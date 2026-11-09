"""FastAPI application for PayPilot AI."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, services
from .agent import LLMError, run_agent
from .config import BASE_DIR, ConfigError, get_settings, llm_chat_url
from .paypal_client import PayPalClient, PayPalError
from .tools import (
    ToolValidationError,
    execute_tool,
    validate_tool_args,
)

logger = logging.getLogger("paypilot")

STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    app.state.paypal = PayPalClient()
    try:
        yield
    finally:
        await app.state.paypal.aclose()


app = FastAPI(
    title="PayPilot AI",
    description="Agentic payment operations for the PayPal sandbox.",
    version="1.0.0",
    lifespan=lifespan,
)


def _error(status: int, error_type: str, message: str, **extra: Any) -> JSONResponse:
    body = {"error": {"type": error_type, "message": message, **extra}}
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(ConfigError)
async def config_error_handler(request: Request, exc: ConfigError) -> JSONResponse:
    return _error(400, "config", str(exc))


@app.exception_handler(PayPalError)
async def paypal_error_handler(request: Request, exc: PayPalError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"error": exc.to_dict()})


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"error": exc.to_dict()})


@app.exception_handler(ToolValidationError)
async def validation_error_handler(
    request: Request, exc: ToolValidationError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": exc.to_dict()})


@app.exception_handler(services.BusinessValidationError)
async def business_error_handler(
    request: Request, exc: services.BusinessValidationError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": exc.to_dict()})


@app.exception_handler(services.NotFoundError)
async def not_found_handler(request: Request, exc: services.NotFoundError) -> JSONResponse:
    return _error(404, "not_found", str(exc))


@app.exception_handler(services.ConflictError)
async def conflict_handler(request: Request, exc: services.ConflictError) -> JSONResponse:
    return _error(409, "conflict", str(exc))


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "loc": list(error.get("loc", [])),
                "msg": error.get("msg"),
                "type": error.get("type"),
            }
        )
    return _error(422, "request_validation", "Invalid request body.", errors=errors)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(
        500,
        "internal",
        "An unexpected server error occurred. Check the server logs for details.",
    )


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: Optional[list[dict[str, Any]]] = None


class DemoSeedRequest(BaseModel):
    amount: str = "49.99"
    currency: str = "USD"
    description: str = "PayPilot AI demo order"


class DemoDeclineRequest(BaseModel):
    order_id: str


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
async def health() -> dict:
    settings = get_settings()

    paypal_configured = settings.paypal_configured
    llm_configured = settings.llm_configured

    paypal_connected = False
    llm_connected = False

    # Verify actual PayPal connectivity (only when credentials are present).
    # A reachable endpoint that returns 401 still counts as connected;
    # only a network failure (status_code == 0) counts as disconnected.
    if paypal_configured:
        client = PayPalClient(settings)
        try:
            try:
                await client.get_access_token()
                paypal_connected = True
            except PayPalError as exc:
                paypal_connected = exc.status_code != 0
            finally:
                await client.aclose()
        except Exception:
            paypal_connected = False

    # Verify actual LLM connectivity (only when credentials are present).
    # Any HTTP response means the endpoint is reachable.
    if llm_configured:
        try:
            url = llm_chat_url(settings.llm_base_url)
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {settings.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.llm_model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "tools": [],
                        "tool_choice": "none",
                        "temperature": 0.2,
                    },
                )
                llm_connected = True
        except Exception:
            llm_connected = False

    return {
        "status": "ok",
        "paypal_configured": paypal_configured,
        "paypal_connected": paypal_connected,
        "llm_configured": llm_configured,
        "llm_connected": llm_connected,
        "paypal_base_url": settings.paypal_base_url,
        "llm_model": settings.llm_model,
        "database": str(settings.db_path),
    }


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request) -> dict:
    settings = get_settings()
    settings.require_llm()
    result = await run_agent(
        user_message=payload.message,
        history=payload.history,
        client=request.app.state.paypal,
        settings=settings,
    )
    return result


@app.get("/api/pending")
async def pending() -> dict:
    return {"pending": db.list_pending_actions("pending")}


@app.get("/api/actions")
async def actions(status: Optional[str] = None) -> dict:
    return {"actions": db.list_pending_actions(status, limit=100)}


@app.post("/api/actions/{action_id}/approve")
async def approve(action_id: str, request: Request) -> dict:
    settings = get_settings()
    settings.require_paypal()
    return await services.approve_action(action_id, request.app.state.paypal)


@app.post("/api/actions/{action_id}/reject")
async def reject(action_id: str) -> dict:
    return services.reject_action(action_id)


@app.post("/api/actions/{action_id}/reconcile")
async def reconcile(action_id: str, request: Request) -> dict:
    settings = get_settings()
    settings.require_paypal()
    return await services.reconcile_action(action_id, request.app.state.paypal)


@app.get("/api/transactions")
async def transactions() -> dict:
    return {"transactions": db.list_transactions()}


@app.get("/api/audit")
async def audit() -> dict:
    return {"audit": db.list_audit()}


@app.post("/api/demo/seed")
async def demo_seed(
    request: Request, payload: Optional[DemoSeedRequest] = None
) -> dict:
    settings = get_settings()
    settings.require_paypal()
    data = payload or DemoSeedRequest()
    args = validate_tool_args(
        "create_order",
        {
            "amount": data.amount,
            "currency": data.currency,
            "description": data.description,
        },
    )
    request_id = uuid.uuid4().hex
    result = await execute_tool(
        "create_order", args, request.app.state.paypal, request_id=request_id
    )
    services.record_transactions("create_order", args, result)
    db.log_audit("user", "create_order", args, result, request_id=request_id)
    approval_url = result.get("approval_url")
    instructions = (
        "Open the approval link and log in with a PayPal sandbox personal "
        "(buyer) account that belongs to the same sandbox as the app. Approve "
        f"the payment, then ask PayPilot to investigate order {result.get('id')}."
    )
    return {
        "order": result,
        "approval_url": approval_url,
        "instructions": instructions,
        "sandbox": True,
    }


@app.post("/api/demo/decline")
async def demo_decline(payload: DemoDeclineRequest) -> dict:
    """Create a pending capture proposal that PayPal will decline in sandbox.

    This never calls PayPal: it only queues a normal, approval-gated action with
    the sandbox mock code so the failure path can be demonstrated reliably.
    """
    args = validate_tool_args(
        "capture_order",
        {
            "order_id": payload.order_id,
            "mock_application_code": "INSTRUMENT_DECLINED",
        },
    )
    action = db.create_pending_action("capture_order", args)
    db.log_audit(
        "demo",
        "capture_order",
        args,
        {"status": "pending_approval", "action_id": action["id"]},
        request_id=action["request_id"],
    )
    return {
        "action": action,
        "sandbox": True,
        "instructions": (
            "Approve this proposal to see PayPal reject the capture with "
            "INSTRUMENT_DECLINED. The failure is audited and nothing is retried."
        ),
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

