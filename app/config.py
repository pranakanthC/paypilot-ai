"""Environment configuration for PayPilot AI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    paypal_client_id: str
    paypal_client_secret: str
    paypal_base_url: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    db_path: Path

    @property
    def paypal_configured(self) -> bool:
        return bool(self.paypal_client_id and self.paypal_client_secret)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)

    def require_paypal(self) -> None:
        if not self.paypal_configured:
            raise ConfigError(
                "PayPal credentials are missing. Add PAYPAL_CLIENT_ID and "
                "PAYPAL_CLIENT_SECRET to your .env file."
            )

    def require_llm(self) -> None:
        if not self.llm_configured:
            raise ConfigError(
                "LLM configuration is missing. Add LLM_API_KEY and LLM_MODEL "
                "to your .env file."
            )


def get_settings() -> Settings:
    """Read settings from the environment on every call (easy to test)."""
    db_path = Path(os.getenv("DB_PATH", "paypilot.db").strip() or "paypilot.db")
    if not db_path.is_absolute():
        db_path = BASE_DIR / db_path
    return Settings(
        paypal_client_id=os.getenv("PAYPAL_CLIENT_ID", "").strip(),
        paypal_client_secret=os.getenv("PAYPAL_CLIENT_SECRET", "").strip(),
        paypal_base_url=os.getenv(
            "PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com"
        ).strip().rstrip("/"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip().rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "deepseek-chat").strip(),
        db_path=db_path,
    )


def llm_chat_url(base_url: str) -> str:
    """Build an OpenAI-compatible chat completions URL from a base URL."""
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/chat/completions"
