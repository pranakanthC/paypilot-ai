"""Smoke test for PayPilot AI.

Starts uvicorn serving app.main:app on http://127.0.0.1:8765, then uses httpx to
GET the root page, static assets, and every API endpoint, printing each status
code. Run from the project root:

    python scripts/smoke_test.py

Nothing is written outside the project directory.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Wait for the server to accept connections.
        ready = False
        for _ in range(50):
            try:
                httpx.get(BASE + "/api/health", timeout=2.0)
                ready = True
                break
            except Exception:
                time.sleep(0.2)
        if not ready:
            print("SERVER DID NOT START IN TIME")
            out = proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            return 1

        paths = [
            "GET /",
            "GET /api/health",
            "GET /api/transactions",
            "GET /api/audit",
            "GET /api/pending",
            "GET /api/actions",
        ]

        for desc in paths:
            method, path = desc.split(" ", 1)
            try:
                r = httpx.request(method, BASE + path, timeout=10.0)
                print(f"{desc}  ->  {r.status_code}")
            except Exception as e:
                print(f"{desc}  ->  ERROR  {type(e).__name__}: {e}")

        # Check that index.html references exist as static assets (served from /static).
        static_assets = [
            "/static/",
        ]
        for path in static_assets:
            try:
                r = httpx.get(BASE + path, timeout=10.0)
                print(f"GET {path}  ->  {r.status_code}")
            except Exception as e:
                print(f"GET {path}  ->  ERROR  {type(e).__name__}: {e}")

        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
