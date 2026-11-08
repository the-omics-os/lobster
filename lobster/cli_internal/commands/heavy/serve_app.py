"""
HTTP app behind ``lobster serve``.

Loopback binds accept only loopback Host headers and need no token. Any other
bind requires a bearer token on every request. Errors never return exception
text to the caller.
"""

import hmac
import ipaddress
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

TOKEN_ENV_VAR = "LOBSTER_SERVE_TOKEN"
INTERNAL_ERROR_DETAIL = "Internal server error"


def _strip_brackets(host: str) -> str:
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def is_loopback_host(host: str) -> bool:
    """True when ``host`` names this machine only (localhost, 127.0.0.0/8, ::1)."""
    name = _strip_brackets(host).lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _host_header_name(value: str) -> str:
    """Host header without the port. Handles ``[::1]:8000`` and ``::1``."""
    value = value.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end != -1 else value
    if value.count(":") == 1:
        return value.split(":", 1)[0]
    return value


def _token_matches(header: Optional[str], token: str) -> bool:
    if not header:
        return False
    scheme, _, supplied = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(supplied.strip().encode(), token.encode())


def build_serve_app(
    host: str,
    token: Optional[str] = None,
    client_factory: Optional[Callable[[], object]] = None,
):
    """Build the FastAPI app for ``host``.

    Raises ValueError when ``host`` is not loopback and no token is given.
    """
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    loopback = is_loopback_host(host)
    token = token or None
    if not loopback and token is None:
        raise ValueError(
            f"A token is required to serve on {host}. Set {TOKEN_ENV_VAR} or pass --token."
        )

    if client_factory is None:
        from lobster.cli_internal.commands.heavy.session_infra import init_client

        client_factory = init_client

    api = FastAPI(
        title="Lobster Agent API",
        description="🦞 Multi-Agent Bioinformatics System by Omics-OS",
        version="2.0",
    )

    @api.middleware("http")
    async def guard(request: Request, call_next):
        if loopback:
            name = _host_header_name(request.headers.get("host", ""))
            # Blocks DNS rebinding: a browser page on another name cannot reach us.
            if not is_loopback_host(name):
                return JSONResponse({"detail": "Invalid host header"}, status_code=400)
        elif not _token_matches(request.headers.get("authorization"), token):
            return JSONResponse(
                {"detail": "Not authenticated"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    class QueryRequest(BaseModel):
        question: str
        session_id: Optional[str] = None
        stream: bool = False

    @api.post("/query")
    async def query_endpoint(request: QueryRequest):
        try:
            client = client_factory()
            return client.query(request.question, stream=request.stream)
        except Exception:
            logger.exception("lobster serve: /query failed")
            raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)

    @api.get("/status")
    async def status_endpoint():
        try:
            client = client_factory()
            return client.get_status()
        except Exception:
            logger.exception("lobster serve: /status failed")
            raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL)

    return api
