"""
Admin Authentication and Request ID Middleware for LLMRouter Control-Plane
===========================================================================

This module provides:
1. Admin API key authentication for control-plane operations (MCP, A2A, hot-reload)
2. Request correlation ID middleware for tracing and error responses
3. Sanitized error responses that don't leak internal exception details

Usage:
    from litellm_llmrouter.auth import admin_api_key_auth, get_request_id

    # Apply admin auth to control-plane endpoints
    @router.post("/config/reload", dependencies=[Depends(admin_api_key_auth)])
    async def reload_config():
        ...

Configuration:
    Environment variables:
    - ADMIN_API_KEYS: Comma-separated list of admin API keys
    - ADMIN_API_KEY: Single admin API key (legacy, use ADMIN_API_KEYS instead)
    - ADMIN_AUTH_ENABLED: Set to "false" to disable admin auth (NOT recommended)

    When no admin keys are configured and ADMIN_AUTH_ENABLED is not explicitly "false",
    control-plane endpoints will deny all requests (fail-closed).
"""

import logging
import os
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Context variable to store request ID for the current request
_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# Header names
REQUEST_ID_HEADER = "X-Request-ID"
ADMIN_API_KEY_HEADER = "X-Admin-API-Key"
AUTHORIZATION_HEADER = "Authorization"


def get_request_id() -> Optional[str]:
    """
    Get the current request's correlation ID.

    Returns:
        The request ID from the current context, or None if not in a request context.
    """
    return _request_id_ctx.get()


def _load_admin_api_keys() -> set[str]:
    """
    Load admin API keys from environment configuration.

    Returns:
        Set of valid admin API keys. Empty set if none configured.

    Configuration sources (in order of precedence):
    1. ADMIN_API_KEYS: Comma-separated list of keys
    2. ADMIN_API_KEY: Single key (legacy fallback)
    """
    keys: set[str] = set()

    # Primary: comma-separated list
    keys_str = os.getenv("ADMIN_API_KEYS", "").strip()
    if keys_str:
        for key in keys_str.split(","):
            key = key.strip()
            if key:
                keys.add(key)

    # Fallback: single key
    single_key = os.getenv("ADMIN_API_KEY", "").strip()
    if single_key:
        keys.add(single_key)

    return keys


def _is_admin_auth_enabled() -> bool:
    """
    Check if admin authentication is enabled.

    Returns:
        True if admin auth is enabled (default), False if explicitly disabled.

    Note:
        Setting ADMIN_AUTH_ENABLED=false is NOT recommended for production.
        When disabled, control-plane endpoints are only protected by user API key auth.
    """
    env_val = os.getenv("ADMIN_AUTH_ENABLED", "true").lower().strip()
    return env_val not in ("false", "0", "no", "off")


def _extract_bearer_token(auth_header: str) -> Optional[str]:
    """Extract token from Bearer authorization header."""
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        return token if token else None
    return None


async def admin_api_key_auth(request: Request) -> dict:
    """
    FastAPI dependency for admin API key authentication.

    Validates that the request contains a valid admin API key.
    Keys can be provided via:
    - X-Admin-API-Key header (preferred)
    - Authorization: Bearer <key> header (fallback, checks against admin keys)

    Returns:
        dict with 'admin_key' on success

    Raises:
        HTTPException 401: Missing or invalid admin key
        HTTPException 403: Admin auth disabled or no admin keys configured (fail-closed)
    """
    request_id = get_request_id() or "unknown"

    # Check if admin auth is enabled
    if not _is_admin_auth_enabled():
        logger.warning(
            "Admin auth disabled via ADMIN_AUTH_ENABLED=false",
            extra={"request_id": request_id},
        )
        # Still allow request through but log warning
        return {"admin_key": "__disabled__"}

    # Load configured admin keys
    admin_keys = _load_admin_api_keys()

    # Fail-closed: if no admin keys configured, deny all requests
    if not admin_keys:
        logger.error(
            "Admin auth configured but no ADMIN_API_KEYS set - denying request",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error": "control_plane_not_configured",
                "message": "Control-plane access denied. Admin API keys not configured.",
                "request_id": request_id,
            },
        )

    # Try X-Admin-API-Key header first
    admin_key = request.headers.get(ADMIN_API_KEY_HEADER, "").strip()

    # Fallback to Authorization header
    if not admin_key:
        auth_header = request.headers.get(AUTHORIZATION_HEADER, "")
        admin_key = _extract_bearer_token(auth_header) or ""

    # Validate the key
    if not admin_key:
        logger.warning(
            "Missing admin API key in request",
            extra={"request_id": request_id, "path": str(request.url.path)},
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "admin_key_required",
                "message": "Admin API key required. Provide via X-Admin-API-Key header.",
                "request_id": request_id,
            },
        )

    if admin_key not in admin_keys:
        logger.warning(
            "Invalid admin API key in request",
            extra={"request_id": request_id, "path": str(request.url.path)},
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_admin_key",
                "message": "Invalid admin API key.",
                "request_id": request_id,
            },
        )

    logger.debug(
        "Admin authentication successful",
        extra={"request_id": request_id, "path": str(request.url.path)},
    )
    return {"admin_key": admin_key}


def sanitize_error_response(
    error: Exception,
    request_id: Optional[str] = None,
    public_message: str = "An internal error occurred",
) -> dict:
    """
    Create a sanitized error response that doesn't leak internal details.

    Args:
        error: The exception that occurred
        request_id: Correlation ID for the request (auto-fetched if None)
        public_message: User-facing message to return

    Returns:
        dict suitable for JSON response with 'error', 'message', and 'request_id'
    """
    req_id = request_id or get_request_id() or "unknown"

    # Log the full error server-side with request ID for debugging
    logger.error(
        f"Request error: {type(error).__name__}: {str(error)}",
        extra={"request_id": req_id, "error_type": type(error).__name__},
        exc_info=True,
    )

    return {
        "error": "internal_error",
        "message": public_message,
        "request_id": req_id,
    }


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to inject request correlation IDs.

    - Reads X-Request-ID from incoming request headers (for passthrough)
    - Generates a UUID if not provided
    - Sets the request ID in context for use throughout the request lifecycle
    - Adds X-Request-ID to response headers
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Get existing request ID or generate new one
        request_id = request.headers.get(REQUEST_ID_HEADER, "").strip()
        if not request_id:
            request_id = str(uuid.uuid4())

        # Set in context for access throughout request handling
        token = _request_id_ctx.set(request_id)

        try:
            # Process request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers[REQUEST_ID_HEADER] = request_id

            return response
        finally:
            # Reset context
            _request_id_ctx.reset(token)


def create_admin_error_response(
    status_code: int,
    error_code: str,
    message: str,
    request_id: Optional[str] = None,
) -> HTTPException:
    """
    Create an HTTPException with a standardized error response format.

    Args:
        status_code: HTTP status code
        error_code: Machine-readable error code
        message: Human-readable error message
        request_id: Correlation ID (auto-fetched if None)

    Returns:
        HTTPException with structured detail
    """
    req_id = request_id or get_request_id() or "unknown"
    return HTTPException(
        status_code=status_code,
        detail={
            "error": error_code,
            "message": message,
            "request_id": req_id,
        },
    )
