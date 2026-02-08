"""
Plugin Middleware: ASGI-Level Request/Response Hooks
=====================================================

Pure ASGI middleware that wires GatewayPlugin on_request/on_response hooks
into the request pipeline. This enables plugins to:

- Inspect and optionally short-circuit incoming requests (on_request)
- Observe response status and headers after processing (on_response)
- Participate in the request lifecycle without buffering or breaking streaming

Design decisions:
- Pure ASGI (not BaseHTTPMiddleware) to preserve streaming responses
- on_request receives parsed request metadata, not raw ASGI scope
- on_response receives status + headers only (no body buffering)
- Plugins are called in priority order (on_request) and reverse order (on_response)
- Failures in plugin hooks are caught and logged, never crash the request

This middleware is inserted AFTER PolicyMiddleware (which handles access control)
and BEFORE BackpressureMiddleware (which handles load shedding).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginRequest:
    """
    Immutable snapshot of an incoming HTTP request for plugin hooks.

    Provides parsed, ergonomic access to request metadata without
    exposing raw ASGI internals to plugin authors.
    """

    method: str
    """HTTP method (GET, POST, etc.)."""

    path: str
    """Request path (e.g., '/v1/chat/completions')."""

    query_string: str
    """Raw query string (e.g., 'model=gpt-4&stream=true')."""

    headers: dict[str, str]
    """Lowercase header name -> value mapping."""

    client_ip: str | None
    """Client IP address if available."""

    request_id: str | None
    """X-Request-ID header value if present."""

    content_type: str | None
    """Content-Type header value if present."""

    @classmethod
    def from_scope(cls, scope: Scope) -> "PluginRequest":
        """
        Create a PluginRequest from an ASGI scope.

        Args:
            scope: ASGI HTTP scope dict

        Returns:
            Parsed PluginRequest instance
        """
        # Parse headers into dict (lowercase keys, last value wins)
        headers: dict[str, str] = {}
        for name_bytes, value_bytes in scope.get("headers", []):
            name = name_bytes.decode("latin-1").lower()
            value = value_bytes.decode("latin-1")
            headers[name] = value

        # Extract client info
        client = scope.get("client")
        client_ip = client[0] if client else None

        return cls(
            method=scope.get("method", "GET"),
            path=scope.get("path", "/"),
            query_string=scope.get("query_string", b"").decode("latin-1"),
            headers=headers,
            client_ip=client_ip,
            request_id=headers.get("x-request-id"),
            content_type=headers.get("content-type"),
        )


@dataclass(frozen=True)
class PluginResponse:
    """
    Short-circuit response from a plugin's on_request hook.

    When a plugin returns a PluginResponse from on_request, the middleware
    sends this response directly to the client without calling the inner app.
    """

    status_code: int = 403
    """HTTP status code."""

    body: dict[str, Any] = field(default_factory=lambda: {"error": "blocked_by_plugin"})
    """JSON-serializable response body."""

    headers: dict[str, str] = field(default_factory=dict)
    """Additional response headers."""


@dataclass(frozen=True)
class ResponseMetadata:
    """
    Response metadata passed to on_response hooks.

    Contains only status and headers (no body) to preserve streaming.
    """

    status_code: int
    """HTTP response status code."""

    headers: dict[str, str]
    """Lowercase header name -> value mapping."""

    duration_ms: float
    """Request duration in milliseconds."""


class PluginMiddleware:
    """
    Pure ASGI middleware that invokes plugin on_request/on_response hooks.

    Middleware ordering (outermost to innermost):
    1. RequestIDMiddleware - correlation
    2. PolicyMiddleware - access control
    3. PluginMiddleware - plugin hooks (this)
    4. RouterDecisionMiddleware - routing telemetry
    5. BackpressureMiddleware - load shedding (wraps app.app)

    This class does NOT use BaseHTTPMiddleware because that breaks
    streaming responses (it buffers the entire body).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._middleware_plugins: list[Any] = []  # GatewayPlugin instances
        self._initialized = False
        # Self-register as the singleton so plugin startup can find us
        set_plugin_middleware(self)

    def set_plugins(self, plugins: list[Any]) -> None:
        """
        Set the list of middleware-capable plugins.

        Called during plugin startup after plugins are loaded and sorted.

        Args:
            plugins: List of GatewayPlugin instances that have
                     on_request or on_response implementations
        """
        self._middleware_plugins = plugins
        self._initialized = True
        if plugins:
            names = [p.name for p in plugins]
            logger.info(f"PluginMiddleware active with {len(plugins)} plugins: {names}")
        else:
            logger.debug("PluginMiddleware active with no middleware plugins")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        # Only intercept HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # If no middleware plugins, pass through immediately
        if not self._middleware_plugins:
            await self.app(scope, receive, send)
            return

        # Parse request
        request = PluginRequest.from_scope(scope)
        start_time = time.monotonic()

        # --- on_request phase ---
        for plugin in self._middleware_plugins:
            try:
                result = await plugin.on_request(request)
                if isinstance(result, PluginResponse):
                    # Plugin wants to short-circuit
                    logger.info(
                        f"Plugin '{plugin.name}' short-circuited request "
                        f"{request.method} {request.path} -> {result.status_code}"
                    )
                    await self._send_json_response(send, result)
                    # Still call on_response for observability
                    duration_ms = (time.monotonic() - start_time) * 1000
                    meta = ResponseMetadata(
                        status_code=result.status_code,
                        headers={},
                        duration_ms=duration_ms,
                    )
                    await self._call_on_response_hooks(request, meta)
                    return
            except Exception as e:
                # Plugin hook failure must never crash the request
                logger.error(
                    f"Plugin '{plugin.name}' on_request failed: {e}",
                    exc_info=True,
                )

        # --- Inner app phase ---
        # Wrap send to capture response status and headers
        captured_status: int | None = None
        captured_headers: dict[str, str] = {}

        async def send_wrapper(message: Message) -> None:
            nonlocal captured_status, captured_headers

            if message["type"] == "http.response.start":
                captured_status = message.get("status", 200)
                # Parse response headers
                for name_bytes, value_bytes in message.get("headers", []):
                    name = name_bytes.decode("latin-1").lower()
                    value = value_bytes.decode("latin-1")
                    captured_headers[name] = value

            elif message["type"] == "http.response.body":
                # If this is the final body chunk, fire on_response hooks
                if not message.get("more_body", False):
                    duration_ms = (time.monotonic() - start_time) * 1000
                    meta = ResponseMetadata(
                        status_code=captured_status or 200,
                        headers=captured_headers,
                        duration_ms=duration_ms,
                    )
                    await self._call_on_response_hooks(request, meta)

            await send(message)

        await self.app(scope, receive, send_wrapper)

    async def _call_on_response_hooks(
        self, request: PluginRequest, response: ResponseMetadata
    ) -> None:
        """
        Call on_response hooks on all middleware plugins in reverse order.

        Reverse order ensures that the first plugin to see the request
        is the last to see the response (symmetric wrapping).
        """
        for plugin in reversed(self._middleware_plugins):
            try:
                await plugin.on_response(request, response)
            except Exception as e:
                logger.error(
                    f"Plugin '{plugin.name}' on_response failed: {e}",
                    exc_info=True,
                )

    async def _send_json_response(self, send: Send, response: PluginResponse) -> None:
        """Send a JSON response for plugin short-circuits."""
        body_bytes = json.dumps(response.body).encode("utf-8")

        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body_bytes)).encode()),
        ]
        for name, value in response.headers.items():
            headers.append((name.encode("latin-1"), value.encode("latin-1")))

        await send(
            {
                "type": "http.response.start",
                "status": response.status_code,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body_bytes,
                "more_body": False,
            }
        )


# Module-level singleton for the plugin middleware instance
_plugin_middleware: PluginMiddleware | None = None


def get_plugin_middleware() -> PluginMiddleware | None:
    """Get the global plugin middleware instance."""
    return _plugin_middleware


def set_plugin_middleware(middleware: PluginMiddleware) -> None:
    """Set the global plugin middleware instance."""
    global _plugin_middleware
    _plugin_middleware = middleware


def reset_plugin_middleware() -> None:
    """Reset the global plugin middleware instance (for testing)."""
    global _plugin_middleware
    _plugin_middleware = None
