"""
Tests for PluginMiddleware: ASGI-level request/response hooks.

Tests cover:
- PluginRequest parsing from ASGI scope
- PluginResponse short-circuit behavior
- on_request/on_response hook invocation and ordering
- Plugin failure isolation (errors don't crash requests)
- Streaming-safe response metadata capture
- ResponseMetadata population
"""

from unittest.mock import AsyncMock

import pytest

from litellm_llmrouter.gateway.plugin_middleware import (
    PluginMiddleware,
    PluginRequest,
    PluginResponse,
    ResponseMetadata,
    get_plugin_middleware,
    reset_plugin_middleware,
)
from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginMetadata,
    PluginCapability,
)


@pytest.fixture(autouse=True)
def _reset_middleware():
    """Reset the global middleware singleton before and after each test."""
    reset_plugin_middleware()
    yield
    reset_plugin_middleware()


# ---------------------------------------------------------------------------
# Mock ASGI helpers
# ---------------------------------------------------------------------------


def make_scope(
    path: str = "/v1/chat/completions",
    method: str = "POST",
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] | None = ("127.0.0.1", 8000),
) -> dict:
    """Create a minimal ASGI HTTP scope."""
    if headers is None:
        headers = [
            (b"content-type", b"application/json"),
            (b"x-request-id", b"req-123"),
            (b"authorization", b"Bearer sk-test"),
        ]
    return {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"stream=true",
        "headers": headers,
        "client": client,
    }


async def mock_receive():
    """Mock ASGI receive callable."""
    return {"type": "http.disconnect"}


def make_200_app():
    """Create a mock ASGI app that returns 200 OK."""

    async def app(scope, receive, send):
        if scope["type"] == "http":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"x-custom", b"test-value"),
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"ok": true}',
                    "more_body": False,
                }
            )

    return app


def make_streaming_app(chunks: int = 3):
    """Create a mock ASGI app that streams response body in chunks."""

    async def app(scope, receive, send):
        if scope["type"] == "http":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            for i in range(chunks):
                await send(
                    {
                        "type": "http.response.body",
                        "body": f"data: chunk {i}\n\n".encode(),
                        "more_body": i < chunks - 1,
                    }
                )

    return app


# ---------------------------------------------------------------------------
# Test plugins
# ---------------------------------------------------------------------------


class PassthroughPlugin(GatewayPlugin):
    """Plugin that does nothing (default behavior)."""

    @property
    def metadata(self):
        return PluginMetadata(
            name="passthrough", capabilities={PluginCapability.MIDDLEWARE}
        )

    async def startup(self, app, context=None):
        pass

    async def shutdown(self, app, context=None):
        pass

    async def on_request(self, request):
        return None

    async def on_response(self, request, response):
        pass


class BlockingPlugin(GatewayPlugin):
    """Plugin that blocks requests to /blocked."""

    @property
    def metadata(self):
        return PluginMetadata(
            name="blocker", capabilities={PluginCapability.MIDDLEWARE}
        )

    async def startup(self, app, context=None):
        pass

    async def shutdown(self, app, context=None):
        pass

    async def on_request(self, request):
        if request.path == "/blocked":
            return PluginResponse(
                status_code=403,
                body={"error": "blocked", "plugin": "blocker"},
                headers={"x-blocked-by": "blocker"},
            )
        return None

    async def on_response(self, request, response):
        pass


class TrackingPlugin(GatewayPlugin):
    """Plugin that records all hook invocations for assertions."""

    def __init__(self):
        self.requests: list[PluginRequest] = []
        self.responses: list[tuple[PluginRequest, ResponseMetadata]] = []

    @property
    def metadata(self):
        return PluginMetadata(
            name="tracker", capabilities={PluginCapability.MIDDLEWARE}
        )

    async def startup(self, app, context=None):
        pass

    async def shutdown(self, app, context=None):
        pass

    async def on_request(self, request):
        self.requests.append(request)
        return None

    async def on_response(self, request, response):
        self.responses.append((request, response))


class ErrorPlugin(GatewayPlugin):
    """Plugin that raises errors in hooks (should not crash the request)."""

    @property
    def metadata(self):
        return PluginMetadata(
            name="error-plugin", capabilities={PluginCapability.MIDDLEWARE}
        )

    async def startup(self, app, context=None):
        pass

    async def shutdown(self, app, context=None):
        pass

    async def on_request(self, request):
        raise RuntimeError("on_request deliberately exploded")

    async def on_response(self, request, response):
        raise RuntimeError("on_response deliberately exploded")


# ===========================================================================
# Tests
# ===========================================================================


class TestPluginRequest:
    """Tests for PluginRequest.from_scope()."""

    def test_basic_parsing(self):
        scope = make_scope()
        req = PluginRequest.from_scope(scope)

        assert req.method == "POST"
        assert req.path == "/v1/chat/completions"
        assert req.query_string == "stream=true"
        assert req.client_ip == "127.0.0.1"
        assert req.request_id == "req-123"
        assert req.content_type == "application/json"
        assert req.headers["authorization"] == "Bearer sk-test"

    def test_missing_client(self):
        scope = make_scope(client=None)
        req = PluginRequest.from_scope(scope)
        assert req.client_ip is None

    def test_missing_headers(self):
        scope = make_scope(headers=[])
        req = PluginRequest.from_scope(scope)
        assert req.request_id is None
        assert req.content_type is None
        assert req.headers == {}

    def test_get_method(self):
        scope = make_scope(method="GET", path="/health")
        req = PluginRequest.from_scope(scope)
        assert req.method == "GET"
        assert req.path == "/health"

    def test_immutable(self):
        req = PluginRequest.from_scope(make_scope())
        with pytest.raises(AttributeError):
            req.path = "/changed"


class TestPluginResponse:
    """Tests for PluginResponse defaults."""

    def test_default_values(self):
        resp = PluginResponse()
        assert resp.status_code == 403
        assert resp.body == {"error": "blocked_by_plugin"}
        assert resp.headers == {}

    def test_custom_values(self):
        resp = PluginResponse(
            status_code=429,
            body={"error": "rate_limited"},
            headers={"retry-after": "60"},
        )
        assert resp.status_code == 429
        assert resp.body["error"] == "rate_limited"
        assert resp.headers["retry-after"] == "60"


class TestPluginMiddlewarePassthrough:
    """Tests for PluginMiddleware with no plugins (passthrough)."""

    @pytest.mark.asyncio
    async def test_passthrough_no_plugins(self):
        """With no plugins, requests pass through unchanged."""
        inner = make_200_app()
        mw = PluginMiddleware(inner)

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(), mock_receive, capture)

        # Verify we got the 200 response
        assert len(responses) == 2
        assert responses[0]["status"] == 200

    @pytest.mark.asyncio
    async def test_passthrough_non_http(self):
        """Non-HTTP scopes (websocket, lifespan) pass through."""
        inner = AsyncMock()
        mw = PluginMiddleware(inner)

        scope = {"type": "websocket", "path": "/ws"}
        await mw(scope, mock_receive, AsyncMock())
        inner.assert_called_once()


class TestPluginMiddlewareOnRequest:
    """Tests for on_request hook invocation."""

    @pytest.mark.asyncio
    async def test_on_request_called(self):
        """on_request is called with parsed PluginRequest."""
        tracker = TrackingPlugin()
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        mw.set_plugins([tracker])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(), mock_receive, capture)

        assert len(tracker.requests) == 1
        assert tracker.requests[0].method == "POST"
        assert tracker.requests[0].path == "/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_on_request_short_circuit(self):
        """Plugin returning PluginResponse short-circuits the request."""
        blocker = BlockingPlugin()
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        mw.set_plugins([blocker])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(path="/blocked"), mock_receive, capture)

        # Should get 403, not 200
        assert responses[0]["status"] == 403
        # Body should be the plugin's response
        import json

        body = json.loads(responses[1]["body"])
        assert body["error"] == "blocked"
        assert body["plugin"] == "blocker"

    @pytest.mark.asyncio
    async def test_on_request_short_circuit_includes_headers(self):
        """Short-circuit response includes plugin-specified headers."""
        blocker = BlockingPlugin()
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        mw.set_plugins([blocker])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(path="/blocked"), mock_receive, capture)

        headers = dict(responses[0]["headers"])
        assert headers[b"content-type"] == b"application/json"
        assert b"x-blocked-by" in headers

    @pytest.mark.asyncio
    async def test_on_request_allows_when_not_blocked(self):
        """Plugin returns None for non-blocked paths."""
        blocker = BlockingPlugin()
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        mw.set_plugins([blocker])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(path="/v1/chat/completions"), mock_receive, capture)

        assert responses[0]["status"] == 200

    @pytest.mark.asyncio
    async def test_on_request_order(self):
        """Plugins are called in order; first short-circuit wins."""
        tracker = TrackingPlugin()
        blocker = BlockingPlugin()
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        # Tracker first, then blocker
        mw.set_plugins([tracker, blocker])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(path="/blocked"), mock_receive, capture)

        # Tracker's on_request should have been called
        assert len(tracker.requests) == 1
        # But response is 403 from blocker
        assert responses[0]["status"] == 403


class TestPluginMiddlewareOnResponse:
    """Tests for on_response hook invocation."""

    @pytest.mark.asyncio
    async def test_on_response_called(self):
        """on_response is called with response metadata after completion."""
        tracker = TrackingPlugin()
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        mw.set_plugins([tracker])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(), mock_receive, capture)

        assert len(tracker.responses) == 1
        req, meta = tracker.responses[0]
        assert req.path == "/v1/chat/completions"
        assert meta.status_code == 200
        assert meta.headers["content-type"] == "application/json"
        assert meta.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_on_response_called_for_short_circuit(self):
        """on_response is called even when a plugin short-circuits."""
        tracker = TrackingPlugin()
        blocker = BlockingPlugin()
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        # Blocker first (short-circuits), tracker second
        mw.set_plugins([blocker, tracker])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(path="/blocked"), mock_receive, capture)

        # on_response should still fire (in reverse order, so tracker is called)
        assert len(tracker.responses) == 1
        _, meta = tracker.responses[0]
        assert meta.status_code == 403

    @pytest.mark.asyncio
    async def test_on_response_reverse_order(self):
        """on_response is called in reverse plugin order."""
        order = []

        class PluginA(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="A")

            async def startup(self, app, ctx=None):
                pass

            async def shutdown(self, app, ctx=None):
                pass

            async def on_request(self, request):
                return None

            async def on_response(self, request, response):
                order.append("A")

        class PluginB(GatewayPlugin):
            @property
            def metadata(self):
                return PluginMetadata(name="B")

            async def startup(self, app, ctx=None):
                pass

            async def shutdown(self, app, ctx=None):
                pass

            async def on_request(self, request):
                return None

            async def on_response(self, request, response):
                order.append("B")

        inner = make_200_app()
        mw = PluginMiddleware(inner)
        mw.set_plugins([PluginA(), PluginB()])

        async def capture(msg):
            pass

        await mw(make_scope(), mock_receive, capture)

        # B should be called first (reverse order), then A
        assert order == ["B", "A"]

    @pytest.mark.asyncio
    async def test_on_response_streaming(self):
        """on_response fires only after the final streaming chunk."""
        tracker = TrackingPlugin()
        inner = make_streaming_app(chunks=3)
        mw = PluginMiddleware(inner)
        mw.set_plugins([tracker])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(), mock_receive, capture)

        # 1 start + 3 body chunks = 4 messages
        assert len(responses) == 4
        # on_response should have been called exactly once (after final chunk)
        assert len(tracker.responses) == 1
        _, meta = tracker.responses[0]
        assert meta.status_code == 200


class TestPluginMiddlewareErrorIsolation:
    """Tests that plugin hook failures don't crash requests."""

    @pytest.mark.asyncio
    async def test_on_request_error_passes_through(self):
        """Error in on_request doesn't prevent the request from completing."""
        error_plugin = ErrorPlugin()
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        mw.set_plugins([error_plugin])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(), mock_receive, capture)

        # Request should still succeed
        assert responses[0]["status"] == 200

    @pytest.mark.asyncio
    async def test_on_response_error_isolated(self):
        """Error in on_response doesn't affect the already-sent response."""
        error_plugin = ErrorPlugin()
        tracker = TrackingPlugin()
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        # Error plugin + tracker - tracker should still be called
        mw.set_plugins([tracker, error_plugin])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(), mock_receive, capture)

        # Response was still sent
        assert responses[0]["status"] == 200
        # Tracker's on_response should still fire (reverse order: error_plugin first, tracker second)
        # error_plugin fires first (reverse), blows up, then tracker fires
        assert len(tracker.responses) == 1

    @pytest.mark.asyncio
    async def test_error_in_one_doesnt_stop_others(self):
        """If one plugin's on_request fails, the next one still runs."""
        error_plugin = ErrorPlugin()
        tracker = TrackingPlugin()
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        # Error first, then tracker
        mw.set_plugins([error_plugin, tracker])

        responses = []

        async def capture(msg):
            responses.append(msg)

        await mw(make_scope(), mock_receive, capture)

        # Tracker should still have been called despite error_plugin blowing up
        assert len(tracker.requests) == 1


class TestPluginMiddlewareSingleton:
    """Tests for the module-level singleton pattern."""

    def test_reset_clears_singleton(self):
        reset_plugin_middleware()
        assert get_plugin_middleware() is None

    def test_set_and_get(self):
        inner = make_200_app()
        mw = PluginMiddleware(inner)
        # __init__ self-registers
        assert get_plugin_middleware() is mw

    def test_self_registration_on_init(self):
        """PluginMiddleware registers itself as singleton on construction."""
        reset_plugin_middleware()
        assert get_plugin_middleware() is None

        inner = make_200_app()
        mw = PluginMiddleware(inner)
        assert get_plugin_middleware() is mw
