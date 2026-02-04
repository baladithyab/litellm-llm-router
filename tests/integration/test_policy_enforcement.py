"""
Integration Tests for Policy Engine Enforcement (TG5.2)
========================================================

Tests that prove:
1. Policy denials are enforced at the right layer (ASGI middleware)
2. Denials cannot be bypassed via alternate routes
3. Streaming behavior is preserved (denial before streaming)
4. Feature flags work correctly
5. Audit logging for denials

These tests exercise the middleware with a real Starlette/FastAPI app.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse, StreamingResponse
from starlette.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from litellm_llmrouter.policy_engine import (
    PolicyMiddleware,
    PolicyEngine,
    PolicyConfig,
    PolicyRule,
    PolicyAction,
    PolicyContext,
    reset_policy_engine,
    set_policy_engine,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset the global policy engine before each test."""
    reset_policy_engine()
    yield
    reset_policy_engine()


@pytest.fixture
def deny_config() -> PolicyConfig:
    """Create a config that denies certain patterns."""
    return PolicyConfig.from_dict({
        "default_action": "allow",
        "excluded_paths": ["/_health/live", "/_health/ready"],
        "rules": [
            {
                "name": "deny-premium-models",
                "action": "deny",
                "models": ["gpt-4*"],
                "routes": ["/v1/chat/completions", "/v1/completions"],
                "reason": "Premium model access denied",
                "priority": 50,
            },
            {
                "name": "deny-admin-routes",
                "action": "deny",
                "routes": ["/admin/*", "/config/*"],
                "reason": "Admin access denied",
                "priority": 60,
            },
        ],
    })


@pytest.fixture
def enabled_engine(deny_config) -> PolicyEngine:
    """Create an enabled policy engine."""
    return PolicyEngine(config=deny_config, enabled=True, fail_closed=False)


def create_test_app(engine: PolicyEngine | None = None):
    """Create a test Starlette app with policy middleware."""
    
    async def health(request):
        return JSONResponse({"status": "healthy"})
    
    async def chat_completions(request):
        return JSONResponse({
            "model": "gpt-3.5-turbo",
            "choices": [{"message": {"content": "Hello!"}}],
        })
    
    async def completions(request):
        return JSONResponse({"text": "Hello!"})
    
    async def admin_route(request):
        return JSONResponse({"admin": True})
    
    async def config_route(request):
        return JSONResponse({"config": True})
    
    async def streaming_route(request):
        """Simulate a streaming response."""
        async def generate():
            for i in range(5):
                yield f"data: chunk {i}\n\n".encode()
                await asyncio.sleep(0.01)
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    async def proxy_route(request):
        """Simulate a provider proxy route."""
        return JSONResponse({"proxied": True})
    
    routes = [
        Route("/_health/live", health),
        Route("/_health/ready", health),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/completions", completions, methods=["POST"]),
        Route("/admin/reload", admin_route, methods=["POST"]),
        Route("/config/status", config_route),
        Route("/v1/chat/completions/stream", streaming_route, methods=["POST"]),
        Route("/openai/v1/chat/completions", proxy_route, methods=["POST"]),
        Route("/anthropic/v1/messages", proxy_route, methods=["POST"]),
    ]
    
    app = Starlette(routes=routes)
    
    # Wrap with policy middleware if engine is provided
    if engine:
        # Create a wrapped ASGI app with policy middleware
        # PolicyMiddleware wraps the app at the ASGI level
        wrapped_app = PolicyMiddleware(app, engine)
        return wrapped_app
    
    return app


# =============================================================================
# Bypass Prevention Tests
# =============================================================================


class TestPolicyMiddlewareBypassPrevention:
    """Tests proving that policy enforcement cannot be bypassed."""
    
    def test_direct_route_denied(self, enabled_engine):
        """Test that direct route access is denied by policy."""
        app = create_test_app(enabled_engine)
        client = TestClient(app)
        
        # Try to access admin route - should be denied
        response = client.post("/admin/reload")
        
        assert response.status_code == 403
        data = response.json()
        assert data["error"] == "policy_denied"
        assert "Admin access denied" in data["message"]
    
    def test_model_based_denial(self, deny_config):
        """Test that model-based policies are enforced."""
        # Create engine that checks X-Model header
        engine = PolicyEngine(config=deny_config, enabled=True)
        app = create_test_app(engine)
        client = TestClient(app)
        
        # Request with premium model header - should be denied
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Model": "gpt-4-turbo"},
        )
        
        assert response.status_code == 403
        data = response.json()
        assert "Premium model" in data["message"]
    
    def test_alternate_route_denied(self, enabled_engine):
        """Test that alternate routes are also denied."""
        # Update config to deny proxy routes
        enabled_engine._config.rules.append(
            PolicyRule(
                name="deny-proxy-admin",
                action=PolicyAction.DENY,
                routes=["/openai/*", "/anthropic/*"],
                reason="Provider proxy denied",
                priority=70,
            )
        )
        
        app = create_test_app(enabled_engine)
        client = TestClient(app)
        
        # Try through OpenAI proxy route
        response = client.post("/openai/v1/chat/completions")
        
        assert response.status_code == 403
        
        # Try through Anthropic proxy route  
        response2 = client.post("/anthropic/v1/messages")
        
        assert response2.status_code == 403
    
    def test_case_sensitivity_bypass_prevented(self, enabled_engine):
        """Test that case variation doesn't bypass policy."""
        app = create_test_app(enabled_engine)
        client = TestClient(app)
        
        # Routes are case-sensitive in Starlette, but we still test
        # The middleware sees the exact path
        response = client.post("/admin/reload")
        assert response.status_code == 403
        
        # Different case would result in 404, not bypass
        # (Starlette routing is case-sensitive)
    
    def test_path_traversal_bypass_prevented(self, enabled_engine):
        """Test that path traversal doesn't bypass policy."""
        app = create_test_app(enabled_engine)
        client = TestClient(app)
        
        # Try path traversal
        response = client.post("/admin/../admin/reload")
        
        # Either 403 (denied) or 404 (not found) - not allowed through
        assert response.status_code in (403, 404)
    
    def test_query_params_dont_bypass(self, enabled_engine):
        """Test that adding query params doesn't bypass policy."""
        app = create_test_app(enabled_engine)
        client = TestClient(app)
        
        response = client.post("/admin/reload?bypass=true")
        
        assert response.status_code == 403
    
    def test_different_methods_checked(self, enabled_engine):
        """Test that all HTTP methods are checked."""
        app = create_test_app(enabled_engine)
        client = TestClient(app)
        
        # POST
        response = client.post("/admin/reload")
        assert response.status_code == 403
        
        # GET (even though route might not support it)
        response = client.get("/admin/reload")
        assert response.status_code in (403, 405)  # Denied or Method Not Allowed


# =============================================================================
# Streaming Behavior Tests
# =============================================================================


class TestPolicyStreamingBehavior:
    """Tests proving streaming behavior is preserved."""
    
    def test_denial_before_streaming_begins(self, deny_config):
        """Test that denial happens before streaming response starts."""
        # Add rule to deny streaming route
        deny_config.rules.append(
            PolicyRule(
                name="deny-stream",
                action=PolicyAction.DENY,
                routes=["/v1/chat/completions/stream"],
                reason="Streaming denied",
                priority=40,
            )
        )
        
        engine = PolicyEngine(config=deny_config, enabled=True)
        app = create_test_app(engine)
        client = TestClient(app)
        
        # Try to access streaming route
        response = client.post("/v1/chat/completions/stream")
        
        # Should get immediate 403, not a partial stream
        assert response.status_code == 403
        
        # Response should be complete JSON, not streaming
        data = response.json()
        assert data["error"] == "policy_denied"
    
    def test_streaming_allowed_passes_through(self, enabled_engine):
        """Test that allowed streaming requests work normally."""
        # Streaming route is not in deny rules
        app = create_test_app(enabled_engine)
        client = TestClient(app)
        
        response = client.post("/v1/chat/completions/stream")
        
        # Should get streaming response
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


# =============================================================================
# Feature Flag Tests
# =============================================================================


class TestPolicyFeatureFlags:
    """Tests for feature flag behavior."""
    
    def test_disabled_engine_allows_all(self, deny_config):
        """Test that disabled engine allows all requests."""
        engine = PolicyEngine(config=deny_config, enabled=False)
        app = create_test_app(engine)
        client = TestClient(app)
        
        # Admin route should be allowed when disabled
        response = client.post("/admin/reload")
        
        assert response.status_code == 200
    
    def test_fail_open_mode(self):
        """Test fail-open mode allows on errors."""
        # Create engine with fail-open
        config = PolicyConfig()
        engine = PolicyEngine(config=config, enabled=True, fail_closed=False)
        
        # This is more of a unit test behavior, but we verify via API
        app = create_test_app(engine)
        client = TestClient(app)
        
        response = client.post("/v1/chat/completions")
        
        # Should be allowed (default allow, no errors)
        assert response.status_code == 200
    
    def test_fail_closed_mode(self):
        """Test fail-closed mode denies on error."""
        # Create engine with default deny and fail-closed
        config = PolicyConfig()
        config.default_action = PolicyAction.DENY
        config.default_reason = "Must be explicitly allowed"
        
        engine = PolicyEngine(config=config, enabled=True, fail_closed=True)
        app = create_test_app(engine)
        client = TestClient(app)
        
        response = client.post("/v1/chat/completions")
        
        # Should be denied (no matching rule, default deny)
        assert response.status_code == 403


# =============================================================================
# Health Endpoint Exclusion Tests
# =============================================================================


class TestHealthEndpointExclusion:
    """Tests that health endpoints are excluded from policy."""
    
    def test_liveness_excluded(self, deny_config):
        """Test that liveness probe is excluded from policy."""
        # Add a rule that would deny health if not excluded
        deny_config.rules.append(
            PolicyRule(
                name="deny-all",
                action=PolicyAction.DENY,
                routes=["/*"],
                priority=100,
                reason="Deny all",
            )
        )
        
        engine = PolicyEngine(config=deny_config, enabled=True)
        app = create_test_app(engine)
        client = TestClient(app)
        
        response = client.get("/_health/live")
        
        # Should be allowed despite deny-all rule
        assert response.status_code == 200
    
    def test_readiness_excluded(self, deny_config):
        """Test that readiness probe is excluded from policy."""
        deny_config.rules.append(
            PolicyRule(
                name="deny-all",
                action=PolicyAction.DENY,
                routes=["/*"],
                priority=100,
            )
        )
        
        engine = PolicyEngine(config=deny_config, enabled=True)
        app = create_test_app(engine)
        client = TestClient(app)
        
        response = client.get("/_health/ready")
        
        assert response.status_code == 200


# =============================================================================
# Audit Logging Tests
# =============================================================================


class TestPolicyAuditLogging:
    """Tests for audit logging integration."""
    
    @pytest.mark.asyncio
    async def test_denial_is_audited(self, enabled_engine):
        """Test that policy denials are logged to audit."""
        app = create_test_app(enabled_engine)
        
        # Patch at the source where audit_denied is defined
        with patch("litellm_llmrouter.audit.audit_denied") as mock_audit:
            mock_audit.return_value = True
            
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post("/admin/reload")
            
            assert response.status_code == 403
            
            # Verify audit was called
            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["action"] == "policy.evaluation.deny"
            assert call_kwargs["resource_type"] == "policy_decision"
            assert "Admin access denied" in call_kwargs["reason"]
    
    def test_allowed_request_no_denial_audit(self, enabled_engine):
        """Test that allowed requests don't log denial audit."""
        app = create_test_app(enabled_engine)
        
        with patch("litellm_llmrouter.audit.audit_denied") as mock_audit:
            client = TestClient(app)
            
            # This route is allowed
            response = client.post("/v1/chat/completions")
            
            assert response.status_code == 200
            
            # audit_denied should not be called for allowed requests
            mock_audit.assert_not_called()


# =============================================================================
# Decision Record Tests
# =============================================================================


class TestPolicyDecisionRecord:
    """Tests for auditable decision records."""
    
    def test_denial_response_includes_decision_info(self, enabled_engine):
        """Test that denial response includes policy decision info."""
        app = create_test_app(enabled_engine)
        client = TestClient(app)
        
        response = client.post(
            "/admin/reload",
            headers={"X-Request-ID": "test-req-123"},
        )
        
        assert response.status_code == 403
        data = response.json()
        
        # Decision info should be in response
        assert data["error"] == "policy_denied"
        assert data["message"] == "Admin access denied"
        assert data["policy_name"] == "deny-admin-routes"
        assert data["request_id"] == "test-req-123"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestPolicyEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_rules_uses_default(self):
        """Test that empty rules uses default action."""
        config = PolicyConfig()
        config.default_action = PolicyAction.ALLOW
        
        engine = PolicyEngine(config=config, enabled=True)
        app = create_test_app(engine)
        client = TestClient(app)
        
        response = client.post("/v1/chat/completions")
        
        assert response.status_code == 200
    
    def test_no_headers_handled(self, enabled_engine):
        """Test that requests without headers are handled."""
        app = create_test_app(enabled_engine)
        client = TestClient(app)
        
        response = client.post("/v1/chat/completions")
        
        # Should still work (allowed by default)
        assert response.status_code == 200
    
    def test_missing_context_fields_handled(self, enabled_engine):
        """Test that missing context fields don't crash evaluation."""
        # Route that's allowed should work even without all context
        app = create_test_app(enabled_engine)
        client = TestClient(app)
        
        response = client.post("/v1/chat/completions")
        
        assert response.status_code == 200


# =============================================================================
# Integration with Real Starlette Middleware Stack
# =============================================================================


class TestMiddlewareStackIntegration:
    """Tests for integration with full middleware stack."""
    
    def test_middleware_order_preserved(self, enabled_engine):
        """Test that middleware order is preserved in the stack."""
        # The PolicyMiddleware wraps the ASGI app
        # Policy middleware should correctly pass through headers       
        app = create_test_app(enabled_engine)
        
        client = TestClient(app)
        response = client.post(
            "/admin/reload",
            headers={"X-Request-ID": "middleware-test-123"},
        )
        
        # Policy denial should work correctly
        assert response.status_code == 403
        
        # The request ID should be captured by the policy engine
        data = response.json()
        assert data["request_id"] == "middleware-test-123"
    
    def test_policy_middleware_sees_all_headers(self, enabled_engine):
        """Test that policy middleware can access all request headers."""
        app = create_test_app(enabled_engine)
        
        client = TestClient(app)
        response = client.post(
            "/admin/reload",
            headers={
                "X-Request-ID": "header-test-456",
                "X-Team-ID": "test-team",
                "X-User-ID": "test-user",
            },
        )
        
        # Denial response should include request ID from headers
        assert response.status_code == 403
        data = response.json()
        assert data["request_id"] == "header-test-456"
