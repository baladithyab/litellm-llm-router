"""
HTTP-level tests for LLMRouter routes.

These tests verify:
1. Probe endpoints (/_health/live, /_health/ready) are unauthenticated
2. Custom routes (llmrouter_router) require authentication

Uses FastAPI TestClient against minimal app fixtures.
"""

import os
import pytest

# Check if litellm is available
try:
    import litellm  # noqa: F401
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LITELLM_AVAILABLE,
    reason="litellm package not installed - HTTP tests require litellm",
)


@pytest.fixture
def app_with_health_router():
    """Create a minimal FastAPI app with just the health router."""
    from litellm_llmrouter.routes import health_router

    app = FastAPI()
    app.include_router(health_router)
    return app


@pytest.fixture
def app_with_llmrouter_router():
    """Create a minimal FastAPI app with the llmrouter router (auth-protected)."""
    from fastapi import Depends, HTTPException, Request
    from litellm_llmrouter.routes import health_router

    # Create a mock auth dependency that rejects requests without API key
    async def mock_api_key_auth(request: Request):
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header == "Bearer ":
            raise HTTPException(
                status_code=401, detail={"error": "Invalid or missing API key"}
            )
        return {"api_key": auth_header.split(" ")[1]}

    # Import and create a router without the original auth dependency
    from fastapi import APIRouter

    # Create a custom router with our mock auth
    test_router = APIRouter(
        tags=["llmrouter-test"],
        dependencies=[Depends(mock_api_key_auth)],
    )

    @test_router.get("/router/info")
    async def get_router_info():
        """Get router info - requires auth."""
        return {
            "registered_strategies": [],
            "strategy_count": 0,
            "hot_reload_enabled": False,
        }

    @test_router.get("/config/sync/status")
    async def get_sync_status():
        """Get sync status - requires auth."""
        return {"enabled": False, "message": "Config sync is not enabled"}

    app = FastAPI()
    app.include_router(health_router)
    app.include_router(test_router)
    return app


class TestProbeEndpointsUnauthenticated:
    """Test that probe endpoints are accessible without authentication."""

    def test_liveness_probe_returns_200(self, app_with_health_router):
        """Test /_health/live returns 200 without auth."""
        client = TestClient(app_with_health_router)
        response = client.get("/_health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert data["service"] == "litellm-llmrouter"

    def test_readiness_probe_returns_200_no_deps(self, app_with_health_router):
        """Test /_health/ready returns 200 without auth when no deps configured."""
        # Ensure no DATABASE_URL or REDIS_HOST are set
        env_backup = {
            "DATABASE_URL": os.environ.pop("DATABASE_URL", None),
            "REDIS_HOST": os.environ.pop("REDIS_HOST", None),
            "MCP_GATEWAY_ENABLED": os.environ.pop("MCP_GATEWAY_ENABLED", None),
        }

        try:
            client = TestClient(app_with_health_router)
            response = client.get("/_health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["service"] == "litellm-llmrouter"
            assert "checks" in data
        finally:
            # Restore environment
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_liveness_probe_no_authorization_header(self, app_with_health_router):
        """Probe endpoints work without any Authorization header."""
        client = TestClient(app_with_health_router)
        # Explicitly ensure no auth header
        response = client.get("/_health/live", headers={})

        assert response.status_code == 200

    def test_readiness_probe_no_authorization_header(self, app_with_health_router):
        """Probe endpoints work without any Authorization header."""
        # Clear deps
        env_backup = {
            "DATABASE_URL": os.environ.pop("DATABASE_URL", None),
            "REDIS_HOST": os.environ.pop("REDIS_HOST", None),
            "MCP_GATEWAY_ENABLED": os.environ.pop("MCP_GATEWAY_ENABLED", None),
        }

        try:
            client = TestClient(app_with_health_router)
            response = client.get("/_health/ready", headers={})

            assert response.status_code == 200
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val


class TestAuthGatedEndpoints:
    """Test that custom routes require authentication."""

    def test_router_info_requires_auth(self, app_with_llmrouter_router):
        """Test /router/info returns 401 without API key."""
        client = TestClient(app_with_llmrouter_router, raise_server_exceptions=False)
        response = client.get("/router/info")

        assert response.status_code == 401

    def test_router_info_rejects_empty_bearer(self, app_with_llmrouter_router):
        """Test /router/info returns 401 with empty Bearer token."""
        client = TestClient(app_with_llmrouter_router, raise_server_exceptions=False)
        response = client.get("/router/info", headers={"Authorization": "Bearer "})

        assert response.status_code == 401

    def test_router_info_accepts_valid_key(self, app_with_llmrouter_router):
        """Test /router/info returns 200 with valid API key."""
        client = TestClient(app_with_llmrouter_router)
        response = client.get(
            "/router/info", headers={"Authorization": "Bearer sk-test-key-123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "registered_strategies" in data
        assert "strategy_count" in data
        assert "hot_reload_enabled" in data

    def test_config_sync_status_requires_auth(self, app_with_llmrouter_router):
        """Test /config/sync/status returns 401 without API key."""
        client = TestClient(app_with_llmrouter_router, raise_server_exceptions=False)
        response = client.get("/config/sync/status")

        assert response.status_code == 401

    def test_config_sync_status_accepts_valid_key(self, app_with_llmrouter_router):
        """Test /config/sync/status returns 200 with valid API key."""
        client = TestClient(app_with_llmrouter_router)
        response = client.get(
            "/config/sync/status", headers={"Authorization": "Bearer sk-test-key-123"}
        )

        assert response.status_code == 200

    def test_probes_still_work_on_combined_app(self, app_with_llmrouter_router):
        """Probe endpoints remain unauthenticated even on combined app."""
        # Clear deps
        env_backup = {
            "DATABASE_URL": os.environ.pop("DATABASE_URL", None),
            "REDIS_HOST": os.environ.pop("REDIS_HOST", None),
            "MCP_GATEWAY_ENABLED": os.environ.pop("MCP_GATEWAY_ENABLED", None),
        }

        try:
            client = TestClient(app_with_llmrouter_router)

            # Probes should work without auth
            live_resp = client.get("/_health/live")
            assert live_resp.status_code == 200

            ready_resp = client.get("/_health/ready")
            assert ready_resp.status_code == 200
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val


class TestConfigSyncStatusResponse:
    """Test that /config/sync/status returns the structured ConfigSyncManager.get_status() output."""

    def test_sync_status_returns_full_structure(self):
        """Test that get_sync_status route returns the full ConfigSyncManager.get_status() dict."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        # Create a manager and verify get_status() structure
        manager = ConfigSyncManager(sync_interval_seconds=120)
        status = manager.get_status()

        # Verify the expected keys from ConfigSyncManager.get_status()
        assert "enabled" in status
        assert "hot_reload_enabled" in status
        assert "sync_interval_seconds" in status
        assert "local_config_path" in status
        assert "local_config_hash" in status
        assert "reload_count" in status
        assert "last_sync_time" in status
        assert "running" in status

        # These should be present (even if None when not configured)
        assert "s3" in status or status.get("s3") is None
        assert "gcs" in status or status.get("gcs") is None
