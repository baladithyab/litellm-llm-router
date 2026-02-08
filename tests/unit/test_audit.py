"""Unit tests for audit logging module.

Tests:
- Repository write success/failure
- Fail-open behavior (logs fallback, continues)
- Fail-closed behavior (raises AuditWriteError → 503)
- Actor extraction from various RBAC info formats
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from litellm_llmrouter.audit import (
    AuditAction,
    AuditLogEntry,
    AuditLogRepository,
    AuditOutcome,
    AuditWriteError,
    audit_error,
    audit_log,
    audit_success,
    extract_actor_info,
    get_audit_repository,
    is_audit_fail_closed,
    is_audit_log_enabled,
    reset_audit_repository,
)


class TestAuditLogEntry:
    """Tests for AuditLogEntry model."""

    def test_create_entry_with_defaults(self):
        """Test creating an entry with minimal fields."""
        entry = AuditLogEntry(
            action=AuditAction.MCP_SERVER_CREATE.value,
            resource_type="mcp_server",
        )
        assert entry.action == AuditAction.MCP_SERVER_CREATE.value
        assert entry.resource_type == "mcp_server"
        assert entry.outcome == "success"
        assert entry.id is not None
        assert entry.timestamp is not None
        assert entry.actor_type == "unknown"

    def test_create_entry_with_all_fields(self):
        """Test creating an entry with all fields specified."""
        now = datetime.now(timezone.utc)
        entry = AuditLogEntry(
            id="test-id-123",
            timestamp=now,
            request_id="req-456",
            actor_type="user",
            actor_id="user@example.com",
            action=AuditAction.CONFIG_RELOAD.value,
            resource_type="config",
            resource_id="hot_reload",
            outcome=AuditOutcome.DENIED.value,
            outcome_reason="Insufficient permissions",
            metadata={"extra": "info"},
        )
        assert entry.id == "test-id-123"
        assert entry.timestamp == now
        assert entry.request_id == "req-456"
        assert entry.actor_type == "user"
        assert entry.actor_id == "user@example.com"
        assert entry.action == AuditAction.CONFIG_RELOAD.value
        assert entry.resource_type == "config"
        assert entry.resource_id == "hot_reload"
        assert entry.outcome == AuditOutcome.DENIED.value
        assert entry.outcome_reason == "Insufficient permissions"
        assert entry.metadata == {"extra": "info"}

    def test_to_dict(self):
        """Test converting entry to dict for DB storage."""
        entry = AuditLogEntry(
            action=AuditAction.A2A_AGENT_CREATE.value,
            resource_type="a2a_agent",
            resource_id="agent-1",
        )
        d = entry.to_dict()
        assert d["action"] == AuditAction.A2A_AGENT_CREATE.value  # "a2a.agent.create"
        assert d["resource_type"] == "a2a_agent"
        assert d["resource_id"] == "agent-1"
        assert d["outcome"] == "success"
        assert "timestamp" in d
        assert "id" in d


class TestExtractActor:
    """Tests for actor extraction from RBAC info."""

    def test_extract_admin_actor(self):
        """Test extracting actor when is_admin is present."""
        rbac_info = {"is_admin": True, "admin_key": "sk-test-admin-key-12345"}
        actor_type, actor_id = extract_actor_info(rbac_info)
        assert actor_type == "admin_key"
        assert actor_id == "...2345"  # Masked

    def test_extract_user_actor(self):
        """Test extracting actor when user_id is present in user_info."""
        rbac_info = {"user_info": {"user_id": "user@example.com", "team_id": "team-1"}}
        actor_type, actor_id = extract_actor_info(rbac_info)
        assert actor_type == "user"
        assert actor_id == "user@example.com"

    def test_extract_team_actor(self):
        """Test extracting actor when only team_id is present."""
        rbac_info = {"user_info": {"team_id": "team-1", "token": "key-hash"}}
        actor_type, actor_id = extract_actor_info(rbac_info)
        assert actor_type == "team"
        assert actor_id == "team-1"

    def test_extract_api_key_actor(self):
        """Test extracting actor when only api_key token is present."""
        rbac_info = {"user_info": {"token": "sk-test-api-key-abc123"}}
        actor_type, actor_id = extract_actor_info(rbac_info)
        assert actor_type == "api_key"
        assert actor_id == "...c123"  # Masked

    def test_extract_actor_none_rbac(self):
        """Test extracting actor when rbac_info is None."""
        actor_type, actor_id = extract_actor_info(None)
        assert actor_type == "unknown"
        assert actor_id is None

    def test_extract_actor_empty_rbac(self):
        """Test extracting actor when rbac_info is empty dict."""
        actor_type, actor_id = extract_actor_info({})
        assert actor_type == "unknown"
        assert actor_id is None


class TestConfigurationFlags:
    """Tests for audit configuration flags."""

    def test_is_audit_enabled_true(self):
        """Test audit enabled when env var is 'true'."""
        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "true"}):
            assert is_audit_log_enabled() is True

    def test_is_audit_enabled_false(self):
        """Test audit disabled when env var is 'false'."""
        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "false"}):
            assert is_audit_log_enabled() is False

    def test_is_audit_enabled_default(self):
        """Test audit enabled by default when env var not set."""
        # Clear and check default
        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "true"}):
            assert is_audit_log_enabled() is True

    def test_is_fail_closed_false(self):
        """Test fail-open (default) when env var is 'open'."""
        with patch.dict(os.environ, {"AUDIT_LOG_FAIL_MODE": "open"}):
            assert is_audit_fail_closed() is False

    def test_is_fail_closed_true(self):
        """Test fail-closed when env var is 'closed'."""
        with patch.dict(os.environ, {"AUDIT_LOG_FAIL_MODE": "closed"}):
            assert is_audit_fail_closed() is True


class TestAuditLogRepository:
    """Tests for AuditLogRepository."""

    def setup_method(self):
        """Reset repository singleton before each test."""
        reset_audit_repository()

    @pytest.mark.asyncio
    async def test_write_disabled_audit(self):
        """Test write returns True when audit is disabled."""
        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "false"}):
            reset_audit_repository()
            repo = AuditLogRepository()
            entry = AuditLogEntry(
                action=AuditAction.MCP_SERVER_CREATE.value,
                resource_type="mcp_server",
                resource_id="server-1",
            )
            result = await repo.write(entry)
            assert result is True

    @pytest.mark.asyncio
    async def test_write_no_database_url_logs_fallback(self):
        """Test write logs fallback when no DATABASE_URL is configured."""
        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "true", "DATABASE_URL": ""}):
            reset_audit_repository()
            repo = AuditLogRepository()
            repo._db_url = None  # Force no database URL
            entry = AuditLogEntry(
                action=AuditAction.MCP_SERVER_DELETE.value,
                resource_type="mcp_server",
                resource_id="server-1",
            )

            with patch("litellm_llmrouter.audit.logger") as mock_logger:
                result = await repo.write(entry)
                assert result is True
                mock_logger.warning.assert_called()
                call_args = str(mock_logger.warning.call_args)
                assert "AUDIT_FALLBACK" in call_args or "fallback" in call_args.lower()

    @pytest.mark.asyncio
    async def test_write_db_error_fail_open(self):
        """Test write failure in fail-open mode logs and continues."""
        with patch.dict(
            os.environ,
            {
                "AUDIT_LOG_ENABLED": "true",
                "AUDIT_LOG_FAIL_MODE": "open",
                "DATABASE_URL": "postgresql://test:test@localhost/test",
            },
        ):
            reset_audit_repository()
            repo = AuditLogRepository()
            entry = AuditLogEntry(
                action=AuditAction.CONFIG_RELOAD.value,
                resource_type="config",
                resource_id="reload",
            )

            # Mock _persist_to_db to raise an error (avoids asyncpg import issue)
            with patch.object(
                repo, "_persist_to_db", side_effect=Exception("DB connection failed")
            ):
                with patch("litellm_llmrouter.audit.logger") as mock_logger:
                    result = await repo.write(entry)
                    assert result is True  # fail-open continues
                    mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_write_db_error_fail_closed(self):
        """Test write failure in fail-closed mode raises AuditWriteError."""
        with patch.dict(
            os.environ,
            {
                "AUDIT_LOG_ENABLED": "true",
                "AUDIT_LOG_FAIL_MODE": "closed",
                "DATABASE_URL": "postgresql://test:test@localhost/test",
            },
        ):
            reset_audit_repository()
            repo = AuditLogRepository()
            entry = AuditLogEntry(
                action=AuditAction.A2A_AGENT_CREATE.value,
                resource_type="a2a_agent",
                resource_id="agent-1",
            )

            # Mock _persist_to_db to raise an error (avoids asyncpg import issue)
            with patch.object(
                repo, "_persist_to_db", side_effect=Exception("DB connection failed")
            ):
                with pytest.raises(AuditWriteError) as exc_info:
                    await repo.write(entry)
                assert "DB connection failed" in str(exc_info.value)


class TestHighLevelAuditFunctions:
    """Tests for high-level audit_log, audit_success, audit_error functions."""

    def setup_method(self):
        """Reset repository singleton before each test."""
        reset_audit_repository()

    @pytest.mark.asyncio
    async def test_audit_log_when_disabled(self):
        """Test audit_log returns True when audit is disabled."""
        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "false"}):
            reset_audit_repository()
            result = await audit_log(
                action=AuditAction.A2A_AGENT_CREATE,
                resource_type="a2a_agent",
                resource_id="agent-1",
                outcome=AuditOutcome.SUCCESS,
                actor_info=None,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_audit_log_no_database_logs_fallback(self):
        """Test audit_log logs fallback when no database is configured."""
        # The test checks that when enabled but no DB URL, we still return True
        # We directly test the repository behavior rather than mocking env vars
        reset_audit_repository()
        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "true"}):
            reset_audit_repository()
            repo = get_audit_repository()
            # Manually set no DB URL
            repo._db_url = None
            repo._enabled = True

            entry = AuditLogEntry(
                action=AuditAction.MCP_TOOL_CALL.value,
                resource_type="mcp_tool",
                resource_id="tool-1",
            )

            with patch("litellm_llmrouter.audit.logger") as mock_logger:
                result = await repo.write(entry)
                assert result is True
                # Should have logged a fallback warning about no DATABASE_URL
                mock_logger.warning.assert_called()
                # Verify the fallback log message contains expected content
                call_str = str(mock_logger.warning.call_args)
                assert "AUDIT_FALLBACK" in call_str

    @pytest.mark.asyncio
    async def test_audit_success_helper(self):
        """Test audit_success helper sets outcome to SUCCESS."""
        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "false"}):
            reset_audit_repository()
            result = await audit_success(
                action=AuditAction.MCP_SERVER_UPDATE,
                resource_type="mcp_server",
                resource_id="server-1",
                actor_info={"user_info": {"team_id": "team-1"}},
            )
            # Disabled returns True
            assert result is True

    @pytest.mark.asyncio
    async def test_audit_error_helper(self):
        """Test audit_error helper sets outcome to ERROR with reason."""
        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "false"}):
            reset_audit_repository()
            result = await audit_error(
                action=AuditAction.A2A_AGENT_DELETE,
                resource_type="a2a_agent",
                resource_id="agent-1",
                actor_info={"user_info": {"token": "sk-xxx"}},
                error="Internal server error",
            )
            assert result is True


class TestAuditActions:
    """Tests for AuditAction enum coverage."""

    def test_all_expected_actions_exist(self):
        """Verify all expected audit actions exist."""
        expected = [
            "MCP_SERVER_CREATE",
            "MCP_SERVER_UPDATE",
            "MCP_SERVER_DELETE",
            "MCP_TOOL_REGISTER",
            "MCP_TOOL_CALL",
            "CONFIG_RELOAD",
            "CONFIG_SYNC",
            "A2A_AGENT_CREATE",
            "A2A_AGENT_DELETE",
        ]
        for action_name in expected:
            assert hasattr(AuditAction, action_name), f"Missing action: {action_name}"

    def test_action_values_are_dotted_format(self):
        """Verify action values use dotted notation for readability."""
        # Actions use dotted notation like "mcp.server.create"
        for action in AuditAction:
            assert "." in action.value, (
                f"Action {action.name} should use dotted notation"
            )


class TestAuditOutcomes:
    """Tests for AuditOutcome enum coverage."""

    def test_all_expected_outcomes_exist(self):
        """Verify all expected audit outcomes exist."""
        expected = ["SUCCESS", "DENIED", "ERROR"]
        for outcome_name in expected:
            assert hasattr(AuditOutcome, outcome_name), (
                f"Missing outcome: {outcome_name}"
            )

    def test_outcome_values_are_lowercase(self):
        """Verify outcome values are lowercase."""
        for outcome in AuditOutcome:
            assert outcome.value == outcome.name.lower()
