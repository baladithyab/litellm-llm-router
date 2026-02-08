"""Integration tests for audit logging.

Tests:
- Control-plane action writes an audit record to database
- Degraded mode: fail-open continues, fail-closed blocks

These tests are self-managed (don't require the local Docker stack)
because the degraded mode tests use mocks, and the database tests
skip themselves if DATABASE_URL is not configured.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest

# Check if asyncpg is available
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


class TestAuditLogIntegration:
    """Integration tests for audit log persistence."""

    @pytest.fixture(autouse=True)
    def reset_audit_singleton(self):
        """Reset the audit repository singleton before and after each test."""
        from litellm_llmrouter.audit import reset_audit_repository

        reset_audit_repository()
        yield
        reset_audit_repository()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
    async def test_control_plane_action_writes_audit_record(self):
        """Test that a control-plane action writes an audit record to the database.

        This test requires a running Postgres instance with DATABASE_URL set.
        Skip if DATABASE_URL is not configured.
        """
        from litellm_llmrouter.audit import (
            AUDIT_LOGS_TABLE_SQL,
            AuditAction,
            AuditOutcome,
            audit_log,
            reset_audit_repository,
        )

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            pytest.skip("DATABASE_URL not configured")

        # Ensure audit is enabled
        with patch.dict(
            os.environ, {"AUDIT_LOG_ENABLED": "true", "AUDIT_LOG_FAIL_MODE": "open"}
        ):
            reset_audit_repository()

            # Create a unique resource ID for this test
            test_resource_id = f"test-mcp-server-{uuid.uuid4().hex[:8]}"

            # Ensure the table exists
            conn = await asyncpg.connect(db_url)
            try:
                await conn.execute(AUDIT_LOGS_TABLE_SQL)

                # Perform an audit log write
                result = await audit_log(
                    action=AuditAction.MCP_SERVER_CREATE,
                    resource_type="mcp_server",
                    resource_id=test_resource_id,
                    outcome=AuditOutcome.SUCCESS,
                    actor_info={"is_admin": True, "admin_key": "sk-test-key-12345"},
                )

                assert result is True, "Audit log write should succeed"

                # Verify the record was written
                row = await conn.fetchrow(
                    "SELECT * FROM audit_logs WHERE resource_id = $1", test_resource_id
                )

                assert row is not None, "Audit record should exist in database"
                assert row["action"] == AuditAction.MCP_SERVER_CREATE.value
                assert row["resource_type"] == "mcp_server"
                assert row["outcome"] == AuditOutcome.SUCCESS.value
                assert row["actor_type"] == "admin_key"

            finally:
                # Clean up test record
                await conn.execute(
                    "DELETE FROM audit_logs WHERE resource_id = $1", test_resource_id
                )
                await conn.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
    async def test_audit_captures_denied_outcome(self):
        """Test that denied actions are properly recorded."""
        from litellm_llmrouter.audit import (
            AUDIT_LOGS_TABLE_SQL,
            AuditAction,
            AuditOutcome,
            audit_denied,
            reset_audit_repository,
        )

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            pytest.skip("DATABASE_URL not configured")

        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "true"}):
            reset_audit_repository()

            test_resource_id = f"test-denied-{uuid.uuid4().hex[:8]}"

            conn = await asyncpg.connect(db_url)
            try:
                await conn.execute(AUDIT_LOGS_TABLE_SQL)

                # Log a denied action
                result = await audit_denied(
                    action=AuditAction.CONFIG_RELOAD,
                    resource_type="config",
                    resource_id="hot_reload",
                    reason="Insufficient permissions",
                    actor_info={"user_info": {"team_id": "team-test"}},
                )

                assert result is True

                # Verify the denied record
                row = await conn.fetchrow(
                    "SELECT * FROM audit_logs WHERE action = $1 ORDER BY timestamp DESC LIMIT 1",
                    AuditAction.CONFIG_RELOAD.value,
                )

                assert row is not None
                assert row["outcome"] == AuditOutcome.DENIED.value
                assert row["outcome_reason"] == "Insufficient permissions"
                assert row["actor_type"] == "team"

            finally:
                await conn.execute(
                    "DELETE FROM audit_logs WHERE resource_id = $1", test_resource_id
                )
                await conn.close()


class TestAuditDegradedMode:
    """Tests for audit logging degraded mode behavior."""

    @pytest.fixture(autouse=True)
    def reset_audit_singleton(self):
        """Reset the audit repository singleton before and after each test."""
        from litellm_llmrouter.audit import reset_audit_repository

        reset_audit_repository()
        yield
        reset_audit_repository()

    @pytest.mark.asyncio
    async def test_fail_open_continues_on_db_error(self):
        """Test that fail-open mode allows request to continue when DB fails."""
        from litellm_llmrouter.audit import (
            AuditAction,
            AuditLogEntry,
            AuditLogRepository,
            reset_audit_repository,
        )

        # Configure fail-open mode with a DB URL
        with patch.dict(
            os.environ,
            {
                "AUDIT_LOG_ENABLED": "true",
                "AUDIT_LOG_FAIL_MODE": "open",
                "DATABASE_URL": "postgresql://invalid:invalid@localhost:5432/invalid",
            },
        ):
            reset_audit_repository()
            repo = AuditLogRepository()

            entry = AuditLogEntry(
                action=AuditAction.MCP_TOOL_CALL.value,
                resource_type="mcp_tool",
                resource_id="test-tool",
            )

            # Mock _persist_to_db to simulate DB failure
            with patch.object(
                repo, "_persist_to_db", side_effect=Exception("Connection refused")
            ):
                with patch("litellm_llmrouter.audit.logger") as mock_logger:
                    # Should not raise, should return True (fail-open)
                    result = await repo.write(entry)
                    assert result is True

                    # Should have logged the fallback
                    mock_logger.warning.assert_called()
                    log_call = str(mock_logger.warning.call_args)
                    assert "AUDIT_FALLBACK" in log_call

    @pytest.mark.asyncio
    async def test_fail_closed_blocks_on_db_error(self):
        """Test that fail-closed mode raises AuditWriteError when DB fails."""
        from litellm_llmrouter.audit import (
            AuditAction,
            AuditLogEntry,
            AuditLogRepository,
            AuditWriteError,
            reset_audit_repository,
        )

        # Configure fail-closed mode
        with patch.dict(
            os.environ,
            {
                "AUDIT_LOG_ENABLED": "true",
                "AUDIT_LOG_FAIL_MODE": "closed",
                "DATABASE_URL": "postgresql://invalid:invalid@localhost:5432/invalid",
            },
        ):
            reset_audit_repository()
            repo = AuditLogRepository()

            entry = AuditLogEntry(
                action=AuditAction.A2A_AGENT_CREATE.value,
                resource_type="a2a_agent",
                resource_id="test-agent",
            )

            # Mock _persist_to_db to simulate DB failure
            with patch.object(
                repo, "_persist_to_db", side_effect=Exception("Connection refused")
            ):
                # Should raise AuditWriteError in fail-closed mode
                with pytest.raises(AuditWriteError) as exc_info:
                    await repo.write(entry)

                assert "Connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fail_closed_triggers_503_in_endpoint(self):
        """Test that fail-closed mode causes endpoint to return 503.

        This simulates what happens in routes.py when AuditWriteError is raised.
        """
        from fastapi import HTTPException

        from litellm_llmrouter.audit import (
            AuditAction,
            AuditOutcome,
            AuditWriteError,
            audit_log,
            reset_audit_repository,
        )

        # Configure fail-closed mode
        with patch.dict(
            os.environ,
            {
                "AUDIT_LOG_ENABLED": "true",
                "AUDIT_LOG_FAIL_MODE": "closed",
                "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
            },
        ):
            reset_audit_repository()

            # Mock the repository to raise AuditWriteError
            with patch("litellm_llmrouter.audit.get_audit_repository") as mock_get_repo:
                mock_repo = AsyncMock()
                mock_repo.is_enabled = True
                mock_repo.write.side_effect = AuditWriteError("Database unavailable")
                mock_get_repo.return_value = mock_repo

                # Simulate what _handle_audit_write does in routes.py
                async def simulate_audit_handler():
                    try:
                        await audit_log(
                            action=AuditAction.MCP_SERVER_CREATE,
                            resource_type="mcp_server",
                            resource_id="test-server",
                            outcome=AuditOutcome.SUCCESS,
                            actor_info=None,
                        )
                    except AuditWriteError as e:
                        raise HTTPException(
                            status_code=503,
                            detail={"error": "audit_failure", "message": str(e)},
                        )

                with pytest.raises(HTTPException) as exc_info:
                    await simulate_audit_handler()

                assert exc_info.value.status_code == 503
                assert "audit_failure" in str(exc_info.value.detail)


class TestAuditLogQuery:
    """Tests for audit log query functionality."""

    @pytest.fixture(autouse=True)
    def reset_audit_singleton(self):
        """Reset the audit repository singleton before and after each test."""
        from litellm_llmrouter.audit import reset_audit_repository

        reset_audit_repository()
        yield
        reset_audit_repository()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
    async def test_query_audit_logs(self):
        """Test querying audit logs with filters."""
        from litellm_llmrouter.audit import (
            AUDIT_LOGS_TABLE_SQL,
            AuditAction,
            AuditOutcome,
            AuditLogRepository,
            audit_log,
            reset_audit_repository,
        )

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            pytest.skip("DATABASE_URL not configured")

        with patch.dict(os.environ, {"AUDIT_LOG_ENABLED": "true"}):
            reset_audit_repository()

            # Create a unique identifier for this test batch
            batch_id = uuid.uuid4().hex[:8]

            conn = await asyncpg.connect(db_url)
            try:
                await conn.execute(AUDIT_LOGS_TABLE_SQL)

                # Write some test records
                for i in range(3):
                    await audit_log(
                        action=AuditAction.MCP_TOOL_CALL,
                        resource_type="mcp_tool",
                        resource_id=f"test-tool-{batch_id}-{i}",
                        outcome=AuditOutcome.SUCCESS,
                        actor_info=None,
                    )

                # Query using the repository
                repo = AuditLogRepository()
                results = await repo.query(
                    action=AuditAction.MCP_TOOL_CALL.value,
                    resource_type="mcp_tool",
                    limit=10,
                )

                # We should find at least our test records
                assert len(results) >= 3

            finally:
                # Clean up test records
                await conn.execute(
                    "DELETE FROM audit_logs WHERE resource_id LIKE $1",
                    f"test-tool-{batch_id}%",
                )
                await conn.close()
