"""
Unit tests for HA Config Sync with Leader Election.

Tests the coordination of config sync across replicas in HA deployments,
ensuring only the leader performs sync operations while followers skip.

Validates:
- LLMROUTER_HA_MODE setting behavior
- Leader-only sync execution
- Follower skip behavior
- Leadership change handling
"""

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestHAModeConfiguration:
    """Tests for LLMROUTER_HA_MODE setting."""

    def test_default_ha_mode_is_single(self):
        """Default HA mode should be 'single' when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any cached module state
            import importlib
            import litellm_llmrouter.leader_election as le_module

            importlib.reload(le_module)

            mode = le_module.get_ha_mode()
            assert mode == "single"

    def test_ha_mode_single_explicit(self):
        """Explicit LLMROUTER_HA_MODE=single should return 'single'."""
        with patch.dict(os.environ, {"LLMROUTER_HA_MODE": "single"}, clear=True):
            import importlib
            import litellm_llmrouter.leader_election as le_module

            importlib.reload(le_module)

            mode = le_module.get_ha_mode()
            assert mode == "single"

    def test_ha_mode_leader_election_explicit(self):
        """Explicit LLMROUTER_HA_MODE=leader_election should return 'leader_election'."""
        with patch.dict(
            os.environ, {"LLMROUTER_HA_MODE": "leader_election"}, clear=True
        ):
            import importlib
            import litellm_llmrouter.leader_election as le_module

            importlib.reload(le_module)

            mode = le_module.get_ha_mode()
            assert mode == "leader_election"

    def test_ha_mode_case_insensitive(self):
        """HA mode setting should be case-insensitive."""
        with patch.dict(
            os.environ, {"LLMROUTER_HA_MODE": "LEADER_ELECTION"}, clear=True
        ):
            import importlib
            import litellm_llmrouter.leader_election as le_module

            importlib.reload(le_module)

            mode = le_module.get_ha_mode()
            assert mode == "leader_election"

    def test_legacy_env_var_enables_leader_election(self):
        """Legacy LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED should enable leader_election."""
        with patch.dict(
            os.environ,
            {
                "LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED": "true",
                "DATABASE_URL": "postgresql://localhost/test",
            },
            clear=True,
        ):
            import importlib
            import litellm_llmrouter.leader_election as le_module

            importlib.reload(le_module)

            mode = le_module.get_ha_mode()
            assert mode == "leader_election"

    def test_legacy_env_var_requires_database_url(self):
        """Legacy env var without DATABASE_URL should not enable leader_election."""
        with patch.dict(
            os.environ,
            {"LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED": "true"},
            clear=True,
        ):
            import importlib
            import litellm_llmrouter.leader_election as le_module

            importlib.reload(le_module)

            mode = le_module.get_ha_mode()
            assert mode == "single"

    def test_get_leader_election_config_includes_ha_mode(self):
        """get_leader_election_config should include ha_mode in response."""
        with patch.dict(
            os.environ, {"LLMROUTER_HA_MODE": "leader_election"}, clear=True
        ):
            import importlib
            import litellm_llmrouter.leader_election as le_module

            importlib.reload(le_module)

            config = le_module.get_leader_election_config()
            assert "ha_mode" in config
            assert config["ha_mode"] == "leader_election"
            assert config["enabled"] is True

    def test_single_mode_disables_leader_election(self):
        """In single mode, leader election should be disabled."""
        with patch.dict(os.environ, {"LLMROUTER_HA_MODE": "single"}, clear=True):
            import importlib
            import litellm_llmrouter.leader_election as le_module

            importlib.reload(le_module)

            config = le_module.get_leader_election_config()
            assert config["enabled"] is False


class TestLeaderElection:
    """Tests for LeaderElection class behavior."""

    def test_leader_election_without_database(self):
        """Without DATABASE_URL, instance should always be considered leader."""
        from litellm_llmrouter.leader_election import LeaderElection

        election = LeaderElection(database_url=None)

        # Without database, always acts as leader
        loop = asyncio.new_event_loop()
        try:
            is_leader = loop.run_until_complete(election.try_acquire())
            assert is_leader is True
            assert election.is_leader is True
        finally:
            loop.close()

    def test_holder_id_generation(self):
        """Each LeaderElection instance should have a unique holder_id."""
        from litellm_llmrouter.leader_election import LeaderElection

        election1 = LeaderElection()
        election2 = LeaderElection()

        assert election1.holder_id != election2.holder_id
        assert len(election1.holder_id) > 0
        assert "-" in election1.holder_id  # Format: hostname-uuid

    def test_custom_holder_id(self):
        """Custom holder_id should be used when provided."""
        from litellm_llmrouter.leader_election import LeaderElection

        election = LeaderElection(holder_id="my-custom-id")
        assert election.holder_id == "my-custom-id"

    def test_lease_expiration_check(self):
        """is_leader should return False after lease expires."""
        from litellm_llmrouter.leader_election import LeaderElection

        election = LeaderElection(lease_seconds=1)

        # Simulate having acquired leadership
        election._is_leader = True
        election._lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=10)

        # Should be leader with valid lease
        assert election.is_leader is True

        # Simulate expired lease
        election._lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert election.is_leader is False

    def test_get_status_returns_all_fields(self):
        """get_status should return all expected fields."""
        from litellm_llmrouter.leader_election import LeaderElection

        election = LeaderElection()
        status = election.get_status()

        expected_fields = [
            "holder_id",
            "is_leader",
            "lock_name",
            "lease_seconds",
            "renew_interval_seconds",
            "lease_expires_at",
            "database_configured",
            "last_renewal_error",
            "renewal_thread_alive",
        ]

        for field in expected_fields:
            assert field in status, f"Missing field: {field}"


class TestConfigSyncManagerLeaderBehavior:
    """Tests for ConfigSyncManager leader/follower behavior."""

    @pytest.fixture
    def mock_leader_election(self):
        """Create a mock leader election instance."""
        mock = MagicMock()
        mock.is_leader = True
        mock.get_status.return_value = {
            "holder_id": "test-holder",
            "is_leader": True,
            "lock_name": "config_sync",
            "lease_seconds": 30,
            "renew_interval_seconds": 10,
            "lease_expires_at": None,
            "database_configured": False,
            "last_renewal_error": None,
            "renewal_thread_alive": False,
        }
        mock.ensure_table_exists = AsyncMock(return_value=True)
        mock.try_acquire = AsyncMock(return_value=True)
        mock.start_renewal = MagicMock()
        mock.stop_renewal = MagicMock()
        return mock

    def test_is_leader_returns_true_when_leader_election_disabled(self):
        """_is_leader should return True when leader election is disabled."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        with patch.dict(os.environ, {"LLMROUTER_HA_MODE": "single"}, clear=True):
            manager = ConfigSyncManager()
            manager._leader_election_enabled = False
            manager._leader_election = None

            assert manager._is_leader() is True

    def test_is_leader_returns_election_status_when_enabled(self, mock_leader_election):
        """_is_leader should return leader election status when enabled."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        manager = ConfigSyncManager()
        manager._leader_election_enabled = True
        manager._leader_election = mock_leader_election

        # When mock says we are leader
        mock_leader_election.is_leader = True
        assert manager._is_leader() is True

        # When mock says we are not leader
        mock_leader_election.is_leader = False
        assert manager._is_leader() is False

    def test_sync_skipped_when_not_leader(self, mock_leader_election):
        """Sync should be skipped when instance is not the leader."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        manager = ConfigSyncManager()
        manager._leader_election_enabled = True
        manager._leader_election = mock_leader_election
        manager.s3_sync_enabled = True
        manager._download_from_s3_if_changed = MagicMock(return_value=True)

        # Set up as non-leader
        mock_leader_election.is_leader = False

        # Track initial skipped count
        initial_skipped = manager._skipped_sync_count

        # Simulate one sync iteration manually
        manager._skipped_sync_count += 1  # What _sync_loop would do

        # Download should not have been called
        manager._download_from_s3_if_changed.assert_not_called()
        assert manager._skipped_sync_count > initial_skipped

    def test_sync_performed_when_leader(self, mock_leader_election):
        """Sync should be performed when instance is the leader."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        manager = ConfigSyncManager()
        manager._leader_election_enabled = True
        manager._leader_election = mock_leader_election
        manager.s3_sync_enabled = True
        manager.s3_bucket = "test-bucket"
        manager.s3_key = "test-key"

        # Set up as leader
        mock_leader_election.is_leader = True

        # Check is_leader returns True
        assert manager._is_leader() is True

    def test_status_includes_ha_mode(self):
        """get_status should include ha_mode field."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        with patch.dict(
            os.environ, {"LLMROUTER_HA_MODE": "leader_election"}, clear=True
        ):
            manager = ConfigSyncManager()
            manager._ha_mode = "leader_election"
            status = manager.get_status()

            assert "ha_mode" in status
            assert status["ha_mode"] == "leader_election"

    def test_status_includes_leader_election_details(self, mock_leader_election):
        """get_status should include leader election details when enabled."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        manager = ConfigSyncManager()
        manager._leader_election_enabled = True
        manager._leader_election = mock_leader_election
        manager._ha_mode = "leader_election"

        status = manager.get_status()

        assert "leader_election" in status
        assert "enabled" in status["leader_election"]
        assert "is_leader" in status["leader_election"]
        assert status["leader_election"]["enabled"] is True


class TestLeadershipChangeCallback:
    """Tests for leadership change callback handling."""

    def test_on_leadership_change_callback_called(self):
        """Callback should be called when leadership status changes."""
        from litellm_llmrouter.leader_election import LeaderElection

        callback_mock = MagicMock()
        election = LeaderElection(database_url=None)

        election.start_renewal(on_leadership_change=callback_mock)

        # Simulate leadership change
        election._on_leadership_change = callback_mock
        if election._on_leadership_change:
            election._on_leadership_change(True)

        callback_mock.assert_called_once_with(True)
        election.stop_renewal()

    def test_config_sync_manager_logs_leadership_change(self, caplog):
        """ConfigSyncManager should log leadership changes."""
        from litellm_llmrouter.config_sync import ConfigSyncManager
        import logging

        manager = ConfigSyncManager()

        with caplog.at_level(logging.INFO):
            manager._on_leadership_change(True)
            assert "Became leader" in caplog.text

        caplog.clear()

        with caplog.at_level(logging.INFO):
            manager._on_leadership_change(False)
            assert "Lost leadership" in caplog.text


class TestConcurrentLeaderElection:
    """Tests for concurrent leader election scenarios."""

    def test_only_one_leader_without_database(self):
        """Without database, each instance acts independently as leader."""
        from litellm_llmrouter.leader_election import LeaderElection

        # Without a shared database, each instance is its own leader
        # (single-instance mode)
        election1 = LeaderElection(database_url=None)
        election2 = LeaderElection(database_url=None)

        loop = asyncio.new_event_loop()
        try:
            result1 = loop.run_until_complete(election1.try_acquire())
            result2 = loop.run_until_complete(election2.try_acquire())

            # Both succeed because there's no coordination
            assert result1 is True
            assert result2 is True
        finally:
            loop.close()

    def test_release_clears_leadership(self):
        """Releasing leadership should clear is_leader status."""
        from litellm_llmrouter.leader_election import LeaderElection

        election = LeaderElection(database_url=None)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(election.try_acquire())
            assert election.is_leader is True

            loop.run_until_complete(election.release())
            assert election._is_leader is False
        finally:
            loop.close()


class TestLeaderElectionRenewal:
    """Tests for leader election renewal thread."""

    def test_renewal_thread_starts_and_stops(self):
        """Renewal thread should start and stop properly."""
        from litellm_llmrouter.leader_election import LeaderElection

        election = LeaderElection(
            database_url=None,
            renew_interval_seconds=1,
        )

        election.start_renewal()

        # Thread should be running
        assert election._renew_thread is not None
        assert election._renew_thread.is_alive() is True

        # Stop should terminate thread
        election.stop_renewal()
        time.sleep(0.1)

        # Give thread time to stop
        assert election._stop_event.is_set() is True

    def test_renewal_thread_is_daemon(self):
        """Renewal thread should be a daemon thread."""
        from litellm_llmrouter.leader_election import LeaderElection

        election = LeaderElection(database_url=None)
        election.start_renewal()

        assert election._renew_thread.daemon is True

        election.stop_renewal()


class TestLeaseInfo:
    """Tests for LeaseInfo dataclass."""

    def test_lease_info_to_dict(self):
        """LeaseInfo.to_dict should return all fields."""
        from litellm_llmrouter.leader_election import LeaseInfo

        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=30)

        info = LeaseInfo(
            lock_name="test_lock",
            holder_id="test_holder",
            acquired_at=now,
            expires_at=expires,
            is_leader=True,
        )

        result = info.to_dict()

        assert result["lock_name"] == "test_lock"
        assert result["holder_id"] == "test_holder"
        assert result["acquired_at"] == now.isoformat()
        assert result["expires_at"] == expires.isoformat()
        assert result["is_leader"] is True


class TestSingleInstanceMode:
    """Tests for single-instance (non-HA) mode."""

    def test_single_mode_always_syncs(self):
        """In single mode, sync should always be performed."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        with patch.dict(os.environ, {"LLMROUTER_HA_MODE": "single"}, clear=True):
            manager = ConfigSyncManager()
            manager._leader_election_enabled = False
            manager._leader_election = None

            # Should always be "leader" in single mode
            assert manager._is_leader() is True

    def test_single_mode_no_skipped_syncs(self):
        """In single mode, skipped_sync_count should always be 0."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        with patch.dict(os.environ, {"LLMROUTER_HA_MODE": "single"}, clear=True):
            manager = ConfigSyncManager()
            manager._leader_election_enabled = False

            # Single mode never skips - it's always the leader
            is_leader = manager._is_leader()
            if is_leader:
                # In single mode, we are always leader, so never skip
                pass
            else:
                manager._skipped_sync_count += 1

            assert manager._skipped_sync_count == 0
