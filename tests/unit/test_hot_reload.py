"""
Unit tests for hot_reload.py - Hot Reload Manager
===================================================

Tests cover:
- HotReloadManager initialization
- register_router_reload_callback
- reload_router (single, all, errors, not found)
- reload_config (success, force_sync, errors)
- get_config_sync_status
- get_router_info
- update_strategy_weights (success, registry unavailable, errors)
- set_active_strategy
- clear_ab_weights
- stage_strategy_config
- promote_staged_strategy
- rollback_staged_strategy
- get_strategy_status
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.hot_reload import HotReloadManager, get_hot_reload_manager


# =============================================================================
# HotReloadManager Basic Tests
# =============================================================================


class TestHotReloadManagerInit:
    """Tests for HotReloadManager initialization."""

    def test_fresh_instance(self):
        mgr = HotReloadManager()
        assert mgr._router_reload_callbacks == {}
        assert mgr._strategy_registry is None
        assert mgr._initialized is False

    def test_register_callback(self):
        mgr = HotReloadManager()
        callback = MagicMock()
        mgr.register_router_reload_callback("test-strategy", callback)
        assert "test-strategy" in mgr._router_reload_callbacks
        assert mgr._router_reload_callbacks["test-strategy"] is callback


# =============================================================================
# reload_router Tests
# =============================================================================


class TestReloadRouter:
    """Tests for HotReloadManager.reload_router."""

    def test_reload_specific_strategy_success(self):
        mgr = HotReloadManager()
        callback = MagicMock()
        mgr.register_router_reload_callback("llmrouter-knn", callback)

        result = mgr.reload_router("llmrouter-knn")

        assert result["status"] == "success"
        assert "llmrouter-knn" in result["reloaded"]
        assert result["errors"] == []
        callback.assert_called_once()

    def test_reload_specific_strategy_not_found(self):
        mgr = HotReloadManager()

        result = mgr.reload_router("nonexistent")

        assert result["status"] == "failed"
        assert result["reloaded"] == []
        assert len(result["errors"]) == 1
        assert "not found" in result["errors"][0]["error"]

    def test_reload_specific_strategy_error(self):
        mgr = HotReloadManager()
        callback = MagicMock(side_effect=RuntimeError("model file missing"))
        mgr.register_router_reload_callback("llmrouter-knn", callback)

        result = mgr.reload_router("llmrouter-knn")

        assert result["status"] == "failed"
        assert result["reloaded"] == []
        assert "model file missing" in result["errors"][0]["error"]

    def test_reload_all_strategies(self):
        mgr = HotReloadManager()
        cb1 = MagicMock()
        cb2 = MagicMock()
        mgr.register_router_reload_callback("strategy-a", cb1)
        mgr.register_router_reload_callback("strategy-b", cb2)

        result = mgr.reload_router()

        assert result["status"] == "success"
        assert len(result["reloaded"]) == 2
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_reload_all_partial_failure(self):
        mgr = HotReloadManager()
        cb1 = MagicMock()
        cb2 = MagicMock(side_effect=RuntimeError("broken"))
        mgr.register_router_reload_callback("good", cb1)
        mgr.register_router_reload_callback("bad", cb2)

        result = mgr.reload_router()

        assert result["status"] == "partial"
        assert "good" in result["reloaded"]
        assert len(result["errors"]) == 1

    def test_reload_all_no_strategies(self):
        mgr = HotReloadManager()
        result = mgr.reload_router()
        assert result["status"] == "success"
        assert result["reloaded"] == []


# =============================================================================
# reload_config Tests
# =============================================================================


class TestReloadConfig:
    """Tests for HotReloadManager.reload_config."""

    @patch("litellm_llmrouter.hot_reload.os.kill")
    @patch("litellm_llmrouter.hot_reload.get_sync_manager")
    def test_basic_reload(self, mock_get_sync, mock_kill):
        mock_sync = MagicMock()
        mock_get_sync.return_value = mock_sync

        mgr = HotReloadManager()
        result = mgr.reload_config()

        assert result["status"] == "success"
        assert result["synced_from_remote"] is False
        mock_kill.assert_called_once()
        mock_sync.force_sync.assert_not_called()

    @patch("litellm_llmrouter.hot_reload.os.kill")
    @patch("litellm_llmrouter.hot_reload.get_sync_manager")
    def test_force_sync_reload(self, mock_get_sync, mock_kill):
        mock_sync = MagicMock()
        mock_sync.force_sync.return_value = True
        mock_get_sync.return_value = mock_sync

        mgr = HotReloadManager()
        result = mgr.reload_config(force_sync=True)

        assert result["status"] == "success"
        assert result["synced_from_remote"] is True
        mock_sync.force_sync.assert_called_once()

    @patch("litellm_llmrouter.hot_reload.get_sync_manager")
    def test_reload_error(self, mock_get_sync):
        mock_get_sync.side_effect = RuntimeError("sync manager broken")

        mgr = HotReloadManager()
        result = mgr.reload_config()

        assert result["status"] == "failed"
        assert result["error"] == "config_reload_failed"


# =============================================================================
# get_config_sync_status Tests
# =============================================================================


class TestGetConfigSyncStatus:
    """Tests for HotReloadManager.get_config_sync_status."""

    @patch("litellm_llmrouter.hot_reload.get_sync_manager")
    def test_returns_sync_status(self, mock_get_sync):
        mock_sync = MagicMock()
        mock_sync.get_status.return_value = {"enabled": True, "last_sync": "2024-01-01"}
        mock_get_sync.return_value = mock_sync

        mgr = HotReloadManager()
        result = mgr.get_config_sync_status()

        assert result["enabled"] is True
        assert "last_sync" in result


# =============================================================================
# get_router_info Tests
# =============================================================================


class TestGetRouterInfo:
    """Tests for HotReloadManager.get_router_info."""

    def test_no_strategies(self):
        mgr = HotReloadManager()
        # Mock _ensure_registry to avoid import issues
        mock_registry = MagicMock()
        mock_registry.get_status.return_value = {"active": None}
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.get_router_info()

        assert result["registered_strategies"] == []
        assert result["strategy_count"] == 0
        assert result["hot_reload_enabled"] is False

    def test_with_strategies(self):
        mgr = HotReloadManager()
        mgr.register_router_reload_callback("knn", MagicMock())
        mgr.register_router_reload_callback("svm", MagicMock())

        mock_registry = MagicMock()
        mock_registry.get_status.return_value = {"active": "knn"}
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.get_router_info()

        assert len(result["registered_strategies"]) == 2
        assert result["hot_reload_enabled"] is True


# =============================================================================
# A/B Testing / Weight Management Tests
# =============================================================================


class TestUpdateStrategyWeights:
    """Tests for HotReloadManager.update_strategy_weights."""

    def test_no_registry_returns_failed(self):
        mgr = HotReloadManager()
        mgr._ensure_registry = MagicMock(return_value=None)

        result = mgr.update_strategy_weights({"a": 50, "b": 50})
        assert result["status"] == "failed"

    def test_successful_update(self):
        mgr = HotReloadManager()
        mock_registry = MagicMock()
        mock_registry.set_weights.return_value = True
        mock_registry.get_status.return_value = {"weights": {"a": 50, "b": 50}}
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.update_strategy_weights({"a": 50, "b": 50}, experiment_id="exp-1")

        assert result["status"] == "success"
        assert result["experiment_id"] == "exp-1"

    def test_unregistered_strategy(self):
        mgr = HotReloadManager()
        mock_registry = MagicMock()
        mock_registry.set_weights.return_value = False
        mock_registry.list_strategies.return_value = ["a"]
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.update_strategy_weights({"a": 50, "unknown": 50})
        assert result["status"] == "failed"


class TestSetActiveStrategy:
    """Tests for HotReloadManager.set_active_strategy."""

    def test_no_registry(self):
        mgr = HotReloadManager()
        mgr._ensure_registry = MagicMock(return_value=None)

        result = mgr.set_active_strategy("knn")
        assert result["status"] == "failed"

    def test_successful_activation(self):
        mgr = HotReloadManager()
        mock_registry = MagicMock()
        mock_registry.set_active.return_value = True
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.set_active_strategy("knn")
        assert result["status"] == "success"
        assert result["ab_disabled"] is True

    def test_strategy_not_registered(self):
        mgr = HotReloadManager()
        mock_registry = MagicMock()
        mock_registry.set_active.return_value = False
        mock_registry.list_strategies.return_value = ["mlp"]
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.set_active_strategy("nonexistent")
        assert result["status"] == "failed"


class TestClearAbWeights:
    """Tests for HotReloadManager.clear_ab_weights."""

    def test_successful_clear(self):
        mgr = HotReloadManager()
        mock_registry = MagicMock()
        mock_registry.get_status.return_value = {"weights": {}}
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.clear_ab_weights()
        assert result["status"] == "success"
        mock_registry.clear_weights.assert_called_once()


# =============================================================================
# Staged Strategy Tests
# =============================================================================


class TestStageStrategyConfig:
    """Tests for HotReloadManager.stage_strategy_config."""

    def test_no_registry(self):
        mgr = HotReloadManager()
        mgr._ensure_registry = MagicMock(return_value=None)

        result = mgr.stage_strategy_config({"name": "test"})
        assert result["status"] == "failed"

    def test_missing_name(self):
        mgr = HotReloadManager()
        mock_registry = MagicMock()
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.stage_strategy_config({})
        assert result["status"] == "failed"
        assert "name required" in result["error"]


class TestPromoteStagedStrategy:
    """Tests for HotReloadManager.promote_staged_strategy."""

    def test_no_registry(self):
        mgr = HotReloadManager()
        mgr._ensure_registry = MagicMock(return_value=None)

        result = mgr.promote_staged_strategy("test")
        assert result["status"] == "failed"

    def test_successful_promotion(self):
        mgr = HotReloadManager()
        mock_registry = MagicMock()
        mock_registry.promote_staged.return_value = (True, None)
        mock_registry.get_status.return_value = {"active": "test"}
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.promote_staged_strategy("test")
        assert result["status"] == "success"
        assert result["promoted"] == "test"

    def test_promotion_failure(self):
        mgr = HotReloadManager()
        mock_registry = MagicMock()
        mock_registry.promote_staged.return_value = (False, "No staged strategy")
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.promote_staged_strategy("test")
        assert result["status"] == "failed"


class TestRollbackStagedStrategy:
    """Tests for HotReloadManager.rollback_staged_strategy."""

    def test_successful_rollback(self):
        mgr = HotReloadManager()
        mock_registry = MagicMock()
        mock_registry.rollback_staged.return_value = True
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.rollback_staged_strategy("test")
        assert result["status"] == "success"

    def test_rollback_not_found(self):
        mgr = HotReloadManager()
        mock_registry = MagicMock()
        mock_registry.rollback_staged.return_value = False
        mgr._ensure_registry = MagicMock(return_value=mock_registry)
        mgr._strategy_registry = mock_registry

        result = mgr.rollback_staged_strategy("nonexistent")
        assert result["status"] == "failed"


# =============================================================================
# Singleton Tests
# =============================================================================


class TestGetHotReloadManager:
    """Tests for get_hot_reload_manager singleton."""

    def test_returns_same_instance(self):
        import litellm_llmrouter.hot_reload as mod

        # Reset singleton
        mod._hot_reload_manager = None

        mgr1 = get_hot_reload_manager()
        mgr2 = get_hot_reload_manager()
        assert mgr1 is mgr2

        # Cleanup
        mod._hot_reload_manager = None
