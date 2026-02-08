"""
Tests for PluginCallbackBridge: LiteLLM callback → plugin hook integration.

Tests cover:
- Async hook dispatch (pre_call, success, failure)
- Plugin error isolation
- kwargs merging from on_llm_pre_call
- Registration/deregistration with litellm.callbacks
- Empty plugin list (no-op behavior)
"""

import pytest

from litellm_llmrouter.gateway.plugin_callback_bridge import (
    PluginCallbackBridge,
    get_callback_bridge,
    register_callback_bridge,
    reset_callback_bridge,
)
from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginMetadata,
)


@pytest.fixture(autouse=True)
def _reset_bridge():
    """Reset callback bridge singleton before and after each test."""
    reset_callback_bridge()
    yield
    reset_callback_bridge()


# ---------------------------------------------------------------------------
# Test plugins for callback hooks
# ---------------------------------------------------------------------------


class TrackingCallbackPlugin(GatewayPlugin):
    """Plugin that records all LLM lifecycle hook invocations."""

    def __init__(self):
        self.pre_calls: list[tuple[str, list, dict]] = []
        self.successes: list[tuple[str, object, dict]] = []
        self.failures: list[tuple[str, Exception, dict]] = []

    @property
    def metadata(self):
        return PluginMetadata(name="tracking-callback")

    async def startup(self, app, context=None):
        pass

    async def shutdown(self, app, context=None):
        pass

    async def on_llm_pre_call(self, model, messages, kwargs):
        self.pre_calls.append((model, messages, kwargs))
        return None

    async def on_llm_success(self, model, response, kwargs):
        self.successes.append((model, response, kwargs))

    async def on_llm_failure(self, model, exception, kwargs):
        self.failures.append((model, exception, kwargs))


class ModifyingPlugin(GatewayPlugin):
    """Plugin that modifies kwargs in on_llm_pre_call."""

    @property
    def metadata(self):
        return PluginMetadata(name="modifier")

    async def startup(self, app, context=None):
        pass

    async def shutdown(self, app, context=None):
        pass

    async def on_llm_pre_call(self, model, messages, kwargs):
        return {"metadata": {**kwargs.get("metadata", {}), "injected": True}}


class ErrorCallbackPlugin(GatewayPlugin):
    """Plugin that raises errors in hooks."""

    @property
    def metadata(self):
        return PluginMetadata(name="error-callback")

    async def startup(self, app, context=None):
        pass

    async def shutdown(self, app, context=None):
        pass

    async def on_llm_pre_call(self, model, messages, kwargs):
        raise RuntimeError("pre_call exploded")

    async def on_llm_success(self, model, response, kwargs):
        raise RuntimeError("success exploded")

    async def on_llm_failure(self, model, exception, kwargs):
        raise RuntimeError("failure exploded")


# ===========================================================================
# Tests
# ===========================================================================


class TestPluginCallbackBridgePreCall:
    """Tests for async_log_pre_api_call dispatching to on_llm_pre_call."""

    @pytest.mark.asyncio
    async def test_pre_call_dispatched(self):
        tracker = TrackingCallbackPlugin()
        bridge = PluginCallbackBridge([tracker])

        messages = [{"role": "user", "content": "hello"}]
        kwargs = {"model": "gpt-4", "temperature": 0.7}

        await bridge.async_log_pre_api_call("gpt-4", messages, kwargs)

        assert len(tracker.pre_calls) == 1
        model, msgs, kw = tracker.pre_calls[0]
        assert model == "gpt-4"
        assert msgs == messages

    @pytest.mark.asyncio
    async def test_pre_call_merges_kwargs(self):
        modifier = ModifyingPlugin()
        bridge = PluginCallbackBridge([modifier])

        kwargs = {"model": "gpt-4", "metadata": {"original": True}}
        await bridge.async_log_pre_api_call("gpt-4", [], kwargs)

        # The modifier should have updated kwargs in-place
        assert kwargs["metadata"]["injected"] is True
        assert kwargs["metadata"]["original"] is True

    @pytest.mark.asyncio
    async def test_pre_call_no_plugins(self):
        """No plugins means no-op (no errors)."""
        bridge = PluginCallbackBridge([])
        await bridge.async_log_pre_api_call("gpt-4", [], {})

    @pytest.mark.asyncio
    async def test_pre_call_error_isolated(self):
        """Plugin error in pre_call doesn't crash the bridge."""
        error_plugin = ErrorCallbackPlugin()
        tracker = TrackingCallbackPlugin()
        bridge = PluginCallbackBridge([error_plugin, tracker])

        await bridge.async_log_pre_api_call("gpt-4", [], {})

        # Tracker should still have been called
        assert len(tracker.pre_calls) == 1


class TestPluginCallbackBridgeSuccess:
    """Tests for async_log_success_event dispatching to on_llm_success."""

    @pytest.mark.asyncio
    async def test_success_dispatched(self):
        tracker = TrackingCallbackPlugin()
        bridge = PluginCallbackBridge([tracker])

        response = {"choices": [{"message": {"content": "hi"}}]}
        kwargs = {"model": "gpt-4"}

        await bridge.async_log_success_event(kwargs, response, None, None)

        assert len(tracker.successes) == 1
        model, resp, kw = tracker.successes[0]
        assert model == "gpt-4"
        assert resp == response

    @pytest.mark.asyncio
    async def test_success_no_plugins(self):
        bridge = PluginCallbackBridge([])
        await bridge.async_log_success_event({}, {}, None, None)

    @pytest.mark.asyncio
    async def test_success_error_isolated(self):
        error_plugin = ErrorCallbackPlugin()
        tracker = TrackingCallbackPlugin()
        bridge = PluginCallbackBridge([error_plugin, tracker])

        await bridge.async_log_success_event({"model": "gpt-4"}, {}, None, None)

        assert len(tracker.successes) == 1


class TestPluginCallbackBridgeFailure:
    """Tests for async_log_failure_event dispatching to on_llm_failure."""

    @pytest.mark.asyncio
    async def test_failure_dispatched(self):
        tracker = TrackingCallbackPlugin()
        bridge = PluginCallbackBridge([tracker])

        exc = ValueError("API error")
        kwargs = {"model": "gpt-4"}

        await bridge.async_log_failure_event(kwargs, exc, None, None)

        assert len(tracker.failures) == 1
        model, error, kw = tracker.failures[0]
        assert model == "gpt-4"
        assert isinstance(error, ValueError)
        assert str(error) == "API error"

    @pytest.mark.asyncio
    async def test_failure_wraps_non_exception_response(self):
        """When LiteLLM passes a non-exception response_obj, it gets wrapped."""
        tracker = TrackingCallbackPlugin()
        bridge = PluginCallbackBridge([tracker])

        await bridge.async_log_failure_event(
            {"model": "gpt-4"}, "some error string", None, None
        )

        assert len(tracker.failures) == 1
        _, error, _ = tracker.failures[0]
        assert isinstance(error, Exception)
        assert "some error string" in str(error)

    @pytest.mark.asyncio
    async def test_failure_no_plugins(self):
        bridge = PluginCallbackBridge([])
        await bridge.async_log_failure_event({}, Exception("err"), None, None)

    @pytest.mark.asyncio
    async def test_failure_error_isolated(self):
        error_plugin = ErrorCallbackPlugin()
        tracker = TrackingCallbackPlugin()
        bridge = PluginCallbackBridge([error_plugin, tracker])

        await bridge.async_log_failure_event(
            {"model": "gpt-4"}, Exception("test"), None, None
        )

        assert len(tracker.failures) == 1


class TestPluginCallbackBridgeSetPlugins:
    """Tests for set_plugins and dynamic plugin updates."""

    @pytest.mark.asyncio
    async def test_set_plugins_updates_list(self):
        bridge = PluginCallbackBridge()

        # Initially no plugins
        await bridge.async_log_pre_api_call("gpt-4", [], {})

        # Add a tracker
        tracker = TrackingCallbackPlugin()
        bridge.set_plugins([tracker])

        await bridge.async_log_pre_api_call("gpt-4", [], {})
        assert len(tracker.pre_calls) == 1

    @pytest.mark.asyncio
    async def test_set_plugins_to_empty(self):
        tracker = TrackingCallbackPlugin()
        bridge = PluginCallbackBridge([tracker])

        # Clear plugins
        bridge.set_plugins([])

        await bridge.async_log_pre_api_call("gpt-4", [], {})
        assert len(tracker.pre_calls) == 0


class TestCallbackBridgeRegistration:
    """Tests for register_callback_bridge with litellm.callbacks."""

    @pytest.mark.asyncio
    async def test_register_appends_to_callbacks(self):
        import litellm

        tracker = TrackingCallbackPlugin()
        original_callbacks = litellm.callbacks
        try:
            litellm.callbacks = []
            bridge = register_callback_bridge([tracker])

            assert bridge is not None
            assert any(isinstance(cb, PluginCallbackBridge) for cb in litellm.callbacks)
        finally:
            litellm.callbacks = original_callbacks

    @pytest.mark.asyncio
    async def test_register_avoids_duplicates(self):
        import litellm

        tracker = TrackingCallbackPlugin()
        original_callbacks = litellm.callbacks
        try:
            existing = PluginCallbackBridge()
            litellm.callbacks = [existing]

            bridge = register_callback_bridge([tracker])

            # Should reuse existing, not add a new one
            plugin_bridges = [
                cb for cb in litellm.callbacks if isinstance(cb, PluginCallbackBridge)
            ]
            assert len(plugin_bridges) == 1
            assert bridge is existing
        finally:
            litellm.callbacks = original_callbacks

    def test_register_empty_plugins_returns_none(self):
        result = register_callback_bridge([])
        assert result is None

    def test_get_callback_bridge_after_reset(self):
        reset_callback_bridge()
        assert get_callback_bridge() is None


class TestCallbackBridgeSyncHooks:
    """Tests for sync hooks (should be no-ops)."""

    def test_sync_pre_call_noop(self):
        bridge = PluginCallbackBridge([TrackingCallbackPlugin()])
        # Should not raise
        bridge.log_pre_api_call("gpt-4", [], {})

    def test_sync_success_noop(self):
        bridge = PluginCallbackBridge([TrackingCallbackPlugin()])
        bridge.log_success_event({}, {}, None, None)

    def test_sync_failure_noop(self):
        bridge = PluginCallbackBridge([TrackingCallbackPlugin()])
        bridge.log_failure_event({}, {}, None, None)
