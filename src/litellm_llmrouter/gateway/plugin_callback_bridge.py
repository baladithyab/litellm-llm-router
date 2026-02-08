"""
Plugin Callback Bridge: LiteLLM Callback → Plugin Hook Integration
===================================================================

Bridges LiteLLM's callback system to GatewayPlugin LLM lifecycle hooks.

LiteLLM calls methods on objects in ``litellm.callbacks`` at various points
in the LLM call lifecycle. This bridge translates those callbacks into
GatewayPlugin hook invocations:

    litellm.log_pre_api_call  →  plugin.on_llm_pre_call
    litellm.log_success_event →  plugin.on_llm_success
    litellm.log_failure_event →  plugin.on_llm_failure

Registration:
    Called during plugin startup in app.py. The bridge is appended to
    ``litellm.callbacks`` alongside the existing RouterDecisionCallback.

Design:
    - Uses duck-typing (no CustomLogger subclass) for minimal coupling
    - Plugin hook failures are caught and logged, never crash the LLM call
    - Plugins are called in priority order
"""

from __future__ import annotations

import logging
from typing import Any

from litellm_llmrouter.gateway.plugins.guardrails_base import GuardrailBlockError

logger = logging.getLogger(__name__)


class PluginCallbackBridge:
    """
    LiteLLM callback that bridges to GatewayPlugin LLM lifecycle hooks.

    Implements the LiteLLM callback interface via duck-typing (same pattern
    as RouterDecisionCallback). Delegates to plugins that override
    on_llm_pre_call, on_llm_success, or on_llm_failure.
    """

    def __init__(self, plugins: list[Any] | None = None) -> None:
        """
        Args:
            plugins: List of GatewayPlugin instances with LLM lifecycle hooks.
                     Can be set later via set_plugins().
        """
        self._plugins: list[Any] = plugins or []

    def set_plugins(self, plugins: list[Any]) -> None:
        """Update the list of callback-capable plugins."""
        self._plugins = plugins
        if plugins:
            names = [p.name for p in plugins]
            logger.info(
                f"PluginCallbackBridge active with {len(plugins)} plugins: {names}"
            )

    # =========================================================================
    # Synchronous hooks (LiteLLM calls these on the main thread)
    # =========================================================================

    def log_pre_api_call(
        self, model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> None:
        """Called before each LLM API call (sync path)."""
        # Sync hooks delegate to async via the event loop if available,
        # but most LiteLLM proxy paths use the async variants below.
        pass

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called after successful LLM API call (sync path)."""
        pass

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called after failed LLM API call (sync path)."""
        pass

    # =========================================================================
    # Async hooks (LiteLLM proxy calls these in the async path)
    # =========================================================================

    async def async_log_pre_api_call(
        self, model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> None:
        """Called before each LLM API call (async path)."""
        if not self._plugins:
            return

        for plugin in self._plugins:
            try:
                result = await plugin.on_llm_pre_call(model, messages, kwargs)
                if isinstance(result, dict):
                    # Merge overrides into kwargs
                    kwargs.update(result)
            except GuardrailBlockError:
                raise  # Let guardrail blocks propagate as request failures
            except Exception as e:
                logger.error(
                    f"Plugin '{plugin.name}' on_llm_pre_call failed: {e}",
                    exc_info=True,
                )

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called after successful LLM API call (async path)."""
        if not self._plugins:
            return

        model = kwargs.get("model", "unknown")
        for plugin in self._plugins:
            try:
                await plugin.on_llm_success(model, response_obj, kwargs)
            except Exception as e:
                logger.error(
                    f"Plugin '{plugin.name}' on_llm_success failed: {e}",
                    exc_info=True,
                )

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called after failed LLM API call (async path)."""
        if not self._plugins:
            return

        model = kwargs.get("model", "unknown")
        # LiteLLM passes the exception as response_obj for failures
        exception = (
            response_obj
            if isinstance(response_obj, Exception)
            else Exception(str(response_obj))
        )
        for plugin in self._plugins:
            try:
                await plugin.on_llm_failure(model, exception, kwargs)
            except Exception as e:
                logger.error(
                    f"Plugin '{plugin.name}' on_llm_failure failed: {e}",
                    exc_info=True,
                )

    # =========================================================================
    # Additional LiteLLM proxy hooks (no-ops for now)
    # =========================================================================

    async def async_post_call_success_hook(
        self, data: dict[str, Any], user_api_key_dict: Any, response: Any
    ) -> Any:
        """Post-call success hook (proxy-specific)."""
        pass

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, Any],
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: str | None = None,
    ) -> Any:
        """Post-call failure hook (proxy-specific)."""
        pass


# Module-level singleton
_callback_bridge: PluginCallbackBridge | None = None


def get_callback_bridge() -> PluginCallbackBridge | None:
    """Get the global callback bridge instance."""
    return _callback_bridge


def register_callback_bridge(plugins: list[Any]) -> PluginCallbackBridge | None:
    """
    Register the plugin callback bridge with LiteLLM.

    Args:
        plugins: List of GatewayPlugin instances with LLM lifecycle hooks

    Returns:
        The registered bridge, or None if no callback plugins or LiteLLM unavailable
    """
    global _callback_bridge

    if not plugins:
        logger.debug("No callback-capable plugins, skipping bridge registration")
        return None

    try:
        import litellm

        bridge = PluginCallbackBridge(plugins)

        if not hasattr(litellm, "callbacks"):
            litellm.callbacks = []

        # Avoid duplicate registration
        for existing in litellm.callbacks:
            if isinstance(existing, PluginCallbackBridge):
                logger.debug(
                    "PluginCallbackBridge already registered, updating plugins"
                )
                existing.set_plugins(plugins)
                _callback_bridge = existing
                return existing

        litellm.callbacks.append(bridge)
        _callback_bridge = bridge
        logger.info(
            f"Registered PluginCallbackBridge with LiteLLM "
            f"({len(plugins)} callback plugins)"
        )
        return bridge

    except ImportError:
        logger.warning("LiteLLM not available, cannot register callback bridge")
        return None
    except Exception as e:
        logger.error(f"Failed to register callback bridge: {e}")
        return None


def reset_callback_bridge() -> None:
    """Reset the global callback bridge (for testing)."""
    global _callback_bridge
    _callback_bridge = None
