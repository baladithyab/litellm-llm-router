"""
Upskill Evaluator Plugin
=========================

Reference implementation of an evaluator plugin that demonstrates
post-execution scoring with OTEL metric emission.

This plugin:
- Records basic metrics (latency, success rate)
- Optionally shells out to `upskill` CLI if present and enabled
- Validates outbound URLs via SSRF protection
- Emits OTEL attributes: eval.plugin, eval.score, eval.status

Configuration:
    - ROUTEIQ_EVALUATOR_ENABLED: Enable evaluator hooks (default: false)
    - ROUTEIQ_UPSKILL_ENABLED: Enable upskill CLI integration (default: false)
    - ROUTEIQ_UPSKILL_ENDPOINT: Endpoint URL for upskill service (optional)
    - ROUTEIQ_UPSKILL_TIMEOUT: Timeout for upskill calls in seconds (default: 5)

Usage:
    export LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.upskill_evaluator.UpskillEvaluatorPlugin
    export ROUTEIQ_EVALUATOR_ENABLED=true
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from litellm_llmrouter.gateway.plugin_manager import (
    PluginCapability,
    PluginContext,
    PluginMetadata,
)
from litellm_llmrouter.gateway.plugins.evaluator import (
    A2AInvocationContext,
    EvaluationResult,
    EvaluatorPlugin,
    MCPInvocationContext,
    register_evaluator,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Default values
DEFAULT_TIMEOUT = 5.0
DEFAULT_SCORE_SUCCESS = 1.0
DEFAULT_SCORE_FAILURE = 0.0


@dataclass
class UpskillConfig:
    """Configuration for the UpskillEvaluatorPlugin."""

    enabled: bool = False
    """Whether upskill CLI integration is enabled."""

    endpoint: str | None = None
    """Optional endpoint URL for upskill service."""

    timeout: float = DEFAULT_TIMEOUT
    """Timeout for upskill calls in seconds."""

    cli_path: str | None = None
    """Path to upskill CLI binary (auto-detected if not set)."""


def _load_config() -> UpskillConfig:
    """Load configuration from environment variables."""
    enabled = os.getenv("ROUTEIQ_UPSKILL_ENABLED", "false").lower() == "true"
    endpoint = os.getenv("ROUTEIQ_UPSKILL_ENDPOINT")
    timeout = float(os.getenv("ROUTEIQ_UPSKILL_TIMEOUT", str(DEFAULT_TIMEOUT)))
    cli_path = os.getenv("ROUTEIQ_UPSKILL_CLI_PATH") or shutil.which("upskill")

    return UpskillConfig(
        enabled=enabled,
        endpoint=endpoint,
        timeout=timeout,
        cli_path=cli_path,
    )


class UpskillEvaluatorPlugin(EvaluatorPlugin):
    """
    Reference evaluator plugin that demonstrates post-execution scoring.

    This plugin:
    1. Records basic metrics for all invocations (latency, success)
    2. Optionally integrates with the `upskill` CLI for advanced scoring
    3. Emits OTEL attributes for observability

    By default, this plugin is disabled. Enable it by setting:
        ROUTEIQ_EVALUATOR_ENABLED=true

    For upskill CLI integration, also set:
        ROUTEIQ_UPSKILL_ENABLED=true

    **Note**: This plugin does NOT have a hard dependency on upskill.
    If upskill is not available, it falls back to basic scoring.
    """

    def __init__(self) -> None:
        """Initialize the plugin."""
        self._config: UpskillConfig | None = None
        self._context: PluginContext | None = None
        self._validate_url: Callable[[str], str] | None = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="upskill-evaluator",
            version="1.0.0",
            capabilities={PluginCapability.EVALUATOR},
            priority=2000,  # Load after core plugins
            description="Reference evaluator with optional upskill integration",
        )

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """
        Initialize the plugin during startup.

        Args:
            app: FastAPI application instance
            context: Plugin context with utilities
        """
        self._context = context
        self._config = _load_config()

        # Get URL validator from context for SSRF protection
        if context and context.validate_outbound_url:
            self._validate_url = context.validate_outbound_url

        # Register as an evaluator
        register_evaluator(self)

        log = context.logger if context else logger
        log.info(
            f"UpskillEvaluatorPlugin started: "
            f"upskill_enabled={self._config.enabled}, "
            f"cli_available={self._config.cli_path is not None}, "
            f"endpoint={self._config.endpoint or 'none'}"
        )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Cleanup on shutdown."""
        self._config = None
        self._context = None
        self._validate_url = None
        logger.debug("UpskillEvaluatorPlugin shutdown complete")

    async def evaluate_mcp_result(
        self, context: MCPInvocationContext
    ) -> EvaluationResult:
        """
        Evaluate an MCP tool invocation result.

        Scoring logic:
        1. If upskill is enabled and available, delegate to upskill
        2. Otherwise, return basic score based on success/failure

        Args:
            context: MCP invocation context

        Returns:
            Evaluation result with score and status
        """
        # Basic scoring based on success/failure
        base_score = DEFAULT_SCORE_SUCCESS if context.success else DEFAULT_SCORE_FAILURE

        # If upskill integration is disabled or unavailable, return basic score
        if not self._config or not self._config.enabled:
            return EvaluationResult(
                score=base_score,
                status="success" if context.success else "error",
                metadata={
                    "source": "basic",
                    "tool_name": context.tool_name,
                    "server_id": context.server_id,
                    "latency_ms": context.duration_ms,
                },
            )

        # Try upskill CLI integration
        if self._config.cli_path:
            try:
                result = await self._evaluate_with_upskill_cli(
                    invocation_type="mcp",
                    tool_name=context.tool_name,
                    success=context.success,
                    duration_ms=context.duration_ms,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Upskill CLI evaluation failed: {e}")

        # Try upskill service endpoint
        if self._config.endpoint:
            try:
                result = await self._evaluate_with_upskill_service(
                    invocation_type="mcp",
                    tool_name=context.tool_name,
                    success=context.success,
                    duration_ms=context.duration_ms,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Upskill service evaluation failed: {e}")

        # Fallback to basic scoring
        return EvaluationResult(
            score=base_score,
            status="success" if context.success else "error",
            metadata={
                "source": "fallback",
                "tool_name": context.tool_name,
                "server_id": context.server_id,
                "latency_ms": context.duration_ms,
            },
        )

    async def evaluate_a2a_result(
        self, context: A2AInvocationContext
    ) -> EvaluationResult:
        """
        Evaluate an A2A agent invocation result.

        Scoring logic:
        1. If upskill is enabled and available, delegate to upskill
        2. Otherwise, return basic score based on success/failure

        Args:
            context: A2A invocation context

        Returns:
            Evaluation result with score and status
        """
        # Basic scoring based on success/failure
        base_score = DEFAULT_SCORE_SUCCESS if context.success else DEFAULT_SCORE_FAILURE

        # If upskill integration is disabled or unavailable, return basic score
        if not self._config or not self._config.enabled:
            return EvaluationResult(
                score=base_score,
                status="success" if context.success else "error",
                metadata={
                    "source": "basic",
                    "agent_id": context.agent_id,
                    "agent_name": context.agent_name,
                    "method": context.method,
                    "latency_ms": context.duration_ms,
                },
            )

        # Try upskill CLI integration
        if self._config.cli_path:
            try:
                result = await self._evaluate_with_upskill_cli(
                    invocation_type="a2a",
                    tool_name=context.agent_id,
                    success=context.success,
                    duration_ms=context.duration_ms,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Upskill CLI evaluation failed: {e}")

        # Try upskill service endpoint
        if self._config.endpoint:
            try:
                result = await self._evaluate_with_upskill_service(
                    invocation_type="a2a",
                    tool_name=context.agent_id,
                    success=context.success,
                    duration_ms=context.duration_ms,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Upskill service evaluation failed: {e}")

        # Fallback to basic scoring
        return EvaluationResult(
            score=base_score,
            status="success" if context.success else "error",
            metadata={
                "source": "fallback",
                "agent_id": context.agent_id,
                "agent_name": context.agent_name,
                "method": context.method,
                "latency_ms": context.duration_ms,
            },
        )

    async def _evaluate_with_upskill_cli(
        self,
        invocation_type: str,
        tool_name: str,
        success: bool,
        duration_ms: float,
    ) -> EvaluationResult | None:
        """
        Evaluate using the upskill CLI.

        This is a stub implementation that demonstrates how to
        shell out to an external evaluator. In a real implementation,
        this would call the upskill CLI with appropriate arguments.

        Args:
            invocation_type: Type of invocation ("mcp" or "a2a")
            tool_name: Name of tool or agent
            success: Whether invocation succeeded
            duration_ms: Duration in milliseconds

        Returns:
            EvaluationResult or None if CLI call fails
        """
        if not self._config or not self._config.cli_path:
            return None

        try:
            # Stub: Just log that we would call upskill
            # In production, this would actually execute the CLI
            logger.debug(
                f"Would call upskill CLI: {self._config.cli_path} "
                f"--type {invocation_type} --tool {tool_name}"
            )

            # For now, return a simulated response
            # Real implementation would parse CLI output
            score = 0.9 if success else 0.1
            return EvaluationResult(
                score=score,
                status="success",
                metadata={
                    "source": "upskill_cli",
                    "tool_name": tool_name,
                    "invocation_type": invocation_type,
                },
            )
        except asyncio.TimeoutError:
            logger.warning("Upskill CLI timed out")
            return None
        except Exception as e:
            logger.warning(f"Upskill CLI error: {e}")
            return None

    async def _evaluate_with_upskill_service(
        self,
        invocation_type: str,
        tool_name: str,
        success: bool,
        duration_ms: float,
    ) -> EvaluationResult | None:
        """
        Evaluate using the upskill HTTP service.

        This is a stub implementation that demonstrates how to
        call an external evaluation service. In a real implementation,
        this would make an HTTP request to the upskill endpoint.

        Args:
            invocation_type: Type of invocation ("mcp" or "a2a")
            tool_name: Name of tool or agent
            success: Whether invocation succeeded
            duration_ms: Duration in milliseconds

        Returns:
            EvaluationResult or None if service call fails
        """
        if not self._config or not self._config.endpoint:
            return None

        # Validate endpoint URL for SSRF protection
        if self._validate_url:
            try:
                self._validate_url(self._config.endpoint)
            except Exception as e:
                logger.warning(f"Upskill endpoint blocked by SSRF protection: {e}")
                return None

        try:
            # Stub: Just log that we would call the service
            # In production, this would make an actual HTTP request
            logger.debug(
                f"Would call upskill service: {self._config.endpoint} "
                f"with type={invocation_type}, tool={tool_name}"
            )

            # For now, return a simulated response
            # Real implementation would parse HTTP response
            score = 0.85 if success else 0.15
            return EvaluationResult(
                score=score,
                status="success",
                metadata={
                    "source": "upskill_service",
                    "tool_name": tool_name,
                    "invocation_type": invocation_type,
                    "endpoint": self._config.endpoint,
                },
            )
        except asyncio.TimeoutError:
            logger.warning("Upskill service timed out")
            return None
        except Exception as e:
            logger.warning(f"Upskill service error: {e}")
            return None
