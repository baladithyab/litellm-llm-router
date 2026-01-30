"""
Tests for the Evaluator Plugin Framework.

These tests verify that:
1. EvaluatorPlugin contract is correct
2. Evaluation hooks are called correctly
3. OTEL attributes are added to spans
4. UpskillEvaluatorPlugin works as expected
5. Evaluator registration and execution works
"""

import os
import pytest
from unittest.mock import MagicMock, patch


class TestEvaluationResult:
    """Test the EvaluationResult dataclass."""

    def test_default_values(self):
        """Test that EvaluationResult has correct defaults."""
        from litellm_llmrouter.gateway.plugins.evaluator import EvaluationResult

        result = EvaluationResult()
        assert result.score is None
        assert result.status == "success"
        assert result.metadata == {}
        assert result.error is None

    def test_custom_values(self):
        """Test EvaluationResult with custom values."""
        from litellm_llmrouter.gateway.plugins.evaluator import EvaluationResult

        result = EvaluationResult(
            score=0.85,
            status="error",
            metadata={"key": "value"},
            error="Test error",
        )
        assert result.score == 0.85
        assert result.status == "error"
        assert result.metadata == {"key": "value"}
        assert result.error == "Test error"


class TestMCPInvocationContext:
    """Test the MCPInvocationContext dataclass."""

    def test_mcp_context_creation(self):
        """Test creating an MCP invocation context."""
        from litellm_llmrouter.gateway.plugins.evaluator import MCPInvocationContext

        ctx = MCPInvocationContext(
            tool_name="test-tool",
            server_id="server-1",
            server_name="Test Server",
            arguments={"arg1": "value1"},
            result={"output": "test"},
            success=True,
            error=None,
            duration_ms=123.45,
            span=None,
        )
        assert ctx.tool_name == "test-tool"
        assert ctx.server_id == "server-1"
        assert ctx.success is True
        assert ctx.duration_ms == 123.45


class TestA2AInvocationContext:
    """Test the A2AInvocationContext dataclass."""

    def test_a2a_context_creation(self):
        """Test creating an A2A invocation context."""
        from litellm_llmrouter.gateway.plugins.evaluator import A2AInvocationContext

        ctx = A2AInvocationContext(
            agent_id="agent-1",
            agent_name="Test Agent",
            agent_url="http://localhost:8080",
            method="message/send",
            request={"message": "hello"},
            result={"response": "hi"},
            success=True,
            error=None,
            duration_ms=456.78,
            span=None,
        )
        assert ctx.agent_id == "agent-1"
        assert ctx.agent_name == "Test Agent"
        assert ctx.method == "message/send"
        assert ctx.success is True


class TestEvaluatorPlugin:
    """Test the EvaluatorPlugin abstract base class."""

    def test_evaluator_metadata_has_evaluator_capability(self):
        """Test that EvaluatorPlugin declares EVALUATOR capability by default."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluatorPlugin,
            EvaluationResult,
        )
        from litellm_llmrouter.gateway.plugin_manager import PluginCapability

        class TestEvaluator(EvaluatorPlugin):
            async def evaluate_mcp_result(self, context):
                return EvaluationResult(score=1.0)

            async def evaluate_a2a_result(self, context):
                return EvaluationResult(score=1.0)

        evaluator = TestEvaluator()
        assert PluginCapability.EVALUATOR in evaluator.metadata.capabilities
        assert evaluator.metadata.priority == 2000

    @pytest.mark.asyncio
    async def test_evaluator_startup_and_shutdown(self):
        """Test that startup and shutdown can be called."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluatorPlugin,
            EvaluationResult,
        )
        from litellm_llmrouter.gateway.plugin_manager import PluginContext

        class TestEvaluator(EvaluatorPlugin):
            async def evaluate_mcp_result(self, context):
                return EvaluationResult()

            async def evaluate_a2a_result(self, context):
                return EvaluationResult()

        evaluator = TestEvaluator()
        mock_app = MagicMock()
        context = PluginContext()

        # Should not raise
        await evaluator.startup(mock_app, context)
        await evaluator.shutdown(mock_app, context)


class TestEvaluatorRegistry:
    """Test evaluator plugin registration and execution."""

    def setup_method(self):
        """Clear registered evaluators before each test."""
        from litellm_llmrouter.gateway.plugins.evaluator import clear_evaluator_plugins

        clear_evaluator_plugins()

    def teardown_method(self):
        """Clear registered evaluators after each test."""
        from litellm_llmrouter.gateway.plugins.evaluator import clear_evaluator_plugins

        clear_evaluator_plugins()

    def test_register_evaluator(self):
        """Test that evaluators can be registered."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluatorPlugin,
            EvaluationResult,
            register_evaluator,
            get_evaluator_plugins,
        )

        class TestEvaluator(EvaluatorPlugin):
            async def evaluate_mcp_result(self, context):
                return EvaluationResult()

            async def evaluate_a2a_result(self, context):
                return EvaluationResult()

        evaluator = TestEvaluator()
        register_evaluator(evaluator)

        plugins = get_evaluator_plugins()
        assert len(plugins) == 1
        assert evaluator in plugins

    def test_is_evaluator_enabled_default_false(self):
        """Test that evaluator is disabled by default."""
        from litellm_llmrouter.gateway.plugins.evaluator import is_evaluator_enabled

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ROUTEIQ_EVALUATOR_ENABLED", None)
            assert is_evaluator_enabled() is False

    def test_is_evaluator_enabled_when_set(self):
        """Test that evaluator can be enabled via env var."""
        from litellm_llmrouter.gateway.plugins.evaluator import is_evaluator_enabled

        with patch.dict(os.environ, {"ROUTEIQ_EVALUATOR_ENABLED": "true"}):
            assert is_evaluator_enabled() is True

        with patch.dict(os.environ, {"ROUTEIQ_EVALUATOR_ENABLED": "false"}):
            assert is_evaluator_enabled() is False

    @pytest.mark.asyncio
    async def test_run_mcp_evaluators_when_disabled(self):
        """Test that run_mcp_evaluators returns empty when disabled."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            MCPInvocationContext,
            run_mcp_evaluators,
        )

        ctx = MCPInvocationContext(
            tool_name="test",
            server_id="s1",
            server_name="Server",
            arguments={},
            result=None,
            success=True,
            error=None,
            duration_ms=0,
        )

        with patch.dict(os.environ, {"ROUTEIQ_EVALUATOR_ENABLED": "false"}):
            results = await run_mcp_evaluators(ctx)
            assert results == []

    @pytest.mark.asyncio
    async def test_run_mcp_evaluators_calls_plugins(self):
        """Test that run_mcp_evaluators calls all registered plugins."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluatorPlugin,
            EvaluationResult,
            MCPInvocationContext,
            register_evaluator,
            run_mcp_evaluators,
        )

        called = []

        class TestEvaluator(EvaluatorPlugin):
            async def evaluate_mcp_result(self, context):
                called.append(context.tool_name)
                return EvaluationResult(score=0.9, status="success")

            async def evaluate_a2a_result(self, context):
                return EvaluationResult()

        register_evaluator(TestEvaluator())

        ctx = MCPInvocationContext(
            tool_name="my-tool",
            server_id="s1",
            server_name="Server",
            arguments={},
            result=None,
            success=True,
            error=None,
            duration_ms=100,
        )

        with patch.dict(os.environ, {"ROUTEIQ_EVALUATOR_ENABLED": "true"}):
            results = await run_mcp_evaluators(ctx)

        assert len(results) == 1
        assert results[0].score == 0.9
        assert "my-tool" in called

    @pytest.mark.asyncio
    async def test_run_a2a_evaluators_calls_plugins(self):
        """Test that run_a2a_evaluators calls all registered plugins."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluatorPlugin,
            EvaluationResult,
            A2AInvocationContext,
            register_evaluator,
            run_a2a_evaluators,
        )

        called = []

        class TestEvaluator(EvaluatorPlugin):
            async def evaluate_mcp_result(self, context):
                return EvaluationResult()

            async def evaluate_a2a_result(self, context):
                called.append(context.agent_id)
                return EvaluationResult(score=0.85, status="success")

        register_evaluator(TestEvaluator())

        ctx = A2AInvocationContext(
            agent_id="agent-1",
            agent_name="Agent",
            agent_url="http://localhost",
            method="message/send",
            request={},
            result=None,
            success=True,
            error=None,
            duration_ms=200,
        )

        with patch.dict(os.environ, {"ROUTEIQ_EVALUATOR_ENABLED": "true"}):
            results = await run_a2a_evaluators(ctx)

        assert len(results) == 1
        assert results[0].score == 0.85
        assert "agent-1" in called

    @pytest.mark.asyncio
    async def test_evaluator_error_handled_gracefully(self):
        """Test that evaluator errors don't crash the system."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluatorPlugin,
            EvaluationResult,
            MCPInvocationContext,
            register_evaluator,
            run_mcp_evaluators,
        )

        class FailingEvaluator(EvaluatorPlugin):
            async def evaluate_mcp_result(self, context):
                raise RuntimeError("Evaluation failed!")

            async def evaluate_a2a_result(self, context):
                return EvaluationResult()

        register_evaluator(FailingEvaluator())

        ctx = MCPInvocationContext(
            tool_name="test",
            server_id="s1",
            server_name="Server",
            arguments={},
            result=None,
            success=True,
            error=None,
            duration_ms=0,
        )

        with patch.dict(os.environ, {"ROUTEIQ_EVALUATOR_ENABLED": "true"}):
            results = await run_mcp_evaluators(ctx)

        # Should return error result, not crash
        assert len(results) == 1
        assert results[0].status == "error"
        assert "Evaluation failed!" in results[0].error


class TestAddEvaluationAttributes:
    """Test OTEL attribute addition."""

    def test_add_evaluation_attributes_to_span(self):
        """Test that evaluation attributes are added to span."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluationResult,
            add_evaluation_attributes,
            ATTR_EVAL_PLUGIN,
            ATTR_EVAL_SCORE,
            ATTR_EVAL_STATUS,
        )

        mock_span = MagicMock()

        result = EvaluationResult(score=0.95, status="success")
        add_evaluation_attributes(mock_span, "test-plugin", result, 50.0, "mcp")

        # Verify attributes were set
        calls = {
            call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list
        }
        assert calls[ATTR_EVAL_PLUGIN] == "test-plugin"
        assert calls[ATTR_EVAL_SCORE] == 0.95
        assert calls[ATTR_EVAL_STATUS] == "success"

    def test_add_evaluation_attributes_with_error(self):
        """Test that error attribute is added when present."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluationResult,
            add_evaluation_attributes,
            ATTR_EVAL_ERROR,
        )

        mock_span = MagicMock()

        result = EvaluationResult(status="error", error="Something failed")
        add_evaluation_attributes(mock_span, "test-plugin", result, 10.0, "a2a")

        calls = {
            call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list
        }
        assert ATTR_EVAL_ERROR in calls
        assert calls[ATTR_EVAL_ERROR] == "Something failed"

    def test_add_evaluation_attributes_handles_none_span(self):
        """Test that None span doesn't cause errors."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluationResult,
            add_evaluation_attributes,
        )

        result = EvaluationResult(score=0.5)
        # Should not raise
        add_evaluation_attributes(None, "plugin", result, 10.0, "mcp")


class TestUpskillEvaluatorPlugin:
    """Test the UpskillEvaluatorPlugin reference implementation."""

    def setup_method(self):
        """Clear evaluators before each test."""
        from litellm_llmrouter.gateway.plugins.evaluator import clear_evaluator_plugins

        clear_evaluator_plugins()

    def teardown_method(self):
        """Clear evaluators after each test."""
        from litellm_llmrouter.gateway.plugins.evaluator import clear_evaluator_plugins

        clear_evaluator_plugins()

    def test_upskill_plugin_metadata(self):
        """Test UpskillEvaluatorPlugin metadata."""
        from litellm_llmrouter.gateway.plugins.upskill_evaluator import (
            UpskillEvaluatorPlugin,
        )
        from litellm_llmrouter.gateway.plugin_manager import PluginCapability

        plugin = UpskillEvaluatorPlugin()
        assert plugin.metadata.name == "upskill-evaluator"
        assert plugin.metadata.version == "1.0.0"
        assert PluginCapability.EVALUATOR in plugin.metadata.capabilities

    @pytest.mark.asyncio
    async def test_upskill_plugin_startup_registers_evaluator(self):
        """Test that startup registers the plugin as an evaluator."""
        from litellm_llmrouter.gateway.plugins.upskill_evaluator import (
            UpskillEvaluatorPlugin,
        )
        from litellm_llmrouter.gateway.plugins.evaluator import get_evaluator_plugins
        from litellm_llmrouter.gateway.plugin_manager import PluginContext

        plugin = UpskillEvaluatorPlugin()
        await plugin.startup(MagicMock(), PluginContext())

        plugins = get_evaluator_plugins()
        assert plugin in plugins

    @pytest.mark.asyncio
    async def test_upskill_plugin_basic_mcp_evaluation(self):
        """Test basic MCP evaluation without upskill enabled."""
        from litellm_llmrouter.gateway.plugins.upskill_evaluator import (
            UpskillEvaluatorPlugin,
        )
        from litellm_llmrouter.gateway.plugins.evaluator import MCPInvocationContext
        from litellm_llmrouter.gateway.plugin_manager import PluginContext

        with patch.dict(os.environ, {"ROUTEIQ_UPSKILL_ENABLED": "false"}, clear=False):
            plugin = UpskillEvaluatorPlugin()
            await plugin.startup(MagicMock(), PluginContext())

            # Success case
            ctx = MCPInvocationContext(
                tool_name="test-tool",
                server_id="s1",
                server_name="Server",
                arguments={"arg": "value"},
                result={"output": "result"},
                success=True,
                error=None,
                duration_ms=100,
            )

            result = await plugin.evaluate_mcp_result(ctx)
            assert result.score == 1.0
            assert result.status == "success"
            assert result.metadata["source"] == "basic"

            # Failure case
            ctx_fail = MCPInvocationContext(
                tool_name="test-tool",
                server_id="s1",
                server_name="Server",
                arguments={},
                result=None,
                success=False,
                error="Tool failed",
                duration_ms=50,
            )

            result_fail = await plugin.evaluate_mcp_result(ctx_fail)
            assert result_fail.score == 0.0
            assert result_fail.status == "error"

    @pytest.mark.asyncio
    async def test_upskill_plugin_basic_a2a_evaluation(self):
        """Test basic A2A evaluation without upskill enabled."""
        from litellm_llmrouter.gateway.plugins.upskill_evaluator import (
            UpskillEvaluatorPlugin,
        )
        from litellm_llmrouter.gateway.plugins.evaluator import A2AInvocationContext
        from litellm_llmrouter.gateway.plugin_manager import PluginContext

        with patch.dict(os.environ, {"ROUTEIQ_UPSKILL_ENABLED": "false"}, clear=False):
            plugin = UpskillEvaluatorPlugin()
            await plugin.startup(MagicMock(), PluginContext())

            ctx = A2AInvocationContext(
                agent_id="agent-1",
                agent_name="Test Agent",
                agent_url="http://localhost:8080",
                method="message/send",
                request={},
                result={"response": "ok"},
                success=True,
                error=None,
                duration_ms=200,
            )

            result = await plugin.evaluate_a2a_result(ctx)
            assert result.score == 1.0
            assert result.status == "success"
            assert result.metadata["source"] == "basic"
            assert result.metadata["agent_id"] == "agent-1"

    @pytest.mark.asyncio
    async def test_upskill_plugin_validates_endpoint_url(self):
        """Test that endpoint URL is validated for SSRF."""
        from litellm_llmrouter.gateway.plugins.upskill_evaluator import (
            UpskillEvaluatorPlugin,
        )
        from litellm_llmrouter.gateway.plugins.evaluator import MCPInvocationContext
        from litellm_llmrouter.gateway.plugin_manager import PluginContext

        # Create a validator that blocks all URLs
        def blocking_validator(url: str) -> str:
            raise ValueError("SSRF blocked!")

        with patch.dict(
            os.environ,
            {
                "ROUTEIQ_UPSKILL_ENABLED": "true",
                "ROUTEIQ_UPSKILL_ENDPOINT": "http://malicious.example.com",
            },
            clear=False,
        ):
            plugin = UpskillEvaluatorPlugin()
            ctx = PluginContext(validate_outbound_url=blocking_validator)
            await plugin.startup(MagicMock(), ctx)

            mcp_ctx = MCPInvocationContext(
                tool_name="test",
                server_id="s1",
                server_name="Server",
                arguments={},
                result=None,
                success=True,
                error=None,
                duration_ms=100,
            )

            # Should fall back to basic scoring, not fail
            result = await plugin.evaluate_mcp_result(mcp_ctx)
            # Will return fallback because service call is blocked
            assert result.score is not None

    @pytest.mark.asyncio
    async def test_upskill_plugin_shutdown(self):
        """Test that shutdown cleans up properly."""
        from litellm_llmrouter.gateway.plugins.upskill_evaluator import (
            UpskillEvaluatorPlugin,
        )
        from litellm_llmrouter.gateway.plugin_manager import PluginContext

        plugin = UpskillEvaluatorPlugin()
        await plugin.startup(MagicMock(), PluginContext())
        await plugin.shutdown(MagicMock(), None)

        # Config should be cleared
        assert plugin._config is None
        assert plugin._context is None


class TestEvaluatorIntegration:
    """Integration tests for evaluator hooks with tracing."""

    def setup_method(self):
        """Clear evaluators before each test."""
        from litellm_llmrouter.gateway.plugins.evaluator import clear_evaluator_plugins

        clear_evaluator_plugins()

    def teardown_method(self):
        """Clear evaluators after each test."""
        from litellm_llmrouter.gateway.plugins.evaluator import clear_evaluator_plugins

        clear_evaluator_plugins()

    @pytest.mark.asyncio
    async def test_mcp_evaluator_hook_integration(self, shared_span_exporter):
        """Test that MCP evaluator hooks add attributes to spans."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluatorPlugin,
            EvaluationResult,
            MCPInvocationContext,
            register_evaluator,
            run_mcp_evaluators,
            ATTR_EVAL_PLUGIN,
            ATTR_EVAL_SCORE,
        )
        from opentelemetry import trace

        class ScoringEvaluator(EvaluatorPlugin):
            async def evaluate_mcp_result(self, context):
                return EvaluationResult(score=0.95, status="success")

            async def evaluate_a2a_result(self, context):
                return EvaluationResult()

        register_evaluator(ScoringEvaluator())

        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("test-mcp-span") as span:
            ctx = MCPInvocationContext(
                tool_name="test-tool",
                server_id="s1",
                server_name="Server",
                arguments={},
                result=None,
                success=True,
                error=None,
                duration_ms=100,
                span=span,
            )

            with patch.dict(os.environ, {"ROUTEIQ_EVALUATOR_ENABLED": "true"}):
                await run_mcp_evaluators(ctx)

        # Check that span has evaluation attributes
        spans = shared_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs.get(ATTR_EVAL_PLUGIN) == "ScoringEvaluator"
        assert attrs.get(ATTR_EVAL_SCORE) == 0.95

    @pytest.mark.asyncio
    async def test_a2a_evaluator_hook_integration(self, shared_span_exporter):
        """Test that A2A evaluator hooks add attributes to spans."""
        from litellm_llmrouter.gateway.plugins.evaluator import (
            EvaluatorPlugin,
            EvaluationResult,
            A2AInvocationContext,
            register_evaluator,
            run_a2a_evaluators,
            ATTR_EVAL_PLUGIN,
            ATTR_EVAL_STATUS,
        )
        from opentelemetry import trace

        class A2AEvaluator(EvaluatorPlugin):
            async def evaluate_mcp_result(self, context):
                return EvaluationResult()

            async def evaluate_a2a_result(self, context):
                return EvaluationResult(score=0.88, status="success")

        register_evaluator(A2AEvaluator())

        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("test-a2a-span") as span:
            ctx = A2AInvocationContext(
                agent_id="agent-1",
                agent_name="Agent",
                agent_url="http://localhost",
                method="message/send",
                request={},
                result=None,
                success=True,
                error=None,
                duration_ms=200,
                span=span,
            )

            with patch.dict(os.environ, {"ROUTEIQ_EVALUATOR_ENABLED": "true"}):
                await run_a2a_evaluators(ctx)

        spans = shared_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs.get(ATTR_EVAL_PLUGIN) == "A2AEvaluator"
        assert attrs.get(ATTR_EVAL_STATUS) == "success"
