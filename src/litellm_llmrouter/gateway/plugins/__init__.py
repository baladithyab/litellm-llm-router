"""
Gateway plugins package.

This package contains built-in plugins that can be enabled via LLMROUTER_PLUGINS.
"""

from litellm_llmrouter.gateway.plugins.evaluator import (
    A2AInvocationContext,
    EvaluationResult,
    EvaluatorPlugin,
    MCPInvocationContext,
    add_evaluation_attributes,
    clear_evaluator_plugins,
    get_evaluator_plugins,
    is_evaluator_enabled,
    register_evaluator,
    run_a2a_evaluators,
    run_mcp_evaluators,
)
from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsDiscoveryPlugin
from litellm_llmrouter.gateway.plugins.upskill_evaluator import UpskillEvaluatorPlugin

__all__ = [
    # Skills Discovery
    "SkillsDiscoveryPlugin",
    # Evaluator Framework
    "EvaluatorPlugin",
    "EvaluationResult",
    "MCPInvocationContext",
    "A2AInvocationContext",
    "register_evaluator",
    "get_evaluator_plugins",
    "clear_evaluator_plugins",
    "is_evaluator_enabled",
    "run_mcp_evaluators",
    "run_a2a_evaluators",
    "add_evaluation_attributes",
    # Reference Evaluator
    "UpskillEvaluatorPlugin",
]
