"""
Prompt Injection Guard Plugin
==============================

Detects and blocks prompt injection attempts in user messages using
regex-based pattern matching.

Detection layers:
    1. Fast regex scan for known injection patterns (~1 ms)

Configuration (environment variables):
    GUARDRAIL_INJECTION_ENABLED    - true/false (default: false)
    GUARDRAIL_INJECTION_ACTION     - block / warn / log (default: block)
    GUARDRAIL_INJECTION_PATTERNS   - JSON list of extra regex patterns (optional)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from litellm_llmrouter.gateway.plugin_manager import (
    PluginCapability,
    PluginMetadata,
)
from litellm_llmrouter.gateway.plugins.guardrails_base import (
    GuardrailDecision,
    GuardrailPlugin,
)

logger = logging.getLogger(__name__)

# Built-in injection patterns (case-insensitive).
# Each tuple is (pattern, human-readable label).
_DEFAULT_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "ignore_previous_instructions"),
    (r"ignore\s+(all\s+)?prior\s+instructions", "ignore_prior_instructions"),
    (r"disregard\s+(all\s+)?previous\s+instructions", "disregard_previous"),
    (r"forget\s+(all\s+)?previous\s+instructions", "forget_previous"),
    (r"override\s+(all\s+)?previous\s+instructions", "override_previous"),
    (r"you\s+are\s+now\s+(?:DAN|an?\s+unrestricted)", "role_hijack_dan"),
    (r"act\s+as\s+(?:if\s+you\s+have\s+no\s+restrictions|DAN)", "role_hijack_act"),
    (
        r"pretend\s+(?:you\s+are|to\s+be)\s+(?:an?\s+)?(?:evil|unrestricted|unfiltered)",
        "role_hijack_pretend",
    ),
    (r"system\s*prompt\s*(?:override|injection|:)", "system_prompt_override"),
    (r"your\s+(?:true|real|actual)\s+instructions\s+are", "instruction_override"),
    (
        r"(?:reveal|show|print|output)\s+(?:your\s+)?system\s*prompt",
        "system_prompt_extraction",
    ),
    (r"do\s+anything\s+now", "dan_prompt"),
    (r"developer\s+mode\s+enabled", "developer_mode"),
    (r"jailbreak", "jailbreak_keyword"),
    (r"\[system\]", "fake_system_tag"),
    (r"<<\s*SYS\s*>>", "fake_llama_system_tag"),
]


class PromptInjectionGuard(GuardrailPlugin):
    """Detects and blocks prompt injection attempts in user messages.

    Uses regex-based fast scanning for known injection patterns.
    """

    def __init__(self) -> None:
        self._enabled = self._env_bool("GUARDRAIL_INJECTION_ENABLED")
        self._action = self._env_str("GUARDRAIL_INJECTION_ACTION", "block")
        self._compiled_patterns = self._compile_patterns()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="prompt-injection-guard",
            version="1.0.0",
            capabilities={PluginCapability.EVALUATOR},
            priority=50,
            description="Detects and blocks prompt injection attempts",
        )

    def _compile_patterns(self) -> list[tuple[re.Pattern[str], str]]:
        """Compile the default patterns into regex objects."""
        compiled: list[tuple[re.Pattern[str], str]] = []
        for pattern, label in _DEFAULT_PATTERNS:
            try:
                compiled.append((re.compile(pattern, re.IGNORECASE), label))
            except re.error as exc:
                logger.warning("Invalid injection pattern %r: %s", pattern, exc)
        return compiled

    def scan(self, text: str) -> tuple[float, str]:
        """Scan *text* for injection patterns.

        Returns:
            (score, matched_label).  Score is 1.0 on match, 0.0 otherwise.
        """
        for regex, label in self._compiled_patterns:
            if regex.search(text):
                return 1.0, label
        return 0.0, ""

    # ----- GuardrailPlugin interface ----

    async def evaluate_input(
        self,
        model: str,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> GuardrailDecision:
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or not content:
                continue

            score, label = self.scan(content)
            if score > 0.0:
                return GuardrailDecision(
                    allowed=(self._action != "block"),
                    action_taken=self._action,
                    guardrail_name=self.name,
                    category="injection",
                    score=score,
                    details={
                        "pattern_matched": label,
                        "reason": f"Prompt injection detected: {label}",
                    },
                )

        return GuardrailDecision(
            allowed=True,
            action_taken="pass",
            guardrail_name=self.name,
            category="injection",
            score=0.0,
        )
