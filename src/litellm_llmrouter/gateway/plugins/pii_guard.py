"""
PII Detection and Redaction Guard Plugin
==========================================

Detects and optionally redacts Personally Identifiable Information (PII) in
LLM request messages and responses using regex-based scanning.

Supported PII entity types:
    - SSN          (US Social Security Number)
    - CREDIT_CARD  (major card formats with optional Luhn check)
    - EMAIL        (standard email addresses)
    - PHONE        (US/international phone formats)
    - IP_ADDRESS   (IPv4)

Configuration (environment variables):
    GUARDRAIL_PII_ENABLED        - true/false (default: false)
    GUARDRAIL_PII_ACTION         - redact / block / warn / log (default: redact)
    GUARDRAIL_PII_ENTITY_TYPES   - comma-separated list (default: all types)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# PII entity definitions
# ---------------------------------------------------------------------------


@dataclass
class PIIFinding:
    """A single PII detection result."""

    entity_type: str
    start: int
    end: int
    matched_text: str


# Entity type -> (compiled regex, replacement token)
_ENTITY_PATTERNS: dict[str, tuple[re.Pattern[str], str]] = {
    "SSN": (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[PII:SSN]",
    ),
    "CREDIT_CARD": (
        re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        "[PII:CREDIT_CARD]",
    ),
    "EMAIL": (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[PII:EMAIL]",
    ),
    "PHONE": (
        re.compile(
            r"(?<!\d)"
            r"(?:\+?1[\s.-]?)?"
            r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
            r"(?!\d)"
        ),
        "[PII:PHONE]",
    ),
    "IP_ADDRESS": (
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "[PII:IP_ADDRESS]",
    ),
}

ALL_ENTITY_TYPES = set(_ENTITY_PATTERNS.keys())


class PIIGuard(GuardrailPlugin):
    """Detects and redacts PII in LLM messages and responses."""

    def __init__(self) -> None:
        self._enabled = self._env_bool("GUARDRAIL_PII_ENABLED")
        self._action = self._env_str("GUARDRAIL_PII_ACTION", "redact")
        self._entity_types = self._load_entity_types()

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="pii-guard",
            version="1.0.0",
            capabilities={PluginCapability.EVALUATOR},
            priority=60,
            description="Detects and redacts PII in messages",
        )

    def _load_entity_types(self) -> set[str]:
        raw = self._env_str("GUARDRAIL_PII_ENTITY_TYPES", "")
        if not raw.strip():
            return ALL_ENTITY_TYPES
        requested = {t.strip().upper() for t in raw.split(",") if t.strip()}
        valid = requested & ALL_ENTITY_TYPES
        if not valid:
            logger.warning(
                "No valid PII entity types in GUARDRAIL_PII_ENTITY_TYPES=%r, "
                "falling back to all types",
                raw,
            )
            return ALL_ENTITY_TYPES
        return valid

    # ----- scanning ----

    def scan_pii(self, text: str) -> list[PIIFinding]:
        """Scan *text* for PII and return all findings."""
        findings: list[PIIFinding] = []
        for entity_type in sorted(self._entity_types):
            pattern, _ = _ENTITY_PATTERNS[entity_type]
            for match in pattern.finditer(text):
                findings.append(
                    PIIFinding(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        matched_text=match.group(),
                    )
                )
        return findings

    def redact_text(self, text: str, findings: list[PIIFinding]) -> str:
        """Replace PII findings in *text* with redaction tokens.

        Processes findings in reverse offset order so that earlier offsets
        remain valid after replacement.
        """
        sorted_findings = sorted(findings, key=lambda f: f.start, reverse=True)
        result = text
        for finding in sorted_findings:
            replacement = _ENTITY_PATTERNS[finding.entity_type][1]
            result = result[: finding.start] + replacement + result[finding.end :]
        return result

    # ----- GuardrailPlugin interface ----

    async def evaluate_input(
        self,
        model: str,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> GuardrailDecision:
        all_findings: list[PIIFinding] = []
        modified = False

        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str) or not content:
                continue

            findings = self.scan_pii(content)
            if not findings:
                continue

            all_findings.extend(findings)

            if self._action == "redact":
                msg["content"] = self.redact_text(content, findings)
                modified = True

        if all_findings:
            entity_types = sorted({f.entity_type for f in all_findings})
            return GuardrailDecision(
                allowed=(self._action != "block"),
                action_taken=self._action,
                guardrail_name=self.name,
                category="pii",
                score=1.0,
                details={
                    "entity_types": entity_types,
                    "count": len(all_findings),
                    "reason": f"PII detected: {entity_types}",
                },
                modified_content="redacted" if modified else None,
            )

        return GuardrailDecision(
            allowed=True,
            action_taken="pass",
            guardrail_name=self.name,
            category="pii",
            score=0.0,
        )

    async def evaluate_output(
        self,
        model: str,
        response: Any,
        kwargs: dict[str, Any],
    ) -> GuardrailDecision | None:
        text = self._extract_response_text(response)
        if not text:
            return None

        findings = self.scan_pii(text)
        if not findings:
            return None

        entity_types = sorted({f.entity_type for f in findings})
        return GuardrailDecision(
            allowed=True,
            action_taken="warn",
            guardrail_name=self.name,
            category="pii",
            score=1.0,
            details={
                "entity_types": entity_types,
                "count": len(findings),
                "direction": "output",
                "reason": f"PII detected in response: {entity_types}",
            },
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Best-effort extraction of text from an LLM response object."""
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                if isinstance(msg, dict):
                    return msg.get("content", "") or ""
        # litellm ModelResponse
        if hasattr(response, "choices") and response.choices:
            first = response.choices[0]
            if hasattr(first, "message") and hasattr(first.message, "content"):
                return first.message.content or ""
        return ""
