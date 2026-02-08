"""
Unit Tests for Policy Engine (TG5.2)
====================================

Tests for OPA-style policy engine including:
- PolicyRule matching logic
- PolicyConfig loading from YAML/dict
- PolicyEngine evaluation with allow/deny decisions
- Fail-open/fail-closed mode behavior
- Auditable decision records

These tests exercise the policy engine WITHOUT external dependencies.
"""

import pytest
from datetime import datetime

from litellm_llmrouter.policy_engine import (
    PolicyAction,
    PolicyDecision,
    PolicyContext,
    PolicyRule,
    PolicyConfig,
    PolicyEngine,
    reset_policy_engine,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset the global policy engine before each test."""
    reset_policy_engine()
    yield
    reset_policy_engine()


@pytest.fixture
def basic_context() -> PolicyContext:
    """Create a basic policy context for testing."""
    return PolicyContext(
        team_id="team-123",
        user_id="user-456",
        api_key_subject="sk-test...1234",
        route="/v1/chat/completions",
        method="POST",
        model="gpt-4",
        source_ip="192.168.1.100",
        headers={"content-type": "application/json"},
        request_id="req-12345",
    )


@pytest.fixture
def deny_rule() -> PolicyRule:
    """Create a deny rule for testing."""
    return PolicyRule(
        name="deny-premium-models",
        action=PolicyAction.DENY,
        models=["gpt-4*", "claude-3-opus*"],
        teams=["free-tier"],
        reason="Premium models not allowed for free-tier",
        priority=50,
    )


@pytest.fixture
def allow_rule() -> PolicyRule:
    """Create an allow rule for testing."""
    return PolicyRule(
        name="allow-admin",
        action=PolicyAction.ALLOW,
        teams=["admin-team"],
        reason="Admin team has full access",
        priority=10,
    )


# =============================================================================
# PolicyRule Tests
# =============================================================================


class TestPolicyRule:
    """Tests for PolicyRule matching logic."""

    def test_rule_matches_team(self, basic_context):
        """Test rule matching by team ID."""
        rule = PolicyRule(
            name="team-match",
            action=PolicyAction.ALLOW,
            teams=["team-123"],
        )
        assert rule.matches(basic_context) is True

        rule_no_match = PolicyRule(
            name="team-no-match",
            action=PolicyAction.ALLOW,
            teams=["other-team"],
        )
        assert rule_no_match.matches(basic_context) is False

    def test_rule_matches_team_pattern(self, basic_context):
        """Test rule matching with team glob pattern."""
        rule = PolicyRule(
            name="team-pattern",
            action=PolicyAction.ALLOW,
            teams=["team-*"],
        )
        assert rule.matches(basic_context) is True

    def test_rule_matches_user(self, basic_context):
        """Test rule matching by user ID."""
        rule = PolicyRule(
            name="user-match",
            action=PolicyAction.ALLOW,
            users=["user-456"],
        )
        assert rule.matches(basic_context) is True

        rule_no_match = PolicyRule(
            name="user-no-match",
            action=PolicyAction.ALLOW,
            users=["other-user"],
        )
        assert rule_no_match.matches(basic_context) is False

    def test_rule_matches_api_key(self, basic_context):
        """Test rule matching by API key pattern."""
        rule = PolicyRule(
            name="apikey-match",
            action=PolicyAction.ALLOW,
            api_keys=["sk-test*"],
        )
        assert rule.matches(basic_context) is True

    def test_rule_matches_route(self, basic_context):
        """Test rule matching by route pattern."""
        rule = PolicyRule(
            name="route-match",
            action=PolicyAction.ALLOW,
            routes=["/v1/*"],
        )
        assert rule.matches(basic_context) is True

        rule_no_match = PolicyRule(
            name="route-no-match",
            action=PolicyAction.ALLOW,
            routes=["/admin/*"],
        )
        assert rule_no_match.matches(basic_context) is False

    def test_rule_matches_method(self, basic_context):
        """Test rule matching by HTTP method."""
        rule = PolicyRule(
            name="method-match",
            action=PolicyAction.ALLOW,
            methods=["POST", "PUT"],
        )
        assert rule.matches(basic_context) is True

        rule_no_match = PolicyRule(
            name="method-no-match",
            action=PolicyAction.ALLOW,
            methods=["GET"],
        )
        assert rule_no_match.matches(basic_context) is False

    def test_rule_matches_model(self, basic_context):
        """Test rule matching by model pattern."""
        rule = PolicyRule(
            name="model-match",
            action=PolicyAction.ALLOW,
            models=["gpt-*"],
        )
        assert rule.matches(basic_context) is True

        rule_no_match = PolicyRule(
            name="model-no-match",
            action=PolicyAction.ALLOW,
            models=["claude-*"],
        )
        assert rule_no_match.matches(basic_context) is False

    def test_rule_matches_source_ip_exact(self, basic_context):
        """Test rule matching by exact source IP."""
        rule = PolicyRule(
            name="ip-match",
            action=PolicyAction.ALLOW,
            source_ips=["192.168.1.100"],
        )
        assert rule.matches(basic_context) is True

    def test_rule_matches_source_ip_cidr(self, basic_context):
        """Test rule matching by source IP CIDR."""
        rule = PolicyRule(
            name="ip-cidr-match",
            action=PolicyAction.ALLOW,
            source_ips=["192.168.0.0/16"],
        )
        assert rule.matches(basic_context) is True

        rule_no_match = PolicyRule(
            name="ip-cidr-no-match",
            action=PolicyAction.ALLOW,
            source_ips=["10.0.0.0/8"],
        )
        assert rule_no_match.matches(basic_context) is False

    def test_rule_matches_multiple_conditions_and(self, basic_context):
        """Test rule matching with multiple conditions (AND logic)."""
        rule = PolicyRule(
            name="multi-match",
            action=PolicyAction.ALLOW,
            teams=["team-123"],
            routes=["/v1/*"],
            methods=["POST"],
        )
        assert rule.matches(basic_context) is True

        # Fail if any condition doesn't match
        rule_partial_match = PolicyRule(
            name="partial-match",
            action=PolicyAction.ALLOW,
            teams=["team-123"],
            routes=["/v1/*"],
            methods=["GET"],  # Context has POST
        )
        assert rule_partial_match.matches(basic_context) is False

    def test_rule_matches_no_conditions(self, basic_context):
        """Test rule with no conditions matches everything."""
        rule = PolicyRule(
            name="match-all",
            action=PolicyAction.ALLOW,
        )
        assert rule.matches(basic_context) is True

    def test_rule_matches_regex_pattern(self, basic_context):
        """Test rule matching with regex pattern (starts with ^)."""
        rule = PolicyRule(
            name="regex-match",
            action=PolicyAction.ALLOW,
            routes=["^/v1/.*"],
        )
        assert rule.matches(basic_context) is True


# =============================================================================
# PolicyConfig Tests
# =============================================================================


class TestPolicyConfig:
    """Tests for policy configuration loading."""

    def test_config_from_dict(self):
        """Test loading config from dictionary."""
        data = {
            "default_action": "deny",
            "default_reason": "Explicit allow required",
            "rules": [
                {
                    "name": "allow-all-get",
                    "action": "allow",
                    "methods": ["GET"],
                    "priority": 10,
                },
                {
                    "name": "deny-premium",
                    "action": "deny",
                    "models": ["gpt-4*"],
                    "reason": "Premium denied",
                    "priority": 50,
                },
            ],
        }

        config = PolicyConfig.from_dict(data)

        assert config.default_action == PolicyAction.DENY
        assert config.default_reason == "Explicit allow required"
        assert len(config.rules) == 2
        # Rules should be sorted by priority
        assert config.rules[0].name == "allow-all-get"
        assert config.rules[1].name == "deny-premium"

    def test_config_from_yaml(self):
        """Test loading config from YAML string."""
        yaml_content = """
default_action: allow
rules:
  - name: deny-admin
    action: deny
    routes:
      - /admin/*
    reason: Admin access denied
    priority: 20
"""
        config = PolicyConfig.from_yaml(yaml_content)

        assert config.default_action == PolicyAction.ALLOW
        assert len(config.rules) == 1
        assert config.rules[0].name == "deny-admin"
        assert config.rules[0].routes == ["/admin/*"]

    def test_config_excluded_paths(self):
        """Test excluded paths in configuration."""
        data = {
            "excluded_paths": ["/_health/live", "/metrics"],
        }

        config = PolicyConfig.from_dict(data)

        assert "/_health/live" in config.excluded_paths
        assert "/metrics" in config.excluded_paths

    def test_config_sort_rules_by_priority(self):
        """Test that rules are sorted by priority."""
        config = PolicyConfig()
        config.rules = [
            PolicyRule("high", PolicyAction.ALLOW, priority=100),
            PolicyRule("low", PolicyAction.ALLOW, priority=10),
            PolicyRule("mid", PolicyAction.ALLOW, priority=50),
        ]

        config.sort_rules()

        assert config.rules[0].name == "low"
        assert config.rules[1].name == "mid"
        assert config.rules[2].name == "high"


# =============================================================================
# PolicyEngine Tests
# =============================================================================


class TestPolicyEngine:
    """Tests for PolicyEngine evaluation."""

    @pytest.mark.asyncio
    async def test_engine_disabled_allows_all(self, basic_context):
        """Test that disabled engine allows all requests."""
        engine = PolicyEngine(enabled=False)

        decision = await engine.evaluate(basic_context)

        assert decision.allowed is True
        assert decision.action == PolicyAction.ALLOW
        assert decision.policy_name == "__disabled__"

    @pytest.mark.asyncio
    async def test_engine_excluded_path_allows(self, basic_context):
        """Test that excluded paths are always allowed."""
        config = PolicyConfig()
        config.excluded_paths.add("/v1/chat/completions")

        engine = PolicyEngine(config=config, enabled=True)

        decision = await engine.evaluate(basic_context)

        assert decision.allowed is True
        assert decision.policy_name == "__excluded__"

    @pytest.mark.asyncio
    async def test_engine_deny_rule_denies(self, basic_context, deny_rule):
        """Test that a matching deny rule denies the request."""
        # Set context to match deny rule
        basic_context.team_id = "free-tier"
        basic_context.model = "gpt-4-turbo"

        config = PolicyConfig()
        config.rules = [deny_rule]

        engine = PolicyEngine(config=config, enabled=True)

        decision = await engine.evaluate(basic_context)

        assert decision.allowed is False
        assert decision.action == PolicyAction.DENY
        assert decision.policy_name == "deny-premium-models"
        assert decision.reason == "Premium models not allowed for free-tier"

    @pytest.mark.asyncio
    async def test_engine_allow_rule_allows(self, basic_context, allow_rule):
        """Test that a matching allow rule allows the request."""
        basic_context.team_id = "admin-team"

        config = PolicyConfig()
        config.rules = [allow_rule]

        engine = PolicyEngine(config=config, enabled=True)

        decision = await engine.evaluate(basic_context)

        assert decision.allowed is True
        assert decision.action == PolicyAction.ALLOW
        assert decision.policy_name == "allow-admin"

    @pytest.mark.asyncio
    async def test_engine_default_action_when_no_match(self, basic_context):
        """Test that default action is used when no rule matches."""
        config = PolicyConfig()
        config.default_action = PolicyAction.DENY
        config.default_reason = "No explicit allow"

        engine = PolicyEngine(config=config, enabled=True)

        decision = await engine.evaluate(basic_context)

        assert decision.allowed is False
        assert decision.policy_name == "__default__"
        assert decision.reason == "No explicit allow"

    @pytest.mark.asyncio
    async def test_engine_first_matching_rule_wins(
        self, basic_context, allow_rule, deny_rule
    ):
        """Test that first matching rule (by priority) wins."""
        # Context: admin-team, gpt-4
        basic_context.team_id = "admin-team"
        basic_context.model = "gpt-4"

        config = PolicyConfig()
        # Allow rule has priority 10, deny rule has priority 50
        config.rules = [deny_rule, allow_rule]
        config.sort_rules()

        engine = PolicyEngine(config=config, enabled=True)

        decision = await engine.evaluate(basic_context)

        # Allow rule should win because of higher priority (lower number)
        assert decision.allowed is True
        assert decision.policy_name == "allow-admin"

    @pytest.mark.asyncio
    async def test_engine_fail_open_on_error(self, basic_context):
        """Test that fail-open mode allows on errors."""
        config = PolicyConfig()
        # Create a rule that will cause an error during evaluation
        # We'll mock this by injecting a broken rule
        rule = PolicyRule("broken", PolicyAction.DENY)
        rule._matches_patterns = lambda *args: (_ for _ in ()).throw(
            Exception("Test error")
        )
        rule.teams = ["team-*"]  # Force pattern matching
        config.rules = [rule]

        engine = PolicyEngine(config=config, enabled=True, fail_closed=False)

        decision = await engine.evaluate(basic_context)

        assert decision.allowed is True
        assert decision.policy_name == "__error__"

    @pytest.mark.asyncio
    async def test_engine_fail_closed_on_error(self, basic_context):
        """Test that fail-closed mode denies on errors."""
        config = PolicyConfig()
        rule = PolicyRule("broken", PolicyAction.DENY)
        rule._matches_patterns = lambda *args: (_ for _ in ()).throw(
            Exception("Test error")
        )
        rule.teams = ["team-*"]
        config.rules = [rule]

        engine = PolicyEngine(config=config, enabled=True, fail_closed=True)

        decision = await engine.evaluate(basic_context)

        assert decision.allowed is False
        assert decision.action == PolicyAction.ERROR

    @pytest.mark.asyncio
    async def test_engine_decision_includes_context(self, basic_context, allow_rule):
        """Test that decision includes context for audit."""
        basic_context.team_id = "admin-team"

        config = PolicyConfig()
        config.rules = [allow_rule]

        engine = PolicyEngine(config=config, enabled=True)

        decision = await engine.evaluate(basic_context)

        assert decision.route == "/v1/chat/completions"
        assert decision.method == "POST"
        assert decision.model == "gpt-4"
        assert decision.subject["team_id"] == "admin-team"
        assert decision.request_id is not None

    @pytest.mark.asyncio
    async def test_engine_evaluation_time_measured(self, basic_context, allow_rule):
        """Test that evaluation time is measured."""
        basic_context.team_id = "admin-team"

        config = PolicyConfig()
        config.rules = [allow_rule]

        engine = PolicyEngine(config=config, enabled=True)

        decision = await engine.evaluate(basic_context)

        assert decision.evaluation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_engine_reload_config(self, basic_context):
        """Test reloading configuration at runtime."""
        config1 = PolicyConfig()
        config1.default_action = PolicyAction.ALLOW

        engine = PolicyEngine(config=config1, enabled=True)

        decision1 = await engine.evaluate(basic_context)
        assert decision1.allowed is True

        # Reload with different config
        config2 = PolicyConfig()
        config2.default_action = PolicyAction.DENY
        await engine.reload_config(config2)

        decision2 = await engine.evaluate(basic_context)
        assert decision2.allowed is False


# =============================================================================
# PolicyDecision Tests
# =============================================================================


class TestPolicyDecision:
    """Tests for PolicyDecision audit record."""

    def test_decision_to_dict(self):
        """Test that decision can be serialized for audit."""
        decision = PolicyDecision(
            allowed=False,
            action=PolicyAction.DENY,
            reason="Access denied",
            policy_name="test-rule",
            evaluation_time_ms=1.5,
            request_id="req-123",
            route="/v1/completions",
            method="POST",
            model="gpt-4",
            subject={"team_id": "team-1"},
        )

        data = decision.to_dict()

        assert data["allowed"] is False
        assert data["action"] == "deny"
        assert data["reason"] == "Access denied"
        assert data["policy_name"] == "test-rule"
        assert data["request_id"] == "req-123"
        assert data["route"] == "/v1/completions"
        assert data["model"] == "gpt-4"
        assert data["subject"]["team_id"] == "team-1"

    def test_decision_timestamp(self):
        """Test that decision has a timestamp."""
        decision = PolicyDecision(
            allowed=True,
            action=PolicyAction.ALLOW,
        )

        assert decision.timestamp is not None
        assert isinstance(decision.timestamp, datetime)


# =============================================================================
# Integration-style Unit Tests
# =============================================================================


class TestPolicyEngineScenarios:
    """Integration-style tests for real-world scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_free_tier_premium_model_denied(self):
        """Test that free tier teams cannot use premium models."""
        config = PolicyConfig.from_dict(
            {
                "default_action": "allow",
                "rules": [
                    {
                        "name": "deny-premium-for-free",
                        "action": "deny",
                        "teams": ["free-*", "trial-*"],
                        "models": ["gpt-4*", "claude-3-opus*"],
                        "reason": "Premium models require paid plan",
                        "priority": 50,
                    },
                ],
            }
        )

        engine = PolicyEngine(config=config, enabled=True)

        # Free tier trying to use GPT-4 - should be denied
        context = PolicyContext(
            team_id="free-tier-123",
            route="/v1/chat/completions",
            method="POST",
            model="gpt-4-turbo",
        )

        decision = await engine.evaluate(context)

        assert decision.allowed is False
        assert "paid plan" in decision.reason

        # Same team using GPT-3.5 - should be allowed
        context.model = "gpt-3.5-turbo"

        decision2 = await engine.evaluate(context)

        assert decision2.allowed is True

    @pytest.mark.asyncio
    async def test_scenario_admin_routes_protected(self):
        """Test that admin routes are protected from user API keys."""
        config = PolicyConfig.from_dict(
            {
                "default_action": "allow",
                "rules": [
                    {
                        "name": "admin-bypass",
                        "action": "allow",
                        "api_keys": ["sk-admin-*"],
                        "routes": ["/admin/*", "/config/*"],
                        "priority": 10,
                    },
                    {
                        "name": "block-admin-routes",
                        "action": "deny",
                        "routes": ["/admin/*", "/config/*"],
                        "reason": "Admin routes require admin API key",
                        "priority": 50,
                    },
                ],
            }
        )

        engine = PolicyEngine(config=config, enabled=True)

        # User API key trying to access admin route - denied
        context = PolicyContext(
            api_key_subject="sk-user-abc123",
            route="/admin/reload",
            method="POST",
        )

        decision = await engine.evaluate(context)

        assert decision.allowed is False

        # Admin API key accessing admin route - allowed
        context.api_key_subject = "sk-admin-xyz789"

        decision2 = await engine.evaluate(context)

        assert decision2.allowed is True

    @pytest.mark.asyncio
    async def test_scenario_ip_allowlist(self):
        """Test IP-based access control."""
        config = PolicyConfig.from_dict(
            {
                "default_action": "deny",
                "default_reason": "Access denied from external network",
                "rules": [
                    {
                        "name": "allow-internal",
                        "action": "allow",
                        "source_ips": ["10.0.0.0/8", "172.16.0.0/12"],
                        "priority": 10,
                    },
                ],
            }
        )

        engine = PolicyEngine(config=config, enabled=True)

        # Internal IP - allowed
        context = PolicyContext(
            route="/v1/chat/completions",
            method="POST",
            source_ip="10.50.100.200",
        )

        decision = await engine.evaluate(context)

        assert decision.allowed is True

        # External IP - denied
        context.source_ip = "203.0.113.50"

        decision2 = await engine.evaluate(context)

        assert decision2.allowed is False
        assert "external network" in decision2.reason
