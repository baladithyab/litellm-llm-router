"""
Unit Tests for Policy Engine Responses API Support
===================================================

Tests for Responses API and emerging LLM API surface support in:
- LLM API path-to-type mapping
- Model extraction from different API body formats
- Policy evaluation with api_type in context
- Telemetry contracts with api_type field
- Backward compatibility for callers that do not supply api_type
"""

import json

import pytest

from litellm_llmrouter.policy_engine import (
    LLM_API_PATHS,
    PolicyConfig,
    PolicyContext,
    PolicyEngine,
    extract_model_from_body,
    get_api_type_for_path,
    reset_policy_engine,
)
from litellm_llmrouter.telemetry_contracts import (
    RouterDecisionEventBuilder,
    RouterDecisionInput,
    RoutingOutcome,
    RoutingOutcomeData,
    extract_router_decision_from_span_event,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the global policy engine before each test."""
    reset_policy_engine()
    yield
    reset_policy_engine()


# =============================================================================
# LLM_API_PATHS mapping tests
# =============================================================================


class TestLLMAPIPaths:
    """Tests for the LLM API path registry."""

    def test_chat_completions_paths(self):
        assert LLM_API_PATHS["/v1/chat/completions"] == "chat_completion"
        assert LLM_API_PATHS["/chat/completions"] == "chat_completion"

    def test_responses_paths(self):
        assert LLM_API_PATHS["/v1/responses"] == "responses"
        assert LLM_API_PATHS["/responses"] == "responses"
        assert LLM_API_PATHS["/openai/v1/responses"] == "responses"

    def test_embeddings_paths(self):
        assert LLM_API_PATHS["/v1/embeddings"] == "embedding"
        assert LLM_API_PATHS["/embeddings"] == "embedding"

    def test_completions_paths(self):
        assert LLM_API_PATHS["/v1/completions"] == "completion"
        assert LLM_API_PATHS["/completions"] == "completion"


class TestGetApiTypeForPath:
    """Tests for get_api_type_for_path helper."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/v1/chat/completions", "chat_completion"),
            ("/chat/completions", "chat_completion"),
            ("/v1/responses", "responses"),
            ("/responses", "responses"),
            ("/openai/v1/responses", "responses"),
            ("/v1/embeddings", "embedding"),
            ("/embeddings", "embedding"),
            ("/v1/completions", "completion"),
            ("/completions", "completion"),
        ],
    )
    def test_known_paths(self, path: str, expected: str):
        assert get_api_type_for_path(path) == expected

    @pytest.mark.parametrize(
        "path",
        [
            "/_health/live",
            "/admin/reload",
            "/v1/models",
            "/mcp",
            "/a2a/agents",
            "/unknown/path",
            "",
        ],
    )
    def test_unknown_paths_return_none(self, path: str):
        assert get_api_type_for_path(path) is None


# =============================================================================
# Model extraction tests
# =============================================================================


class TestExtractModelFromBody:
    """Tests for extract_model_from_body across API surfaces."""

    def test_chat_completions_body(self):
        body = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        assert extract_model_from_body(body, "chat_completion") == "gpt-4-turbo"

    def test_responses_api_body(self):
        body = {
            "model": "gpt-4o",
            "input": "Tell me a joke",
        }
        assert extract_model_from_body(body, "responses") == "gpt-4o"

    def test_responses_api_structured_input(self):
        body = {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "Summarize this"},
            ],
        }
        assert extract_model_from_body(body, "responses") == "gpt-4o"

    def test_embeddings_body(self):
        body = {
            "model": "text-embedding-ada-002",
            "input": "Some text to embed",
        }
        assert extract_model_from_body(body, "embedding") == "text-embedding-ada-002"

    def test_completions_body(self):
        body = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Once upon a time",
        }
        assert extract_model_from_body(body, "completion") == "gpt-3.5-turbo-instruct"

    def test_missing_model_field(self):
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        assert extract_model_from_body(body, "chat_completion") is None

    def test_empty_body(self):
        assert extract_model_from_body({}, "chat_completion") is None

    def test_non_dict_body(self):
        assert extract_model_from_body("not a dict", "chat_completion") is None  # type: ignore[arg-type]
        assert extract_model_from_body(None, "chat_completion") is None  # type: ignore[arg-type]
        assert extract_model_from_body([], "responses") is None  # type: ignore[arg-type]

    def test_none_api_type(self):
        body = {"model": "gpt-4"}
        assert extract_model_from_body(body, None) == "gpt-4"


# =============================================================================
# PolicyContext with api_type
# =============================================================================


class TestPolicyContextApiType:
    """Tests for PolicyContext with the api_type field."""

    def test_api_type_default_none(self):
        ctx = PolicyContext()
        assert ctx.api_type is None

    def test_api_type_set(self):
        ctx = PolicyContext(
            route="/v1/responses",
            method="POST",
            model="gpt-4o",
            api_type="responses",
        )
        assert ctx.api_type == "responses"

    def test_api_type_in_subject_dict_absent(self):
        """api_type is not part of the subject audit dict."""
        ctx = PolicyContext(api_type="responses", team_id="t1")
        subject = ctx.get_subject_dict()
        assert "api_type" not in subject


# =============================================================================
# Policy evaluation with api_type
# =============================================================================


class TestPolicyEngineApiType:
    """Tests for policy evaluation considering api_type in context."""

    @pytest.mark.asyncio
    async def test_route_based_rule_matches_responses_path(self):
        """A route pattern /v1/responses* should match Responses API requests."""
        config = PolicyConfig.from_dict(
            {
                "default_action": "allow",
                "rules": [
                    {
                        "name": "deny-responses-api",
                        "action": "deny",
                        "routes": ["/v1/responses*", "/responses*"],
                        "reason": "Responses API disabled",
                        "priority": 10,
                    }
                ],
            }
        )
        engine = PolicyEngine(config=config, enabled=True)

        ctx = PolicyContext(
            route="/v1/responses",
            method="POST",
            model="gpt-4o",
            api_type="responses",
        )
        decision = await engine.evaluate(ctx)
        assert decision.allowed is False
        assert decision.reason == "Responses API disabled"

    @pytest.mark.asyncio
    async def test_model_rule_works_for_responses_api(self):
        """Model-based deny rules should work when model is extracted from Responses API."""
        config = PolicyConfig.from_dict(
            {
                "default_action": "allow",
                "rules": [
                    {
                        "name": "deny-gpt4",
                        "action": "deny",
                        "models": ["gpt-4*"],
                        "reason": "GPT-4 not allowed",
                        "priority": 10,
                    }
                ],
            }
        )
        engine = PolicyEngine(config=config, enabled=True)

        ctx = PolicyContext(
            route="/v1/responses",
            method="POST",
            model="gpt-4o",
            api_type="responses",
        )
        decision = await engine.evaluate(ctx)
        assert decision.allowed is False
        assert decision.reason == "GPT-4 not allowed"

    @pytest.mark.asyncio
    async def test_allow_chat_deny_responses(self):
        """Route-specific rules can allow chat but deny responses."""
        config = PolicyConfig.from_dict(
            {
                "default_action": "allow",
                "rules": [
                    {
                        "name": "deny-responses",
                        "action": "deny",
                        "routes": ["/v1/responses", "/responses"],
                        "reason": "Use chat/completions instead",
                        "priority": 10,
                    }
                ],
            }
        )
        engine = PolicyEngine(config=config, enabled=True)

        # Chat completions should be allowed
        chat_ctx = PolicyContext(
            route="/v1/chat/completions",
            method="POST",
            model="gpt-4",
            api_type="chat_completion",
        )
        assert (await engine.evaluate(chat_ctx)).allowed is True

        # Responses should be denied
        resp_ctx = PolicyContext(
            route="/v1/responses",
            method="POST",
            model="gpt-4",
            api_type="responses",
        )
        assert (await engine.evaluate(resp_ctx)).allowed is False

    @pytest.mark.asyncio
    async def test_context_without_api_type_still_works(self):
        """Backward compatibility: context without api_type should evaluate normally."""
        config = PolicyConfig.from_dict(
            {
                "default_action": "allow",
                "rules": [
                    {
                        "name": "deny-gpt4",
                        "action": "deny",
                        "models": ["gpt-4*"],
                        "reason": "Denied",
                        "priority": 10,
                    }
                ],
            }
        )
        engine = PolicyEngine(config=config, enabled=True)

        ctx = PolicyContext(
            route="/v1/chat/completions",
            method="POST",
            model="gpt-4",
            # api_type not set (None)
        )
        decision = await engine.evaluate(ctx)
        assert decision.allowed is False

    @pytest.mark.asyncio
    async def test_embeddings_path_model_extraction(self):
        """Model extraction should work for embeddings paths."""
        config = PolicyConfig.from_dict(
            {
                "default_action": "allow",
                "rules": [
                    {
                        "name": "deny-ada",
                        "action": "deny",
                        "models": ["text-embedding-ada*"],
                        "reason": "Use newer embedding model",
                        "priority": 10,
                    }
                ],
            }
        )
        engine = PolicyEngine(config=config, enabled=True)

        ctx = PolicyContext(
            route="/v1/embeddings",
            method="POST",
            model="text-embedding-ada-002",
            api_type="embedding",
        )
        decision = await engine.evaluate(ctx)
        assert decision.allowed is False


# =============================================================================
# Telemetry contract api_type tests
# =============================================================================


class TestTelemetryContractApiType:
    """Tests for api_type in telemetry contract data classes and builder."""

    def test_routing_outcome_data_default_api_type_none(self):
        outcome = RoutingOutcomeData()
        assert outcome.api_type is None

    def test_routing_outcome_data_with_api_type(self):
        outcome = RoutingOutcomeData(
            status=RoutingOutcome.SUCCESS,
            api_type="responses",
        )
        assert outcome.api_type == "responses"

    def test_router_decision_input_default_api_type_none(self):
        inp = RouterDecisionInput()
        assert inp.api_type is None

    def test_router_decision_input_with_api_type(self):
        inp = RouterDecisionInput(
            requested_model="gpt-4o",
            api_type="responses",
        )
        assert inp.api_type == "responses"

    def test_builder_with_input_api_type(self):
        event = (
            RouterDecisionEventBuilder()
            .with_input(
                query_length=50,
                requested_model="gpt-4o",
                api_type="responses",
            )
            .build()
        )
        assert event.input.api_type == "responses"
        assert event.input.requested_model == "gpt-4o"

    def test_builder_with_input_no_api_type(self):
        """Backward compat: builder without api_type defaults to None."""
        event = (
            RouterDecisionEventBuilder()
            .with_input(query_length=50, requested_model="gpt-4")
            .build()
        )
        assert event.input.api_type is None

    def test_builder_with_outcome_api_type(self):
        event = (
            RouterDecisionEventBuilder()
            .with_outcome(
                status=RoutingOutcome.SUCCESS,
                api_type="chat_completion",
            )
            .build()
        )
        assert event.outcome.api_type == "chat_completion"

    def test_builder_with_outcome_no_api_type(self):
        event = (
            RouterDecisionEventBuilder()
            .with_outcome(status=RoutingOutcome.SUCCESS)
            .build()
        )
        assert event.outcome.api_type is None

    def test_event_to_dict_includes_api_type(self):
        event = (
            RouterDecisionEventBuilder()
            .with_input(query_length=10, api_type="embedding")
            .with_outcome(status=RoutingOutcome.SUCCESS, api_type="embedding")
            .build()
        )
        d = event.to_dict()
        assert d["input"]["api_type"] == "embedding"
        assert d["outcome"]["api_type"] == "embedding"

    def test_event_to_dict_api_type_none(self):
        event = RouterDecisionEventBuilder().build()
        d = event.to_dict()
        assert d["input"]["api_type"] is None
        assert d["outcome"]["api_type"] is None

    def test_roundtrip_with_api_type(self):
        """Serialize to JSON, then extract back -- api_type should survive."""
        event = (
            RouterDecisionEventBuilder()
            .with_strategy("llmrouter-knn")
            .with_input(
                query_length=100,
                requested_model="gpt-4o",
                api_type="responses",
            )
            .with_outcome(
                status=RoutingOutcome.SUCCESS,
                api_type="responses",
            )
            .build()
        )
        payload_json = event.to_json()
        attrs = {"routeiq.router_decision.payload": payload_json}

        restored = extract_router_decision_from_span_event(attrs)
        assert restored is not None
        assert restored.input.api_type == "responses"
        assert restored.outcome.api_type == "responses"

    def test_roundtrip_without_api_type(self):
        """Older events without api_type should parse cleanly as None."""
        # Simulate an older payload without api_type in input/outcome
        payload = {
            "contract_version": "v1",
            "contract_name": "routeiq.router_decision.v1",
            "event_id": "test-123",
            "strategy_name": "llmrouter-knn",
            "input": {
                "requested_model": "gpt-4",
                "query_length": 50,
            },
            "outcome": {
                "status": "success",
            },
        }
        attrs = {"routeiq.router_decision.payload": json.dumps(payload)}

        restored = extract_router_decision_from_span_event(attrs)
        assert restored is not None
        assert restored.input.api_type is None
        assert restored.outcome.api_type is None


# =============================================================================
# PolicyMiddleware body extraction (unit-level via _build_context)
# =============================================================================


class TestPolicyMiddlewareBuildContext:
    """Tests for PolicyMiddleware._build_context with api_type and parsed body."""

    def _make_middleware(self):
        from litellm_llmrouter.policy_engine import PolicyMiddleware

        engine = PolicyEngine(enabled=True)

        async def noop_app(scope, receive, send):
            pass

        return PolicyMiddleware(noop_app, engine)

    def _make_scope(self, path: str = "/v1/responses", method: str = "POST"):
        return {
            "type": "http",
            "path": path,
            "method": method,
            "headers": [],
            "client": ("127.0.0.1", 8000),
        }

    def test_build_context_with_responses_body(self):
        mw = self._make_middleware()
        scope = self._make_scope("/v1/responses")
        body = {"model": "gpt-4o", "input": "Hello"}

        ctx = mw._build_context(scope, api_type="responses", parsed_body=body)

        assert ctx.model == "gpt-4o"
        assert ctx.api_type == "responses"
        assert ctx.route == "/v1/responses"

    def test_build_context_with_chat_completions_body(self):
        mw = self._make_middleware()
        scope = self._make_scope("/v1/chat/completions")
        body = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        ctx = mw._build_context(scope, api_type="chat_completion", parsed_body=body)

        assert ctx.model == "gpt-4-turbo"
        assert ctx.api_type == "chat_completion"

    def test_build_context_with_embeddings_body(self):
        mw = self._make_middleware()
        scope = self._make_scope("/v1/embeddings")
        body = {"model": "text-embedding-3-small", "input": "test"}

        ctx = mw._build_context(scope, api_type="embedding", parsed_body=body)

        assert ctx.model == "text-embedding-3-small"
        assert ctx.api_type == "embedding"

    def test_build_context_no_body_falls_back_to_header(self):
        mw = self._make_middleware()
        scope = self._make_scope("/v1/chat/completions")
        scope["headers"] = [
            (b"x-model", b"gpt-3.5-turbo"),
        ]

        ctx = mw._build_context(scope, api_type="chat_completion", parsed_body=None)

        assert ctx.model == "gpt-3.5-turbo"

    def test_build_context_body_model_overrides_header(self):
        mw = self._make_middleware()
        scope = self._make_scope("/v1/responses")
        scope["headers"] = [
            (b"x-model", b"gpt-3.5-turbo"),
        ]
        body = {"model": "gpt-4o"}

        ctx = mw._build_context(scope, api_type="responses", parsed_body=body)

        # Body extraction should take precedence
        assert ctx.model == "gpt-4o"

    def test_build_context_unparseable_body(self):
        """If parsed_body is None (e.g. bad JSON), fall back to header."""
        mw = self._make_middleware()
        scope = self._make_scope("/v1/responses")
        scope["headers"] = [
            (b"x-model", b"fallback-model"),
        ]

        ctx = mw._build_context(scope, api_type="responses", parsed_body=None)

        assert ctx.model == "fallback-model"

    def test_build_context_non_llm_path_no_api_type(self):
        mw = self._make_middleware()
        scope = self._make_scope("/_health/live", "GET")

        ctx = mw._build_context(scope, api_type=None, parsed_body=None)

        assert ctx.api_type is None
        assert ctx.model is None

    def test_build_context_body_without_model_key(self):
        mw = self._make_middleware()
        scope = self._make_scope("/v1/responses")
        body = {"input": "Hello, no model here"}

        ctx = mw._build_context(scope, api_type="responses", parsed_body=body)

        assert ctx.model is None
