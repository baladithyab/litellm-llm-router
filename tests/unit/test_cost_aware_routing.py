"""
Tests for the CostAwareRoutingStrategy.

These tests verify:
1. Strategy selects cheapest model above quality threshold
2. Fallback to best quality when no cheap options meet threshold
3. Cost lookup from litellm.model_cost (mocked)
4. Cost lookup fallback when litellm unavailable
5. Combined scoring with different cost_weight values
6. max_cost_per_1k_tokens filtering
7. Empty candidate list handling
8. Single candidate (always selected)
9. Strategy registration in LLMROUTER_STRATEGIES
10. Integration with inner_strategy delegation
"""

from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.strategies import (
    CostAwareRoutingStrategy,
    LLMROUTER_STRATEGIES,
    DEFAULT_ROUTER_HPARAMS,
)
from litellm_llmrouter.strategy_registry import (
    RoutingContext,
    RoutingStrategy,
    RoutingStrategyRegistry,
    RoutingPipeline,
    reset_routing_singletons,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deployment(model: str, model_name: str = "test-model") -> Dict:
    """Create a deployment dict matching LiteLLM's format."""
    return {
        "model_name": model_name,
        "litellm_params": {"model": model},
    }


def _make_router(deployments: List[Dict]) -> MagicMock:
    """Create a mock Router with the given deployments."""
    router = MagicMock()
    router.model_list = deployments
    router.healthy_deployments = deployments
    return router


def _make_context(
    deployments: List[Dict],
    model_name: str = "test-model",
) -> RoutingContext:
    """Create a RoutingContext for testing."""
    return RoutingContext(
        router=_make_router(deployments),
        model=model_name,
    )


MOCK_MODEL_COST = {
    "gpt-3.5-turbo": {
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
    },
    "gpt-4": {
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-turbo": {
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
    },
    "claude-3-opus": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
    },
}


class MockInnerStrategy(RoutingStrategy):
    """Inner strategy that always selects a specific model."""

    def __init__(self, preferred_model: str):
        self._preferred_model = preferred_model

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        router = context.router
        healthy = getattr(router, "healthy_deployments", router.model_list)
        for dep in healthy:
            if dep.get("litellm_params", {}).get("model") == self._preferred_model:
                return dep
        return None

    @property
    def name(self) -> str:
        return "mock-inner"


class FailingInnerStrategy(RoutingStrategy):
    """Inner strategy that always raises."""

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        raise RuntimeError("Inner strategy failed")

    @property
    def name(self) -> str:
        return "failing-inner"


class NoneInnerStrategy(RoutingStrategy):
    """Inner strategy that always returns None."""

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        return None

    @property
    def name(self) -> str:
        return "none-inner"


# ---------------------------------------------------------------------------
# Test: Strategy registration
# ---------------------------------------------------------------------------


class TestCostAwareRegistration:
    """Test that the strategy is properly registered in the catalog."""

    def test_strategy_in_llmrouter_strategies(self):
        """llmrouter-cost-aware should be in LLMROUTER_STRATEGIES."""
        assert "llmrouter-cost-aware" in LLMROUTER_STRATEGIES

    def test_default_hparams_exist(self):
        """Default hyperparameters should exist for cost-aware."""
        assert "cost-aware" in DEFAULT_ROUTER_HPARAMS
        hparams = DEFAULT_ROUTER_HPARAMS["cost-aware"]
        assert hparams["quality_threshold"] == 0.7
        assert hparams["cost_weight"] == 0.7
        assert hparams["inner_strategy"] is None
        assert hparams["max_cost_per_1k_tokens"] is None

    def test_strategy_name_property(self):
        """Strategy name should return llmrouter-cost-aware."""
        strategy = CostAwareRoutingStrategy()
        assert strategy.name == "llmrouter-cost-aware"

    def test_strategy_version_property(self):
        """Strategy version should return 1.0.0."""
        strategy = CostAwareRoutingStrategy()
        assert strategy.version == "1.0.0"

    def test_strategy_validates_successfully(self):
        """Default parameters should pass validation."""
        strategy = CostAwareRoutingStrategy()
        valid, error = strategy.validate()
        assert valid is True
        assert error is None


# ---------------------------------------------------------------------------
# Test: Registry integration
# ---------------------------------------------------------------------------


class TestCostAwareRegistryIntegration:
    """Test registration in the RoutingStrategyRegistry."""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_routing_singletons()
        yield
        reset_routing_singletons()

    def test_register_in_registry(self):
        """Strategy can be registered in RoutingStrategyRegistry."""
        registry = RoutingStrategyRegistry()
        strategy = CostAwareRoutingStrategy()

        registry.register("llmrouter-cost-aware", strategy)

        assert "llmrouter-cost-aware" in registry.list_strategies()
        assert registry.get("llmrouter-cost-aware") is strategy

    def test_register_and_select(self):
        """Registered strategy is selected when set as active."""
        registry = RoutingStrategyRegistry()
        strategy = CostAwareRoutingStrategy()

        registry.register("llmrouter-cost-aware", strategy)
        registry.set_active("llmrouter-cost-aware")

        result = registry.select_strategy("hash-key")
        assert result.strategy is strategy
        assert result.strategy_name == "llmrouter-cost-aware"

    def test_pipeline_execution(self):
        """Strategy works through the RoutingPipeline."""
        registry = RoutingStrategyRegistry()
        strategy = CostAwareRoutingStrategy(quality_threshold=0.0)

        registry.register("llmrouter-cost-aware", strategy)
        registry.set_active("llmrouter-cost-aware")

        pipeline = RoutingPipeline(registry, emit_telemetry=False)

        deployments = [_make_deployment("gpt-3.5-turbo")]
        context = _make_context(deployments)

        result = pipeline.route(context)
        assert result.deployment is not None
        assert result.strategy_name == "llmrouter-cost-aware"


# ---------------------------------------------------------------------------
# Test: Cost lookup
# ---------------------------------------------------------------------------


class TestCostLookup:
    """Test _get_model_cost lookups."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_known_model_cost(self):
        """Known model returns correct average cost per 1K tokens."""
        strategy = CostAwareRoutingStrategy()
        cost = strategy._get_model_cost("gpt-4")
        # input: 0.00003 * 1000 = 0.03, output: 0.00006 * 1000 = 0.06
        # average: (0.03 + 0.06) / 2 = 0.045
        assert abs(cost - 0.045) < 1e-9

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_cheap_model_cost(self):
        """Cheap model returns lower cost."""
        strategy = CostAwareRoutingStrategy()
        cost = strategy._get_model_cost("gpt-3.5-turbo")
        # input: 0.0000005 * 1000 = 0.0005, output: 0.0000015 * 1000 = 0.0015
        # average: (0.0005 + 0.0015) / 2 = 0.001
        assert abs(cost - 0.001) < 1e-9

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_unknown_model_returns_inf(self):
        """Unknown model returns infinity."""
        strategy = CostAwareRoutingStrategy()
        cost = strategy._get_model_cost("unknown-model-xyz")
        assert cost == float("inf")

    @patch("litellm.model_cost", {})
    def test_empty_cost_db_returns_inf(self):
        """Empty cost database returns infinity."""
        strategy = CostAwareRoutingStrategy()
        cost = strategy._get_model_cost("gpt-4")
        assert cost == float("inf")

    def test_litellm_import_failure_returns_inf(self):
        """When litellm import fails, returns infinity."""
        strategy = CostAwareRoutingStrategy()
        with patch.dict("sys.modules", {"litellm": None}):
            cost = strategy._get_model_cost("gpt-4")
            assert cost == float("inf")

    @patch(
        "litellm.model_cost",
        {
            "zero-cost-model": {
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            }
        },
    )
    def test_zero_cost_model_returns_inf(self):
        """Model with zero cost returns inf (treated as unknown pricing)."""
        strategy = CostAwareRoutingStrategy()
        cost = strategy._get_model_cost("zero-cost-model")
        assert cost == float("inf")


# ---------------------------------------------------------------------------
# Test: Cheapest model selection
# ---------------------------------------------------------------------------


class TestCheapestModelSelection:
    """Test selection of cheapest model above quality threshold."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_selects_cheapest_model(self):
        """With no inner strategy, selects cheapest from all candidates."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.5,
            cost_weight=1.0,  # Pure cost optimization
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
            _make_deployment("gpt-4-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_selects_cheapest_above_threshold_with_inner(self):
        """With inner strategy, filters by quality then selects cheapest."""
        # Inner strategy prefers gpt-4; gpt-3.5 gets lower quality
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.8,
            cost_weight=0.9,
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # gpt-4 gets quality=1.0 (preferred), gpt-3.5 gets 0.5 (below 0.8 threshold)
        # Only gpt-4 meets threshold
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_all_candidates_meet_threshold_picks_cheapest(self):
        """When all meet threshold, picks cheapest by combined score."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,  # All pass
            cost_weight=1.0,  # Pure cost
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
            _make_deployment("claude-3-opus"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Test: Fallback to best quality
# ---------------------------------------------------------------------------


class TestFallbackToBestQuality:
    """Test fallback when no cheap options meet threshold."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_fallback_when_none_meet_threshold(self):
        """Falls back to best quality when no candidate meets threshold."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=1.1,  # Impossible threshold
            cost_weight=0.5,
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Falls back to inner strategy's selection
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_fallback_without_inner_returns_first(self):
        """Without inner strategy, quality=1.0 always meets threshold.

        To test fallback, use NoneInnerStrategy (quality 0.5) with
        a threshold above 0.5 so no candidate qualifies.
        """
        inner = NoneInnerStrategy()
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.9,  # Above 0.5 from NoneInner
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # All get quality 0.5 (below 0.9) -> fallback to best quality
        # Inner returns None -> falls to first candidate
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_fallback_with_failing_inner(self):
        """Fallback handles inner strategy failure gracefully."""
        inner = FailingInnerStrategy()
        strategy = CostAwareRoutingStrategy(
            quality_threshold=1.1,  # Impossible
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Inner fails, falls back to first candidate
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# Test: Combined scoring with cost_weight
# ---------------------------------------------------------------------------


class TestCombinedScoring:
    """Test combined quality-cost scoring with different cost_weight values."""

    def test_combined_score_cost_only(self):
        """cost_weight=1.0 means only cost matters."""
        strategy = CostAwareRoutingStrategy(cost_weight=1.0)
        # quality irrelevant, normalized_cost=0.0 (cheapest) -> score = 1.0
        score = strategy._compute_combined_score(quality=0.0, normalized_cost=0.0)
        assert abs(score - 1.0) < 1e-9

    def test_combined_score_quality_only(self):
        """cost_weight=0.0 means only quality matters."""
        strategy = CostAwareRoutingStrategy(cost_weight=0.0)
        score = strategy._compute_combined_score(quality=0.8, normalized_cost=1.0)
        assert abs(score - 0.8) < 1e-9

    def test_combined_score_balanced(self):
        """cost_weight=0.5 balances quality and cost."""
        strategy = CostAwareRoutingStrategy(cost_weight=0.5)
        # quality=0.8, normalized_cost=0.4 -> (1-0.5)*0.8 + 0.5*(1-0.4) = 0.4+0.3 = 0.7
        score = strategy._compute_combined_score(quality=0.8, normalized_cost=0.4)
        assert abs(score - 0.7) < 1e-9

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_cost_weight_zero_selects_highest_quality(self):
        """cost_weight=0.0 with inner strategy selects highest quality."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.0,  # Only quality matters
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-3.5-turbo"),
            _make_deployment("gpt-4"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # gpt-4 has quality 1.0, gpt-3.5 has 0.5
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_cost_weight_one_selects_cheapest(self):
        """cost_weight=1.0 selects cheapest regardless of quality."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=1.0,  # Only cost matters
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Test: max_cost_per_1k_tokens filtering
# ---------------------------------------------------------------------------


class TestMaxCostFiltering:
    """Test max_cost_per_1k_tokens hard cap."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_filters_expensive_models(self):
        """Models above max_cost_per_1k_tokens are excluded."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.5,
            max_cost_per_1k_tokens=0.01,  # Only gpt-3.5-turbo is cheap enough
        )

        deployments = [
            _make_deployment("gpt-4"),  # avg cost: 0.045
            _make_deployment("gpt-3.5-turbo"),  # avg cost: 0.001
            _make_deployment("claude-3-opus"),  # avg cost: 0.045
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_all_exceed_cap_falls_back(self):
        """When all models exceed cost cap, falls back to best quality."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.5,
            max_cost_per_1k_tokens=0.0001,  # Nothing is this cheap
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Falls back to first candidate
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_no_max_cost_allows_all(self):
        """Without max_cost_per_1k_tokens, all candidates considered."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=1.0,
            max_cost_per_1k_tokens=None,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Cheapest wins
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases in cost-aware routing."""

    def test_empty_candidate_list(self):
        """Empty candidate list returns None."""
        strategy = CostAwareRoutingStrategy()
        deployments: list = []
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is None

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_single_candidate_always_selected(self):
        """Single candidate is always returned."""
        strategy = CostAwareRoutingStrategy(quality_threshold=0.99)

        deployments = [_make_deployment("gpt-4")]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_no_matching_model_name(self):
        """Deployments with non-matching model_name are filtered out."""
        strategy = CostAwareRoutingStrategy()

        deployments = [
            _make_deployment("gpt-4", model_name="other-model"),
        ]
        context = _make_context(deployments, model_name="test-model")

        result = strategy.select_deployment(context)

        assert result is None

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_all_same_cost_picks_highest_quality(self):
        """When all candidates have same cost, picks highest quality."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.5,
            inner_strategy=inner,
        )

        # Same model, same cost -> quality breaks the tie
        deployments = [
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4"},
                "id": "deploy-1",
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4"},
                "id": "deploy-2",
            },
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        # Both have same quality (1.0 since model matches) and same cost
        # First one wins since combined scores are equal and we iterate in order
        assert result["litellm_params"]["model"] == "gpt-4"

    def test_quality_threshold_clamped(self):
        """Quality threshold is clamped to [0, 1]."""
        strategy = CostAwareRoutingStrategy(quality_threshold=2.0)
        assert strategy._quality_threshold == 1.0

        strategy = CostAwareRoutingStrategy(quality_threshold=-0.5)
        assert strategy._quality_threshold == 0.0

    def test_cost_weight_clamped(self):
        """Cost weight is clamped to [0, 1]."""
        strategy = CostAwareRoutingStrategy(cost_weight=1.5)
        assert strategy._cost_weight == 1.0

        strategy = CostAwareRoutingStrategy(cost_weight=-0.2)
        assert strategy._cost_weight == 0.0


# ---------------------------------------------------------------------------
# Test: Inner strategy delegation
# ---------------------------------------------------------------------------


class TestInnerStrategyDelegation:
    """Test delegation to inner strategy for quality prediction."""

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_inner_strategy_quality_scoring(self):
        """Inner strategy's preferred model gets quality=1.0."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.0,  # Pure quality
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-3.5-turbo"),
            _make_deployment("gpt-4"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_no_inner_strategy_all_quality_equal(self):
        """Without inner strategy, all get quality 1.0 -> cheapest wins."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.5,
            inner_strategy=None,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_inner_strategy_returns_none(self):
        """Inner strategy returning None gives quality 0.5."""
        inner = NoneInnerStrategy()
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=0.0,  # Pure quality
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Both get 0.5 quality, first wins with pure quality weighting
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-4"

    @patch("litellm.model_cost", MOCK_MODEL_COST)
    def test_inner_strategy_exception_gives_half_quality(self):
        """Inner strategy raising exception gives quality 0.5."""
        inner = FailingInnerStrategy()
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.0,
            cost_weight=1.0,  # Pure cost
            inner_strategy=inner,
        )

        deployments = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        context = _make_context(deployments)

        result = strategy.select_deployment(context)

        # Both get 0.5 quality (exception), cost_weight=1.0 -> cheapest wins
        assert result is not None
        assert result["litellm_params"]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Test: Quality prediction
# ---------------------------------------------------------------------------


class TestQualityPrediction:
    """Test _predict_quality method directly."""

    def test_no_inner_returns_one(self):
        """Without inner strategy, quality is always 1.0."""
        strategy = CostAwareRoutingStrategy(inner_strategy=None)
        dep = _make_deployment("gpt-4")
        context = _make_context([dep])

        quality = strategy._predict_quality(context, dep)

        assert quality == 1.0

    def test_inner_selects_this_deployment(self):
        """When inner selects this deployment, quality is 1.0."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(inner_strategy=inner)

        dep = _make_deployment("gpt-4")
        context = _make_context([dep])

        quality = strategy._predict_quality(context, dep)

        assert quality == 1.0

    def test_inner_selects_different_deployment(self):
        """When inner selects a different deployment, quality is 0.5."""
        inner = MockInnerStrategy("gpt-4")
        strategy = CostAwareRoutingStrategy(inner_strategy=inner)

        dep = _make_deployment("gpt-3.5-turbo")
        context = _make_context([dep, _make_deployment("gpt-4")])

        quality = strategy._predict_quality(context, dep)

        assert quality == 0.5

    def test_inner_raises_returns_half(self):
        """When inner raises, quality is 0.5."""
        inner = FailingInnerStrategy()
        strategy = CostAwareRoutingStrategy(inner_strategy=inner)

        dep = _make_deployment("gpt-4")
        context = _make_context([dep])

        quality = strategy._predict_quality(context, dep)

        assert quality == 0.5

    def test_inner_returns_none_gives_half(self):
        """When inner returns None, quality is 0.5."""
        inner = NoneInnerStrategy()
        strategy = CostAwareRoutingStrategy(inner_strategy=inner)

        dep = _make_deployment("gpt-4")
        context = _make_context([dep])

        quality = strategy._predict_quality(context, dep)

        assert quality == 0.5


# ---------------------------------------------------------------------------
# Test: Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Test strategy validation."""

    def test_valid_default_params(self):
        """Default parameters pass validation."""
        strategy = CostAwareRoutingStrategy()
        valid, error = strategy.validate()
        assert valid is True
        assert error is None

    def test_valid_custom_params(self):
        """Custom valid parameters pass validation."""
        strategy = CostAwareRoutingStrategy(
            quality_threshold=0.5,
            cost_weight=0.3,
        )
        valid, error = strategy.validate()
        assert valid is True
        assert error is None

    def test_boundary_values_pass(self):
        """Boundary values (0.0 and 1.0) pass validation."""
        for qt in [0.0, 1.0]:
            for cw in [0.0, 1.0]:
                strategy = CostAwareRoutingStrategy(
                    quality_threshold=qt,
                    cost_weight=cw,
                )
                valid, error = strategy.validate()
                assert valid is True, f"Failed for qt={qt}, cw={cw}: {error}"


# ---------------------------------------------------------------------------
# Test: Candidate extraction
# ---------------------------------------------------------------------------


class TestCandidateExtraction:
    """Test _get_candidates method."""

    def test_extracts_matching_candidates(self):
        """Only deployments matching model_name are returned."""
        strategy = CostAwareRoutingStrategy()

        deployments = [
            _make_deployment("gpt-4", model_name="model-a"),
            _make_deployment("gpt-3.5-turbo", model_name="model-b"),
            _make_deployment("claude-3-opus", model_name="model-a"),
        ]
        context = _make_context(deployments, model_name="model-a")

        candidates = strategy._get_candidates(context)

        assert len(candidates) == 2
        models = [c["litellm_params"]["model"] for c in candidates]
        assert "gpt-4" in models
        assert "claude-3-opus" in models

    def test_empty_when_no_match(self):
        """Returns empty list when no deployments match."""
        strategy = CostAwareRoutingStrategy()

        deployments = [
            _make_deployment("gpt-4", model_name="other"),
        ]
        context = _make_context(deployments, model_name="test-model")

        candidates = strategy._get_candidates(context)

        assert candidates == []

    def test_uses_healthy_deployments(self):
        """Uses healthy_deployments attribute when available."""
        strategy = CostAwareRoutingStrategy()

        all_deps = [
            _make_deployment("gpt-4"),
            _make_deployment("gpt-3.5-turbo"),
        ]
        healthy_deps = [_make_deployment("gpt-3.5-turbo")]

        router = MagicMock()
        router.model_list = all_deps
        router.healthy_deployments = healthy_deps

        context = RoutingContext(router=router, model="test-model")
        candidates = strategy._get_candidates(context)

        assert len(candidates) == 1
        assert candidates[0]["litellm_params"]["model"] == "gpt-3.5-turbo"
