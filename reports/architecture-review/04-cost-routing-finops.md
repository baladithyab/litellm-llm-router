# RouteIQ Architecture Review: Cost-Aware Routing and FinOps

**Report**: 04-cost-routing-finops
**Date**: 2025-02-07
**Scope**: Cost tracking, cost-aware routing, budget enforcement, and FinOps patterns
**Status**: Assessment with gap analysis and architecture recommendations

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Industry Landscape: AI Gateway Cost Management](#2-industry-landscape-ai-gateway-cost-management)
3. [RouteIQ Current Capabilities Assessment](#3-routeiq-current-capabilities-assessment)
4. [Gap Analysis](#4-gap-analysis)
5. [Architecture Recommendations](#5-architecture-recommendations)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Appendix: File Reference](#7-appendix-file-reference)

---

## 1. Executive Summary

RouteIQ has foundational cost infrastructure but significant gaps compared to industry
leaders. The existing quota system (`quota.py`) provides pre-request spend reservation
using estimated tokens and LiteLLM's `model_cost` dictionary. The plugin callback bridge
(`plugin_callback_bridge.py`) delivers the hook points needed for post-call cost
reconciliation. However, RouteIQ currently lacks:

- **Post-call cost reconciliation** (actual vs. estimated cost adjustment)
- **Cost-aware routing strategy** (cheapest-model-that-meets-quality-threshold)
- **Persistent cost attribution** (per-user, per-team, per-model spend history)
- **Cost reporting endpoints** (dashboards, spend analytics, budget status)
- **Budget alerting** (soft limit warnings, webhook notifications)
- **Cost-optimized model selection** (quality-cost Pareto frontier routing)

The good news: RouteIQ's architecture is well-positioned for these additions. The plugin
system with LLM lifecycle hooks, the quota enforcement framework, and the telemetry
contracts provide clean extension points. Cost tracking can be implemented as a plugin
without core changes.

---

## 2. Industry Landscape: AI Gateway Cost Management

### 2.1 Per-Request Cost Calculation

Every production gateway calculates per-request cost using the formula:

```
cost = (input_tokens * input_price_per_token) + (output_tokens * output_price_per_token)
```

Some providers also charge for:
- Cached input tokens (reduced rate, e.g., Anthropic prompt caching)
- Image/audio tokens (different pricing tiers)
- Tool use tokens (function calling overhead)
- Batch API discounts (50% off for async batch processing)

#### Pricing Database

LiteLLM maintains `model_prices_and_context_window.json` -- a community-maintained JSON
file mapping model identifiers to pricing:

```json
{
  "gpt-4": {
    "input_cost_per_token": 0.00003,
    "output_cost_per_token": 0.00006,
    "max_tokens": 8192,
    "litellm_provider": "openai"
  }
}
```

This is exposed at runtime as `litellm.model_cost` (a dict). LiteLLM's
`completion_cost()` function uses it to calculate per-response cost from the `usage`
field in API responses. The pricing database is updated with each LiteLLM release.

#### Token Counting Approaches

| Approach | When | Accuracy | Use Case |
|----------|------|----------|----------|
| Pre-call estimation | Before API call | Low (chars/4 heuristic) | Budget reservation, quota guard |
| Provider-reported | In API response `usage` field | High (authoritative) | Cost calculation, billing |
| tiktoken counting | Before/after call | Medium-High (model-specific) | Pre-call cost estimation |
| Streaming accumulation | During SSE stream | High | Real-time cost tracking |

### 2.2 Competitor Analysis

#### Portkey AI Gateway
- **Cost tracking**: Automatic per-request cost calculation from response `usage` data
- **Budget management**: Per-org, per-workspace, per-virtual-key budgets with soft/hard limits
- **Cost attribution**: Tags and metadata for chargeback (team, project, environment)
- **Analytics**: Real-time cost dashboards with model/provider/tag breakdowns
- **Alerts**: Webhook notifications at configurable spend thresholds (50%, 80%, 100%)
- **Virtual keys**: Each key can have a max_budget and rate limits
- **Caching-aware**: Tracks cache hit savings separately

#### Helicone
- **Cost tracking**: Per-request cost from usage data, stored in Clickhouse
- **Budget management**: Per-key, per-user cost limits
- **Cost attribution**: Custom properties for team/project/feature cost allocation
- **Analytics**: SQL-queryable cost analytics, model cost comparison charts
- **Rate limiting**: Cost-based rate limiting (not just request-based)
- **Differentiator**: Deep cost analytics with SQL access and cost anomaly detection

#### Cloudflare AI Gateway
- **Cost tracking**: Estimated per-request cost using provider pricing
- **Budget management**: Basic per-gateway spending caps
- **Cost attribution**: Per-gateway analytics (one gateway per project/team)
- **Analytics**: Dashboard with cost-per-model, cost-per-request breakdown
- **Caching**: Built-in response caching with cost savings tracking
- **Differentiator**: Edge-native with caching cost optimization

#### LiteLLM Proxy (Upstream)
- **Cost tracking**: `completion_cost()` function, `_response_cost` in logging payloads
- **Budget management**: Per-key `max_budget`, per-team budget, per-user budget (DB-backed)
- **Spend tracking**: `LiteLLM_SpendLogs` table tracking per-request spend
- **Provider budgets**: `RouterBudgetLimiting` -- filters out deployments exceeding provider spend limits
- **Spend endpoints**: `/spend/keys`, `/spend/users`, `/spend/tags`, `/global/spend/logs`
- **Reset**: Monthly budget reset, daily reset options
- **Differentiator**: DB-backed spend tracking with Prisma ORM, budget enforcement at router level

### 2.3 Cost-Aware Routing Patterns

The most advanced pattern is **quality-cost-aware routing**: selecting the cheapest model
that meets a quality threshold for each request.

```
For each request:
  1. Estimate request difficulty/complexity
  2. For each candidate model:
     a. Predict quality score (from ML router)
     b. Look up cost (from pricing database)
     c. Compute quality/cost ratio
  3. Select cheapest model where predicted_quality >= threshold
```

| Pattern | Description | Example |
|---------|-------------|---------|
| **Cheapest-adequate** | Select cheapest model above quality threshold | Route simple questions to GPT-3.5, complex to GPT-4 |
| **Cost-weighted routing** | Bias model selection by cost (prefer cheaper) | Weight inverse to price in routing strategy |
| **Cascade/escalation** | Try cheap model first, escalate if quality insufficient | AutoMix, HybridLLMRouter |
| **Budget-constrained** | Route to maximize quality within budget constraint | Optimization problem per-window |
| **Pareto frontier** | Select from quality-cost efficient frontier | Filter dominated models, pick from frontier |

RouteIQ's `llmrouter-automix` and `llmrouter-hybrid` strategies already implement
cascade/escalation patterns. What is missing is explicit cost integration into the
routing decision.

---

## 3. RouteIQ Current Capabilities Assessment

### 3.1 Spend Quota Enforcement (Existing)

**File**: `src/litellm_llmrouter/quota.py`

The quota system provides pre-request spend enforcement:

| Capability | Status | Details |
|-----------|--------|---------|
| Request quota | Implemented | Per-subject request counting (minute/hour/day/month) |
| Token quota | Implemented | Input, output, and total token reservation |
| Spend quota | Implemented | USD spend reservation with `SPEND_USD` metric |
| Multi-dimensional | Implemented | Requests + tokens + spend simultaneously |
| Redis-backed | Implemented | Atomic Lua scripts for check-and-increment |
| Fail-open/closed | Implemented | Configurable behavior on Redis failure |
| Subject derivation | Implemented | team > user > api_key > IP precedence |
| OTel integration | Implemented | Quota span attributes for observability |

**Key limitation**: The spend calculation uses **pre-call estimation** only:

```python
# From quota.py _calculate_spend_reservation()
if model and hasattr(litellm, "model_cost"):
    cost_info = litellm.model_cost.get(model)
    if cost_info:
        input_cost = (input_tokens / 1000) * cost_info.get("input_cost_per_token", 0)
        output_cost = (output_tokens / 1000) * cost_info.get("output_cost_per_token", 0)
        return input_cost + output_cost
```

This estimates input tokens at ~1 token per 4 characters and assumes `max_tokens` will
be fully consumed for output. In practice, actual output tokens are typically 10-50% of
`max_tokens`, so spend reservations are significantly over-estimated. There is no
post-call reconciliation to correct the overcount.

### 3.2 Plugin Callback Bridge (Existing)

**File**: `src/litellm_llmrouter/gateway/plugin_callback_bridge.py`

The callback bridge provides the exact hook points needed for cost tracking:

```
litellm.log_pre_api_call  -->  plugin.on_llm_pre_call(model, messages, kwargs)
litellm.log_success_event -->  plugin.on_llm_success(model, response, kwargs)
litellm.log_failure_event -->  plugin.on_llm_failure(model, exception, kwargs)
```

The `on_llm_success` hook receives the full LLM response object, which contains `usage`
data (prompt_tokens, completion_tokens, total_tokens). This is the ideal point for
accurate post-call cost calculation.

### 3.3 Routing Strategy System (Existing)

**Files**: `src/litellm_llmrouter/strategies.py`, `strategy_registry.py`

The routing system supports 18+ ML-based strategies. The `RoutingStrategy` abstract base
class and `RoutingPipeline` provide clean extension points. The `RoutingContext` includes
a `metadata` dict which could carry cost constraints. The pipeline emits telemetry via
`RouterDecisionEvent` which already tracks `input_tokens`, `output_tokens`, and
`total_tokens` in `RoutingOutcomeData`.

### 3.4 Telemetry Contracts (Existing)

**File**: `src/litellm_llmrouter/telemetry_contracts.py`

The `RoutingOutcomeData` dataclass already has token fields:

```python
@dataclass
class RoutingOutcomeData:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
```

These are currently not populated with cost data, but the structure is extensible via
`custom_attributes: Dict[str, Any]`.

### 3.5 LiteLLM Upstream Capabilities (Inherited)

| Capability | Details |
|-----------|---------|
| `litellm.model_cost` | In-memory pricing dict (4000+ model entries) |
| `litellm.completion_cost()` | Post-call cost calculation from response usage |
| `litellm.cost_per_token()` | Per-token cost lookup |
| `RouterBudgetLimiting` | Provider-level budget filtering |
| `LiteLLM_SpendLogs` | DB-backed per-request spend logging (requires Prisma/Postgres) |
| `/spend/*` endpoints | Spend analytics endpoints (keys, users, tags) |
| `max_budget` on keys | Per-virtual-key spending caps |
| `max_budget` on teams | Per-team spending caps |

**Important caveat**: Many of LiteLLM's budget features require Prisma (PostgreSQL) which
may not be deployed in all RouteIQ configurations. The Redis-backed quota system in
`quota.py` works without a database.

---

## 4. Gap Analysis

### 4.1 Critical Gaps

| Gap | Impact | Industry Comparison |
|-----|--------|---------------------|
| **No post-call cost reconciliation** | Spend quotas over-count by 2-10x due to max_tokens reservation | Portkey, Helicone, LiteLLM all use actual usage data |
| **No cost-aware routing** | Cannot optimize for cost (cheapest-adequate selection) | Portkey and custom gateways offer this |
| **No persistent cost history** | Cannot query historical spend by user/team/model | All competitors store spend history |
| **No budget alerting** | Teams exceed budgets silently (only hard quota rejection) | Portkey sends alerts at 50%/80%/100% |
| **No cost reporting API** | No endpoints for spend analytics or budget status | LiteLLM has /spend/*, Portkey has dashboard API |

### 4.2 Moderate Gaps

| Gap | Impact |
|-----|--------|
| **No cost attribution tags** | Cannot allocate costs to projects/features/environments |
| **No cost anomaly detection** | Unexpected cost spikes go undetected |
| **No cache savings tracking** | Cannot demonstrate ROI of response caching |
| **No streaming cost tracking** | Cost only calculated on complete responses |
| **No batch pricing support** | Batch API requests may be over-priced in estimation |
| **No custom pricing overrides** | Cannot set negotiated pricing for enterprise contracts |

### 4.3 Minor Gaps

| Gap | Impact |
|-----|--------|
| **No pricing freshness check** | Pricing data only updates with LiteLLM releases |
| **No cost forecasting** | Cannot project future spend from current trends |
| **No cost optimization suggestions** | No recommendations for cheaper model alternatives |
| **No multi-currency support** | All costs in USD only |

### 4.4 Strengths (vs. Competitors)

| Strength | Details |
|----------|---------|
| **ML-based routing** | 18+ strategies that could incorporate cost signals |
| **Pre-call quota guard** | Streaming-safe spend reservation (no response buffering) |
| **Plugin architecture** | Cost tracking can be added as a plugin, not core change |
| **OTel integration** | Cost data can flow into existing observability pipeline |
| **A/B testing framework** | Can A/B test cost-aware vs. quality-first routing |
| **Redis-backed quotas** | Scales horizontally without requiring PostgreSQL |

---

## 5. Architecture Recommendations

### 5.1 Cost Tracking Plugin (Priority: P0)

Implement cost tracking as a `GatewayPlugin` that uses the callback bridge hooks. This is
the highest-impact, lowest-risk addition.

**Plugin responsibilities**:

1. **on_llm_pre_call**: Record estimated cost (from quota estimation or tiktoken)
2. **on_llm_success**: Calculate actual cost from response `usage` data
3. **on_llm_success**: Emit cost telemetry (OTel span attributes + metrics)
4. **on_llm_success**: Reconcile spend quota (adjust Redis counter)
5. **on_llm_failure**: Record zero cost (no tokens consumed on failure)

**Design sketch**:

```python
class CostTrackerPlugin(GatewayPlugin):
    """
    Plugin that tracks per-request LLM costs and reconciles spend quotas.

    Hooks into the LLM lifecycle via PluginCallbackBridge:
    - on_llm_pre_call: Record pre-call state (model, estimated tokens)
    - on_llm_success: Calculate actual cost, emit metrics, reconcile quota
    - on_llm_failure: Handle failed requests (zero cost)
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="cost-tracker",
            version="1.0.0",
            capabilities={PluginCapability.EVALUATOR,
                          PluginCapability.OBSERVABILITY_EXPORTER},
            priority=50,  # Early in plugin order
        )

    async def on_llm_success(self, model, response, kwargs):
        # Extract usage from response
        usage = getattr(response, "usage", None)
        if not usage:
            return

        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Calculate actual cost using litellm pricing
        actual_cost = litellm.completion_cost(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )

        # Emit OTel metrics
        self._cost_counter.add(
            actual_cost,
            attributes={
                "model": model,
                "team": kwargs.get("metadata", {}).get("team_id"),
                "user": kwargs.get("metadata", {}).get("user_id"),
            }
        )

        # Reconcile spend quota (adjust overestimation)
        estimated_cost = kwargs.get("metadata", {}).get("_estimated_cost", 0)
        if estimated_cost > actual_cost:
            await self._reconcile_spend(kwargs, estimated_cost - actual_cost)
```

**OTel span attributes to emit**:

```
llm.cost.input_tokens         = 1523
llm.cost.output_tokens        = 487
llm.cost.total_tokens         = 2010
llm.cost.input_cost_usd       = 0.04569
llm.cost.output_cost_usd      = 0.02922
llm.cost.total_cost_usd       = 0.07491
llm.cost.model                = gpt-4
llm.cost.estimated_cost_usd   = 0.25200   # pre-call estimate
llm.cost.estimation_error_pct = -70.3     # how wrong the estimate was
```

**OTel metrics to emit**:

```
llm_cost_total{model, team, user}           Counter (USD)
llm_tokens_total{model, team, direction}    Counter
llm_cost_per_request{model}                 Histogram (USD)
llm_cost_estimation_error{model}            Histogram (percentage)
```

### 5.2 Cost-Aware Routing Strategy (Priority: P1)

Add a new routing strategy that integrates cost into the model selection decision.

**Strategy design: Cheapest-Adequate Model Selection**

```python
class CostAwareRoutingStrategy(RoutingStrategy):
    """
    Selects the cheapest model that meets a quality threshold.

    Algorithm:
    1. For each candidate deployment, look up model cost
    2. Optionally predict quality score (delegate to inner strategy)
    3. Filter to candidates meeting quality threshold
    4. Select cheapest from filtered set

    Configuration:
        quality_threshold: Minimum acceptable quality score (0.0-1.0)
        quality_strategy: Inner strategy for quality prediction (optional)
        cost_weight: How much to weight cost vs quality (0.0=quality, 1.0=cost)
        max_cost_per_request: Hard cap on per-request cost
    """

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        candidates = self._get_candidates(context)

        scored = []
        for deployment in candidates:
            model = deployment["litellm_params"]["model"]
            cost_per_1k = self._get_model_cost_per_1k(model)
            quality = self._predict_quality(context, deployment)

            if quality >= self.quality_threshold:
                scored.append((deployment, cost_per_1k, quality))

        if not scored:
            return self._select_best_quality(candidates, context)

        scored.sort(key=lambda x: (x[1], -x[2]))
        return scored[0][0]
```

**Integration with existing strategies**: The cost-aware strategy wraps any existing
quality prediction strategy. It can use KNN, MLP, SVM, or any `llmrouter-*` strategy as
the quality predictor and add cost optimization on top.

**Registration**: Add `llmrouter-cost-aware` to `LLMROUTER_STRATEGIES` in `strategies.py`.

### 5.3 Spend Reconciliation System (Priority: P0)

The current quota system over-estimates spend because it reserves `max_tokens` worth of
output. A reconciliation system corrects this after each call.

**Design**:

```
Pre-call (quota.py):
  estimated_spend = estimate_cost(input_tokens_est, max_tokens, model)
  Redis INCRBY quota_key estimated_spend
  Store estimated_spend in request metadata

Post-call (cost-tracker plugin):
  actual_spend = completion_cost(response.usage)
  delta = estimated_spend - actual_spend
  Redis DECRBY quota_key delta   # Credit back overestimation
```

**Lua script for atomic reconciliation**:

```lua
-- KEYS[1] = quota key
-- ARGV[1] = delta to credit back (positive = reduce counter)
local key = KEYS[1]
local delta = tonumber(ARGV[1])
local current = tonumber(redis.call('GET', key) or '0')
local new_value = math.max(0, current - delta)
redis.call('SET', key, tostring(new_value))
return tostring(new_value)
```

### 5.4 Budget Enforcement Plugin (Priority: P1)

Extend the cost tracker plugin with budget enforcement features:

| Feature | Description |
|---------|-------------|
| Soft limits | Warning at configurable thresholds (default: 50%, 80%, 90%) |
| Hard limits | Request rejection at 100% (already in quota.py) |
| Webhook alerts | POST to configurable URL when threshold breached |
| OTel events | Span events for budget threshold breaches |
| Budget status API | GET endpoint returning current spend vs. budget |
| Budget reset | Automatic reset at window boundary (daily/monthly) |

**Alert design**:

```python
@dataclass
class BudgetAlert:
    subject: QuotaSubject
    metric: QuotaMetric
    window: QuotaWindow
    current_spend: float
    budget_limit: float
    threshold_pct: float    # e.g., 0.80
    timestamp: float

class BudgetAlertEmitter:
    async def check_thresholds(self, subject, metric, window, current, limit):
        pct = current / limit if limit > 0 else 0
        for threshold in self.thresholds:  # [0.50, 0.80, 0.90]
            if pct >= threshold and not self._already_alerted(subject, threshold):
                await self._emit_alert(BudgetAlert(...))
                self._mark_alerted(subject, threshold)
```

### 5.5 Cost Reporting Endpoints (Priority: P2)

Add admin endpoints for cost analytics. These complement LiteLLM's `/spend/*` endpoints
with RouteIQ-specific data from Redis.

```
GET /admin/cost/summary
    ?window=day|month
    &subject_type=team|user|api_key
    Response: { subjects: [{ key, type, total_spend, request_count, avg_cost }] }

GET /admin/cost/by-model
    ?window=day|month
    &model=gpt-4
    Response: { models: [{ model, total_spend, total_tokens, request_count }] }

GET /admin/cost/budget-status
    ?subject=team:engineering
    Response: { subject, limits: [{ metric, window, current, limit, pct, reset_at }] }

GET /admin/cost/efficiency
    ?window=day
    Response: {
        avg_estimation_error_pct,
        cache_savings_usd,
        cost_per_quality_unit,
        cheapest_adequate_model_savings_pct
    }
```

### 5.6 Cost Attribution via Tags (Priority: P2)

Enable cost attribution by project, feature, or environment using request metadata tags:

```json
{
    "model": "gpt-4",
    "messages": ["..."],
    "metadata": {
        "cost_tags": {
            "project": "customer-support-bot",
            "feature": "summarization",
            "environment": "production",
            "cost_center": "CC-1234"
        }
    }
}
```

The cost tracker plugin would extract these tags and include them in OTel attributes and
Redis keys, enabling per-project cost dashboards, cross-charge to business units, and
feature-level cost optimization.

### 5.7 Integration with Existing Quota System

The cost tracking plugin integrates seamlessly with the existing quota system:

```
Request Flow:
  1. quota_guard (FastAPI Depends) - pre-request quota check + spend reservation
  2. LiteLLM routes request to model
  3. PluginCallbackBridge.async_log_pre_api_call - plugin pre-call hook
  4. LLM API call executes
  5. PluginCallbackBridge.async_log_success_event - plugin success hook
     a. CostTrackerPlugin.on_llm_success
        - Calculate actual cost
        - Reconcile spend quota in Redis
        - Emit OTel metrics
        - Check budget thresholds
        - Persist cost record (if configured)
  6. Response returned to client
```

The key design principle: **quota_guard handles enforcement pre-request, cost tracker
handles reconciliation post-request**. They share the same Redis keys and QuotaSubject
derivation logic.

---

## 6. Implementation Roadmap

### Phase 1: Cost Tracking Foundation (P0)

**Goal**: Accurate per-request cost tracking with post-call reconciliation.

| Task | Effort | Dependencies |
|------|--------|-------------|
| CostTrackerPlugin skeleton (GatewayPlugin subclass) | 1 day | plugin_manager.py, plugin_callback_bridge.py |
| on_llm_success cost calculation (litellm.completion_cost) | 1 day | LiteLLM model_cost |
| OTel span attributes for cost data | 0.5 day | observability.py |
| OTel metrics (counters, histograms) | 1 day | observability.py |
| Spend reconciliation Lua script in QuotaRepository | 1 day | quota.py |
| Wire reconciliation into on_llm_success | 0.5 day | quota.py, cost tracker |
| Unit tests for cost calculation and reconciliation | 1 day | -- |
| Integration test with local test stack | 0.5 day | Docker compose |

**Total**: ~6.5 days

### Phase 2: Budget Intelligence (P1)

**Goal**: Budget alerting and cost-aware routing.

| Task | Effort | Dependencies |
|------|--------|-------------|
| Budget threshold detection (soft limits) | 1 day | Phase 1 |
| Webhook alert emitter | 1 day | http_client_pool.py |
| OTel events for budget breaches | 0.5 day | Phase 1 |
| CostAwareRoutingStrategy implementation | 2 days | strategies.py, strategy_registry.py |
| Quality-cost scoring integration | 1 day | Existing ML strategies |
| Register llmrouter-cost-aware strategy | 0.5 day | strategies.py |
| Unit tests for strategy and alerts | 1 day | -- |

**Total**: ~7 days

### Phase 3: Cost Reporting and Attribution (P2)

**Goal**: Admin visibility into cost data.

| Task | Effort | Dependencies |
|------|--------|-------------|
| Cost summary endpoint (GET /admin/cost/summary) | 1 day | Phase 1 |
| Cost by-model endpoint | 0.5 day | Phase 1 |
| Budget status endpoint | 0.5 day | Phase 1 |
| Cost efficiency endpoint | 1 day | Phase 1 |
| Cost attribution tags in request metadata | 1 day | Phase 1 |
| Tag-based Redis key bucketing | 1 day | quota.py |
| Unit tests for endpoints | 1 day | -- |

**Total**: ~6 days

---

## 7. Appendix: File Reference

### 7.1 Source Files Analyzed

| File | Relevance |
|------|-----------|
| `src/litellm_llmrouter/quota.py` | Core quota enforcement with spend reservation |
| `src/litellm_llmrouter/gateway/plugin_callback_bridge.py` | LiteLLM callback to plugin hook bridge |
| `src/litellm_llmrouter/gateway/plugin_manager.py` | GatewayPlugin base class with LLM lifecycle hooks |
| `src/litellm_llmrouter/gateway/plugin_middleware.py` | ASGI middleware for request/response plugin hooks |
| `src/litellm_llmrouter/gateway/app.py` | App factory wiring plugins and callback bridge |
| `src/litellm_llmrouter/strategies.py` | ML routing strategies (18+ strategies) |
| `src/litellm_llmrouter/strategy_registry.py` | Strategy registry, A/B testing, routing pipeline |
| `src/litellm_llmrouter/router_decision_callback.py` | Routing decision telemetry middleware |
| `src/litellm_llmrouter/telemetry_contracts.py` | Versioned telemetry event schemas |
| `src/litellm_llmrouter/observability.py` | OTel configuration and span attributes |
| `src/litellm_llmrouter/gateway/plugins/evaluator.py` | Reference plugin implementation pattern |

### 7.2 LiteLLM Reference Files

| File | Relevance |
|------|-----------|
| `reference/litellm/litellm/router_strategy/budget_limiter.py` | Provider budget limiting pattern |
| `reference/litellm/litellm/proxy/spend_tracking/spend_management_endpoints.py` | Spend analytics endpoints |
| `reference/litellm/litellm/budget_manager.py` | Per-user budget management |
| `reference/litellm/litellm/search/cost_calculator.py` | Cost calculation utilities |

### 7.3 Key Code Patterns

**Quota spend estimation** (current, in `quota.py:580-610`):

```python
def _calculate_spend_reservation(self, input_tokens, output_tokens, model=None):
    try:
        import litellm
        if model and hasattr(litellm, "model_cost"):
            cost_info = litellm.model_cost.get(model)
            if cost_info:
                input_cost = (input_tokens / 1000) * cost_info.get(
                    "input_cost_per_token", 0)
                output_cost = (output_tokens / 1000) * cost_info.get(
                    "output_cost_per_token", 0)
                return input_cost + output_cost
    except (ImportError, Exception):
        pass
    total_tokens = input_tokens + output_tokens
    return (total_tokens / 1000) * self._config.default_spend_per_1k_tokens
```

**Plugin callback bridge** (how hooks are wired, `plugin_callback_bridge.py:95-112`):

```python
async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
    if not self._plugins:
        return
    model = kwargs.get("model", "unknown")
    for plugin in self._plugins:
        try:
            await plugin.on_llm_success(model, response_obj, kwargs)
        except Exception as e:
            logger.error(
                f"Plugin '{plugin.name}' on_llm_success failed: {e}")
```

**LiteLLM provider budget limiting** (reference, `budget_limiter.py:44-80`):

```python
class RouterBudgetLimiting(CustomLogger):
    """
    Filter out deployments that have exceeded their provider budget limit.
    Uses DualCache (in-memory + Redis) for spend tracking.
    Periodic sync to Redis for multi-instance consistency.
    """
    async def async_filter_deployments(self, model, healthy_deployments, ...):
        # Filters deployments based on cumulative spend vs budget
```

### 7.4 Proposed Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROUTEIQ_COST_TRACKING_ENABLED` | `false` | Enable cost tracking plugin |
| `ROUTEIQ_COST_RECONCILIATION_ENABLED` | `true` | Enable post-call spend reconciliation |
| `ROUTEIQ_COST_ALERT_THRESHOLDS` | `0.50,0.80,0.90` | Budget alert threshold percentages |
| `ROUTEIQ_COST_ALERT_WEBHOOK_URL` | (none) | Webhook URL for budget alerts |
| `ROUTEIQ_COST_ATTRIBUTION_TAGS` | (none) | Allowed cost attribution tag names |
| `ROUTEIQ_COST_AWARE_ROUTING_ENABLED` | `false` | Enable cost-aware routing strategy |
| `ROUTEIQ_COST_AWARE_QUALITY_THRESHOLD` | `0.7` | Minimum quality score for cost-aware routing |
| `ROUTEIQ_COST_AWARE_COST_WEIGHT` | `0.5` | Cost weight in routing decision (0-1) |
| `ROUTEIQ_COST_CUSTOM_PRICING_JSON` | (none) | Custom pricing overrides for negotiated rates |

### 7.5 OTel Semantic Conventions (Proposed)

Span attribute naming following OpenTelemetry semantic conventions for GenAI:

```
# Per-request cost attributes (on LLM call span)
gen_ai.usage.input_tokens          # int
gen_ai.usage.output_tokens         # int
gen_ai.usage.total_tokens          # int
gen_ai.cost.input_usd              # float
gen_ai.cost.output_usd             # float
gen_ai.cost.total_usd              # float
gen_ai.cost.estimated_usd          # float (pre-call)
gen_ai.cost.model                  # string
gen_ai.cost.provider               # string

# Budget attributes (on budget check span)
gen_ai.budget.subject              # string
gen_ai.budget.limit_usd            # float
gen_ai.budget.current_usd          # float
gen_ai.budget.remaining_usd        # float
gen_ai.budget.window               # string (day, month)
gen_ai.budget.threshold_breached   # float (0.80 = 80%)
```

---

*End of report.*
