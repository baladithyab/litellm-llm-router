# 06 - Streaming Observability Architecture Review

**Date**: 2026-02-07
**Scope**: Evaluation of RouteIQ Gateway observability stack against industry-standard AI gateway streaming metrics, OpenTelemetry GenAI Semantic Conventions, and production monitoring best practices.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Industry Standards and Research](#2-industry-standards-and-research)
3. [Current State Assessment](#3-current-state-assessment)
4. [Gap Analysis](#4-gap-analysis)
5. [Architecture Recommendations](#5-architecture-recommendations)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Dashboard and Alerting Design](#7-dashboard-and-alerting-design)
8. [References](#8-references)

---

## 1. Executive Summary

RouteIQ has a solid observability foundation with three-pipeline ADOT Collector integration (traces to X-Ray, metrics to CloudWatch EMF, logs to CloudWatch Logs), distributed tracing for routing decisions, MCP tool calls, and A2A agent invocations, plus structured logging with trace correlation. However, the **MeterProvider produces zero custom metrics** -- no histograms, counters, or gauges are created anywhere in the codebase. This is the single largest observability gap.

The gateway lacks all key AI-specific observability signals: TTFT (Time to First Token), TPS (Tokens Per Second), token usage counters, cost tracking, per-model latency distributions, error rate breakdowns, and streaming chunk timing. These are table-stakes for production AI gateways and are standardized in the OpenTelemetry GenAI Semantic Conventions.

**Severity**: High. Without custom metrics, the gateway is operationally blind to its core function -- proxying LLM requests -- and cannot provide the data needed for cost management, capacity planning, SLA monitoring, or quality-of-service enforcement.

---

## 2. Industry Standards and Research

### 2.1 OpenTelemetry GenAI Semantic Conventions

The OpenTelemetry project has established a dedicated `gen_ai.*` attribute namespace under the Semantic Conventions specification. This is the authoritative standard for LLM/GenAI observability.

#### Span Attributes (gen_ai.*)

| Attribute | Type | Requirement | Description |
|-----------|------|-------------|-------------|
| `gen_ai.system` | string | Required | The GenAI system (e.g., `openai`, `anthropic`, `aws.bedrock`) |
| `gen_ai.request.model` | string | Required | Model name as requested by the caller |
| `gen_ai.request.max_tokens` | int | Recommended | Max tokens requested for completion |
| `gen_ai.request.temperature` | float | Recommended | Temperature parameter |
| `gen_ai.request.top_p` | float | Recommended | Top-p parameter |
| `gen_ai.request.stop_sequences` | string[] | Opt-in | Stop sequences |
| `gen_ai.response.id` | string | Recommended | Provider-specific response ID |
| `gen_ai.response.model` | string | Recommended | Actual model used in response |
| `gen_ai.response.finish_reasons` | string[] | Recommended | Finish reasons (stop, length, etc.) |
| `gen_ai.usage.input_tokens` | int | Recommended | Input/prompt tokens consumed |
| `gen_ai.usage.output_tokens` | int | Recommended | Output/completion tokens generated |

#### Metric Instruments (gen_ai.*)

| Metric | Instrument | Unit | Description |
|--------|-----------|------|-------------|
| `gen_ai.client.token.usage` | Histogram | `{token}` | Token usage per request (input + output) |
| `gen_ai.client.operation.duration` | Histogram | `s` | End-to-end duration of GenAI operations |
| `gen_ai.server.request.duration` | Histogram | `s` | Server-side request duration |
| `gen_ai.server.time_per_output_token` | Histogram | `s` | Time per output token (inter-token latency) |
| `gen_ai.server.time_to_first_token` | Histogram | `s` | Time to first token (TTFT) |

**Recommended histogram bucket boundaries** for duration metrics:
```
[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]
```

**Recommended histogram bucket boundaries** for token usage:
```
[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
```

### 2.2 TTFT (Time to First Token) Measurement

TTFT is the latency from the moment a request is sent to when the first token of the response arrives. For streaming LLM responses, this is the most important user-perceived latency metric.

**Measurement pattern for SSE/streaming**:
1. Record `t_request_sent` when the LLM API call begins
2. Wrap the streaming response iterator
3. On the **first yield/chunk**, record `t_first_token = now()`
4. `TTFT = t_first_token - t_request_sent`
5. Record as a histogram observation with model/provider dimensions

**Implementation approaches**:
- **LiteLLM callback**: Use `log_pre_api_call` for start time, first streaming callback for end time
- **ASGI middleware**: Wrap the response body iterator to detect first chunk
- **Response wrapper**: Wrap `StreamingResponse` to intercept first `__anext__` yield

**Common pitfalls**:
- Must measure at the gateway boundary, not inside the model inference
- For non-streaming requests, TTFT equals the full response time
- Must handle connection establishment time separately from inference latency

### 2.3 TPS (Tokens Per Second) Throughput

TPS measures the rate of token generation during streaming. It is a throughput metric that indicates model inference performance.

**Measurement pattern**:
1. Count output tokens received over the streaming duration
2. `TPS = output_tokens / (t_last_token - t_first_token)`
3. Record as a histogram with model/provider dimensions

**Variant**: Inter-Token Latency (ITL) = time between consecutive tokens. The OTel GenAI convention calls this `gen_ai.server.time_per_output_token`.

**Token counting approaches**:
- Use LiteLLM built-in token counting (it tracks `usage` in response metadata)
- Parse `usage` from the final streaming chunk (many providers send token counts in the last SSE event)
- Approximate via tiktoken for OpenAI-compatible models
- Use provider-specific response headers (e.g., `x-ratelimit-remaining-tokens`)

### 2.4 AI Observability Platform Patterns

#### Langfuse
- Tracks TTFT, total latency, token counts (input/output/total), cost per request
- Uses "generations" as the unit of observation (one LLM call = one generation)
- Calculates cost using per-model token pricing tables
- Provides latency distribution (P50/P90/P99) per model
- Scores quality via user feedback and evaluator functions

#### Helicone
- Measures TTFT for streaming via response header timing
- Tracks tokens/second as a first-class metric
- Cost tracking with model-specific pricing
- Per-user, per-model, per-key analytics
- Custom properties for business dimension tracking

#### Portkey
- Comprehensive streaming metrics: TTFT, TPS, total latency
- Cost tracking across providers with unified pricing
- Per-gateway, per-virtual-key analytics
- Cache hit rate tracking for semantic caching
- Error rate by provider/model/status code

#### OpenLIT
- Open-source, built natively on OpenTelemetry GenAI Semantic Conventions
- Auto-instruments Python LLM libraries (openai, anthropic, langchain, etc.)
- Emits `gen_ai.client.token.usage`, `gen_ai.client.operation.duration`
- Uses OTel histograms for latency distributions
- GPU monitoring integration for self-hosted models
- Exports to any OTel-compatible backend

### 2.5 CloudWatch EMF for AI Metrics

AWS CloudWatch Embedded Metric Format (EMF) enables publishing custom metrics with high-cardinality dimensions directly from structured logs. The ADOT Collector `awsemf` exporter converts OTel metrics to EMF.

**Key patterns for AI workloads**:
- Use histograms for latency distributions (EMF supports statistics: min, max, sum, count, p50, p90, p99)
- Use counters for token usage (monotonic, aggregatable across instances)
- Use gauges for concurrent request counts and circuit breaker states
- Dimension design: `{model, provider, team_id, endpoint}` -- keep cardinality under 10,000 per namespace
- Namespace: `RouteIQ/Gateway` (already configured in otel-collector-config.yaml)

**EMF dimension considerations**:
- Too many unique dimension combinations = high CloudWatch cost
- Roll up optional dimensions (e.g., `team_id` only when quota enforcement is active)
- Use `NoDimensionRollup` (already configured) to prevent CloudWatch from generating every possible dimension combination

### 2.6 Streaming Response Instrumentation

Instrumenting SSE/streaming responses in an ASGI framework requires special care:

**Challenge**: Standard middleware patterns buffer the entire response, which breaks streaming. RouteIQ already handles this correctly with `BackpressureMiddleware` using pure ASGI (not `BaseHTTPMiddleware`).

**Pattern for streaming metrics**:
```
1. ASGI middleware wraps the `send` callable
2. On `http.response.start`: record request metadata
3. On first `http.response.body`: record TTFT
4. On each `http.response.body`: increment chunk counter, record inter-chunk timing
5. On final `http.response.body` (more_body=False): record total duration, TPS
```

**Alternative**: LiteLLM callback approach (preferred for RouteIQ):
- LiteLLM callback interface provides `log_stream_event` (for each chunk) and `log_success_event` (with aggregated usage)
- The `response_obj` in `log_success_event` contains `usage` with `prompt_tokens`, `completion_tokens`, `total_tokens`
- Streaming timing can be derived from `start_time` / `end_time` parameters

---

## 3. Current State Assessment

### 3.1 What Exists

#### Tracing (Good)

| Component | Status | Details |
|-----------|--------|---------|
| TracerProvider initialization | Working | `ObservabilityManager._init_tracing()` - reuses existing SDK providers |
| OTLP trace export | Working | BatchSpanProcessor to ADOT Collector |
| Router decision spans | Working | `set_router_decision_attributes()` sets `router.*` attributes on spans |
| RouterDecisionMiddleware | Working | Emits `router.*` span attributes on `/v1/chat/completions` |
| RouterDecisionCallback | Working | LiteLLM callback for `log_pre_api_call` |
| MCP tracing | Working | `mcp_tracing.py` - spans for tool calls, registration, health checks |
| A2A tracing | Working | `a2a_tracing.py` - spans for agent invocations including streaming |
| Telemetry contracts | Working | `telemetry_contracts.py` - versioned `routeiq.router_decision.v1` schema |
| Evaluator plugin spans | Working | `eval.*` span attributes for post-invocation scoring |
| Sampler configuration | Working | 3-tier env var support (OTEL, ROUTEIQ, legacy LLMROUTER) |

#### Logging (Good)

| Component | Status | Details |
|-----------|--------|---------|
| LoggerProvider initialization | Working | `ObservabilityManager._init_logging()` |
| OTLP log export | Working | BatchLogRecordProcessor to ADOT Collector |
| Trace correlation | Working | `LoggingInstrumentor` adds trace/span IDs to Python logs |
| Structured logging | Working | `log_routing_decision()`, `log_error_with_trace()` |

#### Metrics (Skeleton Only)

| Component | Status | Details |
|-----------|--------|---------|
| MeterProvider initialization | Configured | `ObservabilityManager._init_metrics()` creates provider + OTLP exporter |
| Meter instance | Available | `get_meter()` returns a Meter, but nothing uses it |
| Custom instruments | **MISSING** | No histograms, counters, or gauges created anywhere |
| Token tracking | **MISSING** | No input/output/total token counting |
| Latency histograms | **MISSING** | No request duration or routing latency histograms |
| TTFT measurement | **MISSING** | No streaming first-token timing |
| TPS measurement | **MISSING** | No tokens-per-second calculation |
| Error rate counters | **MISSING** | No error counting by model/provider/status |
| Cost tracking | **MISSING** | No per-request cost calculation |

#### ADOT Collector Pipeline (Good)

The `otel-collector-config.yaml` has a complete three-pipeline configuration:
- **Traces**: OTLP receiver -> batch processor -> AWS X-Ray exporter
- **Metrics**: OTLP receiver -> batch/metrics processor (60s) -> CloudWatch EMF exporter (`RouteIQ/Gateway` namespace)
- **Logs**: OTLP receiver -> batch/logs processor -> CloudWatch Logs exporter (`/routeiq/gateway`)

The metrics pipeline is ready to receive and export metrics -- it just has nothing to export because the application produces zero custom metric observations.

### 3.2 Callback Integration Points

RouteIQ has two LiteLLM callback integrations that could be extended for metrics:

1. **RouterDecisionCallback** (`router_decision_callback.py`): Currently only sets span attributes, does not record metrics. Has `log_pre_api_call` (before LLM call), `log_success_event`, and `log_failure_event` hooks.

2. **PluginCallbackBridge** (`plugin_callback_bridge.py`): Bridges LiteLLM callbacks to GatewayPlugin hooks. Has `async_log_pre_api_call`, `async_log_success_event`, `async_log_failure_event`. Receives `response_obj`, `start_time`, `end_time`.

Both provide the necessary hook points for adding metrics instrumentation without modifying the core request path.

---

## 4. Gap Analysis

### 4.1 OpenTelemetry GenAI Semantic Conventions Compliance

| Convention | RouteIQ Status | Gap |
|------------|---------------|-----|
| `gen_ai.system` span attribute | NOT SET | Must set to provider name (openai, anthropic, bedrock) |
| `gen_ai.request.model` span attribute | PARTIAL | Set as `router.model_selected` (custom prefix, not standard) |
| `gen_ai.request.max_tokens` | NOT SET | Available in request body, not extracted |
| `gen_ai.request.temperature` | NOT SET | Available in request body, not extracted |
| `gen_ai.response.model` | NOT SET | Available in LLM response, not extracted |
| `gen_ai.response.finish_reasons` | NOT SET | Available in LLM response, not extracted |
| `gen_ai.usage.input_tokens` | NOT SET | Available in `response_obj.usage` via callbacks |
| `gen_ai.usage.output_tokens` | NOT SET | Available in `response_obj.usage` via callbacks |
| `gen_ai.client.token.usage` metric | NOT CREATED | Histogram instrument not defined |
| `gen_ai.client.operation.duration` metric | NOT CREATED | Histogram instrument not defined |
| `gen_ai.server.time_to_first_token` metric | NOT CREATED | TTFT not measured |
| `gen_ai.server.time_per_output_token` metric | NOT CREATED | ITL not measured |

### 4.2 Critical Missing Metrics

| Metric | Priority | Impact |
|--------|----------|--------|
| Request latency histogram (P50/P90/P99) | **P0** | Cannot monitor SLAs or detect degradation |
| Token usage counters (input/output) | **P0** | Cannot track cost or capacity |
| Error rate by model/provider | **P0** | Cannot detect provider outages |
| TTFT histogram | **P1** | Cannot measure streaming user experience |
| Requests per second counter | **P1** | Cannot monitor throughput |
| Active request gauge | **P1** | Cannot monitor concurrency |
| TPS (tokens per second) | **P2** | Cannot benchmark model throughput |
| Cost per request | **P2** | Cannot allocate costs to teams |
| Circuit breaker state | **P2** | Cannot monitor resilience system |
| Routing strategy distribution | **P2** | Cannot analyze routing effectiveness |
| Cache hit/miss rate | **P3** | Future: semantic caching observability |

### 4.3 Existing Code That Should Emit Metrics But Does Not

| Location | What Happens | What Should Also Happen |
|----------|-------------|------------------------|
| `RouterDecisionCallback.log_pre_api_call` | Sets span attributes | Record start time, increment active gauge |
| `RouterDecisionCallback.log_success_event` | No-op | Record duration histogram, token counters, success counter |
| `RouterDecisionCallback.log_failure_event` | No-op | Record error counter with model/provider/error_type dims |
| `PluginCallbackBridge.async_log_success_event` | Calls plugin hooks | Record token usage, latency, cost metrics |
| `PluginCallbackBridge.async_log_failure_event` | Calls plugin hooks | Record error metrics |
| `BackpressureMiddleware.__call__` | 503 on overload | Record `gateway.request.rejected` counter, active gauge |
| `CircuitBreaker.execute` | State transitions | Record `gateway.circuit_breaker.state_change` counter |
| `ObservabilityManager._init_metrics` | Creates MeterProvider | Create all instrument definitions |

---

## 5. Architecture Recommendations

### 5.1 Metrics Instrument Registry

Create a centralized metrics module that defines all OTel instruments once during initialization. This follows the OTel pattern of creating instruments at startup and recording observations at runtime.

**Proposed module**: `src/litellm_llmrouter/metrics.py`

**Instruments to create**:

```python
# === GenAI Semantic Convention Metrics ===

# gen_ai.client.operation.duration (Histogram, seconds)
# Dimensions: gen_ai.system, gen_ai.request.model, gen_ai.operation.name
# Bucket boundaries: [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28,
#                     2.56, 5.12, 10.24, 20.48, 40.96, 81.92]

# gen_ai.client.token.usage (Histogram, tokens)
# Dimensions: gen_ai.system, gen_ai.request.model, gen_ai.token.type
# Bucket boundaries: [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144]

# gen_ai.server.time_to_first_token (Histogram, seconds)
# Dimensions: gen_ai.system, gen_ai.request.model
# Bucket boundaries: [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28,
#                     2.56, 5.12, 10.24, 20.48]

# gen_ai.server.time_per_output_token (Histogram, seconds)
# Dimensions: gen_ai.system, gen_ai.request.model
# Bucket boundaries: [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

# === Gateway Operational Metrics ===

# gateway.request.total (Counter)
# Dimensions: method, endpoint, status_code, model, provider

# gateway.request.error (Counter)
# Dimensions: model, provider, error_type, status_code

# gateway.request.active (UpDownCounter)
# Dimensions: endpoint

# gateway.request.rejected (Counter)
# Dimensions: reason (backpressure, drain, policy)

# === Routing Metrics ===

# gateway.routing.decision.duration (Histogram, seconds)
# Dimensions: strategy, outcome

# gateway.routing.strategy.usage (Counter)
# Dimensions: strategy, model_selected

# gateway.routing.fallback.total (Counter)
# Dimensions: original_model, fallback_model, reason

# === Cost Tracking ===

# gateway.cost.total (Counter, USD)
# Dimensions: model, provider, team_id

# === Resilience Metrics ===

# gateway.circuit_breaker.state (ObservableGauge)
# Dimensions: breaker_name, state (closed, open, half_open)

# gateway.circuit_breaker.transitions (Counter)
# Dimensions: breaker_name, from_state, to_state
```

### 5.2 TTFT Measurement via Streaming Response Instrumentation

**Approach**: Use the LiteLLM callback interface, specifically by creating a new `StreamingMetricsCallback` that hooks into LiteLLM's streaming lifecycle.

**Design**:

```
Request flow:
  Client -> Gateway -> LiteLLM Router -> Provider (streaming)
                |
                v
          log_pre_api_call()        -> record t_start per request
          [first streaming chunk]   -> record t_first_token, compute TTFT
          log_success_event()       -> record total duration, token usage, TPS
          log_failure_event()       -> record error metrics
```

**Key design decisions**:
- Store per-request timing state in a thread-safe dictionary keyed by `litellm_call_id`
- Clean up state in `log_success_event` / `log_failure_event` to prevent memory leaks
- Use `time.perf_counter()` for high-resolution timing (not `time.time()`)
- For TTFT on non-streaming requests, TTFT equals total request duration

### 5.3 Token Counting Integration

LiteLLM already performs token counting internally and exposes it via:
1. `response_obj.usage.prompt_tokens` / `response_obj.usage.completion_tokens` in callbacks
2. `kwargs["standard_logging_object"]` which contains token counts
3. `kwargs["litellm_params"]["metadata"]` which may contain cached token counts

**Recommendation**: Extract token usage from `response_obj` in `log_success_event` callbacks. LiteLLM normalizes token counts across providers, so this works for OpenAI, Anthropic, Bedrock, and all other supported providers.

### 5.4 GenAI Span Attribute Enhancement

The existing `RouterDecisionMiddleware` and `RouterDecisionCallback` should be extended (or a parallel callback added) to set `gen_ai.*` span attributes alongside the existing `router.*` attributes.

**Mapping**:
```
gen_ai.system              <- litellm_params["custom_llm_provider"] or model prefix
gen_ai.request.model       <- kwargs["model"] or body["model"]
gen_ai.response.model      <- response_obj["model"]
gen_ai.usage.input_tokens  <- response_obj.usage.prompt_tokens
gen_ai.usage.output_tokens <- response_obj.usage.completion_tokens
gen_ai.response.finish_reasons <- [response_obj.choices[0].finish_reason]
```

### 5.5 Integration with Existing OTel Pipeline

The architecture should follow this flow:

```
Application Code
    |
    v
metrics.py (instrument definitions)
    |
    v
ObservabilityManager._init_metrics() (MeterProvider + OTLP exporter)
    |
    v
ADOT Collector (otel-collector-config.yaml)
    |
    v
awsemf exporter (CloudWatch EMF)
    |
    v
CloudWatch Metrics (RouteIQ/Gateway namespace)
    |
    v
CloudWatch Dashboards and Alarms
```

The existing `otel-collector-config.yaml` already has the metrics pipeline configured. All that is needed is to create the instruments and record observations -- the export pipeline is ready.

**EMF dimension mapping** (should be added to otel-collector-config.yaml):
```yaml
awsemf:
  region: ${AWS_REGION:-us-east-1}
  namespace: RouteIQ/Gateway
  log_group_name: /routeiq/metrics
  dimension_rollup_option: NoDimensionRollup
  metric_declarations:
    - dimensions: [[gen_ai.request.model, gen_ai.system]]
      metric_name_selectors:
        - "gen_ai.client.operation.duration"
        - "gen_ai.client.token.usage"
        - "gen_ai.server.time_to_first_token"
    - dimensions: [[model, endpoint], [model]]
      metric_name_selectors:
        - "gateway.request.total"
        - "gateway.request.error"
    - dimensions: [[strategy]]
      metric_name_selectors:
        - "gateway.routing.decision.duration"
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (P0 - Immediate)

**Goal**: Create the metrics instrument registry and emit basic request-level metrics.

1. **Create `src/litellm_llmrouter/metrics.py`**
   - Define all OTel instruments (histograms, counters, gauges)
   - Initialize from `ObservabilityManager._init_metrics()` or as a lazy singleton
   - Use standard GenAI semantic convention names

2. **Extend `RouterDecisionCallback` or create `StreamingMetricsCallback`**
   - In `log_pre_api_call`: record request start time, increment active request gauge
   - In `log_success_event`: record duration histogram, token usage histogram, decrement active gauge
   - In `log_failure_event`: record error counter, decrement active gauge
   - Extract token counts from `response_obj.usage`

3. **Add `gen_ai.*` span attributes**
   - Set `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*` on the current span
   - Do this alongside existing `router.*` attributes (additive, not replacing)

4. **Update `otel-collector-config.yaml`**
   - Add `metric_declarations` for dimension mapping in the `awsemf` exporter

**Estimated effort**: 2-3 days

### Phase 2: Streaming Metrics (P1 - Near-term)

**Goal**: Measure TTFT and streaming-specific metrics.

1. **Implement TTFT measurement**
   - Use `async_log_stream_event` callback or streaming-aware wrapper
   - Track first-chunk timing per `litellm_call_id`
   - Record `gen_ai.server.time_to_first_token` histogram

2. **Implement TPS measurement**
   - Count output tokens from streaming chunks
   - Calculate TPS = output_tokens / stream_duration
   - Record `gen_ai.server.time_per_output_token` histogram

3. **Add request counter and active request gauge**
   - Record in ASGI middleware or callback
   - Dimensions: endpoint, model, status_code

**Estimated effort**: 2-3 days

### Phase 3: Operational Intelligence (P2 - Short-term)

**Goal**: Cost tracking, resilience metrics, routing analytics.

1. **Cost tracking** with configurable model pricing table
2. **Circuit breaker metrics** hooking into state transitions
3. **Routing analytics** counters for strategy usage and fallback triggers

**Estimated effort**: 2-3 days

### Phase 4: Future Enhancements (P3 - Medium-term)

1. **Cache metrics** (when semantic caching is implemented)
2. **Per-team/per-key analytics** for token usage and cost allocation
3. **Quality metrics** from evaluator scores, guardrail triggers, content filters

---

## 7. Dashboard and Alerting Design

### 7.1 Primary Dashboard: Gateway Overview

```
Row 1: Key Metrics (single-value panels)
+------------------+------------------+------------------+------------------+
| Requests/sec     | P99 Latency      | Error Rate       | Active Requests  |
| (last 5m avg)    | (last 5m)        | (last 5m)        | (current)        |
+------------------+------------------+------------------+------------------+

Row 2: Latency and Throughput (time series)
+--------------------------------------+--------------------------------------+
| Request Latency Distribution         | Token Usage Over Time                |
| (P50, P90, P99 lines by model)       | (stacked: input + output by model)   |
+--------------------------------------+--------------------------------------+

Row 3: TTFT and Errors (time series)
+--------------------------------------+--------------------------------------+
| TTFT Distribution by Model           | Error Rate by Provider               |
| (P50, P90, P99 lines)                | (stacked bar: error_type)            |
+--------------------------------------+--------------------------------------+
```

### 7.2 Streaming Performance Dashboard

```
Row 1: Streaming KPIs
+------------------+------------------+------------------+------------------+
| TTFT P50         | TTFT P99         | Avg TPS          | Stream Error %   |
+------------------+------------------+------------------+------------------+

Row 2: Streaming Detail
+--------------------------------------+--------------------------------------+
| TTFT Heatmap (model x time)          | Tokens/Second by Model               |
+--------------------------------------+--------------------------------------+

Row 3: Streaming Health
+--------------------------------------+--------------------------------------+
| Inter-Token Latency Distribution     | Streaming vs Non-Streaming Split     |
+--------------------------------------+--------------------------------------+
```

### 7.3 Cost and Capacity Dashboard

```
Row 1: Cost KPIs
+------------------+------------------+------------------+------------------+
| Total Cost Today | Cost/1K Requests | Most Expensive   | Token Budget     |
|                  |                  | Model            | Utilization      |
+------------------+------------------+------------------+------------------+

Row 2: Cost Breakdown
+--------------------------------------+--------------------------------------+
| Cost by Model (stacked bar, hourly)  | Cost by Team (pie chart)             |
+--------------------------------------+--------------------------------------+

Row 3: Capacity
+--------------------------------------+--------------------------------------+
| Concurrent Requests vs Limit         | Circuit Breaker States               |
+--------------------------------------+--------------------------------------+
```

### 7.4 Alert Rules

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High P99 Latency | `P99(gen_ai.client.operation.duration) > 30s` for 5m | Warning | Investigate slow model |
| TTFT Degradation | `P99(gen_ai.server.time_to_first_token) > 10s` for 5m | Warning | Check provider status |
| High Error Rate | `error_rate > 5%` for 3m | Critical | Page on-call |
| Provider Down | `error_rate for provider > 50%` for 2m | Critical | Circuit breaker should open |
| Token Budget Exceeded | `daily_tokens > budget * 0.9` | Warning | Notify team lead |
| Backpressure Active | `gateway.request.rejected > 0` for 1m | Warning | Scale up or reduce traffic |
| Circuit Breaker Open | `circuit_breaker.state == OPEN` | Warning | Provider dependency down |
| Cost Spike | `hourly_cost > 2 * avg_hourly_cost` | Warning | Check for runaway requests |

---

## 8. References

### Files Analyzed

| File | Path |
|------|------|
| observability.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/observability.py` |
| telemetry_contracts.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/telemetry_contracts.py` |
| router_decision_callback.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/router_decision_callback.py` |
| mcp_tracing.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/mcp_tracing.py` |
| a2a_tracing.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/a2a_tracing.py` |
| plugin_callback_bridge.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/gateway/plugin_callback_bridge.py` |
| plugin_middleware.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/gateway/plugin_middleware.py` |
| resilience.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/resilience.py` |
| app.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/gateway/app.py` |
| startup.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/startup.py` |
| evaluator.py | `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/gateway/plugins/evaluator.py` |
| otel-collector-config.yaml | `/Users/baladita/Documents/DevBox/RouteIQ/config/otel-collector-config.yaml` |
| pyproject.toml | `/Users/baladita/Documents/DevBox/RouteIQ/pyproject.toml` |

### Standards

- OpenTelemetry Semantic Conventions for GenAI: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- OpenTelemetry Metrics API: https://opentelemetry.io/docs/specs/otel/metrics/
- AWS CloudWatch EMF Specification: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Embedded_Metric_Format_Specification.html
- ADOT Collector awsemf exporter: https://aws-otel.github.io/docs/getting-started/cloudwatch-metrics

### Related Projects

- OpenLIT: https://github.com/openlit/openlit
- Langfuse: https://langfuse.com/
- Helicone: https://www.helicone.ai/
- Portkey: https://portkey.ai/

---

## Appendix A: Codebase Metrics Instrumentation Audit

A grep for OTel metric instrument creation (`create_histogram`, `create_counter`, `create_gauge`) across the entire `src/` directory returned **zero matches**. The `get_meter()` function in `observability.py` is never called by any other module. The `_meter` attribute on `ObservabilityManager` is set during `_init_metrics()` but the returned Meter is never used to create any instruments.

The `telemetry_contracts.py` module defines fields for `input_tokens`, `output_tokens`, and `total_tokens` in the `RoutingOutcomeData` dataclass, but these are only populated if callers explicitly set them via the builder pattern -- and currently no caller does.

The `PluginCallbackBridge.async_log_success_event` receives `response_obj`, `start_time`, and `end_time` from LiteLLM -- all the data needed for metrics -- but currently only passes them to plugin hooks without recording any OTel metrics.

## Appendix B: OTel Dependency Versions

From `pyproject.toml`:
```
opentelemetry-api>=1.22.0
opentelemetry-sdk>=1.22.0
opentelemetry-exporter-otlp>=1.22.0
opentelemetry-instrumentation>=0.43b0
opentelemetry-instrumentation-logging>=0.43b0
opentelemetry-exporter-otlp-proto-grpc>=1.22.0
opentelemetry-exporter-otlp-proto-http>=1.22.0
opentelemetry-instrumentation-fastapi>=0.43b0
opentelemetry-instrumentation-httpx>=0.43b0
opentelemetry-instrumentation-requests>=0.43b0
```

These versions support the full Metrics API including Histogram, Counter, UpDownCounter, and ObservableGauge. No additional dependencies are needed.
