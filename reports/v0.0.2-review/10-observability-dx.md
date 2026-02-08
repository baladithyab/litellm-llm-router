# RouteIQ v0.0.2 Review: Observability & Developer Experience

**Reviewer**: DX/Observability Specialist
**Date**: 2026-02-07
**Scope**: OTel pipeline, metrics, traces, logs, DX by persona, documentation, competitor comparison

---

## Part 1: Observability Maturity Assessment

### Summary Scores (1-5 Scale)

| Signal    | Score | Rationale |
|-----------|-------|-----------|
| **Traces** | 4/5 | Full lifecycle coverage for LLM, MCP, and A2A. GenAI semantic conventions adopted. Router decision attributes are first-class. Missing: streaming TTFT span events, cache hit/miss spans in practice. |
| **Metrics** | 3.5/5 | Solid foundation with `GatewayMetrics` registry and `CostTrackerPlugin`. GenAI semantic convention metric names used. Missing: cache hit rate metric, circuit breaker state gauge, plugin execution time histogram, per-team/per-key rate metrics. |
| **Logs**  | 3/5 | OTel log correlation via `LoggingInstrumentor` is wired. Structured logging used throughout. Missing: consistent structured JSON log format across all modules, no log-level per-module control, no request-scoped log enrichment middleware. |

### Overall Observability Maturity: 3.5/5

The gateway has a surprisingly strong observability foundation for a v0.0.2 project. The three pillars (traces, metrics, logs) are all wired through OTel with OTLP export. The main gap is that several metrics are *defined* but the recording paths are incomplete or only fire in specific code paths.

---

## Part 2: Metric Inventory

### GatewayMetrics Registry (`metrics.py`)

| Metric Name | Type | Labels | Dashboard-Ready? | Notes |
|-------------|------|--------|-------------------|-------|
| `gen_ai.client.operation.duration` | Histogram | `gen_ai.request.model`, `gen_ai.system` | Yes | Custom exponential buckets (10ms-82s). Recorded in `RouterDecisionCallback.log_success_event`. |
| `gen_ai.client.token.usage` | Histogram | `gen_ai.request.model`, `gen_ai.system`, `gen_ai.token.type` | Yes | Separate input/output recordings. Buckets 1-4M tokens. |
| `gen_ai.server.time_to_first_token` | Histogram | -- | **No** | Defined but **no recording site found**. TTFT is never recorded anywhere in the codebase. |
| `gateway.request.total` | Counter | `model`, `provider`, `status` | Yes | Recorded on success in callback. |
| `gateway.request.error` | Counter | `model`, `provider`, `error_type` | Yes | Recorded on failure in callback. |
| `gateway.request.active` | UpDownCounter | `model` | Partial | Incremented in `log_pre_api_call`, decremented in success/failure. Risk: if callback never fires (e.g., middleware short-circuit), gauge drifts. |
| `gateway.routing.decision.duration` | Histogram | -- | **No** | Defined but **no recording site found**. The routing decision latency is set as a span attribute (`router.latency_ms`) but never recorded to this histogram. |
| `gateway.routing.strategy.usage` | Counter | -- | **No** | Defined but **no recording site found**. Strategy name is a span attribute only. |
| `gateway.cost.total` | Counter | -- | **No** | Defined but **no recording site found** in `metrics.py` consumers. The `CostTrackerPlugin` uses its own separate `llm_cost_total` counter. |
| `gateway.circuit_breaker.transitions` | Counter | -- | **No** | Defined but **no recording site found**. `resilience.py` does not import or record to this metric. |

### CostTrackerPlugin Metrics (`gateway/plugins/cost_tracker.py`)

| Metric Name | Type | Labels | Dashboard-Ready? | Notes |
|-------------|------|--------|-------------------|-------|
| `llm_cost_total` | Counter | `model`, `team`, `user` | Yes | Actual USD cost from `litellm.completion_cost`. |
| `llm_tokens_total` | Counter | `model`, `team`, `direction` | Yes | Input/output split. |
| `llm_cost_per_request` | Histogram | `model` | Yes | Cost distribution per request. |
| `llm_active_requests` | UpDownCounter | `model` | Partial | Same gauge drift risk as `gateway.request.active`. |
| `llm_cost_errors_total` | Counter | `model` | Yes | Error count by model. |

### Cardinality Risk Analysis

| Label | Risk Level | Rationale |
|-------|-----------|-----------|
| `model` | **Medium** | Bounded by configured models (~5-20 typically). Safe. |
| `provider` | **Low** | Bounded by supported providers (~10). Safe. |
| `error_type` | **Medium-High** | Python exception class names are unbounded. If custom exceptions proliferate, cardinality grows. Recommend: normalize to categories (auth, rate_limit, timeout, server_error, client_error). |
| `user` (on `llm_cost_total`) | **HIGH** | Unbounded. If per-user cost tracking is enabled with thousands of users, this metric explodes. Recommend: remove `user` label from metrics and track per-user cost in logs/traces only. Use `team` as the highest-granularity metric label. |
| `team` | **Low-Medium** | Bounded by configured teams. Usually safe (<100). |
| `gen_ai.token.type` | **Low** | Only `input`/`output`. Safe. |
| `status` | **Low** | Only `success`/`error`. Safe. |

**Critical Finding**: The `user` label on `llm_cost_total` is the highest cardinality risk in the system. In a multi-tenant deployment with API keys per user, this could create thousands of time series per model. This should be removed from the metric label set and tracked via span attributes or log fields instead.

---

## Part 3: Trace Completeness

### Request Lifecycle Trace Coverage

```
Client Request
  |
  +-- [RequestID Middleware] -- Sets X-Request-ID header, no span
  |
  +-- [PolicyEngine Middleware] -- No OTel span
  |
  +-- [RouterDecisionMiddleware] -- Sets router.* span attributes on current span
  |       |
  |       +-- router.strategy, router.model_selected, router.candidates_evaluated
  |       +-- router.decision_outcome, router.decision_reason, router.latency_ms
  |       +-- gen_ai.request.model, gen_ai.operation.name
  |
  +-- [BackpressureMiddleware] -- No OTel span (wraps ASGI directly)
  |
  +-- [FastAPI Route Handler] -- Auto-instrumented span (if OTEL auto-instrumentation is on)
  |       |
  |       +-- [LiteLLM Router] -- LiteLLM's built-in spans (if OTEL callback enabled)
  |       |       |
  |       |       +-- [RouterDecisionCallback.log_pre_api_call]
  |       |       |       +-- Sets gen_ai.* span attributes
  |       |       |       +-- Records request_active gauge
  |       |       |
  |       |       +-- [RouterDecisionCallback.log_success_event]
  |       |       |       +-- Records duration histogram, token usage, request counter
  |       |       |       +-- Sets gen_ai.usage.input_tokens, gen_ai.usage.output_tokens
  |       |       |
  |       |       +-- [CostTrackerPlugin.on_llm_success]
  |       |               +-- Sets llm.cost.* span attributes (9 attributes)
  |       |               +-- Records cost counter, token counter, cost histogram
  |       |
  |       +-- [MCP Tool Call] -- mcp.tool.call/{tool_name} span
  |       |       +-- mcp.server.id, mcp.tool.name, mcp.transport
  |       |       +-- mcp.success, mcp.duration_ms, mcp.error
  |       |       +-- [Evaluator hooks] -- eval.plugin, eval.score, eval.status
  |       |
  |       +-- [A2A Agent Call] -- a2a.agent.send/{agent_id} span
  |               +-- a2a.agent.id, a2a.agent.name, a2a.agent.url
  |               +-- a2a.method, a2a.success, a2a.duration_ms
  |               +-- W3C trace context propagation on outbound HTTP
  |               +-- [Evaluator hooks] -- eval.plugin, eval.score, eval.status
  |
  +-- Response
```

### Trace Gaps

1. **No dedicated span for routing decision** -- Router attributes are set on the existing HTTP span, not a child span. This means you cannot filter for routing-only latency in trace queries.

2. **No cache operation spans** -- `ObservabilityManager.create_cache_span()` exists but is never called from the semantic cache module. Cache hits/misses are invisible in traces.

3. **No policy engine span** -- Policy evaluation happens in ASGI middleware with no OTel instrumentation. A slow or failing policy check is invisible.

4. **No backpressure span/event** -- When backpressure rejects a request (503), there is no span event recording the rejection reason or queue depth.

5. **No plugin lifecycle spans** -- Plugin startup/shutdown durations are logged but not traced.

6. **Streaming TTFT not instrumented** -- The `gen_ai.server.time_to_first_token` histogram is defined but never recorded. For streaming responses, TTFT is a critical user-facing metric.

### Telemetry Contracts

The `telemetry_contracts.py` module defines a rich `RouterDecisionEvent` v1.1 schema with:
- PII-safe design (only `query_length`, no content)
- Experiment/A/B testing assignment tracking
- Candidate deployment scoring
- Timing breakdown (strategy_ms, embedding_ms, filter_ms)
- Fallback chain recording

This is well-designed but **the contract is not actively emitted** -- the `RouterDecisionEventBuilder` is defined but there is no code path that calls `.build()` and emits the event as a span event. The simpler `set_router_decision_attributes()` function is used instead. The rich contract exists as a schema definition without runtime usage.

---

## Part 4: Log Correlation

### What Works

- `LoggingInstrumentor().instrument(set_logging_format=True)` injects `trace_id`, `span_id`, and `trace_flags` into Python log records
- `OTLPLogExporter` exports logs to the collector
- `LoggingHandler` attached to root logger

### Gaps

1. **No structured JSON formatter** -- Logs use Python's default format with OTel fields appended. For log aggregation (CloudWatch Logs Insights, Elasticsearch), a JSON log formatter would make queries much easier.

2. **No request-scoped log enrichment** -- There is no middleware that adds `request_id`, `model`, `team_id`, or `user_id` to all log records within a request scope. These fields exist in trace spans but not in log lines.

3. **Inconsistent logger naming** -- Some modules use `logging.getLogger(__name__)` (correct), others use `verbose_proxy_logger` from LiteLLM. This means some log lines cannot be filtered by module.

4. **No log sampling correlation** -- When trace sampling is set to 10%, logs still fire at 100%. This means you get log lines without corresponding traces, which breaks the correlation story.

---

## Part 5: Dashboard & Alert Readiness

### Dashboard Readiness Assessment

**Can you build a Grafana dashboard from the current metrics?** Partially.

**Ready panels:**
- Request rate by model/provider (from `gateway.request.total`)
- Error rate by model/provider/error_type (from `gateway.request.error`)
- Request duration percentiles by model (from `gen_ai.client.operation.duration`)
- Token usage distribution (from `gen_ai.client.token.usage`)
- Cost per request distribution (from `llm_cost_per_request`)
- Cumulative cost by model/team (from `llm_cost_total`)
- Active requests gauge (from `gateway.request.active`, with drift caveat)

**Not ready (metric defined but unrecorded):**
- TTFT percentiles (metric exists, no recording)
- Routing decision latency (metric exists, no recording)
- Circuit breaker state (metric exists, no recording)
- Routing strategy usage (metric exists, no recording)

**Not ready (metric not defined):**
- Cache hit/miss rate
- Plugin execution time
- MCP tool call rate/latency (span-only, no metric)
- A2A agent call rate/latency (span-only, no metric)
- Request queue depth / backpressure rejections
- Config hot-reload events
- Health check failure rate

### Alert Readiness Assessment

**Can you set up PagerDuty alerts?** Partially.

**Ready alerts:**
- Error rate spike: `rate(gateway.request.error[5m]) / rate(gateway.request.total[5m]) > 0.05`
- Latency degradation: `histogram_quantile(0.99, gen_ai.client.operation.duration) > 30`
- Cost spike: `rate(llm_cost_total[1h]) > threshold`

**Not ready:**
- Circuit breaker open (no metric)
- Model availability (no metric)
- Backpressure activation (no metric)
- Cache degradation (no metric)
- Plugin failure (no metric)

---

## Part 6: Developer Experience Audit

### DX Scorecard

| Persona | Score | Grade |
|---------|-------|-------|
| **Gateway Operator** | 3.0/5 | B- |
| **Plugin Developer** | 3.5/5 | B |
| **API Consumer** | 2.5/5 | C+ |

### Gateway Operator DX

**First-time setup (3/5):**
- `uv sync && uv run python -m litellm_llmrouter.startup --config config/config.yaml` is clear
- Docker Compose files exist for multiple deployment topologies (basic, HA, OTel, local-test)
- `.env.example` exists at root level
- Missing: a single `make quickstart` or `./scripts/quickstart.sh` command
- Missing: config template with inline comments explaining each section

**Configuration complexity (3/5):**
- 20+ environment variables documented in CLAUDE.md
- `docs/configuration.md` exists but was not found in the docs glob (need to verify)
- No `--validate-config` CLI flag to check config without starting the server
- No config schema validation (Pydantic model or JSON Schema for config.yaml)
- Env var naming is inconsistent: `LITELLM_*`, `LLMROUTER_*`, `ROUTEIQ_*`, `OTEL_*` prefixes coexist

**Error messaging (3/5):**
- Auth errors are clear ("Admin API key required")
- OTel initialization failures are logged clearly
- Plugin loading errors are logged with the module path
- Missing: startup banner showing active configuration (which features enabled, which disabled)
- Missing: config validation errors before startup

**Log readability (3/5):**
- Structured logging with `logging.getLogger(__name__)` in most modules
- Key events are logged at INFO level (plugin startup, OTel init, callback registration)
- Missing: startup summary log showing all enabled features
- Missing: JSON log format for production

**Health check story (3.5/5):**
- `/_health/ready` returns 200 (even for degraded state, documented as intentional)
- `/_health/live` exists
- Health endpoints don't require auth
- Missing: detailed health response showing subsystem status (OTel connected, Redis connected, etc.)

### Plugin Developer DX

**Documentation (4/5):**
- `docs/plugins.md` is comprehensive (577 lines)
- Covers: quick start, capabilities, dependencies, priority, failure modes, context, SSRF prevention
- Evaluator plugin contract is well-documented with code examples
- Missing: a working example plugin in a `examples/` directory

**Plugin template (3/5):**
- No `cookiecutter` or `copier` template
- No `scripts/create_plugin.sh` scaffolding script
- The built-in plugins (evaluator, cost_tracker) serve as reference implementations
- `PluginMetadata` dataclass is self-documenting

**Isolated testing (3.5/5):**
- `create_standalone_app()` exists for testing without LiteLLM
- `reset_*()` functions documented for singleton cleanup in tests
- `conftest.py` provides shared fixtures
- Missing: a `PluginTestHarness` utility class that mocks app, context, and provides assertions

**Error messages during development (3/5):**
- Plugin load failures show the full module path
- Circular dependency detection raises `PluginDependencyError`
- Allowlist/capability violations are logged
- Missing: "Did you mean...?" suggestions for common typos

### API Consumer DX

**API documentation (2/5):**
- `docs/api-reference.md` exists (need to verify contents)
- Routes defined in `routes.py` are not annotated with OpenAPI metadata beyond basic FastAPI
- No interactive API documentation (Swagger UI is available via FastAPI but route descriptions are minimal)
- Missing: request/response examples in route definitions
- Missing: error code reference

**Error responses (2.5/5):**
- Standard HTTP status codes used
- LiteLLM provides structured error responses for LLM endpoints
- Missing: consistent error response schema across all custom endpoints
- Missing: error codes (machine-readable) in addition to messages

**OpenAPI spec (2.5/5):**
- FastAPI auto-generates OpenAPI spec at `/docs` and `/openapi.json`
- Route descriptions are minimal
- No API versioning strategy documented
- Missing: request/response schema examples

**Client SDK examples (1.5/5):**
- No client SDK
- No curl example collection
- `scripts/` has validation scripts but they are for internal testing
- Missing: `examples/` directory with Python, Node, curl examples

---

## Part 7: Documentation Gap Analysis

### Existing Documentation

| Document | Location | Quality | Coverage |
|----------|----------|---------|----------|
| `README.md` | Root | Good | Project overview, features, architecture diagram, quickstart |
| `CLAUDE.md` | Root | Excellent | Comprehensive development guide, architecture, patterns, gotchas |
| `docs/observability.md` | docs/ | Good | 589 lines covering Jaeger, Tempo, CloudWatch, sampling, multiprocess |
| `docs/plugins.md` | docs/ | Good | 577 lines covering full plugin lifecycle |
| `docs/configuration.md` | docs/ | Exists | Need to verify coverage |
| `docs/api-reference.md` | docs/ | Exists | Need to verify coverage |

### Missing Documentation

| Gap | Priority | Impact |
|-----|----------|--------|
| **Quickstart for operators** -- single page going from zero to running gateway with real LLM calls | P0 | Operators cannot deploy without tribal knowledge |
| **Configuration reference** -- complete list of all env vars with defaults, types, and examples | P0 | 60+ env vars across multiple prefixes with no single reference |
| **Metrics reference** -- table of all emitted metrics with labels, types, and PromQL examples | P1 | Operators cannot build dashboards without reading source code |
| **Grafana dashboard JSON** -- pre-built dashboard template | P1 | Every operator rebuilds the same dashboard |
| **Alert runbook** -- for each alertable condition, what to check and how to fix | P1 | On-call engineers have no guidance |
| **API consumer guide** -- getting started with RouteIQ as an API consumer, with curl examples | P1 | API consumers have no onboarding path |
| **Architecture decision records (ADRs)** -- why key decisions were made | P2 | New contributors lack context |
| **Changelog** -- what changed between versions | P2 | Operators cannot assess upgrade risk |
| **Troubleshooting guide** -- common errors and fixes | P2 | Support burden on maintainers |
| **Plugin developer tutorial** -- step-by-step tutorial building a real plugin | P2 | Plugin docs exist but no walkthrough |

---

## Part 8: Competitor DX Comparison

### Feature Comparison Matrix

| Feature | RouteIQ | Helicone | Portkey | LangSmith |
|---------|---------|----------|---------|-----------|
| **Setup complexity** | Config YAML + env vars + Docker | 1-line SDK integration | 1-line SDK integration | SDK + API key |
| **Time to first trace** | ~15 min (need OTel collector) | Instant (SaaS) | Instant (SaaS) | ~5 min (SDK) |
| **Dashboard** | BYO Grafana | Built-in web UI | Built-in web UI | Built-in web UI |
| **Cost tracking** | Per-model/team metrics | Per-model/user dashboard | Per-model/user/team dashboard | Per-run cost |
| **Token analytics** | Histogram metrics | Real-time dashboard | Real-time dashboard | Per-trace |
| **Prompt management** | Not included | Built-in versioning | Via configs | Built-in playground |
| **Cache analytics** | Not instrumented | Built-in | Built-in | N/A |
| **User feedback** | Not included | Not included | Built-in | Built-in |
| **Evaluation** | Plugin system (custom) | N/A | N/A | Built-in datasets + evaluators |
| **Self-hosted** | Yes (primary mode) | Yes (open-source) | Enterprise only | No |
| **OTel native** | Yes (first-class) | Added (via provider) | Partial | Partial |
| **Agent tracing** | A2A + MCP spans | N/A | MCP support | LangGraph native |

### Key Takeaways

1. **Helicone's DX advantage**: Zero-config observability. Their proxy approach means `s/api.openai.com/oai.helicone.ai/` and you get full observability. RouteIQ requires OTel collector setup.

2. **Portkey's dashboard advantage**: Pre-built dashboards for cost, latency, cache, and token analytics. Gartner Cool Vendor in LLM Observability 2025. RouteIQ emits the metrics but provides no visualization.

3. **LangSmith's evaluation advantage**: Built-in dataset management, evaluation framework, and prompt playground. RouteIQ's evaluator plugin system is extensible but requires building your own.

4. **RouteIQ's unique strengths**:
   - Self-hosted, OTel-native (no vendor lock-in)
   - ML-based routing intelligence (no competitor has this)
   - Plugin system for custom evaluation
   - Versioned telemetry contracts for MLOps
   - A2A + MCP protocol support with full tracing

---

## Part 9: CLI & Tooling Review

### `startup.py` CLI

- Minimal CLI: `--config`, `--port`, `--host`
- No `--validate` flag for config checking
- No `--version` flag
- No `--list-plugins` flag
- No `--show-config` flag to dump resolved configuration
- Uses `argparse` -- fine for current scope

### Scripts (`scripts/`)

- 21 scripts covering: testing, validation, security scanning, stub servers
- Good coverage for development workflows
- Missing: `scripts/quickstart.sh` for first-time setup
- Missing: `scripts/doctor.sh` for environment health check
- `install_lefthook.sh` for git hooks -- good

### CI/CD (`.github/workflows/ci.yml`)

- Exists and includes: lint, test, build
- Lefthook pre-commit hooks: ruff, yamllint, secret detection (parallel)
- Lefthook pre-push hooks: unit tests, mypy, security scanning (sequential)
- Good CI hygiene

### Missing Tooling

| Tool | Purpose | Priority |
|------|---------|----------|
| `routeiq doctor` | Check environment health (Python version, deps, OTel reachability, config validity) | P1 |
| `routeiq validate-config` | Validate YAML config without starting server | P1 |
| `routeiq show-metrics` | List all metrics with current values (debug endpoint) | P2 |
| `routeiq export-openapi` | Export OpenAPI spec to file | P2 |
| `routeiq create-plugin` | Scaffold a new plugin | P3 |

---

## Part 10: OTel Collector Config Review

### `config/otel-collector-config.yaml`

The ADOT collector config exports to:
- **Traces**: AWS X-Ray
- **Metrics**: AWS CloudWatch EMF (namespace: `RouteIQ/Gateway`)
- **Logs**: AWS CloudWatch Logs

**Good:**
- Separate batch processors for each signal (traces: 1s, metrics: 60s, logs: 5s)
- Resource attributes for service namespace and environment
- Environment variable substitution (`${AWS_REGION:-us-east-1}`)

**Concerns:**
- `NoDimensionRollup` on EMF exporter means no automatic aggregation across dimensions. This is correct for avoiding unexpected cost but should be documented.
- No `memory_limiter` processor -- in high-traffic scenarios, the collector could OOM.
- No health check extension configured.
- No retry/queue configuration for exporters.
- Hardcoded to AWS backends -- no alternative config for Prometheus/Grafana stack.

---

## Part 11: Priority Recommendations for v0.0.2

### Critical (P0) -- Ship-blocking

1. **Wire unrecorded metrics**: `gen_ai.server.time_to_first_token`, `gateway.routing.decision.duration`, `gateway.routing.strategy.usage`, `gateway.circuit_breaker.transitions` are all defined but never recorded. Either record them or remove the dead code.

2. **Fix `user` label cardinality on `llm_cost_total`**: Remove `user` from metric labels. Track per-user cost in spans/logs only.

3. **Add startup configuration banner**: Log a summary at startup showing: enabled features, loaded plugins, OTel endpoint, sampling rate, health endpoints.

### High (P1) -- First sprint after release

4. **Provide Grafana dashboard JSON**: Pre-built dashboard covering RED metrics, cost, token usage, routing decisions. Export as JSON and include in `docs/dashboards/`.

5. **Create metrics reference doc**: Table of all metrics, labels, types, and example PromQL/TraceQL queries.

6. **Add structured JSON log formatter**: Configurable via `LOG_FORMAT=json` env var.

7. **Add `--validate-config` CLI flag**: Validate config YAML without starting the server.

8. **Create quickstart guide**: Single document from zero to first traced LLM call.

### Medium (P2) -- v0.0.3 planning

9. **Add cache operation spans and metrics**: When semantic cache is used, record `cache.hit`, `cache.miss` spans and a `gateway.cache.hit_ratio` gauge.

10. **Add policy engine span**: Instrument `PolicyEngine.evaluate()` with a span for observability into policy evaluation latency and outcomes.

11. **Add MCP/A2A metric recording**: Currently these are span-only. Add `gateway.mcp.tool_call.total`, `gateway.a2a.agent_call.total` counters and duration histograms.

12. **Normalize `error_type` label**: Map exception classes to categories (auth, rate_limit, timeout, server_error, client_error, unknown) to bound cardinality.

13. **Provide alert rule templates**: Prometheus/CloudWatch alerting rules for common scenarios.

14. **Consolidate env var prefixes**: Document the three prefix generations (`LITELLM_*`, `LLMROUTER_*`, `ROUTEIQ_*`) and migration path.

### Low (P3) -- Backlog

15. **Add `routeiq doctor` CLI command**: Environment health checker.
16. **Create plugin scaffolding tool**: `routeiq create-plugin` command.
17. **Emit `RouterDecisionEvent` contract**: The rich telemetry contract exists as code but is never emitted at runtime.
18. **Add request-scoped log enrichment**: Middleware that injects `request_id`, `model`, `team_id` into all log records within a request.
19. **Provide OTel collector configs for non-AWS stacks**: Prometheus/Grafana, Datadog, Honeycomb.
20. **Create API consumer onboarding guide with curl examples**.

---

## Appendix A: Span Attribute Reference

### Router Decision Attributes (TG4.1)

| Attribute | Type | Source |
|-----------|------|--------|
| `router.strategy` | string | `RouterDecisionMiddleware` / `RouterDecisionCallback` |
| `router.model_selected` | string | Both |
| `router.score` | float | Callback only |
| `router.candidates_evaluated` | int | Both |
| `router.decision_outcome` | string | Both |
| `router.decision_reason` | string | Both |
| `router.latency_ms` | float | Both |
| `router.error_type` | string | Callback only |
| `router.error_message` | string | Callback only |
| `router.strategy_version` | string | Both |
| `router.fallback_triggered` | bool | Both |

### GenAI Semantic Convention Attributes

| Attribute | Type | Source |
|-----------|------|--------|
| `gen_ai.request.model` | string | Middleware + Callback |
| `gen_ai.operation.name` | string | Middleware |
| `gen_ai.system` | string | Callback |
| `gen_ai.usage.input_tokens` | int | Callback (success) |
| `gen_ai.usage.output_tokens` | int | Callback (success) |
| `gen_ai.response.model` | string | Callback (success) |

### Cost Tracking Attributes

| Attribute | Type | Source |
|-----------|------|--------|
| `llm.cost.input_tokens` | int | CostTrackerPlugin |
| `llm.cost.output_tokens` | int | CostTrackerPlugin |
| `llm.cost.total_tokens` | int | CostTrackerPlugin |
| `llm.cost.input_cost_usd` | float | CostTrackerPlugin |
| `llm.cost.output_cost_usd` | float | CostTrackerPlugin |
| `llm.cost.total_cost_usd` | float | CostTrackerPlugin |
| `llm.cost.model` | string | CostTrackerPlugin |
| `llm.cost.estimated_cost_usd` | float | CostTrackerPlugin |
| `llm.cost.estimation_error_pct` | float | CostTrackerPlugin |

### MCP Tracing Attributes

| Attribute | Type | Source |
|-----------|------|--------|
| `mcp.server.id` | string | `mcp_tracing.py` |
| `mcp.server.name` | string | `mcp_tracing.py` |
| `mcp.tool.name` | string | `mcp_tracing.py` |
| `mcp.transport` | string | `mcp_tracing.py` |
| `mcp.success` | bool | `mcp_tracing.py` |
| `mcp.error` | string | `mcp_tracing.py` |
| `mcp.duration_ms` | float | `mcp_tracing.py` |
| `mcp.invocation.disabled` | bool | `mcp_tracing.py` |

### A2A Tracing Attributes

| Attribute | Type | Source |
|-----------|------|--------|
| `a2a.agent.id` | string | `a2a_tracing.py` |
| `a2a.agent.name` | string | `a2a_tracing.py` |
| `a2a.agent.url` | string | `a2a_tracing.py` |
| `a2a.method` | string | `a2a_tracing.py` |
| `a2a.message.id` | string | `a2a_tracing.py` |
| `a2a.stream` | bool | `a2a_tracing.py` |
| `a2a.success` | bool | `a2a_tracing.py` |
| `a2a.error` | string | `a2a_tracing.py` |
| `a2a.duration_ms` | float | `a2a_tracing.py` |

### Evaluator Attributes

| Attribute | Type | Source |
|-----------|------|--------|
| `eval.plugin` | string | evaluator plugin |
| `eval.score` | float | evaluator plugin |
| `eval.status` | string | evaluator plugin |
| `eval.duration_ms` | float | evaluator plugin |
| `eval.error` | string | evaluator plugin |
| `eval.invocation_type` | string | evaluator plugin |

---

## Appendix B: Duplicate Metric Concern

The `GatewayMetrics` registry and `CostTrackerPlugin` define overlapping metrics:

| GatewayMetrics | CostTrackerPlugin | Overlap? |
|----------------|-------------------|----------|
| `gateway.request.active` | `llm_active_requests` | **Yes** -- both track active requests |
| `gateway.cost.total` | `llm_cost_total` | **Yes** -- both track cost |
| `gen_ai.client.token.usage` | `llm_tokens_total` | **Partial** -- different metric types (histogram vs counter) |

Recommendation: Consolidate. Either have `CostTrackerPlugin` record to `GatewayMetrics` instruments, or remove the overlapping instruments from `GatewayMetrics`. Having two active request gauges both decrementing for the same request will produce confusing dashboards.
