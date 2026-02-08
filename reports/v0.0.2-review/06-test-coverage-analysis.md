# RouteIQ v0.0.2 Test Coverage Analysis

**Date:** 2026-02-07
**Scope:** All tests in `tests/unit/`, `tests/property/`, `tests/integration/`, `tests/perf/`, and root-level `tests/test_*.py`

---

## 1. Test Inventory Summary

### Overall Numbers

| Category | Tests Collected | Pass | Skip | Fail |
|----------|----------------|------|------|------|
| Unit tests (`tests/unit/`) | 1,545 | 1,535 | 10 | 0 |
| Property-based (`tests/property/`) | 237 | -- | -- | -- |
| Integration (`tests/integration/`) | 137 | -- | -- | -- |
| Root-level (`tests/test_*.py`) | 235 | -- | -- | -- |
| **Total** | **2,154** | -- | -- | -- |

*Note: Property, integration, and root-level tests were collected but not executed in this analysis. Integration tests require the Docker stack to be running.*

### Unit Test Inventory by File

| Test File | Tests | What Is Tested |
|-----------|-------|----------------|
| `test_observability.py` | 69 | OTel init, tracer/meter/logger creation, provider reuse, metric recording, span attributes, shutdown |
| `test_strategy_registry.py` | 55 | Strategy registration, A/B testing, routing pipeline, hot-swap, concurrent routing |
| `test_metrics.py` | 55 | Prometheus metrics recording, counter increments, histogram observations, metric labels |
| `test_policy_responses_api.py` | 52 | Policy evaluation API responses, status codes, error handling, JSON schema validation |
| `test_guardrails.py` | 51 | GuardrailBlockError, prompt injection detection, PII detection/redaction, action modes, edge cases |
| `test_cost_aware_routing.py` | 49 | Cost-aware model selection, budget constraints, cost estimation, fallback behavior |
| `test_auth.py` | 48 | Admin auth, user auth, RequestID middleware, secret scrubbing, header validation |
| `test_a2a_compliance.py` | 45 | A2A protocol compliance, task states, agent cards, JSON-RPC message format |
| `test_model_artifacts.py` | 41 | ML model hash verification, signature checks, artifact download, integrity validation |
| `test_semantic_cache.py` | 38 | Cache key generation, TTL expiry, LRU eviction, Redis store, cache plugin, cacheability rules |
| `test_mcp_sse_transport.py` | 34 | SSE transport, event formatting, connection lifecycle, heartbeats, reconnection |
| `test_telemetry_contracts.py` | 33 | Telemetry event schemas, contract versioning, serialization round-trips |
| `test_ssrf_async_dns.py` | 33 | Async DNS SSRF validation, rollback flag, DNS timeout, non-blocking behavior, allowlist integration |
| `test_tg4_1_router_decision_telemetry.py` | 32 | Router decision callback, telemetry emission, span attributes |
| `test_mcp_compliance.py` | 32 | MCP protocol compliance, JSON-RPC format, tool discovery, SSE events |
| `test_rbac.py` | 30 | Role-based access control, permission evaluation, role inheritance |
| `test_hot_reload.py` | 29 | Filesystem watch, config reload, debouncing, error handling |
| `test_ha_config_sync.py` | 28 | HA config sync, S3 ETag-based change detection, leader-only sync |
| `test_plugin_manager.py` | 28 | Plugin lifecycle, dependency resolution, load ordering, validation, metadata |
| `test_quota.py` | 25 | Per-team/per-key quota enforcement, window-based limits, quota reset |
| `test_gateway_concurrency.py` | 24 | Concurrent request handling, middleware ordering, race conditions |
| `test_streaming_correctness.py` | 20 | Raw vs buffered streaming, no-line-buffering, byte integrity, cancellation, backpressure |
| `test_http_client_pool.py` | 20 | Client pooling lifecycle, singleton behavior, fallback clients, instantiation tracking |
| `test_skills_discovery.py` | 20 | Skills plugin discovery, registration, metadata, capability reporting |
| `test_cost_tracker.py` | 19 | Cost tracking per model/team, aggregation, reset, persistence |
| `test_policy_engine.py` | 18 | Policy evaluation, fail-open/fail-closed modes, OPA-style rules |
| `test_leader_election.py` | 18 | Leader election, Redis-based locking, renewal, concurrent competition (7 skipped) |
| `test_audit.py` | 18 | Audit log recording, event structure, log rotation, secret redaction |
| `test_a2a_tracing.py` | 17 | A2A OTel tracing, span creation, attribute recording |
| `test_content_filter.py` | 16 | Content filtering, keyword blocking, regex patterns, allow/deny lists |
| `test_mcp_jsonrpc.py` | 16 | JSON-RPC 2.0 parsing, method dispatch, error codes, batch requests |
| `test_inference_knn_router.py` | 16 | KNN routing strategy, model selection, distance calculations |
| `test_a2a_streaming_passthrough.py` | 16 | A2A streaming passthrough, event forwarding, connection management |
| `test_mcp_tracing.py` | 15 | MCP OTel tracing, span attributes, tool invocation tracing |
| `test_conversation_affinity.py` | 15 | Conversation affinity routing, session pinning, TTL, eviction |
| `test_routing_strategy_patch.py` | 13 | LiteLLM Router monkey-patch, strategy injection, idempotent application |
| `test_circuit_breakers.py` | 13 | Circuit breaker states, failure thresholds, recovery, concurrent access |
| `test_evaluator_plugin.py` | 12 | Evaluator plugin lifecycle, scoring, result reporting |
| `test_plugin_middleware.py` | 10 | ASGI plugin middleware, request/response interception, hook dispatch |
| `test_resilience.py` | 9 | Backpressure middleware, drain manager, readiness probes, singleton reset |
| `test_config_loader.py` | 8 | YAML config loading, S3/GCS download, environment variable substitution |
| `test_gateway_app.py` | 7 | App factory, middleware registration, plugin loading, patch safety |
| `test_plugin_callback_bridge.py` | 5 | LiteLLM callback-to-plugin bridge, error isolation, kwargs merging |

**Note:** Three files (`test_streaming_correctness.py`, `test_ssrf_async_dns.py`, `test_http_client_pool.py`) appeared as 0 in bulk collection due to async coroutine test format, but actually contain 20, 33, and 20 tests respectively that pass when run individually.

### Skipped Tests (10 total)

| Test | Reason |
|------|--------|
| 7 tests in `test_leader_election.py` | Require aiosqlite mock database |
| 3 tests in `test_mcp_sse_transport.py` | SSE endpoint testing requires specific transport setup |

---

## 2. Source-to-Test Coverage Matrix

Coverage tool output: **65% overall line coverage** (10,601 statements, 3,733 missed)

| Source Module | Stmts | Miss | Cover | Test File(s) | Assessment |
|---------------|-------|------|-------|--------------|------------|
| `__init__.py` | 13 | 0 | 100% | (implicit) | Complete |
| `a2a_gateway.py` | 448 | 122 | 73% | `test_a2a_compliance.py`, `test_a2a_streaming_passthrough.py` | Good; agent management and some error paths uncovered |
| `a2a_tracing.py` | 308 | 106 | 66% | `test_a2a_tracing.py` | Moderate; large chunks of span processing uncovered (lines 456-570) |
| `audit.py` | 196 | 80 | 59% | `test_audit.py` | Gaps in log rotation, async writing, error recovery |
| `auth.py` | 100 | 7 | 93% | `test_auth.py` | Strong coverage |
| `config_loader.py` | 62 | 5 | 92% | `test_config_loader.py` | Strong coverage |
| `config_sync.py` | 167 | 99 | 41% | `test_ha_config_sync.py` | **LOW**: S3 sync loop, ETag handling, error recovery largely uncovered |
| `conversation_affinity.py` | 156 | 17 | 89% | `test_conversation_affinity.py` | Good; minor gaps in edge cases |
| `database.py` | 345 | 345 | **0%** | **NONE** | **ZERO COVERAGE** - no tests exist |
| `gateway/app.py` | 179 | 66 | 63% | `test_gateway_app.py` | Moderate; lifespan hooks, route registration partially uncovered |
| `gateway/plugin_callback_bridge.py` | 84 | 9 | 89% | `test_plugin_callback_bridge.py` | Good; minor async dispatch edges uncovered |
| `gateway/plugin_manager.py` | 370 | 67 | 82% | `test_plugin_manager.py` | Good; some error paths in dependency resolution uncovered |
| `gateway/plugin_middleware.py` | 120 | 1 | 99% | `test_plugin_middleware.py` | Excellent |
| `gateway/plugins/cache_plugin.py` | 181 | 58 | 68% | `test_semantic_cache.py` | Moderate; cache integration with Redis partially uncovered |
| `gateway/plugins/content_filter.py` | 140 | 2 | 99% | `test_content_filter.py` | Excellent |
| `gateway/plugins/cost_tracker.py` | 139 | 7 | 95% | `test_cost_tracker.py` | Strong |
| `gateway/plugins/evaluator.py` | 156 | 16 | 90% | `test_evaluator_plugin.py` | Good |
| `gateway/plugins/guardrails_base.py` | 97 | 22 | 77% | `test_guardrails.py` | Moderate; some base class methods uncovered |
| `gateway/plugins/pii_guard.py` | 88 | 15 | 83% | `test_guardrails.py` | Good |
| `gateway/plugins/prompt_injection_guard.py` | 40 | 5 | 88% | `test_guardrails.py` | Good |
| `gateway/plugins/skills_discovery.py` | 189 | 20 | 89% | `test_skills_discovery.py` | Good |
| `gateway/plugins/upskill_evaluator.py` | 120 | 47 | 61% | (indirect via plugin_manager) | **LOW**: Major uncovered sections |
| `hot_reload.py` | 186 | 67 | 64% | `test_hot_reload.py` | Moderate; filesystem watcher setup/teardown largely uncovered |
| `http_client_pool.py` | 86 | 0 | 100% | `test_http_client_pool.py` | Complete |
| `leader_election.py` | 226 | 104 | 54% | `test_leader_election.py` | **LOW**: 7 of 18 tests skipped; DB interaction uncovered |
| `mcp_gateway.py` | 428 | 224 | 48% | `test_mcp_compliance.py` | **LOW**: Tool invocation, registry management, proxy largely uncovered |
| `mcp_jsonrpc.py` | 179 | 24 | 87% | `test_mcp_jsonrpc.py` | Good |
| `mcp_parity.py` | 336 | 232 | 31% | `test_mcp_parity.py` (root) | **VERY LOW**: Most parity/proxy endpoints uncovered |
| `mcp_sse_transport.py` | 360 | 154 | 57% | `test_mcp_sse_transport.py` | **LOW**: SSE connection lifecycle, reconnection uncovered |
| `mcp_tracing.py` | 201 | 201 | **0%** | `test_mcp_tracing.py` (exists but 0% coverage) | **ZERO COVERAGE** - tests exist but don't import the actual module |
| `metrics.py` | 31 | 0 | 100% | `test_metrics.py` | Complete |
| `model_artifacts.py` | 540 | 95 | 82% | `test_model_artifacts.py` | Good |
| `observability.py` | 291 | 65 | 78% | `test_observability.py` | Good; shutdown and some factory methods uncovered |
| `policy_engine.py` | 349 | 83 | 76% | `test_policy_engine.py`, `test_policy_responses_api.py` | Moderate; OPA evaluation engine partially uncovered |
| `quota.py` | 359 | 69 | 81% | `test_quota.py` | Good |
| `rbac.py` | 96 | 3 | 97% | `test_rbac.py` | Excellent |
| `resilience.py` | 413 | 37 | 91% | `test_resilience.py`, `test_circuit_breakers.py` | Strong |
| `router_decision_callback.py` | 216 | 216 | **0%** | `test_tg4_1_router_decision_telemetry.py` (exists but 0% coverage) | **ZERO COVERAGE** - tests mock rather than exercise actual module |
| `routes.py` | 458 | 337 | 26% | `test_gateway_app.py` (partial) | **VERY LOW**: Vast majority of routes uncovered at unit level |
| `routing_strategy_patch.py` | 175 | 106 | 39% | `test_routing_strategy_patch.py` | **LOW**: Monkey-patch application and strategy delegation uncovered |
| `semantic_cache.py` | 219 | 32 | 85% | `test_semantic_cache.py` | Good |
| `startup.py` | 177 | 177 | **0%** | **NONE** | **ZERO COVERAGE** - CLI entry point, no tests |
| `strategies.py` | 428 | 148 | 65% | `test_inference_knn_router.py` | Moderate; many strategy implementations uncovered |
| `strategy_registry.py` | 572 | 150 | 74% | `test_strategy_registry.py` | Good; some A/B testing paths uncovered |
| `telemetry_contracts.py` | 216 | 4 | 98% | `test_telemetry_contracts.py` | Excellent |
| `url_security.py` | 354 | 89 | 75% | `test_ssrf_async_dns.py`, `test_ssrf_hardening.py` (root) | Moderate |

---

## 3. Coverage Gap Analysis by Module

### Critical Gaps (0% Coverage)

#### `database.py` (345 lines, 0%)
- **No test file exists at all**
- Contains database connection management, migration logic, and query helpers
- Risk: Database connection leaks, migration failures, SQL injection in dynamic queries
- **Priority: HIGH** -- if this module is actively used

#### `startup.py` (177 lines, 0%)
- **No test file exists**
- CLI entry point, initialization orchestration
- Hard to unit test (runs uvicorn), but startup sequence logic could be tested
- **Priority: MEDIUM** -- integration tests partially cover this

#### `router_decision_callback.py` (216 lines, 0%)
- Test file `test_tg4_1_router_decision_telemetry.py` exists with 32 tests, but they mock rather than exercise the actual callback module
- The telemetry emission code is completely untested
- **Priority: HIGH** -- routing decisions are core functionality

#### `mcp_tracing.py` (201 lines, 0%)
- Test file `test_mcp_tracing.py` exists with 15 tests, but coverage is 0%
- Tests likely mock the tracing module rather than importing/calling it
- **Priority: MEDIUM** -- observability code, not critical path

### Severe Gaps (< 40% Coverage)

#### `routes.py` (458 lines, 26% coverage)
- 337 of 458 lines uncovered
- Most route handlers untested at unit level
- Root-level `test_http_routes.py` and integration tests likely cover some
- **Priority: HIGH** -- routes are the primary API surface

#### `mcp_parity.py` (336 lines, 31% coverage)
- 232 lines uncovered
- OAuth endpoints, protocol proxy, alias routing mostly untested
- Root-level `test_mcp_parity.py` provides some coverage
- **Priority: MEDIUM**

#### `routing_strategy_patch.py` (175 lines, 39% coverage)
- Monkey-patch application and strategy delegation uncovered
- Root-level `test_strategies.py` provides some coverage
- **Priority: HIGH** -- this is a critical runtime patch

#### `config_sync.py` (167 lines, 41% coverage)
- S3 sync loop, ETag change detection, error recovery uncovered
- **Priority: MEDIUM** -- only active in HA mode

### Moderate Gaps (40-70% Coverage)

| Module | Coverage | Key Gaps |
|--------|----------|----------|
| `mcp_gateway.py` | 48% | Tool invocation, proxy, registry CRUD |
| `leader_election.py` | 54% | DB interaction, concurrent competition |
| `mcp_sse_transport.py` | 57% | SSE lifecycle, reconnection, keepalive |
| `audit.py` | 59% | Async log writing, rotation, error recovery |
| `upskill_evaluator.py` | 61% | Evaluation pipeline, scoring logic |
| `gateway/app.py` | 63% | Lifespan hooks, route registration |
| `hot_reload.py` | 64% | Filesystem watcher, debounce, teardown |
| `strategies.py` | 65% | 12+ strategy implementations |
| `a2a_tracing.py` | 66% | Span processing, attribute extraction |
| `cache_plugin.py` | 68% | Redis integration, embedding generation |

---

## 4. Missing Test Categories

### 4.1 Integration Tests for New Components (Priority: HIGH)

The following new v0.0.2 components lack integration tests:

- **Semantic cache end-to-end** -- cache_plugin -> semantic_cache -> Redis/InMemory
- **Guardrails pipeline** -- content_filter + pii_guard + prompt_injection in sequence
- **Cost tracker with routing** -- cost_aware_routing -> cost_tracker -> model selection
- **Plugin callback bridge in live request** -- LiteLLM request -> callback bridge -> plugin hooks
- **Plugin middleware with real ASGI app** -- request flow through plugin_middleware

### 4.2 Property-Based Tests (Priority: MEDIUM)

Existing property tests cover 7 modules. Missing candidates:

- **Semantic cache key generation** -- Verify keys are deterministic for equivalent inputs (hypothesis strategies for message variations)
- **PII detection** -- Fuzz with generated strings containing SSN/CC/phone patterns
- **Policy engine rule evaluation** -- Random policy combinations with random requests
- **SSRF URL validation** -- Fuzzed URLs with various schemes, IPs, encodings
- **Cost estimation** -- Random token counts and model pricing

### 4.3 Load/Stress Tests (Priority: LOW)

- No benchmarks for semantic cache under concurrent load
- No stress tests for plugin middleware with many plugins
- `tests/perf/` exists but only contains `streaming_perf_harness.py` (a harness, not automated tests)
- Circuit breaker behavior under sustained failure load is untested

### 4.4 Negative Security Tests (Priority: HIGH)

- **Guardrail bypass attempts** -- Unicode normalization attacks, homoglyph injection, encoding tricks
- **PII detection evasion** -- Obfuscated SSN/CC (spaces, dashes, mixed formats)
- **Prompt injection sophistication** -- Multi-turn injection, indirect injection, jailbreak patterns
- **SSRF with DNS rebinding** -- Already tested async, but no tests for time-of-check/time-of-use races
- **Auth token reuse** -- Revoked keys, expired tokens, timing attacks
- **Policy engine bypass** -- Malformed policies, recursive rules, resource exhaustion

### 4.5 Regression Tests (Priority: MEDIUM)

No explicit regression test suite exists. Recommended:

- **Streaming line-buffering regression** -- `test_streaming_correctness.py` covers this well
- **Singleton leak regression** -- Tests exist but could be more systematic
- **Plugin loading order regression** -- Ensure deterministic ordering after config changes
- **Hot reload race condition regression** -- Filesystem watch + config reload timing

---

## 5. Test Quality Assessment

### 5.1 Mock Correctness

**Generally Good:**
- `test_semantic_cache.py` uses realistic mock Redis with proper async patterns
- `test_guardrails.py` tests actual detection logic, not just mocked results
- `test_auth.py` tests actual middleware behavior with FastAPI TestClient

**Concerns:**
- `test_tg4_1_router_decision_telemetry.py` and `test_mcp_tracing.py` over-mock to the point of 0% actual coverage of their target modules
- `test_streaming_correctness.py` uses `importlib.reload()` extensively to toggle environment variables, which is fragile and can cause state leaks between test modules
- Some tests mock internal implementation details (private methods, module-level variables) rather than testing through public interfaces

### 5.2 Assertion Quality

**Strong:**
- `test_guardrails.py` checks both positive and negative cases with specific error messages
- `test_semantic_cache.py` verifies cache behavior (TTL, LRU, key determinism) at a behavioral level
- `test_plugin_callback_bridge.py` verifies error isolation (one plugin failure doesn't block others)

**Weak:**
- Some tests only assert `is not None` or `assert len(x) > 0` without checking specific values
- `test_gateway_app.py` has only 7 tests for a 179-line composition root -- too few assertions for the app factory

### 5.3 Fixture Design

**Well-Designed:**
- `conftest.py` provides shared OTel exporter with proper cleanup (clear before/after)
- Most test files use `autouse=True` fixtures for singleton reset (following project convention)
- `_reset_bridge` fixture pattern is consistent across guardrails, callback bridge, and middleware tests

**Concerns:**
- `test_http_client_pool.py` uses `importlib.reload()` in fixtures, which can affect other test modules
- No shared fixture for creating test plugins (each test file creates its own `TrackingPlugin` class)
- `test_leader_election.py` fixtures are overly complex, leading to 7 skipped tests

### 5.4 Test Isolation

**Generally Good:**
- Singleton reset patterns are consistently applied (`autouse=True` fixtures)
- `monkeypatch` is used for environment variables (proper cleanup)

**Concerns:**
- `test_streaming_correctness.py` uses `importlib.reload(gateway_module)` which can pollute module-level state across tests
- `test_http_client_pool.py` uses `os.environ` directly in `TestHttpClientPoolingDisabled` instead of `monkeypatch`, risking state leaks on test failure
- Module-level `_shared_provider` in `conftest.py` means all tracing tests share one TracerProvider -- if any test corrupts it, all subsequent tests are affected

### 5.5 Flakiness Risk

**Low Risk:**
- Most tests are deterministic with mocked I/O
- `asyncio_mode = "auto"` prevents common async fixture issues

**Moderate Risk:**
- `test_streaming_correctness.py::TestIncrementalYields::test_chunks_yielded_incrementally` relies on timing assertions (`gap >= 0.005`) which could flake on slow CI
- `test_ssrf_async_dns.py::TestNonBlockingBehavior::test_event_loop_progress_during_dns_resolution` depends on async scheduling order
- `test_leader_election.py` tests with concurrent asyncio tasks are timing-sensitive

---

## 6. Coverage Tool Output

```
Name                                                              Stmts   Miss  Cover
-----------------------------------------------------------------------------------------------
src/litellm_llmrouter/__init__.py                                    13      0   100%
src/litellm_llmrouter/a2a_gateway.py                                448    122    73%
src/litellm_llmrouter/a2a_tracing.py                                308    106    66%
src/litellm_llmrouter/audit.py                                      196     80    59%
src/litellm_llmrouter/auth.py                                       100      7    93%
src/litellm_llmrouter/config_loader.py                               62      5    92%
src/litellm_llmrouter/config_sync.py                                167     99    41%
src/litellm_llmrouter/conversation_affinity.py                      156     17    89%
src/litellm_llmrouter/database.py                                   345    345     0%
src/litellm_llmrouter/gateway/__init__.py                             3      0   100%
src/litellm_llmrouter/gateway/app.py                                179     66    63%
src/litellm_llmrouter/gateway/plugin_callback_bridge.py              84      9    89%
src/litellm_llmrouter/gateway/plugin_manager.py                     370     67    82%
src/litellm_llmrouter/gateway/plugin_middleware.py                  120      1    99%
src/litellm_llmrouter/gateway/plugins/cache_plugin.py               181     58    68%
src/litellm_llmrouter/gateway/plugins/content_filter.py             140      2    99%
src/litellm_llmrouter/gateway/plugins/cost_tracker.py               139      7    95%
src/litellm_llmrouter/gateway/plugins/evaluator.py                  156     16    90%
src/litellm_llmrouter/gateway/plugins/guardrails_base.py             97     22    77%
src/litellm_llmrouter/gateway/plugins/pii_guard.py                   88     15    83%
src/litellm_llmrouter/gateway/plugins/prompt_injection_guard.py      40      5    88%
src/litellm_llmrouter/gateway/plugins/skills_discovery.py           189     20    89%
src/litellm_llmrouter/gateway/plugins/upskill_evaluator.py          120     47    61%
src/litellm_llmrouter/hot_reload.py                                 186     67    64%
src/litellm_llmrouter/http_client_pool.py                            86      0   100%
src/litellm_llmrouter/leader_election.py                            226    104    54%
src/litellm_llmrouter/mcp_gateway.py                                428    224    48%
src/litellm_llmrouter/mcp_jsonrpc.py                                179     24    87%
src/litellm_llmrouter/mcp_parity.py                                 336    232    31%
src/litellm_llmrouter/mcp_sse_transport.py                          360    154    57%
src/litellm_llmrouter/mcp_tracing.py                                201    201     0%
src/litellm_llmrouter/metrics.py                                     31      0   100%
src/litellm_llmrouter/model_artifacts.py                            540     95    82%
src/litellm_llmrouter/observability.py                              291     65    78%
src/litellm_llmrouter/policy_engine.py                              349     83    76%
src/litellm_llmrouter/quota.py                                      359     69    81%
src/litellm_llmrouter/rbac.py                                        96      3    97%
src/litellm_llmrouter/resilience.py                                 413     37    91%
src/litellm_llmrouter/router_decision_callback.py                   216    216     0%
src/litellm_llmrouter/routes.py                                     458    337    26%
src/litellm_llmrouter/routing_strategy_patch.py                     175    106    39%
src/litellm_llmrouter/semantic_cache.py                             219     32    85%
src/litellm_llmrouter/startup.py                                    177    177     0%
src/litellm_llmrouter/strategies.py                                 428    148    65%
src/litellm_llmrouter/strategy_registry.py                          572    150    74%
src/litellm_llmrouter/telemetry_contracts.py                        216      4    98%
src/litellm_llmrouter/url_security.py                               354     89    75%
-----------------------------------------------------------------------------------------------
TOTAL                                                             10601   3733    65%

1535 passed, 10 skipped in 9.00s
```

---

## 7. Recommended Test Additions (Prioritized)

### P0 -- Critical (Block Release)

1. **Fix `test_tg4_1_router_decision_telemetry.py` to actually exercise `router_decision_callback.py`**
   - Currently 0% coverage despite 32 tests existing
   - Tests mock the module instead of calling it
   - Router decision telemetry is a core v0.0.2 feature (TG4.1)

2. **Fix `test_mcp_tracing.py` to actually exercise `mcp_tracing.py`**
   - 0% coverage despite 15 tests existing
   - Same over-mocking problem

3. **Add unit tests for `database.py`**
   - 345 lines with 0% coverage
   - If module is actively used, this is a critical gap
   - If deprecated/unused, it should be removed from the codebase

4. **Add route-level tests for `routes.py`**
   - 26% coverage at unit level
   - Every public endpoint should have at least one happy-path and one error-path test
   - Use FastAPI TestClient with minimal app fixture

### P1 -- High (Before GA)

5. **Increase `mcp_gateway.py` coverage from 48% to >75%**
   - Add tests for tool invocation, registry CRUD, proxy forwarding
   - Critical for MCP protocol support

6. **Increase `routing_strategy_patch.py` coverage from 39% to >70%**
   - Monkey-patch is the core integration mechanism
   - Test strategy delegation, error handling, idempotent application

7. **Increase `mcp_parity.py` coverage from 31% to >60%**
   - OAuth endpoints, proxy paths, alias routing need tests
   - Security-sensitive (OAuth token handling)

8. **Add security abuse tests for guardrails**
   - Unicode normalization bypass
   - Encoding tricks (URL encoding, HTML entities)
   - Homoglyph attacks on PII detection
   - Multi-turn prompt injection

9. **Fix 7 skipped tests in `test_leader_election.py`**
   - Leader election is critical for HA deployments
   - Tests skip due to missing aiosqlite mock -- provide it

### P2 -- Medium (v0.0.3)

10. **Add integration test for plugin pipeline end-to-end**
    - Request -> plugin_middleware -> callback_bridge -> plugins -> response
    - Verify plugins see correct data at each lifecycle stage

11. **Add property-based tests for semantic cache key generation**
    - Equivalent inputs should produce identical keys
    - Different inputs should produce different keys (collision resistance)

12. **Increase `config_sync.py` coverage from 41% to >70%**
    - S3 ETag-based sync is critical for HA deployments

13. **Increase `leader_election.py` coverage from 54% to >75%**
    - Add tests with mock DB backend

14. **Increase `audit.py` coverage from 59% to >75%**
    - Test async log writing, rotation, concurrent access

15. **Add startup sequence smoke test**
    - Test `startup.py` initialization without running uvicorn
    - Verify patch order, middleware registration, plugin loading

### P3 -- Low (Backlog)

16. **Add load tests for semantic cache under concurrent access**
17. **Add property-based tests for SSRF URL validation (fuzzed URLs)**
18. **Add regression tests for streaming line-buffering (already partially covered)**
19. **Refactor `test_streaming_correctness.py` to avoid `importlib.reload()`**
20. **Add circuit breaker tests under sustained failure load**
21. **Create shared test fixtures for plugin creation (reduce boilerplate across test files)**

---

## 8. Summary

| Metric | Value |
|--------|-------|
| Total unit tests | 1,545 (1,535 pass, 10 skip) |
| Total all tests | 2,154 |
| Unit coverage (lines) | 65% (6,868 / 10,601) |
| Files with 0% coverage | 4 (`database.py`, `startup.py`, `router_decision_callback.py`, `mcp_tracing.py`) |
| Files below 40% coverage | 3 (`routes.py` 26%, `mcp_parity.py` 31%, `routing_strategy_patch.py` 39%) |
| Files above 90% coverage | 10 (auth, config_loader, plugin_middleware, content_filter, cost_tracker, http_client_pool, metrics, rbac, resilience, telemetry_contracts) |
| Test quality | Generally good; over-mocking in 2 tracing test files is the main concern |
| Flakiness risk | Low overall; 3 timing-sensitive tests identified |

The test suite is substantial at 1,545 unit tests for a codebase of 10,601 statements. The main concerns are:
1. Four modules with 0% coverage (two have test files that don't actually exercise the code)
2. Routes module at 26% -- the primary API surface has weak unit coverage
3. MCP-related modules (gateway, parity, SSE) averaging below 50%
4. No integration tests for the new plugin pipeline end-to-end flow
