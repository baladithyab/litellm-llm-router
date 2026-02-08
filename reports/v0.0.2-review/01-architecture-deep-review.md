# RouteIQ v0.0.2 Architecture Deep Review

**Date**: 2026-02-07
**Reviewer**: Architecture Review Agent (Opus 4.6)
**Scope**: All new components from 10 implementation agents + existing architecture

---

## Executive Summary

1. **Duplicate Metric Systems (HIGH)** -- `cost_tracker.py` creates its own OTel metric instruments (`llm_cost_total`, `llm_tokens_total`, etc.) completely bypassing the central `GatewayMetrics` registry in `metrics.py`. This creates two parallel metric pipelines, making dashboarding unreliable and potentially causing OTel instrument name collisions.

2. **Inconsistent Plugin Config Timing (MEDIUM)** -- Guardrail plugins (`PIIGuard`, `PromptInjectionGuard`) read environment variables in `__init__()`, while other plugins (`ContentFilterPlugin`, `SemanticCachePlugin`, `CostTrackerPlugin`) defer to `startup()`. This means guardrail `_enabled` state is frozen at class instantiation and cannot respond to config hot-reload.

3. **Missing `reset_*()` Singletons (MEDIUM)** -- `semantic_cache.py`, `observability.py`, and `router_decision_callback.py` expose module-level mutable state but lack `reset_*()` functions, violating the project's universal singleton-reset testing contract documented in CLAUDE.md.

4. **Content Filter Extends Wrong Base Class (MEDIUM)** -- `ContentFilterPlugin` extends `GatewayPlugin` directly instead of `GuardrailPlugin`, despite being a content security guardrail. It re-implements guardrail logic (score/threshold/block) without the OTel span attributes, `GuardrailDecision` model, or `GuardrailBlockError` propagation that the guardrail base provides.

5. **Module-Level `os.getenv()` Calls Freeze Config at Import (LOW)** -- `conversation_affinity.py` reads `CONVERSATION_AFFINITY_ENABLED`, `_TTL`, and `_MAX_ENTRIES` at module scope. This locks their values at first import and prevents hot-reload from changing them. All other subsystems defer config reads to a function or method.

---

## Detailed Findings

### 1. Architectural Coupling Issues

#### 1.1 Duplicate Metric Registration Systems
**Severity: HIGH**

Two independent metric registration paths exist:

- **Central**: `metrics.py` (`GatewayMetrics`) creates `gateway.cost.total`, `gateway.request.total`, `gateway.routing.decision.duration`, etc. via a single `Meter` instance provided by `ObservabilityManager`.
- **Plugin**: `cost_tracker.py:198-225` creates its own `llm_cost_total`, `llm_tokens_total`, `llm_cost_per_request`, `llm_active_requests`, `llm_cost_errors_total` by calling `get_meter("routeiq.cost_tracker")` directly.

The two systems use different meter names (`routeiq.gateway` vs `routeiq.cost_tracker`) and different metric name prefixes (`gateway.*` vs `llm_*`). Neither references the other.

**Impact**: Dashboards that query `gateway.cost.total` won't see CostTracker data, and vice versa. OTel backends receive two streams of cost data with different schemas.

**Suggested fix**: `CostTrackerPlugin` should use `get_gateway_metrics()` from `metrics.py` or at minimum register instruments through the same `Meter` instance. Define all cost metric names in `metrics.py`.

**Files**: `src/litellm_llmrouter/gateway/plugins/cost_tracker.py:198-225`, `src/litellm_llmrouter/metrics.py:206-211`

#### 1.2 ContentFilterPlugin Bypasses Guardrail Base Class
**Severity: MEDIUM**

`ContentFilterPlugin` at `content_filter.py:181` extends `GatewayPlugin` directly:
```python
class ContentFilterPlugin(GatewayPlugin):
```

Meanwhile, `PromptInjectionGuard` and `PIIGuard` both extend `GuardrailPlugin`, which provides:
- Standardized `GuardrailDecision` emission
- OTel `guardrail.*` span attributes
- `GuardrailBlockError` propagation
- Timing instrumentation

`ContentFilterPlugin` re-implements its own scoring, blocking (via `GuardrailBlockError` import at line 46), and logging. It declares `PluginCapability.GUARDRAIL` but doesn't follow the guardrail contract.

**Impact**: Content filter decisions lack OTel `guardrail.*` span attributes. The `get_guardrail_plugins()` method in `plugin_manager.py:994-1007` will return it but callers expecting `GuardrailPlugin` methods (`evaluate_input`, `evaluate_output`) will fail.

**Suggested fix**: Refactor `ContentFilterPlugin` to extend `GuardrailPlugin` and implement `evaluate_input`/`evaluate_output` instead of `on_llm_pre_call`/`on_llm_success` directly.

**Files**: `src/litellm_llmrouter/gateway/plugins/content_filter.py:181`, `src/litellm_llmrouter/gateway/plugins/guardrails_base.py:119`

---

### 2. Error Handling Patterns

#### 2.1 Silent Fallback in CostTrackerPlugin._calculate_cost
**Severity: MEDIUM**

```python
# cost_tracker.py:121-144
def _calculate_cost(...) -> tuple[float, float, float]:
    try:
        import litellm
        total_cost = litellm.completion_cost(...)
        ...
    except Exception:
        return 0.0, 0.0, 0.0
```

A bare `except Exception` silently returns zero cost. If `litellm.completion_cost` raises due to an unknown model, misconfigured cost map, or API change, the tracker silently reports $0 cost, which is worse than reporting no cost at all -- it creates a false "free" signal.

**Suggested fix**: Log at WARNING level with the exception details before returning zeros. Consider emitting an OTel event for cost estimation failures.

**Files**: `src/litellm_llmrouter/gateway/plugins/cost_tracker.py:143`

#### 2.2 Redis Connection Failure Silently Disables L2 Cache Forever
**Severity: MEDIUM**

In `conversation_affinity.py:196-199`:
```python
except Exception as e:
    logger.warning(...)
    self._redis_available = False
```

Once a single Redis operation fails, `_redis_available` is set to `False` permanently. There is no reconnection logic or periodic retry. The same pattern exists in `cache_plugin.py:160-162` where Redis init failure sets `self._l2 = None` permanently.

**Suggested fix**: Add a reconnection backoff mechanism, or at minimum a periodic health check that re-enables Redis.

**Files**: `src/litellm_llmrouter/conversation_affinity.py:196-200`, `src/litellm_llmrouter/gateway/plugins/cache_plugin.py:160-162`

#### 2.3 PluginCallbackBridge Swallows All Plugin Exceptions
**Severity: LOW**

In `plugin_callback_bridge.py:110-116`, only `GuardrailBlockError` is re-raised. All other exceptions from `on_llm_pre_call` are logged and swallowed. This is correct for fault tolerance but means a misconfigured plugin that always throws will generate log noise without any circuit-breaking.

**Suggested fix**: Consider a per-plugin error counter that quarantines plugins after N consecutive failures, similar to `PluginManager._quarantined`.

**Files**: `src/litellm_llmrouter/gateway/plugin_callback_bridge.py:110-116`

---

### 3. Singleton Management

#### 3.1 Missing reset_*() in semantic_cache.py
**Severity: MEDIUM**

`semantic_cache.py` defines no module-level singleton and no `reset_*()` function. However, `cache_plugin.py:53-85` defines a module-level `_embedder_model` singleton with `_reset_embedder()`. The `InMemoryCache` and `RedisCacheStore` classes are instantiated by the plugin and aren't singletons -- this is fine.

But `cache_plugin.py`'s `_embedder_model` uses a `threading.Lock()` (not `asyncio.Lock`), which is correct since `SentenceTransformer` model loading is CPU-bound. The `_reset_embedder()` function exists, so tests can clean up.

**Status**: `_reset_embedder()` exists in `cache_plugin.py:81-85`. **No issue** for the embedder.

However, **`semantic_cache.py` itself has no singletons or reset functions** -- it's a pure library module. This is architecturally clean.

#### 3.2 Missing reset_*() in observability.py
**Severity: MEDIUM**

`observability.py` manages an `ObservabilityManager` but lacks a `reset_*()` function. The CLAUDE.md explicitly states: "Singletons need `reset_*()` -- every subsystem uses singletons. Tests MUST call `reset_*()` in `autouse=True` fixtures."

If `ObservabilityManager` is a singleton (which it likely is given the pattern), tests that initialize OTel providers will leak state across test cases.

**Suggested fix**: Add `reset_observability_manager()` function.

**Files**: `src/litellm_llmrouter/observability.py`

#### 3.3 Missing reset_*() in router_decision_callback.py
**Severity: MEDIUM**

`router_decision_callback.py` registers a middleware and callback via `register_router_decision_middleware()`, but there's no corresponding `reset_*()` function for tests.

**Suggested fix**: Add `reset_router_decision_callback()`.

**Files**: `src/litellm_llmrouter/router_decision_callback.py`

#### 3.4 PluginMiddleware Self-Registers in __init__
**Severity: LOW**

`plugin_middleware.py:159`:
```python
def __init__(self, app: ASGIApp) -> None:
    ...
    set_plugin_middleware(self)
```

The middleware registers itself as the module singleton during `__init__`. This is documented ("Self-register as the singleton so plugin startup can find us") and works because Starlette creates middleware instances during `app.add_middleware()`. However, if tests create multiple `PluginMiddleware` instances, the last one wins silently.

**Suggested fix**: Document this clearly in test guidelines. The `reset_plugin_middleware()` function exists, which mitigates this.

**Files**: `src/litellm_llmrouter/gateway/plugin_middleware.py:159`

---

### 4. Async Correctness

#### 4.1 Blocking SentenceTransformer Load in Async Context
**Severity: MEDIUM**

`cache_plugin.py:67-77`:
```python
def _get_embedder(model_name: str) -> Any:
    global _embedder_model
    with _embedder_lock:
        if _embedder_model is None:
            _embedder_model = SentenceTransformer(model_name, device="cpu")
    return _embedder_model
```

`SentenceTransformer(model_name, device="cpu")` downloads and loads a neural network model. This can take 5-30 seconds and blocks the event loop since it's called from an async plugin hook (`on_llm_pre_call` / `on_llm_success`) via a synchronous function with a `threading.Lock`.

**Impact**: First cache-enabled request blocks all other requests until the model is loaded.

**Suggested fix**: Load the model during plugin `startup()` using `asyncio.get_event_loop().run_in_executor()`, or use `anyio.to_thread.run_sync()` to load it off the event loop.

**Files**: `src/litellm_llmrouter/gateway/plugins/cache_plugin.py:57-78`

#### 4.2 ConversationAffinityTracker Uses asyncio.Lock Correctly
**Severity: N/A (positive finding)**

`conversation_affinity.py:112` uses `asyncio.Lock()` for the in-memory store, which is correct for async code. The background cleanup task at line 275-284 properly catches `CancelledError`. No issues here.

#### 4.3 PluginManager.startup Uses asyncio.wait_for with Configurable Timeout
**Severity: N/A (positive finding)**

`plugin_manager.py:887-889`:
```python
await asyncio.wait_for(
    plugin.startup(app, self._context),
    timeout=startup_timeout,
)
```

Good pattern -- prevents a rogue plugin from blocking startup indefinitely. Timeout is configurable via `ROUTEIQ_PLUGIN_STARTUP_TIMEOUT`.

---

### 5. Configuration Sprawl

#### 5.1 Inconsistent Config Read Timing
**Severity: MEDIUM**

| Plugin | Config Read Location | Pattern |
|--------|---------------------|---------|
| `PromptInjectionGuard` | `__init__()` | Reads `GUARDRAIL_INJECTION_ENABLED/ACTION` at instantiation |
| `PIIGuard` | `__init__()` | Reads `GUARDRAIL_PII_ENABLED/ACTION/ENTITY_TYPES` at instantiation |
| `ContentFilterPlugin` | `startup()` | Reads `CONTENT_FILTER_ENABLED/THRESHOLD/ACTION/CATEGORIES` during startup |
| `CostTrackerPlugin` | `startup()` | Reads `COST_TRACKER_ENABLED` during startup |
| `SemanticCachePlugin` | `startup()` | Reads all `CACHE_*` vars during startup |
| `ConversationAffinityTracker` | Module scope | Reads `CONVERSATION_AFFINITY_*` at import time |

Three distinct patterns exist. The `__init__()` pattern means the enabled state is frozen before `startup()` runs, and before `PluginContext` is available. The module-scope pattern in `conversation_affinity.py` means values are frozen at first import.

**Suggested fix**: Standardize on reading config in `startup()` for all plugins. This enables hot-reload via `on_config_reload()`.

**Files**: Multiple (see table above)

#### 5.2 No Config Validation for Numeric Env Vars
**Severity: LOW**

Several numeric environment variables are parsed with bare `int()` or `float()`:

- `cache_plugin.py:136`: `int(os.getenv("CACHE_TTL_SECONDS", "3600"))` -- no try/except
- `cache_plugin.py:137`: `int(os.getenv("CACHE_L1_MAX_SIZE", "1000"))` -- no try/except
- `conversation_affinity.py:55-56`: `int(os.getenv(...))` at module scope -- crash on import

If someone sets `CACHE_TTL_SECONDS=abc`, the app crashes with an unhelpful `ValueError`.

**Suggested fix**: Wrap in try/except with a warning and default fallback, or use a centralized config parser.

**Files**: `src/litellm_llmrouter/gateway/plugins/cache_plugin.py:136-144`, `src/litellm_llmrouter/conversation_affinity.py:55-56`

---

### 6. Module Boundaries

#### 6.1 Clean Plugin Architecture
**Severity: N/A (positive finding)**

The plugin system has excellent separation:
- `plugin_manager.py` -- lifecycle and dependency resolution
- `plugin_callback_bridge.py` -- LiteLLM callback integration
- `plugin_middleware.py` -- ASGI request/response hooks
- `gateway/plugins/*` -- individual plugin implementations

Each component has a well-defined responsibility and clean interfaces.

#### 6.2 guardrails_base.py vs content_filter.py Boundary Confusion
**Severity: MEDIUM**

The `guardrails_base.py` module defines the canonical guardrail contract (`GuardrailPlugin`, `GuardrailDecision`, `GuardrailBlockError`). Two of three guardrail plugins follow it (`PIIGuard`, `PromptInjectionGuard`), but `ContentFilterPlugin` goes its own way. This creates an inconsistent module boundary where the "guardrail" concept is split across two class hierarchies.

**Files**: `src/litellm_llmrouter/gateway/plugins/content_filter.py:181`, `src/litellm_llmrouter/gateway/plugins/guardrails_base.py:119`

#### 6.3 gateway/app.py Serves as Clean Composition Root
**Severity: N/A (positive finding)**

`app.py` correctly orchestrates the startup sequence: patch -> middleware -> plugins -> routes -> lifecycle hooks -> backpressure. It imports from sibling modules but doesn't contain business logic. The `_run_plugin_startup()` / `_run_plugin_shutdown()` separation is well-designed.

---

### 7. Import Structure

#### 7.1 Lazy Import of litellm in cost_tracker.py
**Severity: LOW (acceptable)**

`cost_tracker.py:122`:
```python
def _calculate_cost(...):
    try:
        import litellm
        total_cost = litellm.completion_cost(...)
```

`litellm` is imported lazily inside `_calculate_cost()`. This runs on every LLM success callback. The import is cached by Python after the first call, but it's a code smell -- if litellm is unavailable, this silently returns 0.

Contrast with `plugin_callback_bridge.py:213` which also lazily imports litellm but only during registration (once).

**Suggested fix**: Import litellm at module level or during `startup()` and store a reference.

**Files**: `src/litellm_llmrouter/gateway/plugins/cost_tracker.py:122`

#### 7.2 TYPE_CHECKING Guards Used Correctly
**Severity: N/A (positive finding)**

All new modules properly use `TYPE_CHECKING` guards for type-only imports (FastAPI, etc.), preventing circular import issues and unnecessary runtime dependencies. Examples:
- `content_filter.py:48`
- `cost_tracker.py:47`
- `cache_plugin.py:47`
- `plugin_manager.py:71`

#### 7.3 No Circular Import Risks Detected
**Severity: N/A (positive finding)**

The import graph is clean:
- `plugin_manager.py` has zero intra-project imports (except `url_security` in a lazy function)
- `plugins/*` import from `plugin_manager` and `guardrails_base` (parent direction only)
- `plugin_callback_bridge` imports from `guardrails_base` only
- `plugin_middleware` has zero intra-project imports
- `app.py` imports from siblings in the parent package

---

### 8. Code Duplication

#### 8.1 Response Text Extraction Duplicated Across Plugins
**Severity: LOW**

Three separate implementations exist for extracting text from LLM responses:

1. `pii_guard.py:233-247` -- `_extract_response_text(response)` (static method)
2. `content_filter.py:316-335` -- `_extract_response_text(response)` (instance method)
3. `cost_tracker.py:84-108` -- `_extract_usage(response)` (different purpose but similar pattern)

The PII guard and content filter implementations handle the same two response formats (dict and litellm ModelResponse) but with slightly different code paths.

**Suggested fix**: Extract a shared `extract_response_content()` utility into `guardrails_base.py` or a new `gateway/plugins/utils.py`.

**Files**: `src/litellm_llmrouter/gateway/plugins/pii_guard.py:233-247`, `src/litellm_llmrouter/gateway/plugins/content_filter.py:316-335`

#### 8.2 OTel Optional Import Pattern Duplicated
**Severity: LOW**

The try/except pattern for optional OTel imports appears in:
- `guardrails_base.py:60-66`
- `cost_tracker.py:53-61`

Both set `OTEL_AVAILABLE` and fall back to `None` assignments. Minor duplication.

**Suggested fix**: Consider a shared `otel_compat.py` utility, but this is low priority.

---

### 9. Additional Findings

#### 9.1 PluginMiddleware on_response Hooks Fire Before send()
**Severity: MEDIUM**

In `plugin_middleware.py:238-249`:
```python
elif message["type"] == "http.response.body":
    if not message.get("more_body", False):
        ...
        await self._call_on_response_hooks(request, meta)
    await send(message)  # <-- sends AFTER hooks
```

The `on_response` hooks fire before the final body chunk is sent to the client. This means a slow plugin hook delays the response delivery. For streaming responses, this blocks the final chunk.

**Suggested fix**: Fire hooks after `await send(message)` to avoid delaying response delivery. Or fire them in a background task.

**Files**: `src/litellm_llmrouter/gateway/plugin_middleware.py:238-249`

#### 9.2 Cache Plugin Semantic Key Generation Not Used for L1
**Severity: LOW**

`cache_plugin.py:259` computes `CacheKeyGenerator.semantic_key()` which returns a `(prefix, embedding)` tuple, but `prefix` is assigned to an unused variable. The L1 cache always uses exact keys, and semantic lookup only checks L2.

**Files**: `src/litellm_llmrouter/gateway/plugins/cache_plugin.py:259`

#### 9.3 ConversationAffinityTracker Uses time.monotonic() for Serialization
**Severity: LOW**

`conversation_affinity.py:181-188`:
```python
now = time.monotonic()
record = AffinityRecord(
    ...
    recorded_at=now,
    expires_at=now + self._ttl_seconds,
)
```

`time.monotonic()` values are not meaningful across processes or machines. When serialized to Redis (via `to_json()`), a record created on node A with `recorded_at=12345.6` is meaningless on node B. The `is_expired()` check at line 69-73 uses `time.monotonic()` for comparison, which won't match Redis-stored records from other nodes.

**Impact**: In HA Redis mode, affinity records from other nodes are never correctly checked for expiry by the in-memory `is_expired()` method. However, Redis TTL handles expiry server-side, so the practical impact is limited to the `is_expired()` method being unreliable for Redis-sourced records.

**Suggested fix**: Use `time.time()` for serialized records so timestamps are meaningful across nodes.

**Files**: `src/litellm_llmrouter/conversation_affinity.py:69-73, 181-188`

---

## Environment Variable Inventory

All environment variables used by new components:

### Plugin System (plugin_manager.py)
| Variable | Default | Purpose |
|----------|---------|---------|
| `LLMROUTER_PLUGINS` | `""` | Comma-separated plugin module paths |
| `LLMROUTER_PLUGINS_ALLOWLIST` | `None` (all allowed) | Plugin allowlist |
| `LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES` | `None` (all allowed) | Capability restrictions |
| `LLMROUTER_PLUGINS_FAILURE_MODE` | `continue` | Global default failure mode |
| `ROUTEIQ_PLUGIN_STARTUP_TIMEOUT` | `30` | Plugin startup timeout (seconds) |
| `ROUTEIQ_PLUGIN_*` | (varies) | Plugin-specific settings (prefix stripped) |

### Guardrails - Prompt Injection (prompt_injection_guard.py)
| Variable | Default | Purpose |
|----------|---------|---------|
| `GUARDRAIL_INJECTION_ENABLED` | `false` | Enable prompt injection detection |
| `GUARDRAIL_INJECTION_ACTION` | `block` | Action: block / warn / log |
| `GUARDRAIL_INJECTION_PATTERNS` | `None` | Extra regex patterns (JSON list) |

### Guardrails - PII (pii_guard.py)
| Variable | Default | Purpose |
|----------|---------|---------|
| `GUARDRAIL_PII_ENABLED` | `false` | Enable PII detection/redaction |
| `GUARDRAIL_PII_ACTION` | `redact` | Action: redact / block / warn / log |
| `GUARDRAIL_PII_ENTITY_TYPES` | `""` (all) | Comma-separated entity types |

### Content Filter (content_filter.py)
| Variable | Default | Purpose |
|----------|---------|---------|
| `CONTENT_FILTER_ENABLED` | `false` | Enable content filtering |
| `CONTENT_FILTER_THRESHOLD` | `0.7` | Score threshold (0.0-1.0) |
| `CONTENT_FILTER_ACTION` | `block` | Default action: block / warn / log |
| `CONTENT_FILTER_CATEGORIES` | All 5 categories | Active categories (comma-separated) |

### Cost Tracker (cost_tracker.py)
| Variable | Default | Purpose |
|----------|---------|---------|
| `COST_TRACKER_ENABLED` | `true` | Enable cost tracking |

### Semantic Cache (cache_plugin.py)
| Variable | Default | Purpose |
|----------|---------|---------|
| `CACHE_ENABLED` | `false` | Enable response caching |
| `CACHE_SEMANTIC_ENABLED` | `false` | Enable semantic (embedding) matching |
| `CACHE_TTL_SECONDS` | `3600` | Cache TTL |
| `CACHE_L1_MAX_SIZE` | `1000` | Max L1 entries |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Cosine similarity threshold |
| `CACHE_REDIS_URL` | `None` | Redis URL for L2 |
| `CACHE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name |
| `CACHE_MAX_TEMPERATURE` | `0.1` | Max cacheable temperature |

### Conversation Affinity (conversation_affinity.py)
| Variable | Default | Purpose |
|----------|---------|---------|
| `CONVERSATION_AFFINITY_ENABLED` | `false` | Enable affinity tracking |
| `CONVERSATION_AFFINITY_TTL` | `3600` | TTL for affinity records (seconds) |
| `CONVERSATION_AFFINITY_MAX_ENTRIES` | `10000` | Max in-memory entries |

### Strategy Registry (strategy_registry.py)
| Variable | Default | Purpose |
|----------|---------|---------|
| `LLMROUTER_ACTIVE_ROUTING_STRATEGY` | `llmrouter-default` | Active routing strategy |
| `LLMROUTER_STRATEGY_WEIGHTS` | `None` | JSON dict of A/B weights |
| `LLMROUTER_EXPERIMENT_ID` | `None` | Experiment identifier |
| `LLMROUTER_EXPERIMENT_CONFIG` | `None` | Advanced experiment config (JSON) |

### Naming Convention Analysis

Three naming prefixes are used:
- `GUARDRAIL_*` -- guardrail plugins (3 plugins, consistent)
- `CONTENT_FILTER_*` -- content filter (should have been `GUARDRAIL_CONTENT_*`)
- `CACHE_*` -- cache plugin (clean, short)
- `COST_TRACKER_*` -- cost tracker (clean)
- `CONVERSATION_AFFINITY_*` -- affinity tracker (verbose but clear)
- `LLMROUTER_*` -- strategy registry + plugin system (existing convention)
- `ROUTEIQ_*` -- plugin-specific settings (new prefix, used sparingly)

**Inconsistency**: `CONTENT_FILTER_*` doesn't follow the `GUARDRAIL_*` prefix used by the other two guardrail plugins, despite being a guardrail by capability declaration.

---

## Summary of Findings by Severity

| Severity | Count | Key Items |
|----------|-------|-----------|
| CRITICAL | 0 | -- |
| HIGH | 1 | Duplicate metric systems |
| MEDIUM | 8 | Config timing, missing resets, wrong base class, blocking model load, Redis reconnection, response hook timing, time.monotonic cross-node, no numeric validation |
| LOW | 5 | Module-scope config, lazy litellm import, response extraction duplication, OTel import duplication, unused semantic prefix |

No critical defects found. The codebase is well-structured overall with clean module boundaries and good fault isolation. The highest-impact issues are the duplicate metric systems and the ContentFilterPlugin base class mismatch, both of which should be addressed before v0.0.2 release.
