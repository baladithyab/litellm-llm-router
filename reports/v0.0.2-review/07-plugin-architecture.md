# RouteIQ v0.0.2 Plugin Architecture Review

**Reviewer:** plugin-reviewer (automated analysis)
**Date:** 2025-02-07
**Scope:** Plugin system architecture, SDK gaps, competitive research, extensibility recommendations

---

## 1. Current State Summary

### 1.1 Architecture Overview

RouteIQ's plugin system is built around four core components:

| Component | File | Role |
|-----------|------|------|
| **Plugin Manager** | `gateway/plugin_manager.py` (1056 lines) | Plugin lifecycle: discovery, validation, dependency resolution, startup/shutdown |
| **Plugin Middleware** | `gateway/plugin_middleware.py` (317 lines) | ASGI-level request/response hooks for HTTP interception |
| **Callback Bridge** | `gateway/plugin_callback_bridge.py` (250 lines) | Bridges LiteLLM's callback system to plugin LLM lifecycle hooks |
| **App Factory** | `gateway/app.py` (506 lines) | Composition root that wires plugins into the FastAPI application |

### 1.2 Plugin Base Class (`GatewayPlugin`)

The `GatewayPlugin` ABC defines the following hooks:

| Hook | When Called | Can Short-Circuit? | Can Modify Data? |
|------|-----------|-------------------|-----------------|
| `startup(app, context)` | App startup (ordered by deps+priority) | N/A | Can register routes, middleware |
| `shutdown(app, context)` | App shutdown (reverse order) | N/A | Can clean up resources |
| `health_check()` | Readiness probes | No | Returns health dict |
| `on_request(request)` | Before each HTTP request | **Yes** (return `PluginResponse`) | Read-only (immutable `PluginRequest`) |
| `on_response(request, response_meta)` | After each HTTP response | No | Read-only (status+headers only) |
| `on_llm_pre_call(model, messages, kwargs)` | Before LLM API call | **Yes** (raise `GuardrailBlockError`) | **Yes** (return kwargs overrides) |
| `on_llm_success(model, response, kwargs)` | After successful LLM call | No | No (observability only) |
| `on_llm_failure(model, exception, kwargs)` | After failed LLM call | No | No (observability only) |
| `on_config_reload(old, new)` | Config hot-reload | No | No |
| `on_route_register(path, methods)` | Route registration | No | No |
| `on_model_health_change(model, healthy, reason)` | Model health transitions | No | No |

### 1.3 Plugin Capabilities (11 total)

```
ROUTES, ROUTING_STRATEGY, TOOL_RUNTIME, EVALUATOR,
OBSERVABILITY_EXPORTER, MIDDLEWARE, AUTH_PROVIDER,
STORAGE_BACKEND, GUARDRAIL, CACHE, COST_TRACKER
```

### 1.4 Plugin Metadata & Configuration

- **PluginMetadata**: name, version, capabilities, depends_on, priority, failure_mode, description
- **PluginContext**: settings dict, logger, SSRF URL validator
- **FailureMode**: CONTINUE (default), ABORT, QUARANTINE
- **Environment-based config**: `LLMROUTER_PLUGINS`, `LLMROUTER_PLUGINS_ALLOWLIST`, `LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES`, `LLMROUTER_PLUGINS_FAILURE_MODE`
- **Plugin-specific settings**: Any env var prefixed with `ROUTEIQ_PLUGIN_` is passed to plugins via `PluginContext.settings`
- **Startup timeout**: Configurable via `ROUTEIQ_PLUGIN_STARTUP_TIMEOUT` (default: 30s)

### 1.5 Built-in Plugins (9 total)

| Plugin | Capability | Priority | Hooks Used |
|--------|-----------|----------|------------|
| `SkillsDiscoveryPlugin` | ROUTES | 500 | startup (registers routes) |
| `EvaluatorPlugin` (base) | EVALUATOR | 2000 | evaluate_mcp_result, evaluate_a2a_result |
| `UpskillEvaluatorPlugin` | EVALUATOR | 2000 | evaluate_mcp_result, evaluate_a2a_result |
| `GuardrailPlugin` (base) | EVALUATOR | 50 | on_llm_pre_call, on_llm_success |
| `PromptInjectionGuard` | EVALUATOR | 50 | on_llm_pre_call (via evaluate_input) |
| `PIIGuard` | EVALUATOR | 60 | on_llm_pre_call, on_llm_success (via evaluate_input/output) |
| `ContentFilterPlugin` | GUARDRAIL | 70 | on_llm_pre_call, on_llm_success |
| `CostTrackerPlugin` | EVALUATOR, OBSERVABILITY_EXPORTER | 50 | on_llm_pre_call, on_llm_success, on_llm_failure |
| `SemanticCachePlugin` | MIDDLEWARE | 10 | on_llm_pre_call, on_llm_success, health_check |

### 1.6 Security Model

- **Allowlist enforcement**: Checked before importing plugin module (prevents untrusted code execution)
- **Capability policy**: Plugins requesting disallowed capabilities are rejected
- **SSRF protection**: `validate_outbound_url` provided in PluginContext
- **Error isolation**: All hook failures are caught and logged; never crash the request pipeline
- **Quarantine**: Failed plugins are disabled and excluded from future hook calls

### 1.7 Dependency Resolution

Uses Kahn's algorithm for topological sort with priority tiebreaking:
1. Build dependency graph from `depends_on` declarations
2. Detect missing dependencies (raises `PluginDependencyError`)
3. Detect circular dependencies (raises `PluginDependencyError`)
4. Among plugins with equal dependency level, sort by priority (lower = first)

### 1.8 Test Coverage

Three dedicated test files exist:
- `tests/unit/test_plugin_manager.py` (955 lines) -- 28 test cases covering: registration, startup/shutdown ordering, config loading, allowlist, capabilities, dependencies, failure modes, backwards compatibility, singletons, error classes
- `tests/unit/test_plugin_callback_bridge.py` -- Callback bridge integration
- `tests/unit/test_plugin_middleware.py` -- ASGI middleware hooks

---

## 2. Gap Analysis

### 2.1 Hook Coverage Gaps

| Missing Hook Point | Impact | Severity |
|-------------------|--------|----------|
| **Routing decision** (`on_route_decision`) | Plugins cannot intercept or influence model routing decisions | **High** |
| **Streaming chunk** (`on_stream_chunk`) | Cannot inspect/transform individual SSE chunks | **Medium** |
| **Request body access** | `PluginRequest` is read-only with no body; can't inspect/modify POST body at ASGI level | **Medium** |
| **on_startup_complete** | No hook for after ALL plugins have started (useful for cross-plugin coordination) | **Low** |
| **on_shutdown_begin** | No hook before shutdown starts (for graceful drain signaling) | **Low** |
| **on_llm_stream_chunk** | Cannot inspect individual streaming response chunks in the LLM callback path | **Medium** |
| **Authentication hook** (`on_auth`) | Despite AUTH_PROVIDER capability, no actual auth hook exists | **High** |
| **Rate limit / quota hook** | No plugin hook for quota enforcement decisions | **Low** |

### 2.2 Short-Circuit Capability Analysis

| Hook | Can Short-Circuit? | Mechanism | Notes |
|------|-------------------|-----------|-------|
| `on_request` | Yes | Return `PluginResponse` | Works well; sends JSON response directly |
| `on_llm_pre_call` | Partially | Raise `GuardrailBlockError` | Only blocks; cannot return a **cached/synthetic response** directly |
| `on_llm_success` | No | N/A | Cannot modify the response sent to the client |
| `on_response` | No | N/A | Observability only |

**Critical gap**: `on_llm_pre_call` cannot return a synthetic response (e.g., cached response). The `SemanticCachePlugin` works around this by stuffing a `_cache_hit_response` into `kwargs["metadata"]`, but the actual cache short-circuit must happen elsewhere in the LiteLLM pipeline. This is fragile and undocumented.

### 2.3 Plugin Ordering Verification

**Priority IS respected** in both middleware and callback paths:
- `PluginMiddleware` calls `on_request` in plugin order, `on_response` in reverse order (symmetric wrapping) -- **correct**
- `PluginCallbackBridge` calls hooks in registration order -- **correct but subtle**: the callback bridge gets plugins from `get_callback_plugins()` which returns them in registration order, not sorted order. If plugins are sorted during startup, the list order is correct. But `get_callback_plugins()` iterates `self._plugins` (registration order, not sorted order).

**Potential issue**: `get_middleware_plugins()` and `get_callback_plugins()` iterate `self._plugins` which is the **registration order**, not the topologically-sorted order. The sort only happens during `startup()`. This means if plugin A has priority 10 and plugin B has priority 100, but B was registered first, the callback/middleware order would be [B, A] rather than [A, B].

### 2.4 Plugin Modification Visibility

When `on_llm_pre_call` returns kwargs overrides, they are merged into the shared `kwargs` dict via `kwargs.update(result)`. This means:
- Later plugins in the chain **do see** earlier plugins' modifications (they share the same kwargs dict)
- But there's no mechanism to see what specifically another plugin changed
- No audit trail of modifications

### 2.5 Error Isolation Assessment

| Component | Isolation | Mechanism |
|-----------|-----------|-----------|
| Plugin startup | Per-plugin try/catch | Handles TimeoutError, any Exception |
| Plugin shutdown | Per-plugin try/catch | Reverse order, continues on failure |
| on_request | Per-plugin try/catch | Catches, logs, continues to next plugin |
| on_response | Per-plugin try/catch | Catches, logs, continues |
| on_llm_pre_call | Per-plugin try/catch | **Except**: `GuardrailBlockError` propagates intentionally |
| on_llm_success | Per-plugin try/catch | Catches, logs, continues |
| on_llm_failure | Per-plugin try/catch | Catches, logs, continues |

**Verdict**: Error isolation is well-implemented. One plugin failure will not affect others (except the intentional GuardrailBlockError propagation).

### 2.6 Hot Reload

**NOT supported.** Once `PluginManager.startup()` is called:
- `self._started = True` prevents new registrations (`register()` raises `RuntimeError`)
- No `reload()` or `unload()` method exists
- No mechanism to add/remove/update plugins at runtime
- The `on_config_reload` hook exists but is not wired to any config reload system

### 2.7 External Plugin Loading

**Partially supported.** Plugins are loaded via `importlib.import_module()` using fully-qualified Python paths:
- `LLMROUTER_PLUGINS=mypackage.myplugin.MyPlugin` -- works if `mypackage` is pip-installed
- No entry point discovery (e.g., `pkg_resources.iter_entry_points`)
- No plugin registry or marketplace
- No plugin packaging standard
- No way to specify plugin configuration in the LLMROUTER_PLUGINS env var (must use separate env vars)

### 2.8 Plugin Configuration

**Environment-variable only.** Current mechanisms:
- Per-plugin env vars (each plugin reads its own `GUARDRAIL_*`, `COST_TRACKER_*`, etc.)
- `ROUTEIQ_PLUGIN_` prefix forwarded to `PluginContext.settings`
- No YAML config support for plugin-specific settings
- No config validation/schema for plugin settings
- No way to pass structured config (lists, nested objects) via env vars

### 2.9 Plugin Testing

**No test harness exists.** Current state:
- Unit tests directly instantiate plugins and call hooks
- No `MockGateway` or `PluginTestHarness` class
- No way to test plugins in isolation without importing the full gateway
- No pytest fixtures for plugin testing
- No example test template for plugin developers

---

## 3. Competitive Research

### 3.1 Kong Gateway Plugin Development Kit (PDK)

**Architecture**: Kong uses a Lua-based plugin system with a well-defined lifecycle:

| Phase | Kong Hook | Equivalent in RouteIQ |
|-------|-----------|----------------------|
| Certificate | `:certificate()` | Not applicable |
| Rewrite | `:rewrite()` | `on_request` |
| Access | `:access()` | `on_request` + `on_llm_pre_call` |
| Response | `:response()` | `on_response` |
| Header filter | `:header_filter()` | Not available |
| Body filter | `:body_filter()` | **Missing** |
| Log | `:log()` | `on_llm_success/failure` |
| Pre-function | `:preread()` | Not applicable |

**Key Kong SDK features RouteIQ lacks**:
1. **Plugin schema validation** -- Kong uses `schema.lua` to define and validate plugin configuration. RouteIQ has no config schema mechanism.
2. **Plugin DAOs** -- Kong plugins can define database tables. RouteIQ plugins have no storage abstraction.
3. **Admin API extension** -- Kong plugins can add custom Admin API endpoints via `api.lua`. RouteIQ supports this via ROUTES capability + `app.include_router()`.
4. **Body filter phase** -- Kong can inspect/modify response bodies chunk-by-chunk. RouteIQ intentionally omits body access to preserve streaming.
5. **Plugin template/generator** -- Kong provides `kong plugin create` CLI command.

### 3.2 Envoy Extension Model

Envoy supports 6 extension mechanisms:

| Mechanism | Performance | Dev Complexity | Safety |
|-----------|-------------|---------------|--------|
| C++ Filter | Highest | Highest | Low (crash risk) |
| Lua Script | Medium | Low | Medium |
| Wasm Plugin | High | Medium | High (sandboxed) |
| Dynamic Module | Near-native | Medium | Medium |
| ext_proc (gRPC) | Lower | Low | High (out-of-process) |
| ext_authz (gRPC) | Lower | Low | High (out-of-process) |

**Relevant patterns for RouteIQ**:
1. **ext_proc pattern** -- External processing via gRPC. Plugins run out-of-process, communicating via well-defined protocol. Provides maximum isolation but adds latency.
2. **Wasm sandboxing** -- Plugins run in a sandbox with limited host API access. Could inspire a "capability-based access control" model for RouteIQ plugins.
3. **Filter chain** -- Envoy's filter chain is explicit and ordered. RouteIQ's equivalent is the priority-sorted plugin list.

### 3.3 Grafana Plugin SDK

Grafana separates plugins into:
1. **Frontend plugins** -- React components bundled as JavaScript
2. **Backend plugins** -- Go binaries communicating via gRPC (hashicorp/go-plugin)
3. **Plugin types**: Panel, Data Source, App (combined)

**Relevant patterns**:
1. **Plugin scaffolding** -- `create-plugin` CLI generates a complete plugin project with tests, CI, and documentation
2. **Plugin signing** -- Grafana verifies plugin signatures before loading (security)
3. **Plugin marketplace/catalog** -- grafana.com/plugins with search, ratings, versioning
4. **Backend SDK versioning** -- Explicit SDK version compatibility (e.g., "requires SDK >= 0.232.0")

### 3.4 LiteLLM Native Callback System

LiteLLM's callback system (which RouteIQ bridges into):

| Hook | Sync | Async | Proxy-Only |
|------|------|-------|-----------|
| `log_pre_api_call` | Yes | No | No |
| `log_post_api_call` | Yes | No | No |
| `log_success_event` | Yes | Yes | No |
| `log_failure_event` | Yes | Yes | No |
| `async_pre_call_hook` | No | Yes | **Yes** |
| `async_post_call_success_hook` | No | Yes | **Yes** |
| `async_post_call_failure_hook` | No | Yes | **Yes** |

**Key observations**:
- RouteIQ's `PluginCallbackBridge` only wires 3 of the 7 available hooks
- `async_pre_call_hook` (proxy-only, can modify request data) is available but unused
- `async_post_call_success_hook` (can modify response, has user key info) is declared as no-op
- LiteLLM uses `CustomLogger` base class; RouteIQ uses duck-typing (looser coupling, less discoverable)

### 3.5 FastAPI Middleware Best Practices

Key patterns from the FastAPI ecosystem:
1. **Pure ASGI middleware** (not `BaseHTTPMiddleware`) for streaming -- RouteIQ already does this correctly
2. **Dependency injection** for plugin access -- RouteIQ uses singletons instead (simpler but less testable)
3. **Middleware ordering** via `app.add_middleware()` -- outermost added first -- RouteIQ documents this clearly
4. **Background tasks** for post-response processing -- RouteIQ could use this for async plugin hooks

---

## 4. SDK Design Proposal

### 4.1 Proposed `routeiq-plugin-sdk` Package Structure

```
routeiq-plugin-sdk/
  src/routeiq_plugin_sdk/
    __init__.py              # Public API re-exports
    plugin.py                # GatewayPlugin, PluginMetadata, PluginCapability
    hooks.py                 # Hook type definitions (PluginRequest, PluginResponse, etc.)
    context.py               # PluginContext, settings schema
    config.py                # Plugin config schema validation (Pydantic)
    testing/
      __init__.py
      harness.py             # PluginTestHarness (mock gateway for testing)
      fixtures.py            # pytest fixtures (mock_app, mock_context, mock_request, etc.)
      assertions.py          # Custom assertions (assert_hook_called, assert_span_emitted, etc.)
    guardrails/
      __init__.py
      base.py                # GuardrailPlugin, GuardrailDecision, GuardrailBlockError
    evaluators/
      __init__.py
      base.py                # EvaluatorPlugin, EvaluationResult, MCPInvocationContext, etc.
    telemetry/
      __init__.py
      otel.py                # OTel helpers (span attribute setters, metric builders)
    errors.py                # Plugin error types (PluginLoadError, etc.)
    version.py               # SDK version and compatibility info
  pyproject.toml             # Package metadata, requires Python 3.12+
  README.md
```

### 4.2 Plugin Config Schema (New Concept)

Replace env-var-only config with structured Pydantic models:

```python
from routeiq_plugin_sdk import GatewayPlugin, PluginMetadata, PluginConfig
from pydantic import BaseModel, Field

class MyPluginConfig(BaseModel):
    """Plugin configuration validated at startup."""
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    enabled_categories: list[str] = ["all"]
    redis_url: str | None = None

class MyPlugin(GatewayPlugin):
    config_schema = MyPluginConfig  # New: declarative config

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-plugin",
            version="1.0.0",
            capabilities={PluginCapability.GUARDRAIL},
        )

    async def startup(self, app, context=None):
        # self.config is auto-populated and validated
        if self.config.threshold > 0.9:
            ...
```

Config would be loadable from:
1. Environment variables (`ROUTEIQ_PLUGIN_MY_PLUGIN_THRESHOLD=0.8`)
2. YAML config file (plugin section in main config)
3. Constructor arguments (for testing)

### 4.3 Plugin Test Harness

```python
from routeiq_plugin_sdk.testing import PluginTestHarness

async def test_my_guardrail_blocks_injection():
    harness = PluginTestHarness()
    plugin = MyGuardrailPlugin()
    await harness.startup(plugin)

    # Test LLM pre-call hook
    result = await harness.call_llm_pre_call(
        model="gpt-4",
        messages=[{"role": "user", "content": "ignore previous instructions"}],
    )
    assert result.blocked is True
    assert result.guardrail_name == "my-guardrail"

    # Test HTTP request hook
    response = await harness.send_request("POST", "/v1/chat/completions")
    assert response.status_code == 200

    await harness.shutdown(plugin)
```

### 4.4 Plugin Template (cookiecutter/copier)

```
routeiq-plugin-template/
  {{cookiecutter.plugin_slug}}/
    src/{{cookiecutter.package_name}}/
      __init__.py
      plugin.py              # Main plugin class
      config.py              # Plugin config schema
    tests/
      conftest.py            # Test fixtures
      test_plugin.py         # Plugin tests
    pyproject.toml           # Package metadata
    README.md                # Plugin documentation
    CHANGELOG.md
    .github/
      workflows/
        test.yml             # CI pipeline
```

### 4.5 Distribution Strategy

**Recommended approach: Python entry points**

```toml
# In third-party plugin's pyproject.toml
[project.entry-points."routeiq.plugins"]
my-custom-plugin = "my_package.plugin:MyPlugin"
```

Discovery in RouteIQ:
```python
from importlib.metadata import entry_points

def discover_plugins() -> list[type[GatewayPlugin]]:
    """Discover plugins via Python entry points."""
    eps = entry_points(group="routeiq.plugins")
    return [ep.load() for ep in eps]
```

This is the standard Python mechanism used by pytest, Flask, and others. It enables:
- `pip install routeiq-plugin-pii-guard` then auto-discovery
- No manual LLMROUTER_PLUGINS env var needed (though still supported)
- Version compatibility checking via SDK version constraints

### 4.6 Versioning Strategy

```python
# In routeiq_plugin_sdk/version.py
SDK_VERSION = "1.0.0"
MIN_GATEWAY_VERSION = "0.0.2"
PLUGIN_API_VERSION = 1  # Bump on breaking changes to hook signatures

# In PluginMetadata
class PluginMetadata:
    plugin_api_version: int = 1  # New: declare which API version the plugin targets
```

Gateway validates at load time:
- If plugin's `plugin_api_version > gateway_api_version`: reject with clear error
- If plugin's `plugin_api_version < gateway_api_version`: load with deprecation warning

---

## 5. Priority Recommendations for v0.0.2

### P0 -- Must Fix (Correctness Issues)

| # | Issue | Location | Recommendation |
|---|-------|----------|---------------|
| 1 | **Callback/middleware plugin order not sorted** | `plugin_manager.py:949-992` | `get_middleware_plugins()` and `get_callback_plugins()` iterate `self._plugins` (registration order). Should iterate sorted order. Store sorted list after `startup()` and use it. |
| 2 | **`on_config_reload` not wired** | `plugin_manager.py:380-393` | Hook exists in base class but is never called from `hot_reload.py` or `config_sync.py`. Either wire it or remove it (dead code). |
| 3 | **`on_route_register` not wired** | `plugin_manager.py:395-405` | Same -- hook exists but never called from `app.py` or `routes.py`. |
| 4 | **`on_model_health_change` not wired** | `plugin_manager.py:407-421` | Same pattern. Declared but not called from any health check system. |
| 5 | **GuardrailPlugin metadata uses EVALUATOR capability** | `guardrails_base.py:137-144` | GuardrailPlugin base class declares `PluginCapability.EVALUATOR` but should declare `PluginCapability.GUARDRAIL`. PromptInjectionGuard and PIIGuard inherit this wrong capability. ContentFilterPlugin correctly uses GUARDRAIL. |

### P1 -- Should Fix (Gaps Affecting Plugin Developers)

| # | Issue | Recommendation |
|---|-------|---------------|
| 6 | **No plugin test harness** | Create `PluginTestHarness` class with `startup()`, `call_llm_pre_call()`, `send_request()` methods. Include pytest fixtures. |
| 7 | **No config schema validation** | Add optional `config_schema` class attribute (Pydantic model). Validate during startup. |
| 8 | **No entry point discovery** | Add `importlib.metadata.entry_points` discovery alongside env var loading. |
| 9 | **Cache short-circuit is fragile** | `SemanticCachePlugin` stuffs `_cache_hit_response` into metadata but nothing in the LLM pipeline actually uses it to skip the API call. Document or fix the cache hit path. |
| 10 | **Unused LiteLLM proxy hooks** | Wire `async_pre_call_hook` and `async_post_call_success_hook` through the PluginCallbackBridge. These proxy-only hooks provide access to user API key data and response modification. |

### P2 -- Nice to Have (v0.1.0 Roadmap)

| # | Issue | Recommendation |
|---|-------|---------------|
| 11 | **Plugin SDK package** | Extract plugin base classes into `routeiq-plugin-sdk` package for independent versioning. |
| 12 | **Plugin template** | Create cookiecutter/copier template for new plugins. |
| 13 | **Streaming body hooks** | Add `on_stream_chunk(chunk)` for SSE/streaming response inspection (needed for output guardrails on streaming). |
| 14 | **Plugin hot reload** | Add `reload_plugins()` method for zero-downtime plugin updates. |
| 15 | **Request body access** | Optionally allow plugins to access request body in `on_request` (with buffering opt-in flag per plugin). |
| 16 | **Plugin API versioning** | Add `plugin_api_version` to PluginMetadata for forward compatibility. |
| 17 | **Auth provider hook** | Implement actual `on_auth(request, credentials)` hook to make AUTH_PROVIDER capability functional. |

---

## 6. Detailed Findings

### 6.1 Plugin Lifecycle Flow

```
Gateway Startup
  |
  v
create_app() / create_standalone_app()
  |
  +-- _configure_middleware()
  |     +-- RequestIDMiddleware (outermost)
  |     +-- PolicyMiddleware
  |     +-- PluginMiddleware (self-registers singleton)
  |     +-- RouterDecisionMiddleware
  |
  +-- _load_plugins_before_routes()
  |     +-- PluginManager.load_from_config()
  |           +-- Parse LLMROUTER_PLUGINS env var
  |           +-- For each plugin path:
  |                 +-- _validate_allowlist() (BEFORE import)
  |                 +-- importlib.import_module()
  |                 +-- isinstance check (GatewayPlugin)
  |                 +-- _validate_capabilities() (AFTER instantiation)
  |                 +-- manager.register()
  |
  +-- _register_routes()
  |
  +-- Store plugin hooks on app.state as lambdas
  |
  v
startup.py calls app.state.llmrouter_plugin_startup()
  |
  +-- _run_plugin_startup(app)
        +-- PluginManager.startup(app)
        |     +-- _topological_sort() (Kahn's algorithm + priority)
        |     +-- _create_context() (settings, logger, URL validator)
        |     +-- For each sorted plugin:
        |           +-- asyncio.wait_for(plugin.startup(app, context), timeout)
        |           +-- _handle_failure() on error
        |
        +-- Wire middleware plugins -> PluginMiddleware.set_plugins()
        +-- Wire callback plugins -> register_callback_bridge()
```

### 6.2 Request Processing Flow

```
HTTP Request
  |
  v
RequestIDMiddleware
  |
  v
PolicyMiddleware (OPA policy check)
  |
  v
PluginMiddleware.on_request (foreach plugin in order)
  |  <-- Can short-circuit with PluginResponse
  v
RouterDecisionMiddleware (telemetry)
  |
  v
BackpressureMiddleware
  |
  v
FastAPI route handler
  |
  v
LiteLLM Router
  |
  v
PluginCallbackBridge.async_log_pre_api_call (foreach plugin)
  |  <-- Can raise GuardrailBlockError
  |  <-- Can modify kwargs
  v
LLM API Call
  |
  v (success)
PluginCallbackBridge.async_log_success_event (foreach plugin)
  |
  v
Response sent to client
  |
  v
PluginMiddleware.on_response (foreach plugin in REVERSE order)
```

### 6.3 Built-in Plugin Quality Assessment

| Plugin | Code Quality | Test Coverage | Config Validation | OTel Integration | Verdict |
|--------|-------------|---------------|------------------|-----------------|---------|
| SkillsDiscoveryPlugin | Good (path traversal protection, caching) | Not reviewed | Env var only | None | Solid |
| PromptInjectionGuard | Good (compiled regex, extensible patterns) | Not reviewed | Env var only | Via GuardrailPlugin base | Good |
| PIIGuard | Good (multiple entity types, redact+block+warn) | Not reviewed | Env var only | Via GuardrailPlugin base | Good |
| ContentFilterPlugin | Good (multi-category, keyword+pattern scoring) | Not reviewed | Env var only | None (uses GuardrailBlockError directly) | Good |
| CostTrackerPlugin | Good (accurate cost from usage, metric instruments) | Not reviewed | Env var only | Full (5 metrics + span attributes) | Excellent |
| SemanticCachePlugin | Good (two-tier, configurable) | Not reviewed | Env var only | None | Good, but cache-hit short-circuit path unclear |
| UpskillEvaluatorPlugin | Stub (CLI/service integration not implemented) | Not reviewed | Env var only | Via EvaluatorPlugin base | Incomplete |

### 6.4 Capability-to-Hook Mapping Gaps

Several capabilities have no corresponding hooks or enforcement:

| Capability | Hooks Available | Gap |
|-----------|----------------|-----|
| ROUTES | startup (register routes) | None -- working correctly |
| ROUTING_STRATEGY | None | No hook for routing decisions |
| TOOL_RUNTIME | None | No hook for tool execution |
| EVALUATOR | evaluate_mcp_result, evaluate_a2a_result | Working for evaluators |
| OBSERVABILITY_EXPORTER | on_llm_success/failure | Working via callbacks |
| MIDDLEWARE | on_request/on_response | Working |
| AUTH_PROVIDER | None | **No auth hook exists** |
| STORAGE_BACKEND | None | **No storage abstraction** |
| GUARDRAIL | on_llm_pre_call (via GuardrailPlugin) | Working but base class uses wrong capability |
| CACHE | on_llm_pre_call/success | Working via callbacks |
| COST_TRACKER | on_llm_pre_call/success/failure | Working via callbacks |

---

## 7. Summary

### Strengths

1. **Clean architecture** -- Clear separation between plugin manager, middleware, and callback bridge
2. **Security model** -- Allowlist, capability policy, SSRF protection, quarantine on failure
3. **Error isolation** -- Plugins cannot crash the request pipeline
4. **Dependency resolution** -- Topological sort with cycle detection
5. **Multiple hook points** -- ASGI-level (request/response) + LLM lifecycle (pre_call/success/failure) + infrastructure (config/health/routes)
6. **Good base class hierarchy** -- GatewayPlugin -> GuardrailPlugin -> specific guards, GatewayPlugin -> EvaluatorPlugin -> specific evaluators

### Weaknesses

1. **Dead hooks** -- `on_config_reload`, `on_route_register`, `on_model_health_change` are declared but never called
2. **Ordering bug** -- Callback/middleware plugins use registration order, not sorted order
3. **No test harness** -- Plugin developers must understand gateway internals to write tests
4. **No config schema** -- All config is env-var based with no validation
5. **No SDK package** -- Plugin base classes are tightly coupled to the gateway codebase
6. **No entry point discovery** -- Third-party plugins require manual env var configuration
7. **Incomplete capabilities** -- AUTH_PROVIDER, STORAGE_BACKEND, ROUTING_STRATEGY have no hooks
8. **No streaming hooks** -- Cannot inspect/transform streaming response bodies
9. **Cache short-circuit fragility** -- SemanticCachePlugin's cache hit path is not clearly documented or enforced

### Overall Assessment

The plugin system is **architecturally sound** with a good foundation for extensibility. The hook model covers the most critical interception points (HTTP request/response + LLM pre/post call). Error isolation and security policies are well-implemented.

However, the system is **not yet ready for third-party developers** due to: lack of a test harness, no config schema validation, dead hooks, the ordering bug, and tight coupling to the gateway codebase. The P0 fixes (ordering, dead hooks, wrong capability on GuardrailPlugin) should be addressed before declaring the plugin API stable.

For v0.0.2, prioritize the P0 correctness fixes and begin work on the plugin test harness (P1). The full SDK extraction (P2) can be deferred to v0.1.0 when the hook API is stable.
