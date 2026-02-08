# Architecture Review 08: Plugin Ecosystem Analysis

**Date**: 2026-02-07
**Scope**: Plugin/extension system architecture for RouteIQ AI Gateway
**Status**: Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Industry Reference: Gateway Plugin Architectures](#2-industry-reference-gateway-plugin-architectures)
3. [RouteIQ Plugin System: Current State](#3-routeiq-plugin-system-current-state)
4. [Hook Completeness Assessment](#4-hook-completeness-assessment)
5. [Gap Analysis vs. Mature Gateways](#5-gap-analysis-vs-mature-gateways)
6. [Architecture Recommendations](#6-architecture-recommendations)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Appendix: File Reference](#8-appendix-file-reference)

---

## 1. Executive Summary

RouteIQ has completed Phase 1 of its plugin system, delivering a solid foundation with
ASGI-level request/response hooks (PluginMiddleware), LiteLLM callback hooks
(PluginCallbackBridge), lifecycle management with dependency resolution, and three
built-in plugins. The system is well-tested (155+ test assertions across three test files)
and production-oriented with failure modes, quarantine, allowlist security, and
capability-based policy.

However, compared to mature gateway plugin ecosystems (Kong, Envoy, Traefik), RouteIQ's
plugin system has notable gaps in: configuration schema validation, plugin hot-reload,
body access hooks, streaming transformation, SDK tooling, and marketplace/distribution
infrastructure. This report catalogs these gaps and provides a prioritized roadmap.

**Overall Maturity Rating**: Phase 1 Complete (Foundation) -- approximately 40% of a
mature gateway plugin ecosystem.

---

## 2. Industry Reference: Gateway Plugin Architectures

### 2.1 Kong Gateway Plugin Model

Kong is the gold standard for API gateway plugin architecture. Key design elements:

**Phase-Based Execution Model**:
Kong plugins execute at specific phases in the request lifecycle, each with distinct
semantics:

| Phase | When | Typical Use |
|-------|------|-------------|
| `init_worker` | Worker process start | Background timers, global state |
| `certificate` | TLS handshake | mTLS validation |
| `rewrite` | Before routing | URL rewriting, header manipulation |
| `access` | After routing, before upstream | Authentication, authorization, rate limiting |
| `response` | Streaming response from upstream | Response transformation (buffered) |
| `header_filter` | Response headers received | Header manipulation |
| `body_filter` | Each response body chunk | Body transformation (streaming-safe) |
| `log` | After response sent | Logging, analytics, metrics |
| `error` | Error occurred | Custom error responses |

**Plugin Development Kit (PDK)**:
Kong provides `kong.pdk` -- a stable API surface that abstracts Nginx internals:
- `kong.request.*` - Read request headers, body, query params
- `kong.response.*` - Set response headers, status, body
- `kong.service.request.*` - Modify upstream request
- `kong.service.response.*` - Read upstream response
- `kong.ctx.shared` - Cross-plugin shared context
- `kong.log.*` - Structured logging
- `kong.node.*` - Node information
- `kong.ip.*` - IP utilities
- `kong.client.*` - Client certificate info

**Plugin Configuration**:
Each plugin has a `schema.lua` that defines typed configuration with validation:
```lua
return {
  name = "rate-limiting",
  fields = {
    { config = {
        type = "record",
        fields = {
          { second = { type = "number", gt = 0 } },
          { minute = { type = "number", gt = 0 } },
          { policy = { type = "string", default = "local",
                       one_of = { "local", "cluster", "redis" } } },
        },
    }},
  },
}
```

**Plugin Ordering/Priority**:
Kong uses numeric priority (higher = runs first in access phase, lower = runs first in
log phase). Default bundled plugin priorities are well-documented, allowing third-party
plugins to insert at precise points.

**Distribution**:
Kong Hub hosts 100+ plugins. Plugins are distributed as LuaRocks packages with
`kong-plugin-<name>` naming convention, or as Docker layers for custom builds.

### 2.2 Envoy Proxy Filter Chain

Envoy's extensibility model is the most sophisticated in the industry:

**HTTP Filter Chain**:
Envoy processes requests through an ordered chain of HTTP filters. Each filter can:
- Decode (modify request going upstream)
- Encode (modify response going downstream)
- Both (bidirectional)

Filter types: `decoder_filter`, `encoder_filter`, `dual_filter`, `access_log`.

**Extension Mechanisms** (in order of maturity):

1. **Native C++ Filters**: Highest performance, compiled into Envoy binary. Used for
   core functionality (router, RBAC, rate limit).

2. **Lua Filters**: Inline scripting via `envoy.filters.http.lua`. Good for simple
   request/response transformations. Sandboxed with limited API.

3. **ext_proc (External Processing)**: gRPC-based external processing service. Envoy
   sends request/response data to an external gRPC service for decisions. Supports:
   - Request header processing
   - Request body processing (buffered or streamed)
   - Response header processing
   - Response body processing (buffered or streamed)
   - Immediate response (short-circuit)

4. **WebAssembly (Wasm) Filters**: Plugins compiled to Wasm, loaded at runtime.
   Supported languages: Rust, Go, C++, AssemblyScript. Uses the proxy-wasm ABI:
   - `on_request_headers`, `on_request_body`
   - `on_response_headers`, `on_response_body`
   - `on_log`
   - `on_tick` (timer-based background work)
   - Shared memory for cross-filter state
   - HTTP/gRPC callout support
   - Sandboxed execution with resource limits

**Wasm Plugin Lifecycle**:
```
Create VM -> Load Module -> Create Context -> on_vm_start -> on_configure
  -> per-request: on_request_headers -> on_request_body
  -> on_response_headers -> on_response_body -> on_log
```

**Filter Ordering**:
Specified explicitly in Envoy's YAML configuration. Order matters significantly --
the router filter must be last.

### 2.3 Traefik Middleware / Plugin Catalog

Traefik's plugin system is more constrained but developer-friendly:

**Middleware Pattern**:
Traefik middleware wraps HTTP handlers in a chain pattern (similar to Go's
`http.Handler` wrapping). Built-in middleware includes rate limiting, circuit
breaker, retry, headers, IP allowlist, etc.

**Plugin Catalog (Yaegi)**:
Traefik supports dynamic plugins via Yaegi (a Go interpreter):
- Plugins are Go source code fetched from GitHub at startup
- Interpreted at runtime (no compilation needed)
- Limited to a subset of Go standard library
- Plugin catalog at `https://plugins.traefik.io/`
- Plugins declare a `New()` constructor and implement `http.Handler` interface

**Configuration**:
```yaml
experimental:
  plugins:
    my-plugin:
      moduleName: github.com/user/my-plugin
      version: v1.0.0

http:
  middlewares:
    my-middleware:
      plugin:
        my-plugin:
          headers:
            X-Custom: "value"
```

**Limitations**:
- Yaegi interpretation is slower than compiled Go
- Limited standard library access (security sandboxing)
- No Wasm support (as of 2025)
- Plugins must be available at startup (no hot-reload)

### 2.4 AI Gateway Plugin Patterns

AI-specific gateways have unique extensibility needs:

**LiteLLM Callback System**:
LiteLLM provides a `CustomLogger` base class with hooks at:
- `log_pre_api_call(model, messages, kwargs)` -- before LLM call
- `log_success_event(kwargs, response_obj, start_time, end_time)` -- after success
- `log_failure_event(kwargs, response_obj, start_time, end_time)` -- after failure
- `async_log_pre_api_call` / `async_log_success_event` / `async_log_failure_event`
  (async variants for proxy mode)
- `log_stream_event` -- per-chunk for streaming responses
- `async_post_call_success_hook` / `async_post_call_failure_hook` -- proxy-specific

Callbacks are registered via `litellm.callbacks = [MyLogger()]`. Multiple callbacks
can be registered. No priority/ordering mechanism.

**Portkey AI Gateway**:
Portkey uses a "gateway config" pattern with composable features:
- Retry with fallback
- Load balancing
- Caching
- Custom metadata injection
- Guardrails (via hooks into request/response)
- No public plugin SDK -- extensibility is via configuration

**Helicone**:
Helicone operates as a proxy layer with integration hooks:
- Custom properties on requests (metadata injection)
- User tracking
- Caching
- Rate limiting
- Moderation (content filtering)
- Extensibility via HTTP header-based configuration, not a plugin SDK

**Key Observation**: Most AI gateways lack a formal plugin ecosystem. They either provide
hooks/callbacks (LiteLLM) or configuration-driven features (Portkey, Helicone). RouteIQ's
approach of a formal plugin framework with lifecycle management is more advanced than
all current AI gateway competitors.

### 2.5 Plugin Sandboxing and Isolation Patterns

**WebAssembly (Wasm) Isolation**:
- Memory isolation: Each Wasm module has its own linear memory
- System call restriction: No direct filesystem/network access
- Capability-based: Host explicitly grants capabilities (HTTP callout, shared memory)
- Resource limits: CPU instruction counting, memory caps
- Used by: Envoy, APISIX, Istio, WasmCloud

**Process Isolation**:
- Envoy ext_proc: Plugin runs as a separate gRPC service
- Kong: Plugin runs in Nginx worker process (shared memory for isolation)
- Separate process provides strongest isolation but highest latency

**Language-Level Sandboxing**:
- Traefik Yaegi: Go interpreter with restricted standard library
- OpenResty/Kong: LuaJIT sandbox with restricted globals
- Python `RestrictedPython`: Limited but breakable isolation

**Comparison Table**:

| Approach | Isolation | Performance | Languages | Hot-Reload |
|----------|-----------|-------------|-----------|------------|
| In-process (Python) | Weak | Best | Python only | Possible |
| Wasm | Strong | Good | Multi-language | Yes |
| ext_proc (gRPC) | Strong | Higher latency | Any | Yes |
| Yaegi (interpreted) | Medium | Moderate | Go only | At startup |
| Subprocess | Strong | Highest latency | Any | Yes |

### 2.6 Plugin Marketplace and Registry Patterns

**Kong Hub**:
- Central registry at `https://docs.konghq.com/hub/`
- Categories: Authentication, Security, Traffic Control, Analytics, Transformations
- Each plugin has: README, configuration reference, changelog, compatibility matrix
- Trust model: Kong-verified vs. community plugins

**Envoy Extensions**:
- No central marketplace; extensions are compiled into Envoy binary
- Wasm plugins distributed as OCI (container) images
- `GetEnvoy` provided a plugin framework (now deprecated)

**Traefik Plugin Catalog**:
- Hosted at `https://plugins.traefik.io/`
- Plugins reference GitHub repos with specific Go module versions
- Quality scores, download counts, compatibility badges
- Automatic updates via semantic versioning

**Common Registry Metadata**:
```json
{
  "name": "my-plugin",
  "version": "1.2.0",
  "author": "org/author",
  "license": "Apache-2.0",
  "gateway_version_compat": ">=1.0.0",
  "capabilities": ["middleware", "auth"],
  "config_schema": { ... },
  "checksum": "sha256:...",
  "signature": "...",
  "readme_url": "...",
  "source_url": "..."
}
```

### 2.7 Event-Driven Plugin Architectures

Two primary patterns exist for plugin communication:

**Hook-Based (Synchronous Chain)**:
- Plugins are called in order at specific hook points
- Each plugin can modify the request/response or short-circuit
- Used by: Kong, Express.js middleware, ASGI middleware
- Pros: Simple mental model, deterministic ordering
- Cons: Blocking, harder to parallelize

**Event-Bus-Based (Pub/Sub)**:
- Plugins subscribe to events (request_received, response_sent, llm_called, etc.)
- Events dispatched asynchronously
- Used by: VSCode extensions, Grafana plugins, some observability systems
- Pros: Decoupled, parallelizable, extensible without modifying core
- Cons: Harder to reason about ordering, eventual consistency

**Hybrid Pattern** (recommended for AI gateways):
- Synchronous hooks for request/response pipeline (ordering matters)
- Asynchronous event bus for observability/logging (ordering does not matter)
- This is essentially what RouteIQ has started to build with on_request/on_response
  (sync chain) and on_llm_success (async fire-and-forget)

---

## 3. RouteIQ Plugin System: Current State

### 3.1 Architecture Overview

RouteIQ's plugin system consists of three core modules:

```
                     create_app() / create_standalone_app()
                                    |
                    _load_plugins_before_routes()
                    _configure_middleware()
                    _register_routes()
                                    |
                 _run_plugin_startup(app)
                    |              |               |
            PluginManager    PluginMiddleware   PluginCallbackBridge
            (lifecycle)      (ASGI hooks)       (LiteLLM hooks)
                    |              |               |
              GatewayPlugin instances
              - evaluator
              - skills_discovery
              - upskill_evaluator
```

### 3.2 Core Components

**PluginManager** (`src/litellm_llmrouter/gateway/plugin_manager.py` -- 985 lines):
- Plugin registration and lifecycle management
- Dependency resolution via topological sort (Kahn's algorithm)
- Priority-based ordering (lower number = earlier)
- Security: allowlist enforcement, capability-based policy
- Failure modes: CONTINUE (default), ABORT, QUARANTINE
- Startup timeout (`ROUTEIQ_PLUGIN_STARTUP_TIMEOUT`, default 30s)
- Context injection (settings, logger, URL validator)
- Plugin discovery via `LLMROUTER_PLUGINS` env var
- Health check aggregation
- Singleton pattern with `reset_plugin_manager()` for testing

**PluginMiddleware** (`src/litellm_llmrouter/gateway/plugin_middleware.py` -- 317 lines):
- Pure ASGI middleware (not BaseHTTPMiddleware, preserves streaming)
- `on_request(PluginRequest)` hook: inspect/short-circuit incoming requests
- `on_response(PluginRequest, ResponseMetadata)` hook: observe responses
- Plugins called in priority order (on_request) and reverse order (on_response)
- Error isolation: plugin hook failures are caught, logged, never crash the request
- Immutable `PluginRequest` dataclass (method, path, headers, client_ip, request_id)
- `ResponseMetadata` with status_code, headers, duration_ms (no body -- streaming-safe)

**PluginCallbackBridge** (`src/litellm_llmrouter/gateway/plugin_callback_bridge.py` -- 246 lines):
- Bridges LiteLLM callback system to GatewayPlugin LLM lifecycle hooks
- `on_llm_pre_call(model, messages, kwargs)` -- before LLM API call
- `on_llm_success(model, response, kwargs)` -- after successful call
- `on_llm_failure(model, exception, kwargs)` -- after failed call
- Duck-typing integration (no CustomLogger subclass)
- Pre-call can return kwargs overrides (merged into call params)
- Error isolation: plugin failures logged, never crash the LLM call
- Duplicate registration prevention

### 3.3 Plugin Base Class

The `GatewayPlugin` ABC defines the plugin contract:

```python
class GatewayPlugin(ABC):
    @property
    def metadata(self) -> PluginMetadata:   # Optional override
    @abstractmethod
    async def startup(self, app, context)   # Required
    @abstractmethod
    async def shutdown(self, app, context)  # Required
    async def health_check(self) -> dict    # Optional
    async def on_request(self, request)     # Optional (ASGI hook)
    async def on_response(self, request, response)  # Optional (ASGI hook)
    async def on_llm_pre_call(self, model, messages, kwargs)  # Optional (LLM hook)
    async def on_llm_success(self, model, response, kwargs)   # Optional (LLM hook)
    async def on_llm_failure(self, model, exception, kwargs)   # Optional (LLM hook)
```

### 3.4 Plugin Metadata System

```python
@dataclass
class PluginMetadata:
    name: str = ""                           # Unique identifier
    version: str = "0.0.0"                   # Semver
    capabilities: set[PluginCapability]      # What the plugin provides
    depends_on: list[str] = []               # Dependency ordering
    priority: int = 1000                     # Load order (lower = earlier)
    failure_mode: FailureMode = CONTINUE     # Error handling behavior
    description: str = ""                    # Human-readable
```

**Defined Capabilities**:
- `ROUTES` -- Registers HTTP endpoints
- `ROUTING_STRATEGY` -- Custom ML routing strategy
- `TOOL_RUNTIME` -- MCP tool execution
- `EVALUATOR` -- Request/response evaluation
- `OBSERVABILITY_EXPORTER` -- Telemetry export
- `MIDDLEWARE` -- ASGI middleware
- `AUTH_PROVIDER` -- Authentication/authorization
- `STORAGE_BACKEND` -- Storage capabilities

### 3.5 Built-in Plugins

| Plugin | Capability | Priority | Description |
|--------|-----------|----------|-------------|
| `skills-discovery` | ROUTES | 500 | Well-known skills index (`/.well-known/skills/`) |
| `evaluator` (framework) | EVALUATOR | 2000 | Base class + MCP/A2A evaluation hooks |
| `upskill-evaluator` | EVALUATOR | 2000 | Reference evaluator with optional CLI integration |

### 3.6 Security Model

- **Allowlist**: `LLMROUTER_PLUGINS_ALLOWLIST` -- explicit plugin path allowlist
- **Capability Policy**: `LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES` -- restrict what plugins can do
- **SSRF Protection**: `PluginContext.validate_outbound_url` provided to plugins
- **Path Validation**: Skills plugin validates paths against traversal attacks
- **Pre-import Check**: Allowlist checked BEFORE importing plugin module (prevents code execution)

### 3.7 Test Coverage

| Test File | Tests | Coverage Focus |
|-----------|-------|----------------|
| `test_plugin_manager.py` | 30+ tests | Registration, ordering, dependencies, failure modes, allowlist, capabilities |
| `test_plugin_middleware.py` | 20+ tests | ASGI hooks, short-circuit, streaming, error isolation, ordering |
| `test_plugin_callback_bridge.py` | 15+ tests | LLM hooks, kwargs merging, error isolation, registration |

---

## 4. Hook Completeness Assessment

### 4.1 Current Hook Points

| Hook Point | Layer | Direction | Can Modify? | Can Short-Circuit? |
|-----------|-------|-----------|-------------|-------------------|
| `startup(app, context)` | Lifecycle | N/A | App routes/state | N/A |
| `shutdown(app, context)` | Lifecycle | N/A | Cleanup | N/A |
| `health_check()` | Lifecycle | N/A | Health status | N/A |
| `on_request(request)` | ASGI | Inbound | Read-only headers | Yes (PluginResponse) |
| `on_response(request, response_meta)` | ASGI | Outbound | No (observe only) | No |
| `on_llm_pre_call(model, msgs, kwargs)` | LLM | Inbound | kwargs overrides | No |
| `on_llm_success(model, response, kwargs)` | LLM | Outbound | No (observe only) | No |
| `on_llm_failure(model, exception, kwargs)` | LLM | Error | No (observe only) | No |

### 4.2 Missing Hook Points (vs. Kong/Envoy)

| Missing Hook | Kong Equivalent | Priority | Use Case |
|-------------|----------------|----------|----------|
| **Request body access** | `access` phase body read | HIGH | Content filtering, PII redaction, prompt injection detection |
| **Response body access** | `body_filter` phase | HIGH | Response filtering, content moderation, cost tracking |
| **Streaming chunk hook** | `body_filter` (chunked) | HIGH | Per-token processing, streaming guardrails |
| **Request modification** | `rewrite` phase | MEDIUM | Header injection, URL rewriting, model override |
| **Response modification** | `header_filter` phase | MEDIUM | Header injection, CORS, caching headers |
| **Pre-routing hook** | `rewrite` phase | MEDIUM | Custom routing logic, A/B testing |
| **Post-routing hook** | `balancer` phase | LOW | Routing decision telemetry (partially covered by RouterDecisionMiddleware) |
| **Error handling hook** | `error` phase | MEDIUM | Custom error responses, retry decisions |
| **Background timer** | `init_worker` timer | LOW | Periodic tasks, cache refresh, metrics flush |
| **Config change hook** | Admin API events | LOW | React to configuration changes |
| **LLM stream chunk** | LiteLLM `log_stream_event` | HIGH | Token-level processing, streaming cost tracking |

### 4.3 Body Access Design Challenge

The current architecture deliberately avoids body access in plugin hooks to preserve
streaming performance. This is a correct default, but many plugin use cases require
body access:

**Kong's approach**: `body_filter` phase is called for each body chunk, allowing
streaming-safe transformation. The `access` phase can optionally buffer the request
body with `kong.request.get_raw_body()`.

**Recommended RouteIQ approach**: Add opt-in body access hooks:
- `on_request_body(request, body_bytes)` -- only called if plugin declares
  `needs_request_body = True` in metadata
- `on_response_chunk(request, chunk, is_final)` -- streaming-safe per-chunk hook
- Plugins that don't declare body needs get the fast path (current behavior)

---

## 5. Gap Analysis vs. Mature Gateways

### 5.1 Feature Comparison Matrix

| Feature | Kong | Envoy | Traefik | LiteLLM | RouteIQ |
|---------|------|-------|---------|---------|---------|
| **Plugin lifecycle** | init/access/log | filter chain | middleware chain | callbacks | startup/shutdown + hooks |
| **Hook phases** | 9 phases | 5 filter points | 1 (handler wrap) | 6 callbacks | 8 hooks |
| **Dependency resolution** | No (priority only) | No (explicit order) | No | No | Yes (topological sort) |
| **Priority ordering** | Yes (numeric) | Yes (explicit) | Yes (chain order) | No | Yes (numeric) |
| **Failure modes** | Per-plugin | Circuit breaker | No | No | Yes (continue/abort/quarantine) |
| **Config schema validation** | Yes (schema.lua) | Yes (protobuf) | Yes (Go struct) | No | No |
| **Request body access** | Yes | Yes | Yes | Partial (kwargs) | No |
| **Response body access** | Yes (body_filter) | Yes | Yes | Yes (response_obj) | No (metadata only) |
| **Streaming support** | Yes (chunked) | Yes (per-chunk) | No | Partial | Yes (preserves streaming) |
| **Hot-reload plugins** | Yes (DB-backed) | Yes (xDS/Wasm) | No | No | No |
| **Plugin SDK/PDK** | Yes (comprehensive) | Yes (Wasm ABI) | Partial | No | Partial (base class) |
| **Marketplace/registry** | Kong Hub | No (OCI images) | Plugin Catalog | No | No |
| **Sandboxing** | Lua sandbox | Wasm sandbox | Yaegi sandbox | None | None (in-process) |
| **Multi-language** | Lua, Go | C++, Wasm (multi) | Go | Python | Python |
| **Testing framework** | Yes (pongo) | Yes (integration) | No | No | Yes (conftest patterns) |
| **Health check hook** | No | Yes (health check filter) | No | No | Yes |
| **Plugin-specific metrics** | Yes (PDK) | Yes (stats) | No | No | No |
| **Cross-plugin state** | Yes (kong.ctx.shared) | Yes (metadata) | No | No | No (app.state possible) |
| **Security policy** | RBAC, mTLS | RBAC | No | No | Allowlist + capability policy |
| **Plugin documentation** | Extensive | Extensive | Moderate | Minimal | Docstrings only |

### 5.2 Strengths of RouteIQ's System

1. **Dependency Resolution**: RouteIQ is the only gateway in this comparison with
   built-in topological sort for plugin dependencies. Kong and Envoy rely on manual
   ordering.

2. **Failure Modes**: The CONTINUE/ABORT/QUARANTINE system is more sophisticated than
   any competitor. Kong plugins either work or crash the request. Envoy filters can
   have local/remote failure modes but nothing as granular.

3. **AI-Specific Hooks**: The LLM lifecycle hooks (pre_call, success, failure) are
   unique to AI gateways. No general-purpose gateway has these.

4. **Streaming-Safe Design**: The deliberate decision to pass only ResponseMetadata
   (no body) in on_response is correct for a streaming-first architecture.

5. **Security-First**: Pre-import allowlist checking and capability-based policy is
   more restrictive than most gateways. SSRF protection via PluginContext is unique.

6. **Test Patterns**: The autouse fixture pattern with `reset_*()` functions for
   singletons is well-designed for testing plugin interactions.

### 5.3 Weaknesses / Gaps

1. **No Request/Response Body Access**: This is the biggest functional gap. Cannot
   implement content filtering, PII redaction, or prompt injection detection as plugins.

2. **No Plugin Configuration Schema**: Plugins read their own env vars. No validation,
   no typed config, no admin API for plugin config. Kong's schema.lua is the standard.

3. **No Hot-Reload**: Plugins are loaded at startup and cannot be updated without
   restart. Kong supports DB-backed plugin config changes; Envoy supports Wasm module
   hot-swap via xDS.

4. **No Streaming Chunk Hook**: Cannot process individual SSE chunks in streaming
   responses. This is critical for token-level guardrails, cost tracking, and content
   filtering.

5. **No Cross-Plugin Communication**: No shared context or event bus for plugins to
   communicate. Kong has `kong.ctx.shared`; Envoy has request metadata.

6. **No Plugin-Specific Metrics**: Plugins cannot easily emit custom metrics. Need a
   metrics API similar to Kong's PDK `kong.log.serialize()`.

7. **No Multi-Language Support**: Python-only. Wasm would enable Rust, Go, and other
   languages.

8. **No Plugin SDK Tooling**: No CLI for scaffolding, testing, or packaging plugins.

9. **In-Process Only**: All plugins run in the same Python process. No isolation beyond
   try/except error boundaries.

---

## 6. Architecture Recommendations

### 6.1 Phase 2: Body Access and Streaming Hooks (HIGH PRIORITY)

**Goal**: Enable content filtering, guardrails, and cost tracking plugins.

**New hooks to add to GatewayPlugin**:

```python
class GatewayPlugin(ABC):
    # Existing hooks...

    # --- Phase 2 hooks ---

    async def on_request_body(
        self, request: PluginRequest, body: bytes
    ) -> bytes | None:
        """
        Called with the full request body (buffered).
        Only invoked if metadata declares needs_request_body = True.

        Returns:
            None to pass through unchanged, or modified body bytes.
        """
        return None

    async def on_response_chunk(
        self, request: PluginRequest, chunk: bytes, is_final: bool
    ) -> bytes | None:
        """
        Called for each response body chunk (streaming-safe).
        Only invoked if metadata declares needs_response_body = True.

        Returns:
            None to pass through unchanged, or modified chunk bytes.
        """
        return None

    async def on_llm_stream_chunk(
        self, model: str, chunk: Any, kwargs: dict
    ) -> None:
        """
        Called for each streaming SSE chunk from an LLM response.
        Observe-only (cannot modify chunks in flight).
        """
        pass
```

**Metadata extension**:
```python
@dataclass
class PluginMetadata:
    # Existing fields...
    needs_request_body: bool = False   # Opt-in to request body buffering
    needs_response_body: bool = False  # Opt-in to response chunk processing
```

**Implementation notes**:
- Request body buffering should only happen when at least one active plugin declares
  `needs_request_body = True`. Otherwise, the body flows through unbuffered.
- Response chunks should be passed through without buffering. If a plugin returns
  modified bytes, those are forwarded; otherwise the original chunk passes through.
- `on_llm_stream_chunk` bridges to LiteLLM's `log_stream_event` callback.

### 6.2 Phase 3: Plugin Configuration Schema (HIGH PRIORITY)

**Goal**: Typed, validated plugin configuration with admin API support.

**Design**:

```python
from pydantic import BaseModel

class GatewayPlugin(ABC):
    # Existing...

    @classmethod
    def config_schema(cls) -> type[BaseModel] | None:
        """
        Return a Pydantic model class for plugin configuration.
        None means no configuration (env-var only).
        """
        return None

    @property
    def config(self) -> BaseModel | None:
        """Access the validated plugin configuration."""
        return self._config  # Set by PluginManager during loading
```

**Example plugin with config**:
```python
class RateLimitConfig(BaseModel):
    requests_per_minute: int = 60
    burst_size: int = 10
    key_by: Literal["ip", "api_key", "user"] = "api_key"

class RateLimitPlugin(GatewayPlugin):
    @classmethod
    def config_schema(cls):
        return RateLimitConfig

    async def on_request(self, request):
        # self.config is a validated RateLimitConfig instance
        limit = self.config.requests_per_minute
        ...
```

**Configuration sources** (in priority order):
1. Plugin-specific config file: `config/plugins/<plugin-name>.yaml`
2. Environment variables: `ROUTEIQ_PLUGIN_<PLUGIN_NAME>_<FIELD>=value`
3. Admin API: `PUT /admin/plugins/<name>/config` (for hot-reload)
4. Defaults from Pydantic model

**Admin API endpoints**:
- `GET /admin/plugins` -- List all plugins with metadata and config
- `GET /admin/plugins/<name>` -- Plugin details, config, health
- `PUT /admin/plugins/<name>/config` -- Update plugin config (hot-reload)
- `GET /admin/plugins/<name>/schema` -- JSON Schema for plugin config

### 6.3 Phase 4: Plugin SDK and Developer Toolkit (MEDIUM PRIORITY)

**Goal**: Make it easy for external developers to create, test, and distribute plugins.

**Plugin SDK components**:

1. **`routeiq-plugin-sdk` package**:
   - `GatewayPlugin` base class (re-export)
   - `PluginTestHarness` for unit testing plugins
   - `PluginRequest.mock()` / `PluginResponse.mock()` test helpers
   - `MockPluginContext` with pre-configured test settings
   - Type stubs for IDE autocomplete

2. **CLI scaffolding** (`routeiq plugin new`):
   ```
   routeiq plugin new my-guardrails
   # Creates:
   #   my-guardrails/
   #     __init__.py
   #     plugin.py         # GatewayPlugin subclass
   #     config.py          # Pydantic config model
   #     tests/
   #       test_plugin.py   # Pre-wired test template
   #     pyproject.toml     # Package metadata
   #     README.md          # Plugin documentation
   ```

3. **Plugin Test Harness**:
   ```python
   from routeiq_plugin_sdk.testing import PluginTestHarness

   async def test_my_plugin_blocks_harmful_content():
       harness = PluginTestHarness(MyGuardrailPlugin())
       await harness.startup()

       response = await harness.send_request(
           method="POST",
           path="/v1/chat/completions",
           body={"messages": [{"role": "user", "content": "harmful content"}]},
       )

       assert response.status_code == 403
       assert response.body["error"] == "content_blocked"
   ```

4. **Plugin documentation generator**:
   - Auto-generate markdown from plugin metadata + config schema
   - Include hook inventory (which hooks the plugin implements)
   - Generate OpenAPI spec additions for route-providing plugins

### 6.4 Phase 5: Cross-Plugin State and Event Bus (MEDIUM PRIORITY)

**Goal**: Enable plugins to share state and communicate without tight coupling.

**Shared Context**:
```python
@dataclass
class PluginContext:
    # Existing fields...
    shared: dict[str, Any]  # Cross-plugin shared state (namespaced by plugin name)
```

Usage:
```python
# In auth plugin:
async def on_request(self, request):
    user = authenticate(request.headers.get("authorization"))
    self.context.shared["auth"] = {"user_id": user.id, "roles": user.roles}

# In rate-limit plugin (depends on auth):
async def on_request(self, request):
    auth = self.context.shared.get("auth", {})
    user_id = auth.get("user_id", "anonymous")
    # Rate limit by user_id
```

**Event Bus** (for async/observability plugins):
```python
class PluginEventBus:
    async def emit(self, event: str, data: dict) -> None:
        """Emit an event to all subscribed plugins."""

    def subscribe(self, event: str, handler: Callable) -> None:
        """Subscribe to an event type."""
```

Events: `request.received`, `response.sent`, `llm.call.started`, `llm.call.completed`,
`plugin.error`, `config.changed`, `health.degraded`.

### 6.5 Phase 6: Plugin Hot-Reload (MEDIUM PRIORITY)

**Goal**: Update plugin configuration and even plugin code without full restart.

**Configuration hot-reload** (simpler, implement first):
- Watch `config/plugins/*.yaml` for changes (reuse existing `hot_reload.py` pattern)
- Re-validate config against schema
- Call `plugin.on_config_changed(old_config, new_config)` hook
- Admin API trigger: `POST /admin/plugins/<name>/reload`

**Code hot-reload** (complex, implement later):
- Reimport plugin module
- Create new plugin instance
- Run startup on new instance
- Swap in new instance atomically
- Run shutdown on old instance
- Risk: module state leaks, import side effects

**Recommended approach**: Configuration hot-reload first. Code hot-reload only via
full gateway restart (which is fast with container orchestration).

### 6.6 Phase 7: Wasm Plugin Support (LOW PRIORITY, EXPLORATORY)

**Goal**: Enable multi-language plugins with strong isolation.

**Feasibility Assessment**:

Python Wasm runtimes are maturing:
- `wasmtime-py`: Production-ready Python bindings for Wasmtime
- `wasmer-python`: Alternative runtime
- Both support WASI (filesystem, networking capabilities)

**Proposed Architecture**:
```
GatewayPlugin (Python)
    |
WasmPluginHost (Python wrapper)
    |
wasmtime Runtime
    |
Wasm Module (Rust/Go/C++ compiled)
    |
proxy-wasm ABI (on_request_headers, on_request_body, etc.)
```

**Challenges**:
- Latency: Each Wasm call has ~10-50us overhead (acceptable for per-request hooks)
- Memory: Each Wasm instance uses ~1-10MB
- Complexity: Significant development effort for the host bridge
- Ecosystem: Python gateway + Wasm is an unusual combination

**Recommendation**: Defer Wasm to Phase 7 or later. Focus on the Python plugin
ecosystem first, which serves the primary user base. If multi-language support
becomes a requirement, ext_proc-style gRPC sidecar plugins may be simpler.

### 6.7 Plugin Marketplace Concept (LOW PRIORITY, VISION)

**Goal**: Discoverable, distributable, trusted plugin ecosystem.

**Registry Design**:
```yaml
# routeiq-plugins-registry/index.yaml
plugins:
  - name: routeiq-guardrails
    version: "1.2.0"
    author: routeiq-team
    license: Apache-2.0
    capabilities: [MIDDLEWARE]
    gateway_compat: ">=0.2.0"
    install: "pip install routeiq-guardrails"
    source: "https://github.com/routeiq/routeiq-guardrails"
    checksum: "sha256:abc123..."
    verified: true
    config_schema_url: "..."
    description: "Content guardrails for LLM requests and responses"
```

**Distribution**:
- PyPI packages with `routeiq-plugin-` prefix
- OCI images for containerized deployment
- Git-based for development (like Traefik plugin catalog)

**Trust Model**:
- Verified plugins: Reviewed and signed by RouteIQ team
- Community plugins: Community-contributed, code-reviewed
- Internal plugins: Organization-specific, private registry

---

## 7. Implementation Roadmap

### Phase 2: Body Access and Streaming Hooks
**Effort**: 2-3 weeks | **Priority**: HIGH | **Risk**: Medium

| Task | Effort | Notes |
|------|--------|-------|
| Add `needs_request_body` / `needs_response_body` to PluginMetadata | 1 day | |
| Implement request body buffering in PluginMiddleware | 3 days | Only when needed |
| Implement `on_request_body` hook dispatch | 2 days | |
| Implement `on_response_chunk` hook in send wrapper | 3 days | Streaming-safe |
| Bridge `on_llm_stream_chunk` to LiteLLM `log_stream_event` | 2 days | |
| Unit tests for all new hooks | 3 days | |
| Update plugin documentation | 1 day | |

### Phase 3: Plugin Configuration Schema
**Effort**: 2 weeks | **Priority**: HIGH | **Risk**: Low

| Task | Effort | Notes |
|------|--------|-------|
| Add `config_schema()` class method to GatewayPlugin | 1 day | |
| Config loading from YAML + env vars + defaults | 3 days | |
| Config validation via Pydantic | 2 days | |
| Admin API: GET/PUT plugin config | 3 days | |
| JSON Schema generation from Pydantic models | 1 day | |
| Unit tests | 2 days | |

### Phase 4: Plugin SDK
**Effort**: 2 weeks | **Priority**: MEDIUM | **Risk**: Low

| Task | Effort | Notes |
|------|--------|-------|
| Extract `routeiq-plugin-sdk` package | 2 days | |
| PluginTestHarness implementation | 3 days | |
| CLI scaffolding (`routeiq plugin new`) | 2 days | |
| Plugin documentation generator | 2 days | |
| Example plugins (guardrails, cost-tracker, cache) | 3 days | |

### Phase 5: Cross-Plugin State and Event Bus
**Effort**: 1.5 weeks | **Priority**: MEDIUM | **Risk**: Low

| Task | Effort | Notes |
|------|--------|-------|
| Add `shared` dict to PluginContext | 1 day | |
| Implement PluginEventBus | 3 days | |
| Wire event emission into existing hook points | 2 days | |
| Add `on_config_changed` hook | 1 day | |
| Unit tests | 2 days | |

### Phase 6: Plugin Hot-Reload
**Effort**: 1 week | **Priority**: MEDIUM | **Risk**: Medium

| Task | Effort | Notes |
|------|--------|-------|
| Config file watcher for plugins | 2 days | Reuse hot_reload.py |
| `on_config_changed` hook dispatch | 1 day | |
| Admin API reload trigger | 1 day | |
| Integration tests | 2 days | |

### Phase 7: Wasm Plugin Support (Exploratory)
**Effort**: 4-6 weeks | **Priority**: LOW | **Risk**: High

| Task | Effort | Notes |
|------|--------|-------|
| Prototype wasmtime-py integration | 1 week | Feasibility spike |
| Define RouteIQ-specific Wasm ABI | 1 week | Based on proxy-wasm |
| WasmPluginHost wrapper | 2 weeks | |
| Example Wasm plugin (Rust) | 1 week | |

---

## 8. Appendix: File Reference

### Core Plugin System Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/litellm_llmrouter/gateway/plugin_manager.py` | 985 | Plugin lifecycle, dependency resolution, security |
| `src/litellm_llmrouter/gateway/plugin_middleware.py` | 317 | ASGI-level request/response hooks |
| `src/litellm_llmrouter/gateway/plugin_callback_bridge.py` | 246 | LiteLLM callback bridge for LLM hooks |
| `src/litellm_llmrouter/gateway/app.py` | 506 | Plugin wiring in app factory |
| `src/litellm_llmrouter/gateway/__init__.py` | 27 | Public API exports |

### Built-in Plugins

| File | Lines | Plugin |
|------|-------|--------|
| `src/litellm_llmrouter/gateway/plugins/__init__.py` | 41 | Package exports |
| `src/litellm_llmrouter/gateway/plugins/evaluator.py` | 400 | Evaluator framework (base class + hooks) |
| `src/litellm_llmrouter/gateway/plugins/skills_discovery.py` | 509 | Skills discovery endpoints |
| `src/litellm_llmrouter/gateway/plugins/upskill_evaluator.py` | 423 | Reference evaluator implementation |

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_plugin_manager.py` | 30+ | Registration, ordering, deps, failure modes, security |
| `tests/unit/test_plugin_middleware.py` | 20+ | ASGI hooks, short-circuit, streaming, error isolation |
| `tests/unit/test_plugin_callback_bridge.py` | 15+ | LLM hooks, kwargs merging, error isolation |

### Configuration Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLMROUTER_PLUGINS` | (empty) | Comma-separated plugin module paths |
| `LLMROUTER_PLUGINS_ALLOWLIST` | (none) | Allowed plugin paths |
| `LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES` | (none) | Allowed capability types |
| `LLMROUTER_PLUGINS_FAILURE_MODE` | `continue` | Global default failure mode |
| `ROUTEIQ_PLUGIN_STARTUP_TIMEOUT` | `30` | Plugin startup timeout (seconds) |
| `ROUTEIQ_PLUGIN_*` | (varies) | Plugin-specific settings (prefix-stripped) |
| `ROUTEIQ_EVALUATOR_ENABLED` | `false` | Enable evaluator hooks |
| `ROUTEIQ_SKILLS_DIR` | `./skills` | Skills discovery directory |

---

*Report generated from codebase analysis at commit range up to 994a40f on main branch.*
*Industry comparison based on Kong Gateway 3.x, Envoy 1.30+, Traefik 3.x, LiteLLM 1.x documentation.*

# Architecture Review 08: Plugin Ecosystem Analysis

**Date**: 2026-02-07
**Scope**: Plugin/extension system architecture for RouteIQ AI Gateway
**Status**: Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Industry Reference: Gateway Plugin Architectures](#2-industry-reference-gateway-plugin-architectures)
3. [RouteIQ Plugin System: Current State](#3-routeiq-plugin-system-current-state)
4. [Hook Completeness Assessment](#4-hook-completeness-assessment)
5. [Gap Analysis vs. Mature Gateways](#5-gap-analysis-vs-mature-gateways)
6. [Architecture Recommendations](#6-architecture-recommendations)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Appendix: File Reference](#8-appendix-file-reference)

---

## 1. Executive Summary

RouteIQ has completed Phase 1 of its plugin system, delivering a solid foundation with
ASGI-level request/response hooks (PluginMiddleware), LiteLLM callback hooks
(PluginCallbackBridge), lifecycle management with dependency resolution, and three
built-in plugins. The system is well-tested (65+ test assertions across three test files)
and production-oriented with failure modes, quarantine, allowlist security, and
capability-based policy.

However, compared to mature gateway plugin ecosystems (Kong, Envoy, Traefik), RouteIQ's
plugin system has notable gaps in: configuration schema validation, plugin hot-reload,
body access hooks, streaming transformation, SDK tooling, and marketplace/distribution
infrastructure. This report catalogs these gaps and provides a prioritized roadmap.

**Overall Maturity Rating**: Phase 1 Complete (Foundation) -- approximately 40% of a
mature gateway plugin ecosystem.

---

## 2. Industry Reference: Gateway Plugin Architectures

### 2.1 Kong Gateway Plugin Model

Kong is the gold standard for API gateway plugin architecture. Key design elements:

**Phase-Based Execution Model**:
Kong plugins execute at specific phases in the request lifecycle, each with distinct
semantics:

| Phase | When | Typical Use |
|-------|------|-------------|
| `init_worker` | Worker process start | Background timers, global state |
| `certificate` | TLS handshake | mTLS validation |
| `rewrite` | Before routing | URL rewriting, header manipulation |
| `access` | After routing, before upstream | Authentication, authorization, rate limiting |
| `response` | Streaming response from upstream | Response transformation (buffered) |
| `header_filter` | Response headers received | Header manipulation |
| `body_filter` | Each response body chunk | Body transformation (streaming-safe) |
| `log` | After response sent | Logging, analytics, metrics |
| `error` | Error occurred | Custom error responses |

**Plugin Development Kit (PDK)**:
Kong provides `kong.pdk` -- a stable API surface that abstracts Nginx internals:
- `kong.request.*` - Read request headers, body, query params
- `kong.response.*` - Set response headers, status, body
- `kong.service.request.*` - Modify upstream request
- `kong.service.response.*` - Read upstream response
- `kong.ctx.shared` - Cross-plugin shared context
- `kong.log.*` - Structured logging
- `kong.node.*` - Node information
- `kong.ip.*` - IP utilities
- `kong.client.*` - Client certificate info

**Plugin Configuration**:
Each plugin has a `schema.lua` that defines typed configuration with validation:
```lua
return {
  name = "rate-limiting",
  fields = {
    { config = {
        type = "record",
        fields = {
          { second = { type = "number", gt = 0 } },
          { minute = { type = "number", gt = 0 } },
          { policy = { type = "string", default = "local",
                       one_of = { "local", "cluster", "redis" } } },
        },
    }},
  },
}
```

**Plugin Ordering/Priority**:
Kong uses numeric priority (higher = runs first in access phase, lower = runs first in
log phase). Default bundled plugin priorities are well-documented, allowing third-party
plugins to insert at precise points.

**Distribution**:
Kong Hub hosts 100+ plugins. Plugins are distributed as LuaRocks packages with
`kong-plugin-<name>` naming convention, or as Docker layers for custom builds.

### 2.2 Envoy Proxy Filter Chain

Envoy's extensibility model is the most sophisticated in the industry:

**HTTP Filter Chain**:
Envoy processes requests through an ordered chain of HTTP filters. Each filter can:
- Decode (modify request going upstream)
- Encode (modify response going downstream)
- Both (bidirectional)

Filter types: `decoder_filter`, `encoder_filter`, `dual_filter`, `access_log`.

**Extension Mechanisms** (in order of maturity):

1. **Native C++ Filters**: Highest performance, compiled into Envoy binary. Used for
   core functionality (router, RBAC, rate limit).

2. **Lua Filters**: Inline scripting via `envoy.filters.http.lua`. Good for simple
   request/response transformations. Sandboxed with limited API.

3. **ext_proc (External Processing)**: gRPC-based external processing service. Envoy
   sends request/response data to an external gRPC service for decisions. Supports:
   - Request header processing
   - Request body processing (buffered or streamed)
   - Response header processing
   - Response body processing (buffered or streamed)
   - Immediate response (short-circuit)

4. **WebAssembly (Wasm) Filters**: Plugins compiled to Wasm, loaded at runtime.
   Supported languages: Rust, Go, C++, AssemblyScript. Uses the proxy-wasm ABI:
   - `on_request_headers`, `on_request_body`
   - `on_response_headers`, `on_response_body`
   - `on_log`
   - `on_tick` (timer-based background work)
   - Shared memory for cross-filter state
   - HTTP/gRPC callout support
   - Sandboxed execution with resource limits

**Wasm Plugin Lifecycle**:
```
Create VM -> Load Module -> Create Context -> on_vm_start -> on_configure
  -> per-request: on_request_headers -> on_request_body
  -> on_response_headers -> on_response_body -> on_log
```

**Filter Ordering**:
Specified explicitly in Envoy's YAML configuration. Order matters significantly --
the router filter must be last.

### 2.3 Traefik Middleware / Plugin Catalog

Traefik's plugin system is more constrained but developer-friendly:

**Middleware Pattern**:
Traefik middleware wraps HTTP handlers in a chain pattern (similar to Go's
`http.Handler` wrapping). Built-in middleware includes rate limiting, circuit
breaker, retry, headers, IP allowlist, etc.

**Plugin Catalog (Yaegi)**:
Traefik supports dynamic plugins via Yaegi (a Go interpreter):
- Plugins are Go source code fetched from GitHub at startup
- Interpreted at runtime (no compilation needed)
- Limited to a subset of Go standard library
- Plugin catalog at `https://plugins.traefik.io/`
- Plugins declare a `New()` constructor and implement `http.Handler` interface

**Configuration**:
```yaml
experimental:
  plugins:
    my-plugin:
      moduleName: github.com/user/my-plugin
      version: v1.0.0

http:
  middlewares:
    my-middleware:
      plugin:
        my-plugin:
          headers:
            X-Custom: "value"
```

**Limitations**:
- Yaegi interpretation is slower than compiled Go
- Limited standard library access (security sandboxing)
- No Wasm support (as of 2025)
- Plugins must be available at startup (no hot-reload)

### 2.4 AI Gateway Plugin Patterns

AI-specific gateways have unique extensibility needs:

**LiteLLM Callback System**:
LiteLLM provides a `CustomLogger` base class with hooks at:
- `log_pre_api_call(model, messages, kwargs)` -- before LLM call
- `log_success_event(kwargs, response_obj, start_time, end_time)` -- after success
- `log_failure_event(kwargs, response_obj, start_time, end_time)` -- after failure
- `async_log_pre_api_call` / `async_log_success_event` / `async_log_failure_event`
  (async variants for proxy mode)
- `log_stream_event` -- per-chunk for streaming responses
- `async_post_call_success_hook` / `async_post_call_failure_hook` -- proxy-specific

Callbacks are registered via `litellm.callbacks = [MyLogger()]`. Multiple callbacks
can be registered. No priority/ordering mechanism.

**Portkey AI Gateway**:
Portkey uses a "gateway config" pattern with composable features:
- Retry with fallback
- Load balancing
- Caching
- Custom metadata injection
- Guardrails (via hooks into request/response)
- No public plugin SDK -- extensibility is via configuration

**Helicone**:
Helicone operates as a proxy layer with integration hooks:
- Custom properties on requests (metadata injection)
- User tracking
- Caching
- Rate limiting
- Moderation (content filtering)
- Extensibility via HTTP header-based configuration, not a plugin SDK

**Key Observation**: Most AI gateways lack a formal plugin ecosystem. They either provide
hooks/callbacks (LiteLLM) or configuration-driven features (Portkey, Helicone). RouteIQ's
approach of a formal plugin framework with lifecycle management is more advanced than
all current AI gateway competitors.

### 2.5 Plugin Sandboxing and Isolation Patterns

**WebAssembly (Wasm) Isolation**:
- Memory isolation: Each Wasm module has its own linear memory
- System call restriction: No direct filesystem/network access
- Capability-based: Host explicitly grants capabilities (HTTP callout, shared memory)
- Resource limits: CPU instruction counting, memory caps
- Used by: Envoy, APISIX, Istio, WasmCloud

**Process Isolation**:
- Envoy ext_proc: Plugin runs as a separate gRPC service
- Kong: Plugin runs in Nginx worker process (shared memory for isolation)
- Separate process provides strongest isolation but highest latency

**Language-Level Sandboxing**:
- Traefik Yaegi: Go interpreter with restricted standard library
- OpenResty/Kong: LuaJIT sandbox with restricted globals
- Python `RestrictedPython`: Limited but breakable isolation

**Comparison Table**:

| Approach | Isolation | Performance | Languages | Hot-Reload |
|----------|-----------|-------------|-----------|------------|
| In-process (Python) | Weak | Best | Python only | Possible |
| Wasm | Strong | Good | Multi-language | Yes |
| ext_proc (gRPC) | Strong | Higher latency | Any | Yes |
| Yaegi (interpreted) | Medium | Moderate | Go only | At startup |
| Subprocess | Strong | Highest latency | Any | Yes |

### 2.6 Plugin Marketplace and Registry Patterns

**Kong Hub**:
- Central registry at `https://docs.konghq.com/hub/`
- Categories: Authentication, Security, Traffic Control, Analytics, Transformations
- Each plugin has: README, configuration reference, changelog, compatibility matrix
- Trust model: Kong-verified vs. community plugins

**Envoy Extensions**:
- No central marketplace; extensions are compiled into Envoy binary
- Wasm plugins distributed as OCI (container) images
- `GetEnvoy` provided a plugin framework (now deprecated)

**Traefik Plugin Catalog**:
- Hosted at `https://plugins.traefik.io/`
- Plugins reference GitHub repos with specific Go module versions
- Quality scores, download counts, compatibility badges
- Automatic updates via semantic versioning

**Common Registry Metadata**:
```json
{
  "name": "my-plugin",
  "version": "1.2.0",
  "author": "org/author",
  "license": "Apache-2.0",
  "gateway_version_compat": ">=1.0.0",
  "capabilities": ["middleware", "auth"],
  "config_schema": { "..." : "..." },
  "checksum": "sha256:...",
  "signature": "...",
  "readme_url": "...",
  "source_url": "..."
}
```

### 2.7 Event-Driven Plugin Architectures

Two primary patterns exist for plugin communication:

**Hook-Based (Synchronous Chain)**:
- Plugins are called in order at specific hook points
- Each plugin can modify the request/response or short-circuit
- Used by: Kong, Express.js middleware, ASGI middleware
- Pros: Simple mental model, deterministic ordering
- Cons: Blocking, harder to parallelize

**Event-Bus-Based (Pub/Sub)**:
- Plugins subscribe to events (request_received, response_sent, llm_called, etc.)
- Events dispatched asynchronously
- Used by: VSCode extensions, Grafana plugins, some observability systems
- Pros: Decoupled, parallelizable, extensible without modifying core
- Cons: Harder to reason about ordering, eventual consistency

**Hybrid Pattern** (recommended for AI gateways):
- Synchronous hooks for request/response pipeline (ordering matters)
- Asynchronous event bus for observability/logging (ordering does not matter)
- This is essentially what RouteIQ has started to build with on_request/on_response
  (sync chain) and on_llm_success (async fire-and-forget)

---

## 3. RouteIQ Plugin System: Current State

### 3.1 Architecture Overview

RouteIQ's plugin system consists of three core modules:

```
                     create_app() / create_standalone_app()
                                    |
                    _load_plugins_before_routes()
                    _configure_middleware()
                    _register_routes()
                                    |
                 _run_plugin_startup(app)
                    |              |               |
            PluginManager    PluginMiddleware   PluginCallbackBridge
            (lifecycle)      (ASGI hooks)       (LiteLLM hooks)
                    |              |               |
              GatewayPlugin instances
              - evaluator
              - skills_discovery
              - upskill_evaluator
```

### 3.2 Core Components

**PluginManager** (`src/litellm_llmrouter/gateway/plugin_manager.py` -- 985 lines):
- Plugin registration and lifecycle management
- Dependency resolution via topological sort (Kahn's algorithm)
- Priority-based ordering (lower number = earlier)
- Security: allowlist enforcement, capability-based policy
- Failure modes: CONTINUE (default), ABORT, QUARANTINE
- Startup timeout (`ROUTEIQ_PLUGIN_STARTUP_TIMEOUT`, default 30s)
- Context injection (settings, logger, URL validator)
- Plugin discovery via `LLMROUTER_PLUGINS` env var
- Health check aggregation
- Singleton pattern with `reset_plugin_manager()` for testing

**PluginMiddleware** (`src/litellm_llmrouter/gateway/plugin_middleware.py` -- 317 lines):
- Pure ASGI middleware (not BaseHTTPMiddleware, preserves streaming)
- `on_request(PluginRequest)` hook: inspect/short-circuit incoming requests
- `on_response(PluginRequest, ResponseMetadata)` hook: observe responses
- Plugins called in priority order (on_request) and reverse order (on_response)
- Error isolation: plugin hook failures are caught, logged, never crash the request
- Immutable `PluginRequest` dataclass (method, path, headers, client_ip, request_id)
- `ResponseMetadata` with status_code, headers, duration_ms (no body -- streaming-safe)

**PluginCallbackBridge** (`src/litellm_llmrouter/gateway/plugin_callback_bridge.py` -- 246 lines):
- Bridges LiteLLM callback system to GatewayPlugin LLM lifecycle hooks
- `on_llm_pre_call(model, messages, kwargs)` -- before LLM API call
- `on_llm_success(model, response, kwargs)` -- after successful call
- `on_llm_failure(model, exception, kwargs)` -- after failed call
- Duck-typing integration (no CustomLogger subclass)
- Pre-call can return kwargs overrides (merged into call params)
- Error isolation: plugin failures logged, never crash the LLM call
- Duplicate registration prevention

### 3.3 Plugin Base Class

The `GatewayPlugin` ABC defines the plugin contract:

```python
class GatewayPlugin(ABC):
    @property
    def metadata(self) -> PluginMetadata:   # Optional override
    @abstractmethod
    async def startup(self, app, context)   # Required
    @abstractmethod
    async def shutdown(self, app, context)  # Required
    async def health_check(self) -> dict    # Optional
    async def on_request(self, request)     # Optional (ASGI hook)
    async def on_response(self, request, response)  # Optional (ASGI hook)
    async def on_llm_pre_call(self, model, messages, kwargs)  # Optional (LLM hook)
    async def on_llm_success(self, model, response, kwargs)   # Optional (LLM hook)
    async def on_llm_failure(self, model, exception, kwargs)   # Optional (LLM hook)
```

### 3.4 Plugin Metadata System

```python
@dataclass
class PluginMetadata:
    name: str = ""                           # Unique identifier
    version: str = "0.0.0"                   # Semver
    capabilities: set[PluginCapability]      # What the plugin provides
    depends_on: list[str] = []               # Dependency ordering
    priority: int = 1000                     # Load order (lower = earlier)
    failure_mode: FailureMode = CONTINUE     # Error handling behavior
    description: str = ""                    # Human-readable
```

**Defined Capabilities**:
- `ROUTES` -- Registers HTTP endpoints
- `ROUTING_STRATEGY` -- Custom ML routing strategy
- `TOOL_RUNTIME` -- MCP tool execution
- `EVALUATOR` -- Request/response evaluation
- `OBSERVABILITY_EXPORTER` -- Telemetry export
- `MIDDLEWARE` -- ASGI middleware
- `AUTH_PROVIDER` -- Authentication/authorization
- `STORAGE_BACKEND` -- Storage capabilities

### 3.5 Built-in Plugins

| Plugin | Capability | Priority | Description |
|--------|-----------|----------|-------------|
| `skills-discovery` | ROUTES | 500 | Well-known skills index (`/.well-known/skills/`) |
| `evaluator` (framework) | EVALUATOR | 2000 | Base class + MCP/A2A evaluation hooks |
| `upskill-evaluator` | EVALUATOR | 2000 | Reference evaluator with optional CLI integration |

### 3.6 Security Model

- **Allowlist**: `LLMROUTER_PLUGINS_ALLOWLIST` -- explicit plugin path allowlist
- **Capability Policy**: `LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES` -- restrict what plugins can do
- **SSRF Protection**: `PluginContext.validate_outbound_url` provided to plugins
- **Path Validation**: Skills plugin validates paths against traversal attacks
- **Pre-import Check**: Allowlist checked BEFORE importing plugin module (prevents code execution)

### 3.7 Test Coverage

| Test File | Tests | Coverage Focus |
|-----------|-------|----------------|
| `test_plugin_manager.py` | 30+ tests | Registration, ordering, dependencies, failure modes, allowlist, capabilities |
| `test_plugin_middleware.py` | 20+ tests | ASGI hooks, short-circuit, streaming, error isolation, ordering |
| `test_plugin_callback_bridge.py` | 15+ tests | LLM hooks, kwargs merging, error isolation, registration |

---

## 4. Hook Completeness Assessment

### 4.1 Current Hook Points

| Hook Point | Layer | Direction | Can Modify? | Can Short-Circuit? |
|-----------|-------|-----------|-------------|-------------------|
| `startup(app, context)` | Lifecycle | N/A | App routes/state | N/A |
| `shutdown(app, context)` | Lifecycle | N/A | Cleanup | N/A |
| `health_check()` | Lifecycle | N/A | Health status | N/A |
| `on_request(request)` | ASGI | Inbound | Read-only headers | Yes (PluginResponse) |
| `on_response(request, response_meta)` | ASGI | Outbound | No (observe only) | No |
| `on_llm_pre_call(model, msgs, kwargs)` | LLM | Inbound | kwargs overrides | No |
| `on_llm_success(model, response, kwargs)` | LLM | Outbound | No (observe only) | No |
| `on_llm_failure(model, exception, kwargs)` | LLM | Error | No (observe only) | No |

### 4.2 Missing Hook Points (vs. Kong/Envoy)

| Missing Hook | Kong Equivalent | Priority | Use Case |
|-------------|----------------|----------|----------|
| **Request body access** | `access` phase body read | HIGH | Content filtering, PII redaction, prompt injection detection |
| **Response body access** | `body_filter` phase | HIGH | Response filtering, content moderation, cost tracking |
| **Streaming chunk hook** | `body_filter` (chunked) | HIGH | Per-token processing, streaming guardrails |
| **Request modification** | `rewrite` phase | MEDIUM | Header injection, URL rewriting, model override |
| **Response modification** | `header_filter` phase | MEDIUM | Header injection, CORS, caching headers |
| **Pre-routing hook** | `rewrite` phase | MEDIUM | Custom routing logic, A/B testing |
| **Post-routing hook** | `balancer` phase | LOW | Routing decision telemetry (partially covered by RouterDecisionMiddleware) |
| **Error handling hook** | `error` phase | MEDIUM | Custom error responses, retry decisions |
| **Background timer** | `init_worker` timer | LOW | Periodic tasks, cache refresh, metrics flush |
| **Config change hook** | Admin API events | LOW | React to configuration changes |
| **LLM stream chunk** | LiteLLM `log_stream_event` | HIGH | Token-level processing, streaming cost tracking |

### 4.3 Body Access Design Challenge

The current architecture deliberately avoids body access in plugin hooks to preserve
streaming performance. This is a correct default, but many plugin use cases require
body access:

**Kong's approach**: `body_filter` phase is called for each body chunk, allowing
streaming-safe transformation. The `access` phase can optionally buffer the request
body with `kong.request.get_raw_body()`.

**Recommended RouteIQ approach**: Add opt-in body access hooks:
- `on_request_body(request, body_bytes)` -- only called if plugin declares
  `needs_request_body = True` in metadata
- `on_response_chunk(request, chunk, is_final)` -- streaming-safe per-chunk hook
- Plugins that don't declare body needs get the fast path (current behavior)

---

## 5. Gap Analysis vs. Mature Gateways

### 5.1 Feature Comparison Matrix

| Feature | Kong | Envoy | Traefik | LiteLLM | RouteIQ |
|---------|------|-------|---------|---------|---------|
| **Plugin lifecycle** | init/access/log | filter chain | middleware chain | callbacks | startup/shutdown + hooks |
| **Hook phases** | 9 phases | 5 filter points | 1 (handler wrap) | 6 callbacks | 8 hooks |
| **Dependency resolution** | No (priority only) | No (explicit order) | No | No | Yes (topological sort) |
| **Priority ordering** | Yes (numeric) | Yes (explicit) | Yes (chain order) | No | Yes (numeric) |
| **Failure modes** | Per-plugin | Circuit breaker | No | No | Yes (continue/abort/quarantine) |
| **Config schema validation** | Yes (schema.lua) | Yes (protobuf) | Yes (Go struct) | No | No |
| **Request body access** | Yes | Yes | Yes | Partial (kwargs) | No |
| **Response body access** | Yes (body_filter) | Yes | Yes | Yes (response_obj) | No (metadata only) |
| **Streaming support** | Yes (chunked) | Yes (per-chunk) | No | Partial | Yes (preserves streaming) |
| **Hot-reload plugins** | Yes (DB-backed) | Yes (xDS/Wasm) | No | No | No |
| **Plugin SDK/PDK** | Yes (comprehensive) | Yes (Wasm ABI) | Partial | No | Partial (base class) |
| **Marketplace/registry** | Kong Hub | No (OCI images) | Plugin Catalog | No | No |
| **Sandboxing** | Lua sandbox | Wasm sandbox | Yaegi sandbox | None | None (in-process) |
| **Multi-language** | Lua, Go | C++, Wasm (multi) | Go | Python | Python |
| **Testing framework** | Yes (pongo) | Yes (integration) | No | No | Yes (conftest patterns) |
| **Health check hook** | No | Yes (health check filter) | No | No | Yes |
| **Plugin-specific metrics** | Yes (PDK) | Yes (stats) | No | No | No |
| **Cross-plugin state** | Yes (kong.ctx.shared) | Yes (metadata) | No | No | No (app.state possible) |
| **Security policy** | RBAC, mTLS | RBAC | No | No | Allowlist + capability policy |
| **Plugin documentation** | Extensive | Extensive | Moderate | Minimal | Docstrings only |

### 5.2 Strengths of RouteIQ's System

1. **Dependency Resolution**: RouteIQ is the only gateway in this comparison with
   built-in topological sort for plugin dependencies. Kong and Envoy rely on manual
   ordering.

2. **Failure Modes**: The CONTINUE/ABORT/QUARANTINE system is more sophisticated than
   any competitor. Kong plugins either work or crash the request. Envoy filters can
   have local/remote failure modes but nothing as granular.

3. **AI-Specific Hooks**: The LLM lifecycle hooks (pre_call, success, failure) are
   unique to AI gateways. No general-purpose gateway has these.

4. **Streaming-Safe Design**: The deliberate decision to pass only ResponseMetadata
   (no body) in on_response is correct for a streaming-first architecture.

5. **Security-First**: Pre-import allowlist checking and capability-based policy is
   more restrictive than most gateways. SSRF protection via PluginContext is unique.

6. **Test Patterns**: The autouse fixture pattern with `reset_*()` functions for
   singletons is well-designed for testing plugin interactions.

### 5.3 Weaknesses / Gaps

1. **No Request/Response Body Access**: This is the biggest functional gap. Cannot
   implement content filtering, PII redaction, or prompt injection detection as plugins.

2. **No Plugin Configuration Schema**: Plugins read their own env vars. No validation,
   no typed config, no admin API for plugin config. Kong's schema.lua is the standard.

3. **No Hot-Reload**: Plugins are loaded at startup and cannot be updated without
   restart. Kong supports DB-backed plugin config changes; Envoy supports Wasm module
   hot-swap via xDS.

4. **No Streaming Chunk Hook**: Cannot process individual SSE chunks in streaming
   responses. This is critical for token-level guardrails, cost tracking, and content
   filtering.

5. **No Cross-Plugin Communication**: No shared context or event bus for plugins to
   communicate. Kong has `kong.ctx.shared`; Envoy has request metadata.

6. **No Plugin-Specific Metrics**: Plugins cannot easily emit custom metrics. Need a
   metrics API similar to Kong's PDK `kong.log.serialize()`.

7. **No Multi-Language Support**: Python-only. Wasm would enable Rust, Go, and other
   languages.

8. **No Plugin SDK Tooling**: No CLI for scaffolding, testing, or packaging plugins.

9. **In-Process Only**: All plugins run in the same Python process. No isolation beyond
   try/except error boundaries.

---

## 6. Architecture Recommendations

### 6.1 Phase 2: Body Access and Streaming Hooks (HIGH PRIORITY)

**Goal**: Enable content filtering, guardrails, and cost tracking plugins.

**New hooks to add to GatewayPlugin**:

```python
class GatewayPlugin(ABC):
    # Existing hooks...

    # --- Phase 2 hooks ---

    async def on_request_body(
        self, request: PluginRequest, body: bytes
    ) -> bytes | None:
        """
        Called with the full request body (buffered).
        Only invoked if metadata declares needs_request_body = True.

        Returns:
            None to pass through unchanged, or modified body bytes.
        """
        return None

    async def on_response_chunk(
        self, request: PluginRequest, chunk: bytes, is_final: bool
    ) -> bytes | None:
        """
        Called for each response body chunk (streaming-safe).
        Only invoked if metadata declares needs_response_body = True.

        Returns:
            None to pass through unchanged, or modified chunk bytes.
        """
        return None

    async def on_llm_stream_chunk(
        self, model: str, chunk: Any, kwargs: dict
    ) -> None:
        """
        Called for each streaming SSE chunk from an LLM response.
        Observe-only (cannot modify chunks in flight).
        """
        pass
```

**Metadata extension**:
```python
@dataclass
class PluginMetadata:
    # Existing fields...
    needs_request_body: bool = False   # Opt-in to request body buffering
    needs_response_body: bool = False  # Opt-in to response chunk processing
```

**Implementation notes**:
- Request body buffering should only happen when at least one active plugin declares
  `needs_request_body = True`. Otherwise, the body flows through unbuffered.
- Response chunks should be passed through without buffering. If a plugin returns
  modified bytes, those are forwarded; otherwise the original chunk passes through.
- `on_llm_stream_chunk` bridges to LiteLLM's `log_stream_event` callback.

### 6.2 Phase 3: Plugin Configuration Schema (HIGH PRIORITY)

**Goal**: Typed, validated plugin configuration with admin API support.

**Design**:

```python
from pydantic import BaseModel

class GatewayPlugin(ABC):
    # Existing...

    @classmethod
    def config_schema(cls) -> type[BaseModel] | None:
        """
        Return a Pydantic model class for plugin configuration.
        None means no configuration (env-var only).
        """
        return None

    @property
    def config(self) -> BaseModel | None:
        """Access the validated plugin configuration."""
        return self._config  # Set by PluginManager during loading
```

**Example plugin with config**:
```python
class RateLimitConfig(BaseModel):
    requests_per_minute: int = 60
    burst_size: int = 10
    key_by: Literal["ip", "api_key", "user"] = "api_key"

class RateLimitPlugin(GatewayPlugin):
    @classmethod
    def config_schema(cls):
        return RateLimitConfig

    async def on_request(self, request):
        # self.config is a validated RateLimitConfig instance
        limit = self.config.requests_per_minute
        ...
```

**Configuration sources** (in priority order):
1. Plugin-specific config file: `config/plugins/<plugin-name>.yaml`
2. Environment variables: `ROUTEIQ_PLUGIN_<PLUGIN_NAME>_<FIELD>=value`
3. Admin API: `PUT /admin/plugins/<name>/config` (for hot-reload)
4. Defaults from Pydantic model

**Admin API endpoints**:
- `GET /admin/plugins` -- List all plugins with metadata and config
- `GET /admin/plugins/<name>` -- Plugin details, config, health
- `PUT /admin/plugins/<name>/config` -- Update plugin config (hot-reload)
- `GET /admin/plugins/<name>/schema` -- JSON Schema for plugin config

### 6.3 Phase 4: Plugin SDK and Developer Toolkit (MEDIUM PRIORITY)

**Goal**: Make it easy for external developers to create, test, and distribute plugins.

**Plugin SDK components**:

1. **`routeiq-plugin-sdk` package**:
   - `GatewayPlugin` base class (re-export)
   - `PluginTestHarness` for unit testing plugins
   - `PluginRequest.mock()` / `PluginResponse.mock()` test helpers
   - `MockPluginContext` with pre-configured test settings
   - Type stubs for IDE autocomplete

2. **CLI scaffolding** (`routeiq plugin new`):
   ```
   routeiq plugin new my-guardrails
   # Creates:
   #   my-guardrails/
   #     __init__.py
   #     plugin.py         # GatewayPlugin subclass
   #     config.py          # Pydantic config model
   #     tests/
   #       test_plugin.py   # Pre-wired test template
   #     pyproject.toml     # Package metadata
   #     README.md          # Plugin documentation
   ```

3. **Plugin Test Harness**:
   ```python
   from routeiq_plugin_sdk.testing import PluginTestHarness

   async def test_my_plugin_blocks_harmful_content():
       harness = PluginTestHarness(MyGuardrailPlugin())
       await harness.startup()

       response = await harness.send_request(
           method="POST",
           path="/v1/chat/completions",
           body={"messages": [{"role": "user", "content": "harmful content"}]},
       )

       assert response.status_code == 403
       assert response.body["error"] == "content_blocked"
   ```

4. **Plugin documentation generator**:
   - Auto-generate markdown from plugin metadata + config schema
   - Include hook inventory (which hooks the plugin implements)
   - Generate OpenAPI spec additions for route-providing plugins

### 6.4 Phase 5: Cross-Plugin State and Event Bus (MEDIUM PRIORITY)

**Goal**: Enable plugins to share state and communicate without tight coupling.

**Shared Context**:
```python
@dataclass
class PluginContext:
    # Existing fields...
    shared: dict[str, Any]  # Cross-plugin shared state (namespaced by plugin name)
```

Usage:
```python
# In auth plugin:
async def on_request(self, request):
    user = authenticate(request.headers.get("authorization"))
    self.context.shared["auth"] = {"user_id": user.id, "roles": user.roles}

# In rate-limit plugin (depends on auth):
async def on_request(self, request):
    auth = self.context.shared.get("auth", {})
    user_id = auth.get("user_id", "anonymous")
    # Rate limit by user_id
```

**Event Bus** (for async/observability plugins):
```python
class PluginEventBus:
    async def emit(self, event: str, data: dict) -> None:
        """Emit an event to all subscribed plugins."""

    def subscribe(self, event: str, handler: Callable) -> None:
        """Subscribe to an event type."""
```

Events: `request.received`, `response.sent`, `llm.call.started`, `llm.call.completed`,
`plugin.error`, `config.changed`, `health.degraded`.

### 6.5 Phase 6: Plugin Hot-Reload (MEDIUM PRIORITY)

**Goal**: Update plugin configuration and even plugin code without full restart.

**Configuration hot-reload** (simpler, implement first):
- Watch `config/plugins/*.yaml` for changes (reuse existing `hot_reload.py` pattern)
- Re-validate config against schema
- Call `plugin.on_config_changed(old_config, new_config)` hook
- Admin API trigger: `POST /admin/plugins/<name>/reload`

**Code hot-reload** (complex, implement later):
- Reimport plugin module
- Create new plugin instance
- Run startup on new instance
- Swap in new instance atomically
- Run shutdown on old instance
- Risk: module state leaks, import side effects

**Recommended approach**: Configuration hot-reload first. Code hot-reload only via
full gateway restart (which is fast with container orchestration).

### 6.6 Phase 7: Wasm Plugin Support (LOW PRIORITY, EXPLORATORY)

**Goal**: Enable multi-language plugins with strong isolation.

**Feasibility Assessment**:

Python Wasm runtimes are maturing:
- `wasmtime-py`: Production-ready Python bindings for Wasmtime
- `wasmer-python`: Alternative runtime
- Both support WASI (filesystem, networking capabilities)

**Proposed Architecture**:
```
GatewayPlugin (Python)
    |
WasmPluginHost (Python wrapper)
    |
wasmtime Runtime
    |
Wasm Module (Rust/Go/C++ compiled)
    |
proxy-wasm ABI (on_request_headers, on_request_body, etc.)
```

**Challenges**:
- Latency: Each Wasm call has ~10-50us overhead (acceptable for per-request hooks)
- Memory: Each Wasm instance uses ~1-10MB
- Complexity: Significant development effort for the host bridge
- Ecosystem: Python gateway + Wasm is an unusual combination

**Recommendation**: Defer Wasm to Phase 7 or later. Focus on the Python plugin
ecosystem first, which serves the primary user base. If multi-language support
becomes a requirement, ext_proc-style gRPC sidecar plugins may be simpler.

### 6.7 Plugin Marketplace Concept (LOW PRIORITY, VISION)

**Goal**: Discoverable, distributable, trusted plugin ecosystem.

**Registry Design**:
```yaml
# routeiq-plugins-registry/index.yaml
plugins:
  - name: routeiq-guardrails
    version: "1.2.0"
    author: routeiq-team
    license: Apache-2.0
    capabilities: [MIDDLEWARE]
    gateway_compat: ">=0.2.0"
    install: "pip install routeiq-guardrails"
    source: "https://github.com/routeiq/routeiq-guardrails"
    checksum: "sha256:abc123..."
    verified: true
    config_schema_url: "..."
    description: "Content guardrails for LLM requests and responses"
```

**Distribution**:
- PyPI packages with `routeiq-plugin-` prefix
- OCI images for containerized deployment
- Git-based for development (like Traefik plugin catalog)

**Trust Model**:
- Verified plugins: Reviewed and signed by RouteIQ team
- Community plugins: Community-contributed, code-reviewed
- Internal plugins: Organization-specific, private registry

---

## 7. Implementation Roadmap

### Phase 2: Body Access and Streaming Hooks
**Effort**: 2-3 weeks | **Priority**: HIGH | **Risk**: Medium

| Task | Effort | Notes |
|------|--------|-------|
| Add `needs_request_body` / `needs_response_body` to PluginMetadata | 1 day | Backwards compatible |
| Implement request body buffering in PluginMiddleware | 3 days | Only when needed |
| Implement `on_request_body` hook dispatch | 2 days | |
| Implement `on_response_chunk` hook in send wrapper | 3 days | Streaming-safe |
| Bridge `on_llm_stream_chunk` to LiteLLM `log_stream_event` | 2 days | |
| Unit tests for all new hooks | 3 days | |
| Update plugin documentation | 1 day | |

### Phase 3: Plugin Configuration Schema
**Effort**: 2 weeks | **Priority**: HIGH | **Risk**: Low

| Task | Effort | Notes |
|------|--------|-------|
| Add `config_schema()` class method to GatewayPlugin | 1 day | |
| Config loading from YAML + env vars + defaults | 3 days | |
| Config validation via Pydantic | 2 days | |
| Admin API: GET/PUT plugin config | 3 days | |
| JSON Schema generation from Pydantic models | 1 day | |
| Unit tests | 2 days | |

### Phase 4: Plugin SDK
**Effort**: 2 weeks | **Priority**: MEDIUM | **Risk**: Low

| Task | Effort | Notes |
|------|--------|-------|
| Extract `routeiq-plugin-sdk` package | 2 days | |
| PluginTestHarness implementation | 3 days | |
| CLI scaffolding (`routeiq plugin new`) | 2 days | |
| Plugin documentation generator | 2 days | |
| Example plugins (guardrails, cost-tracker, cache) | 3 days | |

### Phase 5: Cross-Plugin State and Event Bus
**Effort**: 1.5 weeks | **Priority**: MEDIUM | **Risk**: Low

| Task | Effort | Notes |
|------|--------|-------|
| Add `shared` dict to PluginContext | 1 day | |
| Implement PluginEventBus | 3 days | |
| Wire event emission into existing hook points | 2 days | |
| Add `on_config_changed` hook | 1 day | |
| Unit tests | 2 days | |

### Phase 6: Plugin Hot-Reload
**Effort**: 1 week | **Priority**: MEDIUM | **Risk**: Medium

| Task | Effort | Notes |
|------|--------|-------|
| Config file watcher for plugins | 2 days | Reuse hot_reload.py |
| `on_config_changed` hook dispatch | 1 day | |
| Admin API reload trigger | 1 day | |
| Integration tests | 2 days | |

### Phase 7: Wasm Plugin Support (Exploratory)
**Effort**: 4-6 weeks | **Priority**: LOW | **Risk**: High

| Task | Effort | Notes |
|------|--------|-------|
| Prototype wasmtime-py integration | 1 week | Feasibility spike |
| Define RouteIQ-specific Wasm ABI | 1 week | Based on proxy-wasm |
| WasmPluginHost wrapper | 2 weeks | |
| Example Wasm plugin (Rust) | 1 week | |

---

## 8. Appendix: File Reference

### Core Plugin System Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/litellm_llmrouter/gateway/plugin_manager.py` | 985 | Plugin lifecycle, dependency resolution, security |
| `src/litellm_llmrouter/gateway/plugin_middleware.py` | 317 | ASGI-level request/response hooks |
| `src/litellm_llmrouter/gateway/plugin_callback_bridge.py` | 246 | LiteLLM callback bridge for LLM hooks |
| `src/litellm_llmrouter/gateway/app.py` | 506 | Plugin wiring in app factory |
| `src/litellm_llmrouter/gateway/__init__.py` | 27 | Public API exports |

### Built-in Plugins

| File | Lines | Plugin |
|------|-------|--------|
| `src/litellm_llmrouter/gateway/plugins/__init__.py` | 41 | Package exports |
| `src/litellm_llmrouter/gateway/plugins/evaluator.py` | 400 | Evaluator framework (base class + hooks) |
| `src/litellm_llmrouter/gateway/plugins/skills_discovery.py` | 509 | Skills discovery endpoints |
| `src/litellm_llmrouter/gateway/plugins/upskill_evaluator.py` | 423 | Reference evaluator implementation |

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_plugin_manager.py` | 30+ | Registration, ordering, deps, failure modes, security |
| `tests/unit/test_plugin_middleware.py` | 20+ | ASGI hooks, short-circuit, streaming, error isolation |
| `tests/unit/test_plugin_callback_bridge.py` | 15+ | LLM hooks, kwargs merging, error isolation |

### Configuration Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLMROUTER_PLUGINS` | (empty) | Comma-separated plugin module paths |
| `LLMROUTER_PLUGINS_ALLOWLIST` | (none) | Allowed plugin paths |
| `LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES` | (none) | Allowed capability types |
| `LLMROUTER_PLUGINS_FAILURE_MODE` | `continue` | Global default failure mode |
| `ROUTEIQ_PLUGIN_STARTUP_TIMEOUT` | `30` | Plugin startup timeout (seconds) |
| `ROUTEIQ_PLUGIN_*` | (varies) | Plugin-specific settings (prefix-stripped) |
| `ROUTEIQ_EVALUATOR_ENABLED` | `false` | Enable evaluator hooks |
| `ROUTEIQ_SKILLS_DIR` | `./skills` | Skills discovery directory |

---

*Report generated from codebase analysis at commit range up to 994a40f on main branch.*
*Industry comparison based on Kong Gateway 3.x, Envoy 1.30+, Traefik 3.x, LiteLLM 1.x documentation.*

# Architecture Review 08: Plugin Ecosystem Analysis

**Date**: 2026-02-07
**Scope**: Plugin/extension system architecture for RouteIQ AI Gateway
**Status**: Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Industry Reference: Gateway Plugin Architectures](#2-industry-reference-gateway-plugin-architectures)
3. [RouteIQ Plugin System: Current State](#3-routeiq-plugin-system-current-state)
4. [Hook Completeness Assessment](#4-hook-completeness-assessment)
5. [Gap Analysis vs. Mature Gateways](#5-gap-analysis-vs-mature-gateways)
6. [Architecture Recommendations](#6-architecture-recommendations)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Appendix: File Reference](#8-appendix-file-reference)

---

## 1. Executive Summary

RouteIQ has completed Phase 1 of its plugin system, delivering a solid foundation with ASGI-level request/response hooks (PluginMiddleware), LiteLLM callback hooks (PluginCallbackBridge), lifecycle management with dependency resolution, and three built-in plugins. The system is well-tested (65+ test assertions across three test files) and production-oriented with failure modes, quarantine, allowlist security, and capability-based policy.

However, compared to mature gateway plugin ecosystems (Kong, Envoy, Traefik), RouteIQ's plugin system has notable gaps in: configuration schema validation, plugin hot-reload, body access hooks, streaming transformation, SDK tooling, and marketplace/distribution infrastructure. This report catalogs these gaps and provides a prioritized roadmap.

**Overall Maturity Rating**: Phase 1 Complete (Foundation) -- approximately 40% of a mature gateway plugin ecosystem.

---

## 2. Industry Reference: Gateway Plugin Architectures

### 2.1 Kong Gateway Plugin Model

Kong is the gold standard for API gateway plugin architecture. Key design elements:

**Phase-Based Execution Model**: Kong plugins execute at specific phases in the request lifecycle, each with distinct semantics:

| Phase | When | Typical Use |
|-------|------|-------------|
| `init_worker` | Worker process start | Background timers, global state |
| `certificate` | TLS handshake | mTLS validation |
| `rewrite` | Before routing | URL rewriting, header manipulation |
| `access` | After routing, before upstream | Authentication, authorization, rate limiting |
| `response` | Streaming response from upstream | Response transformation (buffered) |
| `header_filter` | Response headers received | Header manipulation |
| `body_filter` | Each response body chunk | Body transformation (streaming-safe) |
| `log` | After response sent | Logging, analytics, metrics |
| `error` | Error occurred | Custom error responses |

**Plugin Development Kit (PDK)**: Kong provides `kong.pdk` -- a stable API surface that abstracts Nginx internals:
- `kong.request.*` - Read request headers, body, query params
- `kong.response.*` - Set response headers, status, body
- `kong.service.request.*` - Modify upstream request
- `kong.service.response.*` - Read upstream response
- `kong.ctx.shared` - Cross-plugin shared context
- `kong.log.*` - Structured logging
- `kong.node.*` - Node information
- `kong.ip.*` - IP utilities
- `kong.client.*` - Client certificate info

**Plugin Configuration**: Each plugin has a `schema.lua` that defines typed configuration with validation:
```lua
return {
  name = "rate-limiting",
  fields = {
    { config = {
        type = "record",
        fields = {
          { second = { type = "number", gt = 0 } },
          { minute = { type = "number", gt = 0 } },
          { policy = { type = "string", default = "local",
                       one_of = { "local", "cluster", "redis" } } },
        },
    }},
  },
}
```

**Plugin Ordering/Priority**: Kong uses numeric priority (higher = runs first in access phase, lower = runs first in log phase). Default bundled plugin priorities are well-documented, allowing third-party plugins to insert at precise points.

**Distribution**: Kong Hub hosts 100+ plugins. Plugins are distributed as LuaRocks packages with `kong-plugin-<name>` naming convention, or as Docker layers for custom builds.

### 2.2 Envoy Proxy Filter Chain

Envoy's extensibility model is the most sophisticated in the industry:

**HTTP Filter Chain**: Envoy processes requests through an ordered chain of HTTP filters. Each filter can:
- Decode (modify request going upstream)
- Encode (modify response going downstream)
- Both (bidirectional)

Filter types: `decoder_filter`, `encoder_filter`, `dual_filter`, `access_log`.

**Extension Mechanisms** (in order of maturity):

1. **Native C++ Filters**: Highest performance, compiled into Envoy binary. Used for core functionality (router, RBAC, rate limit).

2. **Lua Filters**: Inline scripting via `envoy.filters.http.lua`. Good for simple request/response transformations. Sandboxed with limited API.

3. **ext_proc (External Processing)**: gRPC-based external processing service. Envoy sends request/response data to an external gRPC service for decisions. Supports request header processing, request body processing (buffered or streamed), response header processing, response body processing (buffered or streamed), and immediate response (short-circuit).

4. **WebAssembly (Wasm) Filters**: Plugins compiled to Wasm, loaded at runtime. Supported languages: Rust, Go, C++, AssemblyScript. Uses the proxy-wasm ABI:
   - `on_request_headers`, `on_request_body`
   - `on_response_headers`, `on_response_body`
   - `on_log`
   - `on_tick` (timer-based background work)
   - Shared memory for cross-filter state
   - HTTP/gRPC callout support
   - Sandboxed execution with resource limits

**Wasm Plugin Lifecycle**:
```
Create VM -> Load Module -> Create Context -> on_vm_start -> on_configure
  -> per-request: on_request_headers -> on_request_body
  -> on_response_headers -> on_response_body -> on_log
```

**Filter Ordering**: Specified explicitly in Envoy's YAML configuration. Order matters significantly -- the router filter must be last.

### 2.3 Traefik Middleware / Plugin Catalog

Traefik's plugin system is more constrained but developer-friendly:

**Middleware Pattern**: Traefik middleware wraps HTTP handlers in a chain pattern (similar to Go's `http.Handler` wrapping). Built-in middleware includes rate limiting, circuit breaker, retry, headers, IP allowlist, etc.

**Plugin Catalog (Yaegi)**: Traefik supports dynamic plugins via Yaegi (a Go interpreter):
- Plugins are Go source code fetched from GitHub at startup
- Interpreted at runtime (no compilation needed)
- Limited to a subset of Go standard library
- Plugin catalog at `https://plugins.traefik.io/`
- Plugins declare a `New()` constructor and implement `http.Handler` interface

**Configuration**:
```yaml
experimental:
  plugins:
    my-plugin:
      moduleName: github.com/user/my-plugin
      version: v1.0.0

http:
  middlewares:
    my-middleware:
      plugin:
        my-plugin:
          headers:
            X-Custom: "value"
```

**Limitations**:
- Yaegi interpretation is slower than compiled Go
- Limited standard library access (security sandboxing)
- No Wasm support (as of 2025)
- Plugins must be available at startup (no hot-reload)

### 2.4 AI Gateway Plugin Patterns

AI-specific gateways have unique extensibility needs:

**LiteLLM Callback System**: LiteLLM provides a `CustomLogger` base class with hooks at:
- `log_pre_api_call(model, messages, kwargs)` -- before LLM call
- `log_success_event(kwargs, response_obj, start_time, end_time)` -- after success
- `log_failure_event(kwargs, response_obj, start_time, end_time)` -- after failure
- `async_log_pre_api_call` / `async_log_success_event` / `async_log_failure_event` (async variants for proxy mode)
- `log_stream_event` -- per-chunk for streaming responses
- `async_post_call_success_hook` / `async_post_call_failure_hook` -- proxy-specific

Callbacks are registered via `litellm.callbacks = [MyLogger()]`. Multiple callbacks can be registered. No priority/ordering mechanism.

**Portkey AI Gateway**: Portkey uses a "gateway config" pattern with composable features:
- Retry with fallback
- Load balancing
- Caching
- Custom metadata injection
- Guardrails (via hooks into request/response)
- No public plugin SDK -- extensibility is via configuration

**Helicone**: Helicone operates as a proxy layer with integration hooks:
- Custom properties on requests (metadata injection)
- User tracking, caching, rate limiting, moderation (content filtering)
- Extensibility via HTTP header-based configuration, not a plugin SDK

**Key Observation**: Most AI gateways lack a formal plugin ecosystem. They either provide hooks/callbacks (LiteLLM) or configuration-driven features (Portkey, Helicone). RouteIQ's approach of a formal plugin framework with lifecycle management is more advanced than all current AI gateway competitors.

### 2.5 Plugin Sandboxing and Isolation Patterns

| Approach | Isolation | Performance | Languages | Hot-Reload |
|----------|-----------|-------------|-----------|------------|
| In-process (Python) | Weak | Best | Python only | Possible |
| Wasm | Strong | Good | Multi-language | Yes |
| ext_proc (gRPC) | Strong | Higher latency | Any | Yes |
| Yaegi (interpreted) | Medium | Moderate | Go only | At startup |
| Subprocess | Strong | Highest latency | Any | Yes |

### 2.6 Plugin Marketplace and Registry Patterns

**Kong Hub**: Central registry with categories (Authentication, Security, Traffic Control, Analytics, Transformations). Each plugin has README, configuration reference, changelog, compatibility matrix. Trust model: Kong-verified vs. community plugins.

**Traefik Plugin Catalog**: Hosted at `https://plugins.traefik.io/`. Plugins reference GitHub repos with specific Go module versions. Quality scores, download counts, compatibility badges. Automatic updates via semantic versioning.

**Common Registry Metadata**:
```json
{
  "name": "my-plugin",
  "version": "1.2.0",
  "author": "org/author",
  "license": "Apache-2.0",
  "gateway_version_compat": ">=1.0.0",
  "capabilities": ["middleware", "auth"],
  "config_schema": {},
  "checksum": "sha256:...",
  "signature": "..."
}
```

### 2.7 Event-Driven Plugin Architectures

Two primary patterns exist for plugin communication:

**Hook-Based (Synchronous Chain)**: Plugins are called in order at specific hook points. Each plugin can modify the request/response or short-circuit. Used by: Kong, Express.js middleware, ASGI middleware. Pros: Simple mental model, deterministic ordering. Cons: Blocking, harder to parallelize.

**Event-Bus-Based (Pub/Sub)**: Plugins subscribe to events. Events dispatched asynchronously. Used by: VSCode extensions, Grafana plugins. Pros: Decoupled, parallelizable, extensible without modifying core. Cons: Harder to reason about ordering, eventual consistency.

**Hybrid Pattern** (recommended for AI gateways): Synchronous hooks for request/response pipeline (ordering matters). Asynchronous event bus for observability/logging (ordering does not matter). This is essentially what RouteIQ has started to build with on_request/on_response (sync chain) and on_llm_success (async fire-and-forget).

---

## 3. RouteIQ Plugin System: Current State

### 3.1 Architecture Overview

RouteIQ's plugin system consists of three core modules:

```
                     create_app() / create_standalone_app()
                                    |
                    _load_plugins_before_routes()
                    _configure_middleware()
                    _register_routes()
                                    |
                 _run_plugin_startup(app)
                    |              |               |
            PluginManager    PluginMiddleware   PluginCallbackBridge
            (lifecycle)      (ASGI hooks)       (LiteLLM hooks)
                    |              |               |
              GatewayPlugin instances
              - evaluator
              - skills_discovery
              - upskill_evaluator
```

### 3.2 Core Components

**PluginManager** (`src/litellm_llmrouter/gateway/plugin_manager.py`):
- Plugin registration and lifecycle management
- Dependency resolution via topological sort (Kahn's algorithm)
- Priority-based ordering (lower number = earlier)
- Security: allowlist enforcement, capability-based policy
- Failure modes: CONTINUE (default), ABORT, QUARANTINE
- Startup timeout (`ROUTEIQ_PLUGIN_STARTUP_TIMEOUT`, default 30s)
- Context injection (settings, logger, URL validator)
- Plugin discovery via `LLMROUTER_PLUGINS` env var
- Health check aggregation
- Singleton pattern with `reset_plugin_manager()` for testing

**PluginMiddleware** (`src/litellm_llmrouter/gateway/plugin_middleware.py`):
- Pure ASGI middleware (not BaseHTTPMiddleware, preserves streaming)
- `on_request(PluginRequest)` hook: inspect/short-circuit incoming requests
- `on_response(PluginRequest, ResponseMetadata)` hook: observe responses
- Plugins called in priority order (on_request) and reverse order (on_response)
- Error isolation: plugin hook failures are caught, logged, never crash the request
- Immutable `PluginRequest` dataclass (method, path, headers, client_ip, request_id)
- `ResponseMetadata` with status_code, headers, duration_ms (no body -- streaming-safe)

**PluginCallbackBridge** (`src/litellm_llmrouter/gateway/plugin_callback_bridge.py`):
- Bridges LiteLLM callback system to GatewayPlugin LLM lifecycle hooks
- `on_llm_pre_call(model, messages, kwargs)` -- before LLM API call
- `on_llm_success(model, response, kwargs)` -- after successful call
- `on_llm_failure(model, exception, kwargs)` -- after failed call
- Duck-typing integration (no CustomLogger subclass)
- Pre-call can return kwargs overrides (merged into call params)
- Error isolation: plugin failures logged, never crash the LLM call
- Duplicate registration prevention

### 3.3 Plugin Base Class

The `GatewayPlugin` ABC defines the plugin contract:

```python
class GatewayPlugin(ABC):
    @property
    def metadata(self) -> PluginMetadata:   # Optional override
    @abstractmethod
    async def startup(self, app, context)   # Required
    @abstractmethod
    async def shutdown(self, app, context)  # Required
    async def health_check(self) -> dict    # Optional
    async def on_request(self, request)     # Optional (ASGI hook)
    async def on_response(self, request, response)  # Optional (ASGI hook)
    async def on_llm_pre_call(self, model, messages, kwargs)  # Optional (LLM hook)
    async def on_llm_success(self, model, response, kwargs)   # Optional (LLM hook)
    async def on_llm_failure(self, model, exception, kwargs)   # Optional (LLM hook)
```

### 3.4 Plugin Metadata System

```python
@dataclass
class PluginMetadata:
    name: str = ""                           # Unique identifier
    version: str = "0.0.0"                   # Semver
    capabilities: set[PluginCapability]      # What the plugin provides
    depends_on: list[str] = []               # Dependency ordering
    priority: int = 1000                     # Load order (lower = earlier)
    failure_mode: FailureMode = CONTINUE     # Error handling behavior
    description: str = ""                    # Human-readable
```

**Defined Capabilities**: ROUTES, ROUTING_STRATEGY, TOOL_RUNTIME, EVALUATOR, OBSERVABILITY_EXPORTER, MIDDLEWARE, AUTH_PROVIDER, STORAGE_BACKEND.

### 3.5 Built-in Plugins

| Plugin | Capability | Priority | Description |
|--------|-----------|----------|-------------|
| `skills-discovery` | ROUTES | 500 | Well-known skills index (`/.well-known/skills/`) |
| `evaluator` (framework) | EVALUATOR | 2000 | Base class + MCP/A2A evaluation hooks |
| `upskill-evaluator` | EVALUATOR | 2000 | Reference evaluator with optional CLI integration |

### 3.6 Security Model

- **Allowlist**: `LLMROUTER_PLUGINS_ALLOWLIST` -- explicit plugin path allowlist
- **Capability Policy**: `LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES` -- restrict what plugins can do
- **SSRF Protection**: `PluginContext.validate_outbound_url` provided to plugins
- **Path Validation**: Skills plugin validates paths against traversal attacks
- **Pre-import Check**: Allowlist checked BEFORE importing plugin module (prevents code execution)

### 3.7 Test Coverage

| Test File | Tests | Coverage Focus |
|-----------|-------|----------------|
| `test_plugin_manager.py` | 30+ | Registration, ordering, dependencies, failure modes, allowlist, capabilities |
| `test_plugin_middleware.py` | 20+ | ASGI hooks, short-circuit, streaming, error isolation, ordering |
| `test_plugin_callback_bridge.py` | 15+ | LLM hooks, kwargs merging, error isolation, registration |

---

## 4. Hook Completeness Assessment

### 4.1 Current Hook Points

| Hook Point | Layer | Direction | Can Modify? | Can Short-Circuit? |
|-----------|-------|-----------|-------------|-------------------|
| `startup(app, context)` | Lifecycle | N/A | App routes/state | N/A |
| `shutdown(app, context)` | Lifecycle | N/A | Cleanup | N/A |
| `health_check()` | Lifecycle | N/A | Health status | N/A |
| `on_request(request)` | ASGI | Inbound | Read-only headers | Yes (PluginResponse) |
| `on_response(request, response_meta)` | ASGI | Outbound | No (observe only) | No |
| `on_llm_pre_call(model, msgs, kwargs)` | LLM | Inbound | kwargs overrides | No |
| `on_llm_success(model, response, kwargs)` | LLM | Outbound | No (observe only) | No |
| `on_llm_failure(model, exception, kwargs)` | LLM | Error | No (observe only) | No |

### 4.2 Missing Hook Points (vs. Kong/Envoy)

| Missing Hook | Kong Equivalent | Priority | Use Case |
|-------------|----------------|----------|----------|
| **Request body access** | `access` phase body read | HIGH | Content filtering, PII redaction, prompt injection detection |
| **Response body access** | `body_filter` phase | HIGH | Response filtering, content moderation, cost tracking |
| **Streaming chunk hook** | `body_filter` (chunked) | HIGH | Per-token processing, streaming guardrails |
| **Request modification** | `rewrite` phase | MEDIUM | Header injection, URL rewriting, model override |
| **Response modification** | `header_filter` phase | MEDIUM | Header injection, CORS, caching headers |
| **Pre-routing hook** | `rewrite` phase | MEDIUM | Custom routing logic, A/B testing |
| **Post-routing hook** | `balancer` phase | LOW | Routing decision telemetry (partially covered) |
| **Error handling hook** | `error` phase | MEDIUM | Custom error responses, retry decisions |
| **Background timer** | `init_worker` timer | LOW | Periodic tasks, cache refresh, metrics flush |
| **Config change hook** | Admin API events | LOW | React to configuration changes |
| **LLM stream chunk** | LiteLLM `log_stream_event` | HIGH | Token-level processing, streaming cost tracking |

### 4.3 Body Access Design Challenge

The current architecture deliberately avoids body access in plugin hooks to preserve streaming performance. This is a correct default, but many plugin use cases require body access.

**Kong's approach**: `body_filter` phase is called for each body chunk, allowing streaming-safe transformation. The `access` phase can optionally buffer the request body with `kong.request.get_raw_body()`.

**Recommended RouteIQ approach**: Add opt-in body access hooks:
- `on_request_body(request, body_bytes)` -- only called if plugin declares `needs_request_body = True` in metadata
- `on_response_chunk(request, chunk, is_final)` -- streaming-safe per-chunk hook
- Plugins that don't declare body needs get the fast path (current behavior)

---

## 5. Gap Analysis vs. Mature Gateways

### 5.1 Feature Comparison Matrix

| Feature | Kong | Envoy | Traefik | LiteLLM | RouteIQ |
|---------|------|-------|---------|---------|---------|
| **Plugin lifecycle** | init/access/log | filter chain | middleware chain | callbacks | startup/shutdown + hooks |
| **Hook phases** | 9 phases | 5 filter points | 1 (handler wrap) | 6 callbacks | 8 hooks |
| **Dependency resolution** | No (priority only) | No (explicit order) | No | No | Yes (topological sort) |
| **Priority ordering** | Yes (numeric) | Yes (explicit) | Yes (chain order) | No | Yes (numeric) |
| **Failure modes** | Per-plugin | Circuit breaker | No | No | Yes (continue/abort/quarantine) |
| **Config schema validation** | Yes (schema.lua) | Yes (protobuf) | Yes (Go struct) | No | No |
| **Request body access** | Yes | Yes | Yes | Partial (kwargs) | No |
| **Response body access** | Yes (body_filter) | Yes | Yes | Yes (response_obj) | No (metadata only) |
| **Streaming support** | Yes (chunked) | Yes (per-chunk) | No | Partial | Yes (preserves streaming) |
| **Hot-reload plugins** | Yes (DB-backed) | Yes (xDS/Wasm) | No | No | No |
| **Plugin SDK/PDK** | Yes (comprehensive) | Yes (Wasm ABI) | Partial | No | Partial (base class) |
| **Marketplace/registry** | Kong Hub | No (OCI images) | Plugin Catalog | No | No |
| **Sandboxing** | Lua sandbox | Wasm sandbox | Yaegi sandbox | None | None (in-process) |
| **Multi-language** | Lua, Go | C++, Wasm (multi) | Go | Python | Python |
| **Testing framework** | Yes (pongo) | Yes (integration) | No | No | Yes (conftest patterns) |
| **Health check hook** | No | Yes (health check) | No | No | Yes |
| **Plugin-specific metrics** | Yes (PDK) | Yes (stats) | No | No | No |
| **Cross-plugin state** | Yes (kong.ctx.shared) | Yes (metadata) | No | No | No |
| **Security policy** | RBAC, mTLS | RBAC | No | No | Allowlist + capability policy |

### 5.2 Strengths of RouteIQ's System

1. **Dependency Resolution**: RouteIQ is the only gateway in this comparison with built-in topological sort for plugin dependencies. Kong and Envoy rely on manual ordering.

2. **Failure Modes**: The CONTINUE/ABORT/QUARANTINE system is more sophisticated than any competitor. Kong plugins either work or crash the request. Envoy filters can have local/remote failure modes but nothing as granular.

3. **AI-Specific Hooks**: The LLM lifecycle hooks (pre_call, success, failure) are unique to AI gateways. No general-purpose gateway has these.

4. **Streaming-Safe Design**: The deliberate decision to pass only ResponseMetadata (no body) in on_response is correct for a streaming-first architecture.

5. **Security-First**: Pre-import allowlist checking and capability-based policy is more restrictive than most gateways. SSRF protection via PluginContext is unique.

6. **Test Patterns**: The autouse fixture pattern with `reset_*()` functions for singletons is well-designed for testing plugin interactions.

### 5.3 Weaknesses / Gaps

1. **No Request/Response Body Access**: This is the biggest functional gap. Cannot implement content filtering, PII redaction, or prompt injection detection as plugins.

2. **No Plugin Configuration Schema**: Plugins read their own env vars. No validation, no typed config, no admin API for plugin config.

3. **No Hot-Reload**: Plugins are loaded at startup and cannot be updated without restart.

4. **No Streaming Chunk Hook**: Cannot process individual SSE chunks in streaming responses. Critical for token-level guardrails, cost tracking, and content filtering.

5. **No Cross-Plugin Communication**: No shared context or event bus for plugins to communicate.

6. **No Plugin-Specific Metrics**: Plugins cannot easily emit custom metrics.

7. **No Multi-Language Support**: Python-only.

8. **No Plugin SDK Tooling**: No CLI for scaffolding, testing, or packaging plugins.

9. **In-Process Only**: All plugins run in the same Python process. No isolation beyond try/except error boundaries.

---

## 6. Architecture Recommendations

### 6.1 Phase 2: Body Access and Streaming Hooks (HIGH PRIORITY)

**Goal**: Enable content filtering, guardrails, and cost tracking plugins.

**New hooks to add to GatewayPlugin**:

```python
class GatewayPlugin(ABC):
    # Existing hooks...

    async def on_request_body(
        self, request: PluginRequest, body: bytes
    ) -> bytes | None:
        """
        Called with the full request body (buffered).
        Only invoked if metadata declares needs_request_body = True.
        Returns None to pass through unchanged, or modified body bytes.
        """
        return None

    async def on_response_chunk(
        self, request: PluginRequest, chunk: bytes, is_final: bool
    ) -> bytes | None:
        """
        Called for each response body chunk (streaming-safe).
        Only invoked if metadata declares needs_response_body = True.
        Returns None to pass through unchanged, or modified chunk bytes.
        """
        return None

    async def on_llm_stream_chunk(
        self, model: str, chunk: Any, kwargs: dict
    ) -> None:
        """
        Called for each streaming SSE chunk from an LLM response.
        Observe-only (cannot modify chunks in flight).
        """
        pass
```

**Metadata extension**:
```python
@dataclass
class PluginMetadata:
    # Existing fields...
    needs_request_body: bool = False   # Opt-in to request body buffering
    needs_response_body: bool = False  # Opt-in to response chunk processing
```

**Implementation notes**:
- Request body buffering should only happen when at least one active plugin declares `needs_request_body = True`. Otherwise, the body flows through unbuffered.
- Response chunks should be passed through without buffering. If a plugin returns modified bytes, those are forwarded; otherwise the original chunk passes through.
- `on_llm_stream_chunk` bridges to LiteLLM's `log_stream_event` callback.

### 6.2 Phase 3: Plugin Configuration Schema (HIGH PRIORITY)

**Goal**: Typed, validated plugin configuration with admin API support.

**Design**:

```python
from pydantic import BaseModel

class GatewayPlugin(ABC):
    @classmethod
    def config_schema(cls) -> type[BaseModel] | None:
        """Return a Pydantic model class for plugin configuration."""
        return None

    @property
    def config(self) -> BaseModel | None:
        """Access the validated plugin configuration."""
        return self._config  # Set by PluginManager during loading
```

**Example plugin with config**:
```python
class RateLimitConfig(BaseModel):
    requests_per_minute: int = 60
    burst_size: int = 10
    key_by: Literal["ip", "api_key", "user"] = "api_key"

class RateLimitPlugin(GatewayPlugin):
    @classmethod
    def config_schema(cls):
        return RateLimitConfig

    async def on_request(self, request):
        limit = self.config.requests_per_minute
        ...
```

**Configuration sources** (in priority order):
1. Plugin-specific config file: `config/plugins/<plugin-name>.yaml`
2. Environment variables: `ROUTEIQ_PLUGIN_<PLUGIN_NAME>_<FIELD>=value`
3. Admin API: `PUT /admin/plugins/<name>/config` (for hot-reload)
4. Defaults from Pydantic model

**Admin API endpoints**:
- `GET /admin/plugins` -- List all plugins with metadata and config
- `GET /admin/plugins/<name>` -- Plugin details, config, health
- `PUT /admin/plugins/<name>/config` -- Update plugin config (hot-reload)
- `GET /admin/plugins/<name>/schema` -- JSON Schema for plugin config

### 6.3 Phase 4: Plugin SDK and Developer Toolkit (MEDIUM PRIORITY)

**Goal**: Make it easy for external developers to create, test, and distribute plugins.

**Plugin SDK components**:

1. **`routeiq-plugin-sdk` package**: GatewayPlugin base class (re-export), PluginTestHarness for unit testing plugins, mock helpers (PluginRequest.mock(), MockPluginContext), type stubs for IDE autocomplete.

2. **CLI scaffolding** (`routeiq plugin new`):
   ```
   routeiq plugin new my-guardrails
   # Creates:
   #   my-guardrails/
   #     __init__.py
   #     plugin.py         # GatewayPlugin subclass
   #     config.py          # Pydantic config model
   #     tests/
   #       test_plugin.py   # Pre-wired test template
   #     pyproject.toml     # Package metadata
   ```

3. **Plugin Test Harness**:
   ```python
   from routeiq_plugin_sdk.testing import PluginTestHarness

   async def test_my_plugin_blocks_harmful_content():
       harness = PluginTestHarness(MyGuardrailPlugin())
       await harness.startup()
       response = await harness.send_request(
           method="POST",
           path="/v1/chat/completions",
           body={"messages": [{"role": "user", "content": "harmful content"}]},
       )
       assert response.status_code == 403
   ```

4. **Plugin documentation generator**: Auto-generate markdown from plugin metadata + config schema. Include hook inventory (which hooks the plugin implements). Generate OpenAPI spec additions for route-providing plugins.

### 6.4 Phase 5: Cross-Plugin State and Event Bus (MEDIUM PRIORITY)

**Goal**: Enable plugins to share state and communicate without tight coupling.

**Shared Context**:
```python
@dataclass
class PluginContext:
    # Existing fields...
    shared: dict[str, Any]  # Cross-plugin shared state
```

**Event Bus** (for async/observability plugins):
```python
class PluginEventBus:
    async def emit(self, event: str, data: dict) -> None:
        """Emit an event to all subscribed plugins."""
    def subscribe(self, event: str, handler: Callable) -> None:
        """Subscribe to an event type."""
```

Events: `request.received`, `response.sent`, `llm.call.started`, `llm.call.completed`, `plugin.error`, `config.changed`, `health.degraded`.

### 6.5 Phase 6: Plugin Hot-Reload (MEDIUM PRIORITY)

**Goal**: Update plugin configuration without full restart.

**Configuration hot-reload** (simpler, implement first):
- Watch `config/plugins/*.yaml` for changes (reuse existing `hot_reload.py` pattern)
- Re-validate config against schema
- Call `plugin.on_config_changed(old_config, new_config)` hook
- Admin API trigger: `POST /admin/plugins/<name>/reload`

**Code hot-reload** (complex, defer):
- Reimport plugin module, create new instance, swap atomically
- Risk: module state leaks, import side effects
- Recommendation: Configuration hot-reload first. Code hot-reload only via full gateway restart.

### 6.6 Phase 7: Wasm Plugin Support (LOW PRIORITY, EXPLORATORY)

**Goal**: Enable multi-language plugins with strong isolation.

**Feasibility Assessment**: Python Wasm runtimes are maturing (`wasmtime-py`, `wasmer-python`). Each Wasm call has ~10-50us overhead (acceptable for per-request hooks). Significant development effort for the host bridge.

**Recommendation**: Defer Wasm to Phase 7 or later. Focus on the Python plugin ecosystem first. If multi-language support becomes a requirement, ext_proc-style gRPC sidecar plugins may be simpler to implement.

### 6.7 Plugin Marketplace Concept (LOW PRIORITY, VISION)

**Goal**: Discoverable, distributable, trusted plugin ecosystem.

**Registry Design**:
```yaml
plugins:
  - name: routeiq-guardrails
    version: "1.2.0"
    author: routeiq-team
    license: Apache-2.0
    capabilities: [MIDDLEWARE]
    gateway_compat: ">=0.2.0"
    install: "pip install routeiq-guardrails"
    source: "https://github.com/routeiq/routeiq-guardrails"
    checksum: "sha256:abc123..."
    verified: true
```

**Distribution**: PyPI packages with `routeiq-plugin-` prefix, OCI images for containerized deployment, Git-based for development.

**Trust Model**: Verified plugins (reviewed and signed by RouteIQ team), community plugins (code-reviewed), internal plugins (organization-specific, private registry).

---

## 7. Implementation Roadmap

### Phase 2: Body Access and Streaming Hooks
**Effort**: 2-3 weeks | **Priority**: HIGH | **Risk**: Medium

| Task | Effort | Notes |
|------|--------|-------|
| Add `needs_request_body` / `needs_response_body` to PluginMetadata | 1 day | Backwards compatible |
| Implement request body buffering in PluginMiddleware | 3 days | Only when needed |
| Implement `on_request_body` hook dispatch | 2 days | |
| Implement `on_response_chunk` hook in send wrapper | 3 days | Streaming-safe |
| Bridge `on_llm_stream_chunk` to LiteLLM `log_stream_event` | 2 days | |
| Unit tests for all new hooks | 3 days | |
| Update plugin documentation | 1 day | |

### Phase 3: Plugin Configuration Schema
**Effort**: 2 weeks | **Priority**: HIGH | **Risk**: Low

| Task | Effort | Notes |
|------|--------|-------|
| Add `config_schema()` class method to GatewayPlugin | 1 day | |
| Config loading from YAML + env vars + defaults | 3 days | |
| Config validation via Pydantic | 2 days | |
| Admin API: GET/PUT plugin config | 3 days | |
| JSON Schema generation from Pydantic models | 1 day | |
| Unit tests | 2 days | |

### Phase 4: Plugin SDK
**Effort**: 2 weeks | **Priority**: MEDIUM | **Risk**: Low

| Task | Effort | Notes |
|------|--------|-------|
| Extract `routeiq-plugin-sdk` package | 2 days | |
| PluginTestHarness implementation | 3 days | |
| CLI scaffolding (`routeiq plugin new`) | 2 days | |
| Plugin documentation generator | 2 days | |
| Example plugins (guardrails, cost-tracker, cache) | 3 days | |

### Phase 5: Cross-Plugin State and Event Bus
**Effort**: 1.5 weeks | **Priority**: MEDIUM | **Risk**: Low

| Task | Effort | Notes |
|------|--------|-------|
| Add `shared` dict to PluginContext | 1 day | |
| Implement PluginEventBus | 3 days | |
| Wire event emission into existing hook points | 2 days | |
| Add `on_config_changed` hook | 1 day | |
| Unit tests | 2 days | |

### Phase 6: Plugin Hot-Reload
**Effort**: 1 week | **Priority**: MEDIUM | **Risk**: Medium

| Task | Effort | Notes |
|------|--------|-------|
| Config file watcher for plugins | 2 days | Reuse hot_reload.py |
| `on_config_changed` hook dispatch | 1 day | |
| Admin API reload trigger | 1 day | |
| Integration tests | 2 days | |

### Phase 7: Wasm Plugin Support (Exploratory)
**Effort**: 4-6 weeks | **Priority**: LOW | **Risk**: High

| Task | Effort | Notes |
|------|--------|-------|
| Prototype wasmtime-py integration | 1 week | Feasibility spike |
| Define RouteIQ-specific Wasm ABI | 1 week | Based on proxy-wasm |
| WasmPluginHost wrapper | 2 weeks | |
| Example Wasm plugin (Rust) | 1 week | |

---

## 8. Appendix: File Reference

### Core Plugin System Files

| File | Purpose |
|------|---------|
| `src/litellm_llmrouter/gateway/plugin_manager.py` | Plugin lifecycle, dependency resolution, security |
| `src/litellm_llmrouter/gateway/plugin_middleware.py` | ASGI-level request/response hooks |
| `src/litellm_llmrouter/gateway/plugin_callback_bridge.py` | LiteLLM callback bridge for LLM hooks |
| `src/litellm_llmrouter/gateway/app.py` | Plugin wiring in app factory |
| `src/litellm_llmrouter/gateway/__init__.py` | Public API exports |

### Built-in Plugins

| File | Plugin |
|------|--------|
| `src/litellm_llmrouter/gateway/plugins/__init__.py` | Package exports |
| `src/litellm_llmrouter/gateway/plugins/evaluator.py` | Evaluator framework (base class + hooks) |
| `src/litellm_llmrouter/gateway/plugins/skills_discovery.py` | Skills discovery endpoints |
| `src/litellm_llmrouter/gateway/plugins/upskill_evaluator.py` | Reference evaluator implementation |

### Test Files

| File | Coverage |
|------|----------|
| `tests/unit/test_plugin_manager.py` | Registration, ordering, deps, failure modes, security |
| `tests/unit/test_plugin_middleware.py` | ASGI hooks, short-circuit, streaming, error isolation |
| `tests/unit/test_plugin_callback_bridge.py` | LLM hooks, kwargs merging, error isolation |

### Configuration Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLMROUTER_PLUGINS` | (empty) | Comma-separated plugin module paths |
| `LLMROUTER_PLUGINS_ALLOWLIST` | (none) | Allowed plugin paths |
| `LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES` | (none) | Allowed capability types |
| `LLMROUTER_PLUGINS_FAILURE_MODE` | `continue` | Global default failure mode |
| `ROUTEIQ_PLUGIN_STARTUP_TIMEOUT` | `30` | Plugin startup timeout (seconds) |
| `ROUTEIQ_PLUGIN_*` | (varies) | Plugin-specific settings (prefix-stripped) |
| `ROUTEIQ_EVALUATOR_ENABLED` | `false` | Enable evaluator hooks |
| `ROUTEIQ_SKILLS_DIR` | `./skills` | Skills discovery directory |

---

*Report generated from codebase analysis at commit range up to 994a40f on main branch.*
*Industry comparison based on Kong Gateway 3.x, Envoy 1.30+, Traefik 3.x, LiteLLM 1.x documentation.*

