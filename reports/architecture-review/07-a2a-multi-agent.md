# Architecture Review: A2A Protocol & Multi-Agent Orchestration

**Date**: 2026-02-07
**Scope**: RouteIQ A2A Gateway implementation vs. Google A2A specification

**Files Reviewed**:
- `/src/litellm_llmrouter/a2a_gateway.py` -- A2A Gateway core (agent registry, invocation, streaming, JSON-RPC models)
- `/src/litellm_llmrouter/a2a_tracing.py` -- A2A OTel tracing middleware, StreamingSpan, W3C trace context injection
- `/src/litellm_llmrouter/routes.py` -- All API routes including A2A convenience endpoints
- `/src/litellm_llmrouter/startup.py` -- Startup orchestration, A2A tracing initialization
- `/src/litellm_llmrouter/gateway/app.py` -- App factory (composition root)
- `/src/litellm_llmrouter/database.py` -- PostgreSQL persistence for A2A agents
- `/src/litellm_llmrouter/gateway/plugins/evaluator.py` -- Evaluator plugin with post-A2A invocation hooks
- `/tests/unit/test_a2a_tracing.py` -- Unit tests for A2A tracing
- `/tests/unit/test_a2a_streaming_passthrough.py` -- Unit tests for A2A streaming
- `/tests/property/test_a2a_gateway_properties.py` -- Property-based tests for agent registration, discovery, invocation, persistence

---

## 1. Google A2A Protocol Specification Summary

The A2A (Agent-to-Agent) protocol is an open standard for inter-agent communication. It defines how autonomous AI agents discover, communicate with, and delegate tasks to each other.

### 1.1 Core Concepts

| Concept | Description |
|---------|-------------|
| **Agent Card** | JSON metadata at `/.well-known/agent.json` describing agent capabilities, skills, auth requirements |
| **Task** | The unit of work; has a lifecycle with well-defined states |
| **Message** | Communication between agents within a task; contains Parts |
| **Part** | Content unit: TextPart, FilePart (binary), DataPart (structured JSON) |
| **Artifact** | Output produced by an agent during task execution |

### 1.2 Transport

- **HTTP + JSON-RPC 2.0**: Primary transport for request/response
- **Server-Sent Events (SSE)**: For streaming responses via `tasks/sendSubscribe`
- All communication is over HTTPS with JSON-RPC 2.0 framing

### 1.3 Key Methods (A2A Spec)

| Method | Purpose | Transport |
|--------|---------|-----------|
| `tasks/send` | Send a task and wait for completion | HTTP POST |
| `tasks/sendSubscribe` | Send a task and subscribe to SSE updates | HTTP POST + SSE |
| `tasks/get` | Retrieve current task state | HTTP POST |
| `tasks/cancel` | Cancel a running task | HTTP POST |
| `tasks/pushNotification/set` | Register a webhook for push notifications | HTTP POST |
| `tasks/pushNotification/get` | Get current push notification config | HTTP POST |
| `tasks/resubscribe` | Re-subscribe to an in-progress task's SSE stream | HTTP POST |

### 1.4 Task Lifecycle States

```
submitted --> working --> completed
                |   \--> failed
                |   \--> canceled
                \-------> input-required --> working (via new message)
```

States: `submitted`, `working`, `input-required`, `completed`, `failed`, `canceled`

### 1.5 Agent Card (/.well-known/agent.json)

Required fields per A2A spec:
- `name`: Human-readable agent name
- `description`: What the agent does
- `url`: Agent's A2A endpoint URL
- `version`: A2A protocol version supported
- `capabilities`: Object with `streaming`, `pushNotifications`, `stateTransitionHistory`
- `skills`: Array of skill descriptors (`id`, `name`, `description`, `tags`, `examples`)
- `authentication`: Auth schemes supported (OAuth 2.0, API keys, bearer tokens)
- `defaultInputModes`: Supported input content types
- `defaultOutputModes`: Supported output content types

### 1.6 Authentication

The A2A spec supports OAuth 2.0 (preferred), bearer tokens, API keys, and custom headers. Authentication requirements are declared in the Agent Card.

### 1.7 Push Notifications

Agents can register webhook URLs to receive asynchronous task updates via `tasks/pushNotification/set`. The server sends task state transitions to registered URLs with HMAC signature verification.

### 1.8 Multi-Part Content

Messages contain arrays of Parts:
- **TextPart**: `{ "type": "text", "text": "..." }`
- **FilePart**: `{ "type": "file", "file": { "name": "...", "mimeType": "...", "bytes": "base64..." } }`
- **DataPart**: `{ "type": "data", "data": { ... } }`

### 1.9 A2A vs MCP (Complementary Protocols)

| Aspect | A2A | MCP |
|--------|-----|-----|
| **Purpose** | Agent-to-agent communication | Agent-to-tool/resource communication |
| **Participants** | Two or more autonomous agents | One agent + one or more tool servers |
| **Communication** | Peer-to-peer, bidirectional | Client-server (agent is client) |
| **Task model** | Stateful tasks with lifecycle | Stateless tool invocations |
| **Discovery** | Agent Cards at well-known URLs | Server capabilities negotiation |
| **Streaming** | SSE for task updates | SSE for real-time events |

They are complementary: an agent might use MCP to access tools and A2A to collaborate with other agents.

---

## 2. RouteIQ A2A Implementation Analysis

### 2.1 Architecture Overview

RouteIQ has a dual A2A architecture:

**LiteLLM Built-in A2A** (primary): LiteLLM provides built-in A2A endpoints:
- `POST /v1/agents` -- Create agent (DB-backed)
- `GET /v1/agents` -- List agents (DB-backed)
- `DELETE /v1/agents/{agent_id}` -- Delete agent (DB-backed)
- `POST /a2a/{agent_id}` -- Invoke agent (A2A JSON-RPC protocol)
- `POST /a2a/{agent_id}/message/stream` -- Streaming alias

**RouteIQ A2A Gateway** (optional layer, `A2A_GATEWAY_ENABLED=false` by default):
- Custom `A2AGateway` class in `a2a_gateway.py`
- In-memory agent registry with thread-safe operations (RLock, MappingProxyType)
- JSON-RPC 2.0 message forwarding (`message/send`, `message/stream`)
- Dual-mode streaming (raw passthrough and line-buffered)
- SSRF protection on all outbound URLs (checked at both registration and invocation)
- W3C trace context propagation

**RouteIQ Convenience Routes** (thin wrappers):
- `GET /a2a/agents` -- List agents (wraps `global_agent_registry`)
- `POST /a2a/agents` -- Register agent (in-memory, admin-only)
- `DELETE /agents/{agent_id}` -- Unregister agent (in-memory, admin-only)

### 2.2 Component Inventory

| Component | File | Purpose |
|-----------|------|---------|
| `A2AGateway` | `a2a_gateway.py` | Agent registry, invocation, streaming |
| `A2AAgent` | `a2a_gateway.py` | Agent dataclass (id, name, url, capabilities) |
| `JSONRPCRequest` / `JSONRPCResponse` | `a2a_gateway.py` | JSON-RPC 2.0 data models |
| `A2ATracingMiddleware` | `a2a_tracing.py` | ASGI middleware for `/a2a/*` OTel spans |
| `StreamingSpan` | `a2a_tracing.py` | Span lifecycle for async streaming |
| `inject_trace_headers` | `a2a_tracing.py` | W3C traceparent/tracestate propagation |
| `instrument_a2a_gateway` | `a2a_tracing.py` | Monkey-patch tracing onto gateway |
| `A2AAgentDB` | `database.py` | PostgreSQL persistence model |
| `AgentRegistration` | `routes.py` | Pydantic registration request model |
| Evaluator hooks | `evaluator.py` | Post-invocation scoring via OTEL |

### 2.3 Strengths

1. **JSON-RPC 2.0 compliance**: Proper framing with correct error codes (-32600, -32601, -32603, -32000).

2. **Streaming infrastructure**: Two streaming modes (raw and line-buffered) with feature flagging (`A2A_RAW_STREAMING_ENABLED`), rollback safety, and configurable chunk sizes (`A2A_RAW_STREAMING_CHUNK_SIZE`).

3. **Security**: SSRF protection at both registration and invocation time, URL validation, secret scrubbing, admin auth separation, RBAC with `PERMISSION_A2A_AGENT_WRITE`.

4. **Observability**: `a2a.agent.send` spans for sync invocations, `a2a.agent.stream` spans with proper lifecycle, `A2ATracingMiddleware` for HTTP-level tracing, W3C trace context propagation to downstream agents, evaluator plugin hooks for post-invocation scoring.

5. **Thread safety**: RLock-protected registry, immutable snapshots via `MappingProxyType`, double-checked locking on singleton.

6. **Testing**: Property-based tests (Hypothesis, ~1800 lines covering 26 properties), unit tests for tracing/streaming/header preservation, JSON-RPC error code compliance tests.

7. **Dual persistence**: In-memory registry for fast access + PostgreSQL persistence for HA.

8. **Audit logging**: All write operations (register, unregister) are audit-logged with RBAC context.

---

## 3. A2A Spec Compliance Assessment

### 3.1 Compliance Matrix

| A2A Spec Feature | RouteIQ Status | Gap Severity | Notes |
|-----------------|---------------|-------------|-------|
| Agent Card at `/.well-known/agent.json` | NOT IMPLEMENTED | HIGH | No endpoint serving Agent Cards |
| Agent Card: name, description, url | PARTIAL | MEDIUM | `get_agent_card()` returns these fields |
| Agent Card: version | MISSING | MEDIUM | No protocol version field |
| Agent Card: authentication | MISSING | MEDIUM | Auth not declared in card |
| Agent Card: skills | PARTIAL | LOW | Derived from capabilities; lack tags/examples/description |
| Agent Card: defaultInputModes/defaultOutputModes | MISSING | LOW | No content type negotiation |
| `tasks/send` | PARTIAL | HIGH | Uses `message/send` instead |
| `tasks/sendSubscribe` (SSE) | PARTIAL | HIGH | Uses `message/stream`; no SSE event framing |
| `tasks/get` | NOT IMPLEMENTED | HIGH | Cannot retrieve task state |
| `tasks/cancel` | NOT IMPLEMENTED | HIGH | Cannot cancel running tasks |
| Task lifecycle states | NOT IMPLEMENTED | HIGH | No state machine |
| Task ID tracking | NOT IMPLEMENTED | HIGH | No task persistence |
| `tasks/pushNotification/set` | NOT IMPLEMENTED | MEDIUM | No webhook registration |
| `tasks/pushNotification/get` | NOT IMPLEMENTED | MEDIUM | No push notification retrieval |
| `tasks/resubscribe` | NOT IMPLEMENTED | LOW | Cannot re-subscribe |
| Multi-turn conversations | NOT IMPLEMENTED | HIGH | Stateless invocations |
| `input-required` state | NOT IMPLEMENTED | HIGH | No clarification support |
| TextPart | IMPLICIT | LOW | Passes through |
| FilePart | NOT VALIDATED | LOW | No validation |
| DataPart | NOT VALIDATED | LOW | No validation |
| Artifacts | NOT IMPLEMENTED | MEDIUM | No artifact model |
| stateTransitionHistory | PARTIAL | MEDIUM | Declared but not implemented |
| Authentication in Agent Card | NOT IMPLEMENTED | MEDIUM | No auth scheme declaration |
| A2A-specific errors | PARTIAL | MEDIUM | Uses JSON-RPC errors only |

### 3.2 Method Name Divergence (Critical)

RouteIQ uses `message/send` and `message/stream`. The A2A spec uses `tasks/send` and `tasks/sendSubscribe`. Any standard A2A client would receive "Method not found" errors.

### 3.3 Missing Task Model (Critical)

The A2A protocol is fundamentally task-oriented. RouteIQ treats each invocation as a stateless pass-through. There is no task ID generation, no state machine, no task persistence, no ability to query task status, and no support for multi-turn interactions where a task enters `input-required`.

### 3.4 Missing Agent Card Endpoint

The `get_agent_card()` method exists on `A2AGateway` but there is no HTTP endpoint at `/.well-known/agent.json`. The card structure is also missing `version`, `authentication`, `defaultInputModes`, and `defaultOutputModes`.

---

## 4. Multi-Agent Orchestration Assessment

### 4.1 Can RouteIQ Act as an A2A Hub?

RouteIQ functions as a **proxy/gateway** for A2A agents, not an orchestration hub.

| Orchestration Capability | Status | Details |
|--------------------------|--------|---------|
| Agent discovery | YES | Registry with capability-based filtering |
| Agent registration | YES | Dynamic registration via API with SSRF protection |
| Request forwarding | YES | JSON-RPC 2.0 forwarding to registered agent URLs |
| Response streaming | YES | SSE streaming with raw and buffered modes |
| Task delegation | NO | No task decomposition or delegation logic |
| Agent routing/selection | NO | No intelligent routing based on capabilities |
| Conversation context | NO | Stateless; no session management |
| Error propagation | PARTIAL | Errors returned as JSON-RPC but no cross-agent correlation |
| Workflow orchestration | NO | No DAG, pipeline, or workflow execution |
| Agent composition | NO | Cannot chain agents or compose results |
| Load balancing | NO | No load balancing across equivalent agents |
| Circuit breaking per agent | NO | No per-agent circuit breakers |
| Rate limiting per agent | NO | No per-agent rate limiting |

### 4.2 Comparison with Multi-Agent Frameworks

| Framework | Pattern | RouteIQ Equivalent |
|-----------|---------|-------------------|
| **CrewAI** | Role-based agent teams with task delegation | Not supported -- no role or team concepts |
| **AutoGen** | Multi-agent conversation with tool use | Not supported -- no conversation management |
| **LangGraph** | Stateful graph-based agent workflows | Not supported -- no graph execution engine |
| **Strands** | Agent-as-tool, A2A protocol, swarm/graph patterns | Partial -- can register agents but no orchestration |
| **Semantic Kernel** | Plugin/function-based agent orchestration | Partial -- plugin system exists but not for agent orchestration |

### 4.3 Current Position

RouteIQ is best described as an **A2A-aware API gateway** with four planes:

1. **Registration plane**: Agents register with the gateway (admin API)
2. **Discovery plane**: Clients query agents by capability
3. **Invocation plane**: Gateway handles auth, SSRF, tracing, streaming
4. **Observability plane**: OTel tracing + evaluator scoring

Missing for multi-agent orchestration: planning, coordination, state management, error recovery.

---

## 5. Architecture Recommendations

### Priority 1: Critical A2A Spec Compliance (HIGH)

**5.1 Implement A2A Method Names**
- Add `tasks/send` as alias for `message/send`
- Add `tasks/sendSubscribe` as alias for `message/stream`
- Add `tasks/get` and `tasks/cancel` as new methods
- Maintain backward compatibility with both method families

**5.2 Implement Task Model and State Machine**
- Create `Task` model with `task_id`, `state` (enum), `messages`, `artifacts`, `history`
- Implement state machine: submitted -> working -> completed/failed/canceled/input-required
- In-memory store with optional Redis/PostgreSQL persistence

**5.3 Add `/.well-known/agent.json` Endpoint**
- Unauthenticated route returning gateway's Agent Card
- Include all required fields: name, description, url, version, capabilities, skills, authentication
- Per-agent cards at `/.well-known/agents/{agent_id}.json`

**5.4 Implement SSE Event Framing**
- Replace NDJSON streaming with SSE format (`event:` + `data:` prefixes)
- Support `task-status-update` and `task-artifact-update` event types
- Include `final: true` on terminal events

### Priority 2: Enhanced A2A Features (MEDIUM)

**5.5 Push Notifications**
- `tasks/pushNotification/set` and `tasks/pushNotification/get`
- HMAC signature for webhook verification, SSRF protection on webhook URLs

**5.6 Content Type Validation**
- Pydantic models for TextPart, FilePart, DataPart
- Validate FilePart fields (name, mimeType, bytes/uri)

**5.7 Authentication in Agent Cards**
- Extend `A2AAgent` model with `authentication.schemes`
- Surface in `get_agent_card()` and well-known endpoint

### Priority 3: Multi-Agent Orchestration (MEDIUM-LOW)

**5.8 Capability-Based Agent Routing**
- Match task descriptions to agent skills/capabilities
- Leverage existing ML routing strategies

**5.9 Per-Agent Circuit Breakers**
- Extend `CircuitBreakerManager` for per-agent health tracking

**5.10 Task Delegation Pipeline**
- Sequential, parallel fan-out, and conditional routing patterns

**5.11 Conversation Context Management**
- Message history per task, `input-required` state support

### Priority 4: Advanced Multi-Agent Patterns (LOW)

**5.12 Agent-as-Tool Pattern**: Bridge A2A agents to MCP tools
**5.13 Agent Health Monitoring**: Periodic health checks, OTel metrics
**5.14 Swarm Pattern Support**: Dynamic task handoff between agents

---

## 6. Summary

### Gap Assessment Scorecard

| Category | Score | Notes |
|----------|-------|-------|
| Transport (HTTP + JSON-RPC 2.0) | 8/10 | Solid JSON-RPC implementation |
| Agent Card compliance | 2/10 | No well-known endpoint; incomplete card structure |
| Task lifecycle | 0/10 | No task model, state machine, or persistence |
| Streaming (SSE) | 6/10 | Good infra but wrong framing (NDJSON vs SSE events) |
| Authentication | 5/10 | Gateway auth exists but not declared in Agent Cards |
| Push notifications | 0/10 | Not implemented |
| Multi-part content | 3/10 | Passes through but no validation |
| Multi-turn conversations | 0/10 | Stateless invocations only |
| Multi-agent orchestration | 2/10 | Registry and forwarding only |
| Observability | 9/10 | Excellent OTel integration |
| Security | 9/10 | Strong SSRF, auth, RBAC, audit |
| Testing | 8/10 | Property tests, unit tests, streaming tests |

### Recommended Phases

1. **Phase 1**: Core A2A compliance -- method names, Task model, Agent Card endpoint, SSE framing
2. **Phase 2**: Enhanced features -- push notifications, content validation, auth in cards
3. **Phase 3**: Multi-agent -- capability routing, circuit breakers, context management
4. **Phase 4**: Advanced -- task delegation, agent-as-tool bridge, health monitoring, swarm

### Current JSON-RPC Methods vs A2A Spec

| Current RouteIQ Method | A2A Spec Equivalent | Status |
|------------------------|---------------------|--------|
| `message/send` | `tasks/send` | Name mismatch |
| `message/stream` | `tasks/sendSubscribe` | Name mismatch |
| (none) | `tasks/get` | Missing |
| (none) | `tasks/cancel` | Missing |
| (none) | `tasks/pushNotification/set` | Missing |
| (none) | `tasks/pushNotification/get` | Missing |
| (none) | `tasks/resubscribe` | Missing |
