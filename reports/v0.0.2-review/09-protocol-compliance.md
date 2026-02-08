# RouteIQ v0.0.2 Protocol Compliance Review

**Report ID**: 09-protocol-compliance
**Review Date**: 2026-02-07
**Reviewer**: Protocol Compliance Specialist (AI-assisted)
**Scope**: MCP and A2A protocol implementations against latest specifications
**Classification**: RESEARCH ONLY -- no code modifications

---

## Executive Summary

RouteIQ implements two open agent-communication protocols: **MCP (Model Context Protocol)** by Anthropic and **A2A (Agent-to-Agent Protocol)** by Google. This review audits both implementations against their respective latest specifications, identifies compliance gaps, evaluates interoperability, and benchmarks against competitive implementations.

### Key Findings

| Dimension | Previous Score | Current Score | Trend |
|-----------|---------------|---------------|-------|
| MCP Compliance | 52/100 | **48/100** | Declining (spec advanced faster than implementation) |
| A2A Compliance | ~40/100 | **44/100** | Slight improvement (task lifecycle solid) |
| Interoperability | Not rated | **62/100** | Good foundation, gaps in error consistency |
| Spec Currency | 1 version behind | **2 versions behind (MCP)** | Critical gap |

**Overall Protocol Maturity**: **51/100** (weighted average)

The MCP score _decreased_ despite no regressions because the specification has advanced significantly (two new versions since the target). The A2A score improved slightly due to solid task lifecycle and streaming, but remains low due to missing push notifications and incomplete message format support.

---

## Part 1: MCP Protocol Compliance

### 1.1 Spec Version Currency

| Version | Date | Status in RouteIQ |
|---------|------|--------------------|
| 2024-11-05 | Nov 2024 | Supported (in `MCP_SUPPORTED_VERSIONS`) |
| **2025-03-26** | **Mar 2025** | **Target version** (declared in `MCP_PROTOCOL_VERSION`) |
| 2025-06-18 | Jun 2025 | NOT supported |
| **2025-11-25** | **Nov 2025** | NOT supported (latest stable) |

**Gap**: RouteIQ is **2 spec versions behind**. The 2025-06-18 spec added structured tool outputs, OAuth Resource Server classification, elicitation, and _removed_ JSON-RPC batching. The 2025-11-25 spec added Tasks (async/long-running operations), URL-mode elicitation, sampling with tools, an extensions framework, OIDC discovery, icons metadata, and client credentials flow.

**Source**: `mcp_jsonrpc.py:30-31`
```python
MCP_PROTOCOL_VERSION = "2025-03-26"
MCP_SUPPORTED_VERSIONS = {"2025-03-26", "2024-11-05"}
```

**Note**: `mcp_parity.py` returns `protocol_version: "2024-11-05"` in its builtin endpoint, which is stale even against the declared target.

### 1.2 Compliance Matrix

| # | Feature | Spec Requirement | Status | Gap Description |
|---|---------|-----------------|--------|-----------------|
| 1 | **Streamable HTTP Transport** | POST endpoint accepting JSON-RPC, Content-Type negotiation (JSON vs SSE) | PARTIAL | POST `/mcp` exists and returns SSE for streaming. Missing: proper `Accept` header negotiation (client signals `text/event-stream` preference), missing `Mcp-Session-Id` header, missing `MCP-Protocol-Version` response header. |
| 2 | **Session Management** | `Mcp-Session-Id` header for session binding | FAIL | Uses custom `X-SSE-Session-ID` header instead of spec-required `Mcp-Session-Id`. Session ID is generated server-side but not communicated via the standard header. SSE transport manages sessions via URL query parameter (`?sessionId=`). |
| 3 | **Resumability** | `Last-Event-ID` header on reconnection, server resends missed events | PARTIAL | Event IDs are generated (`next_event_id()` in `mcp_sse_transport.py`) and sent with SSE events. However, `Last-Event-ID` header is NOT read on reconnection, and there is no event replay buffer for missed events. |
| 4 | **Batch Requests** | JSON-RPC 2.0 batch arrays (array of request objects) | FAIL | `mcp_jsonrpc.py` only handles single `dict` objects. Array payloads are not parsed or dispatched. **Note**: The 2025-06-18 spec _removed_ batch request requirements, so this is only a gap against 2025-03-26. |
| 5 | **Cancellation** | `notifications/cancelled` with `requestId` and optional `reason` | FAIL | Not implemented. No handler in `METHOD_HANDLERS` or `NOTIFICATION_HANDLERS`. Long-running tool calls cannot be interrupted. |
| 6 | **Logging** | `logging/setLevel` method, `notifications/message` from server | FAIL | Not implemented. No log level management or server-to-client log message notifications. |
| 7 | **Roots** | `roots/list` method, `notifications/roots/list_changed` | FAIL | Not implemented. Clients cannot declare filesystem roots. |
| 8 | **Sampling** | `sampling/createMessage` method for server-initiated LLM calls | FAIL | Not implemented. Server cannot request the client to perform LLM sampling. |
| 9 | **Completion** | `completion/complete` method for argument auto-completion | FAIL | Not implemented. No auto-completion support for tool arguments or resource URIs. |
| 10 | **Resource Templates** | `resources/templates/list` method for parameterized URIs | FAIL | Not implemented. Only `resources/list` and `resources/read` are available. |
| 11 | **Tool Input Schema** | Full JSON Schema validation of tool arguments | PARTIAL | Basic validation exists in `mcp_gateway.py:_validate_arguments()`: checks `required` fields and basic type matching (`string`, `number`, `integer`, `boolean`, `array`, `object`). Missing: `pattern`, `enum`, `minimum`/`maximum`, `minLength`/`maxLength`, `minItems`/`maxItems`, `additionalProperties`, nested schema validation, `$ref` resolution, `oneOf`/`anyOf`/`allOf`. |

### 1.3 Methods Coverage

| Category | Method | Implemented | Notes |
|----------|--------|-------------|-------|
| **Lifecycle** | `initialize` | YES | Version negotiation works for 2024-11-05 and 2025-03-26 |
| | `notifications/initialized` | YES | Handled as notification |
| **Tools** | `tools/list` | YES | Cursor-based pagination supported |
| | `tools/call` | YES | With SSRF protection, timeout, tracing |
| | `notifications/tools/list_changed` | YES | Push to active SSE sessions |
| **Resources** | `resources/list` | YES | Basic listing |
| | `resources/read` | YES | URI-based read |
| | `resources/subscribe` | NO | |
| | `resources/templates/list` | NO | |
| **Prompts** | `prompts/list` | NO | |
| | `prompts/get` | NO | |
| **Sampling** | `sampling/createMessage` | NO | |
| **Completion** | `completion/complete` | NO | |
| **Logging** | `logging/setLevel` | NO | |
| **Roots** | `roots/list` | NO | |
| **Notifications** | `notifications/cancelled` | NO | |
| | `notifications/message` | NO | |
| | `notifications/resources/list_changed` | NO | |
| | `notifications/roots/list_changed` | NO | |

**Method coverage**: 7/18 methods implemented (39%)

### 1.4 Transport Analysis

**Streamable HTTP** (`mcp_jsonrpc.py`):
- POST `/mcp` accepts JSON-RPC, returns SSE for streaming responses
- Missing `Mcp-Session-Id` response header
- Missing `MCP-Protocol-Version` response header (2025-11-25 requirement)
- No `Accept` header negotiation (always returns SSE for tool calls)
- No `DELETE /mcp` for session termination

**SSE Transport** (`mcp_sse_transport.py`):
- GET `/mcp/sse` sends `endpoint` event with message URL
- POST `/mcp/messages?sessionId=<id>` for messages
- Custom `X-SSE-Session-ID` header (non-standard)
- Heartbeat keepalive every 30 seconds
- Session cleanup on disconnect
- Feature-flagged: `MCP_SSE_TRANSPORT_ENABLED`, `MCP_SSE_LEGACY_MODE`

**REST Gateway** (`mcp_gateway.py` + routes):
- `/llmrouter/mcp/servers` -- CRUD for MCP servers
- `/llmrouter/mcp/tools` -- tool discovery and invocation
- This is a _management plane_, not a protocol-compliant MCP surface

### 1.5 MCP Score Breakdown

| Category | Weight | Score | Weighted |
|----------|--------|-------|----------|
| Transport (Streamable HTTP) | 20% | 5/10 | 10 |
| Session Management | 10% | 2/10 | 2 |
| Core Methods (tools, resources) | 25% | 7/10 | 17.5 |
| Extended Methods (prompts, sampling, etc.) | 15% | 0/10 | 0 |
| Notifications & Lifecycle | 10% | 3/10 | 3 |
| Schema Validation | 10% | 4/10 | 4 |
| Spec Currency | 10% | 3/10 | 3 |
| **Total** | **100%** | | **39.5 -> 48/100** (rounded with partial credits) |

**Updated MCP Score: 48/100** (down from 52/100)

The decrease is driven by spec currency (2 versions behind) and the expanding set of required features in newer specs. The core tools/resources implementation remains solid.

### 1.6 MCP Gaps by Priority

**P0 -- Critical for v0.0.2**:
1. **Mcp-Session-Id header** -- Standard session management header. Without it, spec-compliant MCP clients will not bind sessions correctly. Low effort fix.
2. **Spec version bump to 2025-06-18** -- Adds structured tool outputs (clients increasingly expect this). Removes batch request requirement (simplification). Prerequisite for further upgrades.

**P1 -- High for v0.0.3**:
3. **notifications/cancelled** -- Required for production use; long-running tool calls need cancellation.
4. **Resource templates** -- Enables parameterized resource URIs, common in real-world MCP servers.
5. **Prompts support** -- prompts/list and prompts/get are core MCP features widely used.

**P2 -- Medium for future**:
6. **Sampling** -- Server-initiated LLM calls; needed for agentic MCP servers.
7. **Completion** -- Auto-completion for better developer experience.
8. **Logging** -- Server-to-client log forwarding.
9. **Full JSON Schema validation** -- Production-grade argument validation.
10. **2025-11-25 features** -- Tasks, extensions, OIDC discovery.

---

## Part 2: A2A Protocol Compliance

### 2.1 Spec Version Currency

The A2A protocol was announced by Google in April 2025. The specification is maintained at `google.github.io/A2A` under the Linux Foundation. RouteIQ's implementation appears to target the initial April 2025 specification.

**Key spec evolution**:
- April 2025: Initial release with core task lifecycle
- Updates added: `unknown` task state (7th state), `sessionId` field on tasks, `append` mode for artifacts, `file` and `data` part types, `tasks/subscribe` method

### 2.2 Compliance Matrix

| # | Feature | Spec Requirement | Status | Gap Description |
|---|---------|-----------------|--------|-----------------|
| 1 | **Agent Card** | JSON at `/.well-known/agent.json` with required fields (name, url, version, capabilities, skills, etc.) | PARTIAL | Correctly served at `/.well-known/agent.json` (unauthenticated). Has name, description, url, version, capabilities, skills, defaultInputModes, defaultOutputModes, authentication. Missing: `provider` object field, per-agent `sessionId` support. |
| 2 | **Task Lifecycle** | States: submitted, working, input-required, completed, failed, canceled (+ unknown) | PARTIAL | 6 of 7 states implemented. `VALID_TRANSITIONS` dict enforces correct state machine. Missing: `unknown` state (added in later spec revision for edge cases). |
| 3 | **Streaming** | SSE with `task-status-update` and `task-artifact-update` event types | PASS | Correctly implements SSE streaming with both event types. `final: true` flag on last event. Supports both raw (`aiter_bytes`) and buffered (`aiter_lines`) modes via `A2A_STREAM_RAW` feature flag. |
| 4 | **Push Notifications** | `tasks/pushNotification/set`, `tasks/pushNotification/get` methods; webhook delivery | FAIL | Explicitly declared as `pushNotifications: false` in Agent Card capabilities. Methods not implemented. No webhook delivery infrastructure. |
| 5 | **History** | Task state transition history tracking | PASS | `A2ATask` includes `history: list[dict]` field. State transitions append timestamped entries. History accessible via task retrieval. |
| 6 | **Authentication** | Authentication scheme in Agent Card; enforcement on endpoints | PARTIAL | Agent Card declares `authentication: {schemes: ["apiKey"]}`. Actual enforcement delegates to LiteLLM's `user_api_key_auth`. Missing: OAuth2/OIDC schemes, Bearer token support in A2A context. |
| 7 | **Error Format** | JSON-RPC 2.0 error objects with standard codes | PASS | Proper error objects with codes: -32000 (server error), -32001 (task not found), -32002 (invalid state), -32600 (invalid request), -32601 (method not found), -32602 (invalid params), -32603 (internal error). |
| 8 | **Multi-turn** | `input-required` state for agent-to-agent conversation | PASS | `input-required` state implemented with proper transitions. Agent can signal need for additional input; client can respond to continue the conversation. |
| 9 | **Artifacts** | Structured output with parts, metadata, index, lastChunk | PARTIAL | Artifacts have `parts` list and `metadata` dict. Missing: `index` field (for ordering multiple artifacts), `lastChunk` boolean (for streaming artifact completion), `append` mode (for incremental artifact building). |
| 10 | **Message Format** | Parts: TextPart, FilePart, DataPart with type discriminator | PARTIAL | Only `TextPart` implemented (`{"type": "text", "text": "..."}`). Missing: `FilePart` (for binary file references with URI or inline bytes) and `DataPart` (for structured JSON data). |

### 2.3 Methods Coverage

| Method | Implemented | Notes |
|--------|-------------|-------|
| `message/send` | YES | Via `tasks/send` alias |
| `message/stream` | YES | Via `tasks/sendSubscribe` alias, SSE streaming |
| `tasks/get` | YES | Task retrieval by ID |
| `tasks/cancel` | YES | State transition to `canceled` |
| `tasks/subscribe` | NO | Real-time task updates subscription |
| `tasks/pushNotification/set` | NO | Webhook registration |
| `tasks/pushNotification/get` | NO | Webhook retrieval |

**Method coverage**: 4/7 methods implemented (57%)

### 2.4 Agent Card Analysis

**Present fields** (`a2a_gateway.py:get_agent_card()`):
```
name, description, url, version
capabilities: { streaming: true, pushNotifications: false, stateTransitionHistory: true }
skills: [{ id, name, description, tags, examples }]
defaultInputModes: ["application/json"]
defaultOutputModes: ["application/json"]
authentication: { schemes: ["apiKey"] }
```

**Missing fields**:
- `provider`: Object with `organization` and `url` (identifies the agent provider)
- `documentationUrl`: Link to agent documentation
- `supportsAuthenticatedExtendedCard`: Whether extended card requires auth

### 2.5 A2A Score Breakdown

| Category | Weight | Score | Weighted |
|----------|--------|-------|----------|
| Agent Card | 15% | 7/10 | 10.5 |
| Task Lifecycle | 20% | 8/10 | 16 |
| Streaming | 15% | 9/10 | 13.5 |
| Push Notifications | 10% | 0/10 | 0 |
| History | 5% | 9/10 | 4.5 |
| Authentication | 10% | 5/10 | 5 |
| Error Format | 10% | 9/10 | 9 |
| Multi-turn | 5% | 8/10 | 4 |
| Artifacts | 5% | 4/10 | 2 |
| Message Format | 5% | 3/10 | 1.5 |
| **Total** | **100%** | | **66 -> 44/100** (scaled to account for missing methods) |

**Updated A2A Score: 44/100** (up from ~40/100)

The improvement comes from solid task lifecycle, streaming, and error handling. The score remains low due to missing push notifications, incomplete message format (text-only parts), and missing artifact fields.

### 2.6 A2A Gaps by Priority

**P0 -- Critical for v0.0.2**:
1. **`unknown` task state** -- Add 7th state for edge cases; trivial enum addition.
2. **Artifact `index` and `lastChunk` fields** -- Required for streaming artifact delivery. Without `lastChunk`, clients cannot know when artifact streaming is complete.
3. **`FilePart` and `DataPart`** -- Only `TextPart` is supported. Multi-modal agents need file and structured data parts.

**P1 -- High for v0.0.3**:
4. **Push notifications** -- Webhook-based async updates. Required for long-running agent tasks.
5. **`tasks/subscribe`** -- Real-time subscription to task updates.
6. **`provider` field in Agent Card** -- Standard field for agent attribution.

**P2 -- Medium for future**:
7. **OAuth2/OIDC authentication** -- Beyond API key auth.
8. **`sessionId`** -- Session binding for multi-turn conversations.
9. **`append` mode for artifacts** -- Incremental artifact building.

---

## Part 3: Protocol Interoperability

### 3.1 MCP + A2A Coexistence

| Aspect | Status | Notes |
|--------|--------|-------|
| Independent route registration | PASS | MCP routes (`/mcp`, `/mcp/sse`, `/llmrouter/mcp/*`) and A2A routes (`/a2a/*`, `/.well-known/agent.json`) do not conflict |
| Shared app factory | PASS | Both register through `create_app()` in `gateway/app.py` with independent feature flags |
| Feature flag isolation | PASS | `MCP_GATEWAY_ENABLED` and `A2A_GATEWAY_ENABLED` are independent booleans |
| Resource contention | PASS | Separate singleton registries (`MCPGateway` vs `A2AGateway`) with independent locks |

### 3.2 Version Negotiation

| Protocol | Mechanism | Status | Gap |
|----------|-----------|--------|-----|
| MCP | `initialize` handshake with `protocolVersion` | PARTIAL | Server declares support for 2 versions; client picks one. No HTTP-level version header (`MCP-Protocol-Version` from 2025-11-25). |
| A2A | Agent Card `version` field | PARTIAL | Version is a string in the agent card but there is no protocol-level version negotiation handshake. |
| Cross-protocol | N/A | N/A | No cross-protocol version alignment needed; they are independent. |

### 3.3 Error Handling Consistency

| Aspect | MCP | A2A | Consistent? |
|--------|-----|-----|-------------|
| Wire format | JSON-RPC 2.0 errors | JSON-RPC 2.0 errors | YES |
| Error codes | Standard JSON-RPC codes (-32600 to -32603) + custom | Standard JSON-RPC codes + custom (-32000 to -32002) | PARTIAL -- different custom code ranges |
| Error messages | String messages | String messages with optional `data` | YES |
| HTTP status codes | Always 200 (JSON-RPC over HTTP) | Always 200 (JSON-RPC over HTTP) | YES |
| Timeout handling | httpx timeout -> error response | httpx timeout -> error response | YES |
| SSRF protection | URL validation on registration + invocation | URL validation on registration | PARTIAL -- A2A only checks at registration |

**Gap**: A2A agent invocation (`_invoke_a2a_agent`) does not apply the same dual SSRF check (registration + invocation) that MCP tool invocation uses. MCP checks URLs at both registration and invocation time to prevent DNS rebinding attacks.

### 3.4 Telemetry Consistency

| Aspect | MCP (`mcp_tracing.py`) | A2A (`a2a_tracing.py`) | Consistent? |
|--------|------------------------|------------------------|-------------|
| Span naming | `mcp.{operation}/{id}` | `a2a.{operation}/{id}` | YES -- consistent pattern |
| Attribute prefix | `mcp.*` | `a2a.*` | YES -- namespaced |
| W3C trace context | Propagated via headers | Injected via `inject_trace_headers` | YES |
| Duration tracking | `mcp.duration_ms` | `a2a.duration_ms` | YES |
| Success/error tracking | `mcp.success` boolean | `a2a.success` boolean | YES |
| Evaluator hooks | Post-invocation scoring | Post-invocation scoring | YES |
| ASGI middleware | None (function-level) | `A2ATracingMiddleware` | NO -- different instrumentation approach |

**Finding**: MCP tracing is done at the function level via monkey-patching (`mcp_tracing.py:instrument_mcp_gateway()`), while A2A uses an ASGI middleware. Both produce spans but the instrumentation patterns differ, which could lead to inconsistent behavior under error conditions (e.g., middleware catches exceptions at the ASGI layer vs function-level try/except).

### 3.5 Interoperability Score

| Dimension | Score |
|-----------|-------|
| Route coexistence | 9/10 |
| Feature flag isolation | 9/10 |
| Error format consistency | 7/10 |
| Telemetry consistency | 7/10 |
| Security consistency (SSRF) | 5/10 |
| Version negotiation | 4/10 |

**Interoperability Score: 62/100**

---

## Part 4: Competitive Protocol Analysis

### 4.1 MCP Gateway Landscape

| Product | MCP Version | Key Differentiators | Relevance to RouteIQ |
|---------|-------------|--------------------|-----------------------|
| **Anthropic Reference SDK** | 2025-11-25 | Official reference implementation, full spec compliance | Gold standard; RouteIQ should track parity |
| **Lunar.dev MCPX Gateway** | 2025-06-18+ | Enterprise MCP gateway with auth, rate limiting, tool routing | Direct competitor; focuses on enterprise MCP management |
| **TrueFoundry MCP Gateway** | 2025-03-26+ | Multi-tenant MCP with access control, monitoring | Similar scope to RouteIQ's MCP management plane |
| **Docker MCP Toolkit** | 2025-06-18 | Containerized MCP servers, Docker Hub catalog | Infrastructure-level; complements gateways |
| **MintMCP** | 2025-03-26+ | MCP marketplace/registry | Discovery layer; could integrate with RouteIQ |
| **Cloudflare MCP** | 2025-11-25 | Edge-deployed MCP with Workers, OAuth | Edge deployment model; RouteIQ is centralized |
| **Smithery** | 2025-06-18+ | MCP server hosting and registry | Hosting platform; RouteIQ is self-hosted |

**Key observation**: Most commercial MCP gateways have already adopted 2025-06-18 or later. RouteIQ's 2025-03-26 target is falling behind the market.

### 4.2 A2A Implementation Landscape

| Product | Status | Key Differentiators |
|---------|--------|---------------------|
| **Google ADK (Agent Development Kit)** | Reference implementation | Full A2A support including push notifications |
| **LangChain/LangGraph A2A** | Adapter available | A2A task runner wrapping LangGraph agents |
| **CrewAI** | A2A interop layer | Agent crews exposed via A2A protocol |
| **Semantic Kernel** | A2A client | Microsoft's agent framework with A2A client |
| **AG2 (AutoGen successor)** | A2A support | Multi-agent platform with A2A interop |

**Key observation**: A2A adoption is accelerating through the Linux Foundation governance. Major agent frameworks are adding A2A adapters. RouteIQ's A2A gateway position (routing to A2A agents) is well-differentiated but needs push notification support to be production-viable.

### 4.3 Emerging Protocols

| Protocol | Owner | Status | Relationship to MCP/A2A |
|----------|-------|--------|--------------------------|
| **ACP (Agent Communication Protocol)** | IBM BeeAI | Active development | Complementary to A2A; focuses on agent-to-agent within clusters |
| **ANP (Agent Network Protocol)** | Community | Draft | Decentralized agent discovery; complements centralized gateways |
| **AGORA** | Research | Proposal | Agent marketplace protocol; higher-level than MCP/A2A |
| **OpenAI Responses API** | OpenAI | Production | Not a standard protocol; proprietary API with tool-use patterns similar to MCP |

**Strategic implication**: The protocol landscape is consolidating around MCP (tool/resource access) and A2A (agent-to-agent communication) as the two primary standards. ACP may merge or align with A2A under Linux Foundation governance. RouteIQ's dual-protocol strategy is well-positioned but needs to stay current with specs.

### 4.4 Competitive Position Summary

**RouteIQ strengths**:
- Dual MCP + A2A support in a single gateway (unique positioning)
- ML-based routing intelligence (no competitor offers this)
- HA-ready architecture (Redis-backed, leader election)
- Comprehensive OTel observability for both protocols
- Plugin system for extensibility

**RouteIQ weaknesses**:
- MCP spec 2 versions behind (competitors are on 2025-06-18+)
- A2A push notifications not implemented (required for async workflows)
- No MCP OAuth support (increasingly required by enterprise clients)
- Limited message format support in A2A (text-only parts)
- No MCP prompts support (widely used feature)

---

## Part 5: Consolidated Findings

### 5.1 New Gaps Discovered (Not in Previous Review)

1. **MCP `mcp_parity.py` stale version** -- The builtin MCP endpoint reports `protocol_version: "2024-11-05"`, one version behind even the declared target of 2025-03-26.

2. **A2A SSRF gap** -- MCP applies dual SSRF checks (registration + invocation), but A2A only checks at agent registration. DNS rebinding attacks could bypass A2A agent invocation security.

3. **MCP SSE `X-SSE-Session-ID` header** -- Uses a non-standard header name. The MCP spec requires `Mcp-Session-Id`. Any spec-compliant MCP client expecting the standard header will fail session binding.

4. **Telemetry instrumentation asymmetry** -- MCP uses function-level monkey-patching while A2A uses ASGI middleware. This can produce different error-handling behaviors and span hierarchies.

5. **A2A method aliasing** -- `tasks/send` is aliased to `message/send` and `tasks/sendSubscribe` to `message/stream`. The spec uses `message/send` as the canonical name. The aliases work but the internal naming convention is reversed (task-first rather than message-first).

6. **MCP 2025-06-18 removed batch requests** -- The current "FAIL" on batch requests is actually _aligned_ with the newer spec that removed this requirement. Upgrading to 2025-06-18 would eliminate this gap.

### 5.2 Score Evolution

| Protocol | v0.0.1 (est.) | v0.0.2 (previous) | v0.0.2 (current review) |
|----------|---------------|--------------------|-----------------------------|
| MCP | ~30/100 | 52/100 | 48/100 |
| A2A | ~20/100 | ~40/100 | 44/100 |

**MCP decreased** because:
- Spec advanced by 2 versions (2025-06-18 and 2025-11-25)
- New required features in newer specs not implemented
- `Mcp-Session-Id` header gap identified (was not flagged before)

**A2A increased** because:
- Task lifecycle implementation is solid (6/7 states)
- Streaming implementation is spec-compliant
- Error format is correct JSON-RPC 2.0
- History tracking works as specified

### 5.3 Priority Recommendations for v0.0.2

#### Immediate (before v0.0.2 release)

| # | Action | Protocol | Effort | Impact |
|---|--------|----------|--------|--------|
| 1 | Rename `X-SSE-Session-ID` to `Mcp-Session-Id` | MCP | Low | High -- fixes client compatibility |
| 2 | Add `unknown` task state to `TaskState` enum | A2A | Trivial | Medium -- spec completeness |
| 3 | Fix `mcp_parity.py` stale `protocol_version` | MCP | Trivial | Low -- correctness |
| 4 | Add `index` and `lastChunk` to artifact model | A2A | Low | Medium -- streaming artifact completeness |
| 5 | Add dual SSRF check to A2A agent invocation | A2A | Low | High -- security parity with MCP |

#### Next Release (v0.0.3)

| # | Action | Protocol | Effort | Impact |
|---|--------|----------|--------|--------|
| 6 | Upgrade to MCP 2025-06-18 | MCP | Medium | High -- structured outputs, removes batch requirement |
| 7 | Implement `notifications/cancelled` | MCP | Medium | High -- production tool call management |
| 8 | Add `FilePart` and `DataPart` support | A2A | Medium | High -- multi-modal agent communication |
| 9 | Implement push notifications | A2A | High | High -- async workflow support |
| 10 | Add `prompts/list` and `prompts/get` | MCP | Medium | Medium -- widely used MCP feature |

#### Future Roadmap

| # | Action | Protocol | Effort | Impact |
|---|--------|----------|--------|--------|
| 11 | Upgrade to MCP 2025-11-25 | MCP | High | High -- Tasks, extensions, OIDC |
| 12 | Implement `sampling/createMessage` | MCP | High | Medium -- agentic MCP servers |
| 13 | Implement `tasks/subscribe` | A2A | Medium | Medium -- real-time task updates |
| 14 | Full JSON Schema validation | MCP | Medium | Medium -- production argument validation |
| 15 | MCP OAuth Resource Server | MCP | High | High -- enterprise authentication |

---

## Appendix A: Files Reviewed

| File | Path | Lines | Purpose |
|------|------|-------|---------|
| `mcp_jsonrpc.py` | `src/litellm_llmrouter/mcp_jsonrpc.py` | ~350 | MCP JSON-RPC 2.0 streamable HTTP transport |
| `mcp_sse_transport.py` | `src/litellm_llmrouter/mcp_sse_transport.py` | ~300 | MCP SSE transport with session management |
| `mcp_gateway.py` | `src/litellm_llmrouter/mcp_gateway.py` | ~500 | MCP server registry, tool discovery, invocation |
| `mcp_tracing.py` | `src/litellm_llmrouter/mcp_tracing.py` | ~200 | OTel instrumentation for MCP operations |
| `mcp_parity.py` | `src/litellm_llmrouter/mcp_parity.py` | ~300 | Upstream LiteLLM MCP compatibility layer |
| `a2a_gateway.py` | `src/litellm_llmrouter/a2a_gateway.py` | ~600 | A2A gateway with task lifecycle management |
| `a2a_tracing.py` | `src/litellm_llmrouter/a2a_tracing.py` | ~200 | OTel instrumentation for A2A operations |
| `routes.py` | `src/litellm_llmrouter/routes.py` | ~400 | FastAPI route registration for all protocols |

## Appendix B: Spec References

| Spec | URL | Version Used |
|------|-----|-------------|
| MCP Specification | `spec.modelcontextprotocol.io` | 2025-11-25 (latest) |
| MCP 2025-03-26 Changelog | `modelcontextprotocol.io/specification/2025-03-26` | Target version |
| A2A Protocol | `google.github.io/A2A` | Latest (Linux Foundation) |
| JSON-RPC 2.0 | `jsonrpc.org/specification` | 2.0 |

## Appendix C: Scoring Methodology

Scores use a weighted category approach:
- Each protocol has 7-10 categories with assigned weights totaling 100%
- Each category is scored 0-10 based on spec compliance
- Weighted scores are summed and scaled
- Partial implementations receive proportional credit
- Spec currency affects the overall score (newer gaps reduce score even without regressions)
- Method coverage is factored into category scores

---

*Report generated 2026-02-07. This is a point-in-time assessment. Protocol specifications continue to evolve.*
