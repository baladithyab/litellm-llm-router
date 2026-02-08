# 09 - Emerging AI API Standards: Assessment and RouteIQ Readiness

**Date**: 2026-02-07
**Scope**: Emerging API standards from OpenAI, Anthropic, Google, and the broader ecosystem, evaluated against RouteIQ Gateway's current architecture.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Emerging Standards Landscape](#2-emerging-standards-landscape)
3. [RouteIQ Current API Surface Analysis](#3-routeiq-current-api-surface-analysis)
4. [Readiness Assessment Matrix](#4-readiness-assessment-matrix)
5. [Gap Analysis](#5-gap-analysis)
6. [Architecture Recommendations](#6-architecture-recommendations)
7. [Implementation Priorities](#7-implementation-priorities)
8. [Risk Assessment](#8-risk-assessment)

---

## 1. Executive Summary

The AI API landscape is undergoing a fundamental architectural shift. The simple request-response proxy model that RouteIQ (and all AI gateways) was built around is being challenged by four converging trends:

1. **Stateful APIs** (OpenAI Responses API) that maintain server-side conversation state, breaking the stateless proxy assumption.
2. **Real-time streaming protocols** (OpenAI Realtime, Gemini Live) that use WebSocket connections for bidirectional audio/text, incompatible with HTTP request-response middleware.
3. **Agent-native APIs** with built-in tool loops, where a single API call triggers multiple internal LLM invocations and tool executions.
4. **Provider divergence** in multimodal, structured output, and tool use formats, requiring sophisticated normalization.

RouteIQ inherits significant Responses API and structured output support from upstream LiteLLM (which already has `/v1/responses` endpoints, completion-to-responses transformation, and provider-specific adapters). However, RouteIQ's own routing intelligence layer (`RouterDecisionMiddleware`, `strategies.py`, and the policy engine) currently only instruments `/v1/chat/completions` and `/chat/completions` paths. This means emerging API patterns bypass RouteIQ's core value proposition: ML-based routing, telemetry, and policy enforcement.

**Overall Readiness: MODERATE with critical gaps in routing instrumentation for new API surfaces.**

---

## 2. Emerging Standards Landscape

### 2.1 OpenAI Responses API

**Status**: Generally Available (successor to Chat Completions)

The Responses API (`POST /v1/responses`) represents OpenAI's most significant API architecture change since GPT-3. Key differences from Chat Completions:

| Aspect | Chat Completions | Responses API |
|--------|-----------------|---------------|
| State | Stateless (client manages history) | Stateful (`previous_response_id` chaining) |
| Tools | Developer-defined functions only | Built-in tools (web search, file search, code interpreter, computer use) + developer functions |
| Multi-turn | Client re-sends full history | Server chains via `previous_response_id` |
| Output format | `choices[].message` | `output[]` items (text, tool calls, reasoning) |
| Background mode | Not supported | `background: true` with polling |
| Streaming | SSE chunks of deltas | SSE with richer event types |
| Input format | `messages[]` array | `input` (string or structured items) |

**Critical Gateway Implications**:

- **Statefulness breaks routing**: When a client sends `previous_response_id`, the request MUST go to the same provider that created that response (or one compatible with its state). RouteIQ's per-request ML routing cannot freely select a different provider mid-conversation.
- **Built-in tools create opacity**: When a model uses `web_search` or `code_interpreter`, the tool execution happens server-side at the provider. The gateway cannot observe, control, or bill for these intermediate steps.
- **Background mode requires async infrastructure**: `background: true` returns immediately with a `polling_id`. The gateway needs to support polling endpoints and async response tracking.
- **Input format divergence**: The `input` field accepts strings, structured items with `type` discriminators, and mixed content -- different from the `messages` array format.

**LiteLLM Upstream Status**: LiteLLM already supports `POST /v1/responses`, `/responses`, and `/openai/v1/responses` endpoints in `litellm/proxy/response_api_endpoints/endpoints.py`. It includes:
- Background mode with Redis-based polling (`response_polling/`)
- Completion-to-responses transformation for non-OpenAI providers (`litellm_completion_transformation/`)
- Streaming support via `BaseResponsesAPIStreamingIterator`
- Session handling for `previous_response_id` chaining

### 2.2 OpenAI Realtime API

**Status**: Generally Available

The Realtime API uses WebSocket connections for bidirectional audio and text streaming. Key characteristics:

- **Protocol**: WebSocket (`wss://api.openai.com/v1/realtime`)
- **Session-based**: Long-lived connections with server-managed state
- **Audio formats**: PCM16, G.711 (mu-law and A-law)
- **Tool use during sessions**: Function calling within a real-time session
- **Voice activity detection (VAD)**: Server-side turn detection
- **Events**: Client sends `input_audio_buffer.append`, `response.create`; server sends `response.audio.delta`, `response.text.delta`, etc.

**Critical Gateway Implications**:

- **WebSocket proxying**: Standard HTTP middleware (request ID, policy engine, backpressure) does not apply to WebSocket frames.
- **Long-lived connections**: Cannot use request-per-connection concurrency tracking. A single WebSocket connection may last minutes to hours.
- **Audio payload routing**: Binary audio data mixed with JSON control messages on the same connection.
- **Provider lock-in**: Once a WebSocket session starts with a provider, it cannot be rerouted.

**LiteLLM Upstream Status**: LiteLLM has `_arealtime()` in `litellm/realtime_api/main.py` with WebSocket pass-through support. It also has Vertex AI Live API WebSocket passthrough at `/vertex_ai/live`. The proxy server includes WebSocket route registration and realtime health checks.

### 2.3 Anthropic Messages API Evolution

**Status**: Production, rapidly evolving

Anthropic's Messages API has diverged significantly from OpenAI-compatible patterns:

- **Tool Use**: Content blocks with `type: "tool_use"` and `type: "tool_result"`, rather than top-level `tool_calls`. The model returns tool calls as content blocks interleaved with text.
- **Computer Use**: Specialized tool definitions for desktop automation (`computer_20241022`, `text_editor_20241022`, `bash_20241022`) with screenshot-driven interaction loops.
- **Citations**: Source attribution in responses with document references.
- **Extended Thinking**: `thinking` content blocks with chain-of-thought reasoning visible to the client. Uses budget tokens for controlling reasoning depth.
- **Batch API**: Asynchronous batch processing with different billing rates.
- **Prompt Caching**: Cache breakpoints in conversations to reduce token costs.

**Critical Gateway Implications**:

- **Content block normalization**: Anthropic's tool use as content blocks vs. OpenAI's top-level `tool_calls` requires deep response transformation for cross-provider compatibility.
- **Computer Use loops**: Multi-step screenshot -> action -> screenshot cycles require maintaining state across multiple API calls.
- **Extended Thinking tokens**: Additional token usage for thinking blocks must be tracked for cost attribution.
- **Provider-specific features**: Citations and prompt caching are Anthropic-only features that have no direct cross-provider equivalent.

### 2.4 Google Gemini API Patterns

**Status**: Production (Gemini 2.0+)

Google's Gemini API introduces several distinctive patterns:

- **Function Calling**: Similar to OpenAI but with `NONE`, `AUTO`, `ANY` modes and parallel function calling.
- **Grounding with Google Search**: Built-in grounding tool that returns search results with source attribution.
- **Code Execution**: Server-side Python code execution in sandboxed environments.
- **Gemini Live API**: WebSocket-based real-time multimodal interaction (text, audio, video).
- **Context Caching**: Explicit cache management for long contexts.
- **System Instructions**: Separate from messages, with specific formatting requirements.

**Critical Gateway Implications**:

- **Grounding transparency**: Google Search grounding responses include `groundingMetadata` with search results and support links -- different metadata structure from any other provider.
- **Code execution results**: Inline code execution output blocks in responses.
- **Video understanding**: Gemini can process video frames, a capability unique to Gemini among major providers.
- **Different error model**: Gemini uses Google Cloud error conventions, not OpenAI-style errors.

### 2.5 Structured Output Standards

**Status**: Converging but provider-specific

All major providers now support some form of structured output, but with different mechanisms:

| Provider | Mechanism | Format |
|----------|-----------|--------|
| OpenAI | `response_format: { type: "json_schema", json_schema: {...} }` | JSON Schema (draft 2020-12 subset) |
| Anthropic | Tool use with schema, or `response_format` (limited) | JSON Schema via tool definitions |
| Google | `response_schema` in `generation_config` | Subset of OpenAPI 3.0 schema |
| Mistral | `response_format: { type: "json_object" }` | Basic JSON mode |

**Critical Gateway Implications**:

- **Schema translation**: Different schema dialect support across providers means the gateway may need to translate schemas.
- **Routing decision input**: Structured output requests may benefit from different routing strategies (some models are better at schema adherence).
- **Validation**: The gateway could validate responses against the declared schema before returning to clients.

### 2.6 Multimodal API Patterns

**Status**: Rapidly expanding across providers

Multimodal capabilities are diverging significantly:

| Capability | OpenAI | Anthropic | Google |
|-----------|--------|-----------|--------|
| Image Input | Vision via `image_url` content parts | `image` content blocks (base64/URL) | Inline `image/...` MIME parts |
| Audio Input | Whisper transcription + GPT-4o native | Not supported natively | Gemini native audio |
| Audio Output | TTS API + Realtime API | Not supported | Gemini Live |
| Video Input | Not supported | Not supported | Gemini native video |
| PDF Input | Via file upload | `document` content blocks | Via file upload |

**Critical Gateway Implications**:

- **Content format translation**: Image references are specified differently across providers (URL vs. base64 vs. file reference).
- **Capability-aware routing**: Not all models support all modalities. Routing must consider modality requirements.
- **Payload size**: Multimodal requests can be orders of magnitude larger than text-only requests. Backpressure and quota systems need adjustment.
- **Cost divergence**: Multimodal tokens are priced differently per provider.

### 2.7 Agent-Native API Patterns

**Status**: Emerging (OpenAI Agents SDK, Anthropic Computer Use, Strands Agents)

The shift to "agent-native" APIs introduces patterns where a single API call triggers complex multi-step execution:

- **Tool loops**: The model calls tools, observes results, and may call additional tools before producing a final response. Some providers handle this server-side (OpenAI Responses API with built-in tools), while others expect the client to orchestrate the loop.
- **Multi-step reasoning**: Extended thinking / chain-of-thought with intermediate tool calls.
- **MCP integration**: OpenAI and Anthropic both support MCP tool definitions in API calls, enabling standardized tool integration.
- **A2A protocol**: Google's Agent-to-Agent protocol for inter-agent communication.

**Critical Gateway Implications**:

- **Multiple LLM calls per request**: A single user request may trigger 3-10+ internal LLM calls during a tool loop. The gateway needs to track and bill for all internal calls.
- **Timeout management**: Tool loops can run for minutes. Standard request timeouts are insufficient.
- **Intermediate state visibility**: Observability of intermediate tool call/result steps within an agent loop.
- **Tool call authorization**: Which tools can an agent call? Policy enforcement needs to operate at the tool-call level, not just the initial request level.

### 2.8 Human-in-the-Loop (HITL) Approval Patterns

**Status**: Emerging across frameworks

HITL patterns enable human oversight of AI agent actions:

- **Pre-execution approval**: Agent proposes a tool call, waits for human approval before execution.
- **Confidence-based gating**: Only require approval for low-confidence or high-impact actions.
- **Approval workflows**: Slack/email notifications with approve/reject actions.
- **Audit trails**: Complete records of all proposed actions, approvals, and outcomes.
- **Escalation**: Automatic escalation if approval is not received within a timeout.

**Critical Gateway Implications**:

- **Asynchronous request lifecycle**: Requests may be paused for minutes to hours awaiting approval.
- **State persistence**: Pending approval state must survive gateway restarts (needs external storage).
- **WebSocket/SSE for notifications**: Real-time notification of pending approvals to human reviewers.
- **Integration with existing auth**: HITL approvals need to integrate with RBAC (who can approve what).

---

## 3. RouteIQ Current API Surface Analysis

Based on codebase analysis, RouteIQ's current API surface is:

### 3.1 Endpoints Provided by RouteIQ

| Category | Endpoint(s) | Source |
|----------|-------------|--------|
| Health | `/_health/live`, `/_health/ready` | `routes.py` (health_router) |
| A2A Agents | `GET/POST /a2a/agents`, `DELETE /agents/{id}` | `routes.py` (admin_router, llmrouter_router) |
| MCP REST | `GET/POST/PUT/DELETE /llmrouter/mcp/servers/*`, `/llmrouter/mcp/tools/*` | `routes.py` (llmrouter_router, admin_router) |
| MCP Parity | `/v1/mcp/server/*`, `/mcp-rest/*` | `mcp_parity.py` |
| MCP JSON-RPC | `POST /mcp` | `mcp_jsonrpc.py` |
| MCP SSE | `GET /mcp/sse`, `POST /mcp/messages` | `mcp_sse_transport.py` |
| MCP Proxy | `/mcp/{server_id}/*` | `mcp_parity.py` (feature-flagged) |
| Config | `POST /llmrouter/reload`, `/config/reload`, `GET /config/sync/status` | `routes.py` |
| Router Info | `GET /router/info` | `routes.py` |

### 3.2 Endpoints Inherited from LiteLLM

| Category | Endpoint(s) | Status |
|----------|-------------|--------|
| Chat Completions | `POST /v1/chat/completions`, `/chat/completions` | Fully instrumented by RouteIQ |
| **Responses API** | `POST /v1/responses`, `/responses`, `/openai/v1/responses` | **Available but NOT instrumented by RouteIQ routing** |
| Embeddings | `POST /v1/embeddings` | Available, not instrumented |
| Models | `GET /v1/models` | Available |
| **Realtime** | `WS /v1/realtime` | **Available as WebSocket pass-through, not instrumented** |
| Agents (DB) | `POST/GET/DELETE /v1/agents`, `POST /a2a/{agent_id}` | Available |
| Completions | `POST /v1/completions` | Available, not instrumented |
| Images | `POST /v1/images/generations` | Available, not instrumented |
| Audio | `POST /v1/audio/*` | Available, not instrumented |
| Files | `POST /v1/files` | Available |
| Batches | `POST /v1/batches` | Available |

### 3.3 RouteIQ Instrumentation Coverage

The following RouteIQ systems only instrument Chat Completions:

1. **`RouterDecisionMiddleware`** (`router_decision_callback.py`, line 77-79):
   ```python
   CHAT_COMPLETION_PATHS = {
       "/v1/chat/completions",
       "/chat/completions",
   }
   ```
   Only emits TG4.1 router decision span attributes for these two paths.

2. **`PolicyEngine`** (`policy_engine.py`): Model extraction for policy evaluation references `/chat/completions` and `/v1/completions`.

3. **`LLMRouterStrategyFamily`** (`strategies.py`): ML-based routing operates via the monkey-patched `get_available_deployment()` on LiteLLM's Router, which is invoked for chat completions routing.

4. **`PluginCallbackBridge`** (`plugin_callback_bridge.py`): Bridges LiteLLM's callback system (`log_pre_api_call`, `log_success_event`, `log_failure_event`) to plugin hooks. These callbacks fire for all LiteLLM-proxied LLM calls, potentially including Responses API calls if LiteLLM routes them through the same callback path.

### 3.4 Existing Capabilities Relevant to Emerging Standards

| Capability | Status | Details |
|-----------|--------|---------|
| SSE Streaming | Implemented | MCP SSE transport with sessions, heartbeats, max duration |
| Plugin Middleware Hooks | Implemented | on_request/on_response at ASGI level for all paths |
| Plugin LLM Hooks | Implemented | on_llm_pre_call/on_llm_success/on_llm_failure via callback bridge |
| Policy Engine | Implemented | OPA-style pre-request policy evaluation at ASGI layer |
| RBAC | Implemented | Permission-based access control |
| Audit Logging | Implemented | Action-level audit trail |
| Circuit Breakers | Implemented | Per-dependency circuit breakers (DB, Redis) |
| Backpressure | Implemented | Concurrency-bounded ASGI middleware |
| WebSocket Support | NOT implemented | RouteIQ adds no WebSocket handling; LiteLLM provides it |

---

## 4. Readiness Assessment Matrix

| Capability | Readiness | Score | Notes |
|-----------|-----------|-------|-------|
| **Responses API Proxying** | HIGH | 8/10 | LiteLLM handles it upstream; RouteIQ passes through |
| **Responses API Routing** | LOW | 2/10 | RouterDecisionMiddleware only covers chat/completions |
| **Responses API Telemetry** | LOW | 2/10 | Router span attributes not emitted for /v1/responses |
| **Responses API Policy** | MODERATE | 5/10 | PolicyEngine ASGI middleware covers all paths, but model extraction is limited |
| **Structured Output Routing** | NONE | 0/10 | No awareness of response_format in routing decisions |
| **Multimodal Request Routing** | NONE | 0/10 | No modality detection in routing strategies |
| **Real-time/WebSocket** | NONE | 0/10 | No WebSocket handling in RouteIQ layer |
| **Tool Use Loop Proxying** | HIGH | 7/10 | LiteLLM handles tool loops; callback bridge fires for each step |
| **HITL Approval Workflows** | NONE | 0/10 | No approval state machine or notification system |
| **Provider Format Translation** | HIGH | 8/10 | LiteLLM provides extensive cross-provider translation |
| **Backward Compatibility** | HIGH | 9/10 | Chat Completions API remains fully supported alongside new APIs |
| **Background/Async Responses** | MODERATE | 6/10 | LiteLLM supports background mode with Redis polling |

---

## 5. Gap Analysis

### 5.1 Critical Gaps (Break Current Value Proposition)

#### Gap 1: Routing Instrumentation Limited to Chat Completions

**Impact**: HIGH
**Description**: `RouterDecisionMiddleware` only instruments `/v1/chat/completions` and `/chat/completions`. All other LLM API calls (`/v1/responses`, `/v1/embeddings`, `/v1/completions`) bypass RouteIQ's routing telemetry and ML-based strategy selection entirely.

**What breaks**: As clients migrate from Chat Completions to Responses API, RouteIQ's primary value -- ML-based intelligent routing -- silently stops working. Traffic shifts to the new API surface without any routing intelligence.

**Location**: `src/litellm_llmrouter/router_decision_callback.py`, lines 77-79.

#### Gap 2: No Statefulness-Aware Routing

**Impact**: HIGH
**Description**: Responses API's `previous_response_id` creates provider affinity for chained conversations. RouteIQ's routing strategies are stateless and do not track which provider served previous responses. If RouteIQ routes a chained request to a different provider, it will fail.

**What breaks**: Multi-turn conversations using Responses API's native state management.

#### Gap 3: Policy Engine Model Extraction Limited

**Impact**: MODERATE
**Description**: The policy engine extracts model names from request bodies for `/chat/completions`-style paths. Responses API requests use a different body format (with `input` instead of `messages`), and the model field may be in different locations.

**Location**: `src/litellm_llmrouter/policy_engine.py`.

### 5.2 Significant Gaps (Missing Features for New Patterns)

#### Gap 4: No WebSocket Middleware

**Impact**: HIGH for real-time use cases
**Description**: RouteIQ's middleware stack (RequestID, Policy, Plugin, RouterDecision, Backpressure) is all HTTP-only. WebSocket connections for Realtime API and Gemini Live bypass all RouteIQ middleware.

#### Gap 5: No Structured Output Awareness in Routing

**Impact**: MODERATE
**Description**: Routing strategies do not consider `response_format` when making decisions. Some models are significantly better at structured output adherence than others.

#### Gap 6: No Multimodal Awareness in Routing

**Impact**: MODERATE
**Description**: Routing strategies do not detect multimodal content (images, audio, video) in requests. Not all models support all modalities, and multimodal requests have vastly different cost profiles.

#### Gap 7: No HITL Approval System

**Impact**: MODERATE (for enterprise/agent use cases)
**Description**: No mechanism for pausing agent tool calls for human approval.

### 5.3 Minor Gaps

#### Gap 8: No Built-in Tool Usage Visibility

**Impact**: LOW
**Description**: When Responses API uses built-in tools (web search, code interpreter), the gateway has no visibility into these intermediate steps.

#### Gap 9: No Provider-Specific Feature Passthrough Tracking

**Impact**: LOW
**Description**: Features like Anthropic's citations, prompt caching, and extended thinking are passed through but not tracked or optimized for.

---

## 6. Architecture Recommendations

### 6.1 Responses API Adapter Design

**Priority**: P0 (Critical)
**Effort**: Medium

**Recommendation**: Extend `RouterDecisionMiddleware` to instrument all LLM API paths.

```
Current state:
  POST /v1/chat/completions  -->  [RouterDecisionMiddleware]  -->  [LiteLLM Router]
  POST /v1/responses         -->  [no RouteIQ instrumentation] -->  [LiteLLM Router]

Proposed state:
  POST /v1/chat/completions  -->  [UnifiedLLMMiddleware]  -->  [LiteLLM Router]
  POST /v1/responses         -->  [UnifiedLLMMiddleware]  -->  [LiteLLM Router]
  POST /v1/embeddings        -->  [UnifiedLLMMiddleware]  -->  [LiteLLM Router]
  POST /v1/completions       -->  [UnifiedLLMMiddleware]  -->  [LiteLLM Router]
```

**Design**:

1. Replace `CHAT_COMPLETION_PATHS` with a comprehensive path registry:
   ```python
   LLM_API_PATHS = {
       "/v1/chat/completions": "chat_completion",
       "/chat/completions": "chat_completion",
       "/v1/responses": "responses",
       "/responses": "responses",
       "/openai/v1/responses": "responses",
       "/v1/embeddings": "embedding",
       "/v1/completions": "completion",
   }
   ```

2. Add API-type-specific model extraction (Responses API uses `input` not `messages`).

3. Emit API type as a span attribute (`router.api_type = "responses"`) for telemetry segmentation.

4. For `previous_response_id` requests, add a span attribute indicating a chained conversation and skip ML routing (delegate to provider affinity).

**Statefulness Handling**:

- Introduce a `ConversationAffinityTracker` that maps `response_id` to `provider_deployment`.
- On requests with `previous_response_id`, look up the affinity and override routing.
- Affinity data stored in Redis (for HA) or in-memory (single-node).
- TTL-based expiry for affinity records (matching Responses API session lifetime).

### 6.2 Multimodal Routing Strategy

**Priority**: P1 (Important)
**Effort**: Medium

**Recommendation**: Add modality detection to the routing pipeline.

1. **Modality Detector**: Inspect request payloads to extract modality requirements (text, image, audio, video).
2. **Modality-Aware Routing**: Filter candidate deployments by supported modalities before applying ML routing strategy.
3. **Cost-Aware Modality Routing**: Factor in modality-specific cost rates when selecting providers.

### 6.3 WebSocket / Real-time Support Architecture

**Priority**: P2 (Important for completeness)
**Effort**: Large

**Recommendation**: Implement a WebSocket middleware layer parallel to the HTTP middleware stack.

```
HTTP Request Flow (existing):
  Request --> RequestID --> Policy --> Plugin --> RouterDecision --> Backpressure --> App

WebSocket Connection Flow (proposed):
  WS Connect --> WSRequestID --> WSPolicy --> WSConnectionTracker --> App
  WS Frame   --> WSFrameAudit --> App
  WS Close   --> WSCleanup --> App
```

Key components:
- **WSConnectionMiddleware**: Connection IDs, connection limits, policy at connection time.
- **WSConnectionTracker**: Active connections for health checks and drain management.
- **WSRealtimeRouter**: Provider selection at connection establishment time.

### 6.4 HITL Approval Plugin Design

**Priority**: P2 (Important for enterprise)
**Effort**: Large

**Recommendation**: Implement HITL as a GatewayPlugin with `MIDDLEWARE` and `TOOL_RUNTIME` capabilities.

Key components:
1. **ApprovalPolicy**: Config defining which tool calls require approval.
2. **ApprovalQueue**: Redis-backed queue with TTL-based expiry.
3. **NotificationService**: Webhooks (Slack, email) for approval notifications.
4. **Approval API**: `GET/POST /admin/approvals/*` endpoints.
5. **Integration point**: `on_llm_pre_call` plugin hook.

### 6.5 Provider Format Translation Layer

**Priority**: P1 (Important)
**Effort**: Handled Upstream

**Recommendation**: Rely on LiteLLM's existing translation layer but extend monitoring with telemetry for translation success/failure rates per provider, latency impact, and feature parity gaps.

### 6.6 API Version Negotiation

**Priority**: P2 (Forward-looking)
**Effort**: Medium

**Recommendation**: Implement version detection and version-aware routing with `X-RouteIQ-API-Version` header support and automatic format detection.

---

## 7. Implementation Priorities

### Phase 1: Close Critical Gaps (Weeks 1-4)

| Item | Description | Effort |
|------|-------------|--------|
| 7.1.1 | Extend `RouterDecisionMiddleware` to cover `/v1/responses` and other LLM paths | Small |
| 7.1.2 | Update `PolicyEngine` model extraction for Responses API request format | Small |
| 7.1.3 | Add Responses API path to telemetry contracts | Small |
| 7.1.4 | Implement `ConversationAffinityTracker` for `previous_response_id` routing | Medium |

### Phase 2: Structured Output and Multimodal (Weeks 5-8)

| Item | Description | Effort |
|------|-------------|--------|
| 7.2.1 | Implement `ModalityDetector` pre-routing filter | Medium |
| 7.2.2 | Add structured output awareness to routing decision telemetry | Small |
| 7.2.3 | Add modality and structured output as routing strategy inputs | Medium |
| 7.2.4 | Add translation monitoring telemetry | Small |

### Phase 3: WebSocket and HITL (Weeks 9-16)

| Item | Description | Effort |
|------|-------------|--------|
| 7.3.1 | Implement `WSConnectionMiddleware` (feature-flagged plugin) | Large |
| 7.3.2 | Implement HITL Approval Plugin (approval queue + notification) | Large |
| 7.3.3 | Add HITL approval API endpoints | Medium |
| 7.3.4 | WebSocket connection tracking in drain manager | Medium |

---

## 8. Risk Assessment

### 8.1 Risk: Upstream LiteLLM Churn

**Probability**: HIGH | **Impact**: MODERATE
LiteLLM is rapidly evolving (RouteIQ pins `>=1.81.3`). New releases may change internal APIs or break the monkey-patch strategy.
**Mitigation**: Pin to minor versions. Test upgrades in CI. Monitor changelog.

### 8.2 Risk: API Standard Fragmentation

**Probability**: HIGH | **Impact**: MODERATE
OpenAI, Anthropic, and Google may continue diverging their APIs.
**Mitigation**: Leverage LiteLLM's translation layer. Track provider-specific features without full parity.

### 8.3 Risk: Stateful API Breaking Routing Model

**Probability**: HIGH | **Impact**: HIGH
As more APIs become stateful, per-request routing becomes less meaningful. Routing decisions shift from per-request to per-session.
**Mitigation**: Implement session-level routing with affinity tracking. This is the most architecturally significant change needed.

### 8.4 Risk: WebSocket Adoption Without Gateway Support

**Probability**: MODERATE | **Impact**: HIGH
Lack of WebSocket middleware means no policy enforcement, telemetry, or rate limiting on real-time traffic.
**Mitigation**: Phase 3 WebSocket middleware. Document current limitations in the interim.

### 8.5 Risk: ML Routing Strategy Irrelevance

**Probability**: LOW | **Impact**: HIGH
As providers add built-in capabilities, model selection becomes less important than capability selection.
**Mitigation**: Evolve routing strategies to consider capabilities (supports_tools, supports_vision, supports_structured_output) alongside text embedding similarity.

---

## Appendix A: Codebase References

| File | Relevance |
|------|-----------|
| `src/litellm_llmrouter/router_decision_callback.py` | RouterDecisionMiddleware -- main gap location |
| `src/litellm_llmrouter/strategies.py` | 18+ ML routing strategies |
| `src/litellm_llmrouter/strategy_registry.py` | A/B testing and hot-swap routing pipeline |
| `src/litellm_llmrouter/routing_strategy_patch.py` | LiteLLM Router monkey-patch |
| `src/litellm_llmrouter/policy_engine.py` | ASGI policy enforcement |
| `src/litellm_llmrouter/gateway/app.py` | App factory / composition root |
| `src/litellm_llmrouter/gateway/plugin_manager.py` | Plugin system |
| `src/litellm_llmrouter/gateway/plugin_callback_bridge.py` | LLM lifecycle hook bridge |
| `src/litellm_llmrouter/gateway/plugin_middleware.py` | ASGI plugin middleware (HTTP only) |
| `src/litellm_llmrouter/mcp_sse_transport.py` | SSE transport (pattern for WebSocket) |
| `src/litellm_llmrouter/resilience.py` | Backpressure / drain |
| `src/litellm_llmrouter/routes.py` | All RouteIQ routes |
| `reference/litellm/litellm/proxy/response_api_endpoints/endpoints.py` | LiteLLM Responses API (upstream) |
| `reference/litellm/litellm/responses/main.py` | LiteLLM Responses API core (upstream) |
| `reference/litellm/litellm/realtime_api/main.py` | LiteLLM Realtime API (upstream) |
| `reference/litellm/litellm/proxy/proxy_server.py` | LiteLLM proxy with WebSocket routes (upstream) |

## Appendix B: LiteLLM Responses API Upstream Support

LiteLLM (as of v1.81.3+) provides the following Responses API support that RouteIQ inherits:

1. **Endpoints**: `POST /v1/responses`, `/responses`, `/openai/v1/responses`
2. **Background mode**: Redis-based polling with `background: true`
3. **Streaming**: `BaseResponsesAPIStreamingIterator` for SSE streaming
4. **Transformation**: `LiteLLMCompletionTransformationHandler` translates Responses API to Chat Completions for non-OpenAI providers
5. **Session handling**: `session_handler.py` manages `previous_response_id` state
6. **Provider adapters**: OpenAI, Gemini, Volcengine, Manus, and proxy-to-proxy adapters
7. **Router integration**: `ResponsesAPIDeploymentCheck` for pre-call deployment validation
8. **MCP tools**: MCP tool integration via `litellm_proxy_mcp_handler.py`
9. **Security**: Response ID security hooks to prevent cross-tenant access

RouteIQ already has a functional Responses API proxy. The gap is that RouteIQ's own intelligence layer (routing strategies, telemetry, policy) does not participate in Responses API requests.
