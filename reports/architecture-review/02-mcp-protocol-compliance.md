# MCP Protocol Compliance Assessment - RouteIQ Gateway

**Date:** 2026-02-07
**MCP Specification Version Assessed:** 2025-03-26 (latest)
**RouteIQ Implementation Version:** 2024-11-05 (one revision behind)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [MCP Standard Overview](#2-mcp-standard-overview)
3. [RouteIQ MCP Implementation Architecture](#3-routeiq-mcp-implementation-architecture)
4. [Compliance Assessment](#4-compliance-assessment)
5. [Gap Analysis](#5-gap-analysis)
6. [Top 10 Prioritized Recommendations](#6-top-10-prioritized-recommendations)
7. [Appendix: File Reference](#7-appendix-file-reference)

---

## 1. Executive Summary

RouteIQ Gateway exposes MCP through five surfaces: JSON-RPC (`/mcp`), SSE (`/mcp/sse`), REST (`/mcp-rest/*`), parity (`/v1/mcp/*`), and proxy (`/mcp-proxy/*`). The implementation provides a solid foundation for tool discovery and invocation with strong security posture (SSRF protection, feature-flagged invocation, admin auth), but targets the **2024-11-05** protocol version rather than the current **2025-03-26** specification.

### Compliance Score: 52/100

| Category | Score | Weight | Notes |
|----------|-------|--------|-------|
| Protocol Version and Lifecycle | 4/10 | High | Targets 2024-11-05, missing initialized notification |
| Transport Support | 6/10 | High | SSE (deprecated) + partial Streamable HTTP |
| Tool Lifecycle | 7/10 | High | Good tools/list and tools/call; missing pagination, annotations |
| Resource Management | 3/10 | Medium | resources/list only; no read, templates, or subscribe |
| Prompt Templates | 0/10 | Medium | Not implemented |
| Capability Negotiation | 5/10 | High | Declares capabilities but ignores client capabilities |
| Authorization | 6/10 | Medium | OAuth flow present (feature-flagged); not MCP-native auth |
| Error Handling | 7/10 | Medium | Good JSON-RPC error codes; missing some MCP-specific codes |
| Sampling | 0/10 | Low | Not implemented |
| Observability and Tracing | 9/10 | Low | Excellent OTel integration |

---

## 2. MCP Standard Overview

The Model Context Protocol specification (2025-03-26) defines a standardized protocol for connecting AI models to external tools and data sources.

### 2.1 Protocol Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 2024-11-05 | Nov 2024 | Initial stable release |
| 2025-03-26 | Mar 2025 | Streamable HTTP replaces SSE; tool annotations; audio content; enhanced pagination |

### 2.2 Core Protocol Requirements

- **Message Format:** JSON-RPC 2.0 over HTTP
- **Lifecycle:** `initialize` request -> `initialized` notification -> operations -> shutdown
- **Capability Negotiation:** Bidirectional during initialize handshake
- **Transports:** stdio (local), SSE (deprecated), Streamable HTTP (current)
- **Server Capabilities:** tools, resources, prompts, logging, completions, experimental
- **Client Capabilities:** roots, sampling, experimental

### 2.3 Key Feature Areas

1. **Tools** - `tools/list`, `tools/call`, `notifications/tools/list_changed`
2. **Resources** - `resources/list`, `resources/read`, `resources/templates/list`, `resources/subscribe`
3. **Prompts** - `prompts/list`, `prompts/get`, `notifications/prompts/list_changed`
4. **Sampling** - `sampling/createMessage`
5. **Authorization** - OAuth 2.0 with PKCE, metadata discovery at `/.well-known/oauth-authorization-server`

---

## 3. RouteIQ MCP Implementation Architecture

### 3.1 Five MCP Surfaces

| Surface | Prefix | Module | Transport | Auth |
|---------|--------|--------|-----------|------|
| JSON-RPC | `POST /mcp` | `mcp_jsonrpc.py` | Streamable HTTP (partial) | `user_api_key_auth` |
| SSE | `GET /mcp/sse` | `mcp_sse_transport.py` | Legacy SSE | `user_api_key_auth` |
| REST | `/mcp-rest/*` | `mcp_parity.py` | REST API | `user_api_key_auth` |
| Parity | `/v1/mcp/*` | `mcp_parity.py` | REST API | user/admin auth |
| Proxy | `/mcp-proxy/*` | `mcp_parity.py` | Proxy | `admin_api_key_auth` |

### 3.2 Core Components

- **`mcp_gateway.py`** (1087 lines) - Central gateway: MCPServer, MCPTransport, MCPGateway, MCPToolResult, MCPToolDefinition. Thread-safe singleton with optional Redis HA sync.
- **`mcp_jsonrpc.py`** (589 lines) - JSON-RPC 2.0 endpoint with method dispatch table for `initialize`, `tools/list`, `tools/call`, `resources/list`.
- **`mcp_sse_transport.py`** (1260 lines) - Legacy SSE transport with session management, heartbeat, and async response queues.
- **`mcp_parity.py`** (1073 lines) - Upstream LiteLLM-compatible endpoints, OAuth flow, protocol proxy, namespace router.
- **`mcp_tracing.py`** (467 lines) - OTel instrumentation with span attributes for tool calls, server registration, health checks.

### 3.3 Key Design Decisions

1. **Gateway pattern** - RouteIQ aggregates multiple MCP servers behind a single gateway, namespacing tools as `<server_id>.<tool_name>`.
2. **Tool invocation gated** - Remote invocation disabled by default (`LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=false`).
3. **SSRF double-check** - URLs validated at both registration (no DNS) and invocation (with DNS) to prevent rebinding attacks.
4. **Custom REST protocol** - Tool invocation uses a custom `POST /mcp/tools/call` with `{"tool_name": ..., "arguments": ...}` payload rather than standard JSON-RPC `tools/call`.

---

## 4. Compliance Assessment

### 4.1 Protocol Version and Lifecycle

**Status: PARTIAL COMPLIANCE**

| Requirement | Spec | Implementation | Status |
|-------------|------|----------------|--------|
| Protocol version | `2025-03-26` | `2024-11-05` | NON-COMPLIANT |
| `initialize` request handling | Required | Implemented in `mcp_jsonrpc.py` L142-192 | COMPLIANT |
| `initialized` notification | Required after initialize response | Not implemented | NON-COMPLIANT |
| Protocol version negotiation | Required | Server returns fixed version; no negotiation logic | PARTIAL |
| Shutdown/cleanup | Optional | Not implemented | N/A |
| Client info capture | Optional | Params accepted but not stored/used | PARTIAL |

**Key Finding:** The implementation hardcodes `MCP_PROTOCOL_VERSION = "2024-11-05"` in both `mcp_jsonrpc.py` (line 52) and `mcp_sse_transport.py` (line 90). The spec requires the server to negotiate the protocol version by examining the client's `protocolVersion` and responding with the latest version both support. The current implementation ignores the client's stated version entirely.

**Critical Missing:** The `initialized` notification, which the MCP spec requires clients to send after receiving the `initialize` response, is neither expected nor validated. Per the spec: "The client MUST send this notification after receiving a successful initialize response, but before sending any other requests."

### 4.2 Transport Support

**Status: PARTIAL COMPLIANCE**

| Transport | Spec Status | Implementation | Compliance |
|-----------|-------------|----------------|------------|
| Streamable HTTP | Current (2025-03-26) | Partial via POST /mcp | PARTIAL |
| SSE (deprecated) | Deprecated (backward-compat) | Full implementation | COMPLIANT (legacy) |
| stdio | Local only | Listed in MCPTransport enum but not used | N/A for gateway |

**Streamable HTTP Analysis:**

The MCP 2025-03-26 spec defines Streamable HTTP as:
- Client sends JSON-RPC via `POST` to server endpoint
- Server can respond with either `application/json` (single response) or `text/event-stream` (streaming)
- Client may issue `GET` to receive server-initiated notifications via SSE stream
- Session management via `Mcp-Session-Id` header

RouteIQ's JSON-RPC endpoint (`POST /mcp`) partially implements this:
- Accepts JSON-RPC POST requests (COMPLIANT)
- Returns `application/json` responses only; no SSE streaming from POST (PARTIAL)
- No `Mcp-Session-Id` header management (NON-COMPLIANT)
- GET `/mcp` returns info page, not SSE notification stream (NON-COMPLIANT for Streamable HTTP)

**SSE Transport Analysis:**

The legacy SSE transport (`mcp_sse_transport.py`) follows the deprecated 2024-11-05 pattern:
- `GET /mcp/sse` establishes SSE connection with `endpoint` event (COMPLIANT with legacy spec)
- `POST /mcp/messages?sessionId=<id>` receives JSON-RPC, responds via SSE (COMPLIANT with legacy spec)
- Session management with timeouts and heartbeats (COMPLIANT)
- However, SSE is deprecated in 2025-03-26 in favor of Streamable HTTP

### 4.3 Tool Lifecycle

**Status: MOSTLY COMPLIANT**

| Requirement | Spec | Implementation | Status |
|-------------|------|----------------|--------|
| `tools/list` | Required if tools capability | Implemented | COMPLIANT |
| `tools/call` | Required if tools capability | Implemented | COMPLIANT |
| Tool name | String, unique per server | Namespaced as `server_id.tool_name` | COMPLIANT (gateway extension) |
| `description` field | Optional string | Provided | COMPLIANT |
| `inputSchema` | Required JSON Schema object | Provided (defaults to `{"type": "object"}`) | COMPLIANT |
| `annotations` field | New in 2025-03-26 | Not implemented | NON-COMPLIANT |
| Pagination (`cursor`) | Optional | Documented but not implemented | PARTIAL |
| `notifications/tools/list_changed` | Required if `listChanged: true` | Declared but not sent | NON-COMPLIANT |
| Tool result `content` array | text, image, audio, resource | Only `text` type produced | PARTIAL |
| `isError` flag | Required for tool errors | Implemented | COMPLIANT |
| Protocol vs tool errors | Distinct categories | Properly separated | COMPLIANT |

**Key Finding (Tool Content Types):** The spec defines four content types for tool results: `text`, `image`, `audio` (new in 2025-03-26), and embedded `resource`. RouteIQ only produces `text` content blocks (lines 344-351 in `mcp_jsonrpc.py`), serializing all results as JSON strings.

**Key Finding (Pagination):** The `tools/list` handler accepts a `cursor` parameter in its docstring but does not implement pagination logic.

**Key Finding (Tool Annotations):** The 2025-03-26 spec introduces tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) that help clients make decisions about tool execution. These are not implemented.

### 4.4 Resource Management

**Status: MINIMALLY COMPLIANT**

| Requirement | Spec | Implementation | Status |
|-------------|------|----------------|--------|
| `resources/list` | Required if resources declared | Implemented (basic) | PARTIAL |
| `resources/read` | Required if resources declared | Not implemented | NON-COMPLIANT |
| `resources/templates/list` | Optional | Not implemented | N/A |
| `resources/subscribe` | Optional | Declared `subscribe: false` | COMPLIANT (opted out) |
| `notifications/resources/list_changed` | Required if `listChanged: true` | Declared but not sent | NON-COMPLIANT |

**Key Finding:** `resources/list` is implemented but `resources/read` is not. The spec states: "If a server provides resources, it MUST implement `resources/list` and `resources/read`." Since the server declares `resources` in capabilities, this is a **spec violation**.

### 4.5 Prompt Templates

**Status: NOT IMPLEMENTED** - Compliant by omission since prompts not declared in capabilities.

### 4.6 Capability Negotiation

**Status: PARTIAL COMPLIANCE**

Server declares `tools` and `resources` capabilities but ignores client capabilities entirely during the initialize handshake. The initialize handler accepts `protocolVersion`, `capabilities`, and `clientInfo` but does not use them.

### 4.7 Authorization

**Status: PARTIAL COMPLIANCE**

Feature-flagged OAuth flow exists (`MCP_OAUTH_ENABLED`, default false) but `/.well-known/oauth-authorization-server` metadata discovery is not implemented.

### 4.8 Error Handling

**Status: MOSTLY COMPLIANT**

Standard JSON-RPC error codes implemented correctly. However, `-32002` is used for `MCP_TOOL_INVOCATION_DISABLED` while the MCP spec reserves it for "resource not found." This is a semantic collision.

### 4.9 Sampling

**Status: NOT IMPLEMENTED** - Acceptable for server/gateway role.

### 4.10 Observability and Tracing

**Status: EXCELLENT** - Comprehensive OTel instrumentation with span attributes, evaluator hooks, and graceful degradation.

---

## 5. Gap Analysis

### 5.1 Critical Gaps (Spec Violations)

| ID | Gap | Impact | Affected Files |
|----|-----|--------|----------------|
| G1 | Protocol version stuck at 2024-11-05 | Clients targeting 2025-03-26 may reject | `mcp_jsonrpc.py:52`, `mcp_sse_transport.py:90` |
| G2 | Missing `initialized` notification handling | Spec-compliant clients send it; server ignores | `mcp_jsonrpc.py`, `mcp_sse_transport.py` |
| G3 | `resources/read` not implemented despite declaring resources | Violates MUST requirement | `mcp_jsonrpc.py`, `mcp_gateway.py` |
| G4 | `notifications/tools/list_changed` never sent | Declared capability never delivered | `mcp_jsonrpc.py`, `mcp_sse_transport.py` |
| G5 | Error code `-32002` semantic collision | Spec uses for resource-not-found | `mcp_jsonrpc.py:99` |

### 5.2 Significant Gaps (Missing Features)

| ID | Gap | Impact | Priority |
|----|-----|--------|----------|
| G6 | No Streamable HTTP session management (`Mcp-Session-Id`) | No stateful sessions | High |
| G7 | No protocol version negotiation logic | No graceful version handling | High |
| G8 | No pagination for tools/list | Scalability issue | Medium |
| G9 | Tool results only support `text` content type | No multimodal outputs | Medium |
| G10 | `/.well-known/oauth-authorization-server` missing | No OAuth auto-discovery | Medium |
| G11 | Client capabilities ignored during initialize | No adaptive behavior | Medium |
| G12 | No tool annotations | No safety metadata | Medium |

### 5.3 Minor Gaps (Enhancements)

| ID | Gap | Priority |
|----|-----|----------|
| G13 | No `prompts/list` or `prompts/get` | Low |
| G14 | No `resources/templates/list` | Low |
| G15 | Custom tool invocation protocol (non-standard REST) | Low |
| G16 | SSE uses deprecated transport pattern | Low |
| G17 | No `logging` capability | Low |
| G18 | No `completions` capability | Low |

### 5.4 Architecture Observations

1. **Dual-path invocation** - Gateway invokes upstream servers via custom REST (`POST /mcp/tools/call` with `{"tool_name", "arguments"}`) rather than MCP JSON-RPC. Cannot act as transparent MCP proxy.
2. **Tool namespacing** - `server_id.tool_name` is non-standard but reasonable for multi-server gateway.
3. **Five surfaces, one registry** - All surfaces share MCPGateway singleton ensuring consistency.
4. **Health check simulation** - `check_server_health()` does not make real HTTP requests; validates URL format only.

---

## 6. Top 10 Prioritized Recommendations

### 1. Upgrade Protocol Version to 2025-03-26

**Priority:** CRITICAL | **Effort:** Low | **Gaps:** G1, G7

Update `MCP_PROTOCOL_VERSION` from `"2024-11-05"` to `"2025-03-26"` and implement version negotiation:
- If client sends `2025-03-26`, respond with `2025-03-26`
- If client sends `2024-11-05`, respond with `2024-11-05` for backward compatibility
- If client sends unsupported version, return `-32602` error

**Files:** `src/litellm_llmrouter/mcp_jsonrpc.py` (line 52, `_handle_initialize`), `src/litellm_llmrouter/mcp_sse_transport.py` (line 90, `_handle_initialize_sse`)

### 2. Implement `initialized` Notification Handling

**Priority:** CRITICAL | **Effort:** Low | **Gaps:** G2

Add `notifications/initialized` to the method dispatch table. Per MCP spec, this is a notification (no `id` field, no response expected). The server should:
1. Accept the notification without returning a response
2. Optionally track that the session is fully initialized
3. Handle for both HTTP and SSE transports

**Files:** `src/litellm_llmrouter/mcp_jsonrpc.py` (add to `METHOD_HANDLERS`), `src/litellm_llmrouter/mcp_sse_transport.py` (add to `_dispatch_jsonrpc_method`)

### 3. Implement `resources/read` Method

**Priority:** HIGH | **Effort:** Medium | **Gaps:** G3

The server declares `resources` in capabilities but does not implement `resources/read`. Options:
1. Implement `resources/read` to proxy reads to upstream servers
2. Remove `resources` from the capabilities object

Option 1 is recommended. Handler needs to accept `uri`, find owning server, proxy the read, return content with `uri`, `mimeType`, and `text` or `blob`.

**Files:** `src/litellm_llmrouter/mcp_jsonrpc.py`, `src/litellm_llmrouter/mcp_sse_transport.py`, `src/litellm_llmrouter/mcp_gateway.py`

### 4. Implement Streamable HTTP Session Management

**Priority:** HIGH | **Effort:** Medium | **Gaps:** G6

The Streamable HTTP transport (2025-03-26) requires:
1. Server issues `Mcp-Session-Id` header in initialize response
2. Client includes `Mcp-Session-Id` in subsequent requests
3. Server rejects requests with invalid/expired session IDs (HTTP 404)
4. Server terminates sessions via HTTP 405 on DELETE

Reuse the `SSESession` pattern from `mcp_sse_transport.py`.

**Files:** `src/litellm_llmrouter/mcp_jsonrpc.py`

### 5. Fix Error Code Collision for `-32002`

**Priority:** HIGH | **Effort:** Low | **Gaps:** G5

The MCP spec reserves `-32002` for "resource not found." RouteIQ uses it for `MCP_TOOL_INVOCATION_DISABLED`. Reassign:
- `MCP_TOOL_INVOCATION_DISABLED` -> `-32004` (unused custom code)
- Reserve `-32002` for resource-not-found

**Files:** `src/litellm_llmrouter/mcp_jsonrpc.py` (line 99), `src/litellm_llmrouter/mcp_sse_transport.py` (lines 809, 1085)

### 6. Implement `notifications/tools/list_changed` Delivery

**Priority:** HIGH | **Effort:** Medium | **Gaps:** G4

The server declares `tools.listChanged: true` but never sends the notification. Implement:
1. When `register_server()` or `unregister_server()` modifies tools, emit notification
2. For SSE: push event to all active sessions
3. For Streamable HTTP: clients establish GET SSE stream for notifications

**Files:** `src/litellm_llmrouter/mcp_gateway.py`, `src/litellm_llmrouter/mcp_sse_transport.py`, `src/litellm_llmrouter/mcp_jsonrpc.py`

### 7. Add Pagination to `tools/list` and `resources/list`

**Priority:** MEDIUM | **Effort:** Medium | **Gaps:** G8

Implement cursor-based pagination:
- Accept optional `cursor` parameter
- Return `nextCursor` when more results available
- Use server-side cursor encoding (base64-encoded offset)
- Default page size configurable via environment variable

**Files:** `src/litellm_llmrouter/mcp_jsonrpc.py`, `src/litellm_llmrouter/mcp_sse_transport.py`

### 8. Support Multiple Tool Result Content Types

**Priority:** MEDIUM | **Effort:** Medium | **Gaps:** G9

Currently all results serialized as `{"type": "text", "text": json.dumps(result)}`. Extend to support:
- `ImageContent`: `{"type": "image", "data": "<base64>", "mimeType": "image/png"}`
- `AudioContent` (2025-03-26): `{"type": "audio", "data": "<base64>", "mimeType": "audio/wav"}`
- `EmbeddedResource`: `{"type": "resource", "resource": {"uri": "...", "text": "..."}}`

Parse upstream responses for content type passthrough.

**Files:** `src/litellm_llmrouter/mcp_jsonrpc.py`, `src/litellm_llmrouter/mcp_sse_transport.py`, `src/litellm_llmrouter/mcp_gateway.py`

### 9. Implement Tool Annotations

**Priority:** MEDIUM | **Effort:** Low | **Gaps:** G12

The 2025-03-26 spec introduces tool annotations: `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`. Extend `MCPToolDefinition` to include `annotations` field and propagate through `tools/list`.

**Files:** `src/litellm_llmrouter/mcp_gateway.py`, `src/litellm_llmrouter/mcp_jsonrpc.py`, `src/litellm_llmrouter/mcp_sse_transport.py`

### 10. Add OAuth Authorization Server Metadata Discovery

**Priority:** MEDIUM | **Effort:** Low | **Gaps:** G10

Implement `/.well-known/oauth-authorization-server` endpoint per RFC 8414 / MCP spec with issuer, authorization_endpoint, token_endpoint, registration_endpoint, response_types_supported, grant_types_supported, and code_challenge_methods_supported fields.

**Files:** `src/litellm_llmrouter/mcp_parity.py`, `src/litellm_llmrouter/routes.py`

---

## 7. Appendix: File Reference

### Source Files Analyzed

| File | Path | Lines | Role |
|------|------|-------|------|
| mcp_gateway.py | `src/litellm_llmrouter/mcp_gateway.py` | 1087 | Core gateway, server registry, tool invocation |
| mcp_jsonrpc.py | `src/litellm_llmrouter/mcp_jsonrpc.py` | 589 | JSON-RPC 2.0 endpoint |
| mcp_sse_transport.py | `src/litellm_llmrouter/mcp_sse_transport.py` | 1260 | Legacy SSE transport |
| mcp_parity.py | `src/litellm_llmrouter/mcp_parity.py` | 1073 | Upstream-compatible endpoints, OAuth, proxy |
| mcp_tracing.py | `src/litellm_llmrouter/mcp_tracing.py` | 467 | OTel tracing instrumentation |
| routes.py | `src/litellm_llmrouter/routes.py` | ~1300 | Route registration |

### Test Files Analyzed

| File | Path | Coverage Area |
|------|------|---------------|
| test_mcp_parity.py | `tests/test_mcp_parity.py` | Parity endpoints, OAuth, proxy, feature flags |
| test_mcp_tool_invocation.py | `tests/test_mcp_tool_invocation.py` | Tool invocation, SSRF, timeouts, auth tokens |
| test_mcp_gateway_properties.py | `tests/property/test_mcp_gateway_properties.py` | Property-based tests for gateway operations |
| test_mcp_jsonrpc.py | `tests/unit/test_mcp_jsonrpc.py` | JSON-RPC endpoint handlers |
| test_mcp_sse_transport.py | `tests/unit/test_mcp_sse_transport.py` | SSE transport sessions and events |
| test_mcp_tracing.py | `tests/unit/test_mcp_tracing.py` | OTel span creation and attributes |

### MCP Specification Pages Referenced

| Page | URL |
|------|-----|
| Specification Overview | https://modelcontextprotocol.io/specification/2025-03-26 |
| Lifecycle | https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle |
| Transports | https://modelcontextprotocol.io/specification/2025-03-26/basic/transports |
| Authorization | https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization |
| Tools | https://modelcontextprotocol.io/specification/2025-03-26/server/tools |
| Resources | https://modelcontextprotocol.io/specification/2025-03-26/server/resources |
| Prompts | https://modelcontextprotocol.io/specification/2025-03-26/server/prompts |

---

*Report generated as part of the RouteIQ Architecture Review series.*
