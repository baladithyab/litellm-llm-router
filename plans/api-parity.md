# API Parity Analysis: RouteIQ Gateway vs Default LiteLLM

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing.

This document analyzes the API parity between our custom RouteIQ Gateway implementation and the default LiteLLM proxy server.

## Executive Summary

| Category | LiteLLM Default | Our Implementation | Parity Status |
|----------|-----------------|-------------------|---------------|
| Core LLM Endpoints | ✅ Full | ✅ Inherited | ✅ Complete |
| Health Endpoints | ✅ Full | ✅ Inherited | ✅ Complete |
| A2A Gateway | ✅ Full (beta) | ⚠️ Simplified | 🔄 Full Parity Planned |
| MCP Gateway | ✅ Full | ⚠️ Simplified | 🔄 Full Parity Planned |
| Hot Reload | ❌ Limited | ✅ Extended | ✅ Enhanced |
| Management Endpoints | ✅ Full | ✅ Inherited | ✅ Complete |

---

## 1. Core LLM Endpoints (Inherited from LiteLLM)

These endpoints are provided by the LiteLLM proxy server and are fully available in our implementation.

### Chat & Completions

| Endpoint | Method | Status | Notes |
|----------|--------|--------|-------|
| `/v1/chat/completions` | POST | ✅ Inherited | OpenAI-compatible chat |
| `/chat/completions` | POST | ✅ Inherited | Alias |
| `/v1/completions` | POST | ✅ Inherited | Text completions |
| `/completions` | POST | ✅ Inherited | Alias |
| `/engines/{model}/chat/completions` | POST | ✅ Inherited | Azure-compatible |
| `/openai/deployments/{model}/chat/completions` | POST | ✅ Inherited | Azure-compatible |

### Embeddings

| Endpoint | Method | Status | Notes |
|----------|--------|--------|-------|
| `/v1/embeddings` | POST | ✅ Inherited | Text embeddings |
| `/embeddings` | POST | ✅ Inherited | Alias |
| `/engines/{model}/embeddings` | POST | ✅ Inherited | Azure-compatible |

### Audio

| Endpoint | Method | Status | Notes |
|----------|--------|--------|-------|
| `/v1/audio/speech` | POST | ✅ Inherited | Text-to-speech |
| `/audio/speech` | POST | ✅ Inherited | Alias |
| `/v1/audio/transcriptions` | POST | ✅ Inherited | Speech-to-text |
| `/audio/transcriptions` | POST | ✅ Inherited | Alias |

### Images

| Endpoint | Method | Status | Notes |
|----------|--------|--------|-------|
| `/v1/images/generations` | POST | ✅ Inherited | Image generation |

### Other Core Endpoints

| Endpoint | Method | Status | Notes |
|----------|--------|--------|-------|
| `/v1/moderations` | POST | ✅ Inherited | Content moderation |
| `/v1/models` | GET | ✅ Inherited | List models |
| `/v1/models/{model_id}` | GET | ✅ Inherited | Get model info |
| `/v1/batches` | POST | ✅ Inherited | Batch processing |
| `/v1/rerank` | POST | ✅ Inherited | Document reranking |

---

## 2. Health Endpoints

| Endpoint | Method | LiteLLM | Ours | Notes |
|----------|--------|---------|------|-------|
| `/health` | GET | ✅ | ✅ Inherited | Model health check |
| `/health/liveliness` | GET | ✅ | ✅ Inherited | Liveness probe |
| `/health/readiness` | GET | ✅ | ✅ Inherited | Readiness probe |
| `/health/services` | GET | ✅ | ✅ Inherited | Service health (Slack, Langfuse, etc.) |
| `/test` | GET | ✅ | ✅ Inherited | Deprecated, use liveliness |
| `/metrics` | GET | ✅ | ✅ Inherited | Prometheus metrics |

---

## 3. A2A (Agent-to-Agent) Gateway

### LiteLLM Default Implementation

LiteLLM has a comprehensive A2A implementation with database persistence:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/agents` | GET | List all agents (with permission filtering) |
| `/v1/agents` | POST | Create agent (DB persistence) |
| `/v1/agents/{agent_id}` | GET | Get agent by ID |
| `/v1/agents/{agent_id}` | PUT | Update agent |
| `/v1/agents/{agent_id}` | PATCH | Partial update agent |
| `/v1/agents/{agent_id}` | DELETE | Delete agent |
| `/v1/agents/{agent_id}/make_public` | POST | Make agent public |
| `/v1/agents/make_public` | POST | Make multiple agents public |
| `/a2a/{agent_id}` | POST | Invoke agent (JSON-RPC 2.0) |
| `/a2a/{agent_id}/message/send` | POST | Send message to agent |
| `/a2a/{agent_id}/.well-known/agent-card.json` | GET | Get agent card |
| `/agent/daily/activity` | GET | Agent analytics |

### Our Implementation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/a2a/agents` | GET | List agents (in-memory) |
| `/a2a/agents` | POST | Register agent (in-memory) |
| `/a2a/agents/{agent_id}` | GET | Get agent details |
| `/a2a/agents/{agent_id}` | DELETE | Unregister agent |
| `/a2a/agents/{agent_id}/card` | GET | Get A2A agent card |

### Gap Analysis - A2A

| Feature | LiteLLM | Ours | Priority | Status |
|---------|---------|------|----------|--------|
| Database persistence | ✅ | ❌ In-memory only | High | 🔄 Planned (Task 9.6) |
| Agent invocation (JSON-RPC) | ✅ | ❌ Missing | High | 🔄 Planned (Task 9.2) |
| PUT/PATCH updates | ✅ | ❌ Missing | Medium | 🔄 Planned (Task 9.8) |
| Permission filtering | ✅ | ❌ Missing | Medium | 🔄 Planned (Task 9.10) |
| Make public endpoints | ✅ | ❌ Missing | Low | 🔄 Planned (Task 9.10) |
| Analytics | ✅ | ❌ Missing | Low | 🔄 Planned (Task 9.11) |
| Streaming support | ✅ | ❌ Missing | Medium | 🔄 Planned (Task 9.4) |

---

## 4. MCP (Model Context Protocol) Gateway

### LiteLLM Default Implementation

LiteLLM has extensive MCP support with OAuth, registry, and management:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/mcp/server` | GET | List all MCP servers |
| `/v1/mcp/server` | POST | Create MCP server (DB) |
| `/v1/mcp/server/{server_id}` | GET | Get server details |
| `/v1/mcp/server/{server_id}` | PUT | Update server |
| `/v1/mcp/server/{server_id}` | DELETE | Delete server |
| `/v1/mcp/server/health` | GET | Health check servers |
| `/v1/mcp/tools` | GET | List all MCP tools |
| `/v1/mcp/access_groups` | GET | List access groups |
| `/v1/mcp/registry.json` | GET | MCP registry (discovery) |
| `/v1/mcp/server/oauth/session` | POST | Temporary OAuth session |
| `/v1/mcp/server/oauth/{server_id}/authorize` | GET | OAuth authorize |
| `/v1/mcp/server/oauth/{server_id}/token` | POST | OAuth token exchange |
| `/mcp/tools/list` | GET | List tools (REST API) |
| `/mcp/tools/call` | POST | Call tool (REST API) |
| `/.well-known/oauth-*` | GET | OAuth discovery |

### Our Implementation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp/servers` | GET | List servers (in-memory) |
| `/mcp/servers` | POST | Register server (in-memory) |
| `/mcp/servers/{server_id}` | GET | Get server details |
| `/mcp/servers/{server_id}` | DELETE | Unregister server |
| `/mcp/tools` | GET | List all tools |
| `/mcp/resources` | GET | List all resources |

### Gap Analysis - MCP

| Feature | LiteLLM | Ours | Priority | Status |
|---------|---------|------|----------|--------|
| Database persistence | ✅ | ❌ In-memory only | High | 🔄 Planned (Task 10.5) |
| Tool invocation | ✅ | ❌ Missing | High | 🔄 Planned (Task 10.3) |
| OAuth support | ✅ | ❌ Missing | Medium | 🔄 Planned (Task 10.7) |
| Registry endpoint | ✅ | ❌ Missing | Medium | 🔄 Planned (Task 10.11) |
| Health checks | ✅ | ❌ Missing | Medium | 🔄 Planned (Task 10.9) |
| Access groups | ✅ | ❌ Missing | Low | 🔄 Planned (Task 10.14) |
| PUT updates | ✅ | ❌ Missing | Low | 🔄 Planned (Task 10.13) |

---

## 5. Hot Reload & Config Sync (Our Extension)

These endpoints are **unique to our implementation** and extend LiteLLM's capabilities:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/router/reload` | POST | Reload routing strategy |
| `/config/reload` | POST | Reload configuration |
| `/config/sync/status` | GET | Get config sync status |
| `/router/info` | GET | Get router information |

### Features Not in Default LiteLLM

- ✅ Dynamic strategy hot-reload without restart
- ✅ S3/GCS config sync with ETag caching
- ✅ Per-strategy reload capability
- ✅ Sync status monitoring

---

## 6. Management Endpoints (Inherited)

These are fully inherited from LiteLLM:

### Key Management
- `/key/generate` - Generate API keys
- `/key/update` - Update keys
- `/key/delete` - Delete keys
- `/key/info` - Key information

### User Management
- `/user/new` - Create user
- `/user/update` - Update user
- `/user/delete` - Delete user
- `/user/info` - User information

### Team Management
- `/team/new` - Create team
- `/team/update` - Update team
- `/team/delete` - Delete team
- `/team/info` - Team information

### Model Management
- `/model/new` - Add model
- `/model/update` - Update model
- `/model/delete` - Delete model
- `/model/info` - Model information
- `/v2/model/info` - Enhanced model info

### Budget Management
- `/budget/new` - Create budget
- `/budget/update` - Update budget
- `/budget/delete` - Delete budget
- `/budget/info` - Budget information

---

## 7. Additional LiteLLM Endpoints (Inherited)

### Assistants API
| Endpoint | Method | Status |
|----------|--------|--------|
| `/v1/assistants` | GET/POST | ✅ Inherited |
| `/v1/assistants/{id}` | DELETE | ✅ Inherited |
| `/v1/threads` | POST | ✅ Inherited |
| `/v1/threads/{id}` | GET | ✅ Inherited |
| `/v1/threads/{id}/messages` | GET/POST | ✅ Inherited |
| `/v1/threads/{id}/runs` | POST | ✅ Inherited |

### Vector Stores
| Endpoint | Method | Status |
|----------|--------|--------|
| `/v1/vector_stores` | POST | ✅ Inherited |
| `/v1/vector_stores/{id}/search` | POST | ✅ Inherited |
| `/v1/vector_stores/{id}/files` | GET/POST | ✅ Inherited |
| `/vector_store/new` | POST | ✅ Inherited |
| `/vector_store/list` | GET | ✅ Inherited |

### Files API
| Endpoint | Method | Status |
|----------|--------|--------|
| `/v1/files` | GET/POST | ✅ Inherited |
| `/v1/files/{id}` | GET/DELETE | ✅ Inherited |
| `/v1/files/{id}/content` | GET | ✅ Inherited |

### Fine-tuning
| Endpoint | Method | Status |
|----------|--------|--------|
| `/v1/fine_tuning/jobs` | GET/POST | ✅ Inherited |
| `/v1/fine_tuning/jobs/{id}` | GET | ✅ Inherited |
| `/v1/fine_tuning/jobs/{id}/cancel` | POST | ✅ Inherited |

### Videos (New)
| Endpoint | Method | Status |
|----------|--------|--------|
| `/v1/videos` | GET/POST | ✅ Inherited |
| `/v1/videos/{id}` | GET | ✅ Inherited |
| `/v1/videos/{id}/content` | GET | ✅ Inherited |
| `/v1/videos/{id}/remix` | POST | ✅ Inherited |

---

## 8. Recommendations

### High Priority (Planned for Implementation)

1. **A2A Agent Invocation**: ✅ Planned - Add `/a2a/{agent_id}` POST endpoint for JSON-RPC 2.0 message handling
2. **MCP Tool Invocation**: ✅ Planned - Add `/mcp/tools/call` POST endpoint for tool execution
3. **Database Persistence**: ✅ Planned - Add PostgreSQL persistence for A2A/MCP registrations

### Medium Priority (Planned for Implementation)

4. **A2A Streaming**: ✅ Planned - Support `message/stream` method for streaming responses
5. **MCP OAuth**: ✅ Planned - Add OAuth flow support for MCP server authentication
6. **Health Checks**: ✅ Planned - Add health check endpoints for MCP servers
7. **PUT/PATCH Updates**: ✅ Planned - Add update endpoints for A2A agents and MCP servers

### Low Priority (Planned for Implementation)

8. **MCP Registry**: ✅ Planned - Add `/v1/mcp/registry.json` for MCP discovery
9. **Access Groups**: ✅ Planned - Add MCP access group management
10. **Analytics**: ✅ Planned - Add agent activity analytics

### Implementation Status

All gaps identified above have been documented in the spec files and are planned for implementation:
- Requirements: `.kiro/specs/production-ai-gateway/requirements.md` (Requirements 7.7-7.14, 8.7-8.15)
- Design: `.kiro/specs/production-ai-gateway/design.md` (Properties 23-31)
- Tasks: `.kiro/specs/production-ai-gateway/tasks.md` (Tasks 9.2-9.11, 10.3-10.14)

---

## 9. API Compatibility Matrix

| Client Expectation | Supported | Notes |
|-------------------|-----------|-------|
| OpenAI SDK | ✅ | Full compatibility |
| Azure OpenAI SDK | ✅ | Full compatibility |
| Anthropic SDK | ✅ | Via passthrough |
| Google AI SDK | ✅ | Via passthrough |
| A2A SDK | ⚠️ | Discovery works, invocation missing |
| MCP SDK | ⚠️ | Registration works, tool calls missing |
| LiteLLM Python SDK | ✅ | Full compatibility |

---

## 10. Version Information

- **LiteLLM Reference Version**: Latest (from submodule)
- **Analysis Date**: January 2026
- **Our Implementation**: RouteIQ Gateway v1.0

---

## Appendix: Endpoint URL Differences

Our implementation uses slightly different URL patterns for A2A and MCP:

| Feature | LiteLLM Default | Our Implementation |
|---------|-----------------|-------------------|
| A2A agents list | `/v1/agents` | `/a2a/agents` |
| A2A agent CRUD | `/v1/agents/{id}` | `/a2a/agents/{id}` |
| A2A invocation | `/a2a/{id}` | ❌ Not implemented |
| MCP servers list | `/v1/mcp/server` | `/mcp/servers` |
| MCP server CRUD | `/v1/mcp/server/{id}` | `/mcp/servers/{id}` |
| MCP tools | `/v1/mcp/tools` | `/mcp/tools` |

Consider aligning URL patterns with LiteLLM defaults for better ecosystem compatibility.
