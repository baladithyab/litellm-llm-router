# Design Document: Production AI Gateway with LiteLLM + LLMRouter

## Overview

This design document describes a production-ready AI Gateway that integrates LiteLLM (unified LLM API gateway) with LLMRouter (ML-based intelligent routing). The system provides a single OpenAI-compatible API endpoint that can route requests to 100+ LLM providers using 18+ intelligent routing strategies, with enterprise features including high availability, persistence, observability, hot reload, and support for modern protocols (A2A, MCP).

### Key Design Principles

1. **Zero-Downtime Updates**: Hot reload for routing models and configurations without service restart
2. **Horizontal Scalability**: Stateless design with Redis for distributed state and PostgreSQL for persistence
3. **Observability First**: Comprehensive tracing, metrics, and logging for production debugging
4. **Protocol Extensibility**: Support for emerging standards (A2A, MCP) alongside OpenAI compatibility
5. **ML-Powered Intelligence**: Use trained routing models to optimize for cost, latency, and quality

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Client Applications                              │
│                    (OpenAI SDK, HTTP Clients, etc.)                      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ HTTP/HTTPS
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Load Balancer (Nginx)                           │
│                      (Optional for HA deployment)                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
┌───────────────────────────┐       ┌───────────────────────────┐
│   Gateway Instance 1      │       │   Gateway Instance N      │
│  ┌─────────────────────┐  │       │  ┌─────────────────────┐  │
│  │  LiteLLM Proxy      │  │       │  │  LiteLLM Proxy      │  │
│  │  (FastAPI Server)   │  │       │  │  (FastAPI Server)   │  │
│  └──────────┬──────────┘  │       │  └──────────┬──────────┘  │
│             │              │       │             │              │
│  ┌──────────▼──────────┐  │       │  ┌──────────▼──────────┐  │
│  │  LLMRouter          │  │       │  │  LLMRouter          │  │
│  │  Strategy Family    │  │       │  │  Strategy Family    │  │
│  │  (18+ Strategies)   │  │       │  │  (18+ Strategies)   │  │
│  └──────────┬──────────┘  │       │  └──────────┬──────────┘  │
│             │              │       │             │              │
│  ┌──────────▼──────────┐  │       │  ┌──────────▼──────────┐  │
│  │  Extension Routes   │  │       │  │  Extension Routes   │  │
│  │  • A2A Gateway      │  │       │  │  • A2A Gateway      │  │
│  │  • MCP Gateway      │  │       │  │  • MCP Gateway      │  │
│  │  • Hot Reload API   │  │       │  │  • Hot Reload API   │  │
│  └─────────────────────┘  │       │  └─────────────────────┘  │
└───────────┬───────────────┘       └───────────┬───────────────┘
            │                                   │
            └───────────────┬───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Redis      │    │ PostgreSQL   │    │  S3 / GCS    │
│  (Cache &    │    │ (Persistence │    │  (Config &   │
│   State)     │    │  & Logs)     │    │   Models)    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   LLM Providers       │
                │  • OpenAI             │
                │  • Anthropic          │
                │  • AWS Bedrock        │
                │  • Azure OpenAI       │
                │  • 100+ others        │
                └───────────────────────┘
```

### Request Flow

1. **Client Request**: Client sends OpenAI-compatible request to Gateway
2. **Authentication**: Gateway validates API key against master_key or database
3. **Routing Decision**: LLMRouter strategy selects optimal model based on query
4. **Cache Check**: Gateway checks Redis cache for identical previous request
5. **LLM Invocation**: LiteLLM forwards request to selected provider
6. **Response Processing**: Gateway processes response, updates cache, logs metrics
7. **Client Response**: Gateway returns OpenAI-compatible response to client

### Data Flow for Hot Reload

```
┌─────────────────────────────────────────────────────────────────┐
│                    Config Sync Manager                          │
│                   (Background Thread)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Periodic Check (ETag-based)
                             ▼
                    ┌────────────────┐
                    │   S3 / GCS     │
                    │  (Remote       │
                    │   Storage)     │
                    └────────┬───────┘
                             │
                             │ Download if ETag changed
                             ▼
                    ┌────────────────┐
                    │  Local Config  │
                    │  /app/config/  │
                    └────────┬───────┘
                             │
                             │ SIGHUP signal
                             ▼
                    ┌────────────────┐
                    │  LiteLLM Proxy │
                    │  (Reload)      │
                    └────────────────┘
```

## Components and Interfaces

### 1. LiteLLM Proxy Server

**Responsibility**: Unified API gateway providing OpenAI-compatible interface to 100+ LLM providers.

**Key Interfaces**:
- `/v1/chat/completions` - Chat completion endpoint (OpenAI-compatible)
- `/v1/embeddings` - Text embeddings endpoint
- `/v1/images/generations` - Image generation endpoint
- `/v1/audio/transcriptions` - Audio transcription endpoint
- `/v1/audio/speech` - Text-to-speech endpoint
- `/v1/moderations` - Content moderation endpoint
- `/v1/batches` - Batch processing endpoint
- `/v1/rerank` - Document reranking endpoint
- `/health/liveliness` - Health check for orchestrators
- `/health/readiness` - Readiness check for traffic routing

**Configuration**:
```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
      
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: os.environ/REDIS_HOST
    port: 6379
    ttl: 3600
```

### 2. LLMRouter Strategy Family

**Responsibility**: ML-based routing strategies for selecting optimal LLM based on query characteristics.

**Class**: `LLMRouterStrategyFamily`

**Key Methods**:
- `__init__(strategy_name, model_path, llm_data_path, hot_reload, reload_interval, **kwargs)` - Initialize strategy with configuration
- `router` (property) - Get router instance, loading/reloading as needed
- `_load_router()` - Load the appropriate LLMRouter model based on strategy name
- `_should_reload()` - Check if model should be reloaded based on file mtime and interval

**Supported Strategies**:
- **Single-round**: `llmrouter-knn`, `llmrouter-svm`, `llmrouter-mlp`, `llmrouter-mf`, `llmrouter-elo`, `llmrouter-routerdc`, `llmrouter-hybrid`, `llmrouter-causallm`, `llmrouter-graph`, `llmrouter-automix`
- **Multi-round**: `llmrouter-r1`
- **Personalized**: `llmrouter-gmt`
- **Agentic**: `llmrouter-knn-multiround`, `llmrouter-llm-multiround`
- **Baseline**: `llmrouter-smallest`, `llmrouter-largest`
- **Custom**: `llmrouter-custom`

**Configuration**:
```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router
    llm_data_path: /app/config/llm_candidates.json
    hot_reload: true
    reload_interval: 300
```

**Thread Safety**: Uses `threading.RLock()` for thread-safe model access and reloading.

### 3. A2A Gateway

**Responsibility**: Agent-to-Agent protocol support for agent communication with full LiteLLM parity.

**Class**: `A2AGateway`

**Key Methods**:
- `register_agent(agent: A2AAgent)` - Register a new A2A agent
- `unregister_agent(agent_id: str)` - Remove an agent
- `get_agent(agent_id: str)` - Retrieve agent by ID
- `update_agent(agent_id: str, agent: A2AAgent)` - Full update of agent
- `patch_agent(agent_id: str, updates: dict)` - Partial update of agent
- `discover_agents(capability: str, user_id: str, team_id: str)` - Find agents by capability with permission filtering
- `get_agent_card(agent_id: str)` - Get A2A protocol agent card
- `invoke_agent(agent_id: str, message: JSONRPCRequest)` - Invoke agent via JSON-RPC 2.0
- `stream_agent_response(agent_id: str, message: JSONRPCRequest)` - Stream agent response via SSE
- `get_daily_activity(agent_id: str, start_date: date, end_date: date)` - Get agent usage analytics

**Data Model**:
```python
@dataclass
class A2AAgent:
    agent_id: str
    name: str
    description: str
    url: str
    capabilities: list[str]
    metadata: dict[str, Any]
    team_id: str | None = None
    user_id: str | None = None
    is_public: bool = False
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class JSONRPCRequest:
    jsonrpc: str = "2.0"
    method: str  # "message/send" or "message/stream"
    params: dict[str, Any]
    id: str | int

@dataclass
class A2AMessage:
    role: str  # "user" or "agent"
    parts: list[A2AMessagePart]
    
@dataclass
class A2AMessagePart:
    type: str  # "text", "file", "data"
    content: str | bytes
    mime_type: str | None = None
```

**API Endpoints**:
- `GET /v1/agents` - List all agents (with permission filtering)
- `POST /v1/agents` - Register new agent (DB persistence)
- `GET /v1/agents/{agent_id}` - Get agent details
- `PUT /v1/agents/{agent_id}` - Full update agent
- `PATCH /v1/agents/{agent_id}` - Partial update agent
- `DELETE /v1/agents/{agent_id}` - Unregister agent
- `POST /v1/agents/{agent_id}/make_public` - Make agent public
- `POST /a2a/{agent_id}` - Invoke agent (JSON-RPC 2.0)
- `GET /a2a/{agent_id}/.well-known/agent-card.json` - Get agent card
- `GET /agent/daily/activity` - Agent analytics

**Database Schema**:
```sql
CREATE TABLE a2a_agents (
    agent_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    url VARCHAR(1024) NOT NULL,
    capabilities JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    team_id VARCHAR(255),
    user_id VARCHAR(255),
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_a2a_agents_team ON a2a_agents(team_id);
CREATE INDEX idx_a2a_agents_user ON a2a_agents(user_id);
CREATE INDEX idx_a2a_agents_public ON a2a_agents(is_public);
```

### 4. MCP Gateway

**Responsibility**: Model Context Protocol support for tool and context integration with full LiteLLM parity.

**Class**: `MCPGateway`

**Key Methods**:
- `register_server(server: MCPServer)` - Register a new MCP server
- `unregister_server(server_id: str)` - Remove a server
- `get_server(server_id: str)` - Retrieve server by ID
- `update_server(server_id: str, server: MCPServer)` - Full update of server
- `list_tools()` - List all available tools across servers
- `list_resources()` - List all available resources across servers
- `call_tool(server_id: str, tool_name: str, arguments: dict)` - Invoke an MCP tool
- `get_registry()` - Get MCP registry for discovery
- `check_server_health(server_id: str)` - Check MCP server health
- `get_access_groups()` - List MCP access groups
- `create_oauth_session(server_id: str)` - Create temporary OAuth session
- `authorize_oauth(server_id: str, redirect_uri: str)` - OAuth authorization
- `exchange_oauth_token(server_id: str, code: str)` - OAuth token exchange

**Data Model**:
```python
@dataclass
class MCPServer:
    server_id: str
    name: str
    url: str
    transport: MCPTransport  # streamable_http, sse, stdio
    tools: list[MCPTool]
    resources: list[MCPResource]
    auth_type: str  # none, api_key, bearer_token, oauth2
    oauth_config: MCPOAuthConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    team_id: str | None = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    server_id: str

@dataclass
class MCPResource:
    uri: str
    name: str
    description: str
    mime_type: str
    server_id: str

@dataclass
class MCPOAuthConfig:
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    scopes: list[str]

@dataclass
class MCPToolCallRequest:
    server_id: str
    tool_name: str
    arguments: dict[str, Any]

@dataclass
class MCPToolCallResponse:
    content: list[dict[str, Any]]
    is_error: bool = False
```

**API Endpoints**:
- `GET /v1/mcp/server` - List all servers
- `POST /v1/mcp/server` - Register new server (DB persistence)
- `GET /v1/mcp/server/{server_id}` - Get server details
- `PUT /v1/mcp/server/{server_id}` - Full update server
- `DELETE /v1/mcp/server/{server_id}` - Unregister server
- `GET /v1/mcp/server/health` - Health check all servers
- `GET /v1/mcp/tools` - List all tools
- `GET /mcp/tools/list` - List tools (REST API)
- `POST /mcp/tools/call` - Call tool (REST API)
- `GET /v1/mcp/registry.json` - MCP registry for discovery
- `GET /v1/mcp/access_groups` - List access groups
- `POST /v1/mcp/server/oauth/session` - Create OAuth session
- `GET /v1/mcp/server/oauth/{server_id}/authorize` - OAuth authorize
- `POST /v1/mcp/server/oauth/{server_id}/token` - OAuth token exchange
- `GET /.well-known/oauth-authorization-server` - OAuth discovery

**Database Schema**:
```sql
CREATE TABLE mcp_servers (
    server_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    url VARCHAR(1024) NOT NULL,
    transport VARCHAR(50) NOT NULL,
    auth_type VARCHAR(50) DEFAULT 'none',
    oauth_config JSONB,
    metadata JSONB DEFAULT '{}',
    team_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE mcp_tools (
    tool_id UUID PRIMARY KEY,
    server_id UUID REFERENCES mcp_servers(server_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    input_schema JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE mcp_resources (
    resource_id UUID PRIMARY KEY,
    server_id UUID REFERENCES mcp_servers(server_id) ON DELETE CASCADE,
    uri VARCHAR(1024) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    mime_type VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE mcp_access_groups (
    group_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    server_ids JSONB DEFAULT '[]',
    team_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_mcp_servers_team ON mcp_servers(team_id);
CREATE INDEX idx_mcp_tools_server ON mcp_tools(server_id);
CREATE INDEX idx_mcp_resources_server ON mcp_resources(server_id);
```

### 5. Hot Reload Manager

**Responsibility**: Manage hot reload operations for routing models and configurations.

**Class**: `HotReloadManager`

**Key Methods**:
- `register_router_reload_callback(strategy_name, callback)` - Register reload callback for a strategy
- `reload_router(strategy: str | None)` - Reload specific or all strategies
- `reload_config(force_sync: bool)` - Reload configuration, optionally syncing from remote
- `get_router_info()` - Get information about current routing configuration

**API Endpoints**:
- `POST /router/reload` - Trigger router reload
- `POST /config/reload` - Trigger config reload
- `GET /router/info` - Get router information

### 6. Config Sync Manager

**Responsibility**: Background synchronization of configuration from S3/GCS with ETag-based change detection.

**Class**: `ConfigSyncManager`

**Key Methods**:
- `start()` - Start background sync thread
- `stop()` - Stop background sync
- `force_sync()` - Force immediate sync from remote
- `get_status()` - Get current sync status
- `_get_s3_etag()` - Get S3 object ETag without downloading
- `_download_from_s3_if_changed()` - Download only if ETag changed
- `_trigger_reload()` - Trigger config reload via SIGHUP signal

**Configuration**:
```bash
CONFIG_S3_BUCKET=my-bucket
CONFIG_S3_KEY=config/config.yaml
CONFIG_HOT_RELOAD=true
CONFIG_SYNC_INTERVAL=60
```

**ETag-Based Optimization**: Uses S3 ETag to avoid unnecessary downloads, only downloading when remote config has actually changed.

**API Endpoints**:
- `GET /config/sync/status` - Get sync status

### 7. OpenTelemetry Observability

**Responsibility**: Unified observability via OpenTelemetry for traces, logs, and metrics.

**Components**:
- **Tracer**: Emits spans for routing decisions, LLM calls, cache operations
- **Logger**: Structured logging with trace correlation
- **Meter**: Metrics for request count, latency, errors, costs

**Configuration**:
```yaml
litellm_settings:
  success_callback: ["prometheus", "otel"]
  otel_config:
    endpoint: "http://otel-collector:4317"
    service_name: "litellm-gateway"
    traces_exporter: "otlp"
    logs_exporter: "otlp"
    metrics_exporter: "prometheus"
```

**Semantic Conventions**: Uses OpenTelemetry semantic conventions for:
- HTTP spans: `http.method`, `http.status_code`, `http.route`
- LLM spans: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`
- Custom spans: `llm.routing.strategy`, `llm.routing.selected_model`

**Trace Correlation**: All logs include `trace_id` and `span_id` for correlation with distributed traces.

**Exporters**:
- **OTLP**: OpenTelemetry Protocol for traces and logs to collectors (Jaeger, Tempo, CloudWatch)
- **Prometheus**: Pull-based metrics at `/metrics` endpoint
- **Langfuse**: LLM-specific observability platform

### 8. Startup Module

**Responsibility**: Entry point that wires all components together and starts LiteLLM proxy.

**Key Functions**:
- `register_routes_with_litellm()` - Register extension routes with LiteLLM's FastAPI app
- `register_strategies()` - Register LLMRouter strategies with LiteLLM
- `start_config_sync_if_enabled()` - Start background config sync if enabled
- `main()` - Main entry point that orchestrates startup

**Startup Sequence**:
1. Parse command-line arguments
2. Register LLMRouter strategies
3. Start config sync (if enabled)
4. Build litellm command with arguments
5. Execute litellm proxy via `os.execvp()`

## Data Models

### Configuration Schema

```yaml
# Model List - LLM provider configurations
model_list:
  - model_name: string          # Logical model name
    litellm_params:
      model: string              # Provider/model format (e.g., "openai/gpt-4")
      api_key: string            # API key (supports os.environ/ prefix)
      api_base: string           # Optional custom endpoint
      rpm: integer               # Requests per minute limit
      timeout: integer           # Request timeout in seconds
      stream_timeout: integer    # Stream timeout in seconds
    model_info:
      id: string                 # Optional model ID
      mode: string               # Optional mode (e.g., "embedding")

# Router Settings - Routing strategy configuration
router_settings:
  routing_strategy: string       # Strategy name (e.g., "llmrouter-knn")
  routing_strategy_args:
    model_path: string           # Path to trained model
    llm_data_path: string        # Path to LLM candidates JSON
    hot_reload: boolean          # Enable hot reload
    reload_interval: integer     # Reload check interval (seconds)
    model_s3_bucket: string      # Optional S3 bucket for models
    model_s3_key: string         # Optional S3 key for models
  num_retries: integer           # Number of retries on failure
  retry_after: integer           # Seconds to wait before retry
  timeout: integer               # Global timeout
  cache_responses: boolean       # Enable response caching
  redis_host: string             # Redis host for distributed state
  redis_password: string         # Redis password
  redis_port: integer            # Redis port

# General Settings - Gateway configuration
general_settings:
  master_key: string             # Master API key for admin access
  database_url: string           # PostgreSQL connection string
  store_model_in_db: boolean     # Store models in database
  proxy_budget_rescheduler_min_time: integer
  proxy_budget_rescheduler_max_time: integer
  proxy_batch_write_at: integer
  database_connection_pool_limit: integer

# LiteLLM Settings - LiteLLM-specific configuration
litellm_settings:
  cache: boolean                 # Enable response caching
  cache_params:
    type: string                 # Cache backend (e.g., "redis")
    host: string                 # Cache host
    port: integer                # Cache port
    ttl: integer                 # Cache TTL in seconds
  set_verbose: boolean           # Enable verbose logging
  success_callback: list[string] # Success callbacks (e.g., ["prometheus", "langfuse"])
  failure_callback: list[string] # Failure callbacks
  num_retries: integer           # Number of retries
  request_timeout: integer       # Request timeout
  telemetry: boolean             # Enable telemetry
  context_window_fallbacks: list # Fallback models for context limits
  default_team_settings: list    # Per-team settings
  drop_params: boolean           # Drop unsupported parameters

# MCP Servers - Model Context Protocol servers
mcp_servers:
  server_name:
    url: string                  # Server URL
    transport: string            # Transport type (streamable_http, sse, stdio)
    spec_path: string            # Optional OpenAPI spec path
    auth_type: string            # Authentication type (none, api_key, bearer_token, oauth2)
```

### LLM Candidates JSON

```json
{
  "models": [
    {
      "model_name": "gpt-4",
      "provider": "openai",
      "cost_per_1k_tokens": 0.03,
      "latency_p50_ms": 500,
      "latency_p95_ms": 1200,
      "context_window": 8192,
      "capabilities": ["chat", "function_calling"]
    }
  ]
}
```

### Database Schema

**Virtual Keys Table**:
```sql
CREATE TABLE virtual_keys (
    key_id UUID PRIMARY KEY,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_alias VARCHAR(255),
    team_id VARCHAR(255),
    user_id VARCHAR(255),
    max_budget DECIMAL(10, 2),
    budget_duration VARCHAR(50),
    budget_reset_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    metadata JSONB
);
```

**Request Logs Table**:
```sql
CREATE TABLE request_logs (
    request_id UUID PRIMARY KEY,
    key_id UUID REFERENCES virtual_keys(key_id),
    model_name VARCHAR(255),
    provider VARCHAR(255),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost DECIMAL(10, 6),
    latency_ms INTEGER,
    status VARCHAR(50),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);
```

## Error Handling

### Error Categories

1. **Authentication Errors** (401)
   - Invalid API key
   - Expired API key
   - Missing Authorization header

2. **Authorization Errors** (403)
   - Budget exceeded
   - Model access denied
   - Rate limit exceeded

3. **Validation Errors** (400)
   - Invalid request format
   - Missing required parameters
   - Unsupported model

4. **Provider Errors** (502, 503, 504)
   - Provider API unavailable
   - Provider timeout
   - Provider rate limit

5. **Internal Errors** (500)
   - Routing strategy failure
   - Database connection failure
   - Cache connection failure

### Retry Strategy

```python
# Exponential backoff with jitter
def calculate_retry_delay(attempt: int, base_delay: int = 5) -> int:
    """Calculate retry delay with exponential backoff and jitter."""
    max_delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0, 0.1 * max_delay)
    return min(max_delay + jitter, 60)  # Cap at 60 seconds
```

### Fallback Strategy

```yaml
litellm_settings:
  context_window_fallbacks:
    - gpt-3.5-turbo: [gpt-3.5-turbo-16k]
    - gpt-4: [gpt-4-32k, claude-3-opus]
```

When a request exceeds the context window, the Gateway automatically falls back to the next model in the list.

## Testing Strategy

### Unit Testing

**Scope**: Individual components and functions

**Framework**: pytest

**Coverage Areas**:
- LLMRouter strategy loading and initialization
- Config sync ETag-based change detection
- Hot reload manager callback registration
- A2A agent registration and discovery
- MCP server registration and tool listing
- Authentication and authorization logic
- Cost calculation and budget tracking

**Example Test**:
```python
def test_llmrouter_strategy_initialization():
    """Test that LLMRouter strategies initialize correctly."""
    strategy = LLMRouterStrategyFamily(
        strategy_name="llmrouter-knn",
        model_path="/app/models/knn_router",
        llm_data_path="/app/config/llm_candidates.json",
        hot_reload=True,
        reload_interval=300
    )
    assert strategy.strategy_name == "llmrouter-knn"
    assert strategy.hot_reload is True
    assert strategy.reload_interval == 300
```

### Integration Testing

**Scope**: Component interactions and end-to-end flows

**Framework**: pytest with Docker Compose

**Coverage Areas**:
- LiteLLM + LLMRouter integration
- Redis caching integration
- PostgreSQL persistence integration
- S3 config sync integration
- A2A protocol compliance
- MCP protocol compliance
- Hot reload end-to-end flow

**Example Test**:
```python
@pytest.mark.integration
async def test_routing_with_cache():
    """Test that routing decisions are cached correctly."""
    # First request - cache miss
    response1 = await client.post("/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response1.status_code == 200
    
    # Second identical request - cache hit
    response2 = await client.post("/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response2.status_code == 200
    assert response2.headers.get("X-Cache-Hit") == "true"
```

### Load Testing

**Scope**: Performance and scalability validation

**Framework**: Locust or k6

**Metrics**:
- Requests per second (RPS)
- P50, P95, P99 latency
- Error rate
- Cache hit rate
- Database connection pool utilization

**Target Performance**:
- 1000 RPS sustained
- P95 latency < 100ms (excluding LLM call)
- Error rate < 0.1%
- Cache hit rate > 80% for repeated queries

### Property-Based Testing

**Scope**: Universal properties that should hold across all inputs

**Framework**: Hypothesis (Python)

**Note**: Property-based testing will be defined in the Correctness Properties section below.


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Request Forwarding Correctness

*For any* valid client request to a supported endpoint with a configured LLM provider, the Gateway should successfully forward the request to the provider using LiteLLM's unified interface and return a response in the expected format.

**Validates: Requirements 1.2**

### Property 2: Authentication Enforcement

*For any* request to the Gateway when master_key is configured, the request should be accepted if and only if it includes a valid API key (either master_key or a valid virtual key from the database).

**Validates: Requirements 1.3, 7.5, 11.1, 11.4**

### Property 3: Configuration Loading

*For any* valid YAML configuration file containing model_list, router_settings, and general_settings, the Gateway should successfully load the configuration and make all configured models and settings available.

**Validates: Requirements 1.5**

### Property 4: Routing Strategy Selection

*For any* configured routing strategy name (either `llmrouter-*` or LiteLLM built-in), when a request is made, the Gateway should use the correct routing strategy to select a model and return a valid model name from the configured model list.

**Validates: Requirements 2.2, 2.5, 2.6**

### Property 5: Model and Config Hot Reload

*For any* model file or configuration file that changes (detected via modification time or ETag), when hot reload is enabled, the Gateway should detect the change and reload the affected component without requiring a service restart.

**Validates: Requirements 3.2, 3.4, 3.5**

### Property 6: Data Persistence

*For any* request processed by the Gateway when database_url is configured, all relevant data (virtual keys, request logs, cost data) should be persisted to PostgreSQL and be retrievable via database queries.

**Validates: Requirements 4.1, 12.2**

### Property 7: Response Caching

*For any* two identical requests made to the Gateway when caching is enabled, the second request should return a cached response (indicated by cache hit metrics) without invoking the LLM provider, and the cached response should expire after the configured TTL.

**Validates: Requirements 4.2, 13.1, 13.3**

### Property 8: Cache Key Generation

*For any* request with provider-specific optional parameters, when enable_caching_on_provider_specific_optional_params is true, the cache key should include those parameters such that requests differing only in provider-specific params produce different cache keys.

**Validates: Requirements 13.4**

### Property 9: Rate Limiting Enforcement

*For any* virtual key with a configured rate limit, when the number of requests exceeds the limit within the time window, the Gateway should reject subsequent requests with HTTP 429 status until the window resets.

**Validates: Requirements 11.6**

### Property 10: Budget Tracking and Enforcement

*For any* virtual key with a configured max_budget, the Gateway should track cumulative costs across all requests and reject requests that would exceed the budget, and the budget should reset after the configured budget_duration.

**Validates: Requirements 11.5, 12.1, 12.3, 12.6**

### Property 11: A2A Agent Registration and Discovery

*For any* A2A agent registered via configuration or API, the agent should be discoverable via the `/v1/agents` endpoint and should be filterable by capability and permission, and the agent card should be retrievable in A2A protocol format at `/.well-known/agent-card.json`.

**Validates: Requirements 7.2, 7.6, 7.13**

### Property 23: A2A Agent Invocation

*For any* registered A2A agent, when a valid JSON-RPC 2.0 request with method `message/send` is POSTed to `/a2a/{agent_id}`, the Gateway should forward the message to the agent backend and return a valid JSON-RPC 2.0 response.

**Validates: Requirements 7.8, 7.9**

### Property 24: A2A Streaming Response

*For any* registered A2A agent, when a valid JSON-RPC 2.0 request with method `message/stream` is POSTed to `/a2a/{agent_id}`, the Gateway should stream the response using Server-Sent Events with proper event formatting.

**Validates: Requirements 7.10**

### Property 25: A2A Database Persistence

*For any* A2A agent registered when database_url is configured, the agent should be persisted to PostgreSQL and should survive Gateway restarts, and should be retrievable via the `/v1/agents` endpoint after restart.

**Validates: Requirements 7.7**

### Property 26: A2A Agent Updates

*For any* registered A2A agent, PUT requests to `/v1/agents/{agent_id}` should fully replace the agent data, and PATCH requests should merge the provided fields with existing data while preserving unspecified fields.

**Validates: Requirements 7.11, 7.12**

### Property 12: MCP Server Tool Loading

*For any* MCP server configured in mcp_servers section or registered via API, the Gateway should load all tool definitions from the server and make them available in the tools list, and requests with `tools` type `mcp` should invoke the correct server.

**Validates: Requirements 8.2, 8.3, 8.4**

### Property 27: MCP Tool Invocation

*For any* registered MCP server with available tools, when a POST request is made to `/mcp/tools/call` with a valid tool name and arguments, the Gateway should invoke the tool on the MCP server and return the tool's response.

**Validates: Requirements 8.8**

### Property 28: MCP Database Persistence

*For any* MCP server registered when database_url is configured, the server and its tools should be persisted to PostgreSQL and should survive Gateway restarts, and should be retrievable via the `/v1/mcp/server` endpoint after restart.

**Validates: Requirements 8.7**

### Property 29: MCP OAuth Flow

*For any* MCP server configured with OAuth authentication, the Gateway should support the complete OAuth 2.0 authorization code flow including session creation, authorization redirect, and token exchange.

**Validates: Requirements 8.10, 8.11**

### Property 30: MCP Server Health Check

*For any* registered MCP server, when a GET request is made to `/v1/mcp/server/health`, the Gateway should check connectivity to the server and return the health status for each server.

**Validates: Requirements 8.13**

### Property 31: MCP Registry Discovery

*For any* set of registered MCP servers, the `/v1/mcp/registry.json` endpoint should return a valid MCP registry document listing all servers and their capabilities for client discovery.

**Validates: Requirements 8.12**

### Property 13: OpenAPI to MCP Conversion

*For any* valid OpenAPI specification provided via spec_path configuration, the Gateway should generate corresponding MCP tool definitions that can be invoked via the MCP protocol.

**Validates: Requirements 8.6**

### Property 14: MLOps Model Training

*For any* valid training data in LLMRouter format, the MLOps pipeline should successfully train a routing model for the specified strategy type and save model artifacts that are compatible with the Gateway's hot reload mechanism.

**Validates: Requirements 5.2, 5.3, 5.5**

### Property 15: Observability Span and Log Emission

*For any* request processed by the Gateway when observability is configured, the system should emit OpenTelemetry spans for all key events (routing decision, LLM call, cache hit/miss), emit structured logs with trace correlation IDs, and update metrics (request count, latency, error rate, cost).

**Validates: Requirements 6.3, 6.4, 6.5, 6.9, 13.5, 14.6, 15.2, 15.4, 15.5**

### Property 16: Per-Team Observability Settings

*For any* team configured in default_team_settings with specific observability callbacks, requests made with that team's virtual keys should send traces, logs, and metrics to the team-specific observability backend (e.g., team-specific Langfuse project) with proper trace correlation.

**Validates: Requirements 6.8**

### Property 17: Retry with Exponential Backoff

*For any* failed request when num_retries is configured, the Gateway should retry the request up to the specified count, and when all retries are exhausted, should return an error response with details about the failure.

**Validates: Requirements 14.1, 14.5**

### Property 18: Context Window Fallback

*For any* request that exceeds the selected model's context window when context_window_fallbacks is configured, the Gateway should automatically fall back to the next model in the fallback list and successfully process the request.

**Validates: Requirements 14.3**

### Property 19: Timeout Enforcement

*For any* request with a configured timeout (either global request_timeout or per-model timeout), the Gateway should cancel the request and return a timeout error if the LLM provider does not respond within the specified time.

**Validates: Requirements 14.4**

### Property 20: Routing Decision Logging with Trace Correlation

*For any* request processed by the Gateway, the system should log the routing decision including the strategy used and model selected via OpenTelemetry structured logging with trace correlation IDs, and when store_model_in_db is false, should not log sensitive prompt content.

**Validates: Requirements 15.2, 15.3, 15.5**

### Property 21: Error Logging with Trace Context

*For any* error that occurs during request processing, the Gateway should log the error via OpenTelemetry with a stack trace, relevant context (request ID, user ID, model name), and trace correlation IDs to enable debugging across distributed traces.

**Validates: Requirements 15.7**

### Property 22: S3 Config Sync with ETag Optimization

*For any* configuration file stored in S3, the Config Sync Manager should only download the file when the ETag changes, avoiding unnecessary downloads when the content is unchanged.

**Validates: Requirements 10.3**

