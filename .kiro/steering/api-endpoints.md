---
inclusion: fileMatch
fileMatchPattern: "**/routes.py"
---

# API Endpoints Reference

## LiteLLM Core Endpoints

These are provided by LiteLLM proxy server:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/images/generations` | POST | Image generation |
| `/v1/audio/transcriptions` | POST | Audio transcription |
| `/v1/audio/speech` | POST | Text-to-speech |
| `/v1/moderations` | POST | Content moderation |
| `/v1/batches` | POST | Batch processing |
| `/v1/rerank` | POST | Document reranking |
| `/health/liveliness` | GET | Liveness check |
| `/health/readiness` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |

## LLMRouter Extension Endpoints

### A2A Gateway

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/a2a/agents` | GET | List all registered agents |
| `/a2a/agents` | POST | Register a new agent |
| `/a2a/agents/{agent_id}` | GET | Get agent details |
| `/a2a/agents/{agent_id}` | DELETE | Unregister agent |
| `/a2a/agents/{agent_id}/card` | GET | Get A2A agent card |

#### Register Agent Request

```json
{
  "agent_id": "my-agent",
  "name": "My Agent",
  "description": "A helpful agent",
  "url": "http://agent-backend:8080",
  "capabilities": ["chat", "code"],
  "metadata": {}
}
```

### MCP Gateway

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp/servers` | GET | List all MCP servers |
| `/mcp/servers` | POST | Register a new server |
| `/mcp/servers/{server_id}` | GET | Get server details |
| `/mcp/servers/{server_id}` | DELETE | Unregister server |
| `/mcp/tools` | GET | List all available tools |
| `/mcp/resources` | GET | List all available resources |

#### Register Server Request

```json
{
  "server_id": "github-mcp",
  "name": "GitHub MCP Server",
  "url": "http://mcp-server:8080",
  "transport": "streamable_http",
  "tools": ["create_issue", "list_prs"],
  "resources": ["repo_contents"],
  "auth_type": "bearer_token",
  "metadata": {}
}
```

### Hot Reload

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/router/reload` | POST | Reload routing strategy |
| `/config/reload` | POST | Reload configuration |
| `/config/sync/status` | GET | Get config sync status |
| `/router/info` | GET | Get router information |

#### Reload Request

```json
{
  "strategy": "llmrouter-knn",  // Optional, null for all
  "force_sync": true            // Force S3/GCS sync first
}
```

#### Reload Response

```json
{
  "status": "success",  // or "partial", "failed"
  "reloaded": ["llmrouter-knn"],
  "errors": []
}
```

#### Sync Status Response

```json
{
  "enabled": true,
  "hot_reload_enabled": true,
  "sync_interval_seconds": 60,
  "s3": {
    "enabled": true,
    "bucket": "my-bucket",
    "key": "config/config.yaml",
    "last_etag": "abc123"
  },
  "reload_count": 5,
  "last_sync_time": 1705142400.0,
  "running": true
}
```

## Authentication

All endpoints require authentication when `master_key` is configured:

```bash
curl -H "Authorization: Bearer sk-1234" \
  http://localhost:4000/v1/chat/completions
```

## Error Responses

### 401 Unauthorized
```json
{
  "error": {
    "message": "Invalid API key",
    "type": "authentication_error",
    "code": "invalid_api_key"
  }
}
```

### 404 Not Found
```json
{
  "detail": "Agent my-agent not found"
}
```

### 429 Rate Limited
```json
{
  "error": {
    "message": "Rate limit exceeded",
    "type": "rate_limit_error",
    "code": "rate_limit_exceeded"
  }
}
```

### 500 Internal Error
```json
{
  "error": {
    "message": "Internal server error",
    "type": "internal_error",
    "code": "internal_error"
  }
}
```

## Pydantic Models

### AgentRegistration

```python
class AgentRegistration(BaseModel):
    agent_id: str
    name: str
    description: str
    url: str
    capabilities: list[str] = []
    metadata: dict[str, Any] = {}
```

### ServerRegistration

```python
class ServerRegistration(BaseModel):
    server_id: str
    name: str
    url: str
    transport: str = "streamable_http"
    tools: list[str] = []
    resources: list[str] = []
    auth_type: str = "none"
    metadata: dict[str, Any] = {}
```

### ReloadRequest

```python
class ReloadRequest(BaseModel):
    strategy: str | None = None
    force_sync: bool = False
```

## Adding New Endpoints

1. Define Pydantic model for request/response
2. Add route function with appropriate decorator
3. Use `get_*_gateway()` to access singleton instances
4. Raise `HTTPException` for errors
5. Return consistent response format

```python
@router.post("/my/endpoint")
async def my_endpoint(request: MyRequest):
    gateway = get_my_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="Feature not enabled")
    
    result = gateway.do_something(request)
    return {"status": "success", "data": result}
```
