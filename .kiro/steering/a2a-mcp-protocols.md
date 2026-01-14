---
inclusion: fileMatch
fileMatchPattern: "**/*gateway*.py"
---

# A2A and MCP Protocol Reference

## A2A Gateway (Agent-to-Agent)

A2A is Google's protocol for agent-to-agent communication, allowing AI agents to discover and communicate with each other.

Reference: https://google.github.io/A2A/

### Enable A2A Gateway

```bash
A2A_GATEWAY_ENABLED=true
```

### A2AAgent Dataclass

```python
@dataclass
class A2AAgent:
    agent_id: str
    name: str
    description: str
    url: str
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### A2A Agent Card Format

The `get_agent_card()` method returns A2A-compliant agent cards:

```python
{
    "name": "My Agent",
    "description": "A helpful agent",
    "url": "http://agent-backend:8080",
    "capabilities": {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": False
    },
    "skills": [
        {"id": "chat", "name": "Chat"},
        {"id": "code", "name": "Code"}
    ]
}
```

### A2A Gateway Methods

| Method | Description |
|--------|-------------|
| `register_agent(agent)` | Register an agent |
| `unregister_agent(agent_id)` | Remove an agent |
| `get_agent(agent_id)` | Get agent by ID |
| `list_agents()` | List all agents |
| `discover_agents(capability)` | Find agents by capability |
| `get_agent_card(agent_id)` | Get A2A-compliant agent card |

---

## MCP Gateway (Model Context Protocol)

MCP is Anthropic's protocol for connecting AI models to external tools and data sources.

Reference: https://modelcontextprotocol.io/

### Enable MCP Gateway

```bash
MCP_GATEWAY_ENABLED=true
```

### MCPServer Dataclass

```python
@dataclass
class MCPServer:
    server_id: str
    name: str
    url: str
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP
    tools: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    auth_type: str = "none"  # none, api_key, bearer_token, oauth2
    metadata: dict[str, Any] = field(default_factory=dict)
```

### MCP Transport Types

```python
class MCPTransport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"
```

### MCP Gateway Methods

| Method | Description |
|--------|-------------|
| `register_server(server)` | Register an MCP server |
| `unregister_server(server_id)` | Remove a server |
| `get_server(server_id)` | Get server by ID |
| `list_servers()` | List all servers |
| `list_tools()` | List all tools from all servers |
| `list_resources()` | List all resources from all servers |

---

## Singleton Pattern

Both gateways use singleton instances:

```python
from litellm_llmrouter.a2a_gateway import get_a2a_gateway
from litellm_llmrouter.mcp_gateway import get_mcp_gateway

a2a = get_a2a_gateway()
mcp = get_mcp_gateway()
```

## Adding New Gateway Features

1. Add methods to the gateway class
2. Add corresponding routes in `routes.py`
3. Use Pydantic models for request/response validation
4. Check `is_enabled()` before operations
5. Log operations with `verbose_proxy_logger`

```python
def new_feature(self, param: str) -> dict:
    if not self.enabled:
        verbose_proxy_logger.warning("Gateway not enabled")
        return {"error": "Gateway disabled"}
    
    # Implementation
    verbose_proxy_logger.info(f"Feature executed: {param}")
    return {"status": "success"}
```
