# MCP Gateway - Model Context Protocol

This guide covers the MCP (Model Context Protocol) gateway functionality for extending LLMs with external tools and data sources.

## Overview

The MCP Gateway enables integration with [Model Context Protocol](https://modelcontextprotocol.io/) servers, allowing LLMs to:

- Access external tools (search, file operations, APIs)
- Query data sources and databases
- Interact with custom services
- Use resources from MCP servers

## Enabling MCP Gateway

Set the environment variable:

```bash
MCP_GATEWAY_ENABLED=true
```

Or in docker-compose:

```yaml
environment:
  - MCP_GATEWAY_ENABLED=true
```

## API Endpoints

### Register an MCP Server

```bash
POST /mcp/servers
Authorization: Bearer <master_key>
Content-Type: application/json

{
  "server_id": "my-mcp-server",
  "name": "My MCP Server",
  "url": "http://mcp-service:8080/mcp",
  "transport": "streamable_http",
  "tools": ["search", "fetch", "write"],
  "resources": ["documents", "images"],
  "auth_type": "bearer_token",
  "metadata": {
    "version": "1.0.0"
  }
}
```

### Transport Types

| Transport | Description | Use Case |
|-----------|-------------|----------|
| `streamable_http` | HTTP with streaming | Most common, production use |
| `sse` | Server-Sent Events | Real-time updates |
| `stdio` | Standard I/O | Local development |

### List All MCP Servers

```bash
GET /mcp/servers
Authorization: Bearer <master_key>
```

### List Available Tools

```bash
GET /mcp/tools
Authorization: Bearer <master_key>
```

Response:
```json
{
  "tools": [
    {
      "server_id": "my-mcp-server",
      "server_name": "My MCP Server",
      "tool": "search"
    }
  ]
}
```

### List Available Resources

```bash
GET /mcp/resources
Authorization: Bearer <master_key>
```

### Unregister an MCP Server

```bash
DELETE /mcp/servers/{server_id}
Authorization: Bearer <master_key>
```

## Python SDK Usage

```python
from litellm_llmrouter import get_mcp_gateway, MCPServer, MCPTransport

# Get gateway instance
gateway = get_mcp_gateway()

# Register an MCP server
server = MCPServer(
    server_id="search-server",
    name="Search Service",
    url="http://localhost:8080/mcp",
    transport=MCPTransport.STREAMABLE_HTTP,
    tools=["web_search", "image_search"],
    resources=["search_results"]
)
gateway.register_server(server)

# List all tools
tools = gateway.list_tools()
for tool in tools:
    print(f"Tool: {tool['tool']} from {tool['server_name']}")
```

## Configuration in YAML

```yaml
mcp_servers:
  search_server:
    url: "http://search-service:8080/mcp"
    transport: "streamable_http"
    tools: ["search"]

  file_server:
    url: "http://file-service:8081/mcp"
    transport: "streamable_http"
    tools: ["read_file", "write_file"]
    auth_type: "api_key"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Gateway                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Server     │  │  Tool       │  │  Resource               │  │
│  │  Registry   │◄─│  Discovery  │◄─│  Manager                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                 │                      │
         ▼                 ▼                      ▼
   ┌─────────┐       ┌─────────┐           ┌─────────┐
   │ Search  │       │ File    │           │ Database│
   │ MCP     │       │ MCP     │           │ MCP     │
   └─────────┘       └─────────┘           └─────────┘
```

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MCP_GATEWAY_ENABLED` | `false` | Enable MCP gateway |
| `STORE_MODEL_IN_DB` | `false` | Persist servers in database |

## See Also

- [A2A Gateway](a2a-gateway.md) - Agent-to-Agent protocol support
- [Vector Stores](vector-stores.md) - Vector database integrations
- [API Reference](api-reference.md) - Complete API documentation
