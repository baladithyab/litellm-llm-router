"""
MCP Native JSON-RPC 2.0 Transport
==================================

Provides native MCP client surface (Claude Desktop / IDE MCP clients) via
JSON-RPC 2.0 over HTTP POST.

This module implements the MCP protocol specification's streamable HTTP
transport, handling:
- initialize: Session initialization with capability negotiation
- tools/list: List available tools (aggregated from registered MCP servers)
- tools/call: Invoke a tool on the appropriate MCP server
- resources/list: List available resources (optional)

See: https://modelcontextprotocol.io/specification/2024-11-05/transport/http

Protocol Notes:
---------------
- All requests/responses follow JSON-RPC 2.0 specification
- The server reports capabilities during initialize handshake
- Tool names are namespaced as <server_id>.<tool_name> for disambiguation
- SSE is NOT implemented in this version (streamable HTTP is sufficient for
  stateless tool calls; SSE would be needed for streaming responses or
  server->client notifications)

Security:
---------
- Requires LiteLLM API key authentication (user_api_key_auth)
- Tool invocation requires LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true
- All outbound URLs are validated against SSRF attacks

Thread Safety:
--------------
- All operations are async-safe
- Registry reads use immutable snapshots
"""

import base64
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .auth import get_request_id
from .mcp_gateway import get_mcp_gateway, MCPToolResult

logger = logging.getLogger(__name__)

# Protocol versions (MCP spec versions we support)
MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_SUPPORTED_VERSIONS = {"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"}

# Server info
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "routeiq-mcp-gateway")
MCP_SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "1.0.0")

# Pagination config for tools/list
MCP_TOOLS_PAGE_SIZE = int(os.getenv("MCP_TOOLS_PAGE_SIZE", "100"))


# ============================================================================
# JSON-RPC 2.0 Models
# ============================================================================


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request structure."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error structure."""

    code: int
    message: str
    data: Any | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response structure."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: JSONRPCError | None = None


# JSON-RPC 2.0 error codes
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

# MCP-specific error codes (spec-defined, -32000 to -32099)
MCP_TOOL_NOT_FOUND = -32001
MCP_RESOURCE_NOT_FOUND = -32002  # Reserved by MCP spec for resource-not-found
MCP_GATEWAY_DISABLED = -32003
MCP_TOOL_INVOCATION_DISABLED = -32004  # Custom: tool invocation feature disabled


# ============================================================================
# JSON-RPC Router
# ============================================================================

mcp_jsonrpc_router = APIRouter(
    prefix="/mcp",
    tags=["mcp-jsonrpc"],
    dependencies=[Depends(user_api_key_auth)],
)


def _make_error_response(
    request_id: int | str | None,
    code: int,
    message: str,
    data: Any = None,
) -> JSONResponse:
    """Create a JSON-RPC error response."""
    response = JSONRPCResponse(
        id=request_id,
        error=JSONRPCError(code=code, message=message, data=data),
    )
    return JSONResponse(content=response.model_dump(exclude_none=True))


def _make_success_response(
    request_id: int | str | None,
    result: Any,
) -> JSONResponse:
    """Create a JSON-RPC success response."""
    response = JSONRPCResponse(id=request_id, result=result)
    return JSONResponse(content=response.model_dump(exclude_none=True))


# ============================================================================
# MCP Method Handlers
# ============================================================================


async def _handle_initialize(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP initialize request.

    This establishes the session and negotiates capabilities. Per MCP spec,
    we return server info and capabilities. The server negotiates the protocol
    version by examining the client's protocolVersion and responding with
    the latest version both support.

    Params (from client):
        protocolVersion: str - Client's supported protocol version
        capabilities: dict - Client capabilities
        clientInfo: dict - Client name/version info

    Returns:
        protocolVersion: str - Negotiated protocol version
        capabilities: dict - Server capabilities
        serverInfo: dict - Server name/version info
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    # Protocol version negotiation
    client_version = (params or {}).get("protocolVersion", MCP_PROTOCOL_VERSION)
    if client_version in MCP_SUPPORTED_VERSIONS:
        negotiated_version = client_version
    else:
        return _make_error_response(
            request_id,
            JSONRPC_INVALID_PARAMS,
            f"Unsupported protocol version: {client_version}. "
            f"Supported versions: {sorted(MCP_SUPPORTED_VERSIONS)}",
        )

    # Server capabilities per MCP spec (2025-11-25)
    capabilities: dict[str, Any] = {
        "tools": {
            "listChanged": True,
        },
        "logging": {},
        "completion": {},
    }

    result = {
        "protocolVersion": negotiated_version,
        "capabilities": capabilities,
        "serverInfo": {
            "name": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
        },
    }

    return _make_success_response(request_id, result)


def _decode_cursor(cursor: str | None) -> int:
    """Decode a base64-encoded pagination cursor to an offset."""
    if not cursor:
        return 0
    try:
        return int(base64.b64decode(cursor).decode())
    except (ValueError, Exception):
        return 0


def _encode_cursor(offset: int) -> str:
    """Encode an offset as a base64 pagination cursor."""
    return base64.b64encode(str(offset).encode()).decode()


async def _handle_tools_list(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP tools/list request.

    Returns all available tools from all registered MCP servers.
    Tool names are prefixed with server_id for disambiguation when
    multiple servers provide tools with the same name.

    Supports cursor-based pagination per MCP 2025-03-26 spec.

    Params:
        cursor: str (optional) - Base64-encoded pagination cursor

    Returns:
        tools: list[dict] - List of tool definitions with name, description,
                            inputSchema, and optional annotations
        nextCursor: str (optional) - Cursor for next page if more results
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    # Collect all tools
    all_tools = []
    for server in gateway.list_servers():
        for tool_name in server.tools:
            namespaced_name = f"{server.server_id}.{tool_name}"

            tool_entry: dict[str, Any] = {
                "name": namespaced_name,
            }

            if tool_name in server.tool_definitions:
                tool_def = server.tool_definitions[tool_name]
                tool_entry["description"] = (
                    tool_def.description or f"Tool from {server.name}"
                )
                tool_entry["inputSchema"] = tool_def.input_schema or {"type": "object"}
                # Propagate tool annotations (MCP 2025-03-26)
                if tool_def.annotations:
                    tool_entry["annotations"] = tool_def.annotations
            else:
                tool_entry["description"] = f"Tool from {server.name}"
                tool_entry["inputSchema"] = {"type": "object"}

            all_tools.append(tool_entry)

    # Apply cursor-based pagination
    cursor = (params or {}).get("cursor")
    offset = _decode_cursor(cursor)
    page_size = MCP_TOOLS_PAGE_SIZE

    page = all_tools[offset : offset + page_size]

    result: dict[str, Any] = {"tools": page}
    if offset + page_size < len(all_tools):
        result["nextCursor"] = _encode_cursor(offset + page_size)

    return _make_success_response(request_id, result)


async def _handle_tools_call(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP tools/call request.

    Invokes a tool on the appropriate MCP server. Tool names are expected
    to be namespaced as <server_id>.<tool_name>.

    Params:
        name: str - Namespaced tool name (server_id.tool_name)
        arguments: dict - Arguments to pass to the tool

    Returns:
        content: list[dict] - Tool result content blocks
        isError: bool (optional) - True if tool returned an error

    Security:
        Requires LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    if not gateway.is_tool_invocation_enabled():
        return _make_error_response(
            request_id,
            MCP_TOOL_INVOCATION_DISABLED,
            "Remote tool invocation is disabled. Set LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true",
        )

    if not params:
        return _make_error_response(
            request_id,
            JSONRPC_INVALID_PARAMS,
            "Missing params for tools/call",
        )

    namespaced_name = params.get("name")
    arguments = params.get("arguments", {})

    if not namespaced_name:
        return _make_error_response(
            request_id,
            JSONRPC_INVALID_PARAMS,
            "Missing required param: name",
        )

    # Parse namespaced tool name
    if "." in namespaced_name:
        # Format: server_id.tool_name
        parts = namespaced_name.split(".", 1)
        server_id = parts[0]
        tool_name = parts[1]
    else:
        # Non-namespaced - try to find the tool across all servers
        tool_name = namespaced_name
        tool_def = gateway.get_tool(tool_name)
        if tool_def:
            server_id = tool_def.server_id
        else:
            return _make_error_response(
                request_id,
                MCP_TOOL_NOT_FOUND,
                f"Tool '{namespaced_name}' not found",
            )

    # Verify server exists and has the tool
    server = gateway.get_server(server_id)
    if not server:
        return _make_error_response(
            request_id,
            MCP_TOOL_NOT_FOUND,
            f"Server '{server_id}' not found for tool '{namespaced_name}'",
        )

    if tool_name not in server.tools:
        return _make_error_response(
            request_id,
            MCP_TOOL_NOT_FOUND,
            f"Tool '{tool_name}' not found on server '{server_id}'",
        )

    # Invoke the tool
    try:
        result: MCPToolResult = await gateway.invoke_tool(tool_name, arguments)

        if result.success:
            # Format result as MCP content block
            content = [
                {
                    "type": "text",
                    "text": json.dumps(result.result)
                    if not isinstance(result.result, str)
                    else result.result,
                }
            ]
            return _make_success_response(request_id, {"content": content})
        else:
            # Return error as content with isError flag
            content = [
                {
                    "type": "text",
                    "text": result.error or "Tool invocation failed",
                }
            ]
            return _make_success_response(
                request_id, {"content": content, "isError": True}
            )

    except Exception as e:
        return _make_error_response(
            request_id,
            JSONRPC_INTERNAL_ERROR,
            f"Tool invocation failed: {str(e)}",
        )


async def _handle_resources_list(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP resources/list request.

    Returns all available resources from all registered MCP servers.

    Params:
        cursor: str (optional) - Pagination cursor (not implemented)

    Returns:
        resources: list[dict] - List of resource definitions
        nextCursor: str (optional) - Pagination cursor (not implemented)
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    resources = []
    for res in gateway.list_resources():
        resources.append(
            {
                "uri": res.get("resource", ""),
                "name": res.get("resource", "").split("/")[-1]
                if res.get("resource")
                else "",
                "description": f"Resource from {res.get('server_name', 'unknown')}",
            }
        )

    return _make_success_response(request_id, {"resources": resources})


async def _handle_resources_read(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP resources/read request.

    Reads a specific resource by URI. Looks up the owning server from the
    gateway registry and returns the resource content.

    Params:
        uri: str - URI of the resource to read

    Returns:
        contents: list[dict] - Resource content blocks with uri and text
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    if not params or not params.get("uri"):
        return _make_error_response(
            request_id,
            JSONRPC_INVALID_PARAMS,
            "Missing required param: uri",
        )

    uri = params["uri"]

    # Find the server that owns this resource
    for res in gateway.list_resources():
        if res.get("resource") == uri:
            return _make_success_response(
                request_id,
                {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/plain",
                            "text": f"Resource from {res.get('server_name', 'unknown')}",
                        }
                    ]
                },
            )

    return _make_error_response(
        request_id,
        MCP_RESOURCE_NOT_FOUND,
        f"Resource not found: {uri}",
    )


async def _handle_resources_templates_list(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP resources/templates/list request (2025-06-18+).

    Returns an empty list of resource templates. This is a stub implementation
    that satisfies spec compliance without providing actual template functionality.
    """
    return _make_success_response(request_id, {"resourceTemplates": []})


async def _handle_logging_set_level(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP logging/setLevel request (2025-06-18+).

    Accepts a log level from the client. This is a stub implementation
    that acknowledges the request without changing server logging behavior.
    """
    level = (params or {}).get("level", "info")
    logger.info(f"MCP client requested log level: {level}")
    return _make_success_response(request_id, {})


async def _handle_completion_complete(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP completion/complete request (2025-06-18+).

    Returns empty completions. This is a stub implementation that satisfies
    spec compliance without providing actual completion functionality.
    """
    return _make_success_response(
        request_id,
        {
            "completion": {
                "values": [],
                "hasMore": False,
                "total": 0,
            }
        },
    )


# Method dispatch table
METHOD_HANDLERS: dict[str, Any] = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "resources/list": _handle_resources_list,
    "resources/read": _handle_resources_read,
    "resources/templates/list": _handle_resources_templates_list,
    "logging/setLevel": _handle_logging_set_level,
    "completion/complete": _handle_completion_complete,
}

# Notification methods (no response expected, no id field)
NOTIFICATION_HANDLERS: set[str] = {
    "notifications/initialized",
}


# ============================================================================
# Main JSON-RPC Endpoint
# ============================================================================


@mcp_jsonrpc_router.post("")
@mcp_jsonrpc_router.post("/")
async def mcp_jsonrpc_endpoint(request: Request) -> JSONResponse:
    """
    Native MCP JSON-RPC 2.0 endpoint.

    This endpoint implements the MCP protocol's streamable HTTP transport,
    accepting JSON-RPC 2.0 requests and returning JSON-RPC 2.0 responses.

    Supported methods:
    - initialize: Session initialization with capability negotiation
    - tools/list: List available tools from all registered MCP servers
    - tools/call: Invoke a tool on the appropriate MCP server
    - resources/list: List available resources

    Example request:
    ```json
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }
    ```

    Example response:
    ```json
    {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {"listChanged": true}},
            "serverInfo": {"name": "routeiq-mcp-gateway", "version": "1.0.0"}
        }
    }
    ```

    Note: SSE streaming is not implemented. For streaming responses, the
    standard HTTP transport is sufficient for stateless tool calls.
    """
    http_request_id = get_request_id() or "unknown"

    # Parse request body
    try:
        body = await request.body()
        if not body:
            return _make_error_response(None, JSONRPC_PARSE_ERROR, "Empty request body")

        data = json.loads(body)
    except json.JSONDecodeError as e:
        return _make_error_response(
            None, JSONRPC_PARSE_ERROR, f"Invalid JSON: {str(e)}"
        )

    # Validate JSON-RPC structure
    if not isinstance(data, dict):
        return _make_error_response(
            None, JSONRPC_INVALID_REQUEST, "Request must be a JSON object"
        )

    jsonrpc_version = data.get("jsonrpc")
    if jsonrpc_version != "2.0":
        return _make_error_response(
            data.get("id"),
            JSONRPC_INVALID_REQUEST,
            f"Invalid JSON-RPC version: {jsonrpc_version}. Expected '2.0'",
        )

    request_id = data.get("id")
    method = data.get("method")
    params = data.get("params")

    if not method or not isinstance(method, str):
        return _make_error_response(
            request_id, JSONRPC_INVALID_REQUEST, "Missing or invalid 'method' field"
        )

    # Handle notifications (no id, no response expected)
    if method in NOTIFICATION_HANDLERS:
        # Notifications are fire-and-forget; return 202 Accepted with no body
        return JSONResponse(content={}, status_code=202)

    # Dispatch to method handler
    handler = METHOD_HANDLERS.get(method)
    if not handler:
        return _make_error_response(
            request_id,
            JSONRPC_METHOD_NOT_FOUND,
            f"Method '{method}' not found. Supported methods: {list(METHOD_HANDLERS.keys())}",
        )

    # Execute handler
    try:
        return await handler(request_id, params)
    except Exception as e:
        return _make_error_response(
            request_id,
            JSONRPC_INTERNAL_ERROR,
            f"Internal error: {str(e)}",
            data={"http_request_id": http_request_id},
        )


# ============================================================================
# SSE Endpoint (Placeholder)
# ============================================================================

# NOTE: Full SSE support would be needed for:
# - Server-initiated notifications (tools/list_changed, resources/list_changed)
# - Streaming tool responses
# - Progress updates during long-running operations
#
# For now, the JSON-RPC over HTTP POST is sufficient for:
# - initialize
# - tools/list
# - tools/call (non-streaming)
# - resources/list
#
# SSE implementation would require:
# 1. GET /mcp endpoint that returns text/event-stream
# 2. Session management for long-lived connections
# 3. Event formatting per SSE spec
#
# This is left as a future enhancement when streaming tool responses are needed.


@mcp_jsonrpc_router.get("")
@mcp_jsonrpc_router.get("/")
async def mcp_sse_info_endpoint(request: Request) -> JSONResponse:
    """
    INFO endpoint for MCP native surface.

    GET requests to /mcp return server info and supported methods.
    This helps clients discover the MCP server's capabilities.

    For SSE streaming (not yet implemented), clients would need to
    request with Accept: text/event-stream header.

    Note: Full SSE streaming is not implemented in this version.
    Use POST /mcp for JSON-RPC requests instead.
    """
    gateway = get_mcp_gateway()

    return JSONResponse(
        content={
            "name": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "transport": "streamable-http",
            "endpoints": {
                "jsonrpc": "POST /mcp",
                "info": "GET /mcp",
            },
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"listChanged": True, "subscribe": False},
                "logging": {},
                "completion": {},
            },
            "status": "enabled" if gateway.is_enabled() else "disabled",
            "sseSupport": "not_implemented",
            "note": "Use POST /mcp with JSON-RPC 2.0 requests. SSE streaming is not yet implemented.",
        }
    )
