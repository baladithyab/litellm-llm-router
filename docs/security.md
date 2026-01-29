# Security Guide

Security is a core design principle of RouteIQ Gateway. This guide outlines the security features and best practices for securing your deployment.

## Authentication Boundaries

RouteIQ Gateway implements a two-tier authentication model to separate user traffic from administrative operations:

### User API Key Authentication

Standard LiteLLM API key authentication (`Authorization: Bearer sk-xxx`) is used for:
- All inference endpoints (`/chat/completions`, `/completions`, etc.)
- Read-only monitoring endpoints (`/router/info`, `/config/sync/status`)
- Read-only MCP/A2A listing endpoints

### Admin API Key Authentication

Control-plane operations require a separate admin API key to prevent accidental or malicious configuration changes. Admin keys are provided via:
- **Preferred**: `X-Admin-API-Key: <key>` header
- **Fallback**: `Authorization: Bearer <key>` (checked against admin key list)

**Protected endpoints (require admin auth):**
- `POST /router/reload` - Hot reload routing strategies
- `POST /config/reload` - Reload configuration
- `POST/PUT/DELETE /llmrouter/mcp/servers/*` - MCP server management
- `POST /llmrouter/mcp/servers/{id}/tools` - MCP tool registration
- `POST /llmrouter/mcp/tools/call` - MCP tool invocation
- `POST/DELETE /a2a/agents` - A2A agent registration

**Configuration:**

```bash
# .env - Generate with: openssl rand -hex 32
ADMIN_API_KEYS=key1,key2,key3

# Or single key (legacy)
ADMIN_API_KEY=your-admin-key

# Disable admin auth (NOT recommended for production)
# ADMIN_AUTH_ENABLED=false
```

**Fail-closed behavior:** If `ADMIN_API_KEYS` is not configured and `ADMIN_AUTH_ENABLED` is not explicitly `false`, control-plane endpoints return `403 Forbidden` with error code `control_plane_not_configured`.

### Unauthenticated Endpoints

Health probe endpoints remain unauthenticated for Kubernetes compatibility:
- `/_health/live` - Liveness probe
- `/_health/ready` - Readiness probe

## Request Correlation

All requests are assigned a correlation ID for tracing and debugging:

- **Header**: `X-Request-ID`
- **Passthrough**: If provided in the request, the same ID is used
- **Generated**: UUID v4 is generated if not provided
- **Response**: ID is returned in `X-Request-ID` response header
- **Error bodies**: All error responses include `request_id` field

Example error response:
```json
{
  "error": "invalid_admin_key",
  "message": "Invalid admin API key.",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Server-side logs include the request ID for correlation:
```
ERROR request_id=550e8400-e29b-41d4-a716-446655440000 Invalid admin API key in request
```

## Error Response Sanitization

Client-facing error responses are sanitized to prevent information leakage:

- **No stack traces** in responses
- **No internal exception messages** (e.g., database connection strings)
- **Generic messages** with machine-readable error codes
- **Request ID** for correlation with server logs

| HTTP Status | Error Code | When |
|------------|------------|------|
| 401 | `admin_key_required` | Missing admin API key |
| 401 | `invalid_admin_key` | Invalid admin API key |
| 403 | `control_plane_not_configured` | Admin keys not set up |
| 500 | `internal_error` | Unexpected server error |
| 503 | (readiness detail) | Dependency health check failed |

## SSRF Protection

Server-Side Request Forgery (SSRF) is a major risk for gateways that proxy requests. RouteIQ Gateway includes built-in SSRF guards with a **secure-by-default (fail-closed)** policy.

### Default Behavior (Secure-by-Default)

By default, the gateway blocks requests to:

| Target | Always Blocked | Reason |
|--------|---------------|--------|
| Loopback (127.0.0.0/8, ::1) | ✅ Yes | Cannot be overridden |
| Link-local (169.254.0.0/16) | ✅ Yes | Includes cloud metadata endpoints |
| Localhost hostnames | ✅ Yes | Cannot be overridden |
| Private IPs (10.x, 172.16.x, 192.168.x) | By default | Can be allowed via allowlist |

**Important:** Loopback and link-local addresses are **always blocked** and cannot be allowed via allowlist configuration. This protects against cloud metadata endpoint attacks.

### Configuration

Configure SSRF protection via environment variables:

```bash
# .env

# Allow ALL private IP ranges (NOT recommended for production)
# Default: false (blocked)
LLMROUTER_ALLOW_PRIVATE_IPS=false

# Allowlist specific hosts/domains (comma-separated)
# Supports exact match and suffix match (prefix with ".")
LLMROUTER_SSRF_ALLOWLIST_HOSTS=mcp.internal,.trusted-corp.com

# Allowlist specific IP ranges in CIDR notation (comma-separated)
LLMROUTER_SSRF_ALLOWLIST_CIDRS=10.100.0.0/16,192.168.50.0/24
```

### Host Allowlist Patterns

The `LLMROUTER_SSRF_ALLOWLIST_HOSTS` setting supports two patterns:

| Pattern | Example | Matches |
|---------|---------|---------|
| Exact match | `mcp.internal` | Only `mcp.internal` |
| Suffix match | `.trusted.com` | `api.trusted.com`, `mcp.trusted.com`, `trusted.com` |

### CIDR Allowlist

The `LLMROUTER_SSRF_ALLOWLIST_CIDRS` setting allows specific IP ranges:

```bash
# Allow a specific /16 for internal MCP servers
LLMROUTER_SSRF_ALLOWLIST_CIDRS=10.100.0.0/16

# Multiple ranges
LLMROUTER_SSRF_ALLOWLIST_CIDRS=10.100.0.0/16,192.168.50.0/24
```

**Note:** CIDR allowlists cannot override the loopback/link-local blocks.

### Where SSRF Protection is Enforced

SSRF protection is enforced at:

1. **MCP Server Registration** (`POST /llmrouter/mcp/servers`)
2. **MCP Server Update** (`PUT /llmrouter/mcp/servers/{id}`)
3. **MCP Tool Invocation** (when calling registered servers)
4. **A2A Agent Registration** (`POST /a2a/agents`)
5. **A2A Agent Invocation** (when forwarding requests to agents)

### Error Responses

When a URL is blocked, the API returns:

```json
{
  "error": "ssrf_blocked",
  "message": "Server URL blocked for security reasons: private IP 10.0.0.1 is blocked",
  "request_id": "abc123"
}
```

### Production Recommendations

1. **Keep `LLMROUTER_ALLOW_PRIVATE_IPS=false`** (the default)
2. Use explicit CIDR allowlists for internal services
3. Prefer hostname allowlists over broad IP ranges
4. Monitor blocked requests via logs for security auditing

## Artifact Safety

RouteIQ Gateway uses machine learning models for routing. To prevent arbitrary code execution from malicious model files:

- **Pickle Disabled**: Loading models via Python's `pickle` module is **disabled by default**.
- **Safe Formats**: We recommend using `safetensors` or ONNX for model weights.
- **Opt-in Only**: If you must use pickle (e.g., for legacy Scikit-Learn models), you must explicitly enable it via environment variable: `LLMROUTER_ALLOW_PICKLE_MODELS=true`.

## Key Management

Never store API keys in your configuration files or code.

- **Environment Variables**: Use environment variables (e.g., `OPENAI_API_KEY`) which are injected into the container at runtime.
- **Secret Managers**: In production, use a secret manager (AWS Secrets Manager, HashiCorp Vault, Kubernetes Secrets) to inject these variables.

## Kubernetes Security Context

When deploying to Kubernetes, we recommend the following security context to minimize the attack surface:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 3000
  fsGroup: 2000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

## Network Policies

Restrict traffic to and from the gateway using Kubernetes Network Policies.

- **Ingress**: Allow traffic only from your application services or ingress controller.
- **Egress**: Allow traffic only to the necessary LLM provider APIs and internal dependencies (Redis, Postgres).
