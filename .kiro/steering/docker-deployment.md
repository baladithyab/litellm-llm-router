---
inclusion: fileMatch
fileMatchPattern: "**/Dockerfile*"
---

# Docker Deployment Reference

## Multi-Architecture Build

The Dockerfile supports `linux/amd64` and `linux/arm64`:

```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ghcr.io/baladithyab/litellm-llm-router:latest \
  -f docker/Dockerfile .
```

## Build Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `PYTHON_VERSION` | `3.12` | Python version |
| `UV_VERSION` | `0.9` | uv package manager version |
| `LITELLM_VERSION` | `1.80.15` | LiteLLM version |
| `LLMROUTER_COMMIT` | `7890cd9...` | LLMRouter git commit |

## Multi-Stage Build

1. **Builder stage**: Clones LLMRouter, builds wheel with uv
2. **Runtime stage**: Minimal image with only runtime dependencies

## Security Features

- Non-root user (`litellm:litellm`, UID/GID 1000)
- Minimal base image (slim variant)
- Read-only filesystem compatible
- Health checks for orchestrators
- tini as init system for signal handling

## Environment Variables

### Core Settings
```bash
LITELLM_CONFIG_PATH=/app/config/config.yaml
LITELLM_MASTER_KEY=sk-your-key
LLMROUTER_MODELS_PATH=/app/models
```

### Gateway Features
```bash
A2A_GATEWAY_ENABLED=false
MCP_GATEWAY_ENABLED=false
```

### Config Sync
```bash
CONFIG_HOT_RELOAD=false
CONFIG_SYNC_ENABLED=true
CONFIG_SYNC_INTERVAL=60
CONFIG_S3_BUCKET=my-bucket
CONFIG_S3_KEY=config/config.yaml
```

### OpenTelemetry
```bash
OTEL_SERVICE_NAME=litellm-gateway
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_TRACES_EXPORTER=otlp
```

---

## Deployment Modes

### Basic (Single Instance)

```bash
docker compose up -d
```

### High Availability

```bash
docker compose -f docker-compose.ha.yml up -d
```

Includes:
- PostgreSQL for persistence
- Redis for caching
- 2 gateway replicas
- Nginx load balancer

### With OpenTelemetry

```bash
docker compose -f docker-compose.otel.yml up -d
```

---

## HA Architecture

```
                    ┌─────────────┐
                    │   Nginx     │ :8080
                    │   (LB)      │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │ Gateway 1 │ │ Gateway │ │ Gateway 2 │
        │   :4000   │ │   ...   │ │   :4001   │
        └─────┬─────┘ └────┬────┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
              ┌────────────┼────────────┐
              │                         │
        ┌─────▼─────┐             ┌─────▼─────┐
        │ PostgreSQL│             │   Redis   │
        │   :5432   │             │   :6379   │
        └───────────┘             └───────────┘
```

---

## AWS Deployment

### ECS Task Definition

```json
{
  "containerDefinitions": [
    {
      "name": "litellm-gateway",
      "image": "ghcr.io/baladithyab/litellm-llm-router:latest",
      "portMappings": [{"containerPort": 4000}],
      "environment": [
        {"name": "LITELLM_CONFIG_PATH", "value": "/app/config/config.yaml"}
      ],
      "secrets": [
        {"name": "LITELLM_MASTER_KEY", "valueFrom": "arn:aws:secretsmanager:..."}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:4000/health || exit 1"]
      }
    }
  ]
}
```

### Use IAM Roles for S3 Access

Instead of hardcoding credentials, use ECS task roles:

```json
{
  "taskRoleArn": "arn:aws:iam::123456789:role/litellm-task-role"
}
```

---

## Health Checks

Built-in health check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:4000/health/liveliness || exit 1
```

Endpoints:
- `/health/liveliness` - Basic liveness
- `/health/readiness` - Full readiness with dependencies
