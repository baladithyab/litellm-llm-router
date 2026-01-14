# LiteLLM + LLMRouter Gateway - Project Overview

## What This Project Is

This is a production-ready AI Gateway that integrates:
- **LiteLLM**: Unified API gateway providing OpenAI-compatible interface to 100+ LLM providers
- **LLMRouter**: ML-based intelligent routing library with 18+ routing strategies

## Architecture Summary

```
Client → Gateway → LLMRouter Strategy → LiteLLM → LLM Provider
                         ↓
              [KNN, SVM, MLP, ELO, Hybrid, etc.]
```

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Strategies | `src/litellm_llmrouter/strategies.py` | 18+ ML routing strategies |
| Routes | `src/litellm_llmrouter/routes.py` | FastAPI endpoints for A2A, MCP, hot reload |
| A2A Gateway | `src/litellm_llmrouter/a2a_gateway.py` | Agent-to-Agent protocol support |
| MCP Gateway | `src/litellm_llmrouter/mcp_gateway.py` | Model Context Protocol support |
| Config Sync | `src/litellm_llmrouter/config_sync.py` | S3/GCS config sync with ETag caching |
| Hot Reload | `src/litellm_llmrouter/hot_reload.py` | Dynamic strategy updates |
| Startup | `src/litellm_llmrouter/startup.py` | Entry point wiring everything together |

## Configuration

Main config: `config/config.yaml`

Key sections:
- `model_list`: LLM provider configurations
- `router_settings`: Routing strategy and args
- `general_settings`: Master key, database URL
- `litellm_settings`: Caching, callbacks, timeouts

## Running the Gateway

```bash
# Basic
docker compose up -d

# HA mode (Redis + PostgreSQL)
docker compose -f docker-compose.ha.yml up -d

# With OpenTelemetry
docker compose -f docker-compose.otel.yml up -d
```

## Reference Implementations

- `reference/litellm/` - Full LiteLLM codebase (submodule)
- `reference/LLMRouter/` - Full LLMRouter codebase (submodule)

## Spec Location

Feature spec: `.kiro/specs/production-ai-gateway/`
- `requirements.md` - EARS-format requirements
- `design.md` - Architecture and correctness properties
- `tasks.md` - Implementation task list
