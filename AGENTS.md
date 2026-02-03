# AGENTS.md - Instructions for AI Coding Agents

This document provides comprehensive instructions for AI coding agents (Claude Code,
Cursor, Codex, etc.) working in the RouteIQ Gateway repository.

## OVERVIEW

**RouteIQ Gateway** is a production-grade, cloud-native General AI Gateway built on
[LiteLLM](https://github.com/BerriAI/litellm). It provides:

- **Unified AI Gateway**: OpenAI-compatible proxy for 100+ LLM providers
- **Intelligent Routing**: ML-based routing strategies (KNN, MLP, SVM, etc.)
- **MCP Gateway**: Model Context Protocol support for tool/agent integration
- **A2A Protocol**: Agent-to-Agent communication support
- **MLOps Pipeline**: Scripts for telemetry collection, model training, and deployment
- **Enterprise Features**: Security, observability, hot-reloading, multi-tenancy

## REPOSITORY STRUCTURE

```
RouteIQ/
├── src/litellm_llmrouter/      # Main application code
│   ├── routes.py               # FastAPI routes and endpoints
│   ├── strategies.py           # ML routing strategies
│   ├── strategy_registry.py    # Strategy registration and management
│   ├── mcp_gateway.py          # MCP protocol implementation
│   ├── mcp_sse_transport.py    # SSE transport for MCP
│   ├── a2a_gateway.py          # Agent-to-Agent protocol
│   ├── observability.py        # OpenTelemetry integration
│   ├── auth.py                 # Authentication and authorization
│   ├── hot_reload.py           # Configuration hot-reloading
│   ├── url_security.py         # SSRF protection
│   └── gateway/                # Gateway subsystem
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── conftest.py             # Pytest fixtures
├── config/                     # Configuration files
├── scripts/                    # Utility scripts
├── examples/mlops/             # MLOps training examples
├── docs/                       # Documentation
├── docker/                     # Docker build files
├── reference/litellm/          # Upstream LiteLLM (git submodule - READ-ONLY)
└── plans/                      # Development planning docs
```

## DEVELOPMENT COMMANDS

### Package Management (uv)

```bash
uv sync                          # Install dependencies
uv add <package>                 # Add a dependency
uv run python -m <module>        # Run a module
```

### Testing

```bash
uv run pytest tests/unit/ -x     # Run unit tests (stop on first failure)
uv run pytest tests/integration/ # Run integration tests
uv run pytest tests/ -x -v       # Run all tests with verbose output
uv run pytest -k "test_name"     # Run specific test
```

### Linting and Formatting

```bash
./scripts/lint.sh format         # Format with ruff
./scripts/lint.sh check          # Check with ruff
uv run ruff check src/ tests/    # Direct ruff check
uv run mypy src/litellm_llmrouter/  # Type checking
```

### Git Hooks (Lefthook)

```bash
lefthook install                 # Install git hooks
lefthook run pre-commit          # Run pre-commit hooks manually
lefthook run pre-push            # Run pre-push hooks manually
```

### Remote Execution (rr - Road Runner)

When local `git push` is blocked by Code Defender, use `rr` to sync and push from
a remote machine with unrestricted access:

```bash
rr sync                          # Sync code to remote
rr push                          # Sync and push to GitHub
rr test                          # Run unit tests on remote
rr ci                            # Run full CI pipeline on remote
rr doctor                        # Diagnose connection issues
```

See [`docs/rr-workflow.md`](docs/rr-workflow.md) for setup and usage details.

### Docker

```bash
docker-compose up -d             # Start basic stack
docker-compose -f docker-compose.ha.yml up -d      # HA stack
docker-compose -f docker-compose.otel.yml up -d    # Observability stack
```

## CODING GUIDELINES

### Code Style

- **Python 3.14+** required
- **Ruff** for linting and formatting (line-length: 88)
- **Type hints** required for all public APIs
- **Pydantic v2** for data validation
- **Async/await** patterns throughout FastAPI routes

### File Patterns

- Test files: `tests/unit/test_*.py`, `tests/integration/test_*.py`
- Source files: `src/litellm_llmrouter/*.py`
- Config files: `config/*.yaml`

### Imports

```python
# Standard library
import asyncio
from typing import Optional, Dict, Any

# Third-party
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# Local
from litellm_llmrouter.observability import setup_telemetry
from litellm_llmrouter.auth import verify_token
```

### Error Handling

- Use specific exception types from `fastapi.HTTPException`
- Include meaningful error messages
- Log errors with context using the observability module
- Handle both sync and async errors consistently

### Testing

- Tests must be async where testing async code
- Use `pytest-asyncio` with `asyncio_mode = "auto"`
- Use `hypothesis` for property-based testing where appropriate
- Mock external services (LLM providers, databases)

## KEY PATTERNS

### 1. Routing Strategies

Strategies are registered via the strategy registry:

```python
from litellm_llmrouter.strategy_registry import register_strategy

@register_strategy("my-strategy")
class MyStrategy(BaseStrategy):
    async def route(self, request: RouteRequest) -> str:
        # Return the chosen model name
        pass
```

### 2. MCP Tool Integration

MCP tools follow the Model Context Protocol specification:

```python
# Tool definitions in mcp_gateway.py
# SSE transport in mcp_sse_transport.py
# JSON-RPC handling in mcp_jsonrpc.py
```

### 3. Observability

All operations should be traced:

```python
from litellm_llmrouter.observability import get_tracer

tracer = get_tracer(__name__)

@tracer.start_as_current_span("operation_name")
async def my_operation():
    pass
```

### 4. Configuration

Use the config loader for YAML configurations:

```python
from litellm_llmrouter.config_loader import load_config

config = load_config("config/config.yaml")
```

## IMPORTANT CONSTRAINTS

### 1. Reference Directory is READ-ONLY

The `reference/litellm/` directory is a git submodule containing upstream LiteLLM.
**DO NOT MODIFY** files in this directory. It exists for reference only.

### 2. Security Requirements

- **No real secrets** in code or tests (use placeholders like `test-api-key`)
- **SSRF protection** via `url_security.py` for all external requests
- **Input validation** for all API endpoints
- Pickle loading is **disabled by default** for ML models

### 3. Test Requirements

- All new features require unit tests
- Integration tests for API endpoints
- Security-sensitive code requires security tests

## COMMON TASKS

### Adding a New Endpoint

1. Add route in `src/litellm_llmrouter/routes.py`
2. Add Pydantic models for request/response
3. Add unit test in `tests/unit/test_*.py`
4. Add integration test if needed
5. Update API documentation

### Adding a New Routing Strategy

1. Implement strategy in `src/litellm_llmrouter/strategies.py`
2. Register in `src/litellm_llmrouter/strategy_registry.py`
3. Add configuration support in `config_loader.py`
4. Add unit tests
5. Update routing documentation

### Debugging Test Failures

```bash
# Run with verbose output
uv run pytest tests/unit/test_file.py -v -s

# Run with debugger
uv run pytest tests/unit/test_file.py --trace

# Run specific test
uv run pytest tests/unit/test_file.py::test_function -v
```

## WORKFLOW & PUSHING CHANGES

### Task Group (TG) Workflow

For each Task Group (TG), follow this branch-based workflow to ensure clean history and
minimize remote push friction:

1.  **Create Feature Branch**:
    Name your branch using the convention `tg<id>-<short-desc>`.
    ```bash
    git checkout -b tg123-feature-name
    ```

2.  **Local Development**:
    Make **local commits freely**. You do not need to push every commit to the remote.
    Run tests and linting locally as needed.
    ```bash
    git add .
    git commit -m "feat: implement part 1"
    # ... more work ...
    git commit -m "fix: correct logic"
    ```

3.  **Squash Merge to Main**:
    When the TG is complete, squash all changes into a single commit on `main`.
    ```bash
    # Switch to main
    git checkout main

    # Squash merge the feature branch
    git merge --squash tg123-feature-name

    # Commit with a descriptive message
    git commit -m "feat: complete TG123 feature name"
    ```

4.  **Bulk Push**:
    Push to remote only when ready to publish (e.g., after squash merge).
    Use `rr` if local push is blocked.

### Remote Execution (rr - Road Runner)

Local `git push` may be blocked by Code Defender. When this happens, use the `rr`
(Road Runner) tool to push from a remote machine with unrestricted access.

**Do not use rr to push every small commit; use it to publish milestones (e.g., squashed TG commit) or to run GitHub Actions when needed.**

#### Available rr Push Tasks

| Task | Command | Description |
|------|---------|-------------|
| Normal push | `rr push` | Sync code and push to GitHub |
| Force push | `rr push-force` | Sync code and force push (`--force-with-lease`) |

#### Post-Push Sync (REQUIRED)

After any rr-based push, your local repo is behind `origin/main` (or the remote branch).
You **must** sync your local repo to avoid conflicts:

```bash
# After rr push of main:
git pull

# After rr push of a branch:
git fetch origin
# Optional: check status
git status

# After rr push-force (requires clean working tree):
git fetch origin
git reset --hard origin/main
```

> ⚠️ **Force-push safety**: Before running `git reset --hard`, ensure your working
> tree is clean (`git status`). Any uncommitted changes will be lost.

### Request Repo Approval (Alternative)

For a one-time permanent fix, request Code Defender approval:

```bash
git-defender --request-repo --url https://github.com/baladithyab/RouteIQ.git --reason 3
```

After approval, local `git push` will work normally.

See [`.rr.yaml`](.rr.yaml) for all available tasks and
[`docs/rr-workflow.md`](docs/rr-workflow.md) for detailed setup instructions.

## DOCUMENTATION

| Document | Purpose |
|----------|---------|
| [`README.md`](README.md) | Project overview |
| [`docs/deployment.md`](docs/deployment.md) | Deployment guide |
| [`docs/configuration.md`](docs/configuration.md) | Configuration options |
| [`docs/mcp-gateway.md`](docs/mcp-gateway.md) | MCP Protocol guide |
| [`docs/security.md`](docs/security.md) | Security considerations |
| [`docs/rr-workflow.md`](docs/rr-workflow.md) | Remote push workflow |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guidelines |

## WHEN IN DOUBT

1. Follow existing patterns in the codebase
2. Check similar implementations in `reference/litellm/` (read-only reference)
3. Run tests before committing
4. Ensure comprehensive test coverage
5. Keep security in mind for all changes
