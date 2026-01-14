# Coding Standards

## Python Style

Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with these specifics:

### Formatting
- Use **Black** for code formatting (line length 88)
- Use **Ruff** for linting
- Use **MyPy** for type checking

### Type Hints
Always use type hints for function signatures:

```python
# Good
def load_router(self, model_path: str) -> Router | None:
    ...

# Bad
def load_router(self, model_path):
    ...
```

### Docstrings
Use Google-style docstrings:

```python
def register_strategy(strategy_name: str, callback: Callable) -> bool:
    """Register a routing strategy callback.
    
    Args:
        strategy_name: Name of the strategy (e.g., 'llmrouter-knn')
        callback: Function to call when strategy is invoked
        
    Returns:
        True if registration succeeded, False otherwise
        
    Raises:
        ValueError: If strategy_name is invalid
    """
```

### Imports
Order imports as:
1. Standard library
2. Third-party packages
3. Local imports

```python
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from .config_sync import get_sync_manager
```

## Error Handling

### Use Specific Exceptions
```python
# Good
raise ValueError(f"Unknown strategy: {strategy_name}")

# Bad
raise Exception("Error")
```

### Log Errors with Context
```python
verbose_proxy_logger.error(
    f"Failed to load router {strategy_name}: {e}",
    extra={"strategy": strategy_name, "model_path": model_path}
)
```

## Thread Safety

Use locks for shared state:

```python
class LLMRouterStrategyFamily:
    def __init__(self):
        self._router_lock = threading.RLock()
        
    @property
    def router(self):
        with self._router_lock:
            if self._should_reload():
                self._router = self._load_router()
        return self._router
```

## Configuration

### Environment Variables
Use `os.environ.get()` with defaults:

```python
model_path = os.environ.get("LLMROUTER_MODEL_PATH", "/app/models")
hot_reload = os.environ.get("CONFIG_HOT_RELOAD", "false").lower() == "true"
```

### YAML Config
Support `os.environ/` prefix for secrets:

```yaml
api_key: os.environ/OPENAI_API_KEY
```

## API Endpoints

### Use Pydantic Models
```python
class AgentRegistration(BaseModel):
    agent_id: str
    name: str
    description: str
    url: str
    capabilities: list[str] = []
```

### Return Consistent Responses
```python
return {"status": "success", "agent_id": agent.agent_id}
return {"status": "failed", "error": str(e)}
```

### Use HTTPException for Errors
```python
if not gateway.is_enabled():
    raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")
```
