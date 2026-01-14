---
inclusion: fileMatch
fileMatchPattern: "**/strategies.py"
---

# LLMRouter Strategies Reference

## Available Strategies

### Single-Round Routers

| Strategy | Class | Description |
|----------|-------|-------------|
| `llmrouter-knn` | KNNRouter | K-Nearest Neighbors based routing |
| `llmrouter-svm` | SVMRouter | Support Vector Machine routing |
| `llmrouter-mlp` | MLPRouter | Multi-Layer Perceptron routing |
| `llmrouter-mf` | MFRouter | Matrix Factorization routing |
| `llmrouter-elo` | EloRouter | Elo Rating based routing |
| `llmrouter-routerdc` | RouterDC | Dual Contrastive learning (NeurIPS 2024) |
| `llmrouter-hybrid` | HybridLLMRouter | Probabilistic hybrid (ICLR 2024) |
| `llmrouter-causallm` | CausalLMRouter | Transformer-based (optional) |
| `llmrouter-graph` | GraphRouter | Graph neural network (ICLR 2025, optional) |
| `llmrouter-automix` | AutomixRouter | Automatic model mixing (NeurIPS 2024) |

### Multi-Round Routers

| Strategy | Class | Description |
|----------|-------|-------------|
| `llmrouter-r1` | RouterR1 | Pre-trained multi-turn router (requires vLLM) |

### Personalized Routers

| Strategy | Class | Description |
|----------|-------|-------------|
| `llmrouter-gmt` | GMTRouter | Graph-based personalized router |

### Agentic Routers

| Strategy | Class | Description |
|----------|-------|-------------|
| `llmrouter-knn-multiround` | KNNMultiRoundRouter | KNN agentic router |
| `llmrouter-llm-multiround` | LLMMultiRoundRouter | LLM agentic router |

### Baseline Routers

| Strategy | Class | Description |
|----------|-------|-------------|
| `llmrouter-smallest` | SmallestLLM | Always picks smallest (cost baseline) |
| `llmrouter-largest` | LargestLLM | Always picks largest (quality baseline) |
| `llmrouter-custom` | Custom | User-defined custom router |

## Strategy Configuration

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router
    llm_data_path: /app/config/llm_candidates.json
    hot_reload: true
    reload_interval: 300
    model_s3_bucket: my-bucket  # Optional
    model_s3_key: models/knn/   # Optional
```

## LLMRouterStrategyFamily Class

### Initialization

```python
strategy = LLMRouterStrategyFamily(
    strategy_name="llmrouter-knn",
    model_path="/app/models/knn_router",
    llm_data_path="/app/config/llm_candidates.json",
    hot_reload=True,
    reload_interval=300,
    model_s3_bucket=None,
    model_s3_key=None
)
```

### Key Methods

- `router` (property): Get router instance, auto-reloads if needed
- `_load_router()`: Load the appropriate LLMRouter model
- `_should_reload()`: Check if model should be reloaded
- `_load_llm_data()`: Load LLM candidates from JSON

### Thread Safety

Uses `threading.RLock()` for thread-safe model access:

```python
@property
def router(self):
    with self._router_lock:
        if self._router is None or self._should_reload():
            self._router = self._load_router()
            self._last_load_time = time.time()
    return self._router
```

## Adding a New Strategy

1. Add to `LLMROUTER_STRATEGIES` list:
```python
LLMROUTER_STRATEGIES = [
    # ... existing strategies
    "llmrouter-newstrategy",
]
```

2. Add to router map in `_load_router()`:
```python
router_map = {
    # ... existing mappings
    "newstrategy": ("NewStrategyRouter", False),  # (class_name, is_optional)
}
```

3. Ensure the router class exists in `llmrouter.models`

## LLM Candidates JSON Format

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
    },
    {
      "model_name": "claude-3-opus",
      "provider": "anthropic",
      "cost_per_1k_tokens": 0.015,
      "latency_p50_ms": 400,
      "latency_p95_ms": 1000,
      "context_window": 200000,
      "capabilities": ["chat", "vision"]
    }
  ]
}
```

## Training Models

Use the MLOps pipeline in `examples/mlops/`:

```bash
cd examples/mlops
docker compose -f docker-compose.mlops.yml up -d

# Train KNN router
python train.py --strategy knn --data training_data.json --output /app/models/knn_router
```

## Hot Reload

When `hot_reload: true`:
1. Strategy checks file mtime every `reload_interval` seconds
2. If mtime changed, reloads model
3. Thread-safe reload with RLock

For S3-based models:
1. Checks S3 ETag periodically
2. Downloads only if ETag changed
3. Triggers reload after download
