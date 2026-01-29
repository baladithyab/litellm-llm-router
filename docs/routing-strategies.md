# Routing Strategies

This document covers all available routing strategies in the LiteLLM + LLMRouter gateway.

## Table of Contents

- [LiteLLM Built-in Strategies](#litellm-built-in-strategies)
- [LLMRouter ML-Based Strategies](#llmrouter-ml-based-strategies)
- [A/B Testing & Runtime Hot-Swapping](#ab-testing--runtime-hot-swapping)
- [Model Artifact Security](#model-artifact-security)
- [Hot Reloading](#hot-reloading)
- [Training Custom Models](#training-custom-models)

## LiteLLM Built-in Strategies

These strategies are built into LiteLLM:

### simple-shuffle (default)
Random load balancing across deployments.

```yaml
router_settings:
  routing_strategy: simple-shuffle
```

### least-busy
Routes to the deployment with the fewest active requests.

```yaml
router_settings:
  routing_strategy: least-busy
```

### latency-based-routing
Routes based on historical response latency.

```yaml
router_settings:
  routing_strategy: latency-based-routing
```

### cost-based-routing
Routes to minimize token costs.

```yaml
router_settings:
  routing_strategy: cost-based-routing
```

## LLMRouter ML-Based Strategies

These strategies use machine learning models from [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) to intelligently route queries. All 18+ strategies are available across four categories.

### Single-Round Routers

#### llmrouter-knn
K-Nearest Neighbors based routing. Fast and interpretable.

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    config_path: /app/config/knn_config.yaml
```

**Best for:** Quick deployment, interpretable decisions

#### llmrouter-svm
Support Vector Machine routing. Good generalization with margin-based selection.

```yaml
router_settings:
  routing_strategy: llmrouter-svm
  routing_strategy_args:
    config_path: /app/config/svm_config.yaml
```

**Best for:** Binary routing decisions, high generalization

#### llmrouter-mlp
Multi-Layer Perceptron routing. Neural network based.

```yaml
router_settings:
  routing_strategy: llmrouter-mlp
  routing_strategy_args:
    config_path: /app/config/mlp_config.yaml
```

**Best for:** Complex query patterns, high accuracy requirements

#### llmrouter-mf
Matrix Factorization routing. Collaborative filtering approach.

```yaml
router_settings:
  routing_strategy: llmrouter-mf
  routing_strategy_args:
    config_path: /app/config/mf_config.yaml
```

**Best for:** User preference learning, collaborative scenarios

#### llmrouter-elo
Elo Rating based routing. Uses competitive ranking to select models.

```yaml
router_settings:
  routing_strategy: llmrouter-elo
  routing_strategy_args:
    config_path: /app/config/elo_config.yaml
```

**Best for:** Model quality ranking, tournament-style selection

#### llmrouter-routerdc
Dual Contrastive learning based routing (RouterDC). Uses contrastive representations.

```yaml
router_settings:
  routing_strategy: llmrouter-routerdc
  routing_strategy_args:
    config_path: /app/config/routerdc_config.yaml
```

**Best for:** High accuracy, representation learning
**Reference:** [RouterDC (NeurIPS 2024)](https://arxiv.org/abs/2409.19886)

#### llmrouter-hybrid
HybridLLM probabilistic routing. Combines multiple signals.

```yaml
router_settings:
  routing_strategy: llmrouter-hybrid
  routing_strategy_args:
    config_path: /app/config/hybrid_config.yaml
```

**Best for:** Production deployments, balanced cost-quality decisions
**Reference:** [Hybrid LLM (ICLR 2024)](https://arxiv.org/abs/2404.14618)

#### llmrouter-causallm
Causal Language Model router. Transformer-based routing decisions.

```yaml
router_settings:
  routing_strategy: llmrouter-causallm
  routing_strategy_args:
    config_path: /app/config/causallm_config.yaml
```

**Best for:** Deep semantic understanding, complex reasoning tasks
**Note:** Requires PyTorch with transformers support

#### llmrouter-graph
Graph Neural Network routing. Models query-model relationships as graphs.

```yaml
router_settings:
  routing_strategy: llmrouter-graph
  routing_strategy_args:
    config_path: /app/config/graph_config.yaml
```

**Best for:** Relationship modeling, complex query dependencies
**Reference:** [GraphRouter (ICLR 2025)](https://arxiv.org/abs/2410.03834)
**Note:** Requires PyTorch Geometric

#### llmrouter-automix
Automatic model mixing. Dynamically blends responses from multiple models.

```yaml
router_settings:
  routing_strategy: llmrouter-automix
  routing_strategy_args:
    config_path: /app/config/automix_config.yaml
```

**Best for:** Ensemble approaches, quality optimization
**Reference:** [AutoMix (NeurIPS 2024)](https://arxiv.org/abs/2310.12963)

### Multi-Round Routers

#### llmrouter-r1
Pre-trained Router-R1 for multi-turn conversations. Uses reinforcement learning.

```yaml
router_settings:
  routing_strategy: llmrouter-r1
  routing_strategy_args:
    config_path: /app/config/r1_config.yaml
```

**Best for:** Multi-turn conversations, complex dialogues
**Reference:** [Router-R1 (NeurIPS 2025)](https://arxiv.org/abs/2506.09033)
**Note:** Requires vLLM (tested with vllm==0.6.3, torch==2.4.0)

### Personalized Routers

#### llmrouter-gmt
Graph-based personalized router with user preference learning.

```yaml
router_settings:
  routing_strategy: llmrouter-gmt
  routing_strategy_args:
    config_path: /app/config/gmt_config.yaml
```

**Best for:** Per-user model preferences, personalized experiences
**Reference:** [GMTRouter](https://arxiv.org/abs/2511.08590)

### Agentic Routers

#### llmrouter-knn-multiround
KNN-based agentic router for complex multi-step tasks.

```yaml
router_settings:
  routing_strategy: llmrouter-knn-multiround
  routing_strategy_args:
    config_path: /app/config/knn_multiround_config.yaml
```

**Best for:** Agentic workflows, multi-step reasoning

#### llmrouter-llm-multiround
LLM-based agentic router that uses an LLM to decide routing.

```yaml
router_settings:
  routing_strategy: llmrouter-llm-multiround
  routing_strategy_args:
    config_path: /app/config/llm_multiround_config.yaml
```

**Best for:** Complex agentic tasks, meta-reasoning
**Note:** Inference-only (no training required)

### Baseline Routers

#### llmrouter-smallest
Always routes to the smallest model. Useful for cost optimization baseline.

```yaml
router_settings:
  routing_strategy: llmrouter-smallest
  routing_strategy_args:
    config_path: /app/config/baseline_config.yaml
```

**Best for:** Cost minimization, testing, baseline comparison

#### llmrouter-largest
Always routes to the largest model. Useful for quality baseline.

```yaml
router_settings:
  routing_strategy: llmrouter-largest
  routing_strategy_args:
    config_path: /app/config/baseline_config.yaml
```

**Best for:** Maximum quality, testing, baseline comparison

### Custom Routers

#### llmrouter-custom
Load your own trained routing model from the custom routers directory.

```yaml
router_settings:
  routing_strategy: llmrouter-custom
  routing_strategy_args:
    config_path: /app/custom_routers/my_router_config.yaml
```

See [Creating Custom Routers](https://github.com/ulab-uiuc/LLMRouter#-creating-custom-routers) for details.

## Model Artifact Security

When using pickle-based models (required for sklearn KNN/SVM/MLP routers), RouteIQ provides manifest-based verification to prevent loading of unauthorized or tampered model files.

### Enabling Pickle Models with Verification

```yaml
# config.yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router.pkl
```

```bash
# Environment variables
LLMROUTER_ALLOW_PICKLE_MODELS=true
LLMROUTER_MODEL_MANIFEST_PATH=/app/models/manifest.json
LLMROUTER_MODEL_PUBLIC_KEY_B64=<your-ed25519-public-key>
```

When `LLMROUTER_ALLOW_PICKLE_MODELS=true`, manifest verification is automatically enforced unless explicitly bypassed with `LLMROUTER_ENFORCE_SIGNED_MODELS=false`.

See the [Security Guide](security.md#artifact-safety) for complete setup instructions.

## Hot Reloading

All LLMRouter strategies support hot reloading:

```yaml
router_settings:
  routing_strategy_args:
    hot_reload: true
    reload_interval: 300  # Check every 5 minutes
```

The gateway will automatically detect model file changes and reload.

## Training Custom Models

- [MLOps Training Guide](mlops-training.md) - Full training pipeline with Docker
- [Training from Observability Data](observability-training.md) - Use Jaeger/Tempo/CloudWatch traces
- [LLMRouter Data Pipeline](https://github.com/ulab-uiuc/LLMRouter#-preparing-training-data) - Official data preparation guide

## A/B Testing & Runtime Hot-Swapping

RouteIQ supports runtime strategy hot-swapping and A/B testing through a strategy registry and routing pipeline architecture. This allows you to:

- **Switch strategies at runtime** without restarts
- **Run A/B tests** with deterministic weighted selection
- **Fall back gracefully** when primary strategies fail
- **Emit telemetry** for observing routing decisions via OpenTelemetry

### Configuration

Configure A/B testing via environment variables:

```bash
# Set a single active strategy
LLMROUTER_ACTIVE_ROUTING_STRATEGY=llmrouter-knn

# Or configure A/B weights (JSON format)
LLMROUTER_STRATEGY_WEIGHTS='{"baseline": 90, "candidate": 10}'
```

When `LLMROUTER_STRATEGY_WEIGHTS` is set, the gateway performs weighted A/B selection. The weights are relative (not percentages):

- `{"baseline": 90, "candidate": 10}` → 90% baseline, 10% candidate
- `{"a": 1, "b": 1, "c": 1}` → ~33% each

### Deterministic Assignment

A/B selection is **deterministic** based on a hash key, ensuring:

1. **Same user → same variant**: If a `user_id` is present in the request, that user always gets the same strategy variant
2. **Same request → same variant**: If only `request_id` is available, that request is consistently assigned
3. **Reproducible experiments**: The same hash key always produces the same selection

Priority for hash key selection:
1. `metadata.user_id` or `user` in request
2. `metadata.request_id` or `litellm_call_id`
3. Random (no stickiness guarantee)

### Programmatic Configuration

You can also configure A/B testing programmatically:

```python
from litellm_llmrouter import (
    get_routing_registry,
    RoutingStrategy,
    RoutingContext,
)

# Get the global registry
registry = get_routing_registry()

# Custom strategy implementation
class MyCustomStrategy(RoutingStrategy):
    def select_deployment(self, context: RoutingContext):
        # Your custom routing logic
        router = context.router
        # ... analyze context.messages, context.model, etc.
        return deployment_dict  # or None

# Register strategies
registry.register("baseline", ExistingStrategy())
registry.register("candidate", MyCustomStrategy())

# Option 1: Set single active strategy
registry.set_active("baseline")

# Option 2: Configure A/B weights
registry.set_weights({"baseline": 90, "candidate": 10})

# Check current status
status = registry.get_status()
# {
#     "registered_strategies": ["baseline", "candidate"],
#     "active_strategy": None,
#     "ab_weights": {"baseline": 90, "candidate": 10},
#     "ab_enabled": True
# }

# Clear A/B and revert to single strategy
registry.clear_weights()
```

### Telemetry & Observability

When A/B testing is enabled, routing decisions emit telemetry via the `routeiq.router_decision.v1` contract as OpenTelemetry span events:

```json
{
  "contract_name": "routeiq.router_decision.v1",
  "strategy_name": "candidate",
  "selected_deployment": "gpt-4-turbo",
  "selection_reason": "ab_test",
  "custom_attributes": {
    "ab_enabled": true,
    "ab_weights": {"baseline": 90, "candidate": 10},
    "ab_hash_key": "user:abc123..."
  },
  "timings": {
    "total_ms": 2.5
  },
  "outcome": {
    "status": "success"
  }
}
```

Use this telemetry to:
- Compare strategy performance in observability tools (Jaeger, Grafana, etc.)
- Build dashboards showing A/B test results
- Extract data for offline analysis and model retraining

### Fallback Behavior

The routing pipeline supports automatic fallback:

1. **Primary strategy selected** via registry (A/B or active)
2. **If primary fails**, fallback to default strategy
3. **Telemetry marks fallback** for analysis

```python
# Fallback is tracked in routing results
result = pipeline.route(context)
if result.is_fallback:
    print(f"Fell back due to: {result.fallback_reason}")
```

### Disabling Pipeline Routing

To disable pipeline routing and revert to direct LLMRouter calls:

```bash
LLMROUTER_USE_PIPELINE=false
```

This maintains backward compatibility while allowing the new A/B testing features to be opted into.

### Example: Canary Deployment

Run a new routing strategy on 5% of traffic:

```bash
# Step 1: Deploy with small canary weight
LLMROUTER_STRATEGY_WEIGHTS='{"production": 95, "canary-v2": 5}'

# Step 2: Monitor telemetry for errors/latency
# Look for strategy_name="canary-v2" in traces

# Step 3: Gradually increase weight
LLMROUTER_STRATEGY_WEIGHTS='{"production": 80, "canary-v2": 20}'

# Step 4: Full rollout
LLMROUTER_ACTIVE_ROUTING_STRATEGY=canary-v2
unset LLMROUTER_STRATEGY_WEIGHTS
```

### Thread Safety

The registry and pipeline are thread-safe:
- Multiple strategies can be registered concurrently
- Weight updates are atomic
- Selection operations don't block on writes

All configuration updates trigger registered callbacks for integration with admin endpoints or monitoring systems.
