# Routing Strategies

This document covers all available routing strategies in the LiteLLM + LLMRouter gateway.

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

These strategies use machine learning models to intelligently route queries.

### llmrouter-knn
K-Nearest Neighbors based routing. Fast and interpretable.

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router.pt
    llm_data_path: /app/config/llm_candidates.json
```

**Best for:** Quick deployment, interpretable decisions

### llmrouter-svm
Support Vector Machine routing. Good generalization.

```yaml
router_settings:
  routing_strategy: llmrouter-svm
  routing_strategy_args:
    model_path: /app/models/svm_router.pt
```

**Best for:** Binary routing decisions, margin-based selection

### llmrouter-mlp
Multi-Layer Perceptron routing. Neural network based.

```yaml
router_settings:
  routing_strategy: llmrouter-mlp
  routing_strategy_args:
    model_path: /app/models/mlp_router.pt
```

**Best for:** Complex query patterns, high accuracy requirements

### llmrouter-mf
Matrix Factorization routing. Collaborative filtering approach.

```yaml
router_settings:
  routing_strategy: llmrouter-mf
  routing_strategy_args:
    model_path: /app/models/mf_router.pt
```

**Best for:** User preference learning, collaborative scenarios

### llmrouter-bert
BERT-based routing. Transformer embeddings for queries.

```yaml
router_settings:
  routing_strategy: llmrouter-bert
  routing_strategy_args:
    model_path: /app/models/bert_router.pt
```

**Best for:** Semantic understanding, complex text analysis

### llmrouter-hybrid
Combines multiple routing signals probabilistically.

```yaml
router_settings:
  routing_strategy: llmrouter-hybrid
  routing_strategy_args:
    model_path: /app/models/hybrid_router.pt
    threshold: 0.7
```

**Best for:** Production deployments, balanced decisions

### llmrouter-custom
Load your own trained routing model.

```yaml
router_settings:
  routing_strategy: llmrouter-custom
  routing_strategy_args:
    model_path: /app/models/my_custom_router.pt
    config_path: /app/configs/custom_router_config.yaml
```

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

See [MLOps Training Guide](mlops-training.md) for training your own routing models.

