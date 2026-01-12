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
