# MLOps Training Setup for LLMRouter

This directory contains a complete MLOps setup for training, evaluating, and deploying LLMRouter routing models with MLflow experiment tracking.

## Overview

The MLOps pipeline enables you to:
- **Generate training data** from Jaeger traces or synthetic data
- **Train multiple router types** (KNN, SVM, MLP, Matrix Factorization, BERT, CausalLM, Hybrid)
- **Track experiments** with MLflow including parameters, metrics, and artifacts
- **Deploy models** to S3 or local storage for use with LiteLLM gateway

## Telemetry Contract

### Contract: `routeiq.router_decision.v1`

The RouteIQ gateway emits a **versioned telemetry event** for every routing decision, following a stable contract for MLOps consumption.

| Property | Value |
|----------|-------|
| **Contract Name** | `routeiq.router_decision.v1` |
| **Event Name** | `routeiq.router_decision.v1` |
| **Payload Key** | `routeiq.router_decision.payload` |
| **Version** | `v1` |
| **PII-Safe** | Yes (no query/response content) |

### Schema Fields

```json
{
  "contract_version": "v1",
  "contract_name": "routeiq.router_decision.v1",
  "event_id": "uuid",
  "trace_id": "32-hex-trace-id",
  "span_id": "16-hex-span-id",
  "timestamp_utc": "2024-01-15T10:30:00.000Z",
  "timestamp_unix_ms": 1705315800000,
  
  "input": {
    "query_length": 150,
    "requested_model": "gpt-4 | null",
    "user_id": "hashed-user-id | null",
    "team_id": "team-id | null"
  },
  
  "strategy_name": "llmrouter-knn",
  "strategy_version": "1.0.0 | null",
  
  "candidate_deployments": [
    {"model_name": "gpt-4", "provider": "openai", "score": 0.95, "available": true}
  ],
  
  "selected_deployment": "gpt-4",
  "selection_reason": "highest_score",
  
  "timings": {
    "total_ms": 15.5,
    "strategy_ms": 10.2,
    "embedding_ms": 3.1
  },
  
  "outcome": {
    "status": "success | failure | fallback | error | no_candidates | timeout",
    "input_tokens": 100,
    "output_tokens": 200,
    "total_tokens": 300
  },
  
  "fallback": {
    "fallback_triggered": false,
    "original_model": null,
    "fallback_reason": null
  }
}
```

### Key Fields for Training

| Field | Description | Training Use |
|-------|-------------|--------------|
| `strategy_name` | Strategy that made the decision | Model comparison |
| `candidate_deployments` | Models considered | Training labels |
| `selected_deployment` | Model selected | Ground truth |
| `outcome.status` | Success/failure | Performance labeling |
| `timings.total_ms` | Decision latency | Performance metrics |
| `input.query_length` | Query size (PII-safe) | Feature engineering |

### End-to-End Pipeline

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Gateway Emits  │───▶│ Jaeger/Tempo     │───▶│ Extract Script  │
│  Telemetry      │    │ Stores Events    │    │ Parses JSON     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Deploy Model   │◀───│ MLflow Tracks    │◀───│ Convert Script  │
│  to Gateway     │    │ Experiments      │    │ Creates JSONL   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

1. **Gateway emits** `routeiq.router_decision.v1` events as OTEL span events
2. **Jaeger/Tempo** stores the spans with embedded JSON payloads
3. **Extract script** queries Jaeger API and parses the versioned events
4. **Convert script** transforms to LLMRouter training format (JSONL + embeddings)
5. **MLflow** tracks training experiments and model artifacts
6. **Deploy** models to S3/local for hot-reload by the gateway

### Source of Truth

The contract is defined in [`src/litellm_llmrouter/telemetry_contracts.py`](../../src/litellm_llmrouter/telemetry_contracts.py):

- `CONTRACT_VERSION = "v1"`
- `CONTRACT_NAME = "routeiq.router_decision"`
- `ROUTER_DECISION_EVENT_NAME` - Span event name
- `ROUTER_DECISION_PAYLOAD_KEY` - Attribute key for JSON payload

The extraction scripts import from this module to ensure consistency.

## Quick Start

### Option 1: Local Development (Recommended for Testing)

```bash
# 1. Install dependencies
uv pip install --system mlflow torch sentence-transformers scikit-learn pandas pyyaml click

# 2. Start MLflow server
cd examples/mlops
docker compose -f docker-compose.mlops.yml up -d mlflow

# 3. Generate synthetic training data
mkdir -p /tmp/llmrouter_training
MLFLOW_TRACKING_URI=http://localhost:5000 \
mlflow run . --entry-point generate_synthetic \
  -P count=500 \
  -P output_dir=/tmp/llmrouter_training \
  --env-manager local

# 4. Train a router model
MLFLOW_TRACKING_URI=http://localhost:5000 \
mlflow run . --entry-point train \
  -P router_type=knn \
  -P config=$(pwd)/configs/test_knn_config.yaml \
  --env-manager local

# 5. View results in MLflow UI
open http://localhost:5000
```

### Option 2: Full Docker Stack

```bash
# Start all services
docker compose -f docker-compose.mlops.yml up -d

# Access services:
# - MLflow UI: http://localhost:5000
# - Jupyter Lab: http://localhost:8888 (token: llmrouter)
# - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
```

## Components

| Service | Port | Description |
|---------|------|-------------|
| MLflow | 5000 | Experiment tracking & model registry |
| MinIO | 9000/9001 | S3-compatible object storage for artifacts |
| Jupyter | 8888 | Interactive development environment |
| Trainer | - | GPU-enabled training container |
| Deployer | - | Model deployment to production |

## MLflow Project Entry Points

The `MLproject` file defines several entry points:

### `generate_synthetic` - Generate Training Data
```bash
mlflow run . --entry-point generate_synthetic \
  -P count=500 \
  -P output_dir=/tmp/llmrouter_training \
  --env-manager local
```

### `train` - Train a Router Model
```bash
mlflow run . --entry-point train \
  -P router_type=knn \
  -P config=configs/knn_config.yaml \
  -P experiment_name=llmrouter-training \
  --env-manager local
```

Supported router types:
- `knn` - K-Nearest Neighbors router
- `svm` - Support Vector Machine router
- `mlp` - Multi-Layer Perceptron router
- `mf` - Matrix Factorization router
- `bert` - BERT-based router (RouterDC)
- `causallm` - Causal Language Model router
- `hybrid` - Hybrid LLM router

### `prepare_data` - Extract from Jaeger Traces
```bash
mlflow run . --entry-point prepare_data \
  -P jaeger_url=http://jaeger:16686 \
  -P service_name=litellm-gateway \
  -P hours_back=24 \
  --env-manager local
```

### `deploy` - Deploy Trained Model
```bash
mlflow run . --entry-point deploy \
  -P run_id=<mlflow_run_id> \
  -P target_bucket=llmrouter-models \
  -P model_name=router_model \
  --env-manager local
```

## Training Data Format

### Input: Traces (JSONL)
```json
{"query": "What is machine learning?", "model_name": "claude-3-haiku", "latency_ms": 150, "cost": 0.001, "quality_score": 0.85, "timestamp": "2024-01-15T10:30:00Z"}
```

### Output: Routing Data (JSONL)
```json
{"query": "What is machine learning?", "model_name": "claude-3-haiku", "performance": 0.85, "embedding_id": 0}
```

### Output: Query Embeddings (PyTorch)
```python
# query_embeddings.pt - Shape: [num_unique_queries, embedding_dim]
torch.Size([100, 384])  # Using all-MiniLM-L6-v2
```

## Configuration Files

### KNN Router Config (`configs/knn_config.yaml`)
```yaml
data_path:
  routing_data_train: /tmp/llmrouter_training/routing_train.jsonl
  routing_data_test: /tmp/llmrouter_training/routing_test.jsonl
  query_embedding_data: /tmp/llmrouter_training/query_embeddings.pt
  llm_data: /tmp/llmrouter_training/llm_data.json

hparam:
  k: 5
  distance_metric: cosine
  weighted: true
```

### MLP Router Config (`configs/mlp_config.yaml`)
```yaml
data_path:
  routing_data_train: /tmp/llmrouter_training/routing_train.jsonl
  routing_data_test: /tmp/llmrouter_training/routing_test.jsonl
  query_embedding_data: /tmp/llmrouter_training/query_embeddings.pt
  llm_data: /tmp/llmrouter_training/llm_data.json

hparam:
  hidden_dims: [256, 128]
  learning_rate: 0.001
  epochs: 100
  batch_size: 32
  dropout: 0.2
```


## Directory Structure

```
examples/mlops/
├── docker-compose.mlops.yml   # MLOps stack definition
├── MLproject                  # MLflow project definition
├── python_env.yaml            # Python environment for MLflow
├── Dockerfile.trainer         # Training environment (GPU-enabled)
├── Dockerfile.deployer        # Deployment tools
├── scripts/
│   ├── train_router.py               # Main training script
│   ├── generate_synthetic_traces.py  # Synthetic data generator
│   ├── convert_traces_to_llmrouter.py # Convert traces to training format
│   ├── extract_jaeger_traces.py      # Extract traces from Jaeger
│   └── deploy_model.py               # Model deployment script
├── configs/
│   ├── knn_config.yaml        # KNN router configuration
│   ├── mlp_config.yaml        # MLP router configuration
│   └── test_knn_config.yaml   # Test configuration
├── data/                      # Training data directory
├── models/                    # Trained models output
└── notebooks/                 # Jupyter notebooks for exploration
```

## Integration with LiteLLM Gateway

After training a model, you can integrate it with the LiteLLM gateway:

### 1. Configure Router Strategy

In your LiteLLM config (`litellm_config.yaml`):

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /path/to/trained/model
    # Or load from S3:
    model_s3_bucket: your-bucket
    model_s3_key: models/knn-router/
    hot_reload: true
    reload_interval: 300  # Check for updates every 5 minutes
```

### 2. Use Custom Routing Callback

```python
from litellm import Router
from litellm.integrations.llmrouter import LLMRouterCallback

router = Router(
    model_list=[...],
    routing_strategy="llmrouter-knn",
    routing_strategy_args={
        "model_path": "/tmp/models/knn_router"
    }
)

# The router will use the trained model for routing decisions
response = router.completion(
    model="gpt-4",  # Will be routed based on query
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

## GPU Training

The trainer container supports NVIDIA GPUs. Ensure you have:
- NVIDIA drivers installed
- nvidia-container-toolkit installed

GPU resources are automatically allocated via Docker for training intensive models like BERT and CausalLM routers.

## Using with AWS S3

Set AWS credentials for artifact storage:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

Deploy trained model to S3:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 \
mlflow run . --entry-point deploy \
  -P run_id=<your_mlflow_run_id> \
  -P target_bucket=your-production-bucket \
  -P model_name=knn-router \
  --env-manager local
```

## Troubleshooting

### MLflow Connection Issues
```bash
# Check MLflow is running
curl http://localhost:5000/health

# Restart MLflow
docker compose -f docker-compose.mlops.yml restart mlflow
```

### Missing Dependencies
```bash
# Install all required packages
uv pip install --system mlflow torch sentence-transformers scikit-learn pandas pyyaml click boto3
```

### Permission Errors with Artifacts
When running locally, artifact logging may fail due to Docker volume permissions. Models are still saved to the local `output_dir`.

### Data Path Issues
Ensure your config file paths match where data was generated:
```yaml
data_path:
  routing_data_train: /tmp/llmrouter_training/routing_train.jsonl  # Must exist!
```

## Example: Full Training Pipeline

```bash
# 1. Start MLflow
cd examples/mlops
docker compose -f docker-compose.mlops.yml up -d mlflow

# 2. Generate training data (500 samples)
mkdir -p /tmp/llmrouter_training
MLFLOW_TRACKING_URI=http://localhost:5000 \
mlflow run . --entry-point generate_synthetic \
  -P count=500 \
  -P output_dir=/tmp/llmrouter_training \
  --env-manager local

# 3. Train KNN router
MLFLOW_TRACKING_URI=http://localhost:5000 \
mlflow run . --entry-point train \
  -P router_type=knn \
  -P config=$(pwd)/configs/test_knn_config.yaml \
  --env-manager local

# 4. Train SVM router (update config paths first)
MLFLOW_TRACKING_URI=http://localhost:5000 \
mlflow run . --entry-point train \
  -P router_type=svm \
  -P config=$(pwd)/configs/test_knn_config.yaml \
  --env-manager local

# 5. View all experiments
open http://localhost:5000

# 6. Clean up
docker compose -f docker-compose.mlops.yml down
```

## Related Documentation

- [LLMRouter Library](https://github.com/your-org/llmrouter) - The underlying routing library
- [LiteLLM Documentation](https://docs.litellm.ai/) - LiteLLM gateway documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/) - MLflow experiment tracking
