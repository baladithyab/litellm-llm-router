# MLOps Training Guide

This guide covers training, evaluating, and deploying custom LLMRouter routing models.

## Quick Start

```bash
cd examples/mlops
docker compose -f docker-compose.mlops.yml up -d
```

This starts:
- **MLflow** (port 5000) - Experiment tracking
- **MinIO** (port 9000/9001) - S3-compatible storage
- **Jupyter** (port 8888) - Interactive development
- **Trainer** - Training environment with GPU support

## Telemetry-Driven Training

RouteIQ emits versioned routing telemetry (contract: `routeiq.router_decision.v1`) that can be used
to continuously improve your routing models. See [Observability Training](observability-training.md)
for the full contract schema.

### Extracting Routing Decisions

```bash
# Extract routing decisions from Jaeger
python examples/mlops/scripts/extract_jaeger_traces.py \
    --jaeger-url http://localhost:16686 \
    --service-name litellm-gateway \
    --hours-back 168 \
    --output traces.jsonl \
    --routing-decisions-output routing_decisions.jsonl

# Convert to LLMRouter training format
python examples/mlops/scripts/convert_traces_to_llmrouter.py \
    --input traces.jsonl \
    --output-dir /app/data \
    --include-routing-metadata
```

The extraction scripts support both:
- **Legacy trace format** (span attributes)
- **Versioned events** (`routeiq.router_decision.v1`)

## Training a Router

### 1. Prepare Training Data

Create a dataset with query-model performance pairs:

```json
{
  "queries": [
    {"text": "Explain quantum computing", "best_model": "claude-3-opus"},
    {"text": "Write a hello world in Python", "best_model": "gpt-3.5-turbo"}
  ]
}
```

### 2. Configure Training

Create a YAML config:

```yaml
# configs/knn_config.yaml
data_path:
  train_data: /app/data/train.json
  llm_data: /app/data/llm_candidates.json

hparam:
  k: 5
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

### 3. Run Training

```bash
docker compose exec llmrouter-trainer python /app/scripts/train_router.py \
  --router-type knn \
  --config /app/configs/knn_config.yaml
```

### 4. View Results in MLflow

Open http://localhost:5000 to see:
- Training metrics
- Model artifacts
- Experiment comparisons

## Deploying Models

### Register in MLflow

```python
import mlflow

mlflow.register_model(
    f"runs:/{run_id}/model",
    "knn-router-production"
)
```

### Deploy to Production

```bash
docker compose exec model-deployer python /app/scripts/deploy_model.py \
  --model-name knn-router-production \
  --model-stage Production \
  --s3-bucket my-models-bucket
```

### Update Gateway Config

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_s3_bucket: my-models-bucket
    model_s3_key: models/knn-router-production/
    hot_reload: true
```

## Training Different Router Types

### KNN Router
```bash
python train_router.py --router-type knn --config configs/knn.yaml
```

### MLP Router
```bash
python train_router.py --router-type mlp --config configs/mlp.yaml
```

### BERT Router
```bash
python train_router.py --router-type bert --config configs/bert.yaml
```

## Jupyter Notebooks

Access Jupyter at http://localhost:8888 (token: `llmrouter`)

Example notebooks:
- `01_data_exploration.ipynb` - Analyze routing data
- `02_train_knn_router.ipynb` - Train KNN router
- `03_evaluate_routers.ipynb` - Compare router performance

## CI/CD Integration

Add to your GitHub Actions:

```yaml
- name: Train Router
  run: |
    docker compose -f examples/mlops/docker-compose.mlops.yml run \
      llmrouter-trainer python /app/scripts/train_router.py \
      --router-type ${{ inputs.router_type }} \
      --config ${{ inputs.config }}

- name: Deploy Model
  if: github.ref == 'refs/heads/main'
  run: |
    docker compose run model-deployer python /app/scripts/deploy_model.py \
      --model-name ${{ inputs.model_name }} \
      --model-stage Production
```

## Telemetry Contract Reference

The `routeiq.router_decision.v1` contract provides:

| Field | Description | Training Use |
|-------|-------------|--------------|
| `strategy_name` | Strategy that made the decision | Model comparison |
| `candidate_deployments` | Models considered | Training labels |
| `selected_deployment` | Model selected | Ground truth |
| `outcome.status` | Success/failure | Performance labeling |
| `timings.total_ms` | Decision latency | Performance metrics |

For the complete schema, see [Observability Training](observability-training.md#contract-routeiqrouter_decisionv1).
