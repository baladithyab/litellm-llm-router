# Observability Guide

LiteLLM + LLMRouter supports OpenTelemetry (OTEL) for distributed tracing. This guide covers three backends:

1. **Jaeger** - Simple local setup
2. **Grafana Tempo** - Production-grade with S3 storage
3. **AWS CloudWatch** - Native AWS integration

---

## Quick Start: Jaeger (Recommended for Development)

```bash
docker compose -f docker-compose.otel.yml up -d
```

- **Gateway**: http://localhost:4000
- **Jaeger UI**: http://localhost:16686

Make a request and view traces:
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-haiku", "messages": [{"role": "user", "content": "Hello"}]}'
```

Then open Jaeger UI → Select "litellm-gateway" service → Find Traces

---

## Option 1: Jaeger

Best for: Local development, debugging, simple deployments

### Environment Variables

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
  - OTEL_SERVICE_NAME=litellm-gateway
  - OTEL_TRACES_EXPORTER=otlp
```

### Standalone Jaeger

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -e COLLECTOR_OTLP_ENABLED=true \
  jaegertracing/all-in-one:1.54
```

---

## Option 2: Grafana Tempo

Best for: Production, S3 storage, Grafana ecosystem integration

### Environment Variables

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
  - OTEL_SERVICE_NAME=litellm-gateway
  - OTEL_TRACES_EXPORTER=otlp
```

### Tempo with S3 Backend

Create `tempo-config.yaml`:
```yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
        http:
          endpoint: 0.0.0.0:4318

storage:
  trace:
    backend: s3
    s3:
      bucket: your-tempo-bucket
      endpoint: s3.us-east-1.amazonaws.com
      region: us-east-1
      # Uses IAM role - no credentials needed on EC2
```

```bash
docker run -d --name tempo \
  -p 3200:3200 \
  -p 4317:4317 \
  -v ./tempo-config.yaml:/etc/tempo.yaml \
  grafana/tempo:latest \
  -config.file=/etc/tempo.yaml
```

### Grafana Dashboard

Add Tempo as a data source in Grafana:
- URL: `http://tempo:3200`
- Enable TraceQL for querying

---

## Option 3: AWS CloudWatch (X-Ray)

Best for: AWS-native, no additional infrastructure, IAM-based auth

### Using AWS Distro for OpenTelemetry (ADOT)

The ADOT Collector receives OTLP traces and exports to CloudWatch X-Ray.

#### Run ADOT Collector as Sidecar

```yaml
# docker-compose.cloudwatch.yml
services:
  litellm-gateway:
    # ... your gateway config ...
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://adot-collector:4317
      - OTEL_SERVICE_NAME=litellm-gateway
      - OTEL_TRACES_EXPORTER=otlp

  adot-collector:
    image: amazon/aws-otel-collector:latest
    command: ["--config=/etc/otel-config.yaml"]
    volumes:
      - ./config/otel-collector-config.yaml:/etc/otel-config.yaml:ro
    # Uses IAM Instance Profile on EC2 - no credentials needed
```

#### ADOT Collector Config (`config/otel-collector-config.yaml`)

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 50

exporters:
  awsxray:
    region: us-east-1
    # No credentials needed on EC2 with IAM role

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [awsxray]
```

#### Required IAM Permissions

Attach this policy to your EC2 instance role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "xray:PutTraceSegments",
        "xray:PutTelemetryRecords",
        "xray:GetSamplingRules",
        "xray:GetSamplingTargets"
      ],
      "Resource": "*"
    }
  ]
}
```

#### View Traces

1. Open AWS Console → CloudWatch → X-Ray traces
2. Filter by service name: `litellm-gateway`
3. Use Service Map for dependency visualization

---

## Trace Attributes

The gateway automatically adds these attributes to spans:

| Attribute | Description |
|-----------|-------------|
| `llm.model` | Target model name |
| `llm.provider` | Provider (bedrock, openai, etc.) |
| `llm.tokens.prompt` | Input token count |
| `llm.tokens.completion` | Output token count |
| `http.status_code` | Response status |

---

## Disabling Tracing

To disable OTEL tracing entirely:

```yaml
environment:
  - OTEL_TRACES_EXPORTER=none
  - OTEL_METRICS_EXPORTER=none
  - OTEL_LOGS_EXPORTER=none
```

---

## Trace Sampling Configuration

In high-traffic production environments, sampling all traces can be expensive (storage, network, processing). RouteIQ Gateway supports configurable trace sampling to control the volume of exported traces.

### Sampling Options

#### Option 1: OTEL Standard Environment Variables (Recommended)

Use standard OpenTelemetry environment variables for maximum portability:

| Variable | Description |
|----------|-------------|
| `OTEL_TRACES_SAMPLER` | Sampler type (see below) |
| `OTEL_TRACES_SAMPLER_ARG` | Argument for ratio-based samplers (0.0-1.0) |

**Supported Sampler Types:**

| Sampler | Description |
|---------|-------------|
| `always_on` | Sample 100% of traces (default) |
| `always_off` | Sample 0% of traces |
| `traceidratio` | Sample based on trace ID ratio |
| `parentbased_always_on` | Respect parent sampling decision, default to always_on |
| `parentbased_always_off` | Respect parent sampling decision, default to always_off |
| `parentbased_traceidratio` | Respect parent sampling decision, default to ratio (recommended) |

**Example: Sample 10% of traces in production**
```yaml
environment:
  - OTEL_TRACES_SAMPLER=parentbased_traceidratio
  - OTEL_TRACES_SAMPLER_ARG=0.1
```

#### Option 2: Simple Rate Configuration

For convenience, use `LLMROUTER_OTEL_SAMPLE_RATE` for simple percentage-based sampling:

```yaml
environment:
  # Sample 10% of traces
  - LLMROUTER_OTEL_SAMPLE_RATE=0.1
```

This is equivalent to `parentbased_traceidratio` with the specified rate.

### Sampling Best Practices

1. **Use ParentBased samplers** - They respect incoming trace context, ensuring distributed traces aren't broken mid-flight.

2. **Start conservative** - In high-traffic environments, start with 1-5% sampling and adjust based on:
   - Storage costs
   - Query performance in your tracing backend
   - Debugging needs

3. **Increase sampling for errors** - Consider using head-based sampling with tail-based sampling in your collector to capture all error traces.

4. **Development vs Production**:
   - Development: `always_on` (100%)
   - Staging: 50-100%
   - Production: 1-10% depending on traffic

### Programmatic Sampler Configuration

For advanced use cases, you can provide a custom sampler when initializing observability:

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased
from litellm_llmrouter.observability import ObservabilityManager

# Custom sampler with 5% sampling
custom_sampler = ParentBased(root=TraceIdRatioBased(0.05))

manager = ObservabilityManager(
    service_name="my-gateway",
    sampler=custom_sampler,
)
manager.initialize()
```

---

## Multiprocess Metrics (Gunicorn/Uvicorn Workers)

When running the gateway with multiple worker processes (common in production), metrics require special handling to avoid duplicates or data loss.

### The Problem

Each worker process maintains its own metric state. Without coordination:
- **Counter/Histogram metrics** may be duplicated (each worker reports independently)
- **Gauge metrics** may conflict (last-write-wins from different workers)
- **Memory usage** increases with worker count

### Recommended: OTEL Collector Aggregation

The recommended approach is to use an OpenTelemetry Collector as an aggregation point:

```
┌─────────────┐     ┌─────────────┐
│  Worker 1   │────►│             │
├─────────────┤     │    OTEL     │     ┌─────────────┐
│  Worker 2   │────►│  Collector  │────►│ Prometheus  │
├─────────────┤     │             │     │  / Jaeger   │
│  Worker N   │────►│             │     └─────────────┘
└─────────────┘     └─────────────┘
                    (aggregation)
```

**Configuration:**
```yaml
# docker-compose.yml
services:
  gateway:
    image: ghcr.io/baladithyab/litellm-llm-router:latest
    deploy:
      replicas: 4  # Multiple workers
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      - OTEL_METRICS_EXPORTER=otlp
      # Each worker sends via OTLP; collector aggregates
```

**OTEL Collector Config:**
```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  batch:
    timeout: 10s
    send_batch_size: 1024

exporters:
  prometheus:
    endpoint: 0.0.0.0:8889
    # Collector handles aggregation across workers

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

### Alternative: Prometheus Multiprocess Mode

For Prometheus pull-based metrics without an OTEL Collector:

```yaml
environment:
  # Directory for shared metric state (use tmpfs for performance)
  - PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
```

**Kubernetes:**
```yaml
# deployment.yaml
spec:
  containers:
    - name: gateway
      env:
        - name: PROMETHEUS_MULTIPROC_DIR
          value: /tmp/prometheus_multiproc
      volumeMounts:
        - name: prometheus-multiproc
          mountPath: /tmp/prometheus_multiproc
  volumes:
    - name: prometheus-multiproc
      emptyDir:
        medium: Memory  # tmpfs for performance
```

**Docker Compose:**
```yaml
services:
  gateway:
    environment:
      - PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
    volumes:
      - type: tmpfs
        target: /tmp/prometheus_multiproc
```

### Worker Configuration Examples

**Gunicorn with multiple workers:**
```bash
gunicorn litellm_llmrouter.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:4000
```

**Uvicorn with workers:**
```bash
uvicorn litellm_llmrouter.main:app \
  --workers 4 \
  --host 0.0.0.0 \
  --port 4000
```

### Multiprocess Metrics Best Practices

1. **Prefer OTEL Collector** - It handles aggregation transparently and works with any backend.

2. **Use tmpfs for PROMETHEUS_MULTIPROC_DIR** - Avoid disk I/O for metric state.

3. **Monitor worker health** - Ensure all workers are exporting successfully.

4. **Consider stateless metrics** - Design metrics that don't require cross-worker state (e.g., request counters per-worker that get summed).

---

## Log Level Configuration

Control the verbosity of structured logging:

```yaml
environment:
  # Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - LOG_LEVEL=INFO
  
  # Enable query logging (disabled by default for privacy)
  - LOG_QUERIES=false
```

---

## Troubleshooting

### No traces appearing
1. Verify `OTEL_EXPORTER_OTLP_ENDPOINT` is set and reachable
2. Check `OTEL_TRACES_EXPORTER=otlp` (not `none`)
3. Ensure sampling isn't set to `always_off` or 0.0

### Duplicate metrics in multiprocess mode
1. Use OTEL Collector for aggregation
2. Or set `PROMETHEUS_MULTIPROC_DIR` correctly

### High trace volume / costs
1. Enable sampling: `LLMROUTER_OTEL_SAMPLE_RATE=0.1` (10%)
2. Use `parentbased_traceidratio` to respect upstream decisions
