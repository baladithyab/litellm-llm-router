# OpenTelemetry Integration Guide

## Overview

This project uses OpenTelemetry for unified observability:
- **Traces**: Distributed tracing for request flow
- **Logs**: Structured logging with trace correlation
- **Metrics**: Prometheus-compatible metrics

## Dependencies

```python
# requirements.txt
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
opentelemetry-instrumentation-fastapi>=0.41b0
opentelemetry-instrumentation-httpx>=0.41b0
opentelemetry-instrumentation-redis>=0.41b0
```

## Configuration

### Environment Variables

```bash
# OTLP Endpoint
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317

# Service identification
OTEL_SERVICE_NAME=litellm-gateway
OTEL_SERVICE_VERSION=1.0.0

# Resource attributes
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=production,service.namespace=ai-gateway

# Sampling (1.0 = 100%)
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=1.0
```

### YAML Config

```yaml
litellm_settings:
  success_callback: ["prometheus", "otel"]
  otel_config:
    endpoint: "http://otel-collector:4317"
    service_name: "litellm-gateway"
    traces_exporter: "otlp"
    logs_exporter: "otlp"
    metrics_exporter: "prometheus"
```

## Initialization

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

def init_telemetry(service_name: str, otlp_endpoint: str):
    """Initialize OpenTelemetry with OTLP exporters."""
    resource = Resource.create({SERVICE_NAME: service_name})
    
    # Traces
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
    )
    trace.set_tracer_provider(trace_provider)
    
    return trace.get_tracer(__name__)
```

## Tracing Patterns

### Span for Routing Decision

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def route_request(query: str, model_list: list[str]) -> str:
    with tracer.start_as_current_span("llm.routing.decision") as span:
        span.set_attribute("llm.routing.strategy", self.strategy_name)
        span.set_attribute("llm.routing.model_count", len(model_list))
        
        selected_model = self.router.route(query)
        
        span.set_attribute("llm.routing.selected_model", selected_model)
        return selected_model
```

### Span for LLM Call

```python
async def call_llm(model: str, messages: list) -> dict:
    with tracer.start_as_current_span("gen_ai.request") as span:
        span.set_attribute("gen_ai.system", "litellm")
        span.set_attribute("gen_ai.request.model", model)
        
        response = await litellm.acompletion(model=model, messages=messages)
        
        span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
        span.set_attribute("gen_ai.response.model", response.model)
        
        return response
```

### Span for Cache Operations

```python
def check_cache(cache_key: str) -> tuple[bool, Any]:
    with tracer.start_as_current_span("cache.lookup") as span:
        span.set_attribute("cache.key", cache_key[:50])  # Truncate for privacy
        
        result = redis_client.get(cache_key)
        hit = result is not None
        
        span.set_attribute("cache.hit", hit)
        return hit, result
```

## Logging with Trace Correlation

### Setup Logging

```python
import logging
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# Instrument logging to add trace context
LoggingInstrumentor().instrument(set_logging_format=True)

# Configure logger
logging.basicConfig(
    format='%(asctime)s %(levelname)s [%(name)s] [trace_id=%(otelTraceId)s span_id=%(otelSpanId)s] %(message)s',
    level=logging.INFO
)
```

### Log with Context

```python
logger = logging.getLogger(__name__)

def process_request(request_id: str, model: str):
    # trace_id and span_id are automatically added
    logger.info(
        "Processing request",
        extra={
            "request_id": request_id,
            "model": model,
            "event": "request.start"
        }
    )
```

### Structured Logging Output

```json
{
  "timestamp": "2025-01-13T10:30:00Z",
  "level": "INFO",
  "message": "Processing request",
  "trace_id": "abc123def456",
  "span_id": "789xyz",
  "request_id": "req-001",
  "model": "gpt-4",
  "event": "request.start"
}
```

## Semantic Conventions

Use OpenTelemetry semantic conventions for consistency:

### HTTP Spans
- `http.method`: GET, POST, etc.
- `http.status_code`: 200, 400, 500, etc.
- `http.route`: /v1/chat/completions
- `http.url`: Full URL (sanitized)

### LLM/GenAI Spans
- `gen_ai.system`: litellm, openai, anthropic
- `gen_ai.request.model`: gpt-4, claude-3-opus
- `gen_ai.usage.input_tokens`: Token count
- `gen_ai.usage.output_tokens`: Token count
- `gen_ai.response.model`: Actual model used

### Custom Spans
- `llm.routing.strategy`: llmrouter-knn
- `llm.routing.selected_model`: Model chosen
- `llm.routing.latency_ms`: Routing decision time

## Docker Compose with OTEL Collector

```yaml
# docker-compose.otel.yml
services:
  gateway:
    build: .
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      - OTEL_SERVICE_NAME=litellm-gateway
    depends_on:
      - otel-collector

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    volumes:
      - ./config/otel-collector-config.yaml:/etc/otelcol-contrib/config.yaml
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
      - "8889:8889"   # Prometheus metrics

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
```

## OTEL Collector Config

```yaml
# config/otel-collector-config.yaml
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
    send_batch_size: 1024

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  prometheus:
    endpoint: "0.0.0.0:8889"
  logging:
    loglevel: debug

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger, logging]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [logging]
```

## AWS X-Ray Integration

For CloudWatch X-Ray, use the ADOT (AWS Distro for OpenTelemetry) collector:

```yaml
# ECS Task Definition
{
  "containerDefinitions": [
    {
      "name": "aws-otel-collector",
      "image": "amazon/aws-otel-collector:latest",
      "essential": true,
      "environment": [
        {"name": "AWS_REGION", "value": "us-east-1"}
      ]
    }
  ]
}
```

Set endpoint to ADOT sidecar:
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```
