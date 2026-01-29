# Deployment Guide

RouteIQ Gateway is designed to be cloud-native and deployment-agnostic. This guide covers deployment using Docker, Docker Compose, and Kubernetes.

## Reproducible Builds

RouteIQ uses lockfile-driven dependency management for deterministic, reproducible builds. This reduces supply-chain risk and ensures consistent behavior across environments.

### Key Features

1. **Lockfile-Driven Installs**: All dependencies are installed via `uv sync --frozen` using `uv.lock`, ensuring exact version pinning.
2. **Pinned Base Images**: Docker base images are pinned to SHA256 digests for supply-chain integrity.
3. **CI Verification**: The CI pipeline verifies lockfile integrity on every PR using `uv lock --check`.
4. **SBOM/Provenance**: Docker builds automatically generate SBOM (Software Bill of Materials) and provenance attestations.

### Updating Dependencies

To update dependencies while maintaining reproducibility:

```bash
# Update all dependencies to latest compatible versions
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package <package-name>

# Verify lockfile is in sync (CI runs this automatically)
uv lock --check
```

### Verifying Build Reproducibility

```bash
# Verify lockfile is up to date before building
uv lock --check

# Build with frozen dependencies (fails if lockfile is stale)
docker build -f docker/Dockerfile -t routeiq-gateway .

# Inspect image SBOMs (available for pushed images)
docker buildx imagetools inspect ghcr.io/baladithyab/litellm-llm-router:latest --format '{{json .SBOM}}'
```

### Base Image Digests

The Dockerfiles pin base images by digest for reproducibility. To update the base image digests:

```bash
# Pull and get the digest of the new image
docker pull ghcr.io/astral-sh/uv:0.9-python3.12-bookworm
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/astral-sh/uv:0.9-python3.12-bookworm

# Update the digest in docker/Dockerfile ARG BUILDER_DIGEST and RUNTIME_DIGEST
```

## Docker

The easiest way to run RouteIQ Gateway is using the official Docker image.

```bash
docker run -p 4000:4000 \
  -e OPENAI_API_KEY="sk-..." \
  ghcr.io/baladithyab/litellm-llm-router:latest
```

## Docker Compose

We provide several Docker Compose configurations for different use cases.

### Standard (Development)
For local development and testing.

```bash
docker compose up -d
```

### High Availability (HA)
Includes Redis for caching/rate-limiting and PostgreSQL for data persistence.

```bash
docker compose -f docker-compose.ha.yml up -d
```

### Observability (OTEL)
Includes Jaeger and Prometheus for tracing and metrics.

```bash
docker compose -f docker-compose.otel.yml up -d
```

## Kubernetes

RouteIQ Gateway is Kubernetes-ready. Below is a blueprint for a production deployment.

### Deployment Blueprint

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: routeiq-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: routeiq-gateway
  template:
    metadata:
      labels:
        app: routeiq-gateway
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
        - name: gateway
          image: ghcr.io/baladithyab/litellm-llm-router:latest
          ports:
            - containerPort: 4000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: routeiq-secrets
                  key: database-url
            - name: REDIS_HOST
              value: "redis-master"
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 4000
            initialDelaySeconds: 5
            periodSeconds: 10
      volumes:
        - name: config-volume
          configMap:
            name: routeiq-config
```

### Helm Chart Values
(Coming soon)

## Configuration Management

RouteIQ Gateway supports multiple configuration sources:

1.  **Local Files**: Mount `config.yaml` to `/app/config/config.yaml`.
2.  **Environment Variables**: Override settings using `LITELLM_*` env vars.
3.  **S3**: Load configuration and models from an S3 bucket.

```bash
# Enable S3 config loading
export CONFIG_SOURCE="s3"
export S3_BUCKET_NAME="my-config-bucket"
```

## Cloud Deployment

For specific cloud provider guides, see:

- [AWS Deployment Guide](architecture/aws-deployment.md)

## Observability Configuration

RouteIQ Gateway supports OpenTelemetry for distributed tracing, metrics, and structured logging.

### Quick Setup

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
  - OTEL_SERVICE_NAME=litellm-gateway
  - OTEL_TRACES_EXPORTER=otlp
  - OTEL_METRICS_EXPORTER=otlp
```

### Production Recommendations

- **Trace Sampling**: In high-traffic environments, enable sampling to control costs:
  ```yaml
  environment:
    - LLMROUTER_OTEL_SAMPLE_RATE=0.1  # Sample 10% of traces
  ```

- **Multiprocess Metrics**: When running with multiple workers (gunicorn/uvicorn), use an OTEL Collector for metric aggregation. See the full guide for configuration details.

For complete observability configuration including sampling options, multiprocess setup, and backend-specific guides, see the [Observability Guide](observability.md).
