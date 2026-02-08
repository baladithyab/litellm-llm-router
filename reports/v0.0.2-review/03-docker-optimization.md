# RouteIQ v0.0.2 Docker & Container Optimization Report

**Date:** 2026-02-07
**Reviewer:** Docker/Container Optimization Specialist
**Scope:** Dockerfile, compose configurations, OTel collector, dependency impact, CI pipeline

---

## Executive Summary

The RouteIQ container setup is well-structured with solid security foundations (non-root user, tini init, pinned digests, multi-stage builds, Trivy scanning in CI). However, the addition of `sentence-transformers` (and its transitive `torch` dependency) in v0.0.2 introduces a **massive image size increase** -- estimated at 2-4 GB on x86_64 due to CUDA libraries bundled with the default PyTorch wheel. This is the single largest optimization opportunity. Secondary issues include `build-essential` left in the runtime image, missing resource limits in compose files, and Nginx configuration gaps for SSE/WebSocket streaming.

**Estimated current image size:** ~4.5-5.5 GB (x86_64 with CUDA torch), ~1.5-2.0 GB (arm64 without CUDA)
**Achievable optimized size:** ~1.5-2.0 GB (x86_64 with CPU-only torch + ONNX backend), ~1.0-1.5 GB (arm64)

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [New Component Impact Assessment](#2-new-component-impact-assessment)
3. [Findings by Category](#3-findings-by-category)
4. [Container Sizing Recommendations](#4-container-sizing-recommendations)
5. [Recommended Dockerfile Improvements](#5-recommended-dockerfile-improvements)
6. [Compose Architecture Improvements](#6-compose-architecture-improvements)
7. [CI/CD Pipeline Analysis](#7-cicd-pipeline-analysis)
8. [Research Findings](#8-research-findings)

---

## 1. Current State Analysis

### 1.1 Dockerfile (Production: `docker/Dockerfile`)

**Strengths:**
- Multi-stage build: builder stage clones LLMRouter, runtime is separate
- Pinned base images via sha256 digests (supply-chain integrity)
- Non-root user (`litellm`, UID 1000)
- `tini` init system for proper signal handling
- BuildKit cache mounts for uv (`--mount=type=cache`)
- `STOPSIGNAL SIGTERM` for graceful shutdown
- `HEALTHCHECK` instruction for orchestrators
- `PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1`
- ReadOnlyRootFilesystem documentation with required volume mounts
- `.dockerignore` is comprehensive (excludes tests, docs, reference/, models, .git)

**Issues Identified:**
- `build-essential` installed in runtime stage (lines 103-109) -- adds ~200MB+ of compilers/headers that are only needed for uvloop compilation
- `src/` is COPY'd twice: once for `uv pip install -e "."` (line 119) and again as app source (line 150), wasting a layer
- Prisma client generation at build time runs as root, then chown -- could be simplified
- The `uv pip install --system /wheels/*.whl || true` silently swallows failures (line 126)
- No `.dockerignore` entry for `reports/` or `ha-gate-report.json` (minor)

### 1.2 Dockerfile (Local: `docker/Dockerfile.local`)

**Strengths:**
- Single-stage for simplicity (appropriate for dev)
- Same base image digest pinning
- Lockfile-driven installs

**Issues:**
- Does NOT install `knn` extra (line 63: `.[db,otel,callbacks]`), so local dev won't test sentence-transformers codepaths
- Runs as root (no USER instruction)
- `git` remains in image after LLMRouter clone (not cleaned up)

### 1.3 Base Image Choice

Current: `ghcr.io/astral-sh/uv:0.9-python3.14-bookworm-slim`

This is a reasonable choice:
- Debian Bookworm slim: ~80MB base, good glibc compatibility
- Includes uv pre-installed (no separate install step)
- Python 3.14 is bleeding edge (released ~2025-10)

**Alpine trade-off:** Alpine uses musl libc, which causes problems with PyTorch, numpy, and many scientific Python packages. **Not recommended** for this project given ML dependencies.

**Distroless trade-off:** Google distroless Python images are ~50MB base, but lack a shell, package manager, and curl. This makes debugging harder and breaks the entrypoint.sh pattern. Feasible only with significant refactoring (move entrypoint logic to Python). The ~30MB savings is negligible compared to the multi-GB torch dependency. **Not recommended for v0.0.2.**

### 1.4 Layer Caching Analysis

The current layer order is:

```
1. apt-get install (system deps)      -- changes rarely
2. COPY pyproject.toml uv.lock        -- changes on dep updates
3. COPY src/                           -- changes on every code change (!)
4. uv pip install -e ".[prod,...]"    -- re-runs when src/ changes
5. Prisma generate + chown
6. COPY src/litellm_llmrouter          -- redundant with step 3
7. COPY entrypoint.sh, config/
```

**Problem:** Step 3 copies `src/` before the pip install. Since `uv pip install -e "."` needs the package source, this is required. However, ANY change to source code invalidates the pip install cache layer even when dependencies haven't changed. This is the classic "cache-busting" problem with editable installs.

**Fix:** Use a two-phase approach: first install dependencies without the project itself, then copy source and install in editable mode. See Section 5.

### 1.5 Health Checks

| File | Health Endpoint | Notes |
|------|----------------|-------|
| `Dockerfile` | `/_health/live` | Correct internal endpoint, unauthenticated |
| `docker-compose.yml` | `/health` | **Mismatch** -- uses LiteLLM's authenticated health endpoint |
| `docker-compose.ha.yml` | `/_health/live` | Correct |
| `docker-compose.local-test.yml` | `/health` with Bearer token | Works but overly complex for health check |

**Issue:** `docker-compose.yml` line 45 uses `/health` which may require auth depending on config, while the Dockerfile uses `/_health/live` which is always unauthenticated. These should be consistent.

### 1.6 Security Assessment

| Check | Status | Notes |
|-------|--------|-------|
| Non-root user | PASS | UID 1000, `/sbin/nologin` shell |
| Pinned base images | PASS | sha256 digests |
| No secrets baked in | PASS | Uses env vars at runtime |
| Read-only FS compatible | PASS | Documented writable paths |
| `STOPSIGNAL SIGTERM` | PASS | Graceful shutdown |
| Trivy scanning in CI | PASS | PR + release workflows |
| SBOM generation | PASS | `sbom: true` in release build |
| Provenance attestation | PASS | `provenance: true` in release build |
| build-essential in runtime | **FAIL** | Unnecessary attack surface |
| `|| true` swallowing errors | **WARN** | Could mask build failures |

### 1.7 Volume Management

| Compose File | Volumes | Concern |
|-------------|---------|---------|
| `docker-compose.ha.yml` | `postgres_data`, `redis_data` | Named volumes, appropriate |
| `docker-compose.local-test.yml` | 5 named volumes + `mcp_node_modules` | `mcp_node_modules` is a node volume that could grow unbounded |
| All compose files | `./config:/app/config:ro` | Good: read-only config mount |
| All compose files | `./models:/app/models:ro` | Good: read-only model mount |

No concerns with volume management. Named volumes are used correctly for persistent state.

---

## 2. New Component Impact Assessment

### 2.1 sentence-transformers + torch (the elephant in the room)

The `knn` optional extra pulls in:
- `sentence-transformers>=5.2.0`
- `scikit-learn>=1.3.0`

Which transitively pulls in:
- `torch 2.10.0` -- **the dominant cost**
- `transformers` (~86MB)
- `huggingface-hub`
- `scipy` (~37MB)
- `tokenizers`

**PyTorch size breakdown (from uv.lock wheel sizes):**

| Platform | Wheel Size | Includes CUDA? |
|----------|-----------|----------------|
| linux/x86_64 | **915 MB** | Yes (CUDA 12, nvidia-*, triton) |
| linux/aarch64 | **146 MB** | No |
| macOS arm64 | **79 MB** | No |

On x86_64 Linux, torch 2.10.0 pulls in these CUDA dependencies (from uv.lock):
- `nvidia-cublas-cu12`
- `nvidia-cuda-cupti-cu12`
- `nvidia-cuda-nvrtc-cu12`
- `nvidia-cuda-runtime-cu12`
- `nvidia-cudnn-cu12`
- `nvidia-cufft-cu12`
- `nvidia-cufile-cu12`
- `nvidia-curand-cu12`
- `nvidia-cusolver-cu12`
- `nvidia-cusparse-cu12`
- `nvidia-cusparselt-cu12`
- `nvidia-nccl-cu12`
- `nvidia-nvjitlink-cu12`
- `nvidia-nvshmem-cu12`
- `nvidia-nvtx-cu12`
- `triton` (~420MB)

**Total CUDA overhead on x86_64: ~3+ GB of unnecessary GPU libraries**

This is the single largest cost in the container image. RouteIQ uses sentence-transformers for KNN routing embeddings with the `all-MiniLM-L6-v2` model, which is a CPU inference task. There is no GPU inference requirement.

### 2.2 all-MiniLM-L6-v2 Model Loading (Runtime Memory)

The `all-MiniLM-L6-v2` model:
- Model file size: ~22MB (safetensors)
- Runtime memory when loaded: ~90-120MB (float32)
- Tokenizer memory: ~5MB
- First-encode warmup: additional ~50MB temporary allocation

**Total memory impact per container:** ~150-200MB for the model alone, on top of baseline Python + LiteLLM memory.

### 2.3 Redis Client Library

The `redis>=5.0.0` Python client adds minimal image size (~1MB). Runtime considerations:
- Default connection pool: 10 connections
- Each connection: ~50KB memory overhead
- Timeout defaults: no timeout (should be set explicitly)
- No connection pooling configuration visible in compose files

### 2.4 Startup Time Impact

With the knn extra, container startup includes:
1. Python interpreter startup + LiteLLM import chain: ~5-8s
2. sentence-transformers model download (first run, if not cached): 30-60s
3. Model loading into memory: ~3-5s
4. Plugin initialization (discovery + validation): ~1-2s
5. Prisma client generation (if DB configured): ~5-10s
6. LiteLLM Router creation + monkey-patch: ~2-3s

**Estimated cold start: 20-30s (model cached), 60-90s (first run, model download)**

The `start_period` in health checks should account for this. Current values:
- Dockerfile: 60s (adequate)
- docker-compose.yml: 10s (**too low** with knn enabled)
- docker-compose.ha.yml: 15s (**too low** with knn enabled)

---

## 3. Findings by Category

### CRITICAL -- Image Size

| # | Finding | Severity | Effort | Impact |
|---|---------|----------|--------|--------|
| C1 | torch ships with CUDA on x86_64, adding ~3GB | Critical | Medium | -3GB image size |
| C2 | build-essential in runtime image (~200MB) | High | Quick Win | -200MB, reduced attack surface |

### HIGH -- Correctness

| # | Finding | Severity | Effort | Impact |
|---|---------|----------|--------|--------|
| H1 | docker-compose.yml health check uses `/health` (may require auth) instead of `/_health/live` | High | Quick Win | Prevent false health failures |
| H2 | `uv pip install --system /wheels/*.whl \|\| true` swallows failures | High | Quick Win | Prevent silent build corruption |
| H3 | start_period too low for knn-enabled containers | High | Quick Win | Prevent restart loops |
| H4 | Dockerfile.local missing knn extra for parity | Medium | Quick Win | Dev/prod parity |

### MEDIUM -- Performance & Resource Management

| # | Finding | Severity | Effort | Impact |
|---|---------|----------|--------|--------|
| M1 | No memory/CPU limits in any compose file | Medium | Quick Win | Prevent OOM, resource contention |
| M2 | Nginx missing SSE/WebSocket support for MCP | Medium | Medium | MCP SSE transport broken via LB |
| M3 | Layer cache invalidation on source changes | Medium | Medium | Faster CI builds |
| M4 | No memory_limiter in OTel collector config | Medium | Quick Win | Prevent OTel OOM |
| M5 | Redis in HA compose has no password auth | Medium | Quick Win | Security hardening |
| M6 | Postgres port exposed to host in HA compose | Medium | Quick Win | Reduce attack surface |

### LOW -- Optimization Opportunities

| # | Finding | Severity | Effort | Impact |
|---|---------|----------|--------|--------|
| L1 | sentence-transformers ONNX backend could replace torch entirely | Low | Large | -2GB+ image, faster inference |
| L2 | Pre-download ML model in Dockerfile | Low | Quick Win | Eliminate first-run download |
| L3 | Duplicate COPY src/ in Dockerfile | Low | Medium | Cleaner layers |
| L4 | Dockerfile.local runs as root | Low | Quick Win | Dev security hygiene |
| L5 | No Compose profiles for optional services | Low | Medium | Better DX |
| L6 | minio image not pinned (uses :latest) | Low | Quick Win | Reproducibility |

---

## 4. Container Sizing Recommendations

### 4.1 Deployment Profiles

| Profile | Description | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|-------------|-------------|-----------|----------------|-------------|
| **Minimal** | Gateway only, no KNN routing | 0.5 CPU | 2 CPU | 512MB | 1GB |
| **Standard** | Gateway + KNN routing + Redis | 1 CPU | 4 CPU | 1GB | 2GB |
| **HA** | Gateway + KNN + DB + Redis + OTel | 2 CPU | 4 CPU | 1.5GB | 3GB |
| **Full Stack** | All services (local-test) | 4 CPU | 8 CPU | 4GB | 8GB |

### 4.2 Per-Service Memory Estimates

| Service | Idle Memory | Peak Memory | Notes |
|---------|-------------|-------------|-------|
| Gateway (no KNN) | ~300MB | ~600MB | LiteLLM + FastAPI + plugins |
| Gateway (with KNN) | ~500MB | ~1.2GB | + sentence-transformers model + embedding computation |
| PostgreSQL | ~50MB | ~256MB | Depends on connection count |
| Redis (256MB limit) | ~10MB | ~256MB | maxmemory configured in HA compose |
| Jaeger all-in-one | ~100MB | ~500MB | Depends on trace volume |
| OTel Collector | ~50MB | ~200MB | Should have memory_limiter |
| Nginx | ~5MB | ~20MB | Minimal |
| MLflow | ~200MB | ~500MB | SQLite backend, MinIO artifacts |
| MinIO | ~100MB | ~512MB | Object storage |

### 4.3 Recommended Resource Limits for Production (per gateway instance)

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 2G
    reservations:
      cpus: '1'
      memory: 1G
```

---

## 5. Recommended Dockerfile Improvements

### 5.1 [C1] Install CPU-only PyTorch (Critical -- saves ~3GB)

The biggest single optimization. Use PyTorch's CPU-only index URL:

```dockerfile
# In the runtime stage, before uv pip install:
# Force CPU-only torch to avoid CUDA libraries on x86_64
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system /wheels/*.whl \
    && cd /app && uv pip install --system -e ".[prod,db,otel,cloud,callbacks,knn]" \
    && rm -rf /wheels
```

Alternatively, use `uv pip install` with `--extra-index-url` and environment constraints. The key is ensuring the CPU-only torch wheel is selected.

**Note:** This requires testing that sentence-transformers works correctly with CPU-only torch. It should, since all-MiniLM-L6-v2 is a CPU inference model.

### 5.2 [C2] Remove build-essential from Runtime (Quick Win -- saves ~200MB)

Move compilation to the builder stage:

```dockerfile
# ==============================================================================
# Stage 1: Builder - compile wheels that need C extensions
# ==============================================================================
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm@${BUILDER_DIGEST} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build LLMRouter wheel
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone ... && uv build --wheel ...

# Pre-compile wheels for packages needing C extensions (uvloop, etc.)
COPY pyproject.toml uv.lock /build/
COPY src/ /build/src/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /build && uv pip install --system -e ".[prod,db,otel,cloud,callbacks,knn]"

# ==============================================================================
# Stage 2: Runtime - no compilers needed
# ==============================================================================
FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-bookworm-slim@${RUNTIME_DIGEST} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tini libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
```

This removes `build-essential`, `gcc`, `g++`, `make`, `dpkg-dev`, and all development headers from the final image.

### 5.3 [H2] Don't Swallow Wheel Install Failures

```dockerfile
# Replace:
#   uv pip install --system /wheels/*.whl || true
# With:
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system /wheels/*.whl \
    && cd /app && uv pip install --system -e ".[prod,db,otel,cloud,callbacks,knn]" \
    && rm -rf /wheels
```

If LLMRouter wheel install fails, the build should fail, not silently continue.

### 5.4 [M3] Improve Layer Caching for Source Changes

Split the editable install into two phases:

```dockerfile
# Phase 1: Install dependencies only (cache-friendly)
COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /app && uv pip install --system --no-deps -e "." 2>/dev/null || true \
    && uv pip install --system ".[prod,db,otel,cloud,callbacks,knn]" --no-install-project

# Phase 2: Copy source and install project (invalidated on code changes)
COPY src/ /app/src/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /app && uv pip install --system --no-deps -e "."
```

This way, dependency installation is cached as long as `pyproject.toml` and `uv.lock` don't change.

### 5.5 [L1] Future: ONNX Backend for sentence-transformers (Large effort, -2GB+)

sentence-transformers v5.2+ natively supports ONNX backend:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2", backend="onnx")
```

This replaces the torch dependency with `onnxruntime` (~24MB vs ~915MB for torch+CUDA on x86_64). The trade-off:
- **Pro:** ~2-3GB image size reduction, potentially faster CPU inference with ONNX optimizations
- **Pro:** Quantized ONNX models (int8) available for even faster inference
- **Con:** Requires code changes in the KNN routing strategy to use ONNX backend
- **Con:** ONNX models need to be pre-exported or downloaded at startup
- **Con:** Some model features may not be available in ONNX mode

**Recommendation:** Implement in v0.0.3 as a configuration option. For v0.0.2, use CPU-only torch.

### 5.6 [L2] Pre-download ML Model in Dockerfile

Eliminate first-run model download latency:

```dockerfile
# After installing dependencies, pre-download the model
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')" \
    && chown -R litellm:litellm /app/.cache
```

This adds ~22MB to the image but eliminates 30-60s first-run download. The model is cached in `/app/.cache/huggingface/` which should be listed in the ReadOnlyRootFilesystem volume mounts.

---

## 6. Compose Architecture Improvements

### 6.1 [M1] Add Resource Limits

None of the compose files set resource limits. Add to all gateway services:

```yaml
services:
  litellm-gateway-1:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

For Redis (already has `--maxmemory 256mb` in HA):
```yaml
  redis:
    deploy:
      resources:
        limits:
          memory: 300M
```

### 6.2 [M2] Fix Nginx for SSE/WebSocket (MCP Streaming)

The current `nginx.conf` has `proxy_buffering off` but lacks WebSocket upgrade support needed for MCP SSE transport (`/mcp/sse`):

```nginx
# Add to the location / block:
location /mcp/ {
    proxy_pass http://litellm_backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $http_host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

    # SSE-specific settings
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 86400s;  # Keep SSE connections open (24h)
    chunked_transfer_encoding on;

    # Disable request buffering for streaming
    proxy_request_buffering off;
}
```

### 6.3 [M5] Redis Authentication in HA

The Redis instance in `docker-compose.ha.yml` has no password. Add:

```yaml
redis:
    command: >
      redis-server
        --appendonly yes
        --maxmemory 256mb
        --maxmemory-policy allkeys-lru
        --requirepass ${REDIS_PASSWORD:-redis_password}
```

And update gateway environment:
```yaml
- REDIS_PASSWORD=${REDIS_PASSWORD:-redis_password}
```

### 6.4 [M6] Don't Expose Postgres to Host in HA

In `docker-compose.ha.yml`, Postgres exposes port 5432 to the host (line 20). This is noted as "Exposed for HA integration tests" but is a security risk in production. Consider removing or making conditional:

```yaml
ports:
  - "${POSTGRES_EXPOSE_PORT:-}:5432"  # Only exposed if POSTGRES_EXPOSE_PORT is set
```

### 6.5 [M4] Add memory_limiter to OTel Collector Config

The `config/otel-collector-config.yaml` lacks a memory limiter processor, risking OOM:

```yaml
processors:
  memory_limiter:
    check_interval: 5s
    limit_mib: 400
    spike_limit_mib: 100

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, resource]  # memory_limiter FIRST
      exporters: [awsxray]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch/metrics, resource]
      exporters: [awsemf]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch/logs, resource]
      exporters: [awscloudwatchlogs]
```

### 6.6 Service Dependency & Startup Ordering

Current ordering is correct:
- Postgres + Redis start first (with health checks)
- Gateway depends on both with `condition: service_healthy`
- Nginx depends on both gateways

**Potential issue:** In `docker-compose.local-test.yml`, the gateway depends on `mlflow: condition: service_healthy`, but MLflow itself depends on MinIO. If MinIO is slow to start, there's a cascading delay. MLflow's `start_period: 30s` helps but consider making MLflow optional via Compose profiles.

### 6.7 Network Segmentation

Current: All services share a single flat network (`litellm-network` or `litellm-test-network`).

**Recommendation for production HA:**
```yaml
networks:
  frontend:     # Nginx <-> Internet
    driver: bridge
  backend:      # Nginx <-> Gateway instances
    driver: bridge
    internal: true
  datastore:    # Gateway <-> Postgres/Redis
    driver: bridge
    internal: true
```

This prevents direct external access to Postgres/Redis even if port mappings are accidentally added.

---

## 7. CI/CD Pipeline Analysis

### 7.1 `docker-build.yml` Workflow

**Strengths:**
- Separate PR (single-arch build+test) and Release (multi-arch) workflows
- Registry cache with zstd compression and per-arch refs
- Trivy scanning for both PR and release images
- SARIF upload to GitHub Security tab
- Non-root container verification in smoke test
- SBOM and provenance attestation on release
- Fork PR detection (cache read-only for forks)
- Disk space cleanup step (the image needs it at ~4GB)

**Issues:**
- The "Free disk space" step (lines 56-65, 323-339) indicates the image is close to GitHub Actions runner disk limits. With CPU-only torch, this would be less of a concern.
- Trivy scan timeout set to 30min (line 501) with note "Increase timeout for large images with CUDA dependencies" -- this is a symptom of the torch+CUDA bloat.
- Smoke test loads the image via multiple fallback methods (lines 146-163) which is fragile. The OCI archive format conversion could be simplified.

### 7.2 Build Time Estimates

| Phase | Current (with CUDA torch) | Optimized (CPU torch) |
|-------|--------------------------|----------------------|
| Base image pull | ~30s | ~30s |
| apt-get install | ~20s | ~15s (no build-essential) |
| LLMRouter wheel build | ~30s | ~30s |
| Python deps install | ~5-8min | ~2-3min |
| Prisma generate | ~15s | ~15s |
| Source copy + finalize | ~5s | ~5s |
| Image push | ~3-5min | ~1-2min |
| Trivy scan | ~10-15min | ~3-5min |
| **Total** | **~15-25min** | **~8-12min** |

---

## 8. Research Findings

### 8.1 PyTorch CPU-Only Installation (Critical for Image Size)

PyTorch offers CPU-only wheels via a dedicated index URL that exclude CUDA/nvidia libraries:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For torch 2.10.0 on x86_64:
- Default wheel (with CUDA): **915 MB** + nvidia deps (~2GB) + triton (~420MB) = **~3.3GB total**
- CPU-only wheel: **~150-200MB** (estimated based on prior versions)

This is by far the highest-impact change available.

**Sources:**
- https://pytorch.org/get-started/locally/
- https://discuss.pytorch.org/t/index-url-to-install-pytorch/198253
- https://stackoverflow.com/questions/78105348/how-to-reduce-python-docker-image-size
- https://shekhargulati.com/2025/02/05/reducing-size-of-docling-pytorch-docker-image/

### 8.2 sentence-transformers ONNX Backend

sentence-transformers v5.x+ supports ONNX as a backend, eliminating the torch dependency entirely:

```python
model = SentenceTransformer("all-MiniLM-L6-v2", backend="onnx")
```

Benefits:
- `onnxruntime` is ~24MB vs torch's 150-915MB
- CPU inference can be faster with ONNX optimizations (O3 level)
- Quantized models (int8) available for further speedup
- Supported natively, no custom code needed

**Source:** https://sbert.net/docs/sentence_transformer/usage/efficiency.html

### 8.3 OTel Collector Resource Tuning

The memory_limiter processor is considered mandatory for production OTel Collector deployments:

- Should be the **first** processor in every pipeline
- Recommended settings: `limit_mib` at 80% of container memory limit
- `spike_limit_mib` at 20% of `limit_mib`
- `check_interval: 5s` is standard

For a gateway processing ~100 req/s:
- Collector CPU: 0.2-0.5 CPU cores
- Collector Memory: 256-512MB with memory_limiter at 400MB
- Batch size: 50-200 for traces, 100+ for metrics

**Sources:**
- https://www.ibm.com/docs/en/zapmc/7.1.0?topic=tuning-opentelemetry-collector-resource
- https://www.dash0.com/guides/opentelemetry-memory-limiter-processor
- https://last9.io/guides/opentelemetry/deploying-opentelemetry-at-scale-production-patterns-that-work/

### 8.4 Distroless Python Images

Google's distroless Python images (gcr.io/distroless/python3) are ~50MB but:
- No shell (breaks entrypoint.sh pattern)
- No package manager (can't install curl for health checks)
- No debugging tools
- Python version lags behind upstream
- Savings are negligible (~30MB) compared to torch dependency

**Verdict:** Not worth the effort for RouteIQ. The uv slim base is a better fit.

**Source:** https://www.joshkasuboski.com/posts/distroless-python-uv/

### 8.5 Container Security Scanning

The project already uses Trivy in CI, which is the industry standard. Additional options:

| Tool | Type | Integration | Notes |
|------|------|------------|-------|
| Trivy (current) | Vulnerability + SBOM | GitHub Actions | Already integrated |
| Grype | Vulnerability | CLI/CI | Anchore's scanner, good complement |
| Syft | SBOM generation | CLI/CI | Pairs with Grype |
| Snyk Container | Vulnerability + base image advice | GitHub Actions | More opinionated recommendations |
| Docker Scout | Vulnerability + supply chain | Docker Hub | Native Docker integration |

**Recommendation:** Current Trivy setup is sufficient. Consider adding Grype as a second opinion in CI for critical releases.

### 8.6 Redis Configuration Best Practices for AI Gateway Caching

For LiteLLM's Redis caching:
- `maxmemory-policy allkeys-lru` (already set in HA compose) is correct for cache workloads
- Consider `volatile-ttl` if mixing cache with persistent data
- Set `timeout 300` to close idle connections
- Enable `tcp-keepalive 60` for connection health
- For semantic caching with embeddings, consider dedicated Redis instance with higher memory

**Source:** https://redis.io/tutorials/howtos/solutions/microservices/api-gateway-caching/

---

## Appendix A: Estimated Image Size Breakdown

### Current (x86_64, with default torch)

| Component | Estimated Size |
|-----------|---------------|
| Base image (bookworm-slim + uv + Python 3.14) | ~250MB |
| apt packages (build-essential, curl, tini, etc.) | ~250MB |
| torch 2.10.0 (default x86_64 wheel) | ~915MB |
| nvidia CUDA libraries (transitive) | ~2,000MB |
| triton (transitive) | ~420MB |
| LiteLLM + dependencies | ~300MB |
| sentence-transformers + transformers + scipy | ~200MB |
| OTel instrumentation | ~50MB |
| Cloud SDKs (boto3, google-cloud, azure) | ~100MB |
| Prisma client | ~50MB |
| Application source | ~5MB |
| **Estimated Total** | **~4.5-5.5GB** |

### Optimized (x86_64, CPU-only torch)

| Component | Estimated Size |
|-----------|---------------|
| Base image (bookworm-slim + uv + Python 3.14) | ~250MB |
| apt packages (curl, tini ONLY -- no build-essential) | ~50MB |
| torch 2.10.0 (CPU-only wheel) | ~200MB |
| LiteLLM + dependencies | ~300MB |
| sentence-transformers + transformers + scipy | ~200MB |
| OTel instrumentation | ~50MB |
| Cloud SDKs (boto3, google-cloud, azure) | ~100MB |
| Prisma client | ~50MB |
| Application source | ~5MB |
| **Estimated Total** | **~1.2-1.5GB** |

### Future (ONNX backend, no torch)

| Component | Estimated Size |
|-----------|---------------|
| Base image | ~250MB |
| apt packages (minimal) | ~50MB |
| onnxruntime (replaces torch) | ~25MB |
| LiteLLM + dependencies | ~300MB |
| sentence-transformers + transformers | ~150MB |
| OTel + Cloud SDKs + Prisma | ~200MB |
| Application source | ~5MB |
| **Estimated Total** | **~1.0GB** |

---

## Appendix B: Priority Action Items

### Immediate (v0.0.2)

1. **[C1] Install CPU-only torch** -- `--index-url https://download.pytorch.org/whl/cpu`
2. **[C2] Remove build-essential from runtime** -- move compilation to builder stage
3. **[H1] Fix health check endpoints** -- standardize on `/_health/live`
4. **[H2] Remove `|| true` from wheel install** -- let builds fail on errors
5. **[H3] Increase start_period** -- 60s minimum in all compose files
6. **[M1] Add resource limits** -- to all compose services

### Short-term (v0.0.3)

7. **[M2] Fix Nginx for MCP SSE** -- WebSocket upgrade + long timeouts
8. **[M4] Add memory_limiter to OTel config**
9. **[M5] Add Redis password auth in HA**
10. **[L2] Pre-download ML model in Dockerfile**
11. **[L1] Evaluate ONNX backend** -- prototype and benchmark

### Medium-term

12. **Network segmentation** in HA compose (frontend/backend/datastore)
13. **Compose profiles** for optional services in local-test
14. **Layer caching optimization** with two-phase install

---

*Report generated from analysis of: `docker/Dockerfile`, `docker/Dockerfile.local`, `docker-compose.yml`, `docker-compose.ha.yml`, `docker-compose.otel.yml`, `docker-compose.local-test.yml`, `config/otel-collector-config.yaml`, `config/nginx.conf`, `pyproject.toml`, `uv.lock`, `.dockerignore`, `docker/entrypoint.sh`, `docker/entrypoint.local.sh`, `.github/workflows/docker-build.yml`*
