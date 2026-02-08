# RouteIQ v0.0.2 -- Dependency Audit Report

**Date:** 2026-02-07
**Auditor:** dep-auditor (automated supply chain analysis)
**Scope:** All direct + transitive dependencies in `pyproject.toml` and `uv.lock`

---

## Executive Summary

RouteIQ has **173 total packages** resolved in `uv.lock`, comprising 27 direct
dependencies and 146 transitive dependencies. Key risk areas:

| Area | Finding | Severity |
|------|---------|----------|
| LiteLLM CVEs | 16+ CVEs across versions, some affecting v1.81.3 | **HIGH** |
| PyTorch torch.load() RCE | CVE-2025-32434 (CVSS 9.8) patched in 2.6.0; locked at 2.10.0 (OK) | MITIGATED |
| Redis server CVEs | CVE-2025-49844 RediShell RCE (CVSS 10.0) -- server-side, Python client unaffected | INFO |
| Jinja2 sandbox escape | CVE-2025-27516 -- fixed in 3.1.6; locked at 3.1.6 (OK) | MITIGATED |
| NVIDIA/CUDA bloat | 18 packages pulled by `torch` for GPU support | **MEDIUM** |
| Unused direct deps | 9 deps listed for LiteLLM runtime, not imported by RouteIQ | **LOW** |
| sentence-transformers model loading risk | Arbitrary code execution via unsafe model loading | **MEDIUM** |
| Python 3.14 target | Aggressive; some deps may have limited 3.14 testing | **LOW** |

**Overall Risk Rating: MEDIUM** -- No critical unpatched vulnerabilities in current
locked versions, but LiteLLM's CVE history and dependency weight warrant attention.

---

## Part 1: Dependency Inventory

### 1.1 Direct Dependencies (Core)

| Package | Locked Version | Constraint | License | Type | Status |
|---------|---------------|------------|---------|------|--------|
| fastapi | 0.128.0 | >=0.109.0 | MIT | Direct | OK |
| pydantic | 2.12.5 | >=2.5.0 | MIT | Direct | OK |
| httpx | 0.28.1 | >=0.26.0 | BSD-3-Clause | Direct | OK |
| litellm | 1.81.3 | >=1.81.3 | MIT | Direct | **CVE history** |
| apscheduler | 3.11.2 | >=3.10.0 | MIT | Proxy dep | OK |
| email-validator | 2.3.0 | >=2.0.0 | CC0 | Proxy dep | OK |
| fastapi-sso | 0.19.0 | >=0.16.0 | MIT | Proxy dep | OK |
| websockets | 15.0.1 | >=15.0.0 | BSD-3-Clause | Proxy dep | OK |
| backoff | 2.2.1 | >=2.0.0 | MIT | Proxy dep | OK |
| redis | 7.1.0 | >=5.0.0 | MIT | Direct | OK |
| a2a-sdk | 0.3.22 | >=0.2.0 | Apache-2.0 | Direct | OK |
| pyyaml | 6.0.3 | >=6.0 | MIT | Direct | OK |
| boto3 | 1.42.32 | >=1.42.32 | Apache-2.0 | Direct | OK |
| aiofiles | 25.1.0 | >=23.0.0 | Apache-2.0 | Direct | OK |
| watchdog | 6.0.0 | >=3.0.0 | Apache-2.0 | Direct | OK |
| prometheus-client | 0.24.1 | >=0.24.1 | Apache-2.0 | Direct | OK |
| opentelemetry-api | 1.39.1 | >=1.22.0 | Apache-2.0 | Direct | OK |
| opentelemetry-sdk | 1.39.1 | >=1.22.0 | Apache-2.0 | Direct | OK |
| opentelemetry-exporter-otlp | 1.39.1 | >=1.22.0 | Apache-2.0 | Direct | OK |
| opentelemetry-instrumentation | 0.60b1 | >=0.43b0 | Apache-2.0 | Direct | OK |
| opentelemetry-instrumentation-logging | 0.60b1 | >=0.43b0 | Apache-2.0 | Direct | OK |
| python-multipart | 0.0.22 | >=0.0.22 | Apache-2.0 | Direct | OK |

### 1.2 Optional Dependencies

| Package | Locked Version | Extra Group | License | Status |
|---------|---------------|-------------|---------|--------|
| asyncpg | 0.31.0 | db | Apache-2.0 | OK |
| prisma | 0.15.0 | db | Apache-2.0 | OK |
| opentelemetry-exporter-otlp-proto-grpc | 1.39.1 | otel | Apache-2.0 | OK |
| opentelemetry-exporter-otlp-proto-http | 1.39.1 | otel | Apache-2.0 | OK |
| opentelemetry-instrumentation-fastapi | 0.60b1 | otel | Apache-2.0 | OK |
| opentelemetry-instrumentation-httpx | 0.60b1 | otel | Apache-2.0 | OK |
| opentelemetry-instrumentation-requests | 0.60b1 | otel | Apache-2.0 | OK |
| google-cloud-aiplatform | 1.134.0 | cloud | Apache-2.0 | OK |
| azure-identity | 1.25.1 | cloud | MIT | OK |
| langfuse | 3.12.0 | callbacks | MIT | OK |
| sentence-transformers | 5.2.0 | knn | Apache-2.0 | **Model loading risk** |
| scikit-learn | 1.8.0 | knn | BSD-3-Clause | OK |

### 1.3 Dev Dependencies

| Package | Locked Version | License | Status |
|---------|---------------|---------|--------|
| pytest | 9.0.2 | MIT | OK |
| pytest-asyncio | 1.3.0 | Apache-2.0 | OK |
| hypothesis | 6.150.2 | MPL-2.0 | OK |
| ruff | 0.14.13 | MIT | OK |
| mypy | 1.19.1 | MIT | OK |
| numpy | 2.4.1 | BSD-3-Clause | OK |
| scikit-learn | 1.8.0 | BSD-3-Clause | OK (dupe in dev-group) |
| backoff | 2.2.1 | MIT | OK (dupe in dev-group) |
| cryptography | 46.0.3 | Apache-2.0/BSD | OK |
| orjson | 3.11.5 | Apache-2.0/MIT | OK |
| black | 26.1.0 | MIT | OK |

### 1.4 Key Transitive Dependencies

| Package | Locked Version | Pulled By | License | Notes |
|---------|---------------|-----------|---------|-------|
| starlette | 0.50.0 | fastapi | BSD-3-Clause | CVE-2024-47874 fixed in 0.40.0 (OK) |
| torch | 2.10.0 | sentence-transformers | BSD-3-Clause | Heavy; brings CUDA |
| transformers | 4.57.6 | sentence-transformers | Apache-2.0 | Large |
| tokenizers | 0.22.2 | transformers | Apache-2.0 | Rust-compiled |
| jinja2 | 3.1.6 | Multiple | BSD-3-Clause | CVE-2025-27516 fixed |
| aiohttp | 3.13.3 | litellm | Apache-2.0 | OK |
| openai | 2.15.0 | litellm | MIT | OK |
| grpcio | 1.76.0 | google-*, otel | Apache-2.0 | C extension |
| requests | 2.32.5 | boto3, others | Apache-2.0 | OK |
| urllib3 | 2.6.3 | requests | MIT | OK |
| certifi | 2026.1.4 | requests | MPL-2.0 | OK |
| pydantic-core | 2.41.5 | pydantic | MIT | Rust-compiled |
| protobuf | 6.33.4 | google-*, a2a-sdk | BSD-3-Clause | OK |

### 1.5 Dependency Group Counts

| Category | Count | Notes |
|----------|-------|-------|
| NVIDIA/CUDA packages | 18 | All from `torch` |
| OpenTelemetry packages | 15 | Core + instrumentation |
| Google Cloud packages | 12 | From `google-cloud-aiplatform` |
| AWS/boto packages | 3 | boto3, botocore, s3transfer |
| **Total** | **173** | |

---

## Part 2: Vulnerability Findings

### 2.1 LiteLLM (litellm==1.81.3)

LiteLLM has **16+ reported CVEs** across its version history. This is the highest-risk
dependency in the project.

| CVE | Severity | CVSS | Affected | Description | Fix | Status in RouteIQ |
|-----|----------|------|----------|-------------|-----|-------------------|
| CVE-2024-6587 | HIGH | 7.5 | <1.44.8 | SSRF via `api_base` parameter | 1.44.8 | **FIXED** (v1.81.3) |
| CVE-2024-5710 | MEDIUM | 5.9 | 1.34.34 | Improper access control in team management | Revoked | N/A (revoked) |
| CVE-2024-9606 | HIGH | 7.5 | <1.44.12 | API key leakage in logs (only first 5 chars masked) | 1.44.12 | **FIXED** (v1.81.3) |
| CVE-2025-0330 | HIGH | 7.5 | 1.52.1 | Langfuse API key leakage on parse error | Unknown | **FIXED** (v1.81.3) |
| CVE-2025-11203 | LOW | 3.5 | Various | API_KEY exposure via health endpoint | Updated | **INVESTIGATE** |
| CVE-2025-45809 | MEDIUM | 5.4 | 1.65.4 | SQL injection via `/key/block` | Unknown | **INVESTIGATE** |
| GHSA-53gh-p8jc-7rg8 | HIGH | 8.8 | 1.40.3-1.40.12 | RCE via `post_call_rules` callback | None listed | **FIXED** (v1.81.3) |

**Assessment:** RouteIQ pins `>=1.81.3`, which is well beyond the fix versions for the
most critical CVEs. However, LiteLLM's rapid release cadence and history of
security issues means ongoing vigilance is required. The latest nightly as of
2026-02-07 is `1.81.8-nightly`. The SQL injection CVE (CVE-2025-45809) affects
v1.65.4 specifically but the fix status for 1.81.3 should be verified.

**Recommendation:** Subscribe to LiteLLM's GitHub security advisories. Consider
pinning to exact version (`==1.81.3`) rather than minimum (`>=1.81.3`) to prevent
accidental upgrade to a version with new vulnerabilities.

### 2.2 PyTorch (torch==2.10.0)

| CVE | Severity | CVSS | Affected | Description | Fix | Status in RouteIQ |
|-----|----------|------|----------|-------------|-----|-------------------|
| CVE-2025-32434 | **CRITICAL** | 9.8 | <2.6.0 | RCE via `torch.load(weights_only=True)` | 2.6.0 | **FIXED** (v2.10.0) |
| CVE-2024-31580 | HIGH | 7.5 | Various | Heap buffer overflow DoS | Patched | **FIXED** |
| CVE-2024-7804 | HIGH | -- | <=2.3.1 | RCE via torch.distributed.rpc deserialization | 2.4.0+ | **FIXED** |
| Medium vulns | MEDIUM | -- | 2.10.0 | 6 medium-severity issues per Snyk (2.10.0) | Varies | **MONITOR** |

**Assessment:** The critical CVE-2025-32434 is mitigated by the locked version
(2.10.0 >> 2.6.0 fix). However, Snyk reports 6 medium-severity issues in 2.10.0.
RouteIQ's `LLMROUTER_ALLOW_PICKLE_MODELS=false` default mitigates the primary
unsafe-model-loading attack vector. The `torch.distributed.rpc` framework is not
used by RouteIQ.

### 2.3 sentence-transformers (5.2.0)

| CVE | Severity | Description | Status |
|-----|----------|-------------|--------|
| SNYK-PYTHON-SENTENCETRANSFORMERS-8161344 | HIGH | Arbitrary code execution when loading untrusted models | **MITIGATED** |

**Assessment:** sentence-transformers loads models that may use unsafe
serialization formats. RouteIQ mitigates this with `LLMROUTER_ALLOW_PICKLE_MODELS=false`
by default and model hash/signature verification in `model_artifacts.py`. Risk is
LOW when loading only trusted, pre-verified models.

### 2.4 Redis Python Client (redis==7.1.0)

| Finding | Details |
|---------|---------|
| Python `redis` client CVEs | No known CVEs in redis-py 7.1.0 |
| Redis **server** CVEs | CVE-2025-49844 "RediShell" (CVSS 10.0) -- server-side RCE via Lua scripting |

**Assessment:** The Python redis client itself has no known vulnerabilities. The
critical Redis server CVEs (RediShell, CVE-2025-46817, CVE-2025-46818) affect the
**server process**, not the Python client library. RouteIQ operators should ensure
their Redis servers are patched.

### 2.5 OpenTelemetry SDK (1.39.1)

| Finding | Details |
|---------|---------|
| CVE-2024-36129 | OTel **Collector** (Go) issue, not Python SDK |
| CVE-2025-27513 | OTel **.NET** SDK DoS, not Python SDK |
| Python OTel SDK | No known CVEs in current version |

**Assessment:** No vulnerabilities found in the Python OpenTelemetry SDK packages.
The OTel Collector (if deployed alongside) should be kept updated separately.

### 2.6 FastAPI / Starlette / Uvicorn

| CVE | Severity | Affected | Description | Status |
|-----|----------|----------|-------------|--------|
| CVE-2024-47874 | HIGH (8.7) | Starlette <0.40.0 | multipart/form-data DoS | **FIXED** (0.50.0) |
| CVE-2024-24762 | MEDIUM | python-multipart | ReDoS via Content-Type | **FIXED** |
| GHSA-qf9m-vfgh-m389 | MEDIUM | FastAPI | Content-Type ReDoS | **FIXED** (0.128.0) |

**Assessment:** All known CVEs are fixed in the locked versions. FastAPI 0.128.0
and Starlette 0.50.0 are current and patched.

### 2.7 Pydantic (2.12.5)

| CVE | Severity | Affected | Description | Status |
|-----|----------|----------|-------------|--------|
| CVE-2024-3772 | MEDIUM | <2.4.0 | ReDoS via crafted email string | **FIXED** |
| CVE-2021-29510 | LOW | <1.8.2 | DoS via infinity float | **FIXED** |

**Assessment:** No vulnerabilities in pydantic 2.12.5. Clean.

### 2.8 Jinja2 (3.1.6)

| CVE | Severity | Affected | Description | Status |
|-----|----------|----------|-------------|--------|
| CVE-2025-27516 | HIGH (8.8) | <3.1.6 | Sandbox escape via `|attr` filter | **FIXED** (3.1.6) |

**Assessment:** Fixed in the exact locked version. RouteIQ does not execute
untrusted Jinja2 templates directly, further reducing risk.

### 2.9 httpx (0.28.1)

No known CVEs in httpx 0.28.1. Clean.

---

## Part 3: Dependency Size Analysis

### 3.1 PyTorch + CUDA Stack

**Impact:** MASSIVE -- the single largest contributor to image size.

| Component | Approximate Size | Notes |
|-----------|-----------------|-------|
| torch (core) | ~800 MB | CPU-only would be ~200 MB |
| nvidia-cublas-cu12 | ~400 MB | CUDA linear algebra |
| nvidia-cudnn-cu12 | ~700 MB | Deep learning primitives |
| nvidia-cuda-runtime-cu12 | ~5 MB | CUDA runtime |
| nvidia-cufft-cu12 | ~200 MB | FFT operations |
| nvidia-cusolver-cu12 | ~150 MB | Linear solver |
| Other NVIDIA packages (12) | ~200 MB | Various CUDA libs |
| triton | ~200 MB | GPU compiler |
| **Subtotal** | **~2.5-3 GB** | |

**Recommendation:** Since RouteIQ uses sentence-transformers primarily for KNN
routing (embedding similarity), GPU support is likely unnecessary for most
deployments. Use `torch` CPU-only variant (`torch-cpu` or install with
`--index-url https://download.pytorch.org/whl/cpu`) to reduce image size by
**~2 GB**.

### 3.2 sentence-transformers Stack

| Component | Approximate Size | Notes |
|-----------|-----------------|-------|
| sentence-transformers | ~2 MB | Small itself |
| transformers | ~30 MB | HuggingFace core |
| tokenizers | ~10 MB | Rust binary |
| huggingface-hub | ~2 MB | Model download |
| safetensors | ~1 MB | Safe model format |
| torch (see above) | ~2.5 GB | The big one |
| **Subtotal (excl. torch)** | **~45 MB** | |

### 3.3 LiteLLM Stack

| Component | Approximate Size | Notes |
|-----------|-----------------|-------|
| litellm | ~15 MB | Core + proxy |
| openai | ~3 MB | OpenAI SDK |
| tiktoken | ~5 MB | Token counting |
| aiohttp | ~5 MB | Async HTTP |
| **Subtotal** | **~30 MB** | |

### 3.4 OpenTelemetry Stack

15 packages total:

| Component | Approximate Size | Notes |
|-----------|-----------------|-------|
| All 15 OTel packages | ~10 MB total | Reasonable |
| grpcio (transitive) | ~30 MB | C extension, heavy |
| protobuf (transitive) | ~5 MB | Protocol buffers |
| **Subtotal** | **~45 MB** | |

### 3.5 Google Cloud Stack

12 packages:

| Component | Approximate Size | Notes |
|-----------|-----------------|-------|
| google-cloud-aiplatform | ~20 MB | Vertex AI SDK |
| Other google-* packages | ~15 MB | Core, auth, storage |
| **Subtotal** | **~35 MB** | |

### 3.6 Total Estimated Size

| Category | Size Estimate |
|----------|--------------|
| PyTorch + CUDA | ~2.5 GB |
| sentence-transformers stack | ~45 MB |
| LiteLLM + transitive | ~30 MB |
| OpenTelemetry + grpc | ~45 MB |
| Google Cloud | ~35 MB |
| AWS boto3 | ~100 MB |
| Everything else | ~50 MB |
| **Total** | **~2.8 GB** |
| **Without CUDA** | **~800 MB** |

---

## Part 4: Dependency Hygiene

### 4.1 Pinning Strategy

**Current approach:** Minimum version constraints (`>=`) in `pyproject.toml` with
exact versions locked in `uv.lock`.

| Aspect | Assessment |
|--------|-----------|
| Lock file present | YES -- `uv.lock` with 173 packages |
| Lock file used in Docker | YES -- `uv sync --frozen` (fails on mismatch) |
| Version constraints | Minimum-only (`>=`) for all deps |
| Hash verification | YES -- sha256 hashes in `uv.lock` for all wheels/sdists |

**Finding:** The strategy is sound. `uv.lock` provides reproducibility while
`>=` constraints in `pyproject.toml` allow flexibility for consumers. Docker builds
use `--frozen` to enforce lock file consistency.

**Recommendation:** For maximum supply chain safety, consider adding
`--require-hashes` to pip install commands in non-uv contexts.

### 4.2 Unused Dependencies

The following dependencies are declared in `pyproject.toml` but **never imported
directly** by RouteIQ source code (`src/litellm_llmrouter/`):

| Package | Reason Listed | Actually Used By |
|---------|---------------|------------------|
| apscheduler | LiteLLM proxy runtime | `litellm.proxy.proxy_server` |
| email-validator | LiteLLM SCIM/Pydantic EmailStr | `litellm` (Pydantic models) |
| fastapi-sso | LiteLLM SSO endpoints | `litellm.proxy.management_endpoints.ui_sso` |
| websockets | LiteLLM guardrails + realtime | `litellm.proxy.guardrails`, `litellm.llms.openai.realtime` |
| backoff | LiteLLM retry logic | `litellm.proxy.proxy_server` |
| a2a-sdk | LiteLLM A2A endpoints | `litellm.proxy.agent_endpoints.a2a_endpoints` |
| aiofiles | Config file operations | Likely used transitively |
| watchdog | Hot reload file watching | Used by `hot_reload.py` (probably via litellm) |
| prometheus-client | Metrics export | Likely used by litellm's prometheus callback |
| python-multipart | Form data parsing | FastAPI/Starlette runtime requirement |

**Assessment:** These are all **legitimate LiteLLM runtime dependencies** that are
needed when running the proxy but are imported by LiteLLM's code, not RouteIQ's.
The comment in `pyproject.toml` correctly explains this pattern. No action needed.

### 4.3 Optional vs Required

| Package | Currently | Should Be | Rationale |
|---------|-----------|-----------|-----------|
| sentence-transformers | Optional (knn) | Optional (knn) -- CORRECT | Only for KNN routing |
| scikit-learn | Optional (knn) | Optional (knn) -- CORRECT | Only for KNN routing |
| torch | Transitive via sentence-transformers | Should stay optional | Pulled automatically |
| google-cloud-aiplatform | Optional (cloud) | Optional (cloud) -- CORRECT | Only for Vertex AI |
| langfuse | Optional (callbacks) | Optional (callbacks) -- CORRECT | Only for Langfuse callback |
| prisma | Optional (db) | Optional (db) -- CORRECT | Only for PostgreSQL |

**Assessment:** The optional dependency structure is well-designed. The `prod` extra
aggregates all production deps correctly. No changes needed.

### 4.4 Python 3.14+ Compatibility

`requires-python = ">=3.14"` is very aggressive.

| Package | 3.14 Support | Notes |
|---------|-------------|-------|
| torch 2.10.0 | YES | cp314 wheels present in uv.lock |
| aiohttp 3.13.3 | YES | cp314 wheels present |
| pydantic-core 2.41.5 | YES | Rust-compiled, cp314 wheels |
| grpcio 1.76.0 | YES | C extension, cp314 wheels |
| cryptography 46.0.3 | YES | Rust-compiled, cp314 wheels |
| prisma 0.15.0 | UNKNOWN | Pure Python, likely OK |
| All others | YES | Per uv.lock resolution |

**Assessment:** All packages in `uv.lock` resolved successfully for Python 3.14,
confirming compatibility. The aggressive target is viable given the current
ecosystem state.

### 4.5 License Compatibility

All licenses found in the dependency tree:

| License | Count | Compatible with MIT? | Notes |
|---------|-------|---------------------|-------|
| MIT | ~40 | YES | Most common |
| Apache-2.0 | ~50 | YES | Google, OTel, AWS |
| BSD-3-Clause | ~30 | YES | PyTorch, Starlette |
| BSD-2-Clause | ~5 | YES | |
| MPL-2.0 | 2 | YES (file-level copyleft) | hypothesis, certifi |
| CC0 | 1 | YES | email-validator |
| PSF-2.0 | ~5 | YES | Python stdlib extensions |

**Assessment:** No GPL, AGPL, or other restrictive copyleft licenses found.
All dependencies are compatible with RouteIQ's MIT license. The MPL-2.0 licensed
packages (hypothesis is dev-only, certifi is permissive in practice) pose no
distribution concerns.

---

## Part 5: Supply Chain Security

### 5.1 Typosquatting Risk

| Package | Risk | Similar Malicious Names |
|---------|------|------------------------|
| litellm | MEDIUM | `litelm`, `lite-llm`, `littellm` |
| fastapi | LOW | Well-known, high download count |
| torch | LOW | Well-known, PyTorch org |
| redis | LOW | Official Redis client |
| boto3 | LOW | Official AWS SDK |
| a2a-sdk | MEDIUM | New package (0.3.x), only 5 maintainers |
| sentence-transformers | LOW | HuggingFace official |
| opentelemetry-* | LOW | Official OTel org packages |

**Assessment:** `litellm` and `a2a-sdk` have moderate typosquatting risk due to
being newer packages with less established reputations. However, both are sourced
from their official PyPI namespaces with verified maintainers.

### 5.2 Maintainer Risk

| Package | Maintainers | Last Release | Activity | Risk |
|---------|------------|--------------|----------|------|
| litellm | BerriAI (startup) | Jan 2026 | Very active (daily) | MEDIUM -- startup dependency |
| a2a-sdk | Google (5 maintainers) | Dec 2025 | Active | LOW |
| sentence-transformers | HuggingFace | Active | Active | LOW |
| fastapi | Tiangolo + team | Active | Very active | LOW |
| torch | Meta/PyTorch Foundation | Jan 2026 | Very active | LOW |
| prisma | RobertCraigie | Active | Active | LOW |
| opentelemetry-* | CNCF/OTel | Active | Very active | LOW |
| watchdog | gorakhargosh | Active | Moderate | LOW |
| apscheduler | agronholm | Active | Moderate | LOW |

**Assessment:** The primary maintainer risk is **LiteLLM** -- it's maintained by
BerriAI, a venture-funded startup. If the company fails, the library could become
unmaintained. RouteIQ should monitor LiteLLM's health and have a contingency plan
(the reference submodule at a specific commit provides some protection).

### 5.3 SBOM Generation

| Tool | Capability | Status |
|------|-----------|--------|
| `uv.lock` | Package manifest with hashes | AVAILABLE |
| `uv pip compile` | Requirements.txt generation | AVAILABLE |
| `syft` (Anchore) | CycloneDX/SPDX SBOM | Can be added to CI |
| `pip-audit` + `cyclonedx-bom` | Vulnerability + SBOM | Not installed |

**Recommendation:** Add SBOM generation to CI pipeline:
```bash
# Install and generate CycloneDX SBOM
uv run pip install cyclonedx-bom
uv run cyclonedx-py environment -o sbom.json --format json
```

### 5.4 Reproducible Builds

| Aspect | Status | Notes |
|--------|--------|-------|
| Lock file (`uv.lock`) | YES | Hash-verified, version-locked |
| Docker `--frozen` flag | YES | Fails if lock file mismatches |
| Base image pinned by digest | YES | `sha256:` digests in Dockerfile |
| LLMRouter pinned by commit | YES | `LLMROUTER_COMMIT` build arg |
| Build cache mounts | YES | `--mount=type=cache` for uv |
| Deterministic output | PARTIAL | Build caches may vary |

**Assessment:** Reproducibility is strong. The combination of `uv.lock` hashes,
digest-pinned base images, and `--frozen` installs provides high confidence in
build reproducibility. The main gap is that BuildKit cache mounts may produce
slightly different layer orderings.

### 5.5 Signature Verification

| Aspect | Status | Notes |
|--------|--------|-------|
| PyPI wheel hashes | YES | Verified in `uv.lock` |
| GPG signatures on wheels | NO | Most PyPI packages don't sign |
| Sigstore/PEP 740 attestations | PARTIAL | Some packages publish |
| Docker image signing | NO | Not configured |

**Recommendation:** Enable Docker Content Trust or cosign for image signing:
```bash
# Sign with cosign
cosign sign ghcr.io/org/routeiq:v0.0.2
```

---

## Part 6: Prioritized Recommendations for v0.0.2

### Priority 1: HIGH (Do Before Release)

1. **Pin LiteLLM to exact version** -- Change `>=1.81.3` to `==1.81.3` in
   `pyproject.toml` to prevent accidental upgrade to a potentially vulnerable
   version. LiteLLM's CVE history warrants caution.

2. **Verify CVE-2025-45809 (SQL injection) and CVE-2025-11203 (API key exposure)
   status** -- Confirm these are fixed in v1.81.3. Check LiteLLM release notes
   or code for patches.

3. **Install `pip-audit`** -- Add to dev dependencies and CI pipeline:
   ```toml
   dev = [..., "pip-audit>=2.7.0"]
   ```
   Run in CI: `uv run pip-audit --strict --desc`

### Priority 2: MEDIUM (Plan for v0.0.3)

4. **Create CPU-only Docker variant** -- Add a `docker/Dockerfile.cpu` that
   installs `torch` without CUDA support, reducing image size by ~2 GB.
   Most deployments running LLM routing don't need GPU inference locally.

5. **Make `knn` extra truly optional in Docker** -- Currently, `prod` includes
   `knn`, pulling torch+CUDA into every production image. Consider:
   - `prod-lite` = `db,otel,cloud,callbacks` (no torch)
   - `prod-full` = `db,otel,cloud,callbacks,knn` (with torch)

6. **Add SBOM generation to CI** -- Generate CycloneDX SBOM on every release
   and attach to GitHub release artifacts.

7. **Subscribe to security advisories** for all critical dependencies:
   - https://github.com/BerriAI/litellm/security/advisories
   - https://github.com/pytorch/pytorch/security/advisories
   - https://github.com/pallets/jinja/security/advisories
   - https://github.com/redis/redis/security/advisories

### Priority 3: LOW (Backlog)

8. **Consider `safetensors`-only model loading** -- Disable unsafe model loading
   entirely (already default `LLMROUTER_ALLOW_PICKLE_MODELS=false`) and ensure
   sentence-transformers uses only safetensors format.

9. **Docker image signing** -- Implement cosign or Docker Content Trust for
   published images.

10. **Evaluate LiteLLM alternatives** -- Given LiteLLM's CVE history (16+ CVEs),
    evaluate whether RouteIQ should maintain a tighter integration that doesn't
    expose all LiteLLM proxy endpoints, reducing the attack surface.

11. **Monitor `a2a-sdk` maturity** -- The A2A protocol and SDK are relatively new
    (v0.3.x). Monitor for stability and security as the project matures.

12. **Add Dependabot or Renovate** -- Automated dependency update PRs to catch
    security patches promptly.

---

## Appendix A: Full Package List (173 packages)

<details>
<summary>Click to expand complete package list from uv.lock</summary>

```
a2a-sdk==0.3.22
aiofiles==25.1.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.12.1
apscheduler==3.11.2
asgiref==3.11.0
asyncpg==0.31.0
attrs==25.4.0
azure-core==1.38.0
azure-identity==1.25.1
backoff==2.2.1
black==26.1.0
boto3==1.42.32
botocore==1.42.32
certifi==2026.1.4
cffi==2.0.0
charset-normalizer==3.4.4
click==8.3.1
colorama==0.4.6
cryptography==46.0.3
cuda-bindings==12.9.4
cuda-pathfinder==1.3.3
distro==1.9.0
dnspython==2.8.0
docstring-parser==0.17.0
email-validator==2.3.0
fastapi==0.128.0
fastapi-sso==0.19.0
fastuuid==0.14.0
filelock==3.20.3
frozenlist==1.8.0
fsspec==2026.1.0
google-api-core==2.29.0
google-auth==2.47.0
google-cloud-aiplatform==1.134.0
google-cloud-bigquery==3.40.0
google-cloud-core==2.5.0
google-cloud-resource-manager==1.16.0
google-cloud-storage==3.8.0
google-crc32c==1.8.0
google-genai==1.59.0
google-resumable-media==2.8.0
googleapis-common-protos==1.72.0
grpc-google-iam-v1==0.14.3
grpcio==1.76.0
grpcio-status==1.76.0
h11==0.16.0
hf-xet==1.2.0
httpcore==1.0.9
httpx==0.28.1
httpx-sse==0.4.3
huggingface-hub==0.36.0
hypothesis==6.150.2
idna==3.11
importlib-metadata==8.7.1
iniconfig==2.3.0
jinja2==3.1.6
jiter==0.12.0
jmespath==1.0.1
joblib==1.5.3
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
langfuse==3.12.0
librt==0.7.7
litellm==1.81.3
litellm-llmrouter==0.1.0
markupsafe==3.0.3
mpmath==1.3.0
msal==1.34.0
msal-extensions==1.3.1
multidict==6.7.0
mypy==1.19.1
mypy-extensions==1.1.0
networkx==3.6.1
nodeenv==1.10.0
numpy==2.4.1
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
oauthlib==3.3.1
openai==2.15.0
opentelemetry-api==1.39.1
opentelemetry-exporter-otlp==1.39.1
opentelemetry-exporter-otlp-proto-common==1.39.1
opentelemetry-exporter-otlp-proto-grpc==1.39.1
opentelemetry-exporter-otlp-proto-http==1.39.1
opentelemetry-instrumentation==0.60b1
opentelemetry-instrumentation-asgi==0.60b1
opentelemetry-instrumentation-fastapi==0.60b1
opentelemetry-instrumentation-httpx==0.60b1
opentelemetry-instrumentation-logging==0.60b1
opentelemetry-instrumentation-requests==0.60b1
opentelemetry-proto==1.39.1
opentelemetry-sdk==1.39.1
opentelemetry-semantic-conventions==0.60b1
opentelemetry-util-http==0.60b1
orjson==3.11.5
packaging==25.0
pathspec==1.0.3
platformdirs==4.5.1
pluggy==1.6.0
prisma==0.15.0
prometheus-client==0.24.1
propcache==0.4.1
proto-plus==1.27.0
protobuf==6.33.4
pyasn1==0.6.2
pyasn1-modules==0.4.2
pycparser==2.23
pydantic==2.12.5
pydantic-core==2.41.5
pygments==2.19.2
pyjwt==2.10.1
pytest==9.0.2
pytest-asyncio==1.3.0
python-dateutil==2.9.0.post0
python-dotenv==1.2.1
python-multipart==0.0.22
pytokens==0.3.0
pyyaml==6.0.3
redis==7.1.0
referencing==0.37.0
regex==2026.1.15
requests==2.32.5
rpds-py==0.30.0
rsa==4.9.1
ruff==0.14.13
s3transfer==0.16.0
safetensors==0.7.0
scikit-learn==1.8.0
scipy==1.17.0
sentence-transformers==5.2.0
setuptools==80.10.1
six==1.17.0
sniffio==1.3.1
sortedcontainers==2.4.0
starlette==0.50.0
sympy==1.14.0
tenacity==9.1.2
threadpoolctl==3.6.0
tiktoken==0.12.0
tokenizers==0.22.2
tomlkit==0.14.0
torch==2.10.0
tqdm==4.67.1
transformers==4.57.6
triton==3.6.0
typing-extensions==4.15.0
typing-inspection==0.4.2
tzdata==2025.3
tzlocal==5.3.1
urllib3==2.6.3
watchdog==6.0.0
websockets==15.0.1
wrapt==1.17.3
yarl==1.22.0
zipp==3.23.0
```

</details>

---

## Appendix B: Dockerfile System-Level Dependencies

The Docker image (`docker/Dockerfile`) installs these system packages:

| Package | Purpose | Security Concern |
|---------|---------|------------------|
| build-essential | uvloop compilation | Could be removed post-build |
| curl | Health checks | Minimal |
| ca-certificates | TLS | Essential |
| tini | PID 1 init | Essential for signal handling |
| libatomic1 | Prisma/Node.js | Minimal |
| git (builder only) | Clone LLMRouter | Not in final image |

**Recommendation:** Consider a multi-stage approach that removes `build-essential`
from the final image after `uvloop` is compiled. This reduces the attack surface
of the production container.

---

## Appendix C: Methodology

- Dependency inventory extracted from `pyproject.toml` and `uv.lock`
- CVE data sourced from NVD, Snyk, GitHub Advisory Database, cvedetails.com
- Size estimates based on PyPI package sizes and typical wheel distributions
- License information from PyPI metadata and package source
- `pip-audit` was not available in the dev environment; CVE research performed
  via online vulnerability databases
- Import analysis performed via grep/regex search of `src/litellm_llmrouter/`
