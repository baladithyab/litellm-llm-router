# Architecture Review: Response Caching for RouteIQ Gateway

**Report**: 05 - Semantic Caching
**Date**: 2025-02-07
**Status**: Research and Recommendation
**Scope**: Exact-match caching, semantic caching, cache architecture, plugin design

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Industry Research: LLM Caching Patterns](#2-industry-research-llm-caching-patterns)
3. [RouteIQ Current State Assessment](#3-routeiq-current-state-assessment)
4. [Architecture Design: Cache Plugin](#4-architecture-design-cache-plugin)
5. [Cache Key Algorithm](#5-cache-key-algorithm)
6. [Semantic Caching Design](#6-semantic-caching-design)
7. [Streaming Response Caching](#7-streaming-response-caching)
8. [Multi-Tier Cache Architecture](#8-multi-tier-cache-architecture)
9. [Observability and Metrics](#9-observability-and-metrics)
10. [Configuration and Controls](#10-configuration-and-controls)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Risk Analysis](#12-risk-analysis)

---

## 1. Executive Summary

RouteIQ currently has **no response caching layer**. Every LLM request, regardless of whether an identical or near-identical request was recently processed, results in a full round-trip to the upstream provider. This represents a significant opportunity for cost reduction, latency improvement, and provider rate limit mitigation.

This report recommends a **two-tier caching system** implemented as a RouteIQ gateway plugin:

- **Tier 1 (Exact Match)**: SHA-256 hash-based cache with deterministic key canonicalization. Targets requests with temperature=0 or identical parameters. Expected hit rates: 15-40% for deterministic workloads.
- **Tier 2 (Semantic Match)**: Embedding-based similarity search for semantically equivalent requests. Uses the same sentence-transformers/all-MiniLM-L6-v2 model already in the RouteIQ dependency tree. Expected hit rates: 5-20% additional savings.

The caching system fits naturally into RouteIQ's existing plugin architecture using on_request (ASGI-level cache lookup with short-circuit) and on_llm_success (callback-level cache store) hooks, with Redis as the shared cache backend for HA deployments.

### Key Metrics (Estimated)

| Metric | Without Cache | With Exact Cache | With Semantic Cache |
|--------|--------------|------------------|---------------------|
| Average latency (cached) | 500-3000ms | 5-15ms | 20-50ms |
| Cost reduction | 0% | 15-40% | 20-55% |
| Provider rate limit headroom | Baseline | +15-40% | +20-55% |

---

## 2. Industry Research: LLM Caching Patterns

### 2.1 Exact-Match Caching

Exact-match caching is the simplest and most reliable form of LLM response caching. It works by computing a deterministic hash of the request parameters and using it as a cache key.

**How it works:**
1. Normalize request parameters (model, messages, temperature, top_p, max_tokens, tools, etc.)
2. Compute SHA-256 hash of the canonical representation
3. Look up hash in cache store
4. On hit: return cached response immediately
5. On miss: forward to provider, store response on success

**When it works best:**
- Deterministic requests (temperature=0 or very low temperature)
- Repeated system prompts (e.g., classification tasks, structured extraction)
- Automated pipelines with identical inputs
- Retry scenarios where the same request is sent multiple times

**Industry implementations:**

- **LiteLLM Proxy**: Built-in caching support with Redis, S3, and in-memory backends. Cache keys are computed from model + messages + temperature + logit_bias + max_tokens + n + stream + top_p + tools + tool_choice + functions + function_call. Supports per-request cache bypass and TTL configuration. The proxy enables caching via litellm_settings.cache: true in config YAML.

- **Portkey**: "Simple Caching" mode that hashes the entire request body. Supports x-portkey-cache header with values simple (exact match) or semantic. Force refresh via x-portkey-cache-force-refresh: true. Responses include x-portkey-cache-status: HIT/MISS.

- **Helicone**: Bucket-based caching with configurable cache-control policies. Uses request hash with opt-in via Helicone-Cache-Enabled: true header.

### 2.2 Semantic Caching

Semantic caching extends exact-match by finding cached responses for requests that are semantically similar but not textually identical.

**How it works:**
1. Extract the "semantic content" from the request (typically the last user message, or a combination of system + user messages)
2. Generate an embedding vector using a sentence-transformer model
3. Search for similar embeddings in a vector index (within a partition defined by model + non-semantic parameters)
4. If similarity exceeds threshold, return the cached response
5. On miss: forward to provider, store response + embedding on success

**Key architectural reference -- GPTCache:**
GPTCache (by Zilliz) is the most well-known open-source semantic caching library for LLMs. Its architecture provides useful design patterns:

- **Pre-processing**: Extracts the "query" from the request (last user message by default, configurable)
- **Embedding**: Converts query to vector (supports OpenAI, Hugging Face, ONNX, Cohere, etc.)
- **Similarity evaluation**: Computes distance between query embedding and cached embeddings. Supports cosine similarity, L2 distance, and learned similarity functions. Default threshold: 0.8 cosine similarity.
- **Cache storage**: Supports SQLite, MySQL, PostgreSQL for metadata; FAISS, Milvus, Qdrant, Redis VSS, Chromadb for vector search.
- **Eviction policy**: LRU and TTL-based eviction.

**Portkey Semantic Caching:**
Portkey's implementation uses cosine similarity with embeddings to match semantically similar prompts. Users select semantic mode via the x-portkey-cache: semantic header. The similarity threshold is not user-configurable in their implementation.

**Redis Vector Similarity Search (VSS):**
Redis Stack (7.0+) supports vector similarity search natively via the RediSearch module:
- Stores vectors as HASH or JSON fields
- Supports FLAT (brute force) and HNSW (approximate nearest neighbor) indexing
- Cosine, L2 (Euclidean), and IP (inner product) distance metrics
- Sub-millisecond search for datasets up to millions of vectors
- Native Redis TTL applies to vector entries

This is particularly relevant for RouteIQ since the HA deployment already includes Redis.

### 2.3 Cache Key Design Patterns

The cache key must be deterministic for the same logical request, regardless of:
- JSON key ordering
- Whitespace variations
- Irrelevant parameter differences (request_id, timestamps)

**Parameters that MUST be included in the cache key:**

| Parameter | Reason |
|-----------|--------|
| model | Different models produce different outputs |
| messages | Core request content |
| temperature | Affects output randomness/determinism |
| top_p | Affects sampling distribution |
| max_tokens | Affects output length |
| stop | Affects where output terminates |
| tools / tool_choice | Affects function calling behavior |
| functions / function_call | Legacy function calling |
| response_format | JSON mode vs text |
| seed | Reproducibility parameter |
| logit_bias | Modifies token probabilities |
| n | Number of completions |
| presence_penalty | Affects token selection |
| frequency_penalty | Affects token selection |
| top_k | Affects sampling (provider-specific) |

**Parameters that SHOULD be excluded from the cache key:**

| Parameter | Reason |
|-----------|--------|
| stream | Same content, different delivery format |
| user | User tracking, not content-affecting |
| metadata | Gateway-internal metadata |
| api_key | Authentication, not content-affecting |
| request_id | Correlation ID |
| timeout | Client-side behavior |
| litellm_params | Gateway-internal routing |
| proxy_server_request | LiteLLM proxy internals |
| litellm_call_id | LiteLLM internal ID |

### 2.4 Cache Invalidation Strategies

LLM responses are generally **immutable for deterministic requests** (temperature=0, same seed). For non-deterministic requests, the cached response represents one valid sample from the distribution.

**TTL-based expiration:**
- Default TTL: 1 hour for exact match, 30 minutes for semantic match
- Model-specific TTLs (e.g., longer for stable models, shorter for frequently updated ones)
- Per-request TTL override via headers

**Explicit invalidation:**
- Admin API endpoint to flush cache by model, key pattern, or entirely
- Model version changes should invalidate cache (e.g., when provider updates a model)

**Capacity-based eviction:**
- LRU (Least Recently Used) for in-memory L1 cache
- Redis maxmemory-policy allkeys-lru for L2 cache (already configured in HA compose)
- Maximum cache entry size limit (prevent caching extremely long responses)

### 2.5 Streaming Response Caching Challenges

Streaming (stream=true) introduces complexity because:

1. **Responses arrive as Server-Sent Events (SSE) chunks** -- must be reassembled before caching
2. **Cache hits must be re-chunked** for streaming clients
3. **Partial responses** (client disconnect mid-stream) should not be cached
4. **Streaming and non-streaming share cache** -- the same content, different delivery

**Solutions employed by the industry:**

- **LiteLLM**: Caches the complete response object. On cache hit for streaming requests, wraps the cached response in a synthetic SSE stream.
- **Portkey**: Assembles streaming chunks into complete response before caching. Replays from cache with synthetic chunking.
- **Buffer-then-cache**: Intercept all SSE chunks in the callback, assemble complete response, cache only after [DONE] sentinel.

**Recommended approach for RouteIQ:**
Use the on_llm_success hook which fires after the complete response is available (LiteLLM reassembles streaming responses before calling success callbacks). On cache hit during on_request, return the cached response; the caller will handle re-streaming if needed. The stream parameter is excluded from the cache key so both streaming and non-streaming requests share cached data.

---

## 3. RouteIQ Current State Assessment

### 3.1 Existing Infrastructure (Strengths)

RouteIQ has several components that directly support a caching implementation:

**Plugin System** (src/litellm_llmrouter/gateway/plugin_manager.py):
- GatewayPlugin base class with well-defined lifecycle hooks
- on_llm_pre_call(model, messages, kwargs) -- receives the full request before it reaches the provider. Can return a dict of kwargs overrides.
- on_llm_success(model, response, kwargs) -- receives the complete response after successful LLM call.
- PluginCallbackBridge (src/litellm_llmrouter/gateway/plugin_callback_bridge.py) wires these into LiteLLM's callback system.
- Plugin middleware (src/litellm_llmrouter/gateway/plugin_middleware.py) supports on_request with PluginResponse for short-circuiting at the ASGI level.

**Redis in HA deployment** (docker-compose.ha.yml):
- Redis 7 Alpine already deployed with allkeys-lru eviction policy and 256MB memory limit
- Used for distributed state and leader election
- redis>=5.0.0 already in project dependencies (pyproject.toml)

**Sentence-Transformers** (src/litellm_llmrouter/strategies.py):
- sentence-transformers already in optional dependencies (knn extra)
- Lazy-loading singleton pattern already implemented (_get_sentence_transformer())
- Default model sentence-transformers/all-MiniLM-L6-v2 already used for KNN routing
- The same model instance can be shared with semantic caching

**Observability** (src/litellm_llmrouter/observability.py):
- create_cache_span(operation, cache_key) method already exists on ObservabilityManager
- Ready to emit cache.lookup, cache.set, cache.delete spans
- OTEL attribute cache.key already defined (truncated for privacy)

**Circuit Breaker** (src/litellm_llmrouter/resilience.py):
- CircuitBreakerManager already has a REDIS breaker
- Can protect cache operations from Redis failures
- Graceful fallback when cache is unavailable

### 3.2 Gaps to Address

| Gap | Impact | Effort |
|-----|--------|--------|
| No cache plugin exists | Core deliverable | High |
| on_llm_pre_call returns kwargs overrides only, cannot short-circuit | Cache hits must use ASGI-level on_request instead | Medium |
| No Redis connection management for cache ops | Need async Redis client integration | Medium |
| No vector search capability | Need Redis VSS or in-process FAISS | Medium |
| No cache key canonicalization logic | Need deterministic hashing | Low |
| No cache admin endpoints | Flush, stats, warm | Low |
| No cache bypass headers | x-routeiq-cache-control | Low |

### 3.3 Critical Design Constraint: Short-Circuiting

The current on_llm_pre_call hook in PluginCallbackBridge (lines 95-112) has a limitation for caching:

```python
async def async_log_pre_api_call(self, model, messages, kwargs):
    for plugin in self._plugins:
        result = await plugin.on_llm_pre_call(model, messages, kwargs)
        if isinstance(result, dict):
            kwargs.update(result)  # Can only modify kwargs, not return cached response
```

This hook can modify kwargs but cannot directly short-circuit the LLM call with a cached response. Two architectural approaches exist:

**Approach A: ASGI-Level Cache (via on_request + PluginResponse)**
- The PluginMiddleware supports returning PluginResponse from on_request to short-circuit
- Cache lookup at the HTTP layer, before the request reaches LiteLLM's routing
- Requires parsing the HTTP request body in the ASGI middleware layer
- Can return cached responses with correct HTTP status and headers
- Advantage: Complete control over response format, works with streaming replay
- Disadvantage: Must parse raw HTTP body; does not benefit from LiteLLM's parameter normalization

**Approach B: LiteLLM Callback-Level Cache (via async_pre_call_hook)**
- LiteLLM's proxy has async_pre_call_hook on CustomLogger that can return a modified response
- The PluginCallbackBridge could be extended to support a cache response sentinel
- The cached response would be injected into LiteLLM's response path
- Advantage: Works within LiteLLM's existing response handling
- Disadvantage: Requires understanding LiteLLM's internal response flow; coupling risk

**Recommended: Hybrid approach**
- Use on_request (ASGI layer) for cache lookup and short-circuit on hits
- Use on_llm_success (callback layer) for cache storage after successful LLM calls
- This gives maximum control on the read path and minimal coupling on the write path

---

## 4. Architecture Design: Cache Plugin

### 4.1 Plugin Structure

```
src/litellm_llmrouter/gateway/plugins/
    response_cache.py          # Main cache plugin (GatewayPlugin subclass)
    cache_backends.py          # Cache backend abstractions (L1 / L2)
    cache_key.py               # Cache key canonicalization
    cache_semantic.py          # Semantic similarity module
```

### 4.2 Plugin Class Design

```python
class ResponseCachePlugin(GatewayPlugin):
    """
    LLM Response Cache Plugin.

    Implements two-tier caching:
    - L1: In-process LRU cache (for single-instance deployments)
    - L2: Redis-backed shared cache (for HA deployments)

    Cache lookup: on_request (ASGI layer, short-circuits with PluginResponse)
    Cache store:  on_llm_success (LiteLLM callback layer)
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="response-cache",
            version="1.0.0",
            capabilities={PluginCapability.MIDDLEWARE, PluginCapability.EVALUATOR},
            priority=50,  # Run early (before other plugins)
            failure_mode=FailureMode.CONTINUE,  # Cache failures are non-fatal
            description=(
                "Two-tier LLM response caching with exact and semantic matching"
            ),
        )

    async def startup(self, app, context):
        """Initialize cache backends, warm cache if configured."""

    async def shutdown(self, app, context):
        """Flush pending writes, close connections."""

    async def on_request(self, request: PluginRequest) -> PluginResponse | None:
        """
        Cache lookup on incoming chat/completion requests.

        Only intercepts POST /v1/chat/completions and /chat/completions.
        Parses body, computes cache key, checks L1 then L2.
        Returns PluginResponse with cached data on hit, None on miss.
        """

    async def on_llm_success(self, model, response, kwargs):
        """
        Cache store after successful LLM response.

        Computes cache key from kwargs, serializes response,
        stores in L1 and L2 with configured TTL.
        """

    async def health_check(self):
        """Report cache health: L1 size, L2 connectivity, hit rates."""
```

### 4.3 Request/Response Flow

```
Client Request
    |
    v
[RequestIDMiddleware]
    |
    v
[PolicyMiddleware]
    |
    v
[PluginMiddleware]
    |
    +-- on_request() called on ResponseCachePlugin
    |       |
    |       +-- Is this a cacheable endpoint? (POST /v1/chat/completions)
    |       |       No -> return None (pass through)
    |       |
    |       +-- Is cache bypass requested? (x-routeiq-cache-control: no-cache)
    |       |       Yes -> return None (pass through)
    |       |
    |       +-- Parse request body, extract cache-relevant params
    |       |
    |       +-- Compute exact-match cache key (SHA-256)
    |       |
    |       +-- L1 lookup (in-memory LRU)
    |       |       Hit -> return PluginResponse(200, cached_body)
    |       |
    |       +-- L2 lookup (Redis GET)
    |       |       Hit -> populate L1, return PluginResponse(200, cached_body)
    |       |
    |       +-- [If semantic caching enabled]
    |       |   +-- Compute embedding of user message
    |       |   +-- Redis VSS / FAISS similarity search
    |       |   +-- If similarity >= threshold
    |       |           Hit -> return PluginResponse(200, cached_body)
    |       |
    |       +-- Cache miss -> return None (pass through)
    |
    v
[LiteLLM Proxy / Router]
    |
    v
[LLM Provider] (OpenAI, Anthropic, Bedrock, etc.)
    |
    v
[PluginCallbackBridge]
    |
    +-- on_llm_success() called on ResponseCachePlugin
    |       |
    |       +-- Is this a cacheable response? (status 200, non-error)
    |       |
    |       +-- Compute cache key from kwargs
    |       |
    |       +-- Serialize response
    |       |
    |       +-- Store in L1 (in-memory, with size limit)
    |       |
    |       +-- Store in L2 (Redis SET with TTL)
    |       |
    |       +-- [If semantic caching enabled]
    |           +-- Compute embedding
    |           +-- Store embedding + key mapping in vector index
    |
    v
Client Response (with x-routeiq-cache: MISS header)
```

### 4.4 Body Parsing Challenge

The on_request hook receives a PluginRequest which contains headers and path but **not the request body** (by design, to avoid buffering that breaks streaming). For cache lookup, we need the body.

**Solution options:**

1. **Buffer body in middleware for cacheable paths only**: When the path matches /v1/chat/completions or /chat/completions, read the body bytes, parse JSON, perform lookup, and if miss, replay the body for the inner app. This is a targeted exception to the no-buffer rule.

2. **Use a dedicated ASGI middleware before PluginMiddleware**: A thin CacheMiddleware that only activates for cacheable endpoints, reads the body, performs lookup, and either short-circuits or passes through.

3. **Rely on on_llm_pre_call for lookup and inject response**: Extend the callback bridge to support a special return value that signals "use this cached response instead of calling the provider."

**Recommended: Option 2** -- A dedicated CacheMiddleware added as an inner ASGI wrapper (between PluginMiddleware and the app). This keeps the caching concern isolated and avoids modifying the general-purpose PluginMiddleware.

---

## 5. Cache Key Algorithm

### 5.1 Exact-Match Key Generation

```python
import hashlib
import json

# Parameters included in cache key (sorted for determinism)
CACHE_KEY_PARAMS = [
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "seed",
    "logit_bias",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "top_k",
]

# Parameters explicitly excluded
CACHE_KEY_EXCLUDED = {
    "stream",
    "user",
    "metadata",
    "api_key",
    "request_id",
    "timeout",
    "litellm_params",
    "proxy_server_request",
    "litellm_call_id",
}


def compute_cache_key(request_params: dict) -> str:
    """
    Compute a deterministic cache key from request parameters.

    Returns:
        "routeiq:cache:v1:<sha256_hex>"
    """
    canonical = {}
    for param in CACHE_KEY_PARAMS:
        if param in request_params:
            value = request_params[param]
            if value is not None:
                canonical[param] = _normalize_value(value)

    canonical_json = json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    )
    key_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return f"routeiq:cache:v1:{key_hash}"


def _normalize_value(value):
    """Normalize a parameter value for deterministic hashing."""
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in sorted(value.items())}
    elif isinstance(value, float):
        return round(value, 10)
    elif isinstance(value, str):
        return value.strip()
    return value
```

### 5.2 Message Normalization

Messages require special handling because:
- Trailing whitespace in content should not affect the key
- Empty name fields vs absent name fields should be equivalent
- Tool call results may have non-deterministic ordering

```python
def _normalize_message(msg: dict) -> dict:
    """Normalize a chat message for cache key computation."""
    normalized = {"role": msg["role"]}

    if "content" in msg and msg["content"] is not None:
        content = msg["content"]
        if isinstance(content, str):
            normalized["content"] = content.strip()
        elif isinstance(content, list):
            normalized["content"] = [
                _normalize_content_part(p) for p in content
            ]

    if msg.get("name"):
        normalized["name"] = msg["name"].strip()

    if "tool_calls" in msg:
        normalized["tool_calls"] = sorted(
            msg["tool_calls"],
            key=lambda tc: tc.get("id", "")
        )

    if "tool_call_id" in msg:
        normalized["tool_call_id"] = msg["tool_call_id"]

    return normalized
```

### 5.3 Cacheability Rules

Not all requests should be cached:

```python
def is_cacheable_request(params: dict) -> tuple[bool, str]:
    """
    Determine if a request is cacheable.

    Returns:
        (is_cacheable, reason)
    """
    if not params.get("model") or not params.get("messages"):
        return False, "missing_model_or_messages"

    temp = params.get("temperature", 1.0)
    if temp > MAX_CACHEABLE_TEMPERATURE:
        return False, "temperature_too_high"

    if params.get("n", 1) > 1:
        return False, "multiple_completions"

    messages = params.get("messages", [])
    total_content_len = sum(
        len(str(m.get("content", ""))) for m in messages
    )
    if total_content_len > 100_000:
        return False, "content_too_long"

    model = params.get("model", "")
    if model in EXCLUDED_MODELS:
        return False, "model_excluded"

    return True, "cacheable"
```

---

## 6. Semantic Caching Design

### 6.1 Embedding Strategy

**Model selection**: sentence-transformers/all-MiniLM-L6-v2
- Already loaded for KNN routing in strategies.py
- 384-dimensional embeddings
- Fast inference (~5ms per query on CPU)
- Good balance of quality vs speed for cache matching
- Shared singleton via _get_sentence_transformer()

**What to embed**: The semantic content of the request, extracted as follows:

```python
def extract_semantic_content(messages: list[dict]) -> str:
    """
    Extract the semantically meaningful content from a message chain.

    Strategy:
    1. Always include system message (defines behavior context)
    2. Include the last user message (the actual query)
    3. Optionally include last assistant message (for multi-turn context)
    """
    parts = []

    for msg in messages:
        if msg["role"] == "system":
            content = msg.get("content", "")
            if content:
                parts.append(f"[system] {content[:500]}")
            break

    for msg in reversed(messages):
        if msg["role"] == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"[user] {content}")
            elif isinstance(content, list):
                text_parts = [
                    p["text"] for p in content if p.get("type") == "text"
                ]
                parts.append(f"[user] {' '.join(text_parts)}")
            break

    return "\n".join(parts)
```

### 6.2 Similarity Search Architecture

**Option A: Redis Vector Similarity Search (VSS)**
- Requires Redis Stack (redis/redis-stack image) or Redis 7+ with RediSearch module
- Native vector indexing with HNSW algorithm
- Supports cosine similarity distance
- Redis TTL applies natively to vector entries
- Shared with existing HA Redis infrastructure

**Option B: In-Process FAISS**
- No additional infrastructure
- Sub-millisecond search for datasets up to 100K vectors
- Not shared across HA instances (each instance has its own index)
- Must be rebuilt on restart

**Option C: pgvector**
- Requires PostgreSQL extension
- Already have PostgreSQL in HA setup
- Good for persistent semantic indices
- Higher latency than Redis VSS or FAISS

**Recommended: Redis VSS (primary) with FAISS fallback (single-instance)**
- In HA deployments with Redis Stack: use Redis VSS for shared semantic index
- In single-instance deployments: use in-process FAISS for zero-dependency operation
- Both use the same embedding model and similarity threshold

### 6.3 Similarity Threshold Tuning

The similarity threshold determines the trade-off between cache hit rate and response quality:

| Threshold | Hit Rate | Quality Risk | Use Case |
|-----------|----------|--------------|----------|
| 0.98+ | Very Low | Minimal | Near-exact matches only |
| 0.95 | Low | Very Low | Conservative production default |
| 0.90 | Medium | Low | Balanced default |
| 0.85 | High | Medium | Cost-optimized workloads |
| 0.80 | Very High | High | Aggressive caching (dev/test) |

**Recommended default: 0.95** with configurable override.

The threshold should also be **partitioned by determinism level**:
- For temperature=0: threshold can be lower (0.90) because responses are deterministic
- For temperature>0: threshold should be higher (0.97) because different wordings may legitimately produce different outputs
- Per-model thresholds to account for model sensitivity differences

### 6.4 Semantic Cache Partitioning

Semantic searches MUST be partitioned by non-semantic parameters to avoid returning cached responses for a different model or parameter configuration:

```
Partition key = model + temperature_bucket + response_format + tools_hash
```

Where temperature_bucket:
- deterministic for temperature=0
- low for 0 < temperature <= 0.5
- medium for 0.5 < temperature <= 1.0

This ensures a semantic match for "What is Python?" sent to gpt-4 will never return a cached response from claude-3-opus.

---

## 7. Streaming Response Caching

### 7.1 Problem Statement

When stream=true, LiteLLM sends the response as Server-Sent Event (SSE) chunks. The on_llm_success callback receives the complete, reassembled response object after all chunks have been sent. This means:

1. **Cache store is straightforward**: on_llm_success provides the complete response
2. **Cache hit replay is the challenge**: Must convert a cached response back to SSE chunks

### 7.2 Cache Storage Format

Store the complete, non-streaming response object:

```python
@dataclass
class CacheEntry:
    """Serialized cache entry."""
    response_json: str        # Complete response (ModelResponse serialized)
    model: str                # Model that generated the response
    created_at: float         # Unix timestamp
    ttl_seconds: int          # Time-to-live
    token_count: int | None   # Total tokens (for metrics)
    cache_key: str            # Exact-match key
    semantic_key: str | None  # Semantic embedding key (if applicable)
```

### 7.3 Streaming Replay Strategy

When a cache hit occurs for a streaming request, the cache middleware must convert the stored non-streaming response into synthetic SSE events:

```python
async def replay_cached_response_as_stream(
    cached_response: dict,
    send: Send,
) -> None:
    """
    Convert a cached non-streaming response to SSE chunks.

    Emits synthetic SSE events that match the OpenAI streaming format:
    1. Initial chunk with role
    2. Content chunks (split into segments)
    3. Final chunk with finish_reason and usage
    4. [DONE] sentinel
    """
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            (b"content-type", b"text/event-stream"),
            (b"cache-control", b"no-cache"),
            (b"x-routeiq-cache", b"HIT"),
        ],
    })

    content = cached_response["choices"][0]["message"]["content"]
    chunks = split_into_sse_chunks(content, cached_response)

    body_parts = []
    for chunk in chunks:
        body_parts.append(f"data: {json.dumps(chunk)}\n\n".encode())
    body_parts.append(b"data: [DONE]\n\n")

    await send({
        "type": "http.response.body",
        "body": b"".join(body_parts),
        "more_body": False,
    })
```

### 7.4 Non-Streaming Cache Hit

For non-streaming requests, cache hit is simpler -- return the cached response body directly via PluginResponse:

```python
async def return_cached_response(entry: CacheEntry) -> PluginResponse:
    return PluginResponse(
        status_code=200,
        body=json.loads(entry.response_json),
        headers={
            "x-routeiq-cache": "HIT",
            "x-routeiq-cache-age": str(
                int(time.time() - entry.created_at)
            ),
        },
    )
```

---

## 8. Multi-Tier Cache Architecture

### 8.1 L1: In-Memory LRU Cache

```python
from collections import OrderedDict
import threading

class L1Cache:
    """
    Thread-safe in-memory LRU cache for hot entries.

    Features:
    - O(1) lookup and insertion
    - Size-bounded (by entry count and total bytes)
    - TTL-aware eviction
    - No external dependencies
    """

    def __init__(
        self,
        max_entries: int = 1000,
        max_bytes: int = 50 * 1024 * 1024,  # 50MB
    ):
        self._store: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
```

**Configuration:**
- ROUTEIQ_CACHE_L1_MAX_ENTRIES=1000 -- Maximum entries in L1
- ROUTEIQ_CACHE_L1_MAX_BYTES=52428800 -- Maximum memory for L1 (50MB)

### 8.2 L2: Redis Cache

```python
import redis.asyncio as aioredis

class L2RedisCache:
    """
    Redis-backed shared cache for HA deployments.

    Features:
    - Shared across all gateway instances
    - TTL-based expiration (Redis native)
    - LRU eviction (Redis maxmemory-policy)
    - Circuit breaker protection for Redis failures
    """

    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "routeiq:cache:v1:",
        default_ttl: int = 3600,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self._redis = aioredis.from_url(redis_url)
        self._prefix = key_prefix
        self._default_ttl = default_ttl
        self._cb = circuit_breaker
```

**Configuration:**
- ROUTEIQ_CACHE_REDIS_URL=redis://redis:6379/1 -- Redis URL (separate DB from main)
- ROUTEIQ_CACHE_REDIS_TTL=3600 -- Default TTL in seconds
- ROUTEIQ_CACHE_REDIS_MAX_ENTRY_SIZE=1048576 -- Max entry size (1MB)

### 8.3 Lookup Order

```
Request arrives
    |
    v
L1 in-memory lookup (< 0.1ms)
    |
    Hit? --> Return immediately
    |
    Miss
    |
    v
L2 Redis exact-match lookup (1-3ms)
    |
    Hit? --> Populate L1, return
    |
    Miss
    |
    v
[If semantic caching enabled]
L2 Redis VSS / FAISS semantic lookup (5-20ms)
    |
    Hit (similarity >= threshold)? --> Populate L1, return
    |
    Miss
    |
    v
Forward to LLM provider (500-3000ms)
    |
    v
On success: Store in L1 + L2 + Vector Index
```

---

## 9. Observability and Metrics

### 9.1 OTel Spans

The existing create_cache_span() method on ObservabilityManager should be used:

| Span Name | Attributes | When |
|-----------|-----------|------|
| cache.lookup | cache.key, cache.hit, cache.tier (L1/L2/semantic), cache.latency_ms | Every cache lookup |
| cache.store | cache.key, cache.tier, cache.entry_size_bytes, cache.ttl_seconds | Every cache store |
| cache.evict | cache.key, cache.tier, cache.reason (ttl/lru/manual) | Every eviction |

### 9.2 Prometheus Metrics

```
routeiq_cache_operations_total{operation, tier, status}
routeiq_cache_operation_duration_seconds{operation, tier}
routeiq_cache_entries{tier}
routeiq_cache_bytes{tier}
routeiq_cache_hits_total{model, tier}
routeiq_cache_misses_total{model}
routeiq_cache_cost_savings_total{model}
routeiq_cache_semantic_similarity{model}
```

### 9.3 Response Headers

Every response should include cache status headers:

| Header | Values | Description |
|--------|--------|-------------|
| x-routeiq-cache | HIT, MISS, BYPASS, DISABLED | Cache status |
| x-routeiq-cache-tier | l1, l2, semantic | Which tier served the hit |
| x-routeiq-cache-age | seconds | Age of cached entry |
| x-routeiq-cache-key | truncated hash | Cache key (first 12 chars) |
| x-routeiq-cache-similarity | 0.97 | Semantic similarity score (semantic hits only) |

### 9.4 Admin Endpoints

```
GET  /admin/cache/stats     # Hit rates, entry counts, memory usage
POST /admin/cache/flush     # Flush all or by model/pattern
POST /admin/cache/warm      # Pre-populate cache from request log
GET  /admin/cache/entries   # List recent cache entries (paginated)
```

---

## 10. Configuration and Controls

### 10.1 Environment Variables

```bash
# Feature toggle
ROUTEIQ_CACHE_ENABLED=true                    # Master switch (default: false)
ROUTEIQ_CACHE_SEMANTIC_ENABLED=false           # Semantic caching (default: false)

# L1 (in-memory) settings
ROUTEIQ_CACHE_L1_ENABLED=true                 # L1 cache toggle (default: true)
ROUTEIQ_CACHE_L1_MAX_ENTRIES=1000             # Max L1 entries (default: 1000)
ROUTEIQ_CACHE_L1_MAX_BYTES=52428800           # Max L1 memory (default: 50MB)

# L2 (Redis) settings
ROUTEIQ_CACHE_L2_ENABLED=true                 # L2 cache toggle (default: true when REDIS_HOST set)
ROUTEIQ_CACHE_REDIS_URL=redis://redis:6379/1  # Redis URL (default: from REDIS_HOST/PORT)
ROUTEIQ_CACHE_REDIS_TTL=3600                  # Default TTL seconds (default: 3600)
ROUTEIQ_CACHE_REDIS_MAX_ENTRY_SIZE=1048576    # Max entry bytes (default: 1MB)

# Semantic caching settings
ROUTEIQ_CACHE_SEMANTIC_THRESHOLD=0.95          # Cosine similarity threshold (default: 0.95)
ROUTEIQ_CACHE_SEMANTIC_MODEL=sentence-transformers/all-MiniLM-L6-v2
ROUTEIQ_CACHE_SEMANTIC_MAX_VECTORS=100000      # Max vectors in index (default: 100K)

# Cacheability rules
ROUTEIQ_CACHE_MAX_TEMPERATURE=1.0              # Max temperature for cacheability (default: 1.0)
ROUTEIQ_CACHE_MIN_TTL=60                       # Minimum TTL seconds (default: 60)
ROUTEIQ_CACHE_EXCLUDED_MODELS=                 # Comma-separated models to exclude

# Per-temperature TTL overrides
ROUTEIQ_CACHE_TTL_TEMP_0=7200                  # TTL for temperature=0 (default: 2 hours)
ROUTEIQ_CACHE_TTL_TEMP_LOW=3600                # TTL for 0 < temp <= 0.5 (default: 1 hour)
ROUTEIQ_CACHE_TTL_TEMP_MEDIUM=1800             # TTL for 0.5 < temp <= 1.0 (default: 30 min)
```

### 10.2 Per-Request Cache Control

Clients can control caching behavior via headers:

```
x-routeiq-cache-control: no-cache      # Bypass cache entirely
x-routeiq-cache-control: no-store      # Force refresh (bypass read, store result)
x-routeiq-cache-ttl: 7200              # Request-specific TTL override
x-routeiq-cache-mode: exact-only       # Disable semantic matching for this request
```

### 10.3 Config YAML Integration

```yaml
cache:
  enabled: true
  semantic_enabled: false
  l1:
    max_entries: 1000
    max_bytes: 52428800
  l2:
    redis_url: redis://redis:6379/1
    ttl: 3600
  semantic:
    threshold: 0.95
    model: sentence-transformers/all-MiniLM-L6-v2
  excluded_models:
    - "dall-e-3"
    - "whisper-1"
```

---

## 11. Implementation Roadmap

### Phase 1: Exact-Match Cache (Weeks 1-2)

**Deliverables:**
1. cache_key.py -- Cache key canonicalization and cacheability rules
2. cache_backends.py -- L1 (in-memory LRU) and L2 (Redis) backend abstractions
3. response_cache.py -- Main plugin with cache middleware for body parsing and short-circuit
4. Dedicated ASGI CacheMiddleware for body-aware cache lookup on cacheable endpoints
5. Cache response headers (x-routeiq-cache: HIT/MISS)
6. Unit tests for key generation, normalization, cacheability rules
7. Integration tests with Redis

**Dependencies:** None (Redis already in HA stack)

### Phase 2: Cache Management (Week 3)

**Deliverables:**
1. Admin endpoints (stats, flush, entries)
2. Per-request cache control headers
3. Cache bypass for excluded models
4. TTL configuration by temperature
5. Streaming response replay (synthetic SSE from cached response)
6. OTel spans and Prometheus metrics

**Dependencies:** Phase 1

### Phase 3: Semantic Caching (Weeks 4-5)

**Deliverables:**
1. cache_semantic.py -- Embedding generation and similarity search
2. Redis VSS integration (with FAISS fallback)
3. Semantic cache partitioning by model/parameters
4. Similarity threshold configuration
5. Semantic cache metrics (similarity distribution, hit rates by threshold)
6. A/B testing support (compare exact-only vs exact+semantic)

**Dependencies:** Phase 2, sentence-transformers (already in knn extra)

### Phase 4: Advanced Features (Weeks 6-7)

**Deliverables:**
1. Cache warming from request logs
2. Cost savings estimation and reporting
3. Adaptive TTL based on query frequency
4. Cache coherence for HA (invalidation broadcast via Redis pub/sub)
5. Response quality validation for semantic hits
6. Performance benchmarks and tuning guide

**Dependencies:** Phase 3

---

## 12. Risk Analysis

### 12.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Semantic cache returns incorrect response | Medium | High | Conservative default threshold (0.95), per-model tuning, quality validation |
| Redis failure causes gateway degradation | Low | High | Circuit breaker (already exists), L1 fallback, cache is non-critical path |
| Body parsing in ASGI breaks streaming | Medium | High | Only parse for cacheable endpoints, careful body replay via dedicated middleware |
| Memory pressure from L1 cache | Low | Medium | Strict size limits, LRU eviction, monitoring |
| Cache poisoning (cached error responses) | Low | Medium | Only cache 200 responses with valid completion content |
| Embedding model load increases startup time | Low | Medium | Lazy loading (existing pattern), shared with KNN routing |
| High-cardinality cache keys waste Redis memory | Medium | Low | TTL eviction, entry size limits, LRU policy |

### 12.2 Behavioral Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| Stale responses | Cached response becomes incorrect after provider model update | TTL-based expiration, admin flush API, model version tracking |
| Non-deterministic caching | Caching temperature > 0 responses may surprise users | Clear documentation, cache headers show HIT/MISS, opt-out header |
| Privacy concerns | Cached responses may contain PII | Per-user cache isolation option, TTL limits, admin flush |
| Streaming timing | Cached streaming replays are instant (no typing effect) | Optional artificial delay in streaming replay |

### 12.3 Operational Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| Redis Stack requirement | Semantic caching needs RediSearch module | FAISS fallback for standard Redis, feature flag for semantic |
| Increased Redis memory usage | Vector embeddings consume ~1.5KB each (384 dims x 4 bytes) | Vector count limits, separate Redis DB |
| Cache invalidation complexity | Multi-instance cache coherence | Redis pub/sub for invalidation broadcast |

---

## Appendix A: Comparison with LiteLLM Built-In Caching

LiteLLM already has a built-in caching system. Here is how a custom RouteIQ cache plugin compares:

| Feature | LiteLLM Built-In | RouteIQ Cache Plugin |
|---------|-----------------|---------------------|
| Exact match | Yes | Yes |
| Semantic match | Via GPTCache integration | Native with shared embedding model |
| Cache backends | Redis, S3, in-memory, disk | L1 (in-memory) + L2 (Redis) |
| Key algorithm | Fixed parameter set | Configurable, extensible |
| Streaming replay | Basic | Full SSE replay with proper chunking |
| Bypass headers | cache: {"no-cache": true} in body | HTTP header x-routeiq-cache-control |
| Admin API | Limited | Full CRUD + stats + warm |
| OTel integration | Limited | Native spans + Prometheus metrics |
| Circuit breaker | No | Yes (existing infrastructure) |
| Plugin system | No | Yes (GatewayPlugin lifecycle) |
| Multi-tier | No | Yes (L1 + L2) |

**Recommendation**: Build a custom cache plugin rather than enabling LiteLLM's built-in caching because:
1. Deeper integration with RouteIQ's observability, plugin system, and resilience primitives
2. Semantic caching shares the existing sentence-transformers model (no additional dependency)
3. Full control over cache key algorithm, bypass controls, and admin API
4. Multi-tier architecture (L1 + L2) for optimal latency
5. Native OTel integration with existing span infrastructure

However, LiteLLM's built-in caching could be used as a **quick win** for Phase 0 (enable Redis caching via LiteLLM config) while the custom plugin is being developed.

---

## Appendix B: Quick Win -- LiteLLM Built-In Cache

For immediate value while the custom plugin is developed, LiteLLM's built-in caching can be enabled via config:

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: "redis"
    host: "redis"
    port: 6379
    ttl: 3600
    supported_call_types:
      - "acompletion"
      - "completion"
```

This provides basic exact-match caching with minimal effort but lacks semantic matching, multi-tier architecture, and deep observability integration.

---

## Appendix C: Redis VSS Schema for Semantic Cache

```
FT.CREATE idx:routeiq_semantic_cache
    ON HASH
    PREFIX 1 "routeiq:semcache:"
    SCHEMA
        embedding VECTOR HNSW 6
            TYPE FLOAT32
            DIM 384
            DISTANCE_METRIC COSINE
        exact_cache_key TAG
        model TAG
        temperature_bucket TAG
        tools_hash TAG
        response_format TAG
        created_at NUMERIC SORTABLE
        content_preview TEXT

HSET routeiq:semcache:{uuid}
    embedding {384-dim float32 blob}
    exact_cache_key routeiq:cache:v1:{hash}
    model "gpt-4"
    temperature_bucket "deterministic"
    tools_hash "none"
    response_format "text"
    created_at 1707350400
    content_preview "What is Python? Python is a..."

EXPIRE routeiq:semcache:{uuid} 3600

FT.SEARCH idx:routeiq_semantic_cache
    "(@model:{gpt\-4} @temperature_bucket:{deterministic})
     =>[KNN 5 @embedding $query_vec AS score]"
    PARAMS 2 query_vec {384-dim float32 blob}
    RETURN 3 exact_cache_key score content_preview
    SORTBY score ASC
    LIMIT 0 1
    DIALECT 2
```

---

End of report.
