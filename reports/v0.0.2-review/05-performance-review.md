# RouteIQ AI Gateway v0.0.2 -- Performance & Scalability Review

**Review date:** 2026-02-07
**Reviewer:** Claude Opus 4.6 (automated code analysis)
**Scope:** Hot path latency, caching, plugin overhead, protocol handlers,
conversation affinity, observability, memory, scalability ceilings
**Method:** Static analysis of 18 source files; no runtime profiling

---

## Executive Summary -- Top 5 Performance Concerns

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| 1 | **CRITICAL** | `semantic_cache.py` | `SentenceTransformer.encode()` is synchronous CPU-bound (~15-50 ms) called from async context -- blocks the event loop |
| 2 | **CRITICAL** | `semantic_cache.py` / `RedisCacheStore` | Brute-force `SCAN` + pure-Python cosine similarity over all cached embeddings -- O(N) per request |
| 3 | **HIGH** | `router_decision_callback.py` | `RouterDecisionMiddleware` extends `BaseHTTPMiddleware` -- buffers entire response body, breaks SSE/streaming |
| 4 | **HIGH** | `a2a_gateway.py` | `threading.RLock` used in async context for `A2ATaskStore` -- blocks event loop thread during contention |
| 5 | **HIGH** | `conversation_affinity.py` | `asyncio.Lock` acquired on every `get_affinity()` and `record_response()` with no Redis reconnection after failure |

---

## Hot Path Latency Budget

Middleware chain order (outermost to innermost):

```
Client -> RequestIDMiddleware -> PolicyMiddleware -> PluginMiddleware
       -> RouterDecisionMiddleware -> LiteLLM Router -> [LLM Provider]
       -> PluginCallbackBridge (pre_call -> success/failure)
       -> BackpressureMiddleware (wraps ASGI app directly)
```

### Estimated per-request latency contribution (non-LLM overhead)

| Layer | Estimated Latency | Blocking? | Notes |
|-------|-------------------|-----------|-------|
| `RequestIDMiddleware` | < 0.01 ms | No | UUID generation + header set |
| `PolicyMiddleware` | 0.01-0.5 ms | No | JSON policy rule evaluation |
| `PluginMiddleware.from_scope()` | 0.01-0.05 ms | No | Header parsing into dict |
| `PluginMiddleware` on_request hooks | 0.05-2 ms | No | Depends on plugin count |
| `RouterDecisionMiddleware` | 0.1-5 ms | **Yes** | `await request.body()` reads full payload; `BaseHTTPMiddleware` buffers response |
| `BackpressureMiddleware` | 0.01-0.1 ms | No | Semaphore acquire/release |
| **PluginCallbackBridge** pre_call | 0.5-50 ms | **Partially** | Sequential: guardrails + cache lookup + cost init |
| -- PromptInjectionGuard | 0.1-2 ms | No | 16 regex patterns x N messages |
| -- PIIGuard | 0.1-3 ms | No | 5 entity types x N messages |
| -- ContentFilter | 0.05-1 ms | No | Keyword density + pattern matching |
| -- SemanticCachePlugin (hit) | 0.5-50 ms | **Yes** | Exact key gen + L1 lookup + L2 Redis + semantic scan |
| -- SemanticCachePlugin (miss) | 0.1-1 ms | No | Key gen + L1 miss + L2 miss, embedding deferred |
| **PluginCallbackBridge** success | 1-60 ms | **Yes** | Cache store (with embedding) + cost tracking |
| -- SemanticCachePlugin store | 1-50 ms | **Yes** | `embedder.encode()` if semantic enabled |
| -- CostTracker | 0.1-1 ms | No | `litellm.completion_cost()` + 5 metrics + 9 span attrs |
| **Total non-LLM overhead** | **~3-120 ms** | | Range depends on cache mode and plugin count |

---

## Memory Sizing Estimates

| Component | Memory Per Unit | Scale Factor | Estimate at 1K concurrent |
|-----------|----------------|--------------|---------------------------|
| SentenceTransformer model (`all-MiniLM-L6-v2`) | ~90 MB | Fixed (one instance) | 90 MB |
| InMemoryCache (L1 LRU) | ~2-10 KB per entry | `max_size` (default 1000) | 2-10 MB |
| Embedding vectors (384-dim float32) | ~1.5 KB per entry | Entries in Redis | 1.5 MB per 1K entries |
| A2ATaskStore (in-memory dict) | ~1-5 KB per task | Active tasks | 1-5 MB per 1K tasks |
| ConversationAffinityStore | ~0.5-2 KB per session | Active sessions | 0.5-2 MB per 1K sessions |
| CircuitBreaker `_failures` deque | ~100 bytes per failure | Breakers x window | < 1 MB |
| RouterDecisionCallback `_start_times` | ~100 bytes per call | In-flight requests | < 0.1 MB |
| OTel BatchSpanProcessor buffer | ~2-5 KB per span | `max_queue_size` (2048 default) | 4-10 MB |
| OTel MetricReader buffers | ~1-2 KB per metric series | ~10 instruments x cardinality | 1-5 MB |
| FastAPI/Starlette per-request state | ~5-20 KB per request | Concurrent requests | 5-20 MB at 1K |
| **Baseline (no requests)** | | | **~100-120 MB** |
| **Under load (1K concurrent)** | | | **~130-170 MB** |
| **With large L1 cache + semantic** | | | **~160-250 MB** |

---

## Detailed Findings

### Part 1: Hot Path Analysis

#### FINDING 1.1 -- `RouterDecisionMiddleware` uses `BaseHTTPMiddleware` [HIGH]

**File:** `src/litellm_llmrouter/router_decision_callback.py`
**Severity:** HIGH
**Impact:** Breaks streaming responses; adds latency by buffering entire response body

`RouterDecisionMiddleware` extends Starlette's `BaseHTTPMiddleware`, which wraps
the response in `StreamingResponse` and consumes the entire async generator before
sending. This has two consequences:

1. **Streaming SSE responses are fully buffered** before the first byte reaches the
   client, defeating the purpose of streaming and dramatically increasing TTFB.
2. **Memory usage scales with response size** -- a 100 KB completion is held entirely
   in memory before forwarding.

The middleware also calls `await request.body()` to extract the model name from the
JSON payload. This reads the entire request body into memory synchronously within
the middleware chain.

Contrast with `PluginMiddleware` and `BackpressureMiddleware`, which correctly use
pure ASGI implementation to preserve streaming.

```python
# router_decision_callback.py -- problematic pattern
class RouterDecisionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = await request.body()  # reads entire request body
        ...
        response = await call_next(request)  # buffers entire response
```

**Recommendation:** Rewrite as pure ASGI middleware (same pattern as `PluginMiddleware`).
Extract model from a parsed scope/header rather than reading the body.

---

#### FINDING 1.2 -- Sequential plugin execution in callback bridge [MEDIUM]

**File:** `src/litellm_llmrouter/gateway/plugin_callback_bridge.py`
**Severity:** MEDIUM
**Impact:** Cumulative latency grows linearly with plugin count

`PluginCallbackBridge.async_log_pre_api_call()` iterates plugins sequentially:

```python
for plugin in self._plugins:
    result = await plugin.on_llm_pre_call(...)
```

With the default plugin set (PromptInjectionGuard + PIIGuard + ContentFilter +
SemanticCachePlugin + CostTracker), each request pays the sum of all plugin
latencies. If any plugin blocks (e.g., `embedder.encode()`), all subsequent
plugins wait.

Guardrail plugins are independent and could theoretically run concurrently via
`asyncio.gather()`. However, ordering matters for the cache plugin (must run
after guardrails to avoid caching blocked requests), so a phased approach would
be needed: run guardrails concurrently, then run cache/cost sequentially.

---

#### FINDING 1.3 -- Backpressure semaphore + lock on every request [LOW]

**File:** `src/litellm_llmrouter/resilience.py`
**Severity:** LOW
**Impact:** Negligible under normal load; could cause contention at saturation

`BackpressureMiddleware` acquires an `asyncio.Semaphore` and checks `DrainManager`
(which holds an `asyncio.Lock`) on every request. Under normal load this is
sub-microsecond. At saturation (semaphore near capacity), rejected requests still
acquire the lock to check drain state before returning 503.

The `CircuitBreaker` acquires `asyncio.Lock` on `execute()`, `record_success()`,
`record_failure()`, and `allow_request()` -- potentially 4 lock acquisitions per
circuit-protected operation. For services with multiple circuit breakers, this
creates per-breaker serialization.

---

### Part 2: Caching Performance

#### FINDING 2.1 -- `SentenceTransformer.encode()` blocks the event loop [CRITICAL]

**File:** `src/litellm_llmrouter/gateway/plugins/cache_plugin.py`
**Severity:** CRITICAL
**Impact:** 15-50 ms event loop stall per semantic cache store operation

When semantic caching is enabled and a cache miss occurs, the success handler
computes an embedding synchronously:

```python
# cache_plugin.py -- in on_llm_success
embedding = self._embedder.encode(cache_text)  # BLOCKS EVENT LOOP
```

`SentenceTransformer.encode()` for `all-MiniLM-L6-v2` takes ~15-50 ms on CPU.
During this time, **all other coroutines on the event loop are starved** -- no
other requests can make progress. At 100 RPS, this means ~1.5-5 seconds of
cumulative blocking per second.

The embedder is loaded lazily with `threading.Lock` (not `asyncio.Lock`):

```python
# cache_plugin.py
self._embedder_lock = threading.Lock()
def _get_embedder(self):
    with self._embedder_lock:  # blocks event loop thread
        if self._embedder is None:
            self._embedder = SentenceTransformer(...)
```

First-load is ~2-5 seconds (model download + initialization), during which the
event loop is completely blocked.

**Recommendation:** Run `embedder.encode()` in a thread pool via
`asyncio.get_event_loop().run_in_executor(None, embedder.encode, text)`.
Replace `threading.Lock` with lazy init behind `asyncio.Lock` or use a
one-shot `asyncio.Event`.

---

#### FINDING 2.2 -- Brute-force cosine similarity in Redis cache [CRITICAL]

**File:** `src/litellm_llmrouter/semantic_cache.py`
**Severity:** CRITICAL
**Impact:** O(N) scan of all cached entries per semantic lookup; pure-Python vector math

`RedisCacheStore.get_similar()` implements semantic search as:

1. `SCAN` all keys matching the prefix pattern (fetches all cache keys)
2. For each key, `HGET` the stored embedding and deserialize with `json.loads()`
3. Compute `_cosine_similarity()` between query embedding and stored embedding
4. Return best match above threshold

```python
# semantic_cache.py
def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    ...
```

This is a pure-Python loop over 384-dimensional float vectors. For N cached entries:
- N Redis round-trips (or pipeline, but code shows individual calls)
- N x 384 float multiplications + additions (pure Python, no NumPy)
- O(N) time complexity with no indexing

At 10K cached entries, this could take **100+ ms per semantic lookup**.

The `InMemoryCache.get_similar()` has the same O(N) brute-force pattern but
avoids Redis round-trips.

**Recommendation:** Use Redis Vector Search (RediSearch module with HNSW index)
or a dedicated vector store. For in-memory, use NumPy vectorized operations or
FAISS for approximate nearest neighbor search.

---

#### FINDING 2.3 -- Cache key generation overhead [LOW]

**File:** `src/litellm_llmrouter/semantic_cache.py`
**Severity:** LOW
**Impact:** ~0.01-0.1 ms per request (negligible)

`CacheKeyGenerator.exact_key()` serializes the request with `json.dumps(sort_keys=True)`
and computes `hashlib.sha256()`. For a typical chat completion request (~1-5 KB JSON),
this is well under 0.1 ms. The `_normalize_messages()` step iterates messages and
strips whitespace -- also negligible.

No concern here; included for completeness.

---

#### FINDING 2.4 -- L1 InMemoryCache has no async locking [MEDIUM]

**File:** `src/litellm_llmrouter/semantic_cache.py`
**Severity:** MEDIUM
**Impact:** Potential race conditions under concurrent writes; data corruption unlikely but possible

`InMemoryCache` uses `OrderedDict` for LRU eviction but has no locking mechanism:

```python
class InMemoryCache:
    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
```

`get()` and `set()` both mutate `_cache` (LRU reordering via `move_to_end()`,
eviction via `popitem()`). With concurrent async operations, interleaved access
could corrupt the OrderedDict's internal linked list.

In practice, since Python's GIL protects individual bytecode operations and
async code yields only at `await` points, corruption is unlikely but not
impossible if `set()` triggers eviction between another coroutine's `get()` steps.

**Recommendation:** Add an `asyncio.Lock` around `get()`/`set()` operations, or
use a thread-safe LRU implementation.

---

### Part 3: Plugin System Overhead

#### FINDING 3.1 -- Guardrail regex scanning scales with message count [MEDIUM]

**Files:**
- `src/litellm_llmrouter/gateway/plugins/prompt_injection_guard.py`
- `src/litellm_llmrouter/gateway/plugins/pii_guard.py`
- `src/litellm_llmrouter/gateway/plugins/content_filter.py`

**Severity:** MEDIUM
**Impact:** 0.3-6 ms per request with typical message counts; grows linearly with conversation length

**Prompt Injection Guard:** 16 compiled regex patterns checked against each user
message. For a 5-message conversation, that is 80 regex evaluations. Patterns use
`re.IGNORECASE` and are pre-compiled (good). Most patterns are simple string
matches that the regex engine handles efficiently. Estimated: 0.1-2 ms for 5
messages.

**PII Guard:** 5 entity types (SSN, credit card, email, phone, IP) with pre-compiled
regexes at module level. `scan_pii()` iterates all entity types per message and
collects findings via `finditer()`. The `redact_text()` method processes findings
in reverse offset order with string slicing (creates new string per replacement).
Estimated: 0.1-3 ms for 5 messages.

**Content Filter:** 5 categories with ~20 keywords + ~3 regex patterns each.
`_score_content()` does `keyword.lower() in text_lower` for each keyword
(substring search). This is O(keywords x text_length) per message but Python's
`in` operator uses Boyer-Moore-Horspool, so practical performance is better.
Estimated: 0.05-1 ms for 5 messages.

**Concern:** For long conversations (20+ messages with full history), cumulative
scanning time could reach 10-20 ms. Guardrails scan the full message list on
every request, including previously-scanned messages from conversation history.

**Recommendation:** Consider scanning only new/delta messages when conversation
history is provided, or caching scan results keyed by message content hash.

---

#### FINDING 3.2 -- Guardrail OTel attribute emission [LOW]

**File:** `src/litellm_llmrouter/gateway/plugins/guardrails_base.py`
**Severity:** LOW
**Impact:** 5 span attributes per guardrail evaluation (negligible)

Each guardrail evaluation emits 5 OTel span attributes via `_emit_otel_attributes()`:
`guardrail.name`, `guardrail.action`, `guardrail.triggered`, `guardrail.score`,
`guardrail.evaluation_time_ms`. With 3 guardrails, that is 15 attributes per
request. OTel attribute setting is O(1) per attribute -- no concern.

---

#### FINDING 3.3 -- CostTracker metric cardinality [LOW]

**File:** `src/litellm_llmrouter/gateway/plugins/cost_tracker.py`
**Severity:** LOW
**Impact:** Metric cardinality grows with unique model names; bounded in practice

CostTracker records 5 metrics per successful completion, each tagged with `model`
label. With M distinct model names, this creates M metric series per instrument
(5M total). For typical deployments (5-20 models), this is well within OTel
limits. At 100+ models, metric cardinality could strain exporters.

`litellm.completion_cost()` is called synchronously but is a pure computation
(dictionary lookup + multiplication) -- negligible latency.

---

### Part 4: Protocol Handlers

#### FINDING 4.1 -- MCP tools_list builds full list before paginating [MEDIUM]

**File:** `src/litellm_llmrouter/mcp_jsonrpc.py`
**Severity:** MEDIUM
**Impact:** O(servers x tools) computation per paginated page request

`_handle_tools_list()` iterates all registered MCP servers, collects all tools
into a single list, then applies cursor-based pagination (Base64-encoded offset).
With S servers and T tools per server, each page request does O(S*T) work
regardless of the requested page size.

The cursor encoding/decoding itself (Base64 of JSON `{"offset": N}`) is trivial.

**Recommendation:** Cache the aggregated tool list with invalidation on server
registration changes.

---

#### FINDING 4.2 -- A2A task store uses `threading.RLock` in async context [HIGH]

**File:** `src/litellm_llmrouter/a2a_gateway.py`
**Severity:** HIGH
**Impact:** Blocks event loop thread during lock contention; affects all concurrent requests

`A2ATaskStore` uses `threading.RLock` for all operations:

```python
class A2ATaskStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._tasks: dict[str, A2ATask] = {}
```

When one coroutine holds the lock (e.g., during task creation or cleanup), all
other coroutines attempting to access the task store will **block the event loop
thread** rather than yielding. This is because `threading.RLock.acquire()` is a
blocking call that does not release the GIL to the event loop.

Similarly, `A2AAgentRegistry` uses `threading.RLock` for the agent registry.

**Recommendation:** Replace `threading.RLock` with `asyncio.Lock`. The RLock
re-entrant behavior is not needed since async code does not have re-entrant
call patterns.

---

#### FINDING 4.3 -- A2A TTL cleanup iterates entire task store [LOW]

**File:** `src/litellm_llmrouter/a2a_gateway.py`
**Severity:** LOW
**Impact:** O(N) iteration under lock; negligible for typical task counts

`_cleanup_expired()` iterates all tasks to find expired ones. With the default
TTL of 3600s and moderate throughput, the active task count stays small. At very
high throughput (10K+ tasks/hour with long TTL), cleanup could hold the lock for
noticeable duration.

---

#### FINDING 4.4 -- SSE event formatting per chunk [LOW]

**File:** `src/litellm_llmrouter/a2a_gateway.py`
**Severity:** LOW
**Impact:** `json.dumps()` per SSE event (~0.01 ms); negligible

Each SSE event is formatted with `f"event: {event_type}\ndata: {json.dumps(data)}\n\n"`.
For a typical streaming response with 50-200 chunks, this adds < 1 ms total.

---

### Part 5: Conversation Affinity

#### FINDING 5.1 -- Global `asyncio.Lock` on every affinity operation [HIGH]

**File:** `src/litellm_llmrouter/conversation_affinity.py`
**Severity:** HIGH
**Impact:** Serializes all affinity lookups; bottleneck at > 500 RPS

`ConversationAffinityStore` uses a single `asyncio.Lock` for both reads and writes:

```python
async def get_affinity(self, conversation_id: str) -> str | None:
    async with self._lock:
        ...

async def record_response(self, conversation_id: str, model: str, ...) -> None:
    async with self._lock:
        ...
```

Every request that uses conversation affinity (which is every chat completion with
a conversation ID) must acquire this lock. Under high concurrency, requests queue
up behind the lock even though most operations are read-only dict lookups.

**Recommendation:** Use a read-write lock pattern, or partition the store by
conversation ID hash to reduce contention. For read-heavy workloads, consider
`asyncio.Lock`-free reads with copy-on-write semantics.

---

#### FINDING 5.2 -- No Redis reconnection after failure [HIGH]

**File:** `src/litellm_llmrouter/conversation_affinity.py`
**Severity:** HIGH
**Impact:** Permanent degradation to in-memory only after any Redis error

When a Redis operation fails, the store sets `_redis_available = False` and
never attempts reconnection:

```python
except Exception as e:
    logger.warning(f"Redis affinity failed, falling back to memory: {e}")
    self._redis_available = False
```

After a transient Redis failure (network blip, timeout), the system permanently
falls back to in-memory storage. In a multi-instance deployment, this means
conversation affinity is lost across instances with no recovery path short of
restarting the process.

**Recommendation:** Implement exponential backoff reconnection, or periodically
retry Redis operations (e.g., every 30 seconds) to detect recovery.

---

#### FINDING 5.3 -- Background cleanup iterates entire store under lock [MEDIUM]

**File:** `src/litellm_llmrouter/conversation_affinity.py`
**Severity:** MEDIUM
**Impact:** Lock held during full store iteration every 60 seconds

The cleanup task runs every 60 seconds and iterates the entire `_store` dict
under the global lock to find expired entries. During cleanup, no other
affinity operations can proceed. With 100K+ active conversations, this could
hold the lock for several milliseconds.

---

### Part 6: Observability Overhead

#### FINDING 6.1 -- OTel span attribute volume per request [LOW]

**File:** `src/litellm_llmrouter/observability.py`, `router_decision_callback.py`, `cost_tracker.py`
**Severity:** LOW
**Impact:** 30-50 span attributes per request; within OTel limits but adds serialization cost

Per request, the following attributes are set:
- `RouterDecisionCallback`: 6+ attributes (model, strategy, selected_deployment, etc.)
- `set_router_decision_attributes()`: up to 11 attributes
- Guardrails: 15 attributes (5 per guardrail x 3 guardrails)
- CostTracker: 9 attributes
- Total: ~40-50 attributes per span

`BatchSpanProcessor` buffers spans (default `max_queue_size=2048`) and exports
in batches. With 40-50 attributes per span at ~5 KB per span, the buffer
consumes ~10 MB at capacity. Export to an OTel Collector is typically async
and non-blocking.

---

#### FINDING 6.2 -- Metric recording frequency [LOW]

**File:** `src/litellm_llmrouter/metrics.py`, `cost_tracker.py`
**Severity:** LOW
**Impact:** ~15-20 metric recordings per request; standard OTel overhead

Per request:
- CostTracker: 5 metric recordings
- GatewayMetrics: active request gauge (up/down), duration histogram, token histograms
- RouterDecisionCallback: routing duration, strategy counter
- Total: ~15-20 recordings

`PeriodicExportingMetricReader` aggregates locally and exports every 60 seconds.
Individual recordings are O(1) (histogram bucket increment or counter add).
No concern.

---

#### FINDING 6.3 -- `RouterDecisionCallback._start_times` potential memory leak [MEDIUM]

**File:** `src/litellm_llmrouter/router_decision_callback.py`
**Severity:** MEDIUM
**Impact:** Unbounded dict growth if LLM calls fail without triggering success/failure callbacks

`RouterDecisionCallback` stores request start times in a dict keyed by
`litellm_call_id`:

```python
self._start_times: dict[str, float] = {}
```

Entries are added in `async_log_pre_api_call()` and removed in
`async_log_success_event()` or `async_log_failure_event()`. If a request is
dropped without triggering either callback (e.g., connection reset, timeout
at the transport layer), the entry remains forever.

**Recommendation:** Add periodic cleanup of entries older than a threshold
(e.g., 5 minutes), or use a TTL dict.

---

### Part 7: Memory Profiling Estimates

See the Memory Sizing Estimates table above for detailed breakdown.

**Key observations:**

1. **SentenceTransformer dominates baseline memory** at ~90 MB. If semantic caching
   is disabled, baseline drops to ~30 MB.

2. **L1 cache memory is bounded** by `max_size` parameter (default 1000 entries).
   At ~5 KB average entry size (response text + metadata), max is ~5 MB.

3. **OTel buffers are the second-largest consumer** at ~10-15 MB for span + metric
   buffers. This is fixed regardless of load.

4. **Per-request memory is modest** at ~5-20 KB (request/response objects, middleware
   state, plugin context). At 1K concurrent requests, this is ~5-20 MB.

5. **In-memory stores (task store, affinity, circuit breakers) are unbounded in theory**
   but have TTL cleanup. The risk is during TTL-cleanup-lock contention when stores
   grow large.

**Memory scaling pattern:**
```
Total = 90 MB (model) + 10 MB (OTel) + 5 MB (L1 cache) + N * 15 KB (concurrent)
     = ~105 MB baseline + 15 KB per concurrent request
```

For 256 MB container: safe up to ~10K concurrent requests (without semantic cache model)
For 512 MB container: safe up to ~10K concurrent requests (with semantic cache model)
For 1 GB container: comfortable headroom for all components

---

### Part 8: Scalability Bottlenecks -- Ceiling Analysis

#### Bottleneck 1: Single-process architecture [ARCHITECTURAL]

RouteIQ is constrained to **1 uvicorn worker** because LiteLLM Router monkey-patches
do not survive `os.execvp()`. This means:

- **CPU-bound work cannot be parallelized** across cores
- **All requests share one event loop** -- any blocking call affects all requests
- **Vertical scaling is limited** to single-core throughput
- **Horizontal scaling requires multiple container instances** with Redis/Postgres for
  shared state

At ~1000-2000 RPS (depending on LLM response times and plugin overhead), the
single event loop becomes the bottleneck.

---

#### Bottleneck 2: Semantic cache scan is O(N) [DATA STRUCTURE]

The brute-force cosine similarity scan grows linearly with cache size. Projected
performance:

| Cache entries | Estimated semantic lookup time |
|---------------|-------------------------------|
| 100 | 5-10 ms |
| 1,000 | 50-100 ms |
| 10,000 | 500 ms - 1 s |
| 100,000 | 5-10 s (unusable) |

This becomes the dominant latency source well before other bottlenecks manifest.

---

#### Bottleneck 3: Conversation affinity lock contention [CONCURRENCY]

With a single `asyncio.Lock` serializing all affinity operations, throughput is
bounded by the lock hold time multiplied by concurrency:

- Lock hold time: ~0.01 ms (dict lookup) to ~1 ms (Redis round-trip)
- At 0.01 ms hold time: theoretical max ~100K ops/sec (no contention)
- At 1 ms hold time (Redis): theoretical max ~1K ops/sec
- With N concurrent requesters: effective throughput = 1/hold_time

This becomes a bottleneck when Redis is involved and concurrency exceeds ~500.

---

#### Bottleneck 4: In-memory stores are single-instance [ARCHITECTURE]

`InMemoryCache`, `A2ATaskStore`, `ConversationAffinityStore` (when Redis is down),
and `_start_times` are all in-process dicts. In a multi-instance deployment:

- L1 cache has no cross-instance coherence (cold start on new instances)
- A2A tasks are invisible to other instances
- Conversation affinity falls back to per-instance after Redis failure
- No cache warming or replication mechanism

---

#### Bottleneck 5: `BaseHTTPMiddleware` response buffering [PROTOCOL]

`RouterDecisionMiddleware`'s use of `BaseHTTPMiddleware` means streaming responses
are buffered. For a streaming completion that produces 500 tokens over 10 seconds:

- Without buffering: client sees first token in ~200 ms (TTFT)
- With buffering: client sees first token in ~10 s (after full completion)

This effectively negates the benefit of streaming for any request that passes
through `RouterDecisionMiddleware`.

---

## Optimization Recommendations

### Quick Wins (< 1 day effort)

| # | Change | Expected Impact | Files |
|---|--------|-----------------|-------|
| QW-1 | Run `embedder.encode()` in thread pool executor | Unblocks event loop; ~0 ms stall instead of 15-50 ms | `cache_plugin.py` |
| QW-2 | Replace `threading.Lock` with `asyncio.Lock` for embedder init | Prevents event loop block on first load | `cache_plugin.py` |
| QW-3 | Replace `threading.RLock` with `asyncio.Lock` in A2A stores | Prevents event loop block during contention | `a2a_gateway.py` |
| QW-4 | Add Redis reconnection with exponential backoff | Recovers from transient Redis failures | `conversation_affinity.py` |
| QW-5 | Add TTL cleanup to `_start_times` dict | Prevents memory leak on dropped requests | `router_decision_callback.py` |

### Medium Effort (1-3 day effort)

| # | Change | Expected Impact | Files |
|---|--------|-----------------|-------|
| ME-1 | Rewrite `RouterDecisionMiddleware` as pure ASGI | Restores streaming; reduces memory; removes body buffering | `router_decision_callback.py` |
| ME-2 | Add `asyncio.Lock` to `InMemoryCache` | Prevents race conditions under concurrent access | `semantic_cache.py` |
| ME-3 | Cache aggregated MCP tool list | O(1) pagination instead of O(S*T) per page | `mcp_jsonrpc.py` |
| ME-4 | Partition affinity store lock by conversation hash | Reduces lock contention proportionally to partition count | `conversation_affinity.py` |
| ME-5 | Scan only new messages in guardrails | Reduces scan time for long conversations from O(N) to O(1) | `prompt_injection_guard.py`, `pii_guard.py`, `content_filter.py` |

### Large Effort (3+ day effort)

| # | Change | Expected Impact | Files |
|---|--------|-----------------|-------|
| LE-1 | Replace brute-force cosine with Redis Vector Search (RediSearch HNSW) | O(log N) semantic lookup instead of O(N) | `semantic_cache.py` |
| LE-2 | Use NumPy for in-memory cosine similarity | 10-100x speedup for vector math | `semantic_cache.py` |
| LE-3 | Implement cross-instance L1 cache coherence (pub/sub invalidation) | Consistent cache across instances | `semantic_cache.py` |
| LE-4 | Run guardrails concurrently via `asyncio.gather()` | Reduces guardrail phase from sum(latencies) to max(latency) | `plugin_callback_bridge.py` |

---

## Benchmark Suggestions for v0.0.2

### Recommended benchmarks

1. **Hot path baseline** -- Measure request latency with all plugins disabled.
   Use `wrk` or `hey` against `/v1/chat/completions` with a mock LLM backend
   (fixed 100 ms response). Target: < 5 ms gateway overhead at p99.

2. **Guardrail throughput** -- Measure per-request latency with guardrails enabled
   at varying conversation lengths (1, 5, 20, 50 messages). Verify linear scaling.

3. **Semantic cache hit/miss** -- Measure latency for exact cache hits, semantic
   cache hits, and misses separately. Vary cache size from 100 to 10K entries
   to quantify O(N) degradation.

4. **Streaming TTFT** -- Compare time-to-first-token with and without
   `RouterDecisionMiddleware`. Expect significant regression with current
   `BaseHTTPMiddleware` implementation.

5. **Concurrent affinity** -- Load test conversation affinity with 1K concurrent
   conversations, measuring p50/p99 lock wait time.

6. **Memory profiling** -- Run under `memray` or `tracemalloc` with 1K concurrent
   requests for 5 minutes. Verify no unbounded growth in `_start_times`,
   task store, or affinity store.

7. **Event loop blocking** -- Use `asyncio` debug mode (`PYTHONASYNCIODEBUG=1`)
   to detect coroutines that block > 100 ms. Expect hits on `embedder.encode()`
   and `threading.Lock` acquisitions.

---

*End of performance review. All findings are based on static code analysis of
RouteIQ v0.0.2 source. Runtime benchmarks are recommended to validate estimates.*
