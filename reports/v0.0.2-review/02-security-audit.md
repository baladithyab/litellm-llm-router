# RouteIQ Gateway v0.0.2 -- Security Audit Report

**Date**: 2026-02-07
**Auditor**: Claude Code (Security Research Agent)
**Scope**: All new v0.0.2 components -- guardrails, plugins, semantic cache, MCP, A2A, auth, routes
**Classification**: CONFIDENTIAL -- Internal Engineering Use Only

---

## Executive Summary

RouteIQ Gateway v0.0.2 introduces a substantial security surface through new guardrail plugins, plugin callback bridges, semantic caching, and expanded MCP/A2A protocol support. The overall security posture is **reasonable for a pre-production release** with several design choices reflecting security awareness (fail-closed admin auth, SSRF protection, plugin allowlisting). However, the audit identified **4 High-severity**, **9 Medium-severity**, and **6 Low-severity** findings that require remediation before production deployment.

| Severity | Count | Category |
|----------|-------|----------|
| Critical | 0 | -- |
| High | 4 | Memory DoS, SSRF fallback, data exposure, dependency RCE |
| Medium | 9 | Regex bypass, missing rate limits, error info leakage, PII gaps |
| Low | 6 | Config parsing, minor info leaks, hardening opportunities |

The most significant risks are: (1) unbounded in-memory stores in A2A task storage allowing memory exhaustion DoS, (2) SSRF protection graceful fallback that silently degrades to a no-op, (3) full LLM responses cached to Redis without PII scrubbing, and (4) upstream LiteLLM CVEs inherited by the gateway.

---

## OWASP LLM Top 10 Coverage Matrix

| ID | Threat | Coverage | Component | Gaps |
|----|--------|----------|-----------|------|
| LLM01 | Prompt Injection | Partial | `prompt_injection_guard.py` | Regex-only; no ML-based detection; limited pattern set; trivially bypassable via encoding/obfuscation (Finding SEC-05) |
| LLM02 | Insecure Output Handling | Partial | `content_filter.py`, `pii_guard.py` | Output scanning is warn-only for PII; content filter on output only logs; no output sanitization/encoding |
| LLM03 | Training Data Poisoning | Not Applicable | -- | Gateway does not train models; semantic cache model loading is a supply-chain concern (Finding SEC-16) |
| LLM04 | Model DoS | Partial | `resilience.py` backpressure | Backpressure middleware exists; no per-request token/cost limits; A2A task store unbounded (Finding SEC-01) |
| LLM05 | Supply Chain | Partial | Plugin allowlist, capability policy | Plugin system has allowlist + capability gating; sentence-transformers model loading has RCE risk (Finding SEC-16); LiteLLM upstream CVEs (Finding SEC-14) |
| LLM06 | Sensitive Info Disclosure | Partial | `pii_guard.py`, `auth.py` secret scrubbing | PII guard covers 5 entity types; no address/name/DOB; semantic cache stores full responses without PII scrub (Finding SEC-03); OTel spans may capture PII (Finding SEC-13) |
| LLM07 | Insecure Plugin Design | Good | `plugin_manager.py` | Plugin security: allowlist, capability policy, SSRF context, dependency resolution, failure modes; dynamic loading via `importlib` is inherent risk |
| LLM08 | Excessive Agency | Partial | MCP tool invocation gated | MCP tool invocation disabled by default; A2A agent registration requires admin auth; no runtime tool call limits per session |
| LLM09 | Overreliance | N/A | -- | Application-layer concern, not gateway-scope |
| LLM10 | Model Theft | Partial | Auth gating on endpoints | Admin auth on control plane; user auth on data plane; model artifact hash verification exists |

---

## Detailed Findings

### SEC-01: A2A Task Store Unbounded Memory Growth (HIGH)

- **Severity**: HIGH
- **CWE**: CWE-400 (Uncontrolled Resource Consumption)
- **File**: `src/litellm_llmrouter/a2a_gateway.py`
- **OWASP LLM**: LLM04 (Model DoS)

**Description**: The `A2ATaskStore` uses an in-memory dictionary (`self._tasks`) with no maximum size limit. While individual tasks have TTL-based expiration (default 3600s), an attacker can create tasks faster than the periodic cleanup runs.

**Exploit Scenario**: An authenticated attacker sends thousands of `tasks/send` requests per second. Each task is stored in memory. The cleanup loop (`_cleanup_loop`) runs at intervals defined by `A2A_TASK_CLEANUP_INTERVAL` (default: 300s). In a 5-minute window, an attacker can allocate gigabytes of task objects, causing OOM and gateway crash.

**Affected Code**:
- Task creation: The store has no bounds check before inserting
- No per-client rate limiting on task creation
- `_cleanup_loop` only runs periodically, not on insertion

**Recommendation**:
1. Add a `max_tasks` limit (configurable via env var) with rejection when exceeded
2. Add per-client task creation rate limiting
3. Consider moving task storage to Redis for HA deployments

---

### SEC-02: SSRF Protection Degrades to No-Op on Import Failure (HIGH)

- **Severity**: HIGH
- **CWE**: CWE-636 (Not Failing Securely)
- **File**: `src/litellm_llmrouter/a2a_gateway.py:50-68`
- **OWASP LLM**: LLM07 (Insecure Plugin Design)

**Description**: The A2A gateway imports `validate_outbound_url` with a graceful `ImportError` fallback that replaces the validation function with a lambda that always returns the URL unchanged.

```python
try:
    from litellm_llmrouter.url_security import validate_outbound_url
except ImportError:
    validate_outbound_url = lambda url, **kwargs: url  # type: ignore
```

If `url_security.py` fails to import (e.g., missing dependency, syntax error, import cycle), all SSRF protection is silently disabled for A2A agent registration and task proxying. No warning is logged at the `WARNING` or `ERROR` level -- only a debug-level message may appear.

**Exploit Scenario**: A packaging error or dependency conflict causes `url_security` import to fail. An attacker registers an A2A agent pointing to `http://169.254.169.254/latest/meta-data/` (AWS metadata). The gateway happily proxies requests to the metadata service.

**Recommendation**:
1. Log at `ERROR` level when the import fails and set a flag
2. Refuse agent registration when SSRF validation is unavailable (fail-closed)
3. Add a startup health check that verifies url_security module is importable

---

### SEC-03: Semantic Cache Stores Full Responses Without PII Scrubbing (HIGH)

- **Severity**: HIGH
- **CWE**: CWE-312 (Cleartext Storage of Sensitive Information)
- **File**: `src/litellm_llmrouter/semantic_cache.py:445-462`
- **OWASP LLM**: LLM06 (Sensitive Info Disclosure)

**Description**: The `RedisCacheStore.set()` method stores the complete serialized LLM response in Redis without any PII detection or redaction. If a user's prompt causes the LLM to include PII in its response (e.g., summarizing a document containing SSNs, emails, or credit card numbers), that PII is persisted in Redis and served to subsequent cache-hitting requests.

**Exploit Scenario**:
1. User A asks the LLM to process a document containing PII. Response is cached with PII intact.
2. User B sends a semantically similar query. The semantic cache returns User A's cached response, including their PII.
3. PII persists in Redis for the duration of the cache TTL.

**Affected Code**:
- `RedisCacheStore.set()` (line 445): Stores `entry.response` raw
- `RedisCacheStore.get()` (line 433): Returns cached entries without filtering
- `RedisCacheStore.get_similar()` (line 465): Returns semantic matches without filtering
- No integration between `pii_guard.py` and the cache layer

**Recommendation**:
1. Run PII detection on responses before caching
2. Either redact PII before storage or skip caching when PII is detected
3. Add a `cache_pii_policy` config option: `skip` | `redact` | `allow`
4. Consider encrypting cached responses at rest in Redis

---

### SEC-04: Error Messages Leak Internal Details (MEDIUM)

- **Severity**: MEDIUM
- **CWE**: CWE-209 (Generation of Error Message Containing Sensitive Information)
- **Files**:
  - `src/litellm_llmrouter/mcp_jsonrpc.py:414,638`
  - `src/litellm_llmrouter/a2a_gateway.py:849,1217`

**Description**: Multiple error handlers include `str(e)` in responses returned to clients. While `sanitize_error_response()` exists in `auth.py` and is used in `routes.py`, the MCP JSON-RPC and A2A gateway modules construct error responses directly with exception details.

**Examples**:
```python
# mcp_jsonrpc.py line ~414
return _jsonrpc_error(req_id, -32603, f"Internal error: {str(e)}")

# a2a_gateway.py line ~849
"message": f"Internal error: {str(e)}"

# a2a_gateway.py line ~1217
"message": f"Streaming error: {str(e)}"
```

**Exploit Scenario**: Exception messages may contain file paths, database connection strings, internal hostnames, stack traces, or configuration details that aid further attacks.

**Recommendation**:
1. Use `sanitize_error_response()` consistently across all modules
2. Return generic error messages to clients; log full details server-side
3. Add a `DEBUG_ERRORS` env var that's `false` by default in production

---

### SEC-05: Prompt Injection Guard Trivially Bypassable (MEDIUM)

- **Severity**: MEDIUM
- **CWE**: CWE-693 (Protection Mechanism Failure)
- **File**: `src/litellm_llmrouter/gateway/plugins/prompt_injection_guard.py:36-59`
- **OWASP LLM**: LLM01 (Prompt Injection)

**Description**: The prompt injection guard uses 16 regex patterns with `re.IGNORECASE` for detection. These patterns are trivially bypassable through:

1. **Unicode substitution**: Replace ASCII characters with lookalikes (e.g., using Cyrillic 'e' for Latin 'e')
2. **Whitespace injection**: Insert zero-width characters between words
3. **Encoding**: Base64-encode the injection payload within a legitimate request
4. **Paraphrasing**: "Please discard everything above" instead of "ignore all previous instructions"
5. **Tokenization splitting**: Break pattern keywords across message boundaries
6. **Indirect injection**: Injection via tool call responses, not user messages (guard only checks `role == "user"`)

Additionally, the guard only evaluates string content. It does not inspect multi-part messages (vision messages with `content: [{type: "text", ...}]`), meaning injection via multi-modal content is undetected.

**Pattern Analysis**:
- Line 37: `r"ignore\s+(all\s+)?previous\s+instructions"` -- requires exact word "ignore" + "previous" + "instructions"
- Line 42: `r"you\s+are\s+now\s+(?:DAN|an?\s+unrestricted)"` -- only catches DAN/unrestricted, not other persona names
- All patterns are English-only

**Recommendation**:
1. Document that regex-based detection is a first layer, not a comprehensive solution
2. Add support for multi-part message content inspection
3. Consider integrating ML-based classifiers (e.g., rebuff, lakera) as a plugin
4. Normalize Unicode before regex matching (NFKD decomposition)
5. Extend patterns to cover paraphrased variants

---

### SEC-06: PII Guard Missing Common Entity Types (MEDIUM)

- **Severity**: MEDIUM
- **CWE**: CWE-311 (Missing Encryption of Sensitive Data)
- **File**: `src/litellm_llmrouter/gateway/plugins/pii_guard.py:56-85`
- **OWASP LLM**: LLM06 (Sensitive Info Disclosure)

**Description**: The PII guard detects 5 entity types (SSN, CREDIT_CARD, EMAIL, PHONE, IP_ADDRESS). Missing common PII types include:

- **Names**: No person name detection
- **Addresses**: No physical address detection
- **Date of Birth**: No DOB patterns
- **Passport/ID Numbers**: No government ID patterns
- **IBAN/Bank Account**: No financial account numbers beyond credit cards
- **AWS Keys**: No `AKIA*` pattern (though `auth.py` has secret scrubbing, the PII guard does not)
- **IPv6 addresses**: Only IPv4 is covered
- **Medical Record Numbers**: No HIPAA identifiers

Additionally, the CREDIT_CARD pattern (`\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b`) does not perform Luhn validation (the docstring mentions "optional Luhn check" but none is implemented), so it will match many non-card 16-digit sequences.

**Recommendation**:
1. Add AWS key pattern (`AKIA[A-Z0-9]{16}`)
2. Implement Luhn check for credit card validation to reduce false positives
3. Add IPv6 detection
4. Document the limited scope and recommend integration with dedicated PII engines (Presidio, AWS Comprehend)

---

### SEC-07: Content Filter Keyword Matching Lacks Word Boundaries (MEDIUM)

- **Severity**: MEDIUM
- **CWE**: CWE-185 (Incorrect Regular Expression)
- **File**: `src/litellm_llmrouter/gateway/plugins/content_filter.py:272-277`
- **OWASP LLM**: LLM02 (Insecure Output Handling)

**Description**: The content filter's keyword matching uses `if keyword.lower() in text_lower` (substring match), which produces false positives for short keywords that appear inside legitimate words:

- "kill" matches "skills", "overkill", "killed" (legitimate use)
- "gun" matches "Gunderson", "begun"
- "bomb" matches "bombastic"
- "attack" matches "counterattack", "heart attack"
- "stab" matches "establish", "stability"
- "hack" matches "hackathon"
- "cut" matches "execute", "shortcut"

The scoring function (line 287: `keyword_score = min(len(matched_keywords) / max(word_count * 0.1, 1), 0.5)`) amplifies false positives because multiple substring matches in a long technical document can easily exceed the threshold.

**Recommendation**:
1. Use word-boundary regex (`\bkill\b`) instead of substring matching for short keywords
2. Add contextual scoring that considers surrounding words
3. Maintain separate "exact" vs "partial" keyword lists

---

### SEC-08: No Request Body Size Limit on JSON-RPC Endpoint (MEDIUM)

- **Severity**: MEDIUM
- **CWE**: CWE-770 (Allocation of Resources Without Limits)
- **File**: `src/litellm_llmrouter/mcp_jsonrpc.py`
- **OWASP LLM**: LLM04 (Model DoS)

**Description**: The MCP JSON-RPC endpoint at `/mcp` accepts POST requests with JSON bodies. There is no explicit content-length limit enforced at the application level. While ASGI servers (uvicorn) may have default body size limits, these are typically generous (e.g., no limit in uvicorn by default).

An attacker can send a multi-gigabyte JSON payload that will be parsed in memory, potentially causing OOM.

**Recommendation**:
1. Add a `max_body_size` check before parsing JSON-RPC requests
2. Configure uvicorn's `--limit-concurrency` and body size limits
3. Consider adding this to the backpressure middleware

---

### SEC-09: DNS Resolution Timeout Allows Request Through (MEDIUM)

- **Severity**: MEDIUM
- **CWE**: CWE-636 (Not Failing Securely)
- **File**: `src/litellm_llmrouter/url_security.py:925-930`

**Description**: When async DNS resolution times out, the SSRF validator allows the request through:

```python
except asyncio.TimeoutError:
    verbose_proxy_logger.warning(
        f"SSRF: Async DNS resolution timed out for {hostname} "
        f"after {dns_timeout}s, allowing"
    )
    pass
```

An attacker could use a DNS server that intentionally delays responses beyond the timeout (default 5s) to bypass SSRF protection. After the validation passes, the actual HTTP client will resolve DNS separately and connect to the internal target.

**Recommendation**:
1. Change to fail-closed: block requests when DNS resolution times out
2. Or: ensure the actual HTTP client uses the same resolved IPs as the validator (pin DNS resolution)

---

### SEC-10: Admin Auth Bypass When Disabled (MEDIUM)

- **Severity**: MEDIUM
- **CWE**: CWE-287 (Improper Authentication)
- **File**: `src/litellm_llmrouter/auth.py:206-212`

**Description**: Setting `ADMIN_AUTH_ENABLED=false` bypasses all admin authentication and returns `{"admin_key": "__disabled__"}`. While the code logs a warning, there is no startup-time warning or health check flag. In a misconfigured deployment, this could expose all control-plane endpoints without authentication.

**Recommendation**:
1. Emit a prominent startup-time WARNING log when admin auth is disabled
2. Include `admin_auth_disabled` in the `/_health/ready` response
3. Consider requiring an additional confirmation env var (e.g., `ADMIN_AUTH_DISABLE_CONFIRM=yes`)

---

### SEC-11: Plugin Dynamic Import Without Module Path Validation (MEDIUM)

- **Severity**: MEDIUM
- **CWE**: CWE-94 (Improper Control of Generation of Code)
- **File**: `src/litellm_llmrouter/gateway/plugin_manager.py:723-727`

**Description**: The plugin manager uses `importlib.import_module(module_path)` to load plugins specified in `LLMROUTER_PLUGINS`. While an allowlist exists, it is optional. Without the allowlist, any Python module accessible on the `PYTHONPATH` can be loaded and instantiated, including modules that execute code on import.

**Exploit Scenario**: An attacker with access to the environment variable (e.g., via a container orchestration misconfiguration) sets `LLMROUTER_PLUGINS` to a path pointing to a malicious module.

**Recommendation**:
1. Enforce the allowlist by default in production (fail-closed if no allowlist is configured)
2. Validate that plugin module paths match a safe pattern (e.g., must start with `litellm_llmrouter.gateway.plugins.`)
3. Add integrity verification for plugin modules (hash check)

---

### SEC-12: Guardrail Block Error Exposes Pattern Match Details (MEDIUM)

- **Severity**: MEDIUM
- **CWE**: CWE-209 (Error Message Info Leak)
- **File**: `src/litellm_llmrouter/gateway/plugins/guardrails_base.py:210-215`

**Description**: When a guardrail blocks a request, the error message includes the matched pattern label:

```python
raise GuardrailBlockError(
    guardrail_name=decision.guardrail_name,
    category=decision.category,
    message=decision.details.get("reason", "Blocked by guardrail"),
    score=decision.score,
)
```

The `reason` field contains strings like `"Prompt injection detected: ignore_previous_instructions"` which reveals exactly which pattern was matched. This information helps an attacker craft bypass variations.

**Recommendation**:
1. Return a generic message to the client: "Request blocked by content security policy"
2. Log the detailed reason server-side only
3. Include a `request_id` for correlation between client error and server log

---

### SEC-13: OTel Spans May Capture PII in Guardrail Attributes (LOW)

- **Severity**: LOW
- **CWE**: CWE-532 (Insertion of Sensitive Information into Log File)
- **File**: `src/litellm_llmrouter/gateway/plugins/guardrails_base.py:259-274`

**Description**: The guardrail base class emits OTel span attributes including `guardrail.name`, `guardrail.action`, `guardrail.category`, and `guardrail.score`. While it does not emit the full text or matched content, the `_log_decision()` method (line 276-288) logs `decision.details` which may contain matched PII patterns or prompt text.

The PII guard's decision details include `entity_types` and `count` (not the actual PII), which is acceptable. However, the content filter logs `matched_keywords` and `matched_patterns` to the logger, which could contain sensitive context.

**Recommendation**:
1. Ensure OTel attributes never include raw matched text or PII
2. Review log output from guardrail decisions for sensitive content
3. Apply OTel redaction processors in the collector config

---

### SEC-14: Callback Bridge Logs Exception Details with `exc_info=True` (LOW)

- **Severity**: LOW
- **CWE**: CWE-532 (Insertion of Sensitive Information into Log File)
- **File**: `src/litellm_llmrouter/gateway/plugin_callback_bridge.py:113-116`

**Description**: The callback bridge logs plugin failures with `exc_info=True`, which includes full stack traces in logs. These traces may contain request data, API keys (if passed as function arguments), or internal file paths.

```python
logger.error(
    f"Plugin '{plugin.name}' on_llm_pre_call failed: {e}",
    exc_info=True,
)
```

**Recommendation**:
1. Scrub the error message through `_scrub_secrets()` from `auth.py` before logging
2. Consider `exc_info=False` in production, with detailed traces only at `DEBUG` level

---

### SEC-15: SSRF Config Cached with `lru_cache` (LOW)

- **Severity**: LOW
- **CWE**: CWE-665 (Improper Initialization)
- **File**: `src/litellm_llmrouter/url_security.py:249`

**Description**: The SSRF configuration is loaded once and cached via `@lru_cache(maxsize=1)`. If environment variables are changed at runtime (e.g., via a container orchestration secret rotation), the SSRF config will not update. While `clear_ssrf_config_cache()` exists, it is not called automatically on config reload.

**Recommendation**:
1. Wire `clear_ssrf_config_cache()` into the hot-reload lifecycle
2. Document that SSRF config changes require explicit cache clear or process restart

---

### SEC-16: A2A Raw Streaming Chunk Size Parsed Without Validation (LOW)

- **Severity**: LOW
- **CWE**: CWE-20 (Improper Input Validation)
- **File**: `src/litellm_llmrouter/a2a_gateway.py`

**Description**: The `A2A_RAW_STREAMING_CHUNK_SIZE` environment variable is parsed with `int()` without range validation. A negative or zero value could cause unexpected behavior in the streaming loop.

**Recommendation**:
1. Add range validation: reject values <= 0 or > reasonable maximum (e.g., 1MB)
2. Use a safe default on parse failure

---

### SEC-17: Semantic Cache Brute-Force Similarity Scan (LOW)

- **Severity**: LOW
- **CWE**: CWE-400 (Uncontrolled Resource Consumption)
- **File**: `src/litellm_llmrouter/semantic_cache.py:481-505`

**Description**: `RedisCacheStore.get_similar()` performs a brute-force scan using `scan_iter()` over all matching keys, deserializes each entry, and computes cosine similarity in Python. Under high cache volume, this becomes O(n) with significant CPU and network overhead per lookup.

**Recommendation**:
1. Document the performance limitations
2. Add a `max_scan_keys` limit to bound the search
3. For production, migrate to Redis Stack with `FT.SEARCH` vector similarity (as noted in the docstring)

---

### SEC-18: Request ID Accepted from Client Without Validation (LOW)

- **Severity**: LOW
- **CWE**: CWE-20 (Improper Input Validation)
- **File**: `src/litellm_llmrouter/auth.py:326-328`

**Description**: The `RequestIDMiddleware` accepts `X-Request-ID` from the client without format validation. A malicious client could inject a crafted request ID containing special characters that could cause log injection or break log parsing.

```python
request_id = request.headers.get(REQUEST_ID_HEADER, "").strip()
if not request_id:
    request_id = str(uuid.uuid4())
```

**Recommendation**:
1. Validate that client-provided request IDs match a safe format (e.g., UUID, alphanumeric + hyphens, max length)
2. Sanitize or reject request IDs containing newlines, control characters, or shell metacharacters

---

## ReDoS Analysis

All regex patterns in the guardrail plugins were analyzed for catastrophic backtracking potential:

| File | Pattern | ReDoS Risk |
|------|---------|------------|
| `prompt_injection_guard.py:37` | `ignore\s+(all\s+)?previous\s+instructions` | **None** -- optional groups are non-overlapping |
| `prompt_injection_guard.py:42` | `you\s+are\s+now\s+(?:DAN\|an?\s+unrestricted)` | **None** -- alternation is non-overlapping |
| `prompt_injection_guard.py:48` | `system\s*prompt\s*(?:override\|injection\|:)` | **None** -- `\s*` between fixed words is linear |
| `pii_guard.py:58` | `\b\d{3}-\d{2}-\d{4}\b` | **None** -- fixed quantifiers |
| `pii_guard.py:62` | `\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b` | **None** -- fixed quantifiers with optional single char |
| `pii_guard.py:66` | `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b` | **Low** -- `[._%+-]+` and `[.-]+` could cause mild backtracking on very long non-matching strings, but character classes don't overlap with `@` boundary |
| `pii_guard.py:70-75` | Phone pattern with optional groups | **None** -- optional groups are single-char or fixed width |
| `pii_guard.py:79-84` | IPv4 octet pattern | **None** -- bounded alternation within fixed structure |
| `content_filter.py:153` | `(?i)\b(?:kill\|murder\|...)` patterns | **None** -- fixed alternations with `\b` |
| `auth.py:54` | `(sk[-_])[a-zA-Z0-9]{16,}` | **None** -- single char class with `{16,}` is linear |
| `auth.py:55` | `(api[-_]?key[=:])\s*[a-zA-Z0-9\-_]{10,}` | **None** -- non-overlapping components |

**Verdict**: No ReDoS vulnerabilities found. All patterns use non-overlapping components with bounded or linear matching behavior.

---

## GitHub Security Research: Upstream Dependencies

### LiteLLM CVEs (Direct Dependency)

RouteIQ builds on LiteLLM, inheriting its security surface. Known CVEs in LiteLLM:

| CVE | Severity | Description | RouteIQ Impact |
|-----|----------|-------------|----------------|
| **CVE-2024-5751** | CRITICAL (9.8) | Server-Side Template Injection via `/completions` endpoint | **Direct** -- endpoint is proxied through RouteIQ |
| **CVE-2024-6587** | HIGH (7.5) | SSRF via chat completion endpoint | **Mitigated** -- RouteIQ adds its own SSRF layer via `url_security.py`, but LiteLLM internal requests may bypass it |
| **CVE-2024-6825** | HIGH | Remote Code Execution via `post_call_rules` | **Direct** if post_call_rules feature is used |
| **GHSA-53gh-p8jc-7rg8** | HIGH (8.8) | RCE via `post_call_rules` parameter | Same as above |
| **CVE-2024-9606** | HIGH (7.5) | API key masking only redacts first 5 characters | **Direct** -- API keys in LiteLLM logs may be recoverable |
| **CVE-2025-11203** | HIGH | API key disclosure via health endpoint | **Check**: Verify RouteIQ health endpoints don't expose upstream LiteLLM health data with API keys |
| **CVE-2024-8984** | MEDIUM | Improper access control in team management | Applies if LiteLLM team features are used |
| **CVE-2025-45809** | MEDIUM | SQL injection via `/key/block` endpoint | Applies if LiteLLM key management is used |
| **CVE-2024-10188** | MEDIUM | DoS via `ast.literal_eval` in input parsing | **Direct** -- user input parsed by LiteLLM |
| **CVE-2025-0330** | LOW | Langfuse API key leaked in error responses | Applies if Langfuse integration is configured |

**Source**: [GitHub Advisory Database](https://github.com/advisories?query=litellm), [NVD](https://nvd.nist.gov/)

**Recommendations**:
1. Pin LiteLLM to a version with all critical CVEs patched
2. Disable `post_call_rules` feature unless explicitly needed
3. Audit health endpoints for API key exposure (CVE-2025-11203)
4. Verify SSRF protection covers LiteLLM internal HTTP requests

### MCP Protocol Vulnerabilities

The MCP ecosystem has seen significant security incidents:

| CVE/ID | Severity | Description | RouteIQ Impact |
|--------|----------|-------------|----------------|
| **CVE-2025-6514** | CRITICAL (9.6) | RCE in `mcp-remote` via malicious SSE | RouteIQ's MCP SSE transport should validate SSE events |
| **CVE-2025-66416** | HIGH | DNS rebinding in MCP Python SDK | RouteIQ's SSRF checks mitigate this at the gateway level |
| **NeighborJack** | HIGH | Hundreds of MCP servers bind to 0.0.0.0 | RouteIQ should bind to 127.0.0.1 by default |
| **Anthropic FS MCP** | HIGH | Filesystem server sandbox escape via symlinks | N/A -- RouteIQ doesn't expose filesystem MCP |
| **MCP Inspector RCE** | HIGH | RCE in Anthropic MCP Inspector | N/A -- development tool, not deployed |
| **Postmark rugpull** | HIGH | Malicious MCP server masquerading as legitimate | Applies to MCP tool invocation from registered servers |
| **EchoLeak (CVE-2025-32711)** | MEDIUM | Tool-based prompt injection in Copilot | Demonstrates injection via MCP tool responses |
| **Figma MCP** | MEDIUM | Command injection via MCP tool parameters | RouteIQ's MCP proxy should sanitize tool params |

**Source**: [Invariant Labs research](https://invariantlabs.ai/), [MCP SDK security advisories](https://github.com/modelcontextprotocol/python-sdk/security)

**Recommendations**:
1. Validate MCP SSE event structure before processing
2. Ensure MCP tool invocation results are treated as untrusted input
3. Document binding address recommendations (never 0.0.0.0 in production)
4. Add input sanitization for MCP tool parameters before forwarding

### A2A Protocol Vulnerabilities

The A2A protocol is newer with fewer reported CVEs, but architectural risks exist:

- **Agent impersonation**: Without mutual TLS or agent identity verification, registered agents rely solely on URL-based identity
- **Task data exfiltration**: A compromised agent can access all task data sent to it; no confidentiality boundary between agents
- **Unvalidated push notifications**: If A2A push notification URLs are registered, SSRF protection must cover these (RouteIQ does this)

### Redis Vulnerabilities

| CVE | Severity | Description | RouteIQ Impact |
|-----|----------|-------------|----------------|
| **CVE-2025-49844 "RediShell"** | CRITICAL (10.0) | RCE via Lua scripting engine | RouteIQ uses `GET`/`SET` only -- no Lua scripts. Low direct risk, but shared Redis instances could be compromised. |
| **CVE-2024-46981** | HIGH | Lua library RCE | Same as above |

**Recommendations**:
1. Use Redis 7.4.2+ to patch known CVEs
2. Ensure Redis is not exposed to untrusted networks
3. Enable Redis AUTH and (where supported) ACLs
4. Use TLS for Redis connections in production
5. Restrict Redis commands via ACLs (only allow `GET`, `SET`, `DEL`, `SCAN`, `EXPIRE`)

### sentence-transformers / PyTorch Model Loading

| ID | Severity | Description | RouteIQ Impact |
|----|----------|-------------|----------------|
| **SNYK-PYTHON-SENTENCETRANSFORMERS-8161344** | HIGH | Arbitrary code execution when loading PyTorch models | **Direct risk** for semantic cache if loading untrusted models |
| **CVE-2024-11392** | HIGH | Hugging Face Transformers RCE via model loading | Same concern |

**Recommendations**:
1. Only load sentence-transformer models from trusted sources (pin model name + hash)
2. Consider using ONNX Runtime instead of PyTorch for inference
3. Audit the `LLMROUTER_ALLOW_PICKLE_MODELS` setting (currently `false` by default -- good)

### OpenTelemetry Data Exposure

- **No CVEs** specific to OTel SDK, but the OTel project documents risks of inadvertent PII capture in spans and logs
- OTel collector configurations should include attribute redaction processors
- Ensure `otel-collector-config.yaml` does not export span attributes to untrusted destinations

**Recommendations**:
1. Add attribute redaction processor to OTel collector config
2. Review span attributes for PII before enabling production export
3. Use `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT` to truncate long values

---

## Dependency Vulnerability Summary

| Dependency | Version Constraint | Known CVEs | Risk Level | Action |
|------------|-------------------|------------|------------|--------|
| LiteLLM | (pinned in pyproject.toml) | 10+ CVEs (see above) | HIGH | Pin to latest patched version; audit `post_call_rules` |
| Redis (server) | External | CVE-2025-49844, CVE-2024-46981 | HIGH | Upgrade to 7.4.2+; restrict ACLs |
| sentence-transformers | (optional) | SNYK-8161344 | HIGH | Pin trusted models; keep `ALLOW_PICKLE_MODELS=false` |
| FastAPI / Starlette | (via LiteLLM) | Generally well-maintained | LOW | Keep updated |
| httpx | (via http_client_pool) | No current CVEs | LOW | Keep updated |
| opentelemetry-* | (optional) | No specific CVEs | LOW | Use attribute redaction |

---

## Prioritized Remediation Plan

### P0 -- Address Before Production (High Severity)

1. **SEC-01**: Add max task limit and rate limiting to A2A task store
2. **SEC-02**: Fail-closed SSRF validation -- refuse operations when `url_security` import fails
3. **SEC-03**: Integrate PII detection into semantic cache before storing responses
4. **Upstream LiteLLM**: Pin to version patching CVE-2024-5751 (SSTI) and GHSA-53gh-p8jc-7rg8 (RCE)

### P1 -- Address Before GA (Medium Severity)

5. **SEC-04**: Standardize error responses -- use `sanitize_error_response()` in MCP/A2A modules
6. **SEC-05**: Document regex-guard limitations; add multi-part message support; normalize Unicode
7. **SEC-08**: Add request body size limit on JSON-RPC endpoint
8. **SEC-09**: Change DNS timeout to fail-closed in SSRF validator
9. **SEC-10**: Add prominent startup warning when admin auth is disabled
10. **SEC-11**: Enforce plugin allowlist by default or restrict to known module paths
11. **SEC-12**: Remove pattern match details from client-facing error messages

### P2 -- Address in Next Release (Low Severity + Hardening)

12. **SEC-06**: Expand PII entity types; implement Luhn validation for credit cards
13. **SEC-07**: Use word-boundary matching in content filter keywords
14. **SEC-13**: Review OTel span attributes for PII; add collector-level redaction
15. **SEC-14**: Scrub secrets from callback bridge error logs
16. **SEC-15**: Wire SSRF config cache clear into hot-reload lifecycle
17. **SEC-16**: Validate `A2A_RAW_STREAMING_CHUNK_SIZE` range
18. **SEC-17**: Add `max_scan_keys` to semantic cache similarity search
19. **SEC-18**: Validate format of client-provided `X-Request-ID`
20. **Redis hardening**: ACLs, TLS, version upgrade
21. **Model loading**: Verify sentence-transformers model integrity

---

## Appendix A: Files Reviewed

| File | Lines | Findings |
|------|-------|----------|
| `src/litellm_llmrouter/gateway/plugins/prompt_injection_guard.py` | 140 | SEC-05 |
| `src/litellm_llmrouter/gateway/plugins/pii_guard.py` | 248 | SEC-06 |
| `src/litellm_llmrouter/gateway/plugins/content_filter.py` | 412 | SEC-07 |
| `src/litellm_llmrouter/gateway/plugins/guardrails_base.py` | 289 | SEC-12, SEC-13 |
| `src/litellm_llmrouter/gateway/plugin_callback_bridge.py` | 250 | SEC-14 |
| `src/litellm_llmrouter/gateway/plugin_middleware.py` | 317 | (clean) |
| `src/litellm_llmrouter/gateway/plugin_manager.py` | 1056 | SEC-11 |
| `src/litellm_llmrouter/gateway/app.py` | 506 | (clean) |
| `src/litellm_llmrouter/semantic_cache.py` | 521 | SEC-03, SEC-17 |
| `src/litellm_llmrouter/mcp_jsonrpc.py` | 703 | SEC-04, SEC-08 |
| `src/litellm_llmrouter/a2a_gateway.py` | 1412 | SEC-01, SEC-02, SEC-04, SEC-16 |
| `src/litellm_llmrouter/routes.py` | 1651 | (well-structured, uses sanitize_error_response) |
| `src/litellm_llmrouter/auth.py` | 373 | SEC-10, SEC-18 |
| `src/litellm_llmrouter/url_security.py` | 979 | SEC-09, SEC-15 |

## Appendix B: Methodology

1. **Static code review**: Manual line-by-line analysis of all v0.0.2 source files
2. **ReDoS analysis**: Each regex pattern tested for catastrophic backtracking using complexity analysis
3. **OWASP mapping**: Each component mapped against OWASP LLM Top 10 (2025)
4. **Dependency research**: GitHub Advisory Database, NVD, and Snyk searched for all direct dependencies
5. **Threat modeling**: Attack scenarios constructed for each finding

---

*Report generated 2026-02-07 by Claude Code Security Research Agent*
