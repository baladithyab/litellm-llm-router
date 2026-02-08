# RouteIQ v0.0.2 - GitHub Issues & Ecosystem Research

**Date:** 2026-02-07
**Researcher:** Claude (automated research agent)
**Scope:** LiteLLM issues, LLMRouter issues, MCP/A2A ecosystem, AI gateway industry, standards updates

---

## Executive Summary

This report identifies **critical upstream issues, ecosystem shifts, and standards updates** that impact RouteIQ v0.0.2 planning. The most impactful findings are:

1. **LiteLLM v1.81.x has significant regressions** -- performance degradation (issue #19921), Redis cache index misalignment (#20456), and DualCache ThreadPool bottleneck (#20260) directly impact RouteIQ's production readiness. RouteIQ pins `litellm>=1.81.3`.

2. **LiteLLM Router has a critical 38x request amplification bug** (#17329) where requests broadcast to ALL configured models instead of the requested one. This could interact with RouteIQ's monkey-patched routing strategies.

3. **Streaming tool_calls are non-spec-compliant** (#20480) -- `id` sent on every chunk + duplicate summary chunk breaks downstream consumers. This affects any MCP tool invocation flowing through RouteIQ.

4. **MCP specification has advanced to version 2025-11-25** with async Tasks, CIMD, XAA, and enhanced OAuth. RouteIQ's MCP implementation is based on the 2025-03-26 spec and needs updating.

5. **A2A protocol donated to Linux Foundation** for neutral governance. Production-ready reference implementations are shipping (ServiceNow, etc.).

6. **OWASP released Top 10 for Agentic Applications** (Dec 2025) -- a new security benchmark directly relevant to RouteIQ's MCP/A2A gateway functionality.

7. **AI gateway market is rapidly evolving** -- Helicone launched a Rust-based AI Gateway (June 2025), Bifrost claims 11us overhead, and Kong AI Gateway is expanding AI-specific features. RouteIQ needs differentiation strategy.

8. **OpenAI deprecating Chat Completions API** target mid-2026 in favor of Responses API. This is a fundamental shift RouteIQ must plan for.

---

## Part 1: LiteLLM GitHub Issues

### Critical Issues (Direct RouteIQ Impact)

| Issue # | Title | Severity | RouteIQ Impact | Action Needed |
|---------|-------|----------|----------------|---------------|
| [#17329](https://github.com/BerriAI/litellm/issues/17329) | Router broadcasts requests to ALL models (38x amplification) | **CRITICAL** | Could drain API budgets; interacts with monkey-patched routing | Verify RouteIQ routing strategies are not affected; add safeguard in `routing_strategy_patch.py` |
| [#20260](https://github.com/BerriAI/litellm/issues/20260) | DualCache ThreadPool Bottleneck (100x perf degradation) | **CRITICAL** | Affects all routing decisions under concurrency; `batch_get_cache()` creates ThreadPoolExecutor per call | Monitor; may need upstream fix or workaround in RouteIQ |
| [#19921](https://github.com/BerriAI/litellm/issues/19921) | Performance regression v1.80.5 -> v1.81.x | **HIGH** | RouteIQ pins `>=1.81.3`; reported slowness on UI + API + model loading | Pin to specific tested version; do not float to latest |
| [#20456](https://github.com/BerriAI/litellm/issues/20456) | Async Batch Embedding + Redis Cache Index Misalignment (Regression v1.81.0) | **HIGH** | Affects HA deployments with Redis cache | Test embedding routes; track upstream fix |
| [#20647](https://github.com/BerriAI/litellm/issues/20647) | Spend logs fail with thread lock serialization error when redaction enabled | **HIGH** | Breaks CostTrackerPlugin if redaction enabled | Avoid enabling redaction until fixed; or work around Pydantic v2 deepcopy |
| [#20507](https://github.com/BerriAI/litellm/issues/20507) | anthropic_messages pass-through raises BaseLLMException (breaks Router retry/fallback) | **HIGH** | Router retry and context_window_fallbacks silently broken for Anthropic messages | Critical for Bedrock/Claude routing; need upstream fix |
| [#20480](https://github.com/BerriAI/litellm/issues/20480) | Streaming tool_calls: id on every chunk + duplicate summary chunk | **MEDIUM** | Breaks downstream MCP tool consumers | May need workaround in MCP SSE transport |
| [#20441](https://github.com/BerriAI/litellm/issues/20441) | 403 Forbidden for messages containing `<script>` string | **MEDIUM** | Overzealous XSS protection blocks legitimate coding prompts | Track upstream fix; may need proxy-level bypass |
| [#20589](https://github.com/BerriAI/litellm/issues/20589) | /v1/messages + Bedrock inference profiles: ChecksumMismatch streaming error | **MEDIUM** | Affects Claude Code users through RouteIQ with Bedrock backend | Track upstream; affects streaming reliability |

### Callback System Issues

| Issue # | Title | Severity | RouteIQ Impact |
|---------|-------|----------|----------------|
| [#20294](https://github.com/BerriAI/litellm/issues/20294) | Team callback reads wrong metadata key, ignores YAML config | **HIGH** | PluginCallbackBridge may be affected if team-level callbacks are used |
| [#8842](https://github.com/BerriAI/litellm/issues/8842) | Router async_completion doesn't trigger CustomLogger callbacks | **MEDIUM** | May cause missing telemetry for async routes |
| [#19806](https://github.com/BerriAI/litellm/issues/19806) | Feature: Add retry callback hook to CustomLogger | **LOW** | Would benefit RouteIQ retry observability if implemented |
| [#17310](https://github.com/BerriAI/litellm/issues/17310) | Regression: log_pre_api_call stopped for passthrough endpoints | **MEDIUM** | Pre-call logging may be missing for passthrough routes |
| [#16589](https://github.com/BerriAI/litellm/issues/16589) | Team/Key-level callback settings don't override global callbacks | **MEDIUM** | Affects multi-tenant callback configuration |

### Cost Calculation Issues

| Issue # | Title | Severity | RouteIQ Impact |
|---------|-------|----------|----------------|
| [#20521](https://github.com/BerriAI/litellm/issues/20521) | 39 OpenRouter models removed from model_prices JSON | **MEDIUM** | `litellm.model_cost` may have stale/missing pricing for OpenRouter models |
| [#20412](https://github.com/BerriAI/litellm/issues/20412) | Cost tracking not working with Vercel Models | **LOW** | Affects Vercel model users |
| [#11364](https://github.com/BerriAI/litellm/issues/11364) | Wrong cost for Anthropic: cached tokens not correctly considered | **MEDIUM** | CostTrackerPlugin may report incorrect costs for cached Anthropic requests |
| [#16884](https://github.com/BerriAI/litellm/issues/16884) | Mismatch between /model/info and cost calculations | **LOW** | Cost reporting inconsistencies |
| [#14457](https://github.com/BerriAI/litellm/issues/14457) | Usage data lost on streaming early disconnect | **MEDIUM** | Cost tracking fails when clients disconnect during streaming |

### Streaming Issues

| Issue # | Title | Severity | RouteIQ Impact |
|---------|-------|----------|----------------|
| [#18842](https://github.com/BerriAI/litellm/issues/18842) | UTF-8 Multibyte Character Corruption in Streaming Mode | **MEDIUM** | Affects international language support through gateway |
| [#20347](https://github.com/BerriAI/litellm/issues/20347) | Anthropic streaming silently completes with empty content on upstream errors | **HIGH** | Silent failures in Anthropic streaming -- no error propagated to client |
| [#20389](https://github.com/BerriAI/litellm/issues/20389) | "dictionary changed size during iteration" in Response streaming | **MEDIUM** | Thread-safety issue in streaming; could cause intermittent failures |
| [#16535](https://github.com/BerriAI/litellm/issues/16535) | x-litellm-model-group header missing in streaming responses | **LOW** | Routing metadata lost in streaming responses |

### Security Issues

| Issue # | Title | Severity | RouteIQ Impact |
|---------|-------|----------|----------------|
| [#20416](https://github.com/BerriAI/litellm/issues/20416) | 32 vulnerabilities reported in security scan (v1.81) | **HIGH** | Upstream SAST findings; 5 unique vulnerability classes |
| [#13906](https://github.com/BerriAI/litellm/issues/13906) | Invite link allows password reset multiple times | **MEDIUM** | N/A -- RouteIQ doesn't use LiteLLM's user management UI |
| [#20494](https://github.com/BerriAI/litellm/issues/20494) | Generating new key with same secret key doesn't throw error | **LOW** | Key collision risk |

### Other Notable Issues

| Issue # | Title | Notes |
|---------|-------|-------|
| [#20562](https://github.com/BerriAI/litellm/issues/20562) | Bedrock Claude Opus 4.6 Model ID is incorrect | Affects Opus 4.6 routing through Bedrock |
| [#20543](https://github.com/BerriAI/litellm/issues/20543) | Bedrock Claude Sonnet 4.5 concatenated JSON in tool call args | Tool calling broken for Sonnet 4.5 on Bedrock |
| [#20570](https://github.com/BerriAI/litellm/issues/20570) | Pydantic ValidationError when provider omits required fields | Provider compatibility regression |
| [#20534](https://github.com/BerriAI/litellm/issues/20534) | Team/key updates becoming enterprise-only in v1.81.6.rc.1? | License concern -- features moving behind paywall |
| [#20519](https://github.com/BerriAI/litellm/issues/20519) | Migrate Claude Opus 4.6 to adaptive thinking | Feature request for latest Claude model support |
| [#20533](https://github.com/BerriAI/litellm/issues/20533) | Support Anthropic structured outputs on Opus 4.6 | Feature request for latest Claude capability |
| [#20498](https://github.com/BerriAI/litellm/issues/20498) | Many clients requiring /mcp/ when connecting | MCP path compatibility issue |
| [#20495](https://github.com/BerriAI/litellm/issues/20495) | MCP OAuth flow fails -- temp server doesn't inherit OAuth URLs | MCP OAuth broken in LiteLLM |
| [#20615](https://github.com/BerriAI/litellm/issues/20615) | Blocked all requests if no budget configured | Dangerous default behavior |

### LiteLLM Release Status

- **Latest stable:** v1.81.0-stable.1 (2026-02-07)
- **Latest nightly:** v1.81.9-nightly (2026-02-07)
- **RouteIQ dependency:** `litellm>=1.81.3`
- **Key concern:** v1.81.x has multiple regressions vs v1.80.5. Consider pinning to a specific version tested with RouteIQ (e.g., `litellm==1.81.0`).

---

## Part 2: LLMRouter GitHub Issues

### Open Issues (ulab-uiuc/LLMRouter)

| Issue # | Title | Category | RouteIQ Impact |
|---------|-------|----------|----------------|
| [#164](https://github.com/ulab-uiuc/LLMRouter/issues/164) | Add model fallback mechanism | Enhancement | Relevant to RouteIQ's fallback strategy |
| [#163](https://github.com/ulab-uiuc/LLMRouter/issues/163) | Router inconsistent behavior with concurrent requests | **Bug** | Could cause routing instability under load |
| [#161](https://github.com/ulab-uiuc/LLMRouter/issues/161) | Optimize embedding computation time | Performance | Directly affects routing latency |
| [#158](https://github.com/ulab-uiuc/LLMRouter/issues/158) | Memory leak in router batch processing | **Bug** | Could affect long-running RouteIQ instances |
| [#155](https://github.com/ulab-uiuc/LLMRouter/issues/155) | Implement A/B testing framework for routers | Enhancement | RouteIQ already has A/B testing in `strategy_registry.py` |
| [#153](https://github.com/ulab-uiuc/LLMRouter/issues/153) | Router fails silently when no suitable model found | **Bug** | Silent failures are dangerous for production |
| [#151](https://github.com/ulab-uiuc/LLMRouter/issues/151) | Embedding cache not invalidated on model config change | **Bug** | Stale cache after hot-reload config changes |
| [#150](https://github.com/ulab-uiuc/LLMRouter/issues/150) | Add distributed router inference support | Enhancement | Relevant for HA deployments |
| [#162](https://github.com/ulab-uiuc/LLMRouter/issues/162) | Add multi-lingual support for queries | Feature | Internationalization support |
| [#159](https://github.com/ulab-uiuc/LLMRouter/issues/159) | Add WebSocket support for streaming responses | Feature | Alternative streaming transport |

### Analysis

- LLMRouter has **3 critical bugs** (concurrent request inconsistency, memory leak, silent failures) that could directly affect RouteIQ routing reliability
- The embedding cache invalidation bug (#151) is particularly relevant since RouteIQ uses hot-reload
- RouteIQ already implements some requested features (A/B testing, telemetry) -- upstream may want contributions
- **No new strategies** have been contributed to upstream LLMRouter recently

---

## Part 3: MCP/A2A Ecosystem Updates

### MCP Specification Evolution

| Version | Date | Key Changes | RouteIQ Status |
|---------|------|-------------|----------------|
| 2024-11-05 | Nov 2024 | Initial release | Legacy |
| 2025-03-26 | Mar 2025 | OAuth 2.1, Streamable HTTP (replaces SSE), tool annotations, audio content | **RouteIQ's current baseline** |
| 2025-06-18 | Jun 2025 | Structured tool output, OAuth Resource Servers, Resource Indicators, elicitation, removed JSON-RPC batching | **RouteIQ needs update** |
| **2025-11-25** | **Nov 2025** | **Async Tasks, CIMD, XAA, sampling with tools, scope management, extensions** | **RouteIQ needs update** |

#### Critical MCP 2025-11-25 Changes for RouteIQ

1. **Async Tasks (experimental)** -- Long-running operations become "call-now, fetch-later". RouteIQ's MCP gateway needs to support this for production agent workflows.

2. **Client ID Metadata Documents (CIMD)** -- Replaces per-server Dynamic Client Registration. Clients use a URL they control as `client_id`. This simplifies multi-server scenarios significantly.

3. **Cross-App Access (XAA)** -- Enterprise authorization where IT admins pre-authorize trusted agents. Critical for enterprise RouteIQ deployments.

4. **Sampling with Tools** -- MCP servers can now initiate sampling requests with tool definitions, enabling server-side agent loops. This affects RouteIQ's MCP proxy architecture.

5. **Version Negotiation** -- Formal mechanism for clients and servers to negotiate protocol version. RouteIQ must implement this.

6. **Extensions Framework** -- Servers can expose experimental/vendor-specific features without spec changes.

7. **OTel Integration** -- OpenTelemetry semantic conventions for MCP are now defined (see `opentelemetry.io/docs/specs/semconv/gen-ai/mcp/`). RouteIQ should adopt these in `mcp_tracing.py`.

### A2A Protocol Updates

| Milestone | Date | Details |
|-----------|------|---------|
| A2A launch | Apr 2025 | Google Cloud Next; 50+ technology partners |
| Linux Foundation donation | Mid-2025 | Neutral governance; open-source project |
| Production implementations | Dec 2025+ | ServiceNow, Salesforce, SAP shipping A2A |
| Enterprise adoption | 2026 | IDC projects agentic AI spending >$1.3T by 2029 |

#### Key A2A Developments

- **A2A is now under Linux Foundation** governance, making it a more credible interoperability standard
- **Production-ready implementations** are shipping from major enterprise vendors (ServiceNow Dec 2025 release)
- **A2A + MCP integration** is becoming a standard pattern (A2A for agent-to-agent, MCP for agent-to-tool)
- **Planned enhancements:** formalized auth in Agent Cards, dynamic UX negotiation, improved streaming, push notifications
- RouteIQ's A2A implementation (`a2a_gateway.py`) should be validated against production reference implementations

### MCP Community Issues

- **MCP OAuth flow bugs** are being reported across implementations (#20495 in LiteLLM)
- **MCP path compatibility** (`/mcp/` vs `/mcp`) causing client connection issues (#20498)
- **MCP token handling** -- OAuth tokens not being properly saved after successful flows (#20493)
- The community is converging on Streamable HTTP as the primary transport (SSE is being phased out)

---

## Part 4: AI Gateway Industry Movement

### Competitive Landscape (as of Feb 2026)

| Gateway | Key Differentiator | Recent Changes | Threat Level |
|---------|-------------------|----------------|--------------|
| **Helicone** | Rust-based, observability-first | Launched AI Gateway (Jun 2025); 8ms P50 latency; passthrough billing | **Medium** -- good DX, lacks ML routing |
| **Bifrost (Maxim AI)** | Zero-config, extreme performance | Claims 11us at 5K RPS; MCP support | **Medium** -- focused on speed, not intelligence |
| **Portkey** | Enterprise governance | MCP + agent integrations; comprehensive governance | **High** -- strong enterprise positioning |
| **Kong AI Gateway** | API management heritage | AI-specific plugins; prompt engineering; token rate limiting | **Medium** -- traditional API gateway adding AI |
| **TensorZero** | Rust, industrial-grade | Low-latency structured workflows | **Low** -- different target market |
| **OpenRouter** | Managed simplicity | 200+ models; automatic routing | **Medium** -- volume play, not enterprise |
| **LiteLLM (upstream)** | Provider breadth | 100+ providers; enterprise features gating | **N/A** -- dependency, not competitor |

### Key Industry Trends

1. **Responses API is the future:** OpenAI plans to deprecate Chat Completions API by mid-2026. The Responses API (launched Mar 2025) is becoming the new standard for agent-native applications. Open Responses API (Jan 2026) extends this as an open standard.

2. **Enterprise share shift:** Anthropic now commands 40% of enterprise LLM API spend (Menlo Ventures Dec 2025), up from 12% in 2023. OpenAI fell from 50% to 27%. RouteIQ's multi-provider routing is well-positioned for this fragmentation.

3. **GPT-4o retiring Feb 2026:** OpenAI shutting down GPT-4o, pushing to GPT-5.2. RouteIQ's cost/routing models need updated pricing.

4. **New model providers:**
   - xAI Grok Batch API requested (#20424)
   - fal.ai video/audio generation (#20468)
   - Cohere Embed v4.5 on Bedrock (#20466)
   - Claude Opus 4.6 with adaptive thinking (#20519)
   - GPT-5.2 Codex (#20575 mentions "gpt-5.2" in bug reports)

5. **AI gateways becoming "AI firewalls":** The market is shifting from pure routing to governance, guardrails, and compliance as primary value props.

---

## Part 5: Standards Updates

### OpenTelemetry GenAI Semantic Conventions

- **Status:** Development (not yet stable)
- **Latest:** Part of OTel semantic conventions with dedicated GenAI section
- **Key additions:**
  - `gen_ai.*` attribute namespace standardized
  - Semantic conventions for MCP operations now defined at `opentelemetry.io/docs/specs/semconv/gen-ai/mcp/`
  - Agent and framework spans conventions emerging
  - Provider-specific conventions: AWS Bedrock, Azure AI Inference, OpenAI
- **RouteIQ impact:** `mcp_tracing.py`, `a2a_tracing.py`, and `observability.py` should adopt these conventions
- **Migration note:** Existing instrumentations need `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` flag

### OWASP Updates

#### OWASP Top 10 for LLM Applications (2025 Edition)
Released November 2024, this is the current version:
1. LLM01: Prompt Injection
2. LLM02: Sensitive Information Disclosure
3. LLM03: Supply Chain
4. LLM04: Data and Model Poisoning
5. LLM05: Improper Output Handling
6. LLM06: Excessive Agency
7. LLM07: System Prompt Leakage
8. LLM08: Vector and Embedding Weaknesses
9. LLM09: Misinformation
10. LLM10: Unbounded Consumption

#### **NEW: OWASP Top 10 for Agentic Applications (Dec 2025)**
This is **directly relevant** to RouteIQ's MCP/A2A gateway:
1. **ASI01: Agent Goal Hijack** -- Attacks that redirect agent objectives
2. **ASI02: Tool Misuse** -- Agents manipulated into executing malicious actions
3. **ASI03: Privilege Compromise** -- Exploiting agent permissions
4. **ASI04: Agentic Supply Chain Vulnerabilities** -- Third-party tool/agent risks
5. **ASI05: Memory Poisoning** -- Corrupting agent memory/context
6. **ASI06: Cascading Hallucinations** -- Error propagation in multi-agent systems
7. **ASI07: Insecure Inter-Agent Communication** -- A2A communication flaws
8. **ASI08: Identity Abuse** -- Agent impersonation
9. **ASI09: Human-Agent Trust Exploitation** -- Social engineering via agents
10. **ASI10: Rogue Agents** -- Agents operating outside intended parameters

**Action:** RouteIQ should map its security controls to these categories, especially ASI02 (tool misuse in MCP), ASI04 (supply chain), ASI07 (inter-agent communication in A2A), and ASI10 (rogue agents).

#### OWASP Top 10:2025 (Web Applications)
Released Nov 2025 with new categories:
- A03:2025 Software Supply Chain Failures (new)
- A09:2025 Logging & Alerting Failures (renamed)
- A10:2025 Mishandling of Exceptional Conditions (new)

### NIST AI Risk Management Framework

- **NIST SP 800-53 Release 5.2.0** finalized Aug 2025 (per Executive Order 14306)
- **NIST Privacy Framework 1.1** draft released Apr 2025, updated for AI
- **Dec 2025:** NIST released draft guidelines rethinking cybersecurity for the AI era
- RouteIQ should track NIST AI RMF updates for enterprise compliance positioning

---

## Prioritized Actions for v0.0.2

### P0 -- Must Address (Blocks Production Readiness)

1. **Pin LiteLLM to a tested stable version** -- The `>=1.81.3` constraint is dangerous given v1.81.x regressions. Pin to a specific version tested with RouteIQ (e.g., `litellm==1.81.0`).

2. **Validate RouteIQ routing is not affected by #17329** (38x amplification bug) -- Test that monkey-patched strategies correctly route to single models, not broadcast.

3. **Test DualCache ThreadPool behavior** (#20260) under concurrent load -- This directly impacts routing decision latency.

4. **Handle streaming tool_call spec violations** (#20480) -- Add normalization in MCP SSE transport or callback bridge to strip duplicate `id` fields.

5. **Validate Anthropic exception typing** (#20507) -- Ensure Router retry/fallback works for Bedrock Claude through RouteIQ.

### P1 -- Should Address (Major Functionality Gaps)

6. **Update MCP implementation to 2025-11-25 spec** -- Add async Tasks support, version negotiation, CIMD awareness. This is 2 spec versions behind.

7. **Adopt OTel GenAI semantic conventions** for MCP tracing -- Replace custom attribute names with standard `gen_ai.*` conventions.

8. **Add OWASP Agentic Top 10 security controls** -- Especially ASI02 (tool misuse validation), ASI04 (supply chain checks for MCP tools), ASI07 (A2A communication security).

9. **Validate A2A implementation against production reference implementations** (ServiceNow, etc.) -- Ensure interoperability.

10. **Track OpenAI Responses API** -- Begin planning for Chat Completions deprecation (mid-2026). RouteIQ should support routing Responses API requests.

### P2 -- Should Track (Market Positioning)

11. **Monitor enterprise feature gating in LiteLLM** (#20534) -- Features moving behind enterprise license may affect RouteIQ's free tier.

12. **Update model pricing data** -- OpenRouter models removed (#20521), new models (Opus 4.6, GPT-5.2 Codex, Gemini 3) need pricing.

13. **Evaluate Rust-based gateway components** -- Competitors (Helicone, Bifrost, TensorZero) are using Rust for performance-critical paths. Consider if RouteIQ needs a Rust proxy layer for latency-sensitive routes.

14. **Prepare for GPT-4o deprecation** (Feb 2026) -- Ensure routing strategies gracefully handle model retirement.

### P3 -- Nice to Have

15. **Contribute A/B testing and telemetry features upstream** to LLMRouter (#155, #156)
16. **Add xAI Grok, fal.ai support** as new providers
17. **Implement MCP Extensions framework** for vendor-specific features
18. **Add NIST AI RMF compliance documentation** for enterprise sales

---

## Appendix: Data Sources

- GitHub Issues: `gh issue list --repo BerriAI/litellm` (50+ issues analyzed)
- GitHub Issues: `gh issue list --repo ulab-uiuc/LLMRouter` (20+ issues analyzed)
- LiteLLM Releases: v1.81.0-stable.1 through v1.81.9-nightly (Feb 2026)
- MCP Specification: modelcontextprotocol.io (versions 2025-03-26, 2025-06-18, 2025-11-25)
- A2A Protocol: developers.googleblog.com, Linux Foundation
- OpenAI Changelog: platform.openai.com/docs/changelog
- OWASP: genai.owasp.org, owasp.org/Top10/2025
- OTel: opentelemetry.io/docs/specs/semconv/gen-ai/
- Industry reports: Menlo Ventures 2025 State of GenAI, Gartner, IDC
- Web search: Tavily advanced search across multiple queries
