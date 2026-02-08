# AI Gateway Industry Landscape Report

**Project**: RouteIQ Gateway
**Date**: February 2025
**Scope**: Competitive analysis of AI Gateway / LLM Proxy products and RouteIQ positioning

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Industry Overview](#2-industry-overview)
3. [Competitor Profiles](#3-competitor-profiles)
   - 3.1 [LiteLLM Proxy (Upstream)](#31-litellm-proxy-upstream)
   - 3.2 [Portkey AI Gateway](#32-portkey-ai-gateway)
   - 3.3 [Helicone](#33-helicone)
   - 3.4 [Cloudflare AI Gateway](#34-cloudflare-ai-gateway)
   - 3.5 [Kong AI Gateway](#35-kong-ai-gateway)
   - 3.6 [Google Apigee with AI Extensions](#36-google-apigee-with-ai-extensions)
   - 3.7 [AWS API Gateway for Bedrock](#37-aws-api-gateway-for-bedrock)
   - 3.8 [Martian (Model Router)](#38-martian-model-router)
   - 3.9 [Unify.ai](#39-unifyai)
   - 3.10 [Other Notable Players](#310-other-notable-players)
4. [RouteIQ Capability Assessment](#4-routeiq-capability-assessment)
5. [Feature Comparison Matrix](#5-feature-comparison-matrix)
6. [Gap Analysis](#6-gap-analysis)
7. [Industry Trends](#7-industry-trends)
8. [Prioritized Recommendations](#8-prioritized-recommendations)
9. [Appendix: RouteIQ Source Inventory](#9-appendix-routeiq-source-inventory)

---

## 1. Executive Summary

The AI Gateway market has matured rapidly from 2023 through early 2025, evolving from simple LLM proxy layers into full-spectrum AI infrastructure platforms. The market segments into three tiers:

1. **Observability-first platforms** (Helicone, Braintrust) -- focused on logging, analytics, and prompt management
2. **Full-stack AI gateways** (Portkey, LiteLLM, Cloudflare) -- routing, caching, guardrails, and observability in one product
3. **Enterprise API management with AI extensions** (Kong, Apigee, AWS) -- traditional API gateway vendors adding AI capabilities

**RouteIQ's positioning is unique**: it is the only gateway in the market that combines LiteLLM's 100+ provider compatibility with genuine ML-based routing intelligence (18+ learned strategies), multi-protocol support (OpenAI API, MCP, A2A), and a closed-loop MLOps pipeline. No other product offers ML routing that learns from production traffic and hot-reloads improved models.

Key findings:
- RouteIQ **leads** in ML-based intelligent routing, A2A protocol support, and MLOps feedback loops
- RouteIQ is **competitive** in observability, security hardening, and plugin extensibility
- RouteIQ has **gaps** in semantic caching, managed UI/dashboard, prompt management, guardrails (content filtering), and multi-modal support
- The industry is converging on **agentic infrastructure** (MCP + A2A), where RouteIQ has early-mover advantage

---

## 2. Industry Overview

### Market Size and Growth

The AI Gateway / LLM infrastructure market is a subset of the broader AI infrastructure market, estimated at $30-40B by 2027. AI gateways specifically address:

- **Cost optimization**: Routing to cheaper models when appropriate (estimated 30-70% savings)
- **Reliability**: Fallbacks, retries, and circuit breakers across providers
- **Observability**: Token counting, latency tracking, cost attribution
- **Security**: Rate limiting, content filtering, PII detection
- **Governance**: Audit logging, access control, policy enforcement

### Market Segmentation

| Segment | Products | Focus |
|---------|----------|-------|
| **Open-source self-hosted** | LiteLLM, Portkey Gateway OSS, RouteLLM | Developer-first, maximum control |
| **Managed SaaS** | Portkey Cloud, Helicone, Cloudflare | Zero-ops, usage-based pricing |
| **Enterprise API mgmt** | Kong AI, Apigee AI, AWS Gateway | Existing API infrastructure + AI bolted on |
| **Specialized routers** | Martian, Unify.ai, RouteLLM | Pure routing intelligence |
| **Hybrid (RouteIQ)** | RouteIQ | Open-source gateway + ML routing + MLOps pipeline |

---

## 3. Competitor Profiles

### 3.1 LiteLLM Proxy (Upstream)

**URL**: https://github.com/BerriAI/litellm
**License**: MIT (OSS core) + Enterprise tier
**GitHub Stars**: ~18,000+ (as of early 2025)
**Status**: RouteIQ's upstream dependency -- RouteIQ extends LiteLLM, not competes with it

#### Core Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| **Routing** | Basic | simple-shuffle, least-busy, latency-based, cost-based, usage-based |
| **Caching** | Yes | Redis-backed response caching, semantic caching (experimental) |
| **Guardrails** | Yes | LakeraGuard, Presidio PII, custom callbacks (Enterprise) |
| **Observability** | Yes | Langfuse, Helicone, custom callbacks; basic OpenTelemetry |
| **Auth** | Yes | Virtual keys, team management, SSO (Enterprise) |
| **Rate Limiting** | Yes | Per-key, per-team, RPM/TPM limits |

#### Differentiating Features

- **100+ LLM providers**: Widest provider support in the market (OpenAI, Anthropic, Bedrock, Azure, Google, Cohere, Replicate, etc.)
- **OpenAI API compatibility**: Drop-in replacement for OpenAI SDK
- **Virtual keys**: Abstract provider keys behind internal virtual keys
- **Spend tracking**: Per-key, per-team, per-model cost tracking
- **Proxy server**: Production-ready FastAPI server with Docker support
- **Enterprise tier**: SSO, advanced guardrails, premium support

#### Protocol Support

- OpenAI-compatible API (chat/completions, embeddings, images, audio)
- Pass-through endpoints for provider-specific APIs
- Basic MCP support (added mid-2024)
- A2A support (added late 2024)

#### Pricing

- **Open Source (MIT)**: Full proxy functionality, community support
- **Enterprise**: Self-hosted with SSO, guardrails, premium support ($$$)
- **Hosted (LiteLLM Cloud)**: Managed SaaS option

#### Relevance to RouteIQ

LiteLLM is RouteIQ's foundation. RouteIQ inherits all of LiteLLM's provider support, API compatibility, virtual keys, and basic routing. RouteIQ extends it with ML routing intelligence, enterprise hardening, and multi-protocol gateway capabilities. RouteIQ must track upstream releases carefully.

---

### 3.2 Portkey AI Gateway

**URL**: https://portkey.ai
**License**: MIT (Gateway OSS) + Cloud (SaaS)
**GitHub Stars**: ~7,000+ (as of early 2025)

#### Core Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| **Routing** | Advanced | Fallbacks, load balancing, conditional routing, weighted distribution |
| **Caching** | Yes | Simple caching + semantic caching (embedding-based similarity) |
| **Guardrails** | Yes | Built-in content filtering, PII detection, custom guardrails |
| **Observability** | Excellent | Real-time logs, traces, analytics dashboard, cost tracking |
| **Auth** | Yes | API keys, workspace-level access control |
| **Rate Limiting** | Yes | Per-key rate limiting |

#### Differentiating Features

- **AI Gateway with managed dashboard**: Polished web UI for observability and management
- **Semantic caching**: Embedding-based response caching that matches semantically similar queries
- **Guardrails framework**: Built-in content filtering, PII detection, topic restriction
- **Prompt management**: Version-controlled prompt templates with A/B testing
- **Universal API**: Single API for 200+ LLMs across providers
- **Reliability features**: Automatic retries, fallbacks, load balancing, timeouts
- **Config-as-code**: Gateway configurations defined in YAML/JSON
- **Feedback and evaluation**: Collect user feedback, run evals against logged data

#### Supported Providers

250+ LLMs including OpenAI, Anthropic, Azure, Google Vertex AI, AWS Bedrock, Cohere, Mistral, Groq, Together AI, Perplexity, and many more.

#### Protocol Support

- OpenAI-compatible API
- Provider-native APIs (pass-through)
- No native MCP support (as of early 2025)
- No A2A support

#### Plugin/Extension Model

- Config-driven "gateway configs" (JSON/YAML) for routing rules
- Webhook-based custom integrations
- No formal plugin API for extending the gateway itself

#### Pricing

- **Open Source**: Self-hosted gateway (MIT license)
- **Developer (Free)**: 10K requests/month on cloud
- **Production**: $49/month for 100K requests
- **Enterprise**: Custom pricing, dedicated support, SSO

#### Competitive Assessment vs RouteIQ

| Dimension | Portkey | RouteIQ |
|-----------|---------|---------|
| Routing Intelligence | Rule-based (fallbacks, weights) | ML-based (18+ learned strategies) |
| Dashboard/UI | Polished managed UI | No built-in UI |
| Semantic Caching | Yes | No |
| Guardrails | Built-in content filtering | Policy engine (OPA-style rules) |
| MCP Support | No | Yes (5 surfaces) |
| A2A Support | No | Yes |
| Plugin System | Limited (config-driven) | Full plugin lifecycle |
| MLOps Pipeline | No | Yes (train, deploy, hot-reload) |

---

### 3.3 Helicone

**URL**: https://helicone.ai
**License**: Apache 2.0 (OSS) + Cloud (SaaS)
**GitHub Stars**: ~4,000+ (as of early 2025)

#### Core Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| **Routing** | None | Observability-focused, no routing |
| **Caching** | Yes | Response caching via proxy headers |
| **Guardrails** | Limited | Threat detection, rate limiting |
| **Observability** | Excellent | Best-in-class logging, analytics, traces, cost tracking |
| **Auth** | Basic | API key based |
| **Rate Limiting** | Yes | Per-key rate limiting |

#### Differentiating Features

- **Zero-integration observability**: Header-based proxy (add 1 header to existing OpenAI calls)
- **Advanced analytics dashboard**: Request visualization, cost breakdown, latency percentiles
- **Prompt management**: Template versioning, A/B testing of prompts
- **User tracking**: Per-user analytics and session tracking
- **Experiments**: A/B test different prompts and models with statistical analysis
- **Datasets and evaluation**: Build datasets from production data, run evals
- **Gateway mode**: Proxy mode for intercepting all LLM calls

#### Supported Providers

OpenAI, Anthropic, Azure OpenAI, Google, Mistral, Groq, Together AI, and others via proxy mode.

#### Protocol Support

- OpenAI-compatible proxy
- Provider-native APIs (header-based interception)
- No MCP or A2A support

#### Pricing

- **Free**: 100K requests/month
- **Pro**: $20/month (unlimited requests, advanced features)
- **Team**: $40/seat/month
- **Enterprise**: Custom

#### Competitive Assessment vs RouteIQ

Helicone is not a direct competitor -- it is an observability platform that could complement RouteIQ. However, Helicone's polished analytics dashboard and prompt management features highlight gaps in RouteIQ's user-facing tooling.

| Dimension | Helicone | RouteIQ |
|-----------|----------|---------|
| Routing | None | ML-based (18+ strategies) |
| Observability UI | Polished dashboard | OTel export to Jaeger/Grafana |
| Prompt Management | Yes | No |
| Cost Analytics | Best-in-class | Via LiteLLM spend tracking |
| Integration Effort | 1 header change | Config + deployment |

---

### 3.4 Cloudflare AI Gateway

**URL**: https://developers.cloudflare.com/ai-gateway/
**License**: Proprietary (SaaS)
**Pricing**: Free tier available; part of Cloudflare Workers platform

#### Core Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| **Routing** | Basic | Provider fallbacks only |
| **Caching** | Yes | Response caching at edge (global CDN) |
| **Guardrails** | Limited | Basic content filtering |
| **Observability** | Good | Real-time logging, analytics, cost tracking in Cloudflare dashboard |
| **Auth** | Yes | Cloudflare Access integration |
| **Rate Limiting** | Yes | Built-in, leverages Cloudflare's global infrastructure |

#### Differentiating Features

- **Edge-native**: Runs on Cloudflare's global CDN (300+ PoPs) -- lowest latency caching
- **Response caching**: Cache LLM responses at the edge for repeated queries
- **Analytics dashboard**: Built into Cloudflare dashboard (familiar for existing customers)
- **Workers AI integration**: Seamless integration with Cloudflare's own AI inference
- **Zero-ops**: Fully managed, no infrastructure to deploy
- **DDoS protection**: Inherits Cloudflare's DDoS mitigation

#### Supported Providers

OpenAI, Anthropic, Azure OpenAI, Google AI, HuggingFace, Replicate, Perplexity, Groq, Cohere, Mistral, Workers AI (Cloudflare).

#### Protocol Support

- OpenAI-compatible (partial)
- Provider-specific APIs (universal endpoint)
- No MCP or A2A support

#### Pricing

- **Free**: 100K requests/day (generous for development)
- **Paid**: Included in Cloudflare Workers plans; AI Gateway-specific features may have usage-based pricing

#### Competitive Assessment vs RouteIQ

Cloudflare AI Gateway excels at edge caching and zero-ops deployment but lacks intelligent routing, ML capabilities, and agent protocol support. It is better suited for teams already on Cloudflare's platform who want simple caching and observability.

| Dimension | Cloudflare | RouteIQ |
|-----------|------------|---------|
| Edge Caching | Global CDN (best-in-class) | No built-in caching |
| Routing Intelligence | Basic fallbacks | ML-based (18+ strategies) |
| Self-hosted | No | Yes |
| MCP/A2A | No | Yes |
| DDoS Protection | Yes (Cloudflare infra) | Backpressure middleware only |
| Setup Effort | Minutes | Docker + config |

---

### 3.5 Kong AI Gateway

**URL**: https://konghq.com/products/kong-ai-gateway
**License**: Apache 2.0 (Kong Gateway OSS) + Enterprise tier
**GitHub Stars**: ~40,000+ (Kong Gateway overall, as of early 2025)

#### Core Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| **Routing** | Advanced | Content-based routing, load balancing, canary, blue-green |
| **Caching** | Yes | Semantic caching, response caching |
| **Guardrails** | Yes | AI prompt/response guardrails plugin, PII detection |
| **Observability** | Excellent | Prometheus, Datadog, Splunk, OpenTelemetry integration |
| **Auth** | Enterprise | OAuth2, JWT, OIDC, mTLS, LDAP, RBAC |
| **Rate Limiting** | Enterprise | Advanced multi-dimensional rate limiting, quotas |

#### Differentiating Features

- **Enterprise API management heritage**: 10+ years of API gateway experience
- **Plugin ecosystem**: 100+ plugins (authentication, traffic control, observability, etc.)
- **AI-specific plugins**: AI Prompt Guard, AI Rate Limiting, AI Semantic Cache, AI Request Transformer
- **Multi-LLM routing**: Route to different models based on request content, headers, or metadata
- **Hybrid deployment**: Cloud, on-prem, Kubernetes, edge
- **Service mesh integration**: Works with Kong Mesh / Kuma
- **Compliance**: SOC2, HIPAA-ready, PCI DSS support

#### Supported Providers

OpenAI, Azure OpenAI, Anthropic, Cohere, Mistral, Llama (self-hosted), and others via AI Proxy plugin.

#### Protocol Support

- OpenAI-compatible via AI Proxy plugin
- REST API gateway (any protocol)
- GraphQL support
- gRPC support
- No native MCP or A2A support (but extensible via custom plugins)

#### Plugin/Extension Model

- **Lua plugins**: Primary extension mechanism (high performance)
- **Go plugins**: For compute-intensive tasks
- **External plugins**: gRPC-based plugin protocol for any language
- **Declarative configuration**: YAML/JSON config with deck CLI

#### Pricing

- **Kong Gateway (OSS)**: Free, community-supported
- **Kong Gateway Enterprise**: Starts at ~$50K/year (includes AI plugins)
- **Konnect (SaaS)**: Usage-based managed platform
- **Enterprise AI features**: Available in Enterprise tier only

#### Competitive Assessment vs RouteIQ

Kong is the enterprise incumbent. Its strengths are in mature API management (auth, rate limiting, compliance) rather than AI-specific innovation. Kong's AI capabilities are bolt-on plugins rather than a native AI-first architecture.

| Dimension | Kong AI | RouteIQ |
|-----------|---------|---------|
| Routing Intelligence | Content-based rules | ML-based (18+ learned strategies) |
| Plugin Ecosystem | 100+ mature plugins | 3 built-in plugins (extensible) |
| Enterprise Auth | OAuth2, OIDC, mTLS, LDAP | Admin keys, RBAC, policy engine |
| Compliance Certs | SOC2, HIPAA, PCI | None (self-hosted control) |
| AI-Native Design | Bolted-on AI plugins | Built for AI from day one |
| MCP/A2A | No | Yes |
| MLOps | No | Yes (train, deploy, hot-reload) |
| Pricing | $50K+/year (Enterprise) | Free (MIT) |

---

### 3.6 Google Apigee with AI Extensions

**URL**: https://cloud.google.com/apigee
**License**: Proprietary (Google Cloud product)

#### Core Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| **Routing** | Advanced | Proxy-based routing, traffic management, API versioning |
| **Caching** | Yes | Response caching, key-value maps |
| **Guardrails** | Yes | Content safety via Vertex AI integration |
| **Observability** | Enterprise | Cloud Monitoring, Cloud Logging, custom analytics |
| **Auth** | Enterprise | OAuth2, API keys, JWT, SAML, integration with Google IAM |
| **Rate Limiting** | Enterprise | Spike arrest, quota enforcement, monetization |

#### Differentiating Features

- **Vertex AI integration**: Native integration with Google's AI platform
- **API monetization**: Built-in monetization for API products
- **Developer portal**: Self-service portal for API consumers
- **API analytics**: Business-level analytics and reporting
- **Full lifecycle API management**: Design, publish, monitor, retire

#### Protocol Support

- REST, gRPC, GraphQL
- OpenAI-compatible (via mediation policies)
- Vertex AI native
- No MCP or A2A support

#### Pricing

- **Apigee Standard**: ~$500/month base
- **Apigee Enterprise**: Custom pricing ($100K+/year)
- **Apigee X**: Google Cloud-managed, consumption-based

#### Competitive Assessment vs RouteIQ

Apigee is an enterprise API management platform with AI bolted on. It is not AI-native and targets organizations that need full API lifecycle management with some AI capabilities. Not a direct competitor for AI-first use cases.

---

### 3.7 AWS API Gateway for Bedrock

**URL**: https://docs.aws.amazon.com/bedrock/
**License**: Proprietary (AWS service)

#### Core Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| **Routing** | Basic | Bedrock model routing, inference profiles |
| **Caching** | Limited | Via API Gateway caching (not AI-aware) |
| **Guardrails** | Yes | Bedrock Guardrails (content filtering, topic denial, PII, word filters) |
| **Observability** | Good | CloudWatch, X-Ray, CloudTrail |
| **Auth** | Enterprise | IAM, Cognito, API keys, resource policies |
| **Rate Limiting** | Yes | API Gateway throttling, Bedrock quotas |

#### Differentiating Features

- **Bedrock Guardrails**: Most mature guardrails in the market (configurable content filters, topic denial, PII detection, word filters, contextual grounding checks)
- **Bedrock Agents**: Native agent orchestration with knowledge bases
- **Cross-region inference**: Automatic cross-region routing for availability
- **Model evaluation**: Built-in model evaluation jobs
- **Fine-tuning**: Managed fine-tuning for supported models
- **Knowledge Bases**: RAG with managed vector store integration
- **IAM integration**: Fine-grained access control via AWS IAM

#### Supported Providers (via Bedrock)

Anthropic (Claude), Meta (Llama), Mistral, Cohere, AI21, Stability AI, Amazon (Titan, Nova). Limited to Bedrock-hosted models.

#### Protocol Support

- Bedrock API (proprietary)
- OpenAI-compatible (via Bedrock converse API, partial)
- Bedrock Agents (proprietary agent protocol)
- No MCP or A2A support as of early 2025

#### Pricing

- **API Gateway**: $3.50 per million requests + data transfer
- **Bedrock**: Per-token pricing varies by model (e.g., Claude Sonnet ~$3/$15 per 1M tokens in/out)
- **Bedrock Guardrails**: $0.75 per 1K text units (for filtering)

#### Competitive Assessment vs RouteIQ

AWS is the incumbent cloud provider. Bedrock Guardrails are the gold standard for content filtering. However, AWS locks users into the Bedrock ecosystem. RouteIQ can sit in front of Bedrock (via LiteLLM) and add multi-provider routing intelligence.

| Dimension | AWS Bedrock | RouteIQ |
|-----------|-------------|---------|
| Guardrails | Best-in-class (content filtering) | Policy engine (rules-based) |
| Provider Lock-in | Bedrock models only | 100+ providers |
| ML Routing | No | Yes (18+ strategies) |
| Self-hosted | No | Yes |
| MCP/A2A | No | Yes |
| Cost | Usage-based (can be expensive) | Free (MIT) |

---

### 3.8 Martian (Model Router)

**URL**: https://withmartian.com
**License**: Proprietary (SaaS)
**Funding**: Raised $9M Series A (2024)

#### Core Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| **Routing** | Core focus | ML-based model selection, cost-quality optimization |
| **Caching** | No | Routing-only service |
| **Guardrails** | No | Not a gateway |
| **Observability** | Basic | Routing decision logs |
| **Auth** | Basic | API key |
| **Rate Limiting** | No | Not applicable |

#### Differentiating Features

- **ML-based model router**: Core product is an API that recommends the best model for each prompt
- **Cost-quality frontier**: Optimizes along the Pareto frontier of cost vs. quality
- **Benchmark-trained**: Models trained on extensive LLM benchmarks
- **Simple API**: Send prompt, get recommended model + confidence score
- **No infrastructure**: Just a routing recommendation service

#### Protocol Support

- REST API (custom)
- Returns routing recommendations (not a proxy)

#### Pricing

- **Free tier**: Limited requests
- **Usage-based**: Per-routing-decision pricing

#### Competitive Assessment vs RouteIQ

Martian is the closest philosophical competitor -- both use ML for routing decisions. However, Martian is a SaaS routing recommendation service, while RouteIQ is a self-hosted full gateway with integrated ML routing.

| Dimension | Martian | RouteIQ |
|-----------|---------|---------|
| ML Routing | Yes (core product) | Yes (18+ strategies) |
| Full Gateway | No (recommendation API) | Yes |
| Self-hosted | No | Yes |
| Training on Your Data | No (benchmark-trained) | Yes (MLOps pipeline) |
| Hot-reload Models | No | Yes |
| A/B Testing | No | Yes (deterministic hashing) |

---

### 3.9 Unify.ai

**URL**: https://unify.ai
**License**: Proprietary (SaaS)

#### Core Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| **Routing** | Core focus | Benchmark-based routing, cost optimization, quality-speed tradeoffs |
| **Caching** | No | Not a caching layer |
| **Guardrails** | No | Not a guardrails platform |
| **Observability** | Basic | Usage analytics |
| **Auth** | Basic | API key |
| **Rate Limiting** | Basic | Account-level limits |

#### Differentiating Features

- **Benchmark-based routing**: Routes based on published LLM benchmark scores
- **Cost optimization**: Automatically selects cheapest model meeting quality threshold
- **Provider comparison**: Dashboard showing quality/cost/speed across providers
- **OpenAI-compatible**: Drop-in replacement API
- **Dynamic routing**: Adjusts routing based on real-time provider availability

#### Supported Providers

OpenAI, Anthropic, Google, Mistral, Together, Groq, Perplexity, Fireworks, and others (endpoints vary).

#### Protocol Support

- OpenAI-compatible API
- No MCP or A2A

#### Pricing

- **Free tier**: Limited credits
- **Pay-as-you-go**: Markup on model costs

#### Competitive Assessment vs RouteIQ

Unify.ai routes based on published benchmarks; RouteIQ routes based on ML models trained on your actual production data. RouteIQ's approach is more personalized but requires more setup.

| Dimension | Unify.ai | RouteIQ |
|-----------|----------|---------|
| Routing Basis | Public benchmarks | Your production data (ML) |
| Setup Effort | API key swap | Deploy + train models |
| Personalization | None (generic benchmarks) | Fully personalized |
| Self-hosted | No | Yes |
| Gateway Features | Minimal | Full (auth, RBAC, policies) |

---

### 3.10 Other Notable Players

#### RouteLLM (Open Source)

**URL**: https://github.com/lm-sys/RouteLLM
**License**: Apache 2.0
**GitHub Stars**: ~3,000+

An open-source framework for LLM routing from LMSYS (the Chatbot Arena team). Provides routing strategies based on quality-cost tradeoffs. RouteIQ's LLMRouter strategies are conceptually related.

- **Strengths**: Academic rigor, Chatbot Arena data, open-source
- **Weaknesses**: Not a gateway (just a routing library), no production infrastructure

#### Braintrust

**URL**: https://braintrust.dev
**License**: Proprietary (SaaS)

AI product development platform focused on evaluation and observability. Includes a proxy layer for routing.

- **Strengths**: Best-in-class eval framework, prompt playground, dataset management
- **Weaknesses**: Not primarily a gateway, evaluation-focused

#### Semantic Router

**URL**: https://github.com/aurelio-labs/semantic-router
**License**: MIT

A lightweight library for intent-based routing using semantic similarity. Not a gateway, but relevant as a routing primitive.

#### OneAPI

**URL**: https://github.com/songquanpeng/one-api
**License**: MIT
**GitHub Stars**: ~20,000+ (popular in Chinese AI community)

Open-source API management for LLMs. Strong in the Chinese AI ecosystem with support for Chinese model providers.

#### Gloo Gateway (Solo.io)

An Envoy-based API gateway with AI extensions. Enterprise-focused with service mesh integration.

---

## 4. RouteIQ Capability Assessment

Based on analysis of the RouteIQ codebase at `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/`, the following capabilities are confirmed:

### 4.1 Routing Intelligence (STRONG LEAD)

**Source**: `strategies.py`, `strategy_registry.py`, `routing_strategy_patch.py`

- 18+ ML routing strategies: KNN, MLP, SVM, Matrix Factorization, ELO, Hybrid, RouterDC (BERT-based), CausalLM (GPT-2), Graph Neural Network, AutoMix
- A/B testing with deterministic hash-based assignment
- Hot-reload for zero-downtime model updates
- Strategy versioning and staged promotion
- Fallback chains on routing failure
- Telemetry emission for every routing decision (versioned contract v1)

### 4.2 Multi-Protocol Gateway (STRONG LEAD)

**Source**: `mcp_gateway.py`, `mcp_jsonrpc.py`, `mcp_sse_transport.py`, `mcp_parity.py`, `a2a_gateway.py`, `routes.py`

- **LLM Proxy**: OpenAI-compatible chat/completions (via LiteLLM -- 100+ providers)
- **MCP Gateway**: 5 transport surfaces (JSON-RPC, SSE, REST, parity, proxy)
- **A2A Gateway**: Agent-to-Agent protocol with agent registry, streaming support
- **Skills**: Anthropic Computer Use, Bash, Text Editor skills

### 4.3 Security and Enterprise Hardening (COMPETITIVE)

**Source**: `auth.py`, `rbac.py`, `policy_engine.py`, `quota.py`, `audit.py`, `url_security.py`

- Two-tier auth (admin + user) with fail-closed design
- RBAC with hierarchical permissions (mcp.server.write, a2a.agent.write, etc.)
- OPA-style policy engine (pre-request, ASGI-level, fail-open/fail-closed)
- Multi-dimensional quotas (requests, tokens, spend) with Redis-backed enforcement
- Postgres-backed audit logging with fail-closed mode
- SSRF protection with deny-by-default, DNS rebinding protection, allowlists
- Secret scrubbing in error logs
- ML model artifact verification (Ed25519/HMAC signatures)

### 4.4 Observability (COMPETITIVE)

**Source**: `observability.py`, `telemetry_contracts.py`, `router_decision_callback.py`, `mcp_tracing.py`, `a2a_tracing.py`

- Full OpenTelemetry integration (traces, metrics, logs)
- Router decision span attributes (strategy, model_selected, score, latency_ms, outcome)
- Versioned telemetry contracts for stable MLOps extraction
- MCP and A2A tracing with W3C trace context propagation
- Prometheus metrics export
- Integration with Jaeger, Grafana, and other OTel backends

### 4.5 Resilience (COMPETITIVE)

**Source**: `resilience.py`, `http_client_pool.py`, `leader_election.py`

- Backpressure middleware (concurrency limits, load shedding, 503 response)
- Circuit breakers (CLOSED/OPEN/HALF_OPEN states) for external dependencies
- Graceful drain manager for zero-downtime shutdown
- Shared HTTP client pool with connection reuse
- HA leader election (PostgreSQL-backed lease lock)

### 4.6 Configuration and Operations (COMPETITIVE)

**Source**: `config_loader.py`, `config_sync.py`, `hot_reload.py`

- YAML config with S3/GCS download support
- ETag-based config change detection
- Hot-reload for config and routing models
- Background config sync (HA leader-only)
- Kubernetes health probes (liveness + readiness)

### 4.7 Plugin System (EMERGING)

**Source**: `gateway/plugin_manager.py`, `gateway/plugins/`

- GatewayPlugin base class with startup/shutdown lifecycle
- Capability declarations, dependency resolution, priority ordering
- Failure modes (continue, abort, quarantine)
- Security: allowlist, capability restrictions
- 3 built-in plugins: evaluator, skills_discovery, upskill_evaluator
- Plugin middleware with on_request/on_response hooks

---

## 5. Feature Comparison Matrix

Legend: FULL = full support, PARTIAL = partial/basic support, NONE = not supported, N/A = not applicable

| Feature | RouteIQ | LiteLLM | Portkey | Helicone | Cloudflare | Kong AI | AWS Bedrock | Martian | Unify.ai |
|---------|---------|---------|---------|----------|------------|---------|-------------|---------|----------|
| **ML Routing** | FULL (18+) | NONE | NONE | NONE | NONE | NONE | NONE | FULL | PARTIAL |
| **Rule-based Routing** | FULL | FULL | FULL | NONE | PARTIAL | FULL | PARTIAL | NONE | NONE |
| **A/B Testing (routing)** | FULL | NONE | PARTIAL | PARTIAL (prompts) | NONE | PARTIAL | PARTIAL | NONE | NONE |
| **Semantic Caching** | NONE | PARTIAL | FULL | NONE | NONE | FULL | NONE | NONE | NONE |
| **Response Caching** | PARTIAL | FULL | FULL | FULL | FULL | FULL | PARTIAL | NONE | NONE |
| **Content Guardrails** | NONE | PARTIAL | FULL | NONE | PARTIAL | FULL | FULL | NONE | NONE |
| **PII Detection** | NONE | PARTIAL | FULL | NONE | NONE | FULL | FULL | NONE | NONE |
| **OTel Observability** | FULL | PARTIAL | PARTIAL | NONE | NONE | FULL | PARTIAL | NONE | NONE |
| **Analytics Dashboard** | NONE | PARTIAL | FULL | FULL | FULL | FULL | FULL | PARTIAL | PARTIAL |
| **Prompt Management** | NONE | NONE | FULL | FULL | NONE | NONE | NONE | NONE | NONE |
| **100+ Providers** | FULL | FULL | FULL | PARTIAL | PARTIAL | PARTIAL | LIMITED | N/A | PARTIAL |
| **OpenAI-Compatible** | FULL | FULL | FULL | FULL | PARTIAL | FULL | PARTIAL | NONE | FULL |
| **MCP Protocol** | FULL | PARTIAL | NONE | NONE | NONE | NONE | NONE | NONE | NONE |
| **A2A Protocol** | FULL | PARTIAL | NONE | NONE | NONE | NONE | NONE | NONE | NONE |
| **Plugin System** | FULL | PARTIAL | PARTIAL | NONE | NONE | FULL | N/A | NONE | NONE |
| **RBAC** | FULL | PARTIAL | PARTIAL | NONE | PARTIAL | FULL | FULL | NONE | NONE |
| **Policy Engine** | FULL | NONE | NONE | NONE | NONE | PARTIAL | PARTIAL | NONE | NONE |
| **Audit Logging** | FULL | PARTIAL | PARTIAL | FULL | PARTIAL | FULL | FULL | NONE | NONE |
| **SSRF Protection** | FULL | NONE | NONE | N/A | N/A | PARTIAL | N/A | N/A | N/A |
| **Circuit Breakers** | FULL | NONE | PARTIAL | NONE | N/A | FULL | N/A | NONE | NONE |
| **HA / Leader Election** | FULL | PARTIAL | N/A | N/A | N/A | FULL | N/A | N/A | N/A |
| **MLOps Pipeline** | FULL | NONE | NONE | NONE | NONE | NONE | PARTIAL | NONE | NONE |
| **Hot Reload** | FULL | PARTIAL | PARTIAL | N/A | N/A | FULL | N/A | NONE | NONE |
| **Quota Enforcement** | FULL | PARTIAL | PARTIAL | NONE | PARTIAL | FULL | FULL | NONE | NONE |
| **Self-hosted** | FULL | FULL | FULL | FULL | NONE | FULL | NONE | NONE | NONE |
| **Managed/SaaS** | NONE | PARTIAL | FULL | FULL | FULL | FULL | FULL | FULL | FULL |
| **Pricing** | Free (MIT) | Free/Ent. | Free/Paid | Free/Paid | Free/Paid | $50K+/yr | Usage-based | Usage-based | Usage-based |

*Note: RouteIQ's "FULL" for provider support and OpenAI compatibility is inherited from LiteLLM upstream.*

---

## 6. Gap Analysis

### 6.1 Features Where RouteIQ Leads

These are unique differentiators with no close competitor:

| Feature | RouteIQ Advantage | Nearest Competitor |
|---------|-------------------|-------------------|
| **ML-based routing (18+ strategies)** | Trained on production data, personalized | Martian (benchmark-trained, not personalized) |
| **MLOps feedback loop** | Observe -> Train -> Deploy -> Hot-reload | No competitor has this |
| **Multi-protocol gateway (MCP + A2A)** | 5 MCP surfaces + A2A agent registry | LiteLLM (basic support only) |
| **Routing A/B testing** | Deterministic hashing, strategy staging, experiment telemetry | No competitor has this level |
| **SSRF protection** | Deny-by-default, DNS rebinding defense, dual validation | No competitor has dedicated SSRF module |
| **Policy engine (OPA-style)** | Pre-request ASGI-level, fail-open/closed | Kong (partial, via plugins) |
| **Model artifact verification** | Ed25519/HMAC signature verification for ML models | No competitor has this |

### 6.2 Features Where RouteIQ is Competitive

RouteIQ is at parity or slightly behind market leaders:

| Feature | RouteIQ Status | Market Leader | Gap Size |
|---------|---------------|---------------|----------|
| **OTel observability** | Full (traces, metrics, logs) | Kong AI (mature ecosystem) | Small |
| **RBAC** | Hierarchical permissions | Kong (OAuth2, OIDC, mTLS) | Medium -- auth protocols |
| **Quota enforcement** | Multi-dimensional, Redis-backed | Kong, AWS | Small |
| **Audit logging** | Postgres-backed, fail-closed | Kong, AWS CloudTrail | Small |
| **Circuit breakers** | Per-service, configurable | Kong, Istio | Small |
| **Plugin system** | Lifecycle-managed, capability-scoped | Kong (100+ plugins) | Medium -- ecosystem size |
| **HA deployment** | Leader election, graceful drain | Kong (mature HA) | Small |

### 6.3 Features Where RouteIQ Has Gaps

These are significant market expectations where RouteIQ is behind:

| Gap | Severity | Market Expectation | Impact |
|-----|----------|-------------------|--------|
| **No semantic caching** | HIGH | Portkey, Kong have it; estimated 30-50% cost savings | Major cost optimization missing |
| **No analytics dashboard/UI** | HIGH | Every competitor has a web UI; RouteIQ requires external tools (Jaeger/Grafana) | Poor developer experience |
| **No content guardrails** | HIGH | Portkey, Kong, AWS Bedrock have content filtering, PII detection, topic blocking | Enterprise blocker |
| **No prompt management** | MEDIUM | Portkey, Helicone offer template versioning, A/B testing | Developer productivity gap |
| **No managed SaaS option** | MEDIUM | All competitors offer hosted/managed deployment | Limits adoption for smaller teams |
| **No multi-modal routing** | MEDIUM | Image, audio, video generation growing; routing only covers text/chat | Future-readiness gap |
| **Limited response caching** | MEDIUM | RouteIQ relies on LiteLLM's caching; no AI-aware caching layer | Cost optimization gap |
| **No SDK/client libraries** | LOW-MEDIUM | Portkey has Python/JS SDKs with retry, fallback built in | Integration friction |
| **Single-worker constraint** | LOW-MEDIUM | Monkey-patching requires 1 uvicorn worker; limits single-node throughput | Scalability concern |
| **No streaming guardrails** | LOW | Real-time content filtering on streaming responses | Edge case but growing need |

### 6.4 Inherited Capabilities (via LiteLLM)

RouteIQ benefits significantly from LiteLLM upstream but should not over-claim:

- 100+ provider support
- OpenAI API compatibility
- Virtual keys and spend tracking
- Basic response caching (Redis)
- LiteLLM Admin UI (basic)
- SSO (Enterprise tier)

**Risk**: LiteLLM upstream changes could break RouteIQ's monkey-patching approach. The single-worker constraint is a direct consequence of this integration pattern.

---

## 7. Industry Trends

### 7.1 Agentic Infrastructure (2025-2026)

The industry is shifting from "LLM proxy" to "AI agent infrastructure." Key developments:

- **MCP adoption**: Anthropic's Model Context Protocol is becoming the standard for tool integration. RouteIQ's 5-surface MCP support is ahead of the market.
- **A2A protocol**: Google's Agent-to-Agent protocol enables multi-agent orchestration. RouteIQ's A2A gateway is early-mover advantage.
- **Agent observability**: Tracing agent chains, tool calls, and multi-step reasoning flows. RouteIQ has MCP and A2A tracing.

**RouteIQ positioning**: STRONG -- early investment in MCP and A2A is paying off.

### 7.2 Cost Intelligence (2025)

Organizations are demanding cost optimization beyond simple routing:

- **Semantic caching**: 30-50% cost reduction for repeated/similar queries
- **Prompt compression**: Reduce token count without losing quality
- **Model downsizing**: Automatically route simple queries to cheaper models
- **Budget enforcement**: Hard caps on spend per team/project

**RouteIQ positioning**: PARTIAL -- ML routing can downsize models, quota enforcement caps spend, but semantic caching and prompt compression are missing.

### 7.3 Guardrails and Safety (2025)

Enterprise AI governance is becoming mandatory:

- **Content filtering**: Block harmful/inappropriate content in prompts and responses
- **PII detection**: Redact sensitive data before sending to LLM providers
- **Topic restriction**: Prevent models from discussing unauthorized topics
- **Hallucination detection**: Verify responses against source data
- **Regulatory compliance**: EU AI Act, NIST AI RMF requirements

**RouteIQ positioning**: WEAK -- policy engine handles access control rules but not content-level guardrails. This is a significant enterprise gap.

### 7.4 Developer Experience (2025)

AI gateway adoption is driven by developer experience:

- **Dashboard/UI**: Visual analytics, request inspection, debugging tools
- **One-line integration**: SDKs with built-in retry, fallback, caching
- **Prompt playgrounds**: Interactive prompt testing and comparison
- **Eval frameworks**: Integrated evaluation for quality assurance

**RouteIQ positioning**: WEAK -- no built-in UI, no SDK, no prompt playground. Relies on external tools (Jaeger, Grafana) for observability.

### 7.5 Edge and Multi-Region (2025-2026)

Low-latency AI is driving edge deployment:

- **Edge caching**: Cache responses at CDN edge (Cloudflare leads here)
- **Multi-region routing**: Route to nearest provider endpoint
- **Latency-aware routing**: Real-time latency monitoring for routing decisions

**RouteIQ positioning**: PARTIAL -- latency-based routing exists but no edge deployment or multi-region awareness.

---

## 8. Prioritized Recommendations

The following improvements are ranked by **impact** (market value, user demand) and **effort** (implementation complexity, risk).

### Priority 1: Semantic Caching Layer

**Impact**: HIGH | **Effort**: MEDIUM | **Timeline**: 4-6 weeks

**Rationale**: Every major competitor offers semantic caching. Estimated 30-50% cost reduction for production workloads. This is the single highest-ROI feature missing from RouteIQ.

**Implementation approach**:
- Add embedding-based similarity matching before routing (reuse sentence-transformers from KNN strategy)
- Redis-backed cache with configurable TTL and similarity threshold
- Cache key: embedding vector of prompt; cache value: response + metadata
- Integrate with existing `http_client_pool.py` and observability
- Feature-flagged via `ROUTEIQ_SEMANTIC_CACHE_ENABLED`

**Files to modify**: New module `semantic_cache.py`, integrate in `gateway/app.py` middleware chain

---

### Priority 2: Content Guardrails Framework

**Impact**: HIGH | **Effort**: MEDIUM-HIGH | **Timeline**: 6-8 weeks

**Rationale**: Content guardrails are an enterprise blocker. AWS Bedrock Guardrails sets the market standard. RouteIQ needs at minimum: content filtering, PII detection, and topic restriction.

**Implementation approach**:
- Create `guardrails/` plugin package with pluggable guardrail providers
- Built-in guardrails: keyword blocking, regex-based PII detection, topic classification
- Integration points: pre-request (prompt filtering) and post-response (response filtering)
- Support external providers: Lakera Guard, Presidio, AWS Comprehend
- Streaming-safe: Buffer minimum tokens for content classification
- Feature-flagged via `ROUTEIQ_GUARDRAILS_ENABLED`

**Files to modify**: New `guardrails/` module, integrate via plugin system in `gateway/plugin_manager.py`

---

### Priority 3: Analytics Dashboard (Web UI)

**Impact**: HIGH | **Effort**: HIGH | **Timeline**: 8-12 weeks

**Rationale**: Every competitor has a web UI. RouteIQ's reliance on external tools (Jaeger, Grafana) creates friction for adoption. A dashboard dramatically improves developer experience.

**Implementation approach**:
- Lightweight React/Next.js dashboard served alongside the gateway
- Core views: request log, cost breakdown, routing decisions, strategy performance
- Read from OTel data (query Jaeger/OTLP backend) or direct from Postgres audit logs
- Admin panel: strategy management, A/B experiment configuration, plugin status
- Optional: serve as static files from the gateway itself

**Alternative (lower effort)**: Pre-built Grafana dashboards + provisioning scripts (2-3 weeks). This provides 70% of the value at 20% of the effort.

---

### Priority 4: Client SDKs (Python + TypeScript)

**Impact**: MEDIUM-HIGH | **Effort**: MEDIUM | **Timeline**: 4-6 weeks

**Rationale**: Portkey's adoption is partly driven by easy-to-use SDKs. Client libraries reduce integration friction and enable client-side features (retry, fallback, caching).

**Implementation approach**:
- Python SDK: Thin wrapper over `openai` SDK with RouteIQ-specific headers
- TypeScript SDK: Thin wrapper over `openai` npm package
- Features: automatic retry, fallback configuration, request tagging, cost tracking
- Publish to PyPI and npm

---

### Priority 5: Prompt Management and Versioning

**Impact**: MEDIUM | **Effort**: MEDIUM | **Timeline**: 4-6 weeks

**Rationale**: Portkey and Helicone both offer prompt templates with versioning. This is increasingly expected by development teams.

**Implementation approach**:
- Prompt template storage (Postgres-backed)
- Version control with rollback
- Variable interpolation (Jinja2 or similar)
- A/B testing of prompt variants (integrate with existing A/B testing in `strategy_registry.py`)
- API endpoints for CRUD operations on templates

---

### Priority 6: Multi-Modal Routing Support

**Impact**: MEDIUM | **Effort**: MEDIUM | **Timeline**: 4-6 weeks

**Rationale**: Image generation (DALL-E, Midjourney), audio (Whisper, ElevenLabs), and video (Sora, Runway) are growing. Routing intelligence should extend beyond text.

**Implementation approach**:
- Extend routing strategy interface to handle modality-specific metadata
- Add modality-aware cost models
- Support image/audio/video endpoints in the routing layer
- LiteLLM already supports many multi-modal providers

---

### Priority 7: Enterprise Auth Protocols (OIDC/OAuth2)

**Impact**: MEDIUM | **Effort**: MEDIUM | **Timeline**: 4-6 weeks

**Rationale**: Enterprise customers expect OIDC/OAuth2 integration. RouteIQ's current auth (API keys + RBAC) is functional but not enterprise-standard.

**Implementation approach**:
- OIDC provider integration for user authentication
- JWT validation middleware
- Claims-based RBAC (map OIDC claims to RouteIQ permissions)
- LiteLLM Enterprise has SSO -- evaluate extending rather than rebuilding

---

### Priority 8: Streaming Guardrails

**Impact**: MEDIUM-LOW | **Effort**: HIGH | **Timeline**: 6-8 weeks

**Rationale**: Content filtering on streaming responses is technically challenging but increasingly required for real-time applications.

**Implementation approach**:
- Token-buffered content classification (accumulate N tokens, classify, release)
- Sliding window approach for continuous monitoring
- Integrate with guardrails framework (Priority 2)

---

### Priority 9: Edge Deployment Support

**Impact**: MEDIUM-LOW | **Effort**: HIGH | **Timeline**: 8-12 weeks

**Rationale**: Edge caching reduces latency and cost. Cloudflare leads here. RouteIQ could offer self-hosted edge nodes.

**Implementation approach**:
- Lightweight edge proxy (cache + forward to central gateway)
- Multi-region-aware routing (route to nearest provider endpoint)
- Cache synchronization between edge and central

---

### Priority 10: Resolve Single-Worker Constraint

**Impact**: LOW-MEDIUM | **Effort**: HIGH | **Timeline**: 6-10 weeks

**Rationale**: The monkey-patching approach requires a single uvicorn worker, limiting single-node throughput. For high-scale deployments, this forces horizontal scaling.

**Implementation approach**:
- Option A: Move to a proper LiteLLM extension mechanism (if upstream supports it)
- Option B: Use shared memory or multiprocessing-safe state for patches
- Option C: Accept the constraint and document horizontal scaling patterns (Kubernetes HPA)
- Recommendation: Option C is pragmatic -- Kubernetes horizontal scaling is standard practice

---

## 9. Appendix: RouteIQ Source Inventory

### Source Modules Analyzed

All modules at `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/`:

| Module | Purpose |
|--------|---------|
| `strategies.py` | 18+ ML routing strategies (KNN, MLP, SVM, ELO, MF, hybrid, etc.) |
| `strategy_registry.py` | A/B testing, hot-swap, routing pipeline |
| `routing_strategy_patch.py` | LiteLLM Router monkey-patch |
| `router_decision_callback.py` | Routing decision telemetry (TG4.1) |
| `gateway/app.py` | FastAPI app factory (composition root) |
| `gateway/plugin_manager.py` | Plugin lifecycle with dependency resolution |
| `gateway/plugin_middleware.py` | Plugin request/response hooks |
| `gateway/plugin_callback_bridge.py` | Plugin callback integration |
| `routes.py` | All API routers and endpoint definitions |
| `mcp_gateway.py` | MCP server registry and tool discovery |
| `mcp_jsonrpc.py` | Native MCP JSON-RPC 2.0 (for Claude Desktop) |
| `mcp_sse_transport.py` | MCP SSE streaming transport |
| `mcp_parity.py` | Upstream-compatible `/v1/mcp/*` aliases |
| `mcp_tracing.py` | OTel instrumentation for MCP |
| `a2a_gateway.py` | A2A agent registry (wraps LiteLLM) |
| `a2a_tracing.py` | OTel instrumentation for A2A |
| `observability.py` | OpenTelemetry init (traces, metrics, logs) |
| `telemetry_contracts.py` | Versioned telemetry event schemas |
| `auth.py` | Admin auth, RequestID middleware, secret scrubbing |
| `rbac.py` | Role-based access control |
| `policy_engine.py` | OPA-style policy evaluation middleware |
| `quota.py` | Per-team/per-key quota enforcement |
| `audit.py` | Audit logging |
| `resilience.py` | Backpressure, drain manager, circuit breakers |
| `http_client_pool.py` | Shared httpx.AsyncClient pool |
| `config_loader.py` | YAML config + S3/GCS download |
| `config_sync.py` | Background config sync (S3 ETag-based) |
| `hot_reload.py` | Filesystem-watching config hot-reload |
| `leader_election.py` | HA leader election (Redis-based) |
| `model_artifacts.py` | ML model verification (hash, signature) |
| `url_security.py` | SSRF protection |
| `startup.py` | CLI entry point |
| `database.py` | Database utilities |

### Built-in Plugins

At `/Users/baladita/Documents/DevBox/RouteIQ/src/litellm_llmrouter/gateway/plugins/`:

- `evaluator.py` -- Response quality evaluation
- `skills_discovery.py` -- Skill capability discovery
- `upskill_evaluator.py` -- Upskill quality evaluation

### Key Dependencies

From `/Users/baladita/Documents/DevBox/RouteIQ/pyproject.toml`:

- Python 3.14+
- LiteLLM >= 1.81.3
- FastAPI >= 0.109.0
- OpenTelemetry (full stack)
- Redis >= 5.0.0
- A2A SDK >= 0.2.0
- sentence-transformers >= 5.2.0 (for KNN routing)
- scikit-learn >= 1.3.0 (for ML strategies)

---

*Report generated February 2025. Based on codebase analysis of RouteIQ at `/Users/baladita/Documents/DevBox/RouteIQ/` and industry knowledge current through early 2025.*
