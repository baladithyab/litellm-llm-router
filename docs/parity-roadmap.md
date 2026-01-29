# Feature Parity & Roadmap

This document tracks the feature parity of RouteIQ Gateway compared to standard LiteLLM Proxy and outlines the roadmap for future enhancements.

## Feature Status

| Feature Area | Status | Notes |
|--------------|--------|-------|
| **Core Routing** | ✅ Available | ML-based routing (KNN, SVM), simple shuffle, least-busy. |
| **A2A (Agent-to-Agent)** | ✅ Available | Full support for A2A protocol, agent registration, and invocation. |
| **MCP (Model Context Protocol)** | ✅ Available | Server registration, tool discovery, and invocation (flag-gated). |
| **Skills Gateway** | ✅ Available | Support for Anthropic Computer Use, Bash, and Text Editor skills. |
| **Vector Stores** | ⚠️ Partial | Inherits OpenAI-compatible endpoints. Deep external DB integration (Pinecone, Qdrant) is planned. |
| **Observability** | ✅ Available | OpenTelemetry (OTel) tracing, metrics, and logging. Jaeger integration. |
| **Security** | ✅ Available | SSRF protection (deny-by-default), Admin Auth, Role-based access. |
| **High Availability** | ✅ Available | Redis-backed state sync, multi-replica support, load balancing. |

## Roadmap

Our roadmap is prioritized into P0 (Critical), P1 (High), and P2 (Medium) items.

### P0: Critical Stability & Core Features (Q1 2026)

- [x] **Unified Docker Container**: Single image for LiteLLM + LLMRouter.
- [x] **ML Routing Strategies**: KNN and SVM implementations.
- [ ] **Hot-Reload Config Sync**: Dynamic updates without restart.
- [ ] **Routing Decision Visibility**: Full OTel instrumentation for *why* a model was chosen.

### P1: Enterprise Hardening (Q2 2026)

- [ ] **Backpressure & Load Shedding**: Protect gateway from overload during spikes.
- [ ] **Multi-Replica Trace Correlation**: End-to-end tracing across load balancers.
- [ ] **Durable Audit Export**: Compliance logging to S3/Kafka.
- [ ] **Secret Rotation**: Patterns for dynamic secret updates without downtime.

### P2: Advanced MLOps & Scale (Q3 2026)

- [ ] **Autoscaling Guidance**: KEDA/HPA metrics based on token throughput.
- [ ] **Degraded Mode**: Circuit breakers for DB/Redis failures.
- [ ] **Multi-Region Sync**: Patterns for global deployments.
- [ ] **Automated MLOps Pipeline**: Trace -> Train -> Deploy loop.

## Parity Notes

### LiteLLM Proxy Compatibility

RouteIQ Gateway is built *on top of* LiteLLM Proxy. It inherits 100% of LiteLLM's features unless explicitly overridden.

- **Configuration**: Uses standard `config.yaml`.
- **API**: Fully compatible with OpenAI API format.
- **Providers**: Supports all 100+ LLM providers supported by LiteLLM.

### Key Differences

1.  **Routing Logic**: RouteIQ injects a custom router that uses ML models instead of simple round-robin.
2.  **Security Defaults**: RouteIQ enforces stricter security defaults (e.g., SSRF protection) suitable for enterprise "Moat Mode".
3.  **Protocol Support**: Native support for MCP and A2A protocols alongside standard chat completion.
