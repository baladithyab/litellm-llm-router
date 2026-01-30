# Documentation Consolidation Plan

**Status:** Draft
**Date:** 2026-01-30
**Objective:** Consolidate the RouteIQ Gateway documentation into a cohesive, non-redundant set with clear navigation.

## 1. Current State Analysis

The current documentation set consists of ~30 files in the `docs/` directory, plus `README.md` and Helm chart docs. There is significant overlap between "Quickstart" guides, "Deployment" guides, and "Architecture" docs. Some files are extremely detailed (e.g., `moat-mode.md`) while others are thinner.

**Key Issues:**
- **Redundancy:** Multiple files cover deployment (Docker, Compose, K8s) with varying levels of detail.
- **Fragmentation:** MLOps and Observability info is split across "training", "guide", and "architecture" docs.
- **Navigation:** `docs/index.md` is a flat list; needs better grouping.
- **Internal vs. User:** Roadmap and validation plans are mixed with user documentation.

## 2. Proposed Structure

We propose organizing the documentation into the following sections:

1.  **Getting Started**: Quickstarts and introduction.
2.  **Deployment & Operations**: How to run it in production (Docker, K8s, AWS, Air-gapped).
3.  **Core Features**: The main gateways (MCP, A2A, Skills) and capabilities (Plugins, Vector Stores).
4.  **Configuration & Routing**: Deep dives into config options and routing strategies.
5.  **Observability & MLOps**: The closed-loop feedback system.
6.  **Reference**: API docs and architecture.
7.  **Project Plans**: Roadmaps and internal docs (moved to `plans/` or `docs/internal/`).

## 3. Consolidation Action Plan

| Topic | Current File | Proposed Action | Target / Notes |
| :--- | :--- | :--- | :--- |
| **Entry Point** | `docs/index.md` | **Update** | Redesign as the main landing page with the new structure. |
| | `README.md` | **Trim** | Reduce to a high-level overview + link to `docs/index.md`. |
| **Getting Started** | `docs/quickstart-docker-compose.md` | **Keep** | Canonical "Local Dev" quickstart. |
| | `docs/quickstart-ha-compose.md` | **Rename** | `docs/tutorials/ha-quickstart.md` (Advanced tutorial). |
| | `docs/quickstart-otel-compose.md` | **Rename** | `docs/tutorials/observability-quickstart.md`. |
| **Deployment** | `docs/deployment.md` | **Update** | Canonical "Deployment Overview". Absorb general concepts. |
| | `docs/high-availability.md` | **Merge** | Merge unique content (Leader Election) into `docs/deployment.md` or `docs/deployment/ha.md`. |
| | `docs/hot-reloading.md` | **Merge** | Merge into `docs/configuration.md` or `docs/deployment.md`. |
| | `docs/architecture/aws-deployment.md` | **Move** | `docs/deployment/aws.md`. |
| | `deploy/charts/routeiq-gateway/README.md` | **Link** | Link from `docs/deployment/kubernetes.md`. |
| | `docs/moat-mode.md` | **Move** | `docs/deployment/air-gapped.md`. |
| **Features** | `docs/mcp-gateway.md` | **Keep** | |
| | `docs/a2a-gateway.md` | **Keep** | |
| | `docs/skills-gateway.md` | **Keep** | |
| | `docs/vector-stores.md` | **Keep** | |
| | `docs/plugins.md` | **Keep** | |
| **Config & Routing** | `docs/configuration.md` | **Keep** | Canonical Configuration guide. |
| | `docs/routing-strategies.md` | **Keep** | Reference for strategies. |
| | `docs/security.md` | **Keep** | Canonical Security guide. |
| **Observability** | `docs/observability.md` | **Keep** | Canonical Observability guide. |
| | `docs/observability-training.md` | **Merge** | Merge into `docs/mlops-training.md` (Data Prep section). |
| **MLOps** | `docs/mlops-training.md` | **Keep** | Canonical MLOps guide. |
| | `docs/architecture/ml-routing-cloud-native.md` | **Move** | `docs/architecture/mlops-loop.md`. |
| **Reference** | `docs/api-reference.md` | **Keep** | |
| | `docs/architecture/overview.md` | **Keep** | |
| | `docs/litellm-cloud-native-enhancements.md` | **Move** | `docs/architecture/cloud-native.md`. |
| **Internal/Plans** | `docs/TECHNICAL_ROADMAP.md` | **Move** | `plans/technical-roadmap.md`. |
| | `docs/VALIDATION_PLAN.md` | **Move** | `plans/validation-plan.md`. |
| | `docs/implementation-backlog.md` | **Move** | `plans/backlog.md`. |
| | `docs/release-checklist.md` | **Move** | `plans/release-checklist.md`. |
| | `docs/parity-roadmap.md` | **Move** | `plans/parity-roadmap.md`. |
| | `docs/api-parity-analysis.md` | **Move** | `plans/api-parity.md`. |

## 4. Proposed Navigation (`docs/index.md`)

```markdown
# RouteIQ Gateway Documentation

## Getting Started
- [Introduction](index.md)
- [Quickstart: Docker Compose](quickstart-docker-compose.md)
- [Tutorial: High Availability](tutorials/ha-quickstart.md)
- [Tutorial: Observability](tutorials/observability-quickstart.md)

## Deployment & Operations
- [Deployment Overview](deployment.md)
- [Kubernetes (Helm)](../deploy/charts/routeiq-gateway/README.md)
- [AWS Deployment](deployment/aws.md)
- [Air-Gapped / Moat Mode](deployment/air-gapped.md)
- [Security Guide](security.md)

## Core Gateways
- [MCP Gateway](mcp-gateway.md)
- [A2A Gateway](a2a-gateway.md)
- [Skills Gateway](skills-gateway.md)
- [Vector Stores](vector-stores.md)
- [Plugins](plugins.md)

## Configuration & Routing
- [Configuration Guide](configuration.md)
- [Routing Strategies](routing-strategies.md)

## Observability & MLOps
- [Observability Guide](observability.md)
- [MLOps Training Pipeline](mlops-training.md)

## Reference
- [API Reference](api-reference.md)
- [Architecture Overview](architecture/overview.md)
```

## 5. Next Steps

1.  **Create Directories**: `docs/deployment/`, `docs/tutorials/`, `docs/internal/`.
2.  **Move Files**: Execute the moves/renames listed above.
3.  **Merge Content**:
    -   Merge `observability-training.md` into `mlops-training.md`.
    -   Merge `hot-reloading.md` into `configuration.md` (or keep as section in deployment).
    -   Merge `high-availability.md` unique content into `deployment.md`.
4.  **Update Links**: Fix all broken relative links resulting from moves.
5.  **Update Index**: Rewrite `docs/index.md`.
