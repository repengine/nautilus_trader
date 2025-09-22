# ML Dashboard Enhancement – Revised Roadmap

## Executive Summary
Deliver a production-ready dashboard by strengthening the existing control plane only where the codebase can support it today. Scope is limited to cold-path improvements, measured observability upgrades, and well-defined integrations. Infrastructure that does not yet exist (Redis subscribers, shared circuit breakers) is treated as backlog or explicitly budgeted.

- **Timeline**: ~15–18 working days across 4 phases plus readiness checks
- **Focus**: Hardening, cached polling, Grafana provisioning, optional store insights once wiring exists
- **Guardrails**: `uv run --active --no-sync mypy ml --strict`, `make ruff`, topic builders + enums, zero hot-path impact

## Guiding Constraints
- Reuse implemented primitives: registry facades, `ml.common.metrics_bootstrap`, `ml.common.retry_utils.retry_with_backoff`, and existing config loaders.
- Keep all work in cold-path modules (`DashboardService`, Flask views, Grafana helpers, templates) and defensive best-effort patterns.
- When a capability is missing (e.g., Redis subscription, store connectors), either build it with an explicit estimate or leave it in backlog; never assume it exists.
- Gate every phase with targeted tests (`make pytest -k dashboard`), validators, and documentation updates.

## Phase 0 – Baseline Hardening (Days 1-4)
### Objectives
- Cut redundant registry work with TTL caching and progressive fallbacks.
- Improve resilience of external probes using retry/backoff that already exists.
- Establish regression coverage around configuration and failure handling.

### Tasks
- [x] Wrap registry getters in cache mixins (e.g., `CacheMixin`, `MicrostructureCache`) with 30–60 s TTL and expose cache-hit metrics.
- [x] Add DummyRegistry/DummyController fallbacks when JSON/DB backends are unavailable, emitting `ml_fallback_activations_total` via `metrics_bootstrap` (gated by `ML_ALLOW_DUMMY=1`).
- [x] Replace direct calls in `_safe_get`/registry lookups with `retry_with_backoff` (from `ml.common.retry_utils`) plus capped attempts; track failures with counters.
- [x] Extend unit tests to cover cache expiration, fallback activation, and retry behaviour; document updated behaviour in `ml/docs/architecture/dashboard_control_plane.md`.

### Deliverables
- Hardened service layer with caching + retries, metrics reflecting cache/fallback usage, and coverage proving behaviour.

## Phase 1 – Event Visibility via Cached Polling (Days 5-8)
### Objectives
- Provide recent event visibility without relying on a non-existent subscriber.
- Reduce Redis load by caching responses and exposing filtered views.
- Lightly enhance the UI for more efficient refresh without full page reloads.

### Tasks
- [x] Implement a polling helper that retrieves `xrevrange` results, caches them in-memory (TTL/size bounded), and serves filtered results to `/api/events`.
- [x] Add background polling hook (optional thread or scheduled call) guarded by env flag; fall back to on-demand fetch with per-request caching when disabled.
- [x] Update `templates/index.html` to use HTMX or timed fetch (e.g., `setInterval` + ETag/If-None-Match) for events, services, and models sections.
- [x] Introduce metrics for polling cadence, cache hits, and Redis failures.
- [x] Add tests covering polling cadence, filter correctness, and cache invalidation.

### Deliverables
- Reliable event endpoint backed by cached polling with defensive fallbacks, plus a UI that updates sections without full reloads.

### Backlog Note
- Building a true Redis subscriber + SSE streaming path remains a separate backlog item (estimate 5–7 days) and is not part of the committed timeline.

## Phase 2 – Observability & Grafana Integration (Days 9-12)
### Objectives
- Provision Grafana dashboards using the existing helpers and embed panels in the dashboard UI.
- Offer cold-path Prometheus snapshots for quick summaries without duplicating metrics.
- Validate observability endpoints end-to-end.

### Tasks
- [x] Extend `ml/dashboard/grafana.py` to compose reusable dashboard bundles using `ml.monitoring.dashboard_factory` panels; add tests with mocked HTTP responses.
- [x] Add an idempotent provisioning hook (manual trigger + optional startup flag) and expose status via API.
- [x] Implement a Prometheus query helper using simple polling (requests + retry_with_backoff) to power P95 latency / request counters on the dashboard cards.
- [x] Embed configurable Grafana iframes in the template with environment-driven base URLs and authentication notes; document setup in README.
- [x] Run `make validate-metrics` and ensure new metrics/topics stay compliant.

### Deliverables
- Grafana provisioning flow with accompanying tests, embedded panels in the UI, and Prometheus-backed status cards.

## Phase 3 – ML Insights & Store Integrations (Days 13-18)
### Objectives
- Surface ML health indicators once store access is properly wired.
- Harden authentication, logging, and documentation for production operation.
- Complete validation, performance checks, and deployment notes.

### Prerequisite Decision
- **If store connectors remain unavailable**, defer store-based insights and instead lean on Prometheus + registry metadata. Record the gap and backlog the work.
- **If store connectors are prioritized**, schedule the following tasks with additional environment preparation (PostgreSQL connection, credentials secured via config).

### Tasks (assuming store wiring proceeds)
- [x] Introduce lightweight adapters that obtain Feature/Model/Strategy/Data store clients via existing factory functions; ensure they respect configuration and cold-path constraints.
- [x] Add health summaries (data freshness, model staleness, feature drift) powered by store APIs; guard with fallbacks when stores are offline.
- [x] Expand metrics to capture store query latencies and fallback activations; include these in Prometheus snapshots.
- [x] Enhance `_require_token` with token rotation guidance, optional expiry, and audit logging hooks (structured logging + metrics).
- [x] Update documentation/runbooks and re-run `make validate-events`, `make validate-metrics`, `pytest -q ml/tests/performance -k microbench --benchmark-only` when loops are introduced.

### Deliverables
- Either: store-backed health panels with fallbacks and documentation, or a clearly documented backlog item explaining why store wiring is deferred.
- Refreshed auth/logging guidance and validated dashboard build.

## Cross-Cutting Quality Gates
- Every phase: lint (`make ruff`), strict typing, targeted tests (`make pytest -k dashboard`), documentation updates.
- Maintain alphabetical `__all__` exports and avoid importing internal modules directly.
- For new metrics or events, ensure enums from `ml.config.events` and `ml.common.message_topics` builders are used; rerun validators after changes.

## Success Metrics
- [ ] Registry-backed endpoints P95 ≤ 200 ms with caching enabled.
- [ ] Event polling cache hit rate ≥ 70% under normal load; fallback behaviour logged.
- [ ] Grafana provisioning succeeds ≥ 95% with retries; embedded panels render without manual steps.
- [ ] When store integrations land, health panels respond within SLA (< 750 ms) or gracefully fall back with metrics emitted.
- [ ] All validators and agreed test suites remain green before release.

## Sequencing Overview
1. Phase 0 (Days 1-4): Caching, fallbacks, retry/backoff, tests, docs.
2. Phase 1 (Days 5-8): Event polling cache + UI refresh + metrics.
3. Phase 2 (Days 9-12): Grafana provisioning, Prometheus snapshots, embeds.
4. Phase 3 (Days 13-18): Store integration (if funded) + ML insights + auth/doc polish.

## Deferred / Out of Scope
- React/SPA rewrite (Flask + HTMX/vanilla JS remains the approach).
- Celery or new async queues (message bus + polling cover the use cases).
- Custom Prometheus collectors (reuse `ml.common.metrics_bootstrap` inputs).
- Redis subscriber/SSE infrastructure (tracked as separate backlog with explicit estimate).

Ship each phase behind configuration flags where feasible, making it easy to roll out incrementally without affecting current dashboard stability.
