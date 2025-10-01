# Nautilus ML Dashboard – UI Integration Roadmap

## Purpose

Provide a single, high-level roadmap that connects the existing dashboard PLAN
documents with the planned services layer and enumerates the milestones required
to turn the current stubs into production integrations. This roadmap synthesises
requirements drawn from:

- `README.md` (planning layer summary)
- `IMPLEMENTATION_INDEX.md`
- `PLAN_*.md` specification documents
- `IMPLEMENTATION_PROGRESS.md`

## Guiding Principles

1. **Contract First** – start from the UI promises documented in the PLAN files
   and encode them as typed service interfaces.
2. **4-Store + 4-Registry Compliance** – every service must respect existing ML
   architecture patterns, including progressive fallbacks and protocol typing.
3. **Observability Everywhere** – instrument integrations via
   `ml.common.metrics_bootstrap`, with hot-path safety.
4. **Graceful Degradation** – ensure dummy/cached fallbacks before connecting to
   live Nautilus components.

## Phase 0 – Planning & Alignment (Complete)

- Consolidate documentation (README, implementation index, progress tracker).
- Catalogue UI expectations per domain (`PLAN_actor_management.md`, etc.).
- Outcome: stubs in `integration_layer.py` plus updated planning artifacts
  (current state).

## Phase 1 – Service Foundations (Weeks 1–2 TBD)

**Goals**
- Define stable, typed service facades in `ml/dashboard/services/` (one module per
  domain) without full backend wiring.
- Establish dependency injection via `MLIntegrationManager` and strict protocols.
- Implement request/response DTOs and validation skeletons.

**Key Tasks**
- [ ] Create `actors_service.py`, `metrics_service.py`, `pipelines_service.py`,
  `trading_service.py`, wiring them into `__all__`.
- [ ] Migrate existing placeholder logic out of `integration_layer.py` into the
  new modules; keep integration layer as a thin orchestrator.
- [ ] Add basic unit tests verifying schema validation and metrics hooks.

**Gate Criteria**
- Interfaces documented and type-checked (`uv run --active --no-sync mypy ml --strict`).
- Ruff clean (`make ruff`).
- Tests covering DTO validation.

## Phase 2 – Core Integrations (Weeks 3–5 TBD)

**Goals**
- Connect each service to real ML components with guarded fallbacks.
- Implement progressive fallback chains (Primary → File → Dummy) via
  `MLIntegrationManager`.
- Surface read-only data flows for dashboard displays.

**Key Tasks**
- [ ] Store metrics: implement KPI aggregations using Strategy/Data store APIs,
  add Prometheus query adapters as needed.
- [ ] Actor lifecycle: enable deploy/stop/hot-reload through `BaseMLInferenceActor`
  utilities, including health monitoring hooks.
- [ ] Pipeline triggers: wire to `MLPipelineOrchestrator` (job submission, status
  polling via persistence layer).
- [ ] Trading controls: integrate with `TradingNode` safety checks and circuit
  breakers; wrap publish operations in try/except.

**Gate Criteria**
- Contract tests verifying service responses against seeded dummy stores.
- Updated integration tests (`ml/tests/integration/test_dashboard_ml_integration.py`)
  exercising real components at least in fallback mode.
- Metrics emitted for each service operation (counter + latency histogram).

## Phase 3 – Real-Time & Advanced Features (Weeks 6–8 TBD)

**Goals**
- Deliver real-time updates (WebSocket or SSE) and advanced UI capabilities from
  the PLAN docs (feature designer, strategy builder, API explorer, terminal).
- Extend services to cover write operations with validation and sandboxing.

**Key Tasks**
- [ ] Implement WebSocket broadcaster for metrics/pipeline status, with polling
  fallback and caching.
- [ ] Build feature engineering sandbox (secure execution + validation using
  `FeatureEngineer` protocols).
- [ ] Strategy builder backtest/deploy pipeline hooking into Nautilus backtest
  engine with circuit breaker guards.
- [ ] API explorer: auto-generate OpenAPI specs and provide request testing helpers
  with rate limiting.
- [ ] Terminal/settings: secure command execution, configuration diff & apply
  pipeline.

**Gate Criteria**
- End-to-end tests marked `@pytest.mark.serial` where appropriate (e.g., sandbox).
- Observability dashboards validated via `make validate-metrics`.
- Security review covering sandbox boundaries and TradingNode access.

## Phase 4 – Production Hardening (Weeks 9–10 TBD)

**Goals**
- Optimise performance (hot-path <5 ms), complete resilience features, and deliver
  deployment-ready documentation.

**Key Tasks**
- [ ] Run micro-benchmarks for hot-path service calls (`pytest -q ml/tests/performance -k microbench --benchmark-only`).
- [ ] Finalise fallback activation metrics (`ml_fallback_activations_total` labels).
- [ ] Harden authentication/authorisation for dashboard API entrypoints.
- [ ] Produce operations runbook updates and Grafana dashboards.

**Gate Criteria**
- Perf targets met (<5 ms P99 for hot-path operations).
- All validators (`make validate-events`, `make validate-metrics`, optional
  `make validate-nautilus-patterns`) passing.
- Updated documentation: README, runbook, API reference.

## Cross-Cutting Workstreams

- **Security & Compliance:** integrate secrets handling, sandboxing, and circuit
  breakers across all phases.
- **Testing Strategy:** add contract, property, and integration tests as behaviour
  evolves (see `ml/tests/docs/TESTING_STRATEGY.md`).
- **Documentation:** update planning docs and service README each time behaviour
  changes; regenerate `IMPLEMENTATION_PROGRESS.md` after major milestones.

## Dependencies & Risks

- Availability of underlying ML services (stores, orchestrator, TradingNode).
- Ensuring strict protocol compliance without widening to `Any`.
- Avoiding hot-path regressions when adding observability or network access.
- Coordinating UI expectations as dashboard evolves (keep PLAN docs synced).

## Next Steps

1. Socialise this roadmap with stakeholders and confirm milestone ordering.
2. Break Phase 1 tasks into tickets, assigning ownership per service domain.
3. Schedule regular reviews to reconcile PLAN documents with implemented code.
4. Use `implementation_tracker.py` to keep progress reporting accurate.
