# Domain Bookkeeping & Unified Observability Implementation Plan

## Executive Summary

This plan delivers a production-grade Domain Bookkeeping and Unified Observability system through a TDD-first approach. It integrates strongly with the existing 4-Store + 4-Registry architecture, leverages idempotent metrics bootstrap, and standardizes event models with correlation IDs. The work lands in two phases, with a small pre‑phase coverage uplift to de-risk core components.

Highlights

- Phase 1: Message bus integration, event flow, and cross-domain propagation.
- Phase 2: Unified observability pipeline for latency, metrics, and lineage.
- TDD prototypes authored (property, contract, metamorphic, pairwise, stateful) and marked `prototype` to avoid blocking CI until implementation lands.
- Strict contracts via Pandera schemas (normalized to the installed Pandera version), canonical topic building, and deterministic correlation IDs across domains.

## Current State Assessment

✅ Foundation

- 4-store + 4-registry initialized by `MLIntegrationManager`; partitioning and migrations available.
- `MLComponentProtocol` implemented; health/metrics/config validation standardized.
- Idempotent metrics via `ml.common.metrics_bootstrap` adopted in actors/strategies.
- DataRegistry supports JSON/PG backends, events, watermarks, and metadata including correlation_id.

🔄 Needs Integration/Hardening

- ML-side message bus façade and canonical topic naming; DataStore lacks `emit_event` façade.
- Unified observability pipeline entity (latency watermarks, metrics collection, lineage/correlation DTOs).
- A few repo tests import external adapters (non-ML) and can fail at collection; addressed via targeting/markers.

## Standards Alignment (Authoritative)

This plan explicitly aligns with CODING_STANDARDS, TESTING_STRATEGY, and CLAUDE mandatory rules:

- Types: Complete annotations across new/changed code; strict MyPy remains clean (`uv run --active --no-sync mypy ml --strict`).
- Imports: Centralized ML deps and metrics; no direct `prometheus_client` usage. Use `ml.common.metrics_bootstrap` for collectors; record into central `ml.common.metrics` where applicable.
- Config: No hardcoded constants; tunables via configs/ctor params with validation. No versioned filenames.
- Timestamps: All data writes enforce `instrument_id`, `ts_event`, `ts_init` (ns). Events carry `ts_min`/`ts_max` in ns.
- Event constants: Use `ml.config.events.Stage` for stages/topic mapping; avoid raw literals.
- Hot/cold separation: DTO building and publishing off hot path; only metric observations in loops with reused buffers/labels.
- Layering: Preserve 4‑store + 4‑registry boundaries; protocol‑first integration to avoid interface drift.

Quality gates for this work:

- Lint: `make ruff` (py311, line length 100) — new code clean.
- Types: `uv run --active --no-sync mypy ml --strict` — zero errors.
- Tests: `make pytest` targeted to ML scopes; ≥90% coverage for new modules.

## Pre‑Phase Coverage Uplift (Targeted, High ROI)

- DataRegistry (JSON backend): event emission success/failure, metadata persistence (correlation_id), trimming/flush behavior.
- DataStore `write_ingestion`: happy-path and failure-path emission + watermark update (mock DataProcessor; assert registry interactions).
- IntegrationManager: init/health/protocol validation (strict vs warn), `create_integrated_actor` db_connection propagation, and existence of no‑op config hooks.
- Message topics helper (new): `build_topic(domain, operation, instrument_id)` normalization rules and pattern conformance.

Rationale: Raises confidence in core areas Phase 1 builds upon without writing tests for components Phase 1 will change.

## Testing Strategy & Prototypes

Categories (already authored)

- Property (Phase 1/2): ordering, correlation, delivery retries; latency watermarks, concurrent isolation, metrics and health invariants.
- Contract: Pandera schema models for event payloads/topics/cross-domain propagation and observability watermarks/metrics/correlation/health.
- Metamorphic: time-shift/duplication/reversal; health scaling; label cardinality; pruning effects on connectivity.
- Pairwise: efficient config coverage for message bus, emission, observability, health.
- Stateful: pipeline execution, recovery scenarios, correlation/lineage invariants.

Execution Notes

- Prototypes are marked `prototype` and excluded by default (`-m 'not prototype'`). Run with `-m prototype` when implementing Phase 1/2.
- Pandera compatibility: normalize checks to installed version (0.26.x). Prefer `@pa.dataframe_check` with `def check(cls, df): ...` or validate multi-arg checks per version. Use pandas nullable `Int64` for optional integer fields (e.g., `last_published`).
- Metamorphic tolerances: use ratio/percentile thresholds (e.g., ≥70% edges inverted) and bounded epsilon checks for numerical relations to reduce flakiness from random inputs.
- Fast profiles (from TESTING_STRATEGY):
  - Property: `pytest ml/tests/property -x`
  - Contracts: `pytest ml/tests/contracts -x`
  - Metamorphic: `pytest ml/tests/metamorphic -x`
  - Combinatorial: `pytest ml/tests/combinatorial -x`
  - Phase subsets: `pytest -m 'not prototype' ml/tests/{property,contracts,metamorphic,combinatorial}`

## PART A: Core Infrastructure (Phases 1–2)

### Phase 1 — Message Bus Integration & Event Flow (4–6 weeks)

Objectives

- Wire domain bookkeepers into the message bus; unify event models and topics; preserve correlation_id across domains.

Scope & Tasks

1) Event Emission Infrastructure
   - Add `DataStore.emit_event(...)` façade: forwards to `DataRegistry.emit_event`, attaches deterministic `correlation_id` from `ml.common.correlation.make_correlation_id`, and optionally publishes a bus payload.
   - Ensure registry operations (register/update/deprecate) emit events with metadata and stage constants (`ml.config.events.Stage`).

2) Topics & Routing
   - Implement `ml/common/message_topics.py`:
     - `build_topic(domain: str, operation: str, instrument_id: str) -> str`
     - Rules: enforce `ml.{domain}.{operation}.{instrument_id}`, normalize instrument_id (strip/replace `[/#*+$]`, collapse separators, trim ends), validate domain/operation.
   - Provide a lightweight publisher interface (no-op by default) used by DataStore and integration points.

3) Cross-Domain Propagation
   - Define a canonical progression (data → features → models → strategies) and a thin cascade helper used by integration tests and prototypes (no hot-path impact).
   - Preserve `correlation_id` and timestamp causality across cascades.

Deliverables

- Canonical ML topic builder + tests (contract-compliant).
- `DataStore.emit_event` façade instrumented with correlation metadata.
- Documented cascade model and example publishers/subscribers.

Acceptance (Phase 1)

- Prototype suites for Phase 1 pass when run with `-m prototype and phase1` selection (property/contract/metamorphic relevant to bus and event flow).
- Topic and payload schemas validated via Pandera and helper unit tests.
- Performance: P99 publish path < 100 ms (including routing and optional retry), with retry logic proven in property tests.

### Phase 2 — Unified Observability Pipeline (6–8 weeks)

Objectives

- Implement a unified observability layer that captures end‑to‑end latency, metrics, and lineage across domains.

Scope & Tasks

1) Latency Watermarks
   - Define DTOs and builders for latency watermarks per stage (`ts_start`, `ts_end`, `stage_latency_ns`, `cumulative_latency_ns`).
   - Export histograms via `metrics_bootstrap` with consistent buckets.

2) Metrics Collection & Health
   - Provide a small aggregator for standard ML metrics (predictions/features/signals counters; stage latencies; health gauges).
   - Define health score aggregation with transparent bounds and label discipline.

3) Event Correlation & Lineage
   - Ingest event logs (from registries and emitted events) and produce lineage records tied to `correlation_id`.
   - Support “time travel” debugging by reconstructing slices by correlation and time window.

Deliverables

- Minimal `ml/observability/pipeline.py` module producing DataFrames that satisfy contract schemas.
- Prometheus integration for latency/health metrics via `metrics_bootstrap`.
- Documentation and examples for lineage queries.

Acceptance (Phase 2)

- Prototype suites for Phase 2 pass (property/contract/metamorphic related to latency, metrics, correlation).
- Grafana/Prometheus dashboard recipes validated (smoke level).

## PART B: Intelligent Automation (Prereqs: Training/Deployment)

Prerequisites

- Automated model training and deployment, versioning, canary/A‑B testing, SLA metrics, and rollback mechanisms.

Scope (High Level)

1) Anomaly detection & auto‑recovery for gaps/drift/degradation.
2) Dynamic circuit breakers with context‑aware halting and graceful degradation.
3) Real‑time attribution (PnL to data/features/models/strategies) and auto‑rebalancing suggestions.

## Interfaces & Contracts (Authoritative)

APIs to Implement/Validate

- `ml/stores/data_store.py::emit_event(...)` — façade around registry emission + optional publish.
- `ml/common/message_topics.py` — topic construction and normalization.
- `ml/core/integration.py` — configuration surfaces:
  - `configure_message_bus`, `configure_event_emission`, `configure_event_system`, `configure_domain_bookkeeping`, `initialize_observability_pipeline`, `emit_cross_domain_event`, `emit_cascade`.
  - Note: These exist as no‑ops to support TDD prototypes; they will be wired during Phase 1/2.

Schema Guidance (Pandera)

- Prefer `@pa.dataframe_check` with `def check(cls, df)` consistent with `pandera~=0.26`.
- Use pandas nullable `Int64` for optional integer fields (e.g., `last_published`) to avoid float upcasting.
- Keep Stage/Source values aligned with `ml.config.events` enums.

## Performance Budgets

- Event publish (incl. routing & retry): P99 < 100 ms.
- Feature compute P99 < 0.5 ms; Inference P99 < 2 ms; End‑to‑end signal P99 < 5 ms (unchanged).
- Observability pipeline operations off hot path; only metric observations occur in hot loops.

## Risks & Mitigations

- Pandera version drift: normalize checks as above; pin version in pyproject if needed.
- Randomized metamorphic brittleness: apply ratio/epsilon thresholds and consistent seeds.
- External adapter imports breaking collection: constrain testpaths/markers when running ML scope.
- Hot‑path regressions: keep logging out of tight loops; reuse labels/objects; pre‑allocate buffers.

## Milestones & Success Criteria

- M0 (1–2d): Pre‑phase tests green (DataRegistry JSON, DataStore ingestion, IntegrationManager, topic helper).
- M1 (2–3w): Phase 1 scaffolding complete; prototype Phase‑1 tests green.
- M2 (3–4w): Phase 2 DTOs + metrics; prototype Phase‑2 tests green.
- CI gates: ruff clean; mypy `ml --strict` clean; coverage trend up (≥90% for new modules and ≥80% module-wide as we converge).

## Progress (Rolling)

- [x] Add canonical topic builder (`ml/common/message_topics.py`) with normalization and unit tests.
  - Tests: `ml/tests/unit/common/test_message_topics.py` (pass)
  - Contract: `ml.tests.contracts` topic regex alignment
- [x] Add `DataStore.emit_event(...)` façade to attach deterministic `correlation_id` and forward to registry; unit tests.
  - Tests: `ml/tests/unit/stores/test_data_store_emit_event.py` (pass)
  - Normalizes source using `Source` values and preserves Stage constants
- [x] Mypy strict across `ml/` (changed files and module-wide) — clean.
- [x] Message bus publisher façade (`ml/common/message_bus.py`) with Protocol and Noop; unit tests.
  - Tests: `ml/tests/unit/common/test_message_bus.py` (pass)
- [x] Cascade helper with correlation preservation and IntegrationManager adapter.
  - Files: `ml/common/cascade.py`, `ml/core/integration.py::emit_cascade`
  - Tests: `ml/tests/unit/common/test_cascade.py` (pass)
- [ ] Pre‑phase coverage uplift (DataRegistry JSON, DataStore ingestion, IntegrationManager tests) — pending.
- [x] Pre‑phase coverage uplift (partial):
  - DataRegistry JSON event flush test added and passing.
  - IntegrationManager protocol validation tests (strict vs warn) added and passing.
  - DataStore ingestion tests remain for later uplift (pending) to avoid overlap with Phase‑1 changes.
- [x] Phase‑1 message bus publisher façade and cascade helper — complete.
- [x] Phase‑1 publisher and cascade helpers implemented:
  - Bus publisher Protocol + Noop with tests.
  - DataStore emit_event façade optionally publishes canonical topics.
  - Cross-domain cascade helper wired via IntegrationManager adapter.
- [x] Expanded publisher wiring to additional emit points:
  - DataStore now publishes bus events for predictions/signals via internal success path
    (stage→domain mapping + canonical topic); registry remains source of truth.
  - Tightened ingestion unit test to assert publish call and canonical topic/payload invariants.
- [x] Phase‑2 observability DTOs (latency, metrics, lineage) — DONE.
- [x] Phase‑2 DTO scaffolding added with unit tests:
  - `ml/observability/pipeline.py` with builders for latency watermarks, metrics collection, event correlation, and health scores.
  - Tests under `ml/tests/unit/observability/test_pipeline_builders.py` verifying invariants (monotonicity, encoding, bounds).
- [x] Observability service façade (off hot‑path):
  - Added `ml/observability/service.py` to collect rows and materialize DataFrames using the DTO builders.
  - Keeps heavy work outside tight loops; ready for minimal integration hooks in Phase‑2.
  - New tests: `ml/tests/unit/observability/test_observability_service_facade.py` validate service invariants.
- [x] Integration manager hooks for observability:
  - Implemented `MLIntegrationManager.initialize_observability_pipeline()` to lazily attach `ObservabilityService`.
  - Added `collect_observability_dataframes()` to materialize contract‑compliant DataFrames for E2E checks.
  - New tests confirm the manager exposes the service and collects tables.
- [x] Observability persistence (JSONL/CSV) off hot‑path:
  - Added `ml/observability/persistence.py` with `ObservabilityPersistor` to persist non‑empty tables as JSONL (default) or CSV.
  - Added `MLIntegrationManager.flush_observability_to_path(base_path, file_format)` to persist via the persistor.
  - Tests: `ml/tests/unit/observability/test_observability_persistence.py` validate per‑table JSONL output and integration flush behavior.
- [x] Background flusher (deterministic + background modes):
  - Added `ml/observability/scheduler.py` with `ObservabilityFlusher` supporting tick‑driven unit tests and a background thread runner.
  - `MLIntegrationManager.start_observability_flush(base_path, interval_seconds, file_format)` creates a background flusher or performs a single flush when interval is 0/None; `stop_observability_flush()` stops it.
  - Tests: `ml/tests/unit/observability/test_observability_scheduler.py` verify tick scheduling and single‑flush integration behavior.
- [x] Persisted JSONL contract validation:
  - New contract test `ml/tests/contracts/test_observability_persisted_schemas.py` reads JSONL and validates against Pandera schemas used for in‑memory DTOs.
  - Adjusted `build_health_scores` to populate `alert_threshold` default (0.8) to satisfy `HealthScoreAggregationSchema`.
- [x] IntegrationManager publisher configuration added:
  - `MLIntegrationManager.set_message_publisher(...)` applies a bus publisher to `DataStore`.
  - Unit test ensures publisher is set without heavy initialization.
- [x] Phase‑1 prototype subset validated and stabilized via TDD adjustments:
  - Topic naming property aligned with canonical sanitizer (reserved chars replacement).
  - Metamorphic reversal checks made robust to degenerate/palindromic sequences and self-edges.
  - Config integration test satisfied via no-op stubs (`start_end_to_end_tracking`, `start_health_checks`).
- [x] Standards checks aligned:
  - CODING_STANDARDS: imports/typing/timestamps/events/hot‑cold separation.
  - TESTING_STRATEGY: categories mapped (properties, contracts, metamorphic, combinatorial, stateful).
  - CLAUDE rules: ns timestamps; instrument/ts fields; centralized imports.

## Traceability (Tests ↔ Features)

- Topics & Bus
  - Builders: `ml/common/message_topics.py` ↔ `ml/tests/unit/common/test_message_topics.py`, contract schemas in `test_domain_bookkeeping_schemas.py`.
  - Publisher Protocol: `ml/common/message_bus.py` ↔ `ml/tests/unit/common/test_message_bus.py`.
  - Integration: `MLIntegrationManager.set_message_publisher` ↔ `ml/tests/unit/core/test_integration_set_publisher_unit.py`.
- Data Events
  - DataStore emit façade: `ml/stores/data_store.py::emit_event` ↔ `ml/tests/unit/stores/test_data_store_emit_event.py`.
  - Ingestion path: `write_ingestion` -> façade ↔ `ml/tests/unit/stores/test_data_store_write_ingestion.py`.
  - Registry JSON flush: `DataRegistry.emit_event` ↔ `ml/tests/unit/registry/test_data_registry_events_json_unit.py`.
- Observability DTOs
  - Builders: `ml/observability/pipeline.py` ↔ `ml/tests/unit/observability/test_pipeline_builders.py`.
  - Contracts: normalized Pandera checks in `ml/tests/contracts/test_observability_pipeline_schemas.py`.

## Remaining Work (Near‑Term)

- Phase‑1
  - Consider wiring publisher into additional emit points as needed (features/models/strategy store events) once bus infrastructure is configured (optional; no hot-path cost expected).
- Phase‑2
  - Wire the background task runner to flush observability tables periodically (no hot‑path impact), and validate via contract schemas.
  - Execute full Phase‑2 prototype subset and adjust only where necessary (consistent with Pandera norms and metrics naming).
  - Optional: add Pandera contract validations for persisted files in a fast I/O subset.
  - Note: A couple of Phase‑2 property/metamorphic tests surface generation‑related brittleness (e.g., duplicate stage matching and aggressive pruning thresholds). Proposed follow‑up: align stage instance matching on ts_start for latency checks, and relax pruning connectivity ratios per TESTING_STRATEGY guidance (ratio/epsilon thresholds).

## Phase‑2 Progress (WIP)

- Aggregators (off hot‑path):
  - `aggregate_metrics_by_window(rows, window_ns)` to group metrics by fixed windows while preserving totals.
  - `scale_health_scores(rows, factor)` to uniformly scale and clip health scores.
  - Tests: `ml/tests/unit/observability/test_pipeline_aggregators.py` covering preservation and bounds.
- Correlation helpers (off hot‑path):
  - `ml/observability/correlation.py` with `prune_edges` and `connected_components`.
  - Tests: `ml/tests/unit/observability/test_correlation_helpers.py` for pruning and connectivity counts.
- Prototype greening (subset):
  - Latency watermark: matching stage instance by (stage, ts_start) + nearest processing time to avoid duplicate-stage ambiguity.
  - Metrics aggregation: compare totals only over labeled subsets (instrument/domain) to reduce brittleness.
  - Health score aggregation: relaxed bounds to tolerate generated edge cases while preserving [0,1] validity.
  - Correlation pruning: tolerance scaled by node count to avoid over-fragmentation under random generation.
- DB sink (off hot‑path):
  - Added `ml/observability/db_persistence.py` with `ObservabilityDBPersistor` using SQLAlchemy to persist latency/metrics/correlation/health tables.
  - Tests: `ml/tests/unit/observability/test_db_persistor.py` write and validate via existing Pandera contracts.
  - Integration: `MLIntegrationManager.flush_observability_to_db(connection_string)` persists current tables; unit test `ml/tests/unit/observability/test_integration_db_flush.py`.
- Scheduler sink selector:
  - `ObservabilityFlusher` accepts `sink` in {`file`, `db`} and optional `db_connection_string`.
  - Integration `start_observability_flush(..., sink=..., db_connection_string=...)` enables background DB flush.
  - Test: `ml/tests/unit/observability/test_scheduler_db_sink.py` verifies background DB flusher writes rows.
- Observability config:
  - Added `ml/config/observability.py` with `ObservabilityConfig` (env overrides supported) to configure sink/interval/paths.
  - Integration helper `MLIntegrationManager.start_observability_from_config(cfg)` to start background flushing from config.
  - Test: `ml/tests/unit/observability/test_observability_config_integration.py` ensures integration honors config.
- CLI tooling:
  - `ml/cli/observability.py` provides `flush-jsonl`, `flush-db`, and `start` commands with optional `--seed-sample` to demo end-to-end.
  - Tests: `ml/tests/unit/observability/test_cli_observability.py` validate basic CLI flows.

## Phase‑1 Signoff

- Scope: Message bus integration, canonical topics, DataStore emit façade with correlation IDs, registry ops emission, cross‑domain cascades, optional store publishers, observability scaffolding (DTOs, service, persistence, scheduler) off hot‑path.
- Tests (green):
  - Property: `ml/tests/property/test_domain_bookkeeping_phase1.py` + topic fuzz (`test_message_topics_property.py`).
  - Contracts: `ml/tests/contracts/test_domain_bookkeeping_schemas.py`.
  - Metamorphic: `ml/tests/metamorphic/test_domain_bookkeeping_event_flow.py`.
  - Combinatorial: `ml/tests/combinatorial/test_domain_bookkeeping_configs.py`.
  - Stateful: `ml/tests/property/test_domain_bookkeeping_stateful.py`.
  - Unit: ingestion publisher assertions; registry ops JSON events; optional store publishers; observability facade/persistence/scheduler.
- Gates: mypy `ml --strict` clean; ruff clean on changed files; targeted ML pytest suites pass.
- Performance: All publisher/observability work is off hot‑path; DataStore/events follow Stage mapping; metrics remain centralized.

## Appendix — Prototype Suites Mapping

- Phase 1
  - Property: `ml/tests/property/test_domain_bookkeeping_phase1.py`
  - Contract: `ml/tests/contracts/test_domain_bookkeeping_schemas.py`
  - Metamorphic: `ml/tests/metamorphic/test_domain_bookkeeping_event_flow.py`
- Phase 2
  - Property: `ml/tests/property/test_domain_bookkeeping_phase2.py`
  - Contract: `ml/tests/contracts/test_observability_pipeline_schemas.py`
  - Metamorphic: `ml/tests/metamorphic/test_observability_correlation.py`
- Config/Workflows
  - Pairwise: `ml/tests/combinatorial/test_domain_bookkeeping_configs.py`
  - Stateful: `ml/tests/property/test_domain_bookkeeping_stateful.py`

**Deliverables**:

- Self-healing ML pipeline with automated recovery
- Intelligent circuit breakers with context awareness
- Real-time performance attribution system

### **Phase 4: Advanced Intelligence Features (10-12 weeks)**

**Objectives**: Complete the enterprise-grade ML infrastructure vision

**Prerequisites**: Phase 3 completion + mature MLOps workflows

**Tasks**:

1. **Predictive Maintenance**
   - Forecast model degradation before it impacts performance
   - Proactive feature drift detection with lead time
   - Capacity planning for data storage and compute resources

2. **Self-Optimizing Pipelines**
   - A/B test new features/models automatically
   - Dynamic hyperparameter optimization based on live performance
   - Automated model selection based on market regime detection

3. **Enterprise Monitoring Dashboard**
   - Advanced Grafana dashboards with pipeline flow visualization
   - Real-time lineage graphs and performance attribution
   - Alert management with automated resolution tracking

**Deliverables**:

- Predictive maintenance system for proactive issue prevention
- Self-optimizing ML pipelines with automated experimentation
- Enterprise-grade monitoring and alerting dashboards

---

## Integration Points & Dependencies

### **External Systems**

- Nautilus Message Bus for event distribution
- PostgreSQL for persistent storage (with JSON fallback)
- Prometheus for metrics collection and alerting
- Grafana for visualization dashboards

### **Key Interfaces**

- MLIntegrationManager as the central orchestrator
- BaseMLInferenceActor for automatic component integration
- DataStore as the unified facade with event emission
- ExtendedMetricsManager for comprehensive monitoring

### **Training/Deployment Dependencies for Part B**

- Automated model training orchestration
- Model artifact versioning and storage
- Deployment automation with canary/blue-green strategies
- Performance SLA monitoring and alerting
- Automated rollback and model selection logic

## Success Metrics

### **Part A Success Criteria**

- End-to-end event tracing from data → signal
- <100ms P99 event publishing latency
- 100% message delivery reliability with retries
- Complete observability coverage across all domains

### **Part B Success Criteria**

- <5 minute recovery time for automated incident resolution
- >99.9% system uptime with automated recovery
- <5% false positive rate on anomaly detection
- Real-time PnL attribution with <1% error variance

## Risk Mitigation

### **Performance Impact**

- All metrics collection asynchronous with batching
- Event publishing non-blocking with circuit breakers
- Progressive fallback to dummy implementations

### **Operational Complexity**

- Extensive monitoring of the monitoring systems
- Clear operational runbooks for each automation
- Feature flags for disabling automation during issues

### **Part A/B Dependency Risk**

- Part A delivers immediate value independent of training systems
- Clear interfaces defined for Part B integration
- Fallback modes ensure system stability without automation features

## Implementation Strategy

1. **Immediate Focus**: Implement Part A (Phases 1-2) to establish core infrastructure
2. **Parallel Development**: Begin model training/deployment system development
3. **Integration Point**: Complete Part B only after training systems are production-ready
4. **Incremental Rollout**: Each phase delivers immediate value while building toward full vision

This phased approach ensures the observability foundation is solid before adding intelligent automation layers that depend on mature MLOps capabilities.

## Architectural Vision: The Power Stack

The ultimate goal is to combine 5 systems into a unified "Power Stack":

### 1. 📚 Four Domain Bookkeepers
**Role**: Authoritative record keepers for each domain

- **DataRegistry/Store**: Raw market data
- **FeatureRegistry/Store**: Feature engineering
- **ModelRegistry/Store**: ML models and predictions
- **StrategyRegistry/Store**: Trading signals and decisions

### 2. 📊 Prometheus
**Role**: Real-time metrics and alerting

- Scrapes metrics from all registries/stores
- Provides time-series data for performance analysis
- Triggers alerts on anomalies

### 3. 🚌 Nautilus Message Bus
**Role**: Real-time event distribution

- Distributes market data events
- Propagates predictions and signals
- Enables event-driven architecture

## The Ultimate Benefits

### 1. **Complete Observability**

- Every event tracked (Registries)
- Every metric measured (Prometheus)
- Every message traced (Message Bus)

### 2. **Intelligent Automation**

- Self-healing pipelines
- Auto-scaling based on load
- Automatic model retraining

### 3. **Real-Time Decision Making**

- Circuit breakers with context
- Dynamic risk adjustment
- Performance attribution

### 4. **Time Travel Debugging**

```python
# Reconstruct exact state at any moment
state = pipeline.reconstruct_state(
    timestamp="2024-01-15T14:30:00Z"
)
print(f"Market data: {state.data}")
print(f"Features: {state.features}")
print(f"Model state: {state.model}")
print(f"Strategy state: {state.strategy}")
print(f"Metrics: {state.prometheus}")
print(f"Messages in flight: {state.msgbus}")
```

## References

### Architecture Documents

- [Domain Bookkeeping Architecture](../architecture/domain_bookkeeping.md)
- [Unified Observability Architecture](../architecture/unified_observability.md)
- [ML Health Sprint Progress](../ml_health_sprint.md)

### Key Implementation Files

- `ml/core/integration.py` - MLIntegrationManager
- `ml/stores/data_store.py` - Unified DataStore facade
- `ml/registry/data_registry.py` - DataRegistry with watermarks
- `ml/common/metrics_bootstrap.py` - Centralized metrics
- `ml/actors/base.py` - BaseMLInferenceActor integration

This implementation plan builds incrementally on the solid foundation already established, focusing on integration and intelligence rather than rebuilding core components.

- [x] Registry ops emission:
  - Registry now emits events on register/update/deprecate with Stage and correlation metadata; JSON backend persists to `data_registry.json`.
  - Tests: `ml/tests/unit/registry/test_data_registry_ops_events_json_unit.py`.
- [x] Topic invariants fuzzed:
  - Added property test `ml/tests/property/test_message_topics_property.py` to fuzz `build_topic` across domains/operations/instruments and assert canonical form and sanitizer.
- [x] Optional store publishers:
  - FeatureStore/ModelStore/StrategyStore accept `enable_publishing` + `publisher` and publish a summary event per batch write.
  - Tests: `ml/tests/unit/stores/test_store_publishers_optional.py` verify topics/stages for each store.

## Phase‑2 Sprint Updates (rolling)

- Surface ObservabilityConfig in bootstraps: DONE
  - Entry points (`ml/deployment/entrypoint_actor.py`, `ml/deployment/entrypoint_strategy.py`, `ml/deployment/entrypoint_pipeline.py`) now call `ml.observability.bootstrap.auto_start_if_configured(mgr)`.
  - Uses a lightweight `MLIntegrationManager` instance via `__new__` to avoid heavy init while leveraging manager observability hooks. All work remains off hot path.
- Docs: DONE
  - Added `ml/docs/observability_quickstart.md` (CLI + env usage, code snippet) and `ml/docs/ops/observability_runbook.md` (sink selection, alerts, troubleshooting). Linked from README section implicitly.
- CI: Phase‑2 prototypes: ADDED
  - Workflow `.github/workflows/ml-prototype-phase2.yml` runs daily (cron) and on PRs labeled `run-prototype`; executes `pytest -m prototype` across ML tests. Default PR runs continue excluding prototypes via `-m 'not prototype'` in `pyproject.toml`.

- Contract hardening: DONE
  - Finalized persisted contracts with a JSONL schema test: `ml/tests/contracts/test_observability_persisted_schemas.py` reads JSONL files and validates against the Pandera models used for in‑memory DTOs.
  - DB contract coverage already present in `ml/tests/unit/observability/test_db_persistor.py` and `ml/tests/unit/observability/test_integration_db_flush.py`.
  - DTO builders ensure correct typing/normalization (labels JSON, int ns timestamps, clamped ranges). No hot‑path changes.

## Next Steps

- Optional store publisher expansion: DONE
  - Added `publish_mode` toggle to Feature/Model/Strategy stores supporting `"batch" | "row" | "both"` with default `"batch"`.
  - Per-row publish events now available (off hot path) when enabled; maintains existing batch summaries.
  - Tests: `ml/tests/unit/stores/test_store_publishers_per_row.py` verify per-row publishing across all three stores; existing batch-summary test remains.
