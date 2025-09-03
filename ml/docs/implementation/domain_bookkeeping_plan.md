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
- [ ] Phase‑1 message bus publisher façade and cascade helper — pending.
- [ ] Phase‑2 observability DTOs (latency, metrics, lineage) — pending.

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
