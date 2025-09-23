# ML 0.1 Alpha Roadmap

This roadmap defines the concrete scope, sequencing, gates, and acceptance criteria for the 0.1 Alpha of the Nautilus Trader ML layer. It aligns with Coding Standards, Testing Strategy, and the Event‑Driven ML Pipeline checklist.

References

- Architecture: ml/docs/architecture/{ml_integration_architecture.md, event_driven_ml_pipeline_exploration.md, unified_observability.md}
- Implementation plans: ml/docs/implementation/{event_driven_ml_pipeline_checklist.md, domain_bookkeeping_plan.md}
- Context docs: ml/docs/context/* (actors, stores, features, models, strategies, monitoring)
- Standards: ml/docs/development/CODING_STANDARDS.md, ml/tests/docs/TESTING_STRATEGY.md, CLAUDE.md

Goals (Alpha)

- Event‑driven ML pipeline wired end‑to‑end with optional bus and DB‑first registries.
- Unified observability DTOs persisted off hot path with seed dashboards and runbook.
- Hard gates for lint/types/tests; soft gate for performance micro‑benchmarks.
- Hot‑path budgets preserved (<5 ms P99 end‑to‑end signal; zero‑allocation patterns).

Out‑of‑Scope (post‑alpha)

- Intelligent automation (drift/fallback), async persistence worker rollout by default, and production broker integrations are tracked in Post‑Alpha Backlog.

Milestones

- M0 Guardrails & Baselines (days 1–3)
- M1 Bus Integration & Status Enum (days 4–9)
- M2 Observability & Topic Scheme Parity (days 10–14)
- M3 Performance/CB Skeleton + Docs Sync (days 15–20)

M0 — Guardrails & Baselines

Deliverables

- [x] CI gates in place: ruff, mypy strict (ml/), pytest with −m "not prototype", validate‑metrics, validate‑events.
- [x] Performance micro‑bench job added (soft gate): feature compute P99, inference P99, end‑to‑end P99 measured and uploaded as artifacts (no fail yet).
- [x] DataStore ingestion test uplift (JSON backend first; Postgres path skippable) implemented with deterministic unit tests.

Tasks

- [x] Add CI job: micro‑bench harness for features/inference/signal; upload JSON report artifacts.
- [x] Ensure all new/changed code imports Prometheus via ml.common.metrics_bootstrap (validate‑metrics green).
- [x] Ensure Stage constants are used (validate‑events green); remove raw literals in changed code.

Acceptance

- CI green with new micro‑bench job running successfully; Coding Standards alignment verified.

M1 — Event Status Enum + Actor Bridge

Deliverables

- [x] EventStatus enum introduced and enforced across emitters (DataRegistry/DataStore/FeatureStore/ModelStore/StrategyStore/MLSignalActor payloads); DB fields continue to store string values via EventStatus.value.
- [x] Actor‑side publish bridge wired; store‑level publish disabled when actor path enabled (mutual exclusion). Non‑blocking publish from actor thread only.
- [x] Deterministic correlation_id attached to emitted payloads; registries/stores remain the source of truth.

Tasks

- [x] Add EventStatus enum in ml.config.events; migrate emitters from string literals.
- [x] Update tests and contracts for status semantics and payload validation.
- [x] Wire MLSignalActor bridge; ensure try/except and no hot‑path allocations.
- [x] Add tests: actor‑thread boundary, mutual exclusion with store publishers, non‑blocking behavior (in‑memory bus), payload schema/status.

Acceptance

- All emitters use EventStatus.value; unit/contract tests pass; actor bridge path observed in tests; store path disabled when actor path enabled.

M2 — Topic Scheme Parity + Observability UX

Deliverables

- [x] End‑to‑end topic scheme selection: domain_op (default) or stage_first; all stores and actors respect MessageBusConfig and build_topic_for_stage(scheme=..., prefix=...).
- [x] Observability pipeline: DTO builders + service + persistence verified end‑to‑end; seed dashboards and ops runbook delivered.

Tasks

- [x] Ensure DataStore/FeatureStore/ModelStore/StrategyStore and actor use build_topic_for_stage consistently; honor env for scheme/prefix; add property tests comparing schemes.
- [x] Observability: validate DTO builders, service façade, and persistence integration under basic load; wire minimal dashboards (latency histograms, watermarks, event rates, CB state, pool status) and document runbook paths.
- [x] DataStore ingestion tests (JSON backend): cover success/failure emissions, watermark updates, correlation_id determinism, and optional bus publish.

Acceptance

- Contract/property tests for topics pass for both schemes; observability JSONL/DB persistence validated; dashboards render with sample data.

M3 — Performance/CB Skeleton + Docs Sync

Deliverables

- [x] Performance budgets measured and reported in CI micro‑bench (soft gate) with trend capture.
- [x] Circuit breaker/backpressure skeleton: basic policies and metrics (gauges/counters) instrumented and unit‑tested.
- [x] Documentation refresh: architecture/implementation/context docs synced to reflect productionized tests and event/bus decisions.

Tasks

- [x] Integrate CB skeleton into actors/stores (metrics, transitions); add unit test for state transitions.
- [x] Add backpressure gauges and counters via metrics_bootstrap; validate metrics import hygiene.
- [x] Docs updates across ml/docs/architecture, ml/docs/implementation, ml/docs/context; link this roadmap.

Acceptance

- CI green; micro‑bench trends available; CB skeleton tests pass; docs updated and cross‑linked.

Quality Gates (Alpha)

- Lint: make ruff (clean)
- Types: uv run --active --no-sync mypy ml --strict (zero errors)
- Tests: make pytest with −m 'not prototype' (green); ≥90% coverage for ml/ (new/changed modules uphold threshold)
- Events/Metrics: make validate-events; make validate-metrics (green)
- Performance: micro‑bench job runs and uploads P99 report (soft gate)

Post‑Alpha Backlog (Beta and beyond)

- Performance Gates → Hard
  - Promote micro‑bench thresholds to CI hard gates once tails stabilize.

- Circuit Breakers + Backpressure Policies
  - Full policies on actors/stores (open/half‑open/close), idempotent retries, backpressure actions; dedicated dashboards and runbook.

- Cross‑Domain Cascade & Lineage
  - End‑to‑end lineage propagation with correlation_id; property/contract tests for ordering, retries, watermarks; consumer templates (aggregator, DLQ/retry, lineage writer).

- Schema Evolution & Event Triggers
  - Versioned manifests, compatibility checks, migration scripts; replace polling with event triggers gated by watermarks; CI fixture migration job.

- Observability UX Expansion
  - Grafana dashboards for event rates, stage latencies, watermarks, CB state, engine pool telemetry; add alert coverage.
  - Seeded Observability row panels for async worker enqueued rate and backpressure drops (DELIVERED).
  - Added Async Queue Depth stat and timeseries panels (DELIVERED).
  - Added Ingestion row panels: ingest rate (by dataset), watermark lag (max), ingest errors (DELIVERED).
  - Added Consumers/Aggregator row panels: buffer size, flushed rate, duplicates rate, watermark lag (DELIVERED).
  - Added alerts: MLIngestErrorsHigh, MLIngestWatermarkLagHigh, MLIngestRateDrop, MLAggregatorDuplicatesHigh, MLAggregatorBufferHigh, MLAggregatorWatermarkLagHigh (DELIVERED).

- Async Persistence Worker (Flagged)
  - DELIVERED behind feature flag in Alpha: `ml/observability/async_worker.py` with bounded queue, non-blocking enqueue, off-thread persistence; integration via `ObservabilityConfig`/`MLIntegrationManager`.
  - DELIVERED optional async DB sink: `ml/observability/async_db_persistence.py` using SQLAlchemy async (`sqlite+aiosqlite`, `postgresql+asyncpg`).
  - CLI: `ml/cli/observability.py start --async` and `status` to control/inspect async worker.
  - Next: extend stress/property tests as load increases.

- Feature Engineering Backlog
  - Fractional differencing (StationarityTransformer) with parity tests; cross‑sectional features (rank/standardize); feature selection/importance tools; enriched L3 trade flow features with hot‑path budget checks.

- Models/Training Backlog
  - Complete TFT teacher with export paths and TorchScript adapter; student distiller enhancements (calibration in ONNX, inference smoke tests, registry manifests with schema hash/lineage); HPO/validation hardening (study persistence, failure behavior, validate_inference_compatibility).

- Databento Ingestion & Store Hardening
- DELIVERED deterministic fixtures: curated TBBO/L2 MBP‑10/Trades DataFrames with manifests (`schema_hash`, `sha256`).
- Adapter contracts: validate mapping to internal schemas; ordering invariants (monotonic sequence/ts), idempotent replay, watermark progression; backpressure + retry semantics.
- Live/backfill bridge: resume from offsets/time windows; retry/backoff on rate limits; reconnection semantics; property tests for partial day boundaries and DST.
  - DELIVERED: provider‑agnostic resume/backoff helper with DST‑aware window planning and tests (`DatabentoIngestor`).
  - DELIVERED: store integration via `SqlMarketDataWriter` + `SqlCoverageProvider` (idempotent writes with ON CONFLICT/OR IGNORE; BRIN guidance) targeting canonical `market_data` (003_market_data.sql); DataStore contracts remain provider‑agnostic.
  - DELIVERED: gap backfill orchestrator integrated with DataRegistry (events + watermarks) and canonical storage (`ml/data/ingest/orchestrator.py`).
  - DELIVERED: DatabentoAPIClient adapter (`ml/data/ingest/databento_adapter.py`) with canonical column mapping for bars/quotes/trades and `ts_event` normalization; CLI supports `--client-mode databento` with `DATABENTO_API_KEY`.
  - DELIVERED observability: ingestion metrics helpers, dashboard row, alerts.
  - Performance: ingestion micro‑benchmarks (CPU, throughput, p95/p99) with budgets and documentation.
  - Acceptance: integration tests pass against offline fixtures; contract/property tests green; dashboards updated; micro‑bench stable within budgets.

- Cross‑Domain Lineage & Consumers
  - DELIVERED consumer templates: AggregatingConsumer (watermark-gated, idempotent), LineageWriter (observability correlation), RetriableConsumer (DLQ/retry).
  - DELIVERED tests: property (ordering/idempotency), unit (DLQ), integration (ingest→aggregate→lineage).
  - Next: batch envelope contracts (optional Pandera schema), consumer metrics hardening, Redis Streams end-to-end example with fixtures.

- Registries/Stores Hardening
  - DELIVERED provider‑agnostic FeatureStore write test using deterministic fixtures (SQLite): idempotent upsert and ordering verified.
  - Schema evolution patterns, dual‑write windows, migration tests (JSON + PG); BRIN/BTREE tuning guidance; ops runbook updates.

- Inference Parity & Guards
  - DELIVERED: Inference Parity Checklist (`ml/docs/implementation/inference_parity_checklist.md`) with verification plan.
  - DELIVERED: Startup parity guards in MLSignalActor (data_requirements L1_ONLY, feature_schema_hash parity, min_bars_warmup, bar_type metadata checks).
  - DELIVERED: Optional parity smoke-check (one‑shot) with metrics `ml_feature_parity_checks_total` and `ml_feature_parity_drift`.

Next (Short‑Term)

- Live/Backfill Bridge
  - DELIVERED: Bootstrap CLI wiring (`ml/cli/ingest_backfill.py`) with coverage modes (sql|catalog), client modes (catalog|databento), state persistence.
  - Persist IngestState between runs (JSON under `checkpoints/` or minimal DB table); add resume examples.
  - DELIVERED: IntegrationManager backfill bootstrap (env‑driven; runs CLI on startup with flags).
- Ingestion Performance Gates
  - CI micro‑bench for ingestion (P95/P99, CPU); convert to hard gates once stable.
- Redis Streams End‑to‑End (Optional)
  - Fixtures → Redis publisher → RedisStreamsConsumer → Aggregator → Lineage with deterministic Redis stub; example + tests.

Risks & Mitigations

- Hot‑path regression: keep bus/observability work off hot path; reuse buffers/labels; pre‑allocate arrays; wrap publish in try/except.
- Pandera/version drift: normalize checks to installed version; pin if needed.
- External adapters in tests: constrain testpaths/markers for ML scope; keep provider integrations mocked.

Ownership & Resourcing (Alpha)

- ML Engineering: EventStatus, actor bridge, topic parity, DataStore tests, CB skeleton, feature micro‑bench.
- Backend: Observability service/persistence, message bus config, in‑memory bus, topic helpers.
- DevOps: CI micro‑bench job, dashboards wiring, validate‑metrics/events gates, artifact retention.
- QA: Contract/property/perf tests; coverage tracking; docs validation.

Definition of Done (Alpha)

- End‑to‑end event flow operating with optional bus; actor‑thread publish path enabled via config, store path disabled when active.
- Observability pipeline writing DTOs off hot path; sample dashboards rendering; runbook available.
- Lint/types/tests/metrics/events gates green; coverage ≥90% for ml/; micro‑bench trends available.
- Documentation synchronized, with this roadmap linked from domain_bookkeeping_plan.md.

Quick Commands

- Types: uv run --active --no-sync mypy ml --strict
- Lint: make ruff
- Tests: make pytest; pytest -q -m 'not prototype'
- Metrics/Events gates: make validate-metrics; make validate-events
- Docs: make docs
- Micro‑bench (local): pytest -q ml/tests/performance -k microbench --benchmark-only

  Key Gaps to Address:

  1. Experiment Tracking & Versioning 🔬

  Missing:
  - Automatic hyperparameter logging
  - Experiment comparison UI
  - Metric visualization over time
  - A/B test results tracking
  - Model lineage visualization

  To Add:
  # Need experiment tracking in MLPipelineOrchestrator
  experiment_id = self.experiment_store.create_experiment(
      name="tft_model_v2",
      params=config.as_dict(),
      tags={"author": user, "branch": git_branch}
  )
  # Log metrics during training
  self.experiment_store.log_metrics(experiment_id, {"loss": 0.23, "sharpe": 1.5})

  2. Distributed Training & Compute 🖥️

  Missing:
  - Kubernetes job orchestration
  - Multi-GPU training support
  - Distributed data processing (Spark/Ray)
  - Resource allocation & scheduling
  - Queue management for jobs

  To Add:
  - Ray/Dask integration for distributed compute
  - Kubernetes operator for ML jobs
  - GPU resource management

  3. Model Serving & Deployment 🚀

  Missing:
  - Model serving endpoints (REST/gRPC)
  - Canary deployments
  - Shadow mode deployments
  - Load balancing across model versions
  - Request batching & caching

  Current: Models are loaded directly in actors
  Need: Dedicated model serving layer with KServe/Seldon-style capabilities

  4. Visualization & Monitoring 📊

  Missing:
  - Training curves visualization
  - Feature importance plots
  - Confusion matrices
  - ROC curves
  - Data drift visualization
  - Model performance degradation alerts

  To Add:
  - TensorBoard integration
  - Custom Grafana dashboards for ML metrics
  - Jupyter notebook integration in dashboard

  5. Collaboration Features 👥

  Missing:
  - User authentication & teams
  - Role-based access control (RBAC)
  - Comments on experiments
  - Shared projects/workspaces
  - Approval workflows for production

  6. Advanced Pipeline Features 🔄

  Missing:
  - DAG visualization
  - Pipeline scheduling (Airflow-style)
  - Conditional execution
  - Pipeline templates
  - Failure recovery & retries

  7. Data Versioning & Lineage 📁

  Missing:
  - Dataset versioning (DVC-style)
  - Data lineage tracking
  - Feature store time travel
  - Dataset comparison tools
  - Data quality monitoring

  8. AutoML Capabilities 🤖

  Missing:
  - Automated feature engineering
  - Neural architecture search
  - Hyperparameter optimization (beyond basic HPO)
  - Automated model selection

  9. Production Monitoring 📡

  Missing:
  - Prediction request logging
  - Model performance monitoring
  - Feature drift detection
  - Outlier detection
  - Champion/challenger testing

  10. Developer Experience 💻

  Missing:
  - CLI for pipeline submission
  - Python SDK for experiment tracking
  - IDE plugins
  - Pipeline YAML/DSL
  - Local pipeline testing

—

Document version: 0.1-alpha
Last updated: 2025-09-10
Status: Active (Alpha)
