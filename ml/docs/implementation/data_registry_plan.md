**Nautilus ML Data Platform Hardening Plan**

- Purpose: implement DataRegistry + DataStore with manifests, contracts, lineage
, watermarks, stage coverage, gap backfill, and training automation.
- Audience: developers building and operating `ml/` pipeline.
- Status: production-ready core; this plan finalizes lineage, contracts, gap rep
air, and training orchestration.

**Scope & Objectives**

- Data contracts: enforce strict dataset schemas and keys (batch/live).
- Lineage + events: record per-stage events and end-to-end coverage.
- Watermarks + coverage: compute lag and completeness per instrument/day.
- Gap repair: T+1 backfill for L2/L3 using L1 coverage. (If L1 is present, and the timestamp is within the last 30 days, pull for L2/L3)
- Observability: metrics and dashboards for coverage, lag, violations.
- Reliability: idempotent writes, backpressure, retries, schema locks.
- Training orchestration: SLO- and drift-driven retraining queues.

**Current Baseline (Where To Look)**

- Data pipeline: `ml/data/scheduler.py` (Databento→Catalog→FeatureStore; Prometh
eus metrics).
- Feature parity: `ml/features/engineering.py`, `ml/features/microstructure.py`,
 `ml/preprocessing/joins.py`.
- Stores: `ml/stores/feature_store.py`, `ml/stores/model_store.py`, `ml/stores/s
trategy_store.py`.
- DB schema (gold): `ml/stores/migrations/001_stores_schema.sql`.
- Health views: `ml/schema/pipeline_health.sql` (freshness, errors, processing).
- Monitoring: `ml/monitoring/README.md`, `ml/monitoring/grafana/GRAFANA_INTEGRAT
ION_PLAN.md`.
- Registry (models/features/strategies): `ml/registry/*`, `ml/docs/context/conte
xt_registry.md`.
- Context docs: `ml/docs/context/context_data.md`, `ml/docs/context/context_stor
es.md`.

**Architecture Additions**

- DataRegistry:
  - Tracks dataset manifests, contracts, lineage edges, events, watermarks.
  - Files to add: `ml/registry/data_registry.py`, migration `ml/stores/migration
s/004_data_registry.sql`.
  - External patterns: Apache Flink event-time watermarks; OpenLineage lineage s
pec.

- DataStore:
  - Typed read/write façade that validates batches against contracts and emits e
vents/watermarks.
  - File to add: `ml/stores/data_store.py`.

- Manifests + Contracts:
  - DatasetManifest: dataset_id, dataset_type (BARS|TRADES|QUOTES|MBP1|TBBO|FEAT
URES|PREDICTIONS|SIGNALS), storage_kind (parquet|postgres), location (path/table
), partitioning, retention, schema (names/dtypes), ts_field (nanoseconds), optio
nal seqno, primary keys, schema_hash, constraints (ranges, nullability), lineage
 (parents), pipeline_signature.
  - DataContract: validation rules mapped to quality flags (types, ranges, uniqu
eness, monotonicity, allowable lateness).

- DB Migration 004 (additions):
  - `ml_dataset_registry`: manifests + constraints + schema_hash.
  - `ml_data_events` (partitioned by ts_event): dataset_id, instrument_id, stage
 (INGESTED|CATALOG_WRITTEN|FEATURE_COMPUTED|PREDICTION_EMITTED|SIGNAL_EMITTED),
source (live|historical|backfill), run_id, ts_min, ts_max, count, seq_min/seq_ma
x, status, error, created_at.
  - `ml_data_watermarks`: dataset_id, instrument_id, source, last_success_ns, la
st_attempt_ns, last_count, completeness_pct.
  - `ml_data_lineage`: transform_id, child_dataset_id, parent_dataset_id, ts_ran
ge, parameters, created_at.
  - Indexing: BRIN on time; lookup `(dataset_id, instrument_id)`; partial indexe
s where useful.

- Instrumentation (emit events + update watermarks):
  - Ingestion: `ml/data/scheduler.py` after `catalog.write_data(...)` → emit `CA
TALOG_WRITTEN`.
  - Feature store: `ml/stores/feature_store.py` after `compute_and_store_histori
cal` and `compute_realtime(store=True)` → emit `FEATURE_COMPUTED`.
  - Predictions: `ml/stores/model_store.py` on batch write → emit `PREDICTION_EM
ITTED`.
  - Signals: `ml/stores/strategy_store.py` on batch write → emit `SIGNAL_EMITTED
`.

- Coverage & Backfill:
  - CLI: `ml/cli/coverage.py`:
    - `coverage report` (per instrument/day): stage coverage %, pass-through cou
nts, lag.
    - `plan backfill` (L1 present, L2/TBBO missing): create backfill jobs.
    - `apply backfill`: dispatch Databento `get_range` jobs with rate/storage th
rottles.
  - Scheduler hook: run daily across previous trading day, update watermarks, pl
an backfills when T+1 available.

- Metrics & Dashboards:
  - Prometheus:
    - `nautilus_ml_data_events_total{dataset,stage,source,status}`
    - `nautilus_ml_watermark_lag_seconds{dataset,instrument,source}`
    - `nautilus_ml_stage_coverage_pct{dataset,from_stage,to_stage}`
    - `nautilus_ml_contract_violations_total{dataset,rule}`
  - Grafana: coverage %, max lag, gap heatmap, contract violations, lineage (via
 views/API).

- Schema Enforcement:
  - Fail-closed on writes: DataStore fetches manifest, validates data (types/nul
ls/ranges/keys), blocks mismatches.
  - Schema change guard: require manifest version bump + `schema_hash` change; a
llow dual-write migration window.

- Training Scheduler (service):
  - Triggers: freshness SLOs, drift/quality flags (from `DataProcessor`), model
health (from `ml.model_performance_summary` view).
  - Actions: enqueue retrain tasks per instrument/model; update ModelRegistry de
ployment states; log audit.

- Realtime Mode (optional):
  - Wire `ml/scripts/run_ml_pipeline.py --mode realtime` to Nautilus live data a
dapter:
    - ingest L1; compute/stash features in FeatureStore; emit events/watermarks.
    - optionally run actors for inference; persist predictions/signals (stores a
lready exist).

- Applied Feature Enhancements (optional):
  - Recent-sample weighting for training (config flag).
  - Cross-instrument correlation features as a `PipelineSpec` transform.
  - Optional PCA/feature-selection with manifest flags and parity tests.

- Reliability & Backpressure:
  - Postgres writes: idempotent upserts with composite keys; retries with backof
f/jitter.
  - Parquet dedup: periodic dedup job (hash buckets) for raw vendor dumps.
  - Backpressure: caps/gauges (concurrency, API rate), circuit-breaker with aler
ting.

**Implementation Plan & Tracking**

- Phase 0: Design (1–2 days)
  - Deliverables: Dataset taxonomy, Manifest + Contract types, migration DDL out
line.
  - Owners: registry + stores devs.
  - DOD: reviewed spec; `mypy --strict` types; docs draft.

- Phase 1: Persistence + API Skeletons (2–3 days)
  - Add: `ml/registry/data_registry.py`, `ml/stores/data_store.py`, migration `0
04_data_registry.sql`.
  - JSON backend for dev + Postgres backend prod via `PersistenceManager`.
  - DOD: unit tests for CRUD, events, watermarks; basic docs.

- Phase 2: Integration (Scheduler + FeatureStore) (2–3 days)
  - Instrument `DataScheduler` and `FeatureStore` to emit events/watermarks; add
 event metrics.
  - DOD: integration test (Day T end-to-end with events/watermarks), metrics vis
ible.

- Phase 3: Integration (Model/Strategy Stores) (1–2 days)
  - Emit prediction/signal events; extend health views for stage coverage.
  - DOD: coverage view shows pass-through; Grafana panels render.

- Phase 4: Coverage + Backfill CLI (2 days)
  - Implement `ml/cli/coverage.py` and scheduler hook; throttle by storage/API.
  - DOD: dry run against sample; backfill jobs queued; documentation.

- Phase 5: Schema Enforcement + Contracts (2–3 days)
  - Enforce contracts in DataStore writes; preflight schema check; schema-change
 guard.
  - DOD: unit/property tests for validation; negative tests fail cleanly.

- Phase 6: Hardening + Docs (1–2 days)
  - E2E failure simulation; retry paths; backpressure tuning; docs update.
  - DOD: test matrix green; docs complete.

**APIs To Implement**

- `ml/registry/data_registry.py`
  - `register_dataset(manifest) -> str`
  - `update_manifest(dataset_id, changes) -> None`
  - `deprecate(dataset_id) -> None`
  - `get_manifest(dataset_id) -> DatasetManifest`
  - `get_contract(dataset_id) -> DataContract`
  - `emit_event(dataset_id, instrument_id, stage, source, run_id, ts_min, ts_max
, count, status, error=None) -> None`
  - `update_watermark(dataset_id, instrument_id, source, last_success_ns, count,
 completeness_pct) -> None`
  - `get_watermark(dataset_id, instrument_id, source) -> Watermark`
  - `link_lineage(child_dataset_id, parent_ids, transform_id, ts_range, params)
-> None`

- `ml/stores/data_store.py`
  - `write_ingestion(dataset_id, records, source, run_id) -> DataEvent`
  - `write_features(...)`, `write_predictions(...)`, `write_signals(...)` (wrapp
ers around existing stores + event emission)
  - `read_range(dataset_id, instrument_id, start_ns, end_ns) -> DataFrame|list`
  - `validate_batch(dataset_id, df|list) -> QualityReport`

- `ml/cli/coverage.py`
  - `coverage report --dataset BARS --start YYYY-MM-DD --end YYYY-MM-DD [--instr
ument ...]`
  - `coverage plan-backfill --from L1 --to MBP1 --date YYYY-MM-DD`
  - `coverage apply-backfill --job-file backfill.json`

**SQL DDL Outline (004_data_registry.sql)**

- Create `ml_dataset_registry` (dataset_id PK, name, version, dataset_type, stor
age_kind, location, partitioning, retention_days, schema JSON, schema_hash, cons
traints JSON, parents JSON, pipeline_signature, created_at, last_modified).
- Create `ml_data_events` partitioned by `ts_event`.
- Create `ml_data_watermarks` (unique (dataset_id, instrument_id, source)).
- Create `ml_data_lineage`.
- Indexes: composite lookups + BRIN on time.
- Views: stage coverage joining `ml_data_events` to `ml_feature_values`/`ml_mode
l_predictions`/`ml_strategy_signals`.

**Metrics Catalog**

- Data events: `nautilus_ml_data_events_total{dataset,stage,source,status}`
- Watermarks: `nautilus_ml_watermark_lag_seconds{dataset,instrument,source}`
- Stage coverage: `nautilus_ml_stage_coverage_pct{dataset,from_stage,to_stage}`
- Contract violations: `nautilus_ml_contract_violations_total{dataset,rule}`
- Existing: scheduler collection/feature latencies, store ops, API requests (`ml
/data/scheduler.py`).

**Testing Strategy**

- Unit (mypy strict):
  - Manifests/contracts hashing and serialization.
  - DataRegistry CRUD; event + watermark updates.
  - DataStore validation rules (types, nulls, ranges, keys).
- Integration:
  - Day T ingest→catalog→features→events/watermarks; assert coverage and lag.
  - Prediction/signal emission; assert end-to-end stage pass-through.
  - Backfill CLI plans and queues L2 when L1 present.
- Property-based:
  - Contract validation fuzzing (invalid dtypes, null proportions, out-of-range)
.
  - Idempotent upserts (duplicate batches).
- Performance:
  - Event logging overhead < 2% of pipeline duration.
  - Query pruning on partitioned tables.

**Acceptance Gates**

- Lint/type: `make ruff` clean; `uv run --active --no-sync mypy ml --strict` cle
an.
- Tests: `make pytest` with ≥90% coverage for `ml/` (focus on new modules).
- Health: Grafana shows watermarks, coverage ≥95% for tracked symbols in last 5
trading days; no critical contract violations.
- Backfill: CLI generates a plan for a synthetic gap and successfully applies (d
ry-run if live API not enabled).

**Runbooks & Ops**

- Schema change:
  - Bump DatasetManifest version; compute new `schema_hash`; deploy dual-write w
indow; update consumers; retire old version.
- Backfill:
  - Run `coverage report` to identify gaps; `plan-backfill`; review job file; ru
n `apply-backfill`; monitor `watermark_lag_seconds`.
- Realtime:
  - Start pipeline in realtime mode; verify watermarks update within SLA; monito
r contract violations.
- Incident:
  - On persistent contract violations: pause ingestion for dataset_id; investiga
te upstream schema drift; update manifest or add normalization transform.

**References & Further Info**

- Internal docs:
  - `ml/docs/context/context_data.md` – data architecture and scheduler.
  - `ml/docs/context/context_stores.md` – stores, partitioning, DataProcessor.
  - `ml/docs/context/context_registry.md` – registry architecture and manifests.
  - `ml/schema/pipeline_health.sql` – health/freshness views.
  - `ml/monitoring/README.md`, `ml/monitoring/grafana/GRAFANA_INTEGRATION_PLAN.m
d`.
- Code anchors:
  - `ml/data/scheduler.py` – add ingestion event emissions.
  - `ml/stores/feature_store.py` – emit feature events; enforce contracts.
  - `ml/stores/migrations/001_stores_schema.sql` – patterns for partitioning/ind
exing.
- External patterns to consult:
  - Apache Flink/Beam event-time watermarks.
  - OpenLineage + Marquez for lineage modeling.
  - Airbnb Zipline feature store whitepaper (point-in-time, backfills).

**Tracking Checklist**

- Manifests/Contracts:
  - Define DatasetManifest/DataContract types and tests.
  - Register baseline datasets (BARS, QUOTES, TRADES, MBP1/TBBO, FEATURES, PREDI
CTIONS, SIGNALS).
- Migration 004:
  - Create DDL; run locally; add down migration or safe re-run.
- Registry/Store:
  - Implement DataRegistry + DataStore skeletons; add unit tests.
- Instrumentation:
  - Scheduler emits catalog events; FeatureStore emits feature events; Model/Str
ategy stores emit prediction/signal events.
- Coverage CLI:
  - Implement report/plan/apply; add unit and smoke tests.
- Metrics & Dashboards:
  - Add new metrics; update Grafana to show coverage, lag, violations.
- Enforcement:
  - Integrate contract validation into DataStore writes; add preflight checks.
- Training Scheduler:
  - Implement queue based on SLOs/drift/health; integrate with ModelRegistry; ad
d audit logs.
- Realtime (if enabled):
  - Wire live ingestion; confirm near-real-time watermarks.
- Hardening:
  - Idempotency, retries, backpressure knobs; failover docs.
