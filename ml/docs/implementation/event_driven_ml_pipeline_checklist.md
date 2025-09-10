# Event‑Driven ML Pipeline — Phased Checklist (Master Plan)

This document is a self‑contained, phased checklist for implementing and hardening the event‑driven ML pipeline while preserving hot‑path budgets and keeping DB‑first registries authoritative. It aggregates links to architecture decisions, patterns, protocols, code locations, tests, CI, and ops guidance so a fresh LLM context window can execute tasks end‑to‑end.

## Canonical Context & Standards

Use these as the authoritative references during implementation:

- Architecture & Plans
  - Domain Bookkeeping (active plan): `ml/docs/implementation/domain_bookkeeping_plan.md`
  - Event‑Driven Exploration (this report synthesized): `ml/docs/architecture/event_driven_ml_pipeline_exploration.md`
  - ML Integration Architecture: `ml/docs/architecture/ml_integration_architecture.md`
  - ADRs (selected):
    - Centralized metrics bootstrap: `ml/docs/architecture/decisions/ADR-005-centralized-metrics-bootstrap.md`
- Coding & Testing
  - Coding standards: `ml/docs/development/CODING_STANDARDS.md`
  - Testing strategy: `ml/tests/docs/TESTING_STRATEGY.md`
  - CLAUDE rules (imports/timestamps): `CLAUDE.md`
- Protocols & Patterns (code)
  - Metrics bootstrap API: `ml/common/metrics_bootstrap.py`, central metrics: `ml/common/metrics.py`
  - Topic helpers: `ml/common/message_topics.py`
  - Publisher abstraction: `ml/common/message_bus.py` (MessagePublisherProtocol)
  - Integration manager: `ml/core/integration.py`
  - Stores: `ml/stores/{data_store.py,feature_store.py,model_store.py,strategy_store.py}`
  - Registries: `ml/registry/*`
  - Observability: `ml/observability/{pipeline.py,service.py,persistence.py,db_persistence.py,scheduler.py,migrations.py}`
- CI/Dev
  - ML coverage gate: `.github/workflows/build.yml` (≥90% ML, excludes `-m prototype`)
  - CLI examples: `ml/cli/observability.py`, backfill script: `ml/scripts/observability_backfill.py`

---

## Roadmap Addendum — Next Actions

Near‑term (2–4 weeks)

- [x] Wire actor‑side publishing in `MLSignalActor` via `ActorBusConfig.from_env()` using `DomainEventBridge`; store‑path publishing disabled when actor‑path enabled (mutual exclusion)
- Topic scheme unification
  - [x] Stores (Feature/Model/Strategy): publish via `build_topic_for_stage` with env‑driven scheme/prefix
  - [x] DataStore topic scheme parity: read env and honor stage‑first when configured
- [ ] Standardize event `status` to an enum and enforce across emitters to align with Pandera contracts.
- [x] CI: include `make validate-metrics` and `make validate-events` steps (present in workflows).
- [ ] CI perf smoke: add micro‑bench job (soft gate) for P99 budgets (feature compute, inference, end‑to‑end).
- [ ] Coverage uplift: add deterministic tests for `DataStore` ingestion (JSON backend) covering `emit_event`/watermarks and failure paths; add actor‑thread boundary tests once bridge is wired.
- [ ] Docs sync: update architecture/plan statuses and remove “prototype” language now that tests are productionized.

Mid‑term (4–8 weeks)

- [ ] Circuit breakers + backpressure policies using `MLComponentProtocol` health; expose metrics and drop/degrade policies on bus/DB paths.
- [ ] Cross‑domain cascades: finish lineage propagation and add property/contract tests for correlation, ordering, retries.
- [ ] Observability UX: ship Grafana dashboards + alerts; finalize runbook wiring.
- [ ] Schema evolution: versioned manifests, compatibility checks, migration scripts; move from polling to event triggers gated by watermarks.
- [ ] Bus consumers (reference): aggregator, DLQ/retry, lineage writer templates with idempotent/watermark gates.

Longer‑term (8+ weeks)

- [ ] Enforce perf budgets as hard CI gates after stabilizing thresholds.
- [ ] Intelligent automation: drift/perf monitors, calibrators, model fallback/selection, PnL attribution across domains.
- [ ] Production hardening: broader Postgres‑path tests, observability retention/partition ops, fail‑injection scenarios for hot/cold paths.

## Phase 0 — Guardrails & Setup

Checklist

- [ ] Actor boundary (design): All MessageBus interactions occur on the actor thread (or companion actor). No blocking I/O on the hot path.
- [ ] Optional bus: `MessagePublisherProtocol` set to noop by default; publishing is off unless enabled.
- [ ] Idempotency: All consumers implement correlation_id + registry watermarks gates.
- [ ] Lint/types/tests gates configured:
  - [ ] `ruff` clean for changed ML code
  - [ ] `mypy ml --strict` clean
  - [ ] `pytest -m 'not prototype'` green
  - [ ] `make validate-metrics` / `make validate-events` green
- [ ] Docs: link this checklist from `domain_bookkeeping_plan.md`.

Additions

- [x] CI includes `validate-metrics` and `validate-events` steps in workflows.
- [ ] Perf smoke (soft gate): add CI micro‑bench job to measure P99 targets (feature compute, inference, end‑to‑end) without failing builds initially.

Deliverables

- Architecture note in `ml/docs/architecture/event_driven_ml_pipeline_exploration.md` describing actor boundary and optional bus.

---

## Phase 1 — Message Bus Integration & Event Flow

Objective

- Publish ML domain events after DB/registry commits from the actor thread; keep stores/registries DB‑first and avoid hot‑path I/O.

Checklist

- Topic schema & helpers
  - [x] Add stage‑first builder (configurable): `build_stage_topic(stage, instrument_id=None, prefix='events.ml')`
    - Implemented in `ml/common/message_topics.py` (also added `build_topic_for_stage` to switch between schemes)
  - [x] Maintain compatibility with `ml.<domain>.<operation>.<instrument_id>` via existing `build_topic` + mapping
    - `build_topic` + `map_stage_to_topic_segments(Stage)` retained
  - [x] Add unit tests for optional instrument suffix and normalization
    - Unit + property tests under `ml/tests/unit/common/test_message_topics.py` and `ml/tests/property/test_message_topics_property.py`
    - [x] Wildcard filter behavior (subscription side)
- Actor bridge
  - [x] Add actor‑side hook (companion module `ml/actors/ml_domain_events.py`) that:
    - [ ] Receives internal “commit complete” notifications (wiring TBD)
    - [x] Publishes non‑blocking on the actor thread (enqueue only)
  - [x] Leave store‑level publishing optional on actor paths; default No‑op unless publisher configured
  - [ ] Wire `DomainEventBridge` into `MLSignalActor` behind `ML_BUS_FROM_ACTOR`; ensure mutual exclusion with store‑path publishing via `ActorBusConfig`
- Publishing across stores
  - [ ] Enable end‑to‑end topic scheme selection in Data/Feature/Model/Strategy stores using `build_topic_for_stage(stage, scheme=..., prefix=...)`
- Bus publisher
  - [x] Keep `MessagePublisherProtocol` abstraction; add a Redis Streams adapter behind a flag (optional)
    - `RedisStreamsPublisher` + `publisher_from_config` in `ml/common/message_bus.py`
  - [x] Feature flag/config: `ML_BUS_ENABLE`, `ML_BUS_BACKEND`, `ML_BUS_SCHEME`, `ML_BUS_TOPIC_PREFIX`, `ML_BUS_REDIS_URL`, `ML_BUS_REDIS_STREAM`, `ML_BUS_REDIS_MAXLEN`
    - Parsed by `ml/config/bus.py::MessageBusConfig.from_env`
- Contracts & payloads
  - [ ] Standardize `status` to an enum and enforce across emitters (DataRegistry/DataStore/stores) to align with Pandera contracts
- Idempotency/backpressure
  - [x] Consumer templates showing correlation_id + watermark gating
  - [x] Optional throttler on noisy topics (non‑blocking; drop/aggregate)

Registry‑driven event flow (implemented)

- [x] Unified DataRegistry instance injected into FeatureStore/ModelStore (IntegrationManager & Actors wire a shared registry)
- [x] Feature events after commit: FeatureStore emits `FEATURE_COMPUTED` with metadata `{feature_set_id}` and updates watermarks (dataset_id=`features`)
- [x] Prediction events after commit: ModelStore emits `PREDICTION_EMITTED` with metadata `{model_id}` and updates watermarks (dataset_id=`predictions`)
- [x] Persistence correctness for FeatureRegistry (Postgres): stage changes and `update_manifest` persist via `_save_feature_to_db` (tests include JSON; Postgres test skipped if DB unavailable)
- [x] Unit tests: artifact attach + perf_digest persistence; ModelStore → DataRegistry event emission

Implementation notes (current)

- Code paths:
  - Store/Registry injection: `ml/core/integration.py` (`_init_registries`) and `ml/actors/base.py` (prod path) inject the same `DataRegistry` into stores via `set_data_registry`.
  - Feature events: `ml/stores/feature_store.py` emits `FEATURE_COMPUTED` with `{feature_set_id}` metadata and updates watermark.
  - Prediction events: `ml/stores/model_store.py` emits `PREDICTION_EMITTED` with `{model_id}` metadata and updates watermark.
  - FeatureRegistry persistence: `ml/registry/feature_registry.py` `update_manifest(...)`, `promote/deprecate/scrap` persist on both JSON/Postgres backends.
- Tests:
  - JSON: `ml/tests/unit/registry/test_feature_registry_artifacts.py`, `ml/tests/unit/registry/test_feature_registry_update_manifest.py`
  - Postgres (skippable): `ml/tests/integration/registry/test_feature_registry_postgres_update.py`
  - Store→Registry events: `ml/tests/unit/stores/test_model_store_events.py`

Next tasks (bus‑specific)

- [ ] Wire actor bridge to internal “commit complete” notifications (explicit hook point)
  - Current: actor enqueues SIGNAL_EMITTED on _publish_signal; registry commit hook for non-signal events remains open
- [ ] Add deployment notes for event rate dashboards/alerts (optional)

Tests

- [x] Contract: topic format builders (domain_op and stage_first); payload normalization validated via store emission tests
- [x] Property: wildcard topic filter behavior (with/without instrument suffix)
- [x] Integration: single‑thread end‑to‑end publish/subscribe with idempotent consumers (in‑memory bus)

Docs

- [x] Add events context doc: `ml/docs/context/context_events.md` (schemes, flags, adapters, bridge)
- [x] Add consumer examples doc: `ml/docs/context/context_consumers.md` (idempotency, wildcard filters, in-memory pub/sub, Redis consumer)
- [x] Update `domain_bookkeeping_plan.md` with actor bridge and topic schema options
- [x] Add events consumer CLI doc stub within the consumer examples (usage examples)

Acceptance

- Optional bus off by default; no hot‑path regression (passes)
- End‑to‑end publish/subscribe in a single‑thread context (passes with in‑memory bus; actor hook wiring pending)

Links

- Topic helpers: `ml/common/message_topics.py`
- Publisher proto/adapters: `ml/common/message_bus.py`
- Bus config: `ml/config/bus.py`
- Actor bridge: `ml/actors/ml_domain_events.py`
- Integration helper: `ml/core/bus_integration.py`
- Integration: `ml/core/integration.py`

---

## Phase 2 — Unified Observability Pipeline

Status: Implemented (in repo). Use this checklist to verify gates whenever changes occur.

Checklist

- DTO builders: `ml/observability/pipeline.py`
  - [x] Latency watermarks, metrics collection, event correlation, health scores
- Service façade: `ml/observability/service.py` (off hot path)
- Sinks: JSONL/CSV, DB
  - [x] File: `ml/observability/persistence.py` (+ rotation/compaction)
  - [x] DB: `ml/observability/db_persistence.py` (+ retention/apply_retention)
- Background flusher: `ml/observability/scheduler.py` (tick + background thread)
- DB indices/partitions: `ml/observability/migrations.py` (BRIN, monthly partitions)
- CLI + backfill: `ml/cli/observability.py`, `ml/scripts/observability_backfill.py`
- Alerts/Dashboards: `ml/deployment/alerts.yml`, Grafana JSON references in docs

Tests

- [x] Unit: builders, service, sinks, scheduler
- [x] Contracts: Pandera schemas in‑memory + persisted JSONL/DB
- [x] Partitioning: Postgres partition tests (skip if not available)
- [x] Fault‑injection: disk/DB failures don’t affect hot path
- [ ] Micro‑bench: DTO build + sink latency

Docs

- [x] Quickstart: `ml/docs/observability_quickstart.md`
- [x] Ops Runbook: `ml/docs/ops/observability_runbook.md`

Acceptance

- Tables written per interval; no hot‑path impact; dashboards render stable metrics

Current integration notes

- Registry events (features/predictions) include essential metadata and update watermarks for downstream correlation.
- Model/Feature stores avoid hot‑path DB calls beyond their write operations; bus publishing remains optional and off by default.

---

## Phase 3 — Performance & Circuit Breakers

Objective

- Guarantee hot‑path budgets and provide guardrails on failure.

Checklist

- Budgets & CI
  - [ ] Feature compute P99 < 500µs (micro‑bench + gate)
  - [ ] Inference P99 < 2ms (micro‑bench + gate)
  - [ ] End‑to‑end P99 < 5ms (integration benchmark)
- Circuit breakers
  - [ ] Policies on actors/stores (open/half‑open/close) with idempotent retries
  - [ ] Health hooks: use `MLComponentProtocol` to derive health for CB decisions

Tests

- [ ] CI perf jobs with thresholds and regression detection
- [ ] CB state transitions under synthetic fault loads

Docs

- [ ] Performance targets + CB runbook entries

---

## Phase 4 — Intelligent Automation (Prereqs: MLOps)

Checklist

- [ ] Drift/perf monitors; threshold alerts
- [ ] Model fallback + selection logic
- [ ] PnL attribution to data/feature/model/strategy components

Tests

- [ ] Synthetic drift and fallback scenarios

---

## Phase 5 — Event‑Driven Migration & Schema Evolution

Checklist

- [ ] Replace polling with event triggers (idempotent via watermarks)
- [ ] Versioned manifests; compatibility checks; migration scripts

Tests

- [ ] Fixture migrations + compatibility CI job

---

## Phase 6 — Testing & QA Standards (Always On)

Checklist

- [ ] Contract tests (publishing + schemas)
- [ ] Property tests (validation rules/lineage integrity)
- [ ] Coverage ≥90% for `ml/`
- [ ] CI green with budget gates (lint/types/tests/metrics/events)
  - [x] CI includes `validate-metrics` and `validate-events` steps
  - [ ] Add perf micro‑bench job and later promote thresholds to hard gates

Added gates/tests (recent)

- [x] FeatureRegistry persistence (JSON + Postgres‑skippable) for `update_manifest` and stage transitions
- [x] Store→Registry event emission tests (JSON backend) for predictions
- [x] Strict typing and ruff for new modules (promotion/evaluate CLIs, runner, registry updates)

Links

- Standards: `ml/docs/development/CODING_STANDARDS.md`, `ml/tests/docs/TESTING_STRATEGY.md`, `CLAUDE.md`

---

## Async Persistence Worker (Optional Flag)

Objective

- Keep actor single‑threaded; move I/O (DB/bus/observability) to a background thread running an asyncio loop.

Checklist

- [ ] Worker with bounded queue + asyncio loop
- [ ] Async DB (sqlalchemy.ext.asyncio + asyncpg) & async bus (redis.asyncio) adapters
- [ ] Sync facade in actor: enqueue only; ns‑scale
- [ ] Shutdown barrier; backpressure metrics; drop/degrade policies

Tests

- [ ] Enqueue latency; end‑to‑end drain; orderly shutdown; stress with backpressure

---

## Optional Consumers (Examples)

- Ops/Observability: aggregator, lineage, DLQ handler
- Data Quality/Compliance: contract validator, audit logger
- ML Monitoring: drift monitor, SLO watchdog, circuit breaker coordinator
- Research/Data Eng.: training set builder, lakehouse writer/CDC forwarder, backfill/replay orchestrator
- Trading/Risk: PnL attribution, exposure aggregator, strategy scorer
- Integrations: alert fan‑out, search indexer

Each consumer must be idempotent (correlation_id + watermark), tolerate out‑of‑order, and avoid blocking publishers.

---

## Deliverables Matrix (Per Phase)

- Code: feature toggles, adapters, protocols, helpers
- Tests: unit, contract, property, integration, perf, fault‑injection
- CI: job updates, gates (lint/types/coverage/perf), labels/schedules
- Docs: quickstarts, runbooks, ADR/architecture updates, this checklist
- Ops: alerts, dashboards, runbooks, rollback toggles

---

## Status & Tracking

- Use checkboxes above to track work. Update `ml/docs/implementation/domain_bookkeeping_plan.md` and this checklist at each milestone.
- Default execution profile for PRs: `-m 'not prototype'`. Prototype suites run on schedule or PR label.

---

## Quick Commands

- Types: `uv run --active --no-sync mypy ml --strict`
- Lint: `ruff check ml -q`
- Tests (observability): `pytest -q ml/tests/unit/observability`
- Prototypes: `pytest -q -m prototype ml/tests`
- CLI (local sinks):
  - `uv run -m ml.cli.observability flush-jsonl --base-path ./observability --format jsonl --seed-sample`
  - `uv run -m ml.cli.observability flush-db --db-url sqlite:///./observability.db --seed-sample`
  - `uv run -m ml.cli.observability start --sink db --db-url sqlite:///./observability.db --interval 10 --duration 30 --seed-sample`
