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

Deliverables
- Architecture note in `ml/docs/architecture/event_driven_ml_pipeline_exploration.md` describing actor boundary and optional bus.

---

## Phase 1 — Message Bus Integration & Event Flow

Objective
- Publish ML domain events after DB/registry commits from the actor thread; keep stores/registries DB‑first and avoid hot‑path I/O.

Checklist
- Topic schema & helpers
  - [ ] Add stage‑first builder (configurable): `build_stage_topic(stage, instrument_id=None, prefix='events.ml')`
  - [ ] Maintain compatibility with `ml.<domain>.<operation>.<instrument_id>` via existing `build_topic` + mapping
  - [ ] Add unit tests for wildcard filters and optional instrument suffix
- Actor bridge
  - [ ] Add actor‑side hook (or companion actor `ml/actors/ml_domain_events.py`) that:
    - [ ] Receives internal “commit complete” notifications
    - [ ] Publishes non‑blocking on the actor thread (enqueue only)
  - [ ] Leave store‑level publishing disabled by default on actor paths; keep for dev/tests
- Bus publisher
  - [ ] Keep `MessagePublisherProtocol` abstraction; add a Redis Streams adapter behind a flag (optional)
  - [ ] Feature flag: `ML_BUS_ENABLE` + config for topic prefix and stage‑first vs domain‑op topics
- Idempotency/backpressure
  - [ ] Consumer templates showing correlation_id + watermark gating
  - [ ] Optional throttler on noisy topics (non‑blocking; drop/aggregate)

Tests
- [ ] Contract: publishing contracts (topic format, payload fields)
- [ ] Property: wildcard topic filter behavior (with/without instrument suffix)
- [ ] Integration: single‑thread end‑to‑end publish/subscribe with idempotent consumers

Docs
- [ ] Update `domain_bookkeeping_plan.md` with actor bridge and topic schema options
- [ ] Add short “Consumer Examples” section linking to Redis adapter (if added)

Acceptance
- End‑to‑end publish/subscribe in a single‑thread actor context
- Optional bus off by default; no hot‑path regression

Links
- Topic helpers: `ml/common/message_topics.py`
- Publisher proto: `ml/common/message_bus.py`
- Integration: `ml/core/integration.py`

---

## Phase 2 — Unified Observability Pipeline

Status: Implemented (in repo). Use this checklist to verify gates whenever changes occur.

Checklist
- DTO builders: `ml/observability/pipeline.py`
  - [ ] Latency watermarks, metrics collection, event correlation, health scores
- Service façade: `ml/observability/service.py` (off hot path)
- Sinks: JSONL/CSV, DB
  - [ ] File: `ml/observability/persistence.py` (+ rotation/compaction)
  - [ ] DB: `ml/observability/db_persistence.py` (+ retention/apply_retention)
- Background flusher: `ml/observability/scheduler.py` (tick + background thread)
- DB indices/partitions: `ml/observability/migrations.py` (BRIN, monthly partitions)
- CLI + backfill: `ml/cli/observability.py`, `ml/scripts/observability_backfill.py`
- Alerts/Dashboards: `ml/deployment/alerts.yml`, Grafana JSON references in docs

Tests
- [ ] Unit: builders, service, sinks, scheduler
- [ ] Contracts: Pandera schemas in‑memory + persisted JSONL/DB
- [ ] Partitioning: Postgres partition tests (skip if not available)
- [ ] Fault‑injection: disk/DB failures don’t affect hot path
- [ ] Micro‑bench: DTO build + sink latency

Docs
- [ ] Quickstart: `ml/docs/observability_quickstart.md`
- [ ] Ops Runbook: `ml/docs/ops/observability_runbook.md`

Acceptance
- Tables written per interval; no hot‑path impact; dashboards render stable metrics

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

