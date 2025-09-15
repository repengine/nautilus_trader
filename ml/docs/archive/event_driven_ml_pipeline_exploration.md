# Event‑Driven ML Pipeline Exploration Plan

## Purpose

Capture design options and migration steps for evolving the ML layer to an event‑driven pipeline while preserving hot‑path budgets and keeping DB‑first registries authoritative.

## Operating Principles

- Actor boundary: Keep Nautilus actors single‑threaded; do not block hot paths.
- DB‑first: Registries/stores remain source of truth; events are a read‑side view.
- Optional bus: Publishing is optional and non‑blocking; no‑op in tests/scripts.
- Idempotency: Gate consumers with `correlation_id` + registry watermarks.
- Off hot path: Observability, persistence, and bus I/O run outside actor hot loops.

## Phases (Report Outline)

- Phase 0 — Guardrails
  - Single‑thread boundary for MessageBus usage (actor thread only)
  - Optional bus (no‑op path)
  - Idempotency via watermarks/correlation
  - CI gates: validate‑metrics, validate‑events, mypy strict, ruff clean

- Phase 1 — Bus Integration & Flow
  - Actor (or companion actor) publishes ML domain events after DB commits
  - Topics: `events.ml.data.<Stage>[.<instrument_id>]` and registry ops
  - Acceptance: End‑to‑end pub/sub, wildcard filters, idempotent processing

- Phase 2 — Unified Observability
  - Latency histograms, watermark lag gauge, domain health
  - Correlation/lineage queries by `correlation_id`
  - Dashboards + alerts

- Phase 3 — Performance & Circuit Breakers
  - P99 budgets: feature (<500µs), inference (<2ms), end‑to‑end (<5ms)
  - CI perf guards; standard circuit breaker configs

- Phase 4 — Intelligent Automation (requires MLOps)
  - Gaps/drift/degradation monitoring; model fallback
  - Dynamic circuit breakers; PnL attribution

- Phase 5 — Event‑Driven Migration & Schema Evolution
  - Replace polling with event triggers (idempotent via watermarks)
  - Versioning/compat checks and migrations across datasets/manifests

- Phase 6 — Testing & QA Standards
  - Contract tests (schemas, publishing contracts), property tests
  - ≥90% ML coverage; CI green with budget gates

## Current Plan vs Report — Comparison

- Aligned
  - DB‑first registries/stores with idempotency
  - Phased rollout; strong observability with DTOs/sinks/alerts
  - Optional/no‑op bus path; tests remain simple

- Differences
  - Actor boundary: Report enforces actor‑thread publishing; code currently allows optional store publishers outside actor context (now gated and off hot path)
  - Topic schema: Report favors stage‑first topics with optional instrument suffix; code uses `ml.<domain>.<operation>.<instrument>` (stage in payload)
  - Perf budgets: Report calls for tight CI perf gates; code has micro‑benches but no strict CI P99 budgets for feature/inference
  - Backpressure: Report mentions throttler; code relies on non‑blocking publish

- Recommendations
  - Adopt actor‑only publishing (bridge) as default; leave store‑level publish for dev/tests
  - Introduce stage‑first topic builder behind a flag; keep backward compatibility
  - Add perf budgets for feature/inference in CI; keep publish off hot path
  - Consider a throttle/shadow‑publish mode for high‑volume topics

## Async Design (Keep Actor Sync)

- Pattern: Sync facade + async worker in a background thread
  - Actor enqueues tasks (ns‑scale) to a bounded queue; returns immediately
  - Background loop uses `sqlalchemy.ext.asyncio` + `asyncpg` (DB) and `redis.asyncio` (bus)
  - Provide drain/flush barrier for shutdown; capture metrics; drop/degrade under sustained backpressure
- Benefits: Tighter hot‑path P99, better throughput for I/O bound work, clean escalation to Redis/Kafka later

## Actor Centralization vs Redis vs Build‑Your‑Own

- Actor centralization (publish in actor thread)
  - Pros: Lowest latency; deterministic; simple; resilient when bus optional
  - Cons: Limited fan‑out; no durable replay; external systems need a bridge

- Redis Streams
  - Pros: Cross‑process fan‑out; at‑least‑once + replay; central observability
  - Cons: Network latency; ops overhead; exactly‑once still via idempotency

- Build your own bus
  - Pros: Tailored control
  - Cons: Reinventing durability/replay/HA/backpressure; high maintenance

- Recommendation: Use actor centralization for hot paths; adopt Redis Streams behind a feature flag for fan‑out/replay; don’t build a bespoke distributed bus.

## Decision Trees

- Do we need cross‑process consumers or replay?
  - Yes → Enable Redis Streams publisher (flag); keep consumers idempotent; actor enqueues non‑blocking
  - No → Keep actor bridge only; DB‑first remains authoritative

- Do we need tighter P99 tails in actor?
  - Yes → Add AsyncPersistenceWorker (flag) with bounded queue; keep enqueue as the only hot‑path work
  - No → Current off‑path flush/persist is sufficient

- Topic schema choice
  - Need selective routing by stage → Use stage‑first topics (flag), optional instrument suffix
  - Prefer compatibility/simplicity → Keep current domain‑operation form; stage in payload

## Possible Consumers (Examples)

- Ops/Observability: observability aggregator, lineage materializer, DLQ handler
- Data Quality/Compliance: schema/contract validator, audit logger
- ML Monitoring: drift monitor, SLO watchdog, circuit breaker coordinator
- Research/Data Eng.: training set builder, lakehouse writer/CDC forwarder, backfill/replay orchestrator
- Trading/Risk: PnL attribution, exposure aggregator, strategy scorer
- Integrations: alert fan‑out (Slack/PagerDuty), search indexer

Each consumer must be idempotent (correlation_id + watermark), assume out‑of‑order, and avoid blocking publishers.

## Actionable Next Steps (Flag‑Driven)

- Actor bridge (default): Publish from actor thread after DB/registry commit; disable store‑level publish by default
- Stage‑first topics: Add a `build_stage_topic(stage, instrument_id=None, prefix='events.ml')` helper and a flag to switch
- Perf budgets: Add CI guards for feature/inference P99; keep publish path off hot path
- Async worker: Introduce AsyncPersistenceWorker behind a flag; measure improvements in shadow mode first
- Redis Streams: Add a publisher adapter behind a flag; enable when ≥3 consumers or replay needs appear

## Acceptance

- Hot path remains single‑threaded and within budgets; pub/sub optional and non‑blocking
- Idempotent consumers validated via correlation_id + watermarks
- Observability dashboards stable; perf and coverage gates green
