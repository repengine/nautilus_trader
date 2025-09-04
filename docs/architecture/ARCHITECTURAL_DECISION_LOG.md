# Architectural Decision Log

This log records notable decisions in the ML integration layer.

## 2025-08-29: Universal ML Architecture Patterns

- Problem: Inconsistent architectural patterns across ML domains led to integration gaps and maintenance overhead. Analysis revealed sophisticated components (DataStore, DataRegistry) were missing from mandatory integration patterns, causing manual imports and inconsistent usage.
- Decision: Codify 5 universal patterns as mandatory for all ML components:
  1. **Mandatory 4-store + 4-registry pattern** for all ML actors (FeatureStore, ModelStore, StrategyStore, DataStore + FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry)
  2. **Protocol-first interface design** using typing.Protocol for structural contracts without implementation coupling
  3. **Hot/cold path separation** with <5ms P99 performance budgets and zero-allocation patterns
  4. **Progressive fallback chains** for all external dependencies (PostgreSQL → DummyStore, Registry → Direct loading)
  5. **Centralized metrics bootstrap** for all Prometheus metrics via ml.common.metrics_bootstrap
- Consequences: Consistent architecture across domains, reduced integration bugs, enforced performance standards, eliminated manual component wiring, comprehensive data lifecycle management for all actors.

## 2025-08-29: Idempotent Metrics Bootstrap

- Problem: Multiple modules re-created Prometheus metrics using private `REGISTRY._names_to_collectors` APIs, which is brittle.
- Decision: Add `ml/common/metrics_bootstrap.py` providing `get_counter/get_histogram/get_gauge` that cache and return collectors idempotently. Refactor `ml/actors/signal.py` and `ml/strategies/base.py` to use it.
- Consequences: No reliance on prometheus internals; safe double initialization across modules/tests.

## 2025-08-29: ONNX-only in Production for Actors

- Problem: Allowing arbitrary formats (`.joblib`, pickled) risks code-execution and inconsistent inference semantics.
- Decision: Add `allow_non_onnx_in_dev: bool = False` to `MLActorConfig` and enforce ONNX-only when loading via path (registry remains ONNX-only). Non-ONNX is allowed only if the flag is explicitly set.
- Consequences: Safer default. Dev/test can opt-in to non-ONNX.

## 2025-08-29: DB Fallback Uses DummyStore

- Problem: SQLite fallback conflicted with PostgreSQL-specific upsert code paths, causing runtime errors.
- Decision: When Postgres is unavailable, instantiate `DummyStore` for features/models/strategy (no persistence) and log a warning.
- Consequences: Actors start in dev/test without DB; persistence disabled until Postgres is available.

## 2025-08-29: Integration Manager Opt-in

- Problem: Automatically starting Docker and migrations is intrusive in CI/ops.
- Decision: Default `MLIntegrationManager` `auto_start_postgres` and `auto_migrate` to False; allow env flags `ML_AUTO_START_DB` and `ML_AUTO_MIGRATE` to opt-in.
- Consequences: Safer behavior by default; explicit opt-in for local convenience.

## 2025-08-29: Universal ML Component Protocol

- Problem: Health, metrics, and configuration validation patterns varied across actors, stores, and registries, making it hard to reason about component state and enforce consistency.
- Decision: Introduce a runtime-checkable `MLComponentProtocol` with three methods:
  - `get_health_status() -> dict[str, Any]`
  - `get_performance_metrics() -> dict[str, float]`
  - `validate_configuration() -> list[str]`
  Provide `MLComponentMixin` with safe defaults and adopt it in base actor, base stores, DataStore, and all registries. Add protocol compliance validation in `MLIntegrationManager` with a strictness toggle via `ML_STRICT_PROTOCOL_VALIDATION`.
- Consequences: Standardized health/metrics/config API across components, early detection of non-compliance, and zero hot-path overhead (methods are not invoked in hot loops). Backward compatible defaults minimize churn.

## 2025-08-29: Standardize Metrics Bootstrap (Task 2.3)

- Problem: Modules instantiated Prometheus collectors directly, causing duplicate registration risks in tests and making metrics definitions hard to reason about.
- Decision: Enforce centralized, idempotent metrics acquisition via `ml/common/metrics_bootstrap.py` (get_counter/get_histogram/get_gauge). Refactor monitoring collectors and DataScheduler to use the bootstrap; keep commonly shared metrics defined in `ml/common/metrics.py`.
- Consequences: No duplicate registration, simpler imports, consistent naming/labels, and safer reuse across modules. Hot paths remain unaffected (observations only; collectors created at init).

## 2025-08-29: Registry Event Metadata + Correlation IDs

- Problem: ts_event alone is insufficient for end‑to‑end traceability across domains (collisions, replays, and multi‑event ambiguity). No place to persist per‑event trace data.
- Decision: Extend DataRegistry events with optional `metadata: JSONB` to persist correlation data (e.g., `correlation_id`). Add new SQL migration and `emit_data_event_ext` function; code prefers extended function and falls back to legacy when unavailable. DataStore now generates deterministic `correlation_id` and attaches it to event metadata.
- Consequences: Backward‑compatible tracing across Data→Features→Predictions→Signals without touching manifests or store schemas. Enables lineage queries and simplifies debugging. Manifests and schema hashes remain unchanged.

## 2025-08-29: Canonical Event Constants

- Problem: Stage names were hardcoded across modules, risking drift and typos that break DB constraints and analytics.
- Decision: Add `ml/config/events.py` with a typed `Stage` enum (values persisted to DB). Refactor core emitters (DataStore, DataScheduler) to use constants. Add a lightweight validator script to flag raw stage literals.
- Consequences: Consistent event stage usage across codebase; easier refactors and safer migrations. Prevents accidental string drift.

## 2025-08-29: Aggregated Health Summaries

 - Problem: Health checks were ad-hoc per component; no canonical, typed summary for domains or the system as a whole.
 - Decision: Add `MLIntegrationManager.aggregate_health()` returning per-component health/metrics (via `MLComponentProtocol`), aggregated to domain (data/features/model/strategy) and overall system status.
 - Consequences: Single entry point for ops and CI to assess ML system health. Enables simple CLI tooling and integration with monitoring/alerts.

## 2025-08-30: Testability and Hot-Path Adjustments

- Problem: Several unit/property tests required lightweight runtime stats from `MLSignalActor`, JSON registry persistence was non-deterministic at initialization, older tests used a legacy circuit-breaker field, and performance tests enforced zero-allocation behavior for online features.
- Decision:
  - Add `MLSignalActor.get_signal_statistics()` exposing non-hot-path counters (bars processed, history sizes, performance stats).
  - Accept legacy `half_open_attempts` in `CircuitBreakerConfig` as an alias for `success_threshold` (deprecation path).
  - Ensure `DataRegistry` JSON backend writes immediately on initialization and add a `flush()` method for explicit persistence.
  - Return a view of the pre-allocated feature buffer in online computation to guarantee zero-allocation semantics.
- Consequences:
  - Improved test determinism and observability without affecting hot-path performance.
  - Legacy config compatibility preserved in tests; alias may be removed after consumers migrate.
  - Callers persisting feature rows across bars must copy explicitly; hot path continues to reuse the buffer.

## 2025-08-30: Development-Time Relaxed Model Parity Enforcement

- Problem: Unit/property tests register models without a colocated FeatureRegistry or feature_set_id, but production requires model↔feature parity for serveable models.
- Decision: Keep production contract but add a development-time relaxation controlled by `ML_STRICT_FEATURE_PARITY` (default off):
  - If `feature_set_id` is missing or the FeatureRegistry is not present, log a warning and proceed when strict mode is disabled.
  - When `ML_STRICT_FEATURE_PARITY=1`, enforce the full parity checks and raise errors on violations.
- Consequences: Test environments can exercise registry behavior without heavy setup, while production deployments can enable strict enforcement.
