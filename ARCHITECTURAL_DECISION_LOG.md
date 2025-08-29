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

## 2025-08-29: Universal ML Component Protocol (Task 2.2)

- Problem: Health, metrics, and configuration validation patterns varied across actors, stores, and registries, making it hard to reason about component state and enforce consistency.
- Decision: Introduce a runtime-checkable `MLComponentProtocol` with three methods:
  - `get_health_status() -> dict[str, Any]`
  - `get_performance_metrics() -> dict[str, float]`
  - `validate_configuration() -> list[str]`
  Provide `MLComponentMixin` with safe defaults and adopt it in base actor, base stores, DataStore, and all registries. Add protocol compliance validation in `MLIntegrationManager` with a strictness toggle via `ML_STRICT_PROTOCOL_VALIDATION`.
- Consequences: Standardized health/metrics/config API across components, early detection of non-compliance, and zero hot-path overhead (methods are not invoked in hot loops). Backward compatible defaults minimize churn.
