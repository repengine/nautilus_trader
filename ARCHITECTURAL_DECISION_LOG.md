# Architectural Decision Log

This log records notable decisions in the ML integration layer.

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

