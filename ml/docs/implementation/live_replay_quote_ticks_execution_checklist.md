# Live Replay Quote Ticks Execution Checklist

## Goal
Enable replay harness runs where ML actors stay bar-driven while strategies subscribe to quote ticks (MBP-1) for execution, with staleness guards and full test coverage.

## Scope and Constraints
- Keep ML inference actors bar-only for training/inference parity.
- Subscribe to quote ticks in strategies for execution market state.
- Replay harness must load quote ticks alongside bars when enabled.
- All tunables live in config classes under `ml/config`.
- New metrics use `ml.common.metrics_bootstrap`.
- Guardrails: mypy strict, ruff, fixture validation, targeted tests, and coverage checks.

## Checklist

### Configuration and Wiring
- [x] Add `subscribe_quote_ticks`, `quote_schema`, and `max_quote_age_ms` to `ml/config/base.py` `MLStrategyConfig` with docstrings and env parsing.
- [x] Add matching fields to `ml/config/replay_harness.py` `StrategyReplayConfig` with validation where needed.
- [x] Thread new config fields through `ml/orchestration/parquet_live_replay_harness.py` when building `MLStrategyConfig`.
- [x] Add CLI flags in `ml/cli/parquet_live_replay_harness.py` for `--subscribe-quote-ticks`, `--quote-schema`, and `--max-quote-age-ms`.

### Lifecycle Subscription
- [x] Extend `ml/strategies/common/lifecycle.py` to optionally subscribe to quote ticks when enabled (using schema params).
- [x] Pass the quote subscription callback and config fields from `ml/strategies/base_facade.py` into `LifecycleComponent`.

### Execution Safety (Stale Quote Guard)
- [x] Add a stale-quote guard in `ml/strategies/common/order_submission.py` that skips smart order creation when the latest quote is older than `max_quote_age_ms`.
- [x] Emit a structured debug log and a `ml.common.metrics_bootstrap` counter when staleness triggers.

### Replay Harness Enhancements
- [x] Add catalog quote tick loading to `ml/orchestration/parquet_live_replay_harness.py` when `subscribe_quote_ticks` is enabled.
- [x] Add quote ticks to the backtest engine via a separate `engine.add_data` call.
- [x] Extend `ParquetLiveReplayHarnessResult` to report `quote_ticks_loaded` for visibility.

### Tests
- [x] Unit: update `ml/tests/unit/strategies/common/test_lifecycle_component.py` to assert quote tick subscription wiring and schema params.
- [x] Unit: add stale-quote guard coverage in `ml/tests/unit/strategies/common/test_order_submission_component.py`.
- [x] Integration: update `ml/tests/integration/test_parquet_live_replay_harness.py` to write quote ticks to the catalog, enable quote subscription, and assert `quote_ticks_loaded`.

### Validation Commands
- [x] `poetry run pytest -k "lifecycle_component or order_submission_component"`
- [x] `poetry run pytest -k parquet_live_replay_harness`
- [x] `poetry run mypy ml --strict`
- [x] `poetry run ruff check ml`
- [x] `make validate-fixtures`
- [x] `make validate-metrics`
- [x] `make validate-events`
- [ ] `coverage run -m pytest ml/tests/ && coverage report` (fails without `Cython` for coverage plugins)
