# Option 2 Plan: Small-Scope End-to-End Pipeline Validation

## Goal

Validate the full ML trading pipeline end-to-end with a small, GPU-safe scope
(5-10 symbols) so we can confirm orchestration, feature alignment, model loading,
signal flow, strategy decisions, order intent creation, health checks, and logging.
This is a plumbing validation, not a fidelity or profitability evaluation.

## Scope

- Symbols: 5-10 high-coverage symbols from `data/catalog` (example: SPY, AAPL, MSFT, NVDA, AMZN).
- Model: ONNX artifact for actor inference (Chronos AutoGluon predictors are not directly loadable by `MLSignalActor`).
- Data source: local parquet catalog (`data/catalog`) with `ML_TFT_ALLOW_PARQUET_FALLBACK=1`.
- Broker: not configured. Order intents will be serialized for manual inspection.
- Message bus: Noop/Redis only today; rely on JSONL store outputs unless we add a file-backed publisher.
- Replay pacing: use TestClock (fast) for validation first; live-paced replay is deferred until the fast path is proven.
- Registries are metadata catalogs, not compute/runtime systems. Feature and model parity checks must pass.

## Registration Semantics (What "Register" Means)

- FeatureRegistry: schema contract (ordered feature names + dtypes + schema hash + pipeline signature). It does not compute features.
- DataRegistry: dataset manifest + lineage. It does not store data itself.
- ModelRegistry: model manifest + artifact path; enforces ONNX and feature parity (strict when `ML_STRICT_FEATURE_PARITY=1`).
- StrategyRegistry: strategy manifests + model requirements.

All runtime components reference registry IDs to validate compatibility and deployments.

## Existing Strategy Architecture and Readiness

### Strategy Layer

- `BaseMLStrategyFacade` composes six components (signal routing, decision persistence,
  position management, order submission, lifecycle, performance tracking). This matches
  the protocol-first, component-based architecture.
- `MLTradingStrategy` and `SimpleMLStrategy` implement core decision logic and support
  `execute_trades=False` for safe dry runs.
- `DecisionPersistenceComponent` persists decisions and can publish events via
  `StrategyDecisionPublisher` (topics via `build_topic_for_stage` and enums).
- Strategy store support exists for both PostgreSQL (`StrategyStore`) and JSONL
  file fallback (`FileStrategyStore`) via `ML_FILE_STORE_PATH`.
- Tests exist for strategy logic, store conformance, and integration backtests
  (`ml/tests/unit/strategies/*`, `ml/tests/integration/test_ml_strategy_backtest.py`).

### Actor Layer

- `BaseMLInferenceActor` integrates the 4 stores + 4 registries and enforces hot-path
  rules (pre-allocated features, no allocations in inference).
- `MLSignalActorFacade` provides production-ready signal generation, optional domain
  event emission, and publishes `MLSignal` objects to Nautilus data bus.
- Actor-side event publishing uses `Stage/Source/EventStatus` and topics from
  `build_topic_for_stage` when enabled via `ActorBusConfig` + `MessageBusConfig`.

### Readiness Summary

- Core architecture is present and tested for strategy + actor + store interactions.
- File-backed stores (`FileModelStore`, `FileStrategyStore`) already serialize predictions
  and decisions to JSONL, suitable for manual inspection.
- File-backed stores only activate when PostgreSQL is unavailable; otherwise persistence
  goes to Postgres.
- Missing for this plan: a simple order-intent serialization hook (broker stub), a
  light orchestration harness to run actor + strategy together on the chosen symbol set,
  and a clear ONNX-compatible inference artifact.

## Plan Phases

### Phase 0: Preflight and Environment

- Pick 5-10 symbols with confirmed coverage in `data/catalog`.
- Exclude symbols that require suffix parsing (e.g., `BRK.B`) until symbol parsing is fixed.
- Export:
  - `ML_TFT_ALLOW_PARQUET_FALLBACK=1`
  - `ML_FILE_STORE_PATH=ml_out/pipeline_validation_option2`
- Decide persistence target:
  - JSONL outputs require PostgreSQL to be unavailable (file fallback), otherwise data
    is written to Postgres.
- Ensure an ONNX model artifact exists for the actor (Chronos predictors are not loadable
  directly by `MLSignalActor`).
- Ensure registry layout is compatible with parity checks:
  - `ml_registry/features/feature_registry.json`
  - `ml_registry/models/registry.json`

### Phase 1: Dataset + Feature Contract

- Build a small dataset slice from `data/catalog` and register the feature set.
- Confirm the dataset columns align with the registered feature order and schema hash.
- Record: `feature_set_id`, schema hash, and dataset manifest metadata.

### Phase 2: Model Artifact Preparation (ONNX)

- Preferred path (Chronos teacher → ONNX student):
  - Train Chronos (teacher or bolt_small) and generate rolling soft labels.
  - Train a LightGBM student on soft labels and export ONNX for `MLSignalActor`.
- Alternative path (higher effort):
  - Add a Chronos inference adapter that loads AutoGluon `TimeSeriesPredictor`
    and bridges to actor inference. Defer unless ONNX distillation is blocked.
- If not using Chronos:
  - Use an existing ONNX baseline (e.g., LightGBM/XGBoost export) to validate plumbing.

### Phase 3: Actor + Strategy Harness (Signals + Decisions)

- Configure `MLSignalActor` (or `MLSignalActorFacade`) to load the ONNX model.
- Configure `MLStrategyConfig` with:
  - `execute_trades=False`
  - `use_strategy_store=True`
  - `ml_signal_source` pointing at the actor ID
- Run a backtest-style or controlled run (catalog-backed replay) to emit signals and
  decisions; confirm JSONL/DB outputs and logs.
- Use TestClock (fast) execution; live-paced replay is explicitly out of scope until this succeeds.

### Phase 4: Order-Intent Serialization (Manual Review)

- Use file-backed stores (`ML_FILE_STORE_PATH`) to capture (file fallback only):
  - Predictions: `ml_out/pipeline_validation_option2/models/predictions.jsonl`
  - Strategy decisions: `ml_out/pipeline_validation_option2/strategies/signals.jsonl`
- Provide a minimal order submission stub (submit-order callback) that serializes
  order intents to `ml_out/pipeline_validation_option2/orders/order_intents.jsonl`.
- Re-run with `execute_trades=True` so the strategy actually calls the submit path,
  but keep the stubbed callback in place to avoid broker integration.
- Optional: add a file-backed `MessagePublisherProtocol` if we want bus-style events;
  otherwise rely on JSONL store outputs.

### Phase 5: Observability + Health

- Validate metrics and event schemas:
  - `make validate-metrics`
  - `make validate-events`
- Confirm actor/strategy health checks are green and no hot-path violations are logged.
- Ensure domain events use enums and topics are built via `build_topic_for_stage`.

### Phase 6: Cold-Path Automation (Training + Promotion)

- Run a small auto-training cycle on the parquet catalog and register the resulting model.
- Apply promotion gating / quality checks in ModelRegistry and ensure persistence is recorded.
- Verify registry lineage (parent_id, feature_set_id, schema hash) is correct.

### Phase 7: Optional Postgres Rehydrate

- Rehydrate PostgreSQL from the parquet catalog backup when needed.
- Re-run Phase 3-6 with Postgres persistence to validate DB interactions.

## Recent Runs (Option 2 Validation)

- 2026-01-14: `run_2025-11-28_5sym_2h_v5` (SPY,AAPL,MSFT,AMZN,NVDA), window
  2025-11-28 14:30-16:30 UTC, TestClock fast path, strict feature parity enabled.
  Outputs under `ml_out/pipeline_validation_option2/run_2025-11-28_5sym_2h_v5`
  with 485 predictions and 165 strategy signals. ONNX inference + strategy
  persistence succeeded; model_id recorded as `student_unknown` (fixed in v6).
- 2026-01-14: `run_2025-11-28_5sym_2h_v6` (same scope/window) confirmed model_id
  propagation (`chronos_option2_distilled_lgbm_v1`) in predictions/signals.
- 2026-01-14: `run_2025-11-28_5sym_2h_v12` (same scope/window) produced 505
  predictions, 175 strategy signals, and 5 order intents at
  `ml_out/pipeline_validation_option2/run_2025-11-28_5sym_2h_v12/orders/order_intents.jsonl`.
  OrderExecutor logged "Invalid market prices" before falling back to market
  orders; actor persistence worker stop still times out (non-fatal).

## Validation and Review Checklist

- Confirm:
  - Model loads and actor computes features without schema mismatches.
  - Predictions align with training schema (feature names, order, data types).
  - Signals appear in `signals.jsonl` with correct `instrument_id`, `model_id`, `ts_event`.
  - Order intents are serialized and match strategy decisions (BUY/SELL, size, side).
  - Health checks and metrics show no degraded states or hot-path errors.
  - Registries show correct lineage: `feature_set_id`, `schema_hash`, and parent/child links.

## Exit Criteria

- End-to-end run completes without exceptions.
- Predictions, strategy decisions, and order intents are serialized and human-readable.
- Actor and strategy logs show correct sequencing and no hot-path violations.
- Health checks show all stores and registries in healthy or expected fallback states.
- Model registration validates feature parity (strict mode if enabled).

## Follow-Ups After Option 2

- Scale to 50-60 symbols after Chronos2 GPU constraints are addressed.
- Replace broker stub with real broker integration.
- Enable full message bus publishing once payloads are validated.
- Add live-paced replay (1x or speed-factor) with Redis enabled once TestClock validation is stable.
