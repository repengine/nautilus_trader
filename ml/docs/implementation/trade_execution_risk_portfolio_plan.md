# Trade, Execution, Risk, and Portfolio Management Plan

## Overall Goals

- Make trade entry, exit, and reversals explicit, auditable, and config-driven.
- Align execution behavior with quote freshness and risk constraints.
- Keep portfolio state authoritative via the positions provider with safe fallbacks.
- Preserve hot-path safety (no I/O, no blocking calls) while capturing intent metadata.

## Overall Guardrails

- All tunables live in `ml/config` dataclasses with validation in `__post_init__`.
- Hot-path limits: no file I/O, network calls, or heavy allocations.
- Always use `ml.common.metrics_bootstrap` for metrics and `exc_info=True` in exception logs.
- Use `PositionsProviderProtocol` for positions access; do not bypass with ad hoc cache lookups.
- Keep order intents metadata consistent (positions + quote staleness + exit reason).

## Current State Summary

- Trade direction and reversals are signal-driven; reversals close with reduce-only market orders (`ml/strategies/ml_strategy.py`).
- Exit policy (stop-loss, take-profit, max holding) is wired into the strategy loop with reduce-only orders (`ml/strategies/base_facade.py`).
- Smart execution uses `OrderExecutor` with stale-quote fallback, limit price fallbacks, and TTL cancel/replace support (`ml/strategies/common/order_submission.py`, `ml/strategies/execution.py`).
- Risk checks enforce per-trade limits, exposure, correlation, daily loss, and drawdown; risk halts gate order submission (`ml/strategies/risk.py`, `ml/strategies/common/order_submission.py`).
- Correlation checks use provider snapshots with freshness/fallback config (`ml/config/base.py`, `ml/strategies/risk.py`, `ml/strategies/portfolio.py`).
- Realized PnL from position closes feeds risk and sizing updates with event-time reset alignment (`ml/strategies/base_facade.py`, `ml/strategies/risk.py`).
- Positions provider uses a fallback chain and readiness checks with metrics (`ml/strategies/common/positions_provider.py`, `ml/strategies/base_facade.py`).
- Order intents include positions, quote staleness, exit, and execution metadata (`ml/strategies/base_facade.py`, `ml/strategies/common/order_submission.py`).

## Trade Lifecycle

### Goals

- Define explicit exit policy (stop-loss, take-profit, timeouts) in addition to signal reversals.
- Ensure exits are reduce-only and intent metadata includes exit reasons.

### Guardrails

- Exits must be config-driven; no hard-coded thresholds.
- All exit decisions are persisted before submission.
- Avoid new hot-path I/O; use cache and provider interfaces only.

### Tasks

- [x] Define an `ExitPolicyConfig` (or equivalent) under `ml/config` and validate ranges.
- [x] Wire exit evaluation into the strategy loop (price-based + time-based + reversal).
- [x] Place reduce-only exit orders via `OrderSubmissionComponent`.
- [x] Persist exit intent metadata (reason, trigger price, time-in-trade).
- [x] Add unit tests for exit triggers (stop-loss, take-profit, timeout, reversal).

### Definition of Done

- [x] Exits are triggered deterministically from config in live and replay runs.
- [x] Order intents include `exit_reason` metadata for all closes.
- [x] Unit tests cover all exit paths and pass in `pytest -k trade_exit`.
- [x] Updated docs reflect exit policy and config defaults.

### Model-Driven Exits (Prediction-Based)

#### Goals

- Use model signal changes to exit, reduce, or reverse positions with explicit thresholds and hysteresis.
- Keep stop-loss, take-profit, and timeouts as safety rails even when model exits are enabled.

#### Guardrails

- Config-driven in `ml/config` dataclasses with validation in `__post_init__`.
- DRY: shared helper in `ml/strategies/common/` (protocol-first) reused by strategies.
- Explicitly typed: all new helpers, configs, and metadata structures include full annotations.
- Persist exit intent metadata before any order submission; no hot-path I/O.

#### Tasks

- [x] Add a `ModelExitConfig` (or similar) in `ml/config/base.py` with thresholds such as
  `exit_on_flip`, `exit_confidence_threshold`, `exit_prediction_band`, `min_hold_ms`,
  and `reverse_on_flip`; validate ranges.
- [x] Extend `MLStrategyConfig` and `ml/config/replay_harness.py` to surface model-exit knobs;
  keep `exit_policy_config` in sync and avoid duplicated fields.
- [x] Implement a shared helper in `ml/strategies/common/` (e.g., `model_exit_policy.py`)
  that returns a typed `ExitDecision` dataclass (`action`, `reason`, `trigger_price`,
  `time_in_trade_ns`, `confidence`) and uses Protocols for inputs.
- [x] Wire the helper into `ml/strategies/ml_strategy.py` so model exits run after
  stop-loss/take-profit/timeout checks but before reversal; honor config for
  exit-versus-reverse and hysteresis.
- [x] Emit metrics for model-driven exits (via `ml.common.metrics_bootstrap`) and include
  exit metadata in decision persistence and order intents.

#### TDD + Tests

- [x] Start with failing unit tests for the model-exit helper covering flip,
  confidence drop, neutral-zone exit, min-hold, and no-op cases.
- [x] Update `ml/tests/unit/strategies/test_ml_trading_strategy_exit_policy.py`
  (or add a new suite) to assert ordering: stop-loss/TP/timeout first, then
  model exit, then reversal.
- [x] Add property tests for invariants (no exit when confidence is stable above
  threshold; no reverse when configured to exit-to-flat on flip).

#### Validation (Replay Harness)

- [ ] Run the replay harness over a longer window (>= 5 trading days or the maximum
  available catalog range) with `--execute-trades` and quote ticks enabled.
- [ ] Inspect `backtest_result.json` and logs for exit reasons distribution,
  positions lifecycle, and absence of order rejections.

#### Definition of Done

- [x] Model-driven exits are config-driven, type-safe, and DRY across strategies.
- [x] Decisions and order intents include model-exit reasons and thresholds.
- [x] Tests cover flip/neutral/confidence/ordering and pass `pytest -k model_exit`.
- [ ] Replay harness run demonstrates rational exits across multiple sessions.

## Execution

### Goals

- Enforce quote freshness and spread constraints for smart execution.
- Provide deterministic behavior for limit-order TTL (or explicitly disable it).

### Guardrails

- Execution must degrade safely when quotes are stale or missing.
- Submit path must remain non-blocking; order management runs off the hot path.
- Metrics emitted for fallback reasons and order types.

### Tasks

- [x] Decide execution policy: keep smart execution and emit TTL plan metadata (advisory) until a cancel/replace loop exists.
- [x] If TTL support is required, add an order-management loop to cancel/replace limits.
- [x] Make stale-quote policy explicit (config, log, and metrics).
- [x] Add tests for stale quotes and executor fallback paths.
- [x] Define the order-type decision matrix (market vs aggressive/passive limit) using confidence, spread, quote freshness, and `ExecutionConfig` thresholds.
- [x] Treat missing quotes as an explicit market fallback with `quote_unavailable` metadata.
- [x] Specify the canonical price source + fallback chain for limit pricing (e.g., quote mid/BBO → last trade → market) with config knobs.
- [x] Clarify TTL semantics (apply to resting limits only, cadence vs TTL, replacement uses fresh market state, stop after attempts).
- [x] Emit low-cardinality metrics for execution mode and fallback reason.
- [x] Add unit tests for TTL replacement edge cases.

### Order-Type Decision Matrix

- **Eligibility**: if `signal.confidence < ExecutionConfig.min_confidence` → no order.
- **Urgency determination** (`OrderExecutor._determine_urgency`):
  - `confidence >= market_order_threshold` → high urgency.
  - `spread_bps > max_spread_bps` → low urgency.
  - `confidence >= limit_order_threshold` → medium urgency.
  - else → low urgency.
- **Maker preference adjustment**: if urgency is high and `prefer_maker_orders` is true with
  `spread_bps <= prefer_maker_spread_bps`, downgrade to medium urgency (aggressive limit).
- **Order type**:
  - high urgency → market order (IOC if `use_time_in_force_ioc` else GTC).
  - medium urgency → aggressive limit (IOC if `use_time_in_force_ioc` else GTC).
  - low urgency → passive limit (GTC, `post_only` when `prefer_maker_orders`).
- **Limit pricing**: resolve via `LimitPriceConfig.source_priority` (BBO/mid → last trade → cache).
- **TTL semantics**: TTL applies only to resting limits; replacements stop once attempts exhausted.
- **Tests**: `ml/tests/unit/strategies/test_order_executor_selection_logic.py`.

### Definition of Done

- [x] Execution behavior is fully defined for each order type (market/limit).
- [x] Order-type decision matrix is documented and unit-tested.
- [x] Limit pricing + fallback chain is explicit and covered by tests.
- [x] TTL replacement semantics are documented and covered by tests.
- [x] Execution metrics exist for order mode and fallback reason.
- [x] TTL behavior is either enforced with tests or disabled by config.
- [x] Order intents include execution mode and fallback reason.

## Risk

### Goals

- Enforce risk limits with real PnL and exposure data.
- Replace heuristic correlation with data-driven correlation inputs.
- Add staged risk brakes (halt → liquidate) with multiple config-driven triggers.

### Guardrails

- All limits and thresholds are config-driven.
- Risk checks must degrade gracefully if positions are unavailable.
- Risk-triggered halts must be observable and reversible.
- Liquidation triggers must be explicit and emit metrics per action/reason.
- Reduce-only exits may be allowed during halts only when configured.

### Tasks

- [x] Feed realized PnL and equity updates into `RiskManager` from fill events (include fees and reset boundaries).
- [x] Replace correlation heuristic with data-driven correlations (registry or rolling) and validate data freshness.
- [x] Expose risk-halt state + reason from `RiskManager` and gate order submission via the breaker (with metrics).
- [x] Add property/contract tests for risk invariants and fallback behavior (daily loss, drawdown, correlation, exposure).
- [x] Add staged risk action config (`RiskLiquidationConfig`) with multiple trigger options
  (daily loss, drawdown, unrealized loss, cooldown, require full positions list).
- [x] Implement staged risk decisioning in `RiskManager` (`RiskAction` + metrics) and surface
  allow-reduce-only-when-halted behavior.
- [x] Wire staged risk actions into strategy flow: liquidation submits reduce-only exits,
  halt blocks new entries but allows configured reduce-only exits.
- [x] Fix `max_positions` gating so exit logic still runs when a position is already open.
- [x] Add unit tests for staged risk actions, reduce-only halt bypass, liquidation intents,
  and max-position exit handling.

### Definition of Done

- [x] Daily PnL and drawdown metrics update from real fills.
- [x] Correlation checks use data sources with tests to verify thresholds.
- [x] Risk halts block new orders, expose halt reason, and emit metrics/logs.
- [x] Staged risk actions are config-driven with clear metrics and observability.
- [x] Reduce-only exit behavior during halts is deterministic and tested.
- [x] `max_positions` no longer prevents exits or liquidation while in a position.

## Portfolio Management

### Goals

- Keep positions access authoritative via the provider with explicit readiness checks.
- Require full positions list for live trading when configured.

### Guardrails

- No direct portfolio/cache access outside provider or dedicated adapters.
- Degraded mode must be explicit and observable.

### Tasks

- [x] Tighten live readiness: require full positions list when `positions_required_for_live` is set.
- [x] Expand contract tests for all provider fallbacks (cache/portfolio/net).
- [x] Ensure positions metadata is persisted in decisions and intents for audits.

### Definition of Done

- [x] Live mode fails fast if positions readiness is not met.
- [x] Provider fallbacks are covered by tests and emit metrics.
- [x] Audit metadata includes positions source and readiness state.

## References

- `ml/strategies/ml_strategy.py`
- `ml/strategies/base_facade.py`
- `ml/strategies/common/order_submission.py`
- `ml/strategies/execution.py`
- `ml/strategies/risk.py`
- `ml/strategies/common/positions_provider.py`
