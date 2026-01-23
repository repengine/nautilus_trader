# Positions Management Plan (Strategy + Portfolio Integration)

## Executive Summary

This plan formalizes a single, reliable positions management flow across ML
strategies, risk checks, and order submission. The goal is to make positions
readable and consistent across Nautilus Portfolio variants (live and backtest),
while keeping hot-path execution safe and fast. The plan introduces a small,
typed positions provider interface, a clear source priority chain, and
lightweight observability so E2E validation can trust position state.

## Goals

- Provide a single, authoritative positions access path for strategies, risk
  management, and order submission.
- Support multiple Nautilus Portfolio APIs (positions, positions_open,
  net_position) with safe fallbacks and metrics.
- Make position availability a first-class health check (degrade gracefully
  when missing).
- Keep hot-path behavior under P99 latency budgets (no I/O, no heavy
  allocations).

## Non-goals

- Building a new persistent PositionsStore (we rely on Portfolio/Cache).
- Introducing live broker execution changes or new order types.
- Changing risk or sizing logic beyond adapting the positions input source.

## Scope Assumptions

- Single brokerage account (IBKR) and a single venue for initial live hookup.
- Account routing and multi-account selection logic are out of scope.

## Current State (Implemented)

- Position sizing uses `PositionsProviderProtocol` via `PositionManagementComponent`
  with a cache-first fallback chain and logs provider failures.
  - Files: `ml/strategies/common/position_management.py`,
    `ml/strategies/common/positions_provider.py`
- Risk checks use provider snapshots for exposure/correlation and skip
  portfolio-wide checks when only `net_position` is available.
  - File: `ml/strategies/risk.py`
- Notional exposure uses a config-driven price source priority
  (quote tick midpoint -> position avg -> cache last) with cache-backed quote
  tick access when available.
  - Files: `ml/strategies/risk.py`, `ml/config/base.py`
- Order intent serialization includes quote tick staleness metadata
  (availability, age, max age, stale flag) captured during order submission.
  - Files: `ml/strategies/common/order_submission.py`,
    `ml/strategies/base_facade.py`
- Provider fallback order is configurable via `PositionsConfig`.
  - File: `ml/config/base.py`
- Strategy facade wires the provider into sizing and risk components.
  - File: `ml/strategies/base_facade.py`
- Strategy initialization performs a positions readiness check with degraded
  logging for live trading.
  - File: `ml/strategies/base_facade.py`
- Provider emits positions snapshot and readiness degradation metrics.
  - File: `ml/strategies/common/positions_provider.py`

## Validated Assumptions (Code Read)

- `PortfolioFacade` does not expose `positions()`/`positions_open()` in this
  build; it provides `net_position(...)` and exposure helpers instead.
  - Files: `nautilus_trader/portfolio/base.pyx`,
    `nautilus_trader/portfolio/portfolio.pyx`
- The cache exposes full position list APIs and is already used by strategies.
  - Files: `nautilus_trader/cache/base.pyx`, `ml/strategies/base_facade.py`
- `Position` stores quantity (absolute) and signed quantity; notional exposure
  is not stored directly and must be computed from price and multiplier.
  - File: `nautilus_trader/model/position.pyx`
- `Portfolio.net_position(...)` is derived from cached positions and can serve
  as a single-instrument fallback, but not for portfolio-wide exposure or
  correlation checks.

## Principles and Guardrails

- Hot-path safe: no I/O, no DataFrame construction, no heavy allocations.
- Config-driven: no hard-coded thresholds or source priorities.
- Explicit fallbacks with metrics and logs (`exc_info=True` on exceptions).
- Protocol-first: typed interfaces and adapters, no direct coupling to
  Nautilus concrete classes in strategy logic.
- Degrade safely: when positions are missing, skip correlation/exposure checks
  rather than failing entire strategy loops.
- Target: exposure math uses notional (price * quantity * multiplier).
- Price source fallback for exposure is explicit and configurable:
  quote tick midpoint -> position avg -> cache last; degrade to quantity only
  when no price source is available.

## Target Architecture

### Positions Provider (New)

Introduce a lightweight `PositionsProviderProtocol` that returns a
`PositionsSnapshot` (list of open positions + metadata).

Responsibilities:

- Normalize positions access across Portfolio/Cache variants.
- Track the positions source used (portfolio, cache, net_position-only).
- Provide minimal fields needed for sizing/risk checks.

### Source Priority (Configurable)

Default priority order for open positions:

1) `Cache.positions_open()` (authoritative open list in this repo)
2) `Cache.positions(...)` (if closed positions are required for analytics)
3) `Portfolio.net_position(instrument)` (single-instrument fallback)
4) `Portfolio.positions()` / `Portfolio.positions_open()` (only if available
   in a future build)

If only per-instrument positions are available, portfolio-level exposure and
correlation checks should be marked "limited" and skipped with metrics.

### Integration Points

- Position sizing uses provider snapshots for open positions, not direct cache.
- RiskManager uses the provider for exposure/correlation checks.
- Order submission continues to use cache instruments and prices, but logs the
  positions source used in decision metadata (for auditing).

## Phased Plan

### Phase 0 - Specification and Contracts (DONE)

- Define `PositionsProviderProtocol` and `PositionsSnapshot` types in
  `ml/strategies/common/positions.py`.
- Add a small `PositionsConfig` under `ml/config/base.py` to control:
  - Source priority order.
  - Whether positions are required for live trading.
  - Whether to allow degraded operation (skip exposure/correlation).
- Document snapshot fields and minimal invariants:
  - instrument_id present
  - quantity present
  - is_open flag (or equivalent)

### Phase 1 - Adapter + Strategy Integration (DONE)

- Implemented `NautilusPositionsProvider` with cache/portfolio fallbacks and
  fallback metrics (`ml_fallback_activations_total{component="positions_provider"}`).
  - File: `ml/strategies/common/positions_provider.py`
- Updated `PositionManagementComponent` to use provider snapshots with a safe
  cache fallback.
  - File: `ml/strategies/common/position_management.py`
- Updated `RiskManager` to consume provider snapshots and skip checks when only
  `net_position` is available.
  - File: `ml/strategies/risk.py`
- Wired provider into strategy facade for sizing and risk.
  - File: `ml/strategies/base_facade.py`

### Phase 2 - Health Checks and Observability (DONE)

- Add a `positions_ready` health check in strategy initialization to verify a
  usable positions source before live trading.
- Emit metrics:
  - `ml_positions_snapshot_total{source=...}`
  - `ml_positions_snapshot_empty_total{source=...}`
  - `ml_positions_checks_degraded_total{reason=...}`
- Ensure all exception logs include `exc_info=True`.

### Phase 3 - E2E Validation and Expansion (IN PROGRESS)

- TestClock validation (small symbol set) completed with
  `ml/orchestration/parquet_live_replay_harness.py`:
  - Run ID: `option2_small_scope_v1`
  - Instruments: `SPY.EQUS`, `AAPL.EQUS`, `MSFT.EQUS`, `NVDA.EQUS`, `AMZN.EQUS`
  - Window: 2025-11-28 14:30-16:30 UTC
  - Bars loaded: 605
  - Quote ticks loaded: 0 (catalog window lacks quote ticks for this run)
  - Output: `ml_out/parquet_live_replay_harness/option2_small_scope_v1`
- Quote-tick validation (small symbol set) completed with quote subscriptions:
  - Run ID: `option2_quote_ticks_small_v1`
  - Instruments: `SPY.EQUS`, `AAPL.EQUS`, `MSFT.EQUS`, `NVDA.EQUS`, `AMZN.EQUS`
  - Window: 2024-10-02 14:30-16:30 UTC
  - Bars loaded: 605
  - Quote ticks loaded: 600
  - Output: `ml_out/parquet_live_replay_harness/option2_quote_ticks_small_v1`
- Quote-tick validation (expanded set) completed:
  - Run ID: `option2_quote_ticks_15sym_v1`
  - Instruments: `SPY.EQUS`, `AAPL.EQUS`, `MSFT.EQUS`, `NVDA.EQUS`, `AMZN.EQUS`,
    `GOOG.EQUS`, `GOOGL.EQUS`, `META.EQUS`, `TSLA.EQUS`, `QQQ.EQUS`, `AMD.EQUS`,
    `AVGO.EQUS`, `CRM.EQUS`, `COST.EQUS`, `NFLX.EQUS`
  - Window: 2024-10-02 14:30-16:30 UTC
  - Bars loaded: 1_789
  - Quote ticks loaded: 1_800
  - Output: `ml_out/parquet_live_replay_harness/option2_quote_ticks_15sym_v1`
- Order-intent serialization validation (execute-trades on, broker stub):
  - Run ID: `option2_quote_ticks_order_intents_v1`
  - Window: 2024-10-02 14:30-16:30 UTC
  - Bars loaded: 605
  - Quote ticks loaded: 600
  - Order intents: 4 written to
    `ml_out/parquet_live_replay_harness/option2_quote_ticks_order_intents_v1/orders/order_intents.jsonl`
- Order-intent serialization validation (expanded set):
  - Run ID: `option2_quote_ticks_15sym_order_intents_v1`
  - Window: 2024-10-02 14:30-16:30 UTC
  - Bars loaded: 1_789
  - Quote ticks loaded: 1_800
  - Order intents: 13 written to
    `ml_out/parquet_live_replay_harness/option2_quote_ticks_15sym_order_intents_v1/orders/order_intents.jsonl`
- Order-intent serialization validation (quote metadata enabled, low thresholds):
  - Run ID: `option2_quote_ticks_order_intents_v3`
  - Window: 2024-10-02 14:30-16:30 UTC
  - Bars loaded: 605
  - Quote ticks loaded: 600
  - Order intents: 5 (quote metadata present in 5/5)
  - Model: `dummy_bullish_model` (prediction_threshold=0.1, min_confidence=0.1)
  - Max quote age: 60_000 ms
  - Output: `ml_out/parquet_live_replay_harness/option2_quote_ticks_order_intents_v3`
- Order-intent serialization validation (expanded set, quote metadata enabled):
  - Run ID: `option2_quote_ticks_15sym_order_intents_v3`
  - Window: 2024-10-02 14:30-16:30 UTC
  - Bars loaded: 1_789
  - Quote ticks loaded: 1_800
  - Order intents: 15 (quote metadata present in 15/15)
  - Model: `dummy_bullish_model` (prediction_threshold=0.1, min_confidence=0.1)
  - Max quote age: 60_000 ms
  - Output: `ml_out/parquet_live_replay_harness/option2_quote_ticks_15sym_order_intents_v3`
- Expand symbol scope once positions health checks remain green and quote tick
  coverage is confirmed for the chosen window.

## Testing Strategy

Unit tests (prefer property/contract):

- Provider fallback matrix:
  - Portfolio with `positions()` available.
  - Portfolio with only `positions_open()`.
  - Portfolio with only `net_position(...)` (single-instrument).
  - Cache-only fallback.
- Risk manager behavior when positions list is unavailable:
  - Exposure/correlation checks skipped with metrics.
- Position sizing uses provider snapshot and handles empty list gracefully.

Suggested locations:

- `ml/tests/unit/strategies/common/test_positions_provider.py`
- `ml/tests/unit/strategies/common/test_positions_contracts.py`
- `ml/tests/unit/strategies/test_sizing_and_risk_invariants.py`
- `ml/tests/unit/strategies/test_portfolio_and_exposure_invariants.py`

## Risks and Open Questions

- Price availability: notional exposure now uses available position price
  fields and emits degraded metrics when missing; a dedicated live price source
  may still be needed for full accuracy.
- Backtest vs live data parity: backtests may lack quote ticks; ensure
  positions access does not require tick data.
- Multi-instrument strategies: if only per-instrument positions are available,
  portfolio-wide risk checks will be limited until a full list is available.

## Next Tasks (Phase 3+)

- Expand symbol scope once positions health checks and notional exposure
  metrics remain green.
- Expand quote-tick-backed validation to a wider symbol set once the initial
  quote-mid path remains stable.
- Expand property/contract coverage for provider fallbacks and risk invariants
  using shared fixtures.

## References

- `ml/strategies/common/position_management.py`
- `ml/strategies/risk.py`
- `ml/strategies/common/order_submission.py`
- `ml/config/base.py`
