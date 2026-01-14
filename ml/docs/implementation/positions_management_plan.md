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

## Current State (Relevant Components)

- Position sizing uses `PositionManagementComponent` and relies on:
  - Account + instrument from cache.
  - `cache.positions_open(...)` for current positions.
  - Risk checks via `RiskManager` (portfolio-based).
  - File: `ml/strategies/common/position_management.py`
- Risk checks use `Portfolio.positions()` or `Portfolio.positions_open()` and
  fall back to empty list when unavailable.
  - File: `ml/strategies/risk.py`
- Portfolio API can vary by build; some variants expose only
  `net_position(...)` and `update_position(...)`.
- Strategy can serialize order intents (dry run) with min-quantity fallback to
  keep test pipelines moving.
  - File: `ml/config/base.py`

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
- Exposure math uses notional (price * quantity * multiplier), not raw
  quantity. Prefer quote mid; fall back to last/bar close; degrade if no price.

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

### Phase 0 - Specification and Contracts

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

### Phase 1 - Adapter + Strategy Integration

- Implement `NautilusPositionsProvider` that detects Portfolio/Cache APIs and
  returns a normalized snapshot.
- Update `PositionManagementComponent` to use the provider instead of directly
  calling `cache.positions_open(...)`.
- Update `RiskManager` to consume provider snapshots and record the source in
  metrics/logs.
- Keep fallbacks non-blocking; emit `ml_fallback_activations_total` when
  downgrading positions checks.

### Phase 2 - Health Checks and Observability

- Add a `positions_ready` health check in strategy initialization to verify a
  usable positions source before live trading.
- Emit metrics:
  - `ml_positions_snapshot_total{source=...}`
  - `ml_positions_snapshot_empty_total{source=...}`
  - `ml_positions_checks_degraded_total{reason=...}`
- Ensure all exception logs include `exc_info=True`.

### Phase 3 - E2E Validation and Expansion

- Validate positions behavior in Option 2 TestClock runs (small symbol set).
- Confirm order intents include position context (source + size basis).
- Expand symbol scope once positions health checks are green.

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
- `ml/tests/unit/strategies/test_portfolio_and_exposure_invariants.py`

## Risks and Open Questions

- Price availability: if no quote/last is available, notional exposure checks
  are skipped and must emit a degraded metric.
- Backtest vs live data parity: backtests may lack quote ticks; ensure
  positions access does not require tick data.
- Multi-instrument strategies: if only per-instrument positions are available,
  portfolio-wide risk checks will be limited until a full list is available.

## References

- `ml/strategies/common/position_management.py`
- `ml/strategies/risk.py`
- `ml/strategies/common/order_submission.py`
- `ml/config/base.py`
