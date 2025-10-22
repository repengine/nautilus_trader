# Nautilus Strategy Specification – Accepted Parameters

**Version:** 1.0  
**Date:** 2025-10-17

This specification captures the parameter values approved for the Phase 3 3D
Factor Risk Model when preparing the Nautilus Trader integration. The values are
sourced from `ml.config.playground.ThreeDRiskBacktestDefaults` to guarantee that
all orchestration layers (backtest runner, sensitivity/grid-search utilities,
walk-forward CLI, and documentation) stay aligned.

## Core Backtest Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Risk-free rate | **2.0% annual** | Aligns Sharpe/Sortino calculations across metrics modules and walk-forward analytics. |
| Stable-turnover smoothing | **0.30** | Provides moderate damping for the stable-beta strategy, balancing TC impact vs responsiveness. |
| Rolling-turnover smoothing | **0.40** | Additional smoothing for dynamic betas to offset higher rebalance frequency. |
| Training window (minimum) | **1,250 trading days** (≈5 years) | Ensures factor estimation sees at least one full market cycle. |
| Testing window (minimum) | **250 trading days** (≈1 year) | Requires a full year of out-of-sample coverage before publishing metrics. |
| Coverage tolerance | **±7 calendar days** | Allows for weekly datasets while preventing material drift beyond dataset coverage. |

## Liquidity Mitigation Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Severe liquidity threshold | **-2.0%** annualised | Regime-level mitigation activates when attribution drops below this level. |
| Moderate liquidity threshold | **-1.0%** annualised | Partial mitigation band prior to severe activation. |
| Severe regime multiplier | **0.85** | Applies broad forecast damping for severe liquidity drag. |
| Moderate regime multiplier | **0.92** | Softer damping for moderate drag environments. |
| Severe liquidity multiplier | **0.55** | Factor-specific override applied to `factor_liquidity`. |
| Moderate liquidity multiplier | **0.70** | Factor-specific override for moderate stress. |
| Neutral liquidity multiplier | **1.0** | Preserves exposure when attribution is non-negative. |
| Multiplier floor | **0.40** | Prevents liquidity scaling from zeroing out exposure entirely. |

## Baseline Strategies

| Strategy | Purpose |
|----------|---------|
| Equal Weight | Canonical baseline for all comparisons. |
| 60/40 Portfolio | Core benchmark representing traditional allocation. |
| Risk Parity | Captures diversified risk contribution benchmark. |

## Walk-Forward & Stress Scenario Notes

- Walk-forward outputs now include `metadata.json` that records the defaults in
  effect (risk-free rate, turnover smoothing, liquidity config) for traceability.
- Liquidity mitigation experiments ship with a **Turnover Stress Test** scenario
  that dials turnover smoothing down to stress transaction-cost sensitivity while
  retaining the shared liquidity configuration.

## Usage Guidelines

1. Consume `ThreeDRiskBacktestDefaults` in all new orchestration/CLI code so
   defaults remain centralized.
2. Update this document **and** the defaults dataclass in tandem when approving
   parameter changes.
3. Backtests or experiments that deviate from these values should record the
   overrides explicitly (e.g., via `turnover_overrides` or scenario metadata).
