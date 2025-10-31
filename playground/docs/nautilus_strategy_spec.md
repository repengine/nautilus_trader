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
| Minimum Variance | Low-volatility baseline to stress downside protection. |
| 3D Factor (Rolling Betas) | Approved Phase 3 live candidate used as the fifth benchmark. |

Benchmark summaries and baseline metrics are mirrored under
`playground/reports/backtesting/benchmarks/` for every suite execution, allowing
Grafana/PagerDuty automation to ingest consistent CSV/JSON artefacts.

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
4. Parameter heatmaps, diagnostics, proxy validations, and monitoring snapshots
   inherit their defaults from `ThreeDRiskBacktestDefaults`; regenerate artefacts
   via the CLI flags (`--parameter-heatmaps`, `--extended-diagnostics`,
   `--proxy-validation`, `--monitoring-export`) whenever these configurations are
   tuned.

## Monte Carlo Overlay Catalog (Phase 3)

Current overlays encoded in `ThreeDRiskBacktestDefaults().monte_carlo_stress`:

- `rate_hike_shock` *(rates)* – Tightening-aligned drawdown applied to rate sensitive regimes.
- `growth_scare` *(growth)* – Multi-session negative drift reflecting macro slowdowns.
- `liquidity_crunch` *(liquidity)* – Acute liquidity withdrawal aligned with stress regimes.
- `volatility_breakout` *(volatility)* – Elevated realised vol hitting returns over ~1 week.
- `cross_asset_contagion` *(cross_asset)* – Equity/credit/commodity deleveraging tandem event.
- `compound_liquidity_growth` *(compound)* – Sequential liquidity then growth shock cascade.
- `credit_spread_widening` *(credit)* – Cyclical credit spread blowout across tightening regimes.
- `inflation_repricing` *(inflation)* – Surprise inflation repricing duration-sensitive sectors.
- `energy_supply_shock` *(commodities)* – Energy supply disruption impacting equities and credit.

Per-path overlay activations, category aggregates, and baseline metrics are
persisted under `stress/monte_carlo/` for Grafana dashboards.

## Proxy & Vintage Coverage

Proxy datasets now include:

- International sectors (baseline proxy)
- Factor ETF proxy universe
- Global macro overlay dataset
- Treasury futures hedge proxy *(new)* – guards duration hedging assumptions

Vintage simulations cover:

- Five-Year Rolling (5y/1y)
- Seven-Year Rolling (7y/1y)
- Three-Year High Frequency (3y/1y)
- Crisis Response 2y/1y *(new)* – emphasises rapid regime adaptation

All proxies/vintages persist metadata (status, allow-missing, fold counts) for
monitoring and telemetry.

## Phase 3 Automation Flags

- `--heatmap-specs` triggers the parameter heatmap suite automatically when
  specific spec slugs are provided, enabling targeted refreshes without setting
  `--parameter-heatmaps`.
- `--phase3-battery` executes the full validation stack (walk-forward, Monte
  Carlo, heatmaps, diagnostics, proxy/vintage suites, monitoring export) in a
  single command for nightly smoke tests and pre-deployment rehearsals.
- Monitoring exports also persist ``monitoring/grafana_dashboard_payload.json``
  and ``monitoring/pagerduty_alert_payload.json`` to streamline dashboard and
  escalation automation.
- Use ``poetry run python playground/scripts/publish_phase3_monitoring_integrations.py``
  after refreshing the snapshot to update manifests and generate optional PagerDuty
  rehearsal payloads for on-call drills.
- CI/cron jobs can call ``make publish-phase3-monitoring`` with
  ``SIMULATE_ESCALATION=1`` to refresh payloads and rehearsal artefacts.
- ``--stress-runs`` repeats the chosen suites to simulate heavier nightly load,
  and ``--profile-output`` captures cProfile stats per execution for regression
  tracking alongside the runtime histogram metric.
- Phase 4 harnesses:
  - ``uv run --active --no-sync python playground/scripts/run_phase4_sensitivity.py``
    sweeps the robustness grids defined in ``ThreeDRiskBacktestDefaults.parameter_sensitivity``.
  - ``uv run --active --no-sync python playground/scripts/run_phase4_data_quality.py``
    audits missing data coverage and emits JSON artefacts plus Prometheus
    telemetry.
  - ``uv run --active --no-sync python playground/scripts/run_phase4_outlier_detection.py``
    evaluates >3σ factor outliers, compares winsorisation versus exclusion, and
    updates ``reports/backtesting/outliers/factor_outlier_report.json``.
  - ``uv run --active --no-sync python playground/scripts/generate_phase4_sensitivity_report.py``
    transforms the consolidated sensitivity summary into ``reports/backtesting/sensitivity/sensitivity_analysis.pdf`` for
    executive review packets.

### PagerDuty Rehearsal Playbook

1. Refresh the monitoring snapshot with the full validation battery:

   .. code-block:: bash

       uv run --active --no-sync python playground/scripts/run_phase3_walk_forward.py \
           --phase3-battery \
           --monitoring-export

2. Publish the integration payloads with rehearsal output enabled:

   .. code-block:: bash

       uv run --active --no-sync python playground/scripts/publish_phase3_monitoring_integrations.py \
           --simulate-escalation

3. The step above writes ``playground/reports/backtesting/monitoring/pagerduty_escalation_dry_run.json``.
   Load this file in the PagerDuty developer console as a custom event to walk through the escalation
   path without paging production responders. The payload includes proxy dataset status, vintage coverage,
   and diagnostics metadata so responders can validate context during the rehearsal.

4. Document the rehearsal outcome (success/follow-ups) alongside the timestamp and service rule name in
   the on-call runbook. The ``generated_at`` field within the rehearsal payload must match the recorded
   dry-run entry to maintain auditability.

5. Remove rehearsal artefacts once the drill completes to avoid accidental reuse, or regenerate via the
   same commands for subsequent walk-throughs.
