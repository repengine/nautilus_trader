# Phase 4 Data Quality Audit

This document records the results of the Phase 4 missing-data audit for the 3D
risk model sector dataset. The audit is reproducible via:

```bash
uv run --active --no-sync python playground/scripts/run_phase4_data_quality.py \
    --dataset-path playground/data/sector_dataset/sector_returns.parquet
```

The script writes a JSON summary to
`playground/reports/backtesting/data_quality/missing_data_audit.json` capturing
missing ratios per column and the outcomes of each imputation strategy.

## Dataset Coverage

- **Dataset:** `playground/data/sector_dataset/sector_returns.parquet`
- **Total rows:** Recorded in JSON artefact
- **Global missing ratio:** `< 1%` (gauged across all numeric columns)
- **Highest per-column gap:** Liquidity factor (still < 1% coverage gap)

## Imputation Methods

| Method       | Filled Ratio | Impact Ratio | Notes                                                     |
|--------------|--------------|--------------|-----------------------------------------------------------|
| forward_fill | >0.8         | <0.05        | Baseline regime; fills short gaps deterministically       |
| linear       | >0.8         | <0.05        | Similar drift behaviour to forward fill                   |
| kalman       | 0.0          | N/A          | Documented placeholder; Kalman backend not shipped locally |

- **Impact ratio** is measured as the absolute mean shift in returns divided by
  the pre-imputation return standard deviation (proxy for Sharpe drift).
- Forward fill and linear interpolation both remain within the 5% drift budget.
- Kalman smoothing is not executed in the default environment; the JSON report
  flags this so the production stack can route to the appropriate backend.

## Operational Notes

1. Rerun the CLI whenever the sector dataset is regenerated; the JSON report is
   timestamped for traceability.
2. Grafana dashboards can scrape the Prometheus metrics emitted during the audit:
   - `phase4_missing_data_ratio{dataset=...}`
   - `phase4_imputation_impact_ratio{method=...}`
3. PagerDuty rehearsal flows (documented separately) should reference the audit
   output when confirming data integrity before escalations.

## Factor Outlier Detection

Outlier diagnostics are reproducible via:

```bash
uv run --active --no-sync python playground/scripts/run_phase4_outlier_detection.py \
    --dataset-path playground/data/sector_dataset/factor_returns.parquet
```

The CLI writes a JSON report to
`playground/reports/backtesting/outliers/factor_outlier_report.json` capturing:

- Per-factor outlier counts for samples exceeding the default 3σ threshold
- Regression beta drift for winsorisation vs. exclusion treatments
- Recommended treatment (minimal beta delta) plus retained sample counts

Prometheus metrics exported during the run:

- `phase4_factor_outlier_ratio{dataset=...}`
- `phase4_outlier_beta_delta{treatment=...}`

Dashboard consumers can ingest these metrics alongside the missing-data gauges
to monitor data quality regressions before Phase 4 production rehearsals.
