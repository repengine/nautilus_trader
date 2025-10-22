# Phase 3.3 Attribution Visual Specification

This note captures the draft structure for the visualization layer that will
consume the CSV exports produced by `playground/backtest/runner.py`.

## Inputs
- `playground/reports/backtesting/performance_comparison_table.csv`
- `playground/reports/backtesting/train_vs_test_metrics.csv`
- `playground/reports/backtesting/regime_summary.csv`
- `playground/reports/backtesting/attribution/*.csv`
- `playground/reports/backtesting/attribution/regime/*.csv`

## Visuals
1. **Rolling vs Benchmark Sharpe**
   - Line chart: Sharpe ratio across strategies (Equal Weight, 60/40, Rolling,
     Stable) by train/test period.
   - Annotate periods where rolling Sharpe exceeds 60/40.
2. **Regime Contribution Bars**
   - Stacked bar per regime showing factor contributions and alpha for 3D
     Rolling Betas.
   - Highlight liquidity bar in red when negative.
3. **Liquidity Stress Panel**
   - Scatter of liquidity contribution vs regime Sharpe for 3D Rolling Betas.
   - Overlay thresholds used by liquidity scaling heuristics.
4. **Sharpe vs Transaction Costs**
   - Scatter plot comparing Sharpe ratio and turnover costs across factor
     strategies and the 60/40 benchmark.
5. **Attribution Waterfall (Test Period)**
   - Waterfall from benchmark return to strategy return using factor
     contribution deltas.

## Deliverables
- `playground/reports/backtesting/visuals/phase3_attribution.ipynb`
  (prototype notebook; deterministic data loads).
- Rendered PNG/SVG exports stored alongside Markdown report.
- `python playground/scripts/export_phase3_visuals.py` regenerates the PNG
  artefacts (`rolling_vs_benchmark_sharpe.png`, `regime_contributions.png`,
  `liquidity_stress_panel.png`, `sharpe_vs_tc.png`,
  `attribution_waterfall.png`) without opening the notebook.
- Updated Markdown section in `playground/reports/backtesting/latest.md`
  referencing generated figures.
