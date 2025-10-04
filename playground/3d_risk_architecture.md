# 3D Risk Matrix Architecture Notes

## Purpose

We want a parallel research effort that maps portfolio and asset exposures into a three-axis risk space (Duration, Credit, Liquidity). The system should:

- Produce factor coordinates for each instrument using historical macro data and rolling regressions.
- Define and track an “ideal” point in the 3D space based on risk-adjusted portfolio targets.
- Monitor how assets drift relative to that point and recommend rebalancing actions.
- Eventually invert the model to suggest an optimal portfolio location given predicted asset positions and macro context.

This document captures existing capabilities, data sourcing requirements, architectural integration points, and open issues.

---

## Existing Infrastructure We Can Reuse

### Macro Data Loaders and Joins
- `ml/data/loaders/fred_loader.py` fetches FRED spot series with caching, retries, and Prometheus metrics.
- `ml/data/loaders/alfred_loader.py` downloads full ALFRED vintage histories.
- `ml/data/fred_join.py` performs point-in-time as-of joins between market data and FRED/ALFRED series with configurable lags and revision modes.
- `ml/data/macro_revisions.py` converts vintage releases into revision-aware features (current, prior values, revision deltas, net signals) for batch use.

### Macro Feature Layer
- `ml/features/macro_transforms.py` exposes a `MacroFeatureTransform` with batch (`compute_batch`) and realtime (`compute_realtime`) parity backed by `MacroDataCache` (`ml/features/macro_cache.py`).
- `ml/features/macro_composites.py` defines 26 composite signals (credit spreads, term structure, liquidity, growth/inflation, FX) derived from the base macro series.
- `ml/features/pipeline.py` registers both the base macro transform (`name="macro"`) and the composites transform (`name="macro_composites"`).
- `ml/features/engineering.py` exposes toggles for macro, calendar, and composite transforms; enabling `include_macro_composites` now injects the derived factors end-to-end via the pipeline spec.

### Cross-Asset Exposure Tooling
- `ml/features/cross_asset/beta.py`, `ml/features/cross_asset/state.py`, and `ml/features/pipeline.py` (EWMA beta transform) provide rolling beta computation with hot/cold parity.
- `ml/schema/cross_asset_features.sql` contains table definitions for storing EWMA betas, spreads, and correlations keyed by feature set, asset, benchmark, and timestamp.
- Portfolio and risk managers (`ml/strategies/portfolio.py`, `ml/strategies/risk.py`) already track exposure, correlation limits, and drawdowns; they can consume new factor signals once persisted.
- `ml/exposure/factor_exposure.py` converts factor composites into factor returns and materializes EWMA betas aligned with `ml_cross_asset_betas`.
- `ml/exposure/optimizer.py` defines the default 3D target point and a weight solver (`compute_optimal_weights`) for steering portfolios toward that target subject to non-negative weights.

### Playground Research Implementation (2025-02)
- `playground/risk_model/dataset.py` adds a typed `SectorDatasetAssembler` that aligns sector return series with factor data, emits metrics, and optionally persists parquet artifacts under `playground/data/`.
- `playground/risk_model/analysis.py` orchestrates annual risk profile computation (Sharpe-weighted blends with target alignment), sector distance calculations, and summary statistics reusable across notebooks/tests.
- `playground/risk_model/visualization.py` serializes the resulting coordinates into a JSON payload compatible with `playground/portfolio-3d-risk.html`.
- Public exports live in `playground/risk_model/__init__.py`; top-level `playground/__init__.py` re-exports the package for notebook ergonomics while keeping production modules untouched.
- Tests covering dataset alignment, exposure summaries, distance metrics, and payload serialization land in `playground/tests/unit/risk_model/` to guarantee invariants before iterating in notebooks or the Three.js view.
- `playground/risk_model/fetchers.py` wires sector returns to the shared yfinance adapter and factor levels to FRED (duration/credit/liquidity proxies) with optional parquet caching, while `playground/risk_model/pipeline.py` provides the end-to-end build that emits annual risk profiles and visualization payloads for the Three.js view. Optimizer fallbacks are logged and surfaced in the generated metadata so notebooks can flag years that required Sharpe-only weights.
- Historical replay now honours the `XNYS` trading calendar when computing expected session counts; coverage summaries (sector, factor, and composite) are persisted alongside parquet outputs (`coverage_summary.json`) and are injected into visualization payload metadata for downstream consumers. Coverage shortfalls are materialised separately in `coverage_alerts.json` and exposed via the CLI/Visualization payloads.
- The Three.js playground (`playground/portfolio-3d-risk.html`) now hydrates directly from the generated visualization payloads, renders dynamic axis labels based on factor names, and surfaces composite/sector coverage alerts as status badges with Mahalanobis tooltips. Missing alert metadata degrades gracefully to “all clear” messaging so older payloads remain viewable.
- Coverage alerts are exported as Prometheus-compatible gauges (`playground_coverage_alert_total`, `playground_coverage_alert_ratio`) to support dashboards that highlight deficit counts and residual coverage ratios by dimension.
- Cached FRED factor pulls automatically expand to satisfy wider date ranges (e.g., 1970s backfills) without manual cache invalidation.
- Mahalanobis distances between sector coordinates and the ideal point are computed (with pseudo-inverse fallback) and recorded as metrics/payload fields to highlight outlier regimes.
- Eigenvalue diagnostics captured during annual profile generation are aggregated into decade buckets so the Three.js view (or notebooks) can surface long-horizon covariance regime shifts.

## Roadmap Checklist

- [x] Real-time composite generation parity (`MacroFeatureTransform` batch/realtime).
- [x] Coverage alert surfacing in CLI, payloads, and Three.js visualization.
- [x] FeatureConfig wiring for macro composites (`include_macro_composites`) plus parity tests.
- [x] MacroCoverageValidator rollout across macro pipelines (batch + realtime) with documentation.
- [ ] External data ingestion expansion (Fama/French loader, ETF price ingest prototype → orchestrated path).
- [ ] Factor exposure persistence and constrained optimizer delivering target-aligned weights.
- [ ] Documentation & validation artefacts (macro coverage, factor drift, optimizer health reports).

### Research Roadmap (High-Grade Focus)
- Extend sector proxies and FRED factors back to the 1970s (or earliest credible data), documenting proxy substitutions pre-ETF and re-running the pipeline for multi-regime coverage.
- Introduce calendar-aware coverage checks, missing-session metrics, and factor completeness validation to prevent silent gaps before EWMA betas are computed.
- Persist diagnostic metrics (factor covariance eigenvalues, Mahalanobis distances, fallback counts) to explain how the 3D sector cloud evolves and when optimisation downgrades occur.
- Add configurable optimisation constraints (turnover, sector min/max) using regularised solvers or QP backends, with structured logging for solver choices and degradation events.
- Package a reproducible notebook and CLI runner that builds datasets, publishes visualisation payloads, and wires status/diagnostics into `playground/portfolio-3d-risk.html`.

### TFT Dataset Builder
- `ml/data/tft_dataset_builder.py` joins macro features into training datasets when `include_macro=True`, preserving point-in-time integrity. This makes the same transforms available to models used for factor prediction.

---

## External Data Sources

### FRED (Federal Reserve Economic Data)
- Yields: `DGS1`, `DGS2`, `DGS5`, `DGS10`, `DGS30`
- Term spreads: `T10Y2Y`, `T10Y3M`
- Real yields: `DFII10`
- Fed policy & liquidity: `FEDFUNDS`, `SOFR`, `WALCL`, `TOTBKCR`, `M2SL`
- Credit spreads: `BAMLC0A0CM` (IG OAS), `BAMLH0A0HYM2` (HY OAS), `BAMLC0A4CBBB` (BBB OAS)
- Volatility proxies: `VIXCLS`, `MOVE` (if accessible via FRED or alternative API)
- Macro activity: `PAYEMS`, `UNRATE`, `INDPRO`, `CFNAI`, `CPIAUCSL`, `PCEPI`, `PPIACO`
- FX: `DTWEXBGS`, `DEXUSEU`, `DEXUSAL`, `DEXJPUS`

Access: Use `fredapi` (already vendored via `ml._imports.fredapi`). Requires `FRED_API_KEY` (we already enforce this in the loaders). Historical depth ranges from the 1950s to present depending on the series. Update cadence is daily/weekly/monthly.

### ALFRED (Vintage Archive)
- Same series as FRED but with full release histories for revision-aware features.
- `ALFREDDataLoader` writes per-series release calendars (parquet) into `data/fred/vintages/<series>/release_calendar.parquet`.
- The vintage information powers `join_fred_asof(... include_revisions=True)` and real-time cache coverage.

### Fama/French Data Library (for Factor Benchmarks)
- Downloadable CSV/TXT sets (Daily/Weekly/Monthly) covering:
  - U.S. factors: 3-factor, 5-factor (Rm-Rf, SMB, HML, RMW, CMA), Momentum (Mom), Short-Term & Long-Term Reversal, etc.
  - Portfolio sorts (Size/BM, Size/Profitability, Size/Investment) and global/regional counterparts.
- International developed & emerging market factors and portfolios.
- CRSP format change: from FIZ to CIZ in Jan 2025; monthly returns now compounded from daily data with dividends reinvested on ex-date. Be aware when aligning historical calculations.

Access Patterns:
- Direct HTTP download from [https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). For automation, parse zipped TXT/CSV and stage into our catalog. Python utilities: `pandas-datareader` (FF data), but raw CSV ingestion gives more control.
- Document new loader scripts under `ml/data/loaders/fama_french_loader.py` (TODO) with metadata (frequency, coverage, compression) and optional caching.

### Asset Price Data (for Factor Betas)
- For prototyping: `yfinance` (`SPY`, `TLT`, `HYG`, `IWM`, `GLD`, etc.).
- Production alternatives: Polygon.io, AlphaVantage, IEX Cloud, Quandl (Nasdaq Data Link). Choose provider based on licensing & SLA. Integrate via existing ingestion framework (e.g., dataset discovery + IngestionOrchestrator) for parity with other stores.

---

## Proposed 3D Risk Workflow

### 1. Build Factor Series
- Use `compute_macro_composites_pl` to compute daily composite scores. Align frequency (daily) and fill missing values carefully (forward-fill within acceptable windows).
- Optionally supplement with Fama/French factor returns for benchmarking or as alternative coordinates.

### 2. Compute Asset Coordinates
- Collect daily returns for each asset (ETFs, futures, indices) and align with factor series.
- Run rolling regressions to estimate betas:
  - EWMA beta (already implemented) for fast, incremental updates.
  - Optional OLS over fixed windows for offline validation.
- Persist exposures into `ml_cross_asset_betas` with `asset_id`, `benchmark_id` (e.g., `DURATION`, `CREDIT`, `LIQUIDITY` synthetic factor IDs) and metadata (window, alpha, sample count).

### 3. Define the “Ideal” Point
- Choose methodology:
  - Historical optimal portfolios (e.g., highest Sharpe per quarter) converted to factor coordinates.
  - Risk-parity or custom optimization results anchored to macro regimes.
  - Manual specification for initial experiments.
- Store the target vector `(β_d*, β_c*, β_l*)` in a configuration or derived table (`ml_cross_asset_targets`—new schema) with timestamp/versioning.

### 4. Monitor Positions in 3D Space
- For each asset, track current coordinates relative to the ideal point. Compute distance metrics (Euclidean, Mahalanobis) and drift velocity.
- Visualize via `playground/portfolio-3d-risk.html`, eventually fed by a backend endpoint (e.g., `/api/risk/3d` returning asset coordinates, target point, portfolio allocations).

### 5. Suggest Portfolio Adjustments
- Solve a constrained optimization problem to find weights `w` minimizing `||Bᵀ w − t||`, where `B` is the matrix of asset betas and `t` is the target vector.
- Add constraints: sum of weights = 1, weights ≥ 0 (long-only) or allow leverage/shorts per policy, exposure caps, correlation limits.
- Integrate with existing portfolio manager to stage allocations and ensure risk manager enforces boundaries.

### 6. Optional Inversion (Predict Target)
- Train a model mapping macro state (composite levels, spreads, event flags) to the ideal point location or the desired factor mix. Requires a historical label (e.g., best-performing portfolio each month) or a utility function that defines “ideal”. Without labels, the problem is underdetermined.
- Use TFT or other sequence models to predict future coordinates and infer the target location given predicted macro/factor trajectories.

---

## Architectural Integration Points

| Component | Role |
|-----------|------|
| `ml/data/loaders/fred_loader.py`, `ml/data/loaders/alfred_loader.py` | Source spot and vintage macro data |
| `ml/data/fred_join.py`, `ml/data/macro_revisions.py` | Point-in-time joins and revision features |
| `ml/features/macro_transforms.py`, `ml/features/macro_composites.py` | Feature transforms producing macro & composite signals |
| `ml/features/engineering.py`, `ml/features/pipeline.py` | Pipeline builder and FeatureConfig flags controlling transforms |
| `ml/features/cross_asset/beta.py`, `ml/features/cross_asset/state.py` | Rolling exposure computation |
| `ml/schema/cross_asset_features.sql` | Persistent storage for betas/spreads/correlations |
| `ml/strategies/portfolio.py`, `ml/strategies/risk.py` | Consume exposures for allocation/risk decisions |
| `ml/dashboard/services/metrics_service.py` | Surfacing portfolio metrics; extend to show factor drift |
| `playground/portfolio-3d-risk.html` | Visualization prototype for interactive 3D risk space |

---

## Coverage Diagnostics Workflow

1. Execute the pipeline via `python -m playground.risk_model.cli --coverage-report <path>` to persist
   both `coverage_summary.json` and the alert payload (`coverage_alerts.json`). The CLI mirrors the
   structure stored under `playground/tests/validation_reports/coverage_alerts_example.json` and prints
   the alert dictionary to stdout for quick inspection.
2. Open `playground/portfolio-3d-risk.html?payload=playground/data/visualizations/risk_<year>.json` to
   visualise the alert state. Composite and factor deficits appear as colour-coded badges in the sidebar
   and include Mahalanobis-based tooltips for sector-level diagnostics.
   The CLI now also emits a human-readable coverage summary so analysts can sanity-check ratios without
   opening the JSON artefacts.
3. Share updated snapshots by refreshing the validation artefacts in
   `playground/tests/validation_reports/` whenever coverage thresholds or pipeline inputs change.

---

## Outstanding Tasks & Issues

### Historical Proxy Mapping (2025-03)

The CLI now accepts multiple `--sector-proxy` overrides; the default candidate list focuses on ETF tickers, but multi-decade replays should include S&P sector indices that pre-date modern ETFs. Recommended overrides (verify coverage via `uv run --active --no-sync python -m playground.risk_model.cli --sector-proxy ... --calendar XNYS`):

| Sector ETF | Primary Candidates | Pre-ETF Proxy | Notes |
|------------|-------------------|---------------|-------|
| XLF (Financials) | XLF, IYF | `^SP500-40` | S&P 500 Financials index, available back to 1989 on Yahoo Finance. |
| XLK (Technology) | XLK, VGT | `^SP500-45` | S&P 500 Information Technology index; confirm coverage before 1989 and fall back to `^NDX` if needed. |
| XLE (Energy) | XLE | `^SP500-10` | S&P 500 Energy index; pairs well with fossil fuel proxy futures when ETF gaps arise. |
| XLY (Consumer Discretionary) | XLY | `^SP500-25` | S&P 500 Consumer Discretionary index. |
| XLC (Communication Services) | XLC | `^SP500-50` | Legacy Telecom index; supplement with `^XTC` if telecom data is sparse. |
| XLI (Industrials) | XLI | `^SP500-20` | Industrials index; retains history into the 1970s. |
| XLB (Materials) | XLB | `^SP500-15` | Materials index; use alongside futures-based proxies when liquidity drops. |
| XLV (Health Care) | XLV | `^SP500-35` | Health Care index. |
| XLU (Utilities) | XLU | `^SP500-55` | Utilities index with deep history.

Document any deviations (e.g., regional mutual funds) in this file as coverage experiments evolve. Upcoming work should integrate automated selection heuristics that prefer the deepest proxy meeting the coverage threshold.

1. **FeatureConfig Flag for Composites**  
   - Add `include_macro_composites` to `FeatureConfig`, update `build_pipeline_spec_from_feature_config`, and extend realtime parity coverage.  
   - Expand macro parity tests to cover composite factors end-to-end.

2. **Coverage Validation**  
   - Implement `MacroCoverageValidator` as outlined in `ml/features/CODEX_RECOMMENDATIONS_STATUS.md` to ensure all requested macro series stay populated.  
   - Prime `MacroDataCache`, log gaps, and document the validator flow.
   - _Status_: MacroFeatureTransform now enforces coverage during batch assembly and logs cache availability for realtime paths.

3. **External Data Ingestion**  
   - Ship `ml/data/loaders/fama_french_loader.py` (daily/monthly support) with schema-normalized outputs and caching strategy.  
   - Prototype ETF/asset return ingestion via yfinance, then graduate to an orchestrated pipeline with quality metrics.

4. **Exposure & Optimization**  
   - ✅ EWMA betas for the synthetic factors now persist into `ml_cross_asset_betas` via the risk pipeline (`CrossAssetBetaPersistenceConfig`) with unit coverage.  
   - ✅ Constrained optimizer (long-only with optional caps) emits recommended weights, surfaced through pipeline results and the CLI (`--max-weight`, `--weight-cap`, `--persist-betas`).

5. **Visualization Backend**  
   - Provide an API endpoint streaming coordinates, target, and recommendations; incorporate live Mahalanobis diagnostics and coverage badges in the UI timeline controls.

6. **Documentation & Validation**  
   - Refresh context docs with composite coverage flows; publish validation artefacts (coverage, factor drift, optimizer health) alongside release notes.  
   - Keep unit/property tests current for validators, loaders, optimizers, and visualization payload contracts.
   - Produce validation reports for factor coverage and exposure accuracy.

---

## References & Useful Links

- FRED API documentation: https://fred.stlouisfed.org/docs/api/fred/  
- fredapi Python wrapper: https://github.com/mortada/fredapi  
- Fama/French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html  
- RiskMetrics (1996) EWMA methodology: technical background for `alpha=0.94`  
- Macro feature documentation in repo: `ml/features/README_MACRO_FEATURES.md`

---

## Next Steps

1. Enable macro composites via FeatureConfig and ensure both batch and realtime paths emit the new features.  
2. Stage Fama/French + enhanced macro series into our catalog, ensuring ALFRED coverage is monitored.  
3. Build the exposure computation pipeline (betas, storage schema, APIs).  
4. Prototype the optimization layer that maps desired factor coordinates to portfolio weights.  
5. Connect the playground visualization to live data and iterate on UX for monitoring and recommendations.
