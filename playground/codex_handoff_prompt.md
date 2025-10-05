# Codex Handoff Prompt – 3D Risk Matrix Initiative

Use this prompt verbatim (adjusting the final task checkboxes as you complete them) when continuing the 3D risk model effort. It captures goals, context, constraints, and current progress.

---

## System Context & Standards

- **Repo:** Nautilus Trader ML (`ml/` package) – strict typing, Ruff linting, Pattern compliance.  
- **Code Style:** see `ml/docs/development/CODING_STANDARDS.md`. Always run `uv run --active --no-sync mypy ml --strict` and `make ruff` before delivery.  
- **Universal Patterns:** Public API via `__init__.py`, hot/cold parity, 4-store + 4-registry, no heavy work in hot paths.  
- **Macro Data:** FRED/ALFRED loaders under `ml/data/loaders`, revision joins via `ml/data/fred_join.py` and `ml/data/macro_revisions.py`.  
- **Feature Pipelines:** `ml/features/engineering.py`, `ml/features/pipeline.py`, macro transforms in `ml/features/macro_transforms.py`, composites in `ml/features/macro_composites.py`.  
- **Cross-Asset Exposure:** EWMA beta utils (`ml/features/cross_asset/`), storage schema (`ml/schema/cross_asset_features.sql`).  
- **Testing Philosophy:** Unit + parity tests (Polars/pandas), property tests for transforms, deterministic dataset validators. Write tests alongside new loaders/validators/optimizers.

## High-Level Goals

1. **Complete macro composite integration** (FeatureConfig flag, realtime parity, validators).  
2. **Source and stage Fama/French & extended macro datasets** with ingestors or loaders.  
3. **Build the 3D risk positioning pipeline**: factor creation, asset exposures, target definitions, optimization, visualization.

## Current Status

- FRED & ALFRED loaders operational; point-in-time join + revision features in place.  
- Macro composites implemented (`ml/features/macro_composites.py`) and registered, but not yet exposed via FeatureConfig or realtime path.  
- TFT dataset builder already supports `include_macro` for base transforms; composites not wired.  
- `playground/3d_risk_architecture.md` documents architecture, data sources, and todos.  
- `playground/portfolio-3d-risk.html` contains a Three.js prototype for visualizing assets vs. an ideal point.  
- No Fama/French ingestion yet; no cross-asset exposures persisted. Python users access via pandas-datareader while R users employ the FFresearch package.  
- Risk pipeline upgrades (2025-03): calendar-aware expected session counts, persisted `coverage_summary.json`, Mahalanobis distance metrics, decade-level eigenvalue trend summaries, and CLI support for custom `--calendar`/proxy mappings.

## Directory Layout (relevant pieces)

- `ml/data/loaders/*`: data ingestion (FRED/ALFRED).  
- `ml/data/fred_join.py`, `ml/data/macro_revisions.py`: macro joins & revisions.  
- `ml/features/`: FeatureConfig, transforms, composites, cross-asset utilities.  
- `ml/schema/cross_asset_features.sql`: beta/spread/correlation tables.  
- `ml/strategies/portfolio.py`, `ml/strategies/risk.py`: downstream consumers.  
- `playground/`: prototype docs (`3D_Risk_Model_Idea.md`, `3d_risk_architecture.md`), risk visual (`portfolio-3d-risk.html`).

## Task Roadmap

1. **Macro Composite Wiring**  
   - Add `include_macro_composites` to `FeatureConfig`; update `build_pipeline_spec_from_feature_config`.  
   - Extend `MacroFeatureTransform.compute_realtime` to emit composites (either inline or via `MacroDataCache`).  
   - Ensure training/inference parity tests cover composites (update parity test suite).  

2. **Coverage Validators**  
   - Implement `MacroCoverageValidator` (as per `ml/features/CODEX_RECOMMENDATIONS_STATUS.md`).  
   - Integrate into `TFTDatasetBuilder` after macro join; log/raise on missing/sparse series.  
   - Prime `MacroDataCache` on init, log coverage gaps.

3. **Fama/French & External Factors**  
   - Write a loader (e.g., `ml/data/loaders/fama_french_loader.py`) that downloads and stages factor CSV/TXT files (daily/monthly frequencies).  
   - Normalize column names, handle CIZ format changes, and integrate with dataset manifesting if needed.  
   - Add tests ensuring downloads parse correctly and include metadata (frequency, start date).

4. **Asset Return Sourcing**  
   - Prototype ETF/asset return ingestion using yfinance (or configure IngestionOrchestrator to pull from a paid source).  
   - Store returns in a consistent schema (for regression inputs).  
   - Document fallback plans if API limits hit.

5. **Factor Exposure Pipeline**  
   - Compute factor return series from composites (daily differences, standardized).  
   - Run rolling EWMA betas for each instrument vs. synthetic factors; store results in `ml_cross_asset_betas`.  
   - Create or reuse CLI/service to update exposures daily, with unit tests verifying parity (EWMA vs. batch).  
   - _Status_: Risk pipeline persists betas via `CrossAssetBetaPersistenceConfig`; CLI supports `--persist-betas` for on-demand writes with unit coverage. Dedicated compose service `postgres_playground` (host port `${PLAYGROUND_POSTGRES_HOST_PORT:-5435}`) keeps these artifacts siloed from market-data and core ML databases.

6. **Target Point Definition**  
   - Choose an initial “ideal” risk point (e.g., from historical best-performing portfolio).  
   - Persist the target vector (create new schema/file if necessary).  
   - Document methodology in `playground/3d_risk_architecture.md`.

7. **Optimization & Monitoring**  
   - Implement an optimizer that maps factor betas to weights to hit the target point (least squares/QP with constraints).  
   - Expose API endpoints or CLI to compute recommended allocations.  
   - Extend dashboard or playground visual to consume live coordinates, target point, and recommendations.  
   - _New_: surface Mahalanobis outliers and eigenvalue trend metadata in the Three.js panel; add badges for fallback years using the payload metadata.
   - _Status_: Optimizer upgraded (cvxpy) with long-only + cap constraints; pipeline/CLI output `[optimizer]` block summarising weights and beta persistence.

8. **Documentation & Validation**  
   - Update docs under `ml/docs/context/context_features.md` or new dedicated sections.  
   - Add validation scripts/reports under `ml/tests/validation_reports/` tracking macro coverage, factor drift.  
   - Ensure final PRs include mypy + Ruff + relevant tests (`make pytest -k ...`).

## Important Methods / Modules

- `FREDDataLoader`, `ALFREDDataLoader` – data ingestion.  
- `join_fred_asof`, `compute_revision_features_pl` – point-in-time joins.  
- `FeatureConfig`, `FeatureEngineer`, `MacroFeatureTransform`, `compute_macro_composites_pl`.  
- `compute_ewma_beta_incremental/batch`, `EWMABetaState`.  
- Dashboard metrics aggregator: `ml/dashboard/services/metrics_service.py` (extend to display new risk metrics).

## Deliverables Checklist

- [x] FeatureConfig & pipeline support for macro composites (batch + realtime parity).  
- [x] Macro coverage validator + documentation.  
- [x] Fama/French (and other necessary) datasets ingested with tests.  
- [x] Factor exposure storage pipeline (schema updates, CLI/service, tests).  
- [x] Optimization module that outputs recommended weights to move portfolio toward target.  
- [x] Calendar-aware coverage metadata persisted to disk and payloads (Mahalanobis + eigenvalue diagnostics surfaced).  
- [x] Risk pipeline enforces minimum sector/factor coverage (MacroCoverageValidator + metrics).  
- [x] Updated docs & validation reports (3D architecture doc refreshed with proxy map; extend validation reports next).  
- [x] Playground/frontend integration demonstrating live 3D risk positioning.  
- [ ] All new pieces covered by mypy, Ruff, and targeted pytest suites.

## Validation Steps (before handing off)

1. Run `uv run --active --no-sync mypy ml --strict`.  
2. Run `make ruff`.  
3. Execute relevant tests (`make pytest -k 'macro or cross_asset or risk'` etc.).  
4. Verify coverage of new validators/optimizers with unit tests.  
5. Generate or update validation report for macro coverage / factor drift.  
6. If data ingestion scripts run, confirm they respect caching, rate limits, and emit metrics (factor cache now expands for historical runs).  
7. Update `playground/3d_risk_architecture.md` with any architectural changes (proxy mapping + diagnostics recorded).

## Communication Notes

- Keep this prompt updated as milestones complete—tick deliverables, adjust tasks.  
- Mention outstanding blockers or required approvals (e.g., data provider credentials) in the prompt before handing off.  
- Use the “playground/3d_risk_architecture.md” file as the living long-form design doc; keep this prompt focused on actionable next steps.

---

_Ready for next agent: copy/paste this prompt as-is and begin with the highest-priority unchecked tasks._
