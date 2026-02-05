# Feature Parity Plan (Builder vs Pipeline)

## Purpose
Convert the current feature parity snapshot into an actionable plan that enforces a single canonical feature spec with batch + streaming execution backends, then removes legacy/duplicate feature paths once parity is proven.

## Goals
- One canonical feature spec (names + definitions) owned by `ml/features/pipeline.py`.
- Two execution backends for that spec: batch (dataset builder) and stream (online/inference).
- Training‑inference parity for all feature families once their live inputs are available.
- Remove legacy/duplicate feature computation modules after parity is validated.

## Non‑Goals (for now)
- Rewriting all data ingestion pipelines.
- Introducing new feature families not already present in the codebase.
- Changing data store schemas unrelated to feature parity.

## Current State Summary
- Three feature paths exist today: batch assembly in `ml/data/**`, online computation in `ml/features/common/feature_calculator.py`, and a declarative spec in `ml/features/pipeline.py` that is not used for batch execution.
- Several feature families overlap conceptually but diverge in names and definitions (OHLCV, calendar, events, microstructure).
- Macro revision features are largely aligned by name, but macro deltas remain builder‑only.
- Micro/L2 aggregates are canonical in batch datasets/schemas, but the online path uses OHLC approximations instead of L1/L2 aggregates.
- L2 derived columns and `is_l2_available` exist in `ml/data/tft_dataset_builder_facade.py` and are not referenced elsewhere.

## Live Input Dependencies (Parity Constraints)
| Feature Family | Required Inputs | Current Live Availability | Notes |
| --- | --- | --- | --- |
| OHLCV technical | L1 bars (open/high/low/close/volume) | Available | Online uses IndicatorManager; batch uses FeatureAlignmentComponent with different names. |
| Static covariates | Instrument metadata | Available | Builder emits a subset of pipeline static covariates. |
| Calendar features | Market calendar source | Available if provider wired | Provider emits more fields than pipeline spec. |
| Event schedule | Events + earnings calendars | Available if provider wired | Provider emits more fields than pipeline spec. |
| Macro (FRED/ALFRED) | Macro release calendar + as‑of join | Available if store + join wired | Names largely align with pipeline macro transform. |
| Earnings | Earnings store (actuals + estimates) | Available if store wired | Batch joins exist; no online backend yet. |
| Microstructure aggregates | L1 quotes + trades (tick data) | Not in online pipeline today | Batch uses MicrostructureAggregator or cache. |
| L2 depth aggregates | L2 depth feed (order book) | Not in online pipeline today | Batch uses L2Aggregator or cache. |

## Inventory (Feature Family Drift)

### OHLCV Technical Features
- Builder source: `ml/data/common/feature_alignment.py`
- Builder columns: `return_1`, `return_5`, `return_20`, `volume_ratio_20`, `volatility_20`, `price_sma_5`, `price_sma_20`, `price_position_20`
- Pipeline transforms: `returns`, `volatility`, `volume_ratio`, `core_indicators` in `ml/features/pipeline.py`
- Alignment status: Partial
- Notes: Pipeline names differ (`volume_ratio_{period}`, `price_sma_5/20`, `price_position_20`) and pipeline includes RSI/BB/ATR/EMA/MACD/HL spread not present in builder.

### Static Covariates
- Builder source: `ml/data/common/feature_alignment.py`
- Builder columns: `asset_class`, `tick_size`, `exchange`
- Pipeline transform: `static_covariates` in `ml/features/pipeline.py`
- Alignment status: Partial
- Notes: Pipeline expects additional static features (lot_size, contract_size, currency, fee_class, etc.) that builder does not emit.

### Known‑Future Time Features (Base)
- Builder source: `ml/data/common/known_future_features.py`
- Builder columns: `hour`, `minute`, `tod_sin`, `tod_cos`, `dow`, `dow_sin`, `dow_cos`, `is_market_open`, `is_premarket`, `is_aftermarket`
- Pipeline transform: `calendar` in `ml/features/pipeline.py`
- Alignment status: Low
- Notes: Pipeline expects `hour_sin/hour_cos/minute_sin/minute_cos` and month/quarter features. Builder uses `tod_*` and session flags with different names.

### Calendar Provider Features (Known‑Future)
- Provider source: `ml/data/providers/calendar.py`
- Provider columns: `is_trading_day`, `is_pre_market`, `is_after_hours`, `minutes_to_close`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `is_weekend`, `is_month_start`, `is_month_end`, `is_quarter_start`, `is_quarter_end`, `days_to_month_end`, `days_from_month_start`
- Pipeline transform: `calendar` in `ml/features/pipeline.py`
- Alignment status: Partial
- Notes: Several provider columns are not in the pipeline spec (`is_trading_day`, `minutes_to_close`, `is_pre_market`, `is_after_hours`).

### Event Schedule Features
- Provider source: `ml/data/providers/events.py`
- Provider columns include: `hours_to_{event}`, `has_{event}_in_24h`, `has_{event}_in_week`, `{event}_within_{h}h`, `total_events_24h`, `total_events_week`, `event_density_24h`, `event_density_week`, `is_triple_witching`, `is_fomc_week`, `is_earnings_season`, `is_holiday_week`, `days_to_next_holiday`
- Pipeline transform: `event_schedule` in `ml/features/pipeline.py`
- Alignment status: Partial
- Notes: Provider also emits `has_*_today`, `days_to_next_*`, `days_since_last_*`, `event_importance_score`, `event_clustering_score` which are not in the pipeline spec.

### Macro Features (FRED / ALFRED)
- Builder source: `ml/data/fred_join.py`, `ml/data/macro_revisions.py`
- Builder columns: `{series}__value_real_time`, `{series}_prior_1m`, `{series}_revision_1m`, `{series}_mom_1m`, `{series}_pct_1m`, `{series}_net_signal_1m`, `{series}__value_vintage_ts`, optional `{series}__value_final`
- Pipeline transform: `macro` in `ml/features/pipeline.py`
- Alignment status: Strong
- Notes: Naming aligns for base + revision features. Builder’s macro deltas are not represented in the pipeline spec.

### Cross‑Asset Transforms
- Pipeline transforms: `ewma_beta`, `zscore_spread` in `ml/features/pipeline.py`
- Builder source: None (no batch execution path today)
- Alignment status: None
- Notes: These transforms exist in the declarative spec but have no batch backend and no builder integration. They require multi‑instrument input alignment and a canonical batch execution path.

### Microstructure Aggregates (L1)
- Source: `ml/features/micro_aggregate.py`
- Columns: `MICRO_COLUMNS` (`midprice`, `spread_bps`, `quote_imbalance`, `trade_imbalance`, `realized_vol`)
- Pipeline transform: `microstructure` in `ml/features/pipeline.py`
- Alignment status: None
- Notes: Pipeline microstructure uses OHLC approximations with different column names (`spread_mean`, `size_imbalance_mean`, etc.). These are distinct feature families.

### L2 Aggregates (Depth)
- Source: `ml/features/l2_aggregate.py`
- Columns: `L2_MINUTE_COLUMNS` (`midprice`, `spread_bps`, `microprice_bps`, depth imbalance, dwp_bps, bid/ask slopes)
- Pipeline transform: None (no L2 aggregate transform)
- Alignment status: None
- Notes: Legacy builder adds derived L2 columns (`pressure_accel_top{k}`, `liquidity_gradient_top{k}`, `session_rel_spread`) and `is_l2_available` that are not in schemas or pipeline spec.

## Canonicalization Decisions (Locked 2026-02-04)
### Naming (Canonical = Pipeline Spec)
- Canonical OHLCV names follow `ml/features/pipeline.py`.
- Use aliases during transition:\n  - `volume_ratio` -> `volume_ratio_20`\n  - `sma_5` / `sma_20` -> `price_sma_5` / `price_sma_20`\n  - `price_position` -> `price_position_20`\n  - `volatility_20` remains; add `volatility_5` to match pipeline.\n  - `return_{period}` names remain canonical.\n+- Canonical calendar names follow pipeline spec for time encodings:\n  - `tod_sin` / `tod_cos` -> `hour_sin` / `hour_cos` (minute precision)\n  - Add `minute_sin` / `minute_cos` to batch backend to satisfy pipeline spec.\n+- Canonical session/market flags align to provider naming and are added to the spec:\n  - `is_market_open` -> `is_market_hours`\n  - `is_premarket` -> `is_pre_market`\n  - `is_aftermarket` -> `is_after_hours`\n  - `is_trading_day`, `minutes_to_close` remain canonical.\n+\n+### Calendar/Event Field Scope\n+- Canonical calendar fields = pipeline time encodings + provider session flags + month/quarter flags.\n+- Canonical event schedule fields = pipeline `event_schedule` outputs.\n+- Provider extras (`has_*_today`, `days_since_last_*`, `event_importance_score`, `event_clustering_score`) are supplemental and should be removed or explicitly added to the spec via a decision.\n+\n+### Macro Features\n+- Canonical macro fields = pipeline macro transform outputs (base + revisions).\n+- Macro deltas become a distinct optional transform (`macro_deltas`) in the canonical spec (no longer builder-only).\n+\n+### Micro/L2 and Cross-Asset Gating (DataRequirements)\n+- Microstructure aggregates: require `DataRequirements.L1_L2`.\n+- L2 depth aggregates: require `DataRequirements.L1_L2` (upgrade to L1_L2_L3 only if full depth is required beyond MBP-10).\n+- Trade flow (pipeline): require `DataRequirements.L1_L2_L3`.\n+- Cross-asset transforms (`ewma_beta`, `zscore_spread`): require multi-instrument inputs and should be gated by `DataRequirements.L1_ONLY` plus an explicit multi-instrument capability flag.

## Phased Plan (Actionable)

### Phase 0 — Decisions and Canonical Spec
Checklist:
- [x] Decide canonical names for OHLCV overlaps and write a definitive mapping table.
- [x] Decide which calendar/event fields are canonical vs supplemental.
- [x] Decide whether macro deltas become part of the canonical spec.
- [x] Decide parity policy for micro/L2 (gated by `DataRequirements`).
- [x] Record decisions in this plan and update any impacted configs.

### Phase 1 — Canonical Spec + Batch Backend
Checklist:
- [x] Extend `ml/features/pipeline.py` with any missing canonical feature names.
- [x] Add a batch execution backend for `PipelineSpec` (Polars/Pandas).
- [x] Add batch execution tests for each transform in the canonical spec.
- [x] Ensure `FeatureConfig.get_feature_names()` and batch backend produce identical names across all canonical transforms.

### Phase 2 — Builder Refactor to Canonical Spec
Checklist:
- [x] Refactor `FeatureAlignmentComponent` to delegate to the batch backend.
- [x] Refactor `KnownFutureFeatureComponent` to delegate to canonical calendar/event transforms.
- [x] Align dataset builder output columns to canonical names (remove aliases or add shims as needed).
- [x] Update dataset build metadata capability flags to match canonical spec.

### Phase 3 — Live Backend Coverage
Checklist:
- [x] Implement `compute_stream` for each canonical transform where inputs are available live. (OHLCV + micro/trade_flow via `PipelineStreamExecutor`; calendar/event/macro supported with preloaded providers/transform. `macro_deltas`/`macro_indicators` remain gated.)
- [x] For macro/events/earnings, add live join helpers with cache/lookup semantics. (Macro via `macro_transform` cache, events via `EventScheduleProvider`; earnings features beyond event schedule remain gated.)
- [x] For micro/L2, wire streaming inputs (quotes/trades/depth) or gate features by `DataRequirements` until available. (Gated via `PipelineRunner` with stream executor tests.)
- [ ] Add parity tests comparing batch vs stream outputs where feasible. (OHLCV + calendar/event + macro parity tests added; micro/L2 parity pending.)

### Phase 4 — Parity Validation & Contracts
Checklist:
- [x] Add contract tests for every feature family’s canonical column set.
- [ ] Add parity tolerance tests for overlapping batch/stream families. (OHLCV + calendar/event + macro + micro/trade_flow done; L2 pending until streaming backend exists and the Databento subscription is active again.)
- [x] Update audit harnesses to validate canonical feature names.

### Phase 5 — Cleanup and Removal (Post‑Parity)
Checklist:
- [x] Remove or slim `ml/data/common/feature_alignment.py` (retain only adapters if needed).
- [x] Remove or slim `ml/data/common/known_future_features.py` (retain only adapters if needed).
- [x] Remove legacy L2 derived columns unless explicitly approved into canonical spec.
- [x] Remove any remaining legacy column aliases and update tests/docs.
- [x] Add guardrails to prevent new feature implementations outside the canonical spec.

## Validation & Test Plan
- Types: `poetry run mypy ml --strict`
- Lint: `poetry run ruff check ml`
- Fixtures: `make validate-fixtures`
- Feature unit tests: `poetry run pytest ml/tests/unit/features/test_microstructure.py ml/tests/unit/features/test_l2_aggregate.py`
- Builder tests: `poetry run pytest ml/tests/unit/data/test_tft_dataset_builder_phase_one.py ml/tests/unit/data/test_tft_dataset_builder_store.py ml/tests/unit/data/test_tft_builder_integration.py`
- E2E: `pytest -q ml/tests/e2e/test_tft_dataset_builder_e2e.py`

## Risks
- Changing canonical names may break downstream consumers unless aliases are maintained during transition.
- Live parity for micro/L2 requires data feeds that the online pipeline does not currently use.
- Calendar/event features have overlapping sources with inconsistent naming; unification may require data migration.

## Notes
- Legacy‑only L2 columns (`pressure_accel_*`, `liquidity_gradient_*`, `session_rel_spread`, `is_l2_available`) are not referenced elsewhere and should not be carried forward without explicit approval.
- Canonical cache behavior: micro/L2 joins are cache-first via `micro_cache_policy`/`l2_cache_policy` (default `cache_first`) and fill newly joined numeric columns with 0. `ML_TFT_FORCE_MICRO_CACHE=1` is mapped in orchestration config loading to enforce cache-first policies for audit runs.
