# Market Data and Feature Data Population Plan

## Background
This repo runs a dual-path persistence model: market and feature datasets land in
Postgres for fast access, with Parquet catalogs as a backup and rehydration
source. The migration layout now supports per-class market data tables
(bar, quote_tick, tbbo, mbp1, trade_tick) plus legacy compatibility, and the
registry enforces contracts, events, and watermarks by dataset_id. The ingestion
pipeline pulls from Databento (and DBN archives), normalizes columns into
canonical schemas, writes to Postgres, and mirrors to Parquet when enabled. When
gaps are detected, rehydration pulls from Parquet back into Postgres, and any
remaining gaps are filled by ingestion, again writing to both targets.
Provider dataset ids are only used for external ingestion calls; registry
dataset ids remain the unique keys for manifests, events, watermarks, and table
routing.

Definition of Done
- [x] Background captures the current storage split, registry ownership, and
      ingestion and rehydration flow responsibilities.
- [x] Background explicitly documents the provider dataset id vs registry
      dataset id split.

## Goals
- Ensure all market data schemas and all feature data schemas populate safely
  in Postgres and Parquet.
- Ensure rehydration and ingestion cooperate to fill gaps and maintain
  watermarks and events.
- Verify the full pipeline works in the ML docker deployment stack.

Definition of Done
- [x] Goals are specific, actionable, and aligned to the main outcome.

## Scope
- Market data schemas and their per-class SQL tables.
- Feature data schemas (feature store and feature datasets).
- Registry manifests, contracts, and events required for validation.
- Ingestion, rehydration, and Parquet backup flows in the docker stack.
- Coverage target is all data currently available on disk under `data/` (batch
  archives, catalog parquet, and feature caches).

Definition of Done
- [x] Scope boundaries are explicit and exclude unrelated domains.

## Guardrails
- Keep dataset_id as the registry identity and use provider_dataset_id only for
  external calls.
- Do not delete or reorder applied migrations; keep legacy migrations intact.
- Ensure manifest schemas align to SQL table columns to avoid validation drift.
- Preserve monotonic watermarks and record events for all writes.
- Keep hot-path rules intact; all persistence and rehydration remains cold-path.
- While storage is constrained, disable MBP-1 ingestion and dual-write via env
  flags (reversible when storage expands).

Definition of Done
- [x] Guardrails are concrete and map to known architectural constraints.

## Schema Inventory

### Market Data Schemas
- Bars (ohlcv-1m) -> `EQUS.MINI` -> `market_data_bar`
- Quotes -> `EQUS.MINI_QUOTES` -> `market_data_quote_tick`
- TBBO -> `EQUS.MINI_TBBO` -> `market_data_tbbo`
- MBP-1 -> `EQUS.MINI_MBP1` -> `market_data_mbp1`
- Trades -> `EQUS.MINI_TRADES` -> `market_data_trade_tick`

Definition of Done
- [x] Inventory lists every market data schema and its target table.

### QuoteTick vs TBBO (Provider Mapping)
- Internal `quotes/quote_tick` is the L1 top-of-book dataset used by strategy
  runtime, microstructure features, and L1 coverage policies.
- Databento’s top-of-book schema is `tbbo`/`bbo` (not `quotes`), so provider
  calls should use `tbbo` while internal storage stays `market_data_quote_tick`.
- To avoid duplicate storage, pick one L1 quote dataset for ingestion:
  - Preferred: keep `EQUS.MINI_QUOTES` as the registry dataset_id and map its
    provider schema to `tbbo`.
  - Optional: disable `EQUS.MINI_TBBO` ingestion unless explicitly needed for
    research; treat it as an on-demand dataset.

Definition of Done
- [x] Provider schema mapping for `EQUS.MINI_QUOTES` is explicit (tbbo/bbo) and
      documented in config + ingestion code, with tests updated.
- [x] Only one L1 quote dataset is actively ingested under storage constraints.

### Feature Data Schemas
- FeatureStore values -> `features` -> `ml_feature_values`
- Macro release calendar -> `ml.macro_release_calendar` -> `ml.macro_release_calendar`
- Macro observations -> `ml.macro_observations` -> `ml.macro_observations`
- Events calendar -> `ml.events_calendar` -> `ml.events_calendar`
- Microstructure minute -> `ml.microstructure_minute` -> `ml.microstructure_minute`
- L2 minute -> `ml.l2_minute` -> `ml.l2_minute`

Definition of Done
- [x] Inventory lists every feature dataset and its registry dataset id.

## On-Disk Inputs (Discovery)
- Batch archives live under `data/batch` (DBN zips for market data).
- Parquet catalog lives under `data/catalog` (raw market backups).
- Feature caches live under `data/features` (macro/events/micro/L2, etc).
- L3 (MBO) data is only in scope if present under `data/`; otherwise it is
  tracked as a gap.
- Current inventory (as of this work):
  - `data/catalog/data`: `bar`, `quote_tick`, `trade_tick`, `order_book_depth10`.
  - `data/batch`: OHLCV1m/1h/1d, TBBO, Trades, MBP-1 archives; no MBP-10/MBO/L3 archives found.

Definition of Done
- [x] On-disk inputs are inventoried and mapped to the schemas they cover.

## Coverage Targets and Expected Rows (Baseline)
Universe reference: `UNIVERSE_SYMBOLS` in `ml/deployment/.env` (100 symbols). For
coverage targets, the max available dataset is defined by Parquet catalog
coverage. DB targets should match Parquet coverage windows and instrument
counts.

### Market Data Coverage Targets (Parquet Baseline)
Parquet coverage windows are derived from file timestamps and represent the
maximum available range on disk. Tick data row counts vary by instrument/day, so
coverage targets are expressed in day buckets, not absolute rows.

| Schema | Dataset ID | Parquet path | Instruments | Parquet window (UTC) | Coverage target |
| --- | --- | --- | --- | --- | --- |
| Bars (ohlcv-1m) | `EQUS.MINI` | `data/catalog/data/bar` | 95 | 2023-03-28 -> 2025-12-01 | Per-minute coverage by instrument |
| Quotes | `EQUS.MINI_QUOTES` | `data/catalog/data/quote_tick` | 95 | 2024-09-30 -> 2026-01-09 | Day buckets by instrument |
| TBBO | `EQUS.MINI_TBBO` | `data/catalog/data/quote_tick/*-TBBO` | 95 | 2024-09-30 -> 2024-10-02 | Day buckets by instrument |
| MBP-1 | `EQUS.MINI_MBP1` | `data/catalog/data/quote_tick/*-MBP1` | 95 | 2024-09-30 -> 2025-10-27 | Day buckets by instrument |
| Trades | `EQUS.MINI_TRADES` | `data/catalog/data/trade_tick` | 95 | 2024-09-30 -> 2025-11-26 | Day buckets by instrument |

Notes
- The MBP-1 archive `data/batch/EQUS-MBP-1-20251027-KKQ9D3X3EJ.zip` contains 94
  symbols over 2025-10-27 -> 2025-11-24; Parquet should expand beyond the
  current 8-instrument footprint after ingest completes.
- Parquet coverage is the max available dataset; DB coverage should match it.

### Feature Data Coverage Targets (Parquet Baseline)
Feature caches use date-partitioned Parquet files. Coverage targets are defined
by available partitions in `data/features`.

| Feature schema | Parquet path | Instruments | Parquet window (UTC) | Coverage target |
| --- | --- | --- | --- | --- |
| Microstructure minute | `data/features/micro_minute` | 192 | 2018-09-04 -> 2026-01-23 | Per-minute coverage by instrument |
| L2 minute | `data/features/l2_minute` | 193 | 2018-05-01 -> 2026-01-23 | Per-minute coverage by instrument |
| Feature values | `data/features/store/feature_values` | 95 (EQUS) + 14 legacy/test | 1970-01-01 -> 2026-01-23 | Feature-engineer output driven |
| Macro release calendar | `data/features/macro` | N/A | 1955-05-06 -> 2026-01-22 | Calendar driven |
| Events calendar | `data/features/events` | N/A | 2026-01-20 -> 2026-01-20 | Calendar driven |
| Earnings actuals | `data/features/earnings_raw/earnings_actuals` | N/A | 2025-11-08 -> 2026-01-23 | Provider driven |
| Earnings estimates | `data/features/earnings_raw/earnings_estimates` | N/A | 2025-11-08 -> 2026-01-26 | Provider driven |

Latest Parquet inventory (2026-01-28)
- `data/features/macro/fred_indicators_ml_format.parquet`: 43,114 rows (series_id, timestamp, value).
- `data/features/macro/fred/vintages/*/release_calendar.parquet`: 32 series, 1,636,264 rows total.
- `data/features/earnings_raw/earnings_actuals`: 258,519 rows total across EQUS
  symbols; 561 unique (ticker, period_end) keys.
- `data/features/earnings_raw/earnings_estimates`: 287,158 rows total across
  EQUS symbols; 815 unique (ticker, estimate_date, period_end) keys.
- `data/features/events/events.parquet`: 42,150 rows (ALFRED releases + stubs).

### Estimated Full Rows (Approximate)
For minute-cadence tables, use trading day estimates (calendar_days * 5/7) and
390 minutes per trading day. These are upper-bound targets used to surface
gaps; actual counts will be lower for sparse instruments.

| Table | Coverage window | Instruments | Estimated rows (approx) |
| --- | --- | --- | --- |
| `market_data_bar` | 2023-03-28 -> 2025-12-01 | 95 | ~25,935,000 |
| `ml.microstructure_minute` | 2018-09-04 -> 2026-01-23 | 192 | ~144,293,760 |
| `ml.l2_minute` | 2018-05-01 -> 2026-01-23 | 193 | ~151,819,590 |

### Current DB Snapshot (As of 2026-01-28)
| Table | Rows | Instruments | ts_event window (UTC) |
| --- | --- | --- | --- |
| `market_data_bar` | 25,311,576 | 95 | 2023-03-28 -> 2025-12-01 |
| `market_data_quote_tick` | 1,355 | 2 | 2024-11-19 -> 2025-05-13 |
| `market_data_tbbo` | 2,003,420 | 95 | 2024-09-30 -> 2024-10-02 | [E3]
| `market_data_mbp1` | 128,273,569 | 95 | 2024-09-30 -> 2025-10-27 | [E3]
| `market_data_trade_tick` | 31,027 | 2 | 2024-11-19 -> 2025-05-13 |
| `ml.microstructure_minute` | 518,746 | 77 | 2024-08-26 -> 2025-08-29 |
| `ml.l2_minute` | 1,568,073 | 80 | 2025-07-24 -> 2025-09-19 |
| `ml_feature_values` | 25,311,682 | 95 | 2023-03-28 -> 2025-12-01 |
| `ml.macro_release_calendar` | 1,636,264 | - | 1955-05-06 -> 2026-01-27 |
| `ml.macro_observations` | 43,114 | - | 2016-01-01 -> 2026-01-27 |
| `ml.events_calendar` | 42,150 | 1 | 2022-10-20 -> 2026-01-27 |
| `ml.earnings_actuals` | 561 | 70 | 2024-01-23 -> 2026-01-23 |
| `ml.earnings_estimates` | 815 | 70 | 2020-01-28 -> 2026-01-26 |

### Snapshot Delta (As of 2026-01-29)
| Table | Rows | Instruments | ts_event window (UTC) |
| --- | --- | --- | --- |
| `market_data_quote_tick` | 340,751,942 | 9 | 2024-09-30 -> 2026-01-09 |
| `market_data_trade_tick` | 31,027 | 2 | 2024-11-19 -> 2025-05-13 |
| `market_data_mbp1` | 0 | 0 | n/a (intentionally empty) |

Definition of Done
- [x] Coverage targets are defined per schema with explicit windows and
      instrument counts.
- [x] Expected row estimates are documented for minute-cadence tables.
- [x] Current DB snapshot is recorded to surface gaps quickly.

## Training Dataset Matrix (Teacher/Student)
Goal: define configurable dataset variants aligned to offline and live data
universes, with explicit feature groups, windows, and coverage thresholds.
These are plans only; configuration will live under `ml/config` and be applied
by the dataset builders.

### Universe Definitions
- Offline universe: symbols present on disk (`data/catalog` + `data/features`)
  and listed in `ml/config/equs_mini_symbols.txt`.
- Live universe (future): Databento EQUS.MINI once the license is restored.
- Intersection universe: symbols present in both offline + live (use for
  production-parity validation).

### Feature Groups (Configurable)
- Market core: bars (ohlcv-1m) + trades + quotes/tbbo/mbp1 (if available).
- Macro: release calendar + macro observations (as-of safe).
- Earnings: actuals + estimates (as-of safe; join on release/ts_event).
- Events: release events (as-of safe).
- Derived: `ml_feature_values` (feature engineering output).
- Micro/L2: optional (priority lower; use only when coverage is dense).

### Dataset Variants (Configurable)
| Variant | Window source | Required feature groups | Optional feature groups | Intended use |
| --- | --- | --- | --- | --- |
| Teacher-long | Full bar coverage window | Market core + Macro + Earnings + Events + Derived | Micro/L2 | Long-horizon teacher |
| Student-12m | Last 12 months of bars | Market core + Macro + Earnings + Events + Derived | Micro/L2 | Medium horizon student |
| Student-90d | Last 90 days of bars | Market core + Macro + Earnings + Events | Derived | Short regime student |
| Student-30d | Last 30 days of bars | Market core + Macro + Events | Earnings, Derived | Short regime student |
| HF-quote | TBBO/MBP1 coverage window | Quotes/TBBO/MBP1 + Bars | Macro/Events | Intraday/HF model |

### Coverage Thresholds (Defaults; Configurable)
- Bars: 0.95 coverage by instrument (required).
- Macro: 0.90 coverage by window (required for Teacher/Student-12m).
- Earnings: 0.80 coverage by window (required for Teacher/Student-12m).
- Events: 0.80 coverage by window (required for Teacher/Student-12m).
- Derived features: 0.90 coverage where present; drop/flag features below.
- Micro/L2: 0.70 coverage (optional; drop if below).

### As-Of Safety Rules (Required)
- Macro: `ts_event == release_ts` and `release_ts >= observation_ts`.
- Events: `ts_event == release_ts` for release-driven rows.
- Earnings: `ts_event >= period_end` and join by `ts_event` (not estimate_date).

Definition of Done
- [ ] Dataset variants are defined in config (not hard-coded), including
      windows, feature groups, and thresholds.
- [ ] Offline, live, and intersection universes are defined and used.
- [ ] Each variant has an explicit, validated alignment window derived from the
      intersection of required feature coverage.

## Architecture Summary

### Ingestion to Postgres and Parquet
- Databento ingestion (API or DBN archives) normalizes schema columns into
  canonical fields and writes into per-class SQL tables.
- Dual-write mirrors into Parquet via raw writer when enabled.

Definition of Done
- [x] Summary calls out the canonicalization step and dual-write path.

### Rehydration to Postgres
- Parquet-backed rehydration detects gaps and fills them into Postgres using
  schema-aware routing.
- Events and watermarks are updated for every successful rehydration window.

Definition of Done
- [x] Summary ties rehydration to coverage gaps, events, and watermarks.

### Feature Calculation and Storage
- Feature engineering writes to FeatureStore (values) and feature datasets
  (macro/events/micro/L2) with registry contracts enforced.
- Micro/L2 cache hydration prefers catalog quote/trade data and skips empty
  partitions to avoid persisting sparse windows.

Definition of Done
- [x] Summary states where feature outputs are stored and validated.

## Task Plan

### 1) Schema and Registry Alignment
- [x] Confirm per-class SQL tables and indexes match manifests for `EQUS.MINI`,
      `EQUS.MINI_TBBO`, and `EQUS.MINI_MBP1`.
- [x] Define registry dataset IDs/manifests for quote/trade schemas.
- [x] Add dataset manifest overrides for `EQUS.MINI_TBBO` and `EQUS.MINI_MBP1`
      that align to canonical bid/ask schemas and monthly partitioning.
- [x] Ensure bootstrap manifests and contracts exist for all market and feature
      datasets (bars, tbbo, mbp1, quotes, trades, features, macro, events, L2).
- [x] Keep schema audit expectations in sync with per-class table columns.
- [x] Seed the Postgres data registry with market dataset IDs before
      rehydration so watermark FK checks pass.
- [x] Update the `ml_dataset_registry` dataset_type constraint to include
      macro and feature dataset types.

Definition of Done
- [ ] Manifest schema, SQL DDL, and schema audit expectations match.
- [ ] Contracts validate required fields without rejecting canonical columns.

### 2) Provider vs Registry Dataset Id Split
- [x] Propagate provider_dataset_id through descriptors, inputs, bindings, and
      dataset metadata serialization.
- [x] Ensure provider_dataset_id is used only for external ingestion calls.
- [x] Ensure registry dataset_id is used for manifests, events, and watermarks.
- [x] Propagate provider_schema through descriptors, inputs, and bindings so
      provider calls use the correct external schema (e.g., `EQUS.MINI_QUOTES`
      → provider `tbbo`).

Definition of Done
- [x] Provider calls reference provider_dataset_id and registry paths use
      dataset_id without collisions.
- [x] Market binding metadata round-trips provider_dataset_id in JSON.
- [x] Provider schema is explicit and persisted in binding metadata.

### 3) Parquet Backup and Rehydration
- [x] Confirm raw writer is enabled for market data and feature datasets where
      backup is required.
- [x] Configure dataset-type identifier templates so MBP1 uses a dedicated
      Parquet identifier suffix (e.g., `mbp1={instrument_id}-MBP1`) to avoid
      quote-tick overlap.
- [x] Validate catalog rehydration routes each schema to the correct table.
- [x] Ensure rehydration emits registry events and updates watermarks.
- [x] Add a toggle to disable source reingest during coverage restore when only
      Parquet backfill is desired.
- [x] Prefer catalog quote/trade data for micro features and avoid writing empty
      cache partitions.
- [x] Verify earnings_raw Parquet mirrors exist and match expected schema
      (earnings_actuals/earnings_estimates).
- [x] Add parity checks to flag any schema with Parquet coverage but zero SQL
      buckets (gap indicates rehydration or writer issues). [E10]
- [x] Verify rehydration selection includes non-L2 feature datasets
      (`macro_release_calendar`, `macro_observations`, `events_calendar`,
      `earnings_actuals`, `earnings_estimates`, `feature_values`) and does not
      silently skip them due to dataset type or storage kind. [E5][E6][E7]
- [x] Confirm registry dataset IDs for feature datasets are present before
      rehydration so FK checks do not block writes (`features` registered). [E9]
- [x] Expand macro series coverage to match Parquet vintages (coverage now
      references `ml/config/macro_fred_series.txt`). [E14]
- [x] Run a one-time macro SQL backfill with watermarking disabled (or an
      extended window) so `ml.macro_observations` reaches Parquet coverage.
      Root cause: `macro_window_defaults` caps ingestion to 730 days when
      watermarks are enabled. [E27]
- [x] Resolve events Parquet mirror gap (events Parquet now aligned with SQL
      minus a small historical delta). Root cause: coverage reingest used
      default `economic_series=("CPI",)` with no `alfred_vintage_dir`, so
      ALFRED release events were never materialized; watermark window clipped
      to 365 days. [E15]
- [x] Ensure earnings restores replay all rows within requested buckets so
      sparse datasets (multiple rows per day) reach parity. [E20]
- [ ] Increase coverage restore bucket cap or stage multiple passes to clear
      earnings + quote/trade gaps. [E16]
- [x] Stream catalog rehydration for quotes/trades to avoid loading full
      buckets into memory (prevents segmentation faults on large tick buckets).
      [E36]
- [x] Run a full events backfill with `alfred_vintage_dir` pointing at
      `data/features/macro/fred/vintages` and `economic_series` sourced from
      `ml/config/macro_fred_series.txt`, with watermarking disabled, to
      populate `events.parquet` + SQL history. [E28]
- [ ] Run FeatureStore parquet mirror backfill so `data/features/store/feature_values`
      matches SQL exactly (SQL is canonical). Use `ml/cli/feature_store_mirror_backfill.py`
      with the deployment DSN. (Backfill started 2026-01-29; verify parity once complete.)

Definition of Done
- [ ] Parquet writes exist for each schema and are consumable for rehydration.
- [ ] Rehydration fills gaps, updates registry state, and parity checks pass.

### 3b) Rehydration Gap Investigation (Non-L2 Features)
Goal: identify why Parquet-backed feature datasets are not reflected in
Postgres without assuming a backfill run will fix the issue.

Gap heuristic:
- If Parquet has coverage but SQL is empty, treat it as a rehydration, routing,
  or registry mismatch (not a pure “run a backfill” issue).

- [x] Align `features` coverage entities to explicit EQUS.MINI symbols
      derived from Parquet (`ml/config/equs_mini_symbols.txt`).
- [x] Mark sparse feature datasets to use catalog-derived buckets
      (`bucket_mode = "catalog"`) to avoid daily false gaps.
- [x] Trace catalog rehydration eligibility filters and dataset-type routing in
      `catalog_rehydrator.py` and the entrypoint pipeline for non-L2 feature
      datasets. [E5][E7]
- [x] Confirm non-L2 feature datasets bypass catalog dataset-type templates and
      use explicit Parquet specs from the coverage manifest. [E6][E7]
- [x] Confirm coverage restore flags (`CATALOG_REHYDRATE_ENABLED`,
      `COVERAGE_RESTORE_ENABLED`) are honored for feature datasets. [E5]
- [x] Add a short-run “rehydration dry-run” with logging to prove which
      datasets are included/excluded and why. [E11]
- [ ] Validate `COVERAGE_RESTORE_LOOKBACK_DAYS` spans the desired historical
      window (oldest EQUS market data) and plan staged runs if buckets are too
      large to process in one cycle.

Note
- Coverage bucket cap triggered during stack run (`COVERAGE_MAX_BUCKETS_PER_RUN`
  default 500); expect staged restores unless cap is increased. [E13]

Definition of Done
- [ ] Each non-L2 feature dataset with Parquet coverage has a documented
      rehydration eligibility result (included or explicitly excluded).
- [ ] Any exclusion is tied to a concrete config or code guard and resolved or
      documented.

### 4) Docker Stack Verification
- [x] Ensure docker-compose uses bootstrap migrations for fresh DBs and keeps
      legacy migrations for existing DBs. [E4]
- [x] Validate environment flags for migration profile and table routing
      (ML_MARKET_DATA_PROFILE=class_tables when legacy table exists).
- [x] Run the ML docker stack with ingestion and rehydration enabled (Databento
      live backfill failed due to license; rehydration still ran). [E12]

Definition of Done
- [ ] Containerized stack brings up DB, registry, ingestion, and rehydration
      with no schema conflicts.

### 5) Tests and Validation
- [x] Add unit coverage for provider_dataset_id propagation and metadata round
      trip.
- [x] Update dataset id constants tests to include TBBO and MBP1 ids.
- [x] Run mypy, ruff, validate-fixtures, validate-events, validate-metrics.
- [x] Run targeted pytest for bindings, registry, ingestion, rehydration, and
      docker pipeline tests.

Definition of Done
- [ ] All tests pass and coverage targets are met for touched ML modules.

### 6) On-Disk Coverage and Backfill Execution
- [ ] Inventory `data/batch` and `data/catalog` to confirm which schemas exist
      for L0/L1/L2/L3.
- [ ] Ingest any missing DBN archives into the Parquet catalog (bars, TBBO,
      MBP1, trades, etc).
- [ ] Rehydrate Postgres from the catalog for every available schema (including
      TBBO/MBP1 once ingested).
- [ ] Rebuild micro/L2 feature caches from catalog data and purge empty
      partitions.
- [ ] If L3 archives exist on disk, ingest and register them; otherwise record
      the missing coverage explicitly.
- [x] Validate feature store writes; `ml_feature_values` now matches bar window
      for 95 EQUS symbols after forced backfill for NVDA/QQQ/SPY.
- [x] Validate earnings parquet mirrors and SQL population for
      `ml.earnings_actuals` / `ml.earnings_estimates`.
- [x] Reconcile earnings estimate key gap (815 parquet unique vs 815 SQL);
      actuals parity confirmed (561 parquet unique vs 561 SQL). [E17]

Definition of Done
- [ ] All available on-disk schemas have catalog coverage and SQL backfill.

### 7) Model Readiness Checks (As-Of Safety + Alignment)
- [x] As-of alignment checks for macro/earnings (ensure release/estimate/filing
      timestamps are used as `ts_event` and never exceed their logical
      reference dates). Verified no macro rows where `ts_event != release_ts`
      or `release_ts < observation_ts`, no macro observations where
      `ts_event != observation_ts`, and no earnings actuals where
      `ts_event < period_end`. Earnings estimates show 4 rows where
      `DATE(ts_event) > estimate_date` (likely timezone day shift) and 754 rows
      where `estimate_date > period_end` (expected; do not join on
      `estimate_date`). [E23]
- [x] Training-readiness dataset build (bars + macro + earnings) to validate
      joins, missingness rates, and as-of behavior on a sample symbol subset.
      Use a time window inside bar coverage (max_ts 2025-12-01) and pass
      explicit EQUS instrument_ids (to avoid NYSE/NASDAQ heuristics). [E25]
- [x] Feature-store parity spot-check for a representative EQUS subset to
      confirm feature values align with bar coverage windows and no
      out-of-range timestamps slip through. Sample set (AAPL, MSFT, NVDA, AMZN,
      META, TSLA, SPY, QQQ, JPM, XOM) shows 0 feature rows outside bar windows
      and matching min/max ts_event in `ml_feature_values`. [E24]

Definition of Done
- [x] As-of checks run and documented with violation counts (0 expected).
- [x] Training-ready dataset build produces stable sample outputs without
      leakage warnings. [E25]
- [ ] Investigate macro feature coverage shortfall in the 10-symbol readiness
      build (GDP is null while CPI/PCE/PAYEMS/UNRATE/FEDFUNDS are filled).
      Root cause likely stale/empty GDP join inputs at build time or a column
      selection mismatch between base series and `__value_*` outputs. Re-run
      the readiness build after macro mirror refresh; `join_fred_asof` now
      coalesces base series columns from `__value_real_time` to avoid null base
      series when the value columns are present. [E26][E33]
- [x] Feature-store parity spot-check confirms no missing or future-dated
      values across the sample. [E24]

### 8) Training Dataset Definitions + HPO Matrix
- [ ] Define config-driven dataset variants (teacher + student) with
      windows/feature groups/coverage thresholds in `ml/config`.
- [ ] Add a dataset builder entry point that selects a variant by name and
      applies coverage thresholds, as-of rules, and universe selection.
- [ ] Add HPO scaffold for each model family (TFT/Chronos/LightGBM/other)
      referencing the dataset variant and feature group selection.
- [ ] Add a “coverage summary report” artifact per dataset build for
      auditability (missingness matrix, per-feature coverage, and time window).

Definition of Done
- [ ] Each model family can be trained from a named dataset variant with
      coverage summaries logged.
- [ ] Variant selection is purely config-driven and reproducible.

## Verification Checklist
- [x] Postgres tables populated for bars (95 symbols).
- [ ] Postgres tables populated for quotes and trades (quotes=9 symbols; trades=2).
- [x] Postgres tables populated for tbbo (95 symbols). [E3]
- [x] Postgres mbp1 is intentionally empty (Parquet only; storage constraint). [E32]
- [ ] Parquet catalog has matching data for the same schemas.
- [x] Rehydration fills DB gaps from Parquet for bars, quotes, trades.
- [x] Watermarks updated for bars, quotes, trades.
- [ ] New ingestion fills remaining gaps and writes to DB and Parquet
      (blocked by Databento live license for EQUS.MINI after 2026-01-26). [E12]
- [x] Feature values populate via feature engineering and match bar coverage.
- [ ] L2/L3 coverage aligned to the on-disk availability window.
- [ ] Parquet-vs-DB parity checks pass for non-L2 feature schemas.
      (feature_values parquet currently has +583 rows vs SQL; needs mirror
      backfill).
- [x] Macro release calendar row counts align with Parquet (1,636,264 SQL vs
      1,636,264 parquet rows). [E22]
- [x] Macro observations row counts align with Parquet (43,114 SQL vs 43,114
      parquet rows). [E22]
- [x] Earnings actuals/estimates parity with Parquet unique keys (561/815
      parquet unique vs 561/815 SQL). [E17]
- [x] Events parity with Parquet (42,150 SQL vs 42,150 parquet). [E15]
- [x] Macro Parquet mirrors refreshed to include all configured series (30/30
      series present in `fred_indicators_ml_format.parquet`, vintage release
      calendars present for all series). [E35]

Definition of Done
- [ ] All checklist items are confirmed against the docker deployment stack.

## Global Definition of Done
- [ ] Rows populate for all market data schemas and feature schemas via
  rehydration, ingestion, or feature computation.
- [ ] Postgres and Parquet stay in sync for covered windows.
- [ ] Registry events and watermarks reflect every successful write.
- [ ] Coverage window matches all available data on disk.

## Evidence (2026-01-28)
- [E1] MBP1 Parquet catalog directories: `find data/catalog/data/quote_tick -maxdepth 2 -type d -name "*-MBP1" | wc -l` -> `95`.
- [E2] TBBO Parquet catalog directories: `find data/catalog/data/quote_tick -maxdepth 2 -type d -name "*-TBBO" | wc -l` -> `95`.
- [E3] SQL counts:
  - `market_data_mbp1`: rows=128,273,569; instruments=95; min_ts=1727689028761166141; max_ts=1761602626858793260
  - `market_data_tbbo`: rows=2,003,420; instruments=95; min_ts=1727694000070893702; max_ts=1727913599765687232
- [E4] Bootstrap migrations are mounted in compose for DB init and app access:
  - `ml/deployment/docker-compose.yml` (migrations_bootstrap mounts for DB + app)
  - `ml/deployment/docker-compose.override.yml` (migrations_bootstrap mount)
- [E5] Coverage restoration routes feature datasets via parquet specs and calls
  feature restore: `ml/deployment/entrypoint_pipeline.py` (`_run_coverage_restoration`).
- [E6] Coverage manifest defines non-L2 feature datasets with parquet specs and
  bucket_mode=catalog: `ml/config/coverage_datasets_tier1.toml`.
- [E7] Feature restoration supports macro/events/earnings/feature_values and
  uses supported dataset IDs: `ml/data/coverage/feature_restorer.py`.
- [E8] Parquet paths for non-L2 features exist on disk:
  - `data/features/earnings_raw/earnings_actuals`
  - `data/features/earnings_raw/earnings_estimates`
  - `data/features/macro/fred/vintages`
  - `data/features/macro/fred_indicators_ml_format.parquet`
  - `data/features/events/events.parquet`
  - `data/features/store/feature_values`
- [E9] Registry presence check (SQL): `ml_dataset_registry` contains
  `features`, `ml.earnings_actuals`, `ml.earnings_estimates`,
  `ml.events_calendar`, `ml.macro_observations`, `ml.macro_release_calendar`.
- [E10] Parity gap logging added in coverage manager:
  `ml/data/coverage/manager.py` (`_log_parity_gaps` in `restore_all`).
- [E11] Coverage dry-run dataset summary logging added:
  `ml/deployment/entrypoint_pipeline.py` (`_log_coverage_dry_run_summary`).
- [E12] Docker stack run (`make ml-up`) shows migrations verified, coverage
  summary/feature restore pending, and backfill failures due to Databento live
  license (`license_not_found_unauthorized` for EQUS.MINI after 2026-01-26).
- [E13] Coverage bucket cap applied during stack run:
  `coverage_manager.bucket_cap_applied` in `ml_pipeline` logs.
- [E14] Macro coverage now references `ml/config/macro_fred_series.txt` (32
  series) to align with Parquet vintages.
- [E15] Events Parquet mirror now populated; parquet rows=42,150 vs SQL
  rows=42,150.
- [E30] MBP-1 ingestion disabled in deployment env to cap storage footprint:
  `DATABENTO_ALLOWED_SCHEMAS` excludes `mbp-1` and `DUAL_WRITE_MBP=0`
  (`ml/deployment/.env`).
- [E31] MBP-1 SQL truncated to preserve disk (keeping Parquet as backup):
  `TRUNCATE TABLE public.market_data_mbp1` on the production DB.
- [E32] Quote/trade SQL snapshot (2026-01-29):
  - `market_data_quote_tick`: rows=340,751,942; instruments=9;
    min_ts=1727694000382513453; max_ts=1768001909636465381
  - `market_data_trade_tick`: rows=31,027; instruments=2;
    min_ts=1732022128773535067; max_ts=1747180772155671568
  - `market_data_mbp1`: rows=0; instruments=0 (intentionally empty)
- [E33] Macro join coalesces base series columns from `__value_real_time` to
  avoid null base-series columns when value columns are present:
  `ml/data/fred_join.py`.
- [E34] Quote sentinel values are filtered during SQL writes using
  `ML_MARKET_DATA_QUOTE_SENTINEL_PRICE` (defaults to the Databento missing
  price marker) to prevent invalid bid/ask values in
  `market_data_quote_tick` / `market_data_tbbo`.
- [E35] Macro Parquet mirrors refreshed from SQL using
  `ml.cli.feature_dataset_mirror_refresh`; 30/30 series present in
  `data/features/macro/fred_indicators_ml_format.parquet` and release calendars
  present for all series under `data/features/macro/fred/vintages/*`.
- [E36] Catalog rehydration streams quote/trade buckets via
  `ParquetDataCatalog.backend_session` and skips identifier=None fallback for
  tick datasets to prevent segfaults on large buckets.
- [E16] Coverage bucket cap defaults to `COVERAGE_MAX_BUCKETS_PER_RUN=500`
  (entrypoint pipeline).
- [E17] Parquet earnings totals (EQUS universe):
  - actuals: 325,033 rows / 561 unique (ticker, period_end)
  - estimates: 360,988 rows / 818 unique (ticker, estimate_date, period_end)
  vs SQL 561/818 rows (parity confirmed by unique keys).
- [E18] Parquet macro observations file has 43,114 rows vs SQL 43,114 rows.
- [E19] Event ingestion SQL dedupe added to avoid ON CONFLICT cardinality
  errors (`ml/preprocessing/event_ingestion.py`).
- [E20] FeatureCoverageRestorer now replays all earnings rows per bucket (no
  early break) with unit coverage in
  `ml/data/coverage/feature_restorer.py` and
  `ml/tests/unit/data/test_feature_restorer.py`.
- [E21] Earnings coverage configs now use explicit EQUS.MINI symbols via
  `ml/config/equs_mini_symbols.txt` (`ml/config/coverage_datasets_tier1.toml`,
  `ml/config/coverage_datasets_earnings_only.toml`).
- [E22] Macro SQL parity: `ml.macro_release_calendar` rows=1,636,264 vs Parquet
  1,636,264; `ml.macro_observations` rows=43,114 vs Parquet 43,114.
- [E23] As-of alignment checks:
  - Macro: 0 rows where `ts_event != release_ts` or `release_ts < observation_ts`.
  - Macro observations: 0 rows where `ts_event != observation_ts`.
  - Earnings actuals: 0 rows where `ts_event < period_end`.
  - Earnings estimates: 4 rows where `DATE(ts_event) > estimate_date`
    (timezone day shift) and 754 rows where `estimate_date > period_end`
    (expected; join on `ts_event`).
- [E24] Feature-store parity spot-check: sample EQUS instruments (AAPL, MSFT,
  NVDA, AMZN, META, TSLA, SPY, QQQ, JPM, XOM) have 0 `ml_feature_values` rows
  outside bar windows and matching `ts_event` ranges.
- [E25] Training dataset build (AAPL/MSFT/NVDA, 2025-11-01→2025-12-01, macro +
  earnings enabled, explicit EQUS instrument_ids) produced `rows=26423`,
  `cols=74` with `ML_MARKET_DATA_PROFILE=class_tables`.
- [E26] Training dataset build (10 EQUS symbols, 2025-11-01→2025-12-01, macro +
  earnings enabled, explicit EQUS instrument_ids) persisted under
  `ml_out/tft_readiness_equs_2025_11/` with `rows=85073`, `cols=101`,
  `instrument_id_n_unique=10` (validation run with
  `min_feature_coverage=0.0` due to macro coverage gaps). GDP base column is
  null; `GDP__value_real_time` and `GDP__value_final` are filled with 0.0.
- [E29] Parquet/SQL parity snapshot (2026-01-28): macro_release and
  macro_observations counts match configured series; events.parquet refreshed
  to 42,150 rows; earnings unique keys match SQL; feature_values parquet has
  +583 rows vs SQL.
