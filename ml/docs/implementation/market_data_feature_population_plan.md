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

### Feature Data Schemas
- FeatureStore values -> `ml.feature_values` -> `ml_feature_values`
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
| MBP-1 | `EQUS.MINI_MBP1` | `data/catalog/data/quote_tick/*-MBP1` | 8 | 2024-09-30 -> 2025-10-27 | Day buckets by instrument |
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
| Feature values | `data/features/store/feature_values` | 16 | 1970-01-01 -> 2026-01-23 | Feature-engineer output driven |
| Macro release calendar | `data/features/macro` | N/A | 1955-05-06 -> 2026-01-22 | Calendar driven |
| Events calendar | `data/features/events` | N/A | 2026-01-20 -> 2026-01-20 | Calendar driven |
| Earnings actuals | `data/features/earnings_raw/earnings_actuals` | N/A | 2025-11-08 -> 2026-01-23 | Provider driven |
| Earnings estimates | `data/features/earnings_raw/earnings_estimates` | N/A | 2025-11-08 -> 2026-01-23 | Provider driven |

### Estimated Full Rows (Approximate)
For minute-cadence tables, use trading day estimates (calendar_days * 5/7) and
390 minutes per trading day. These are upper-bound targets used to surface
gaps; actual counts will be lower for sparse instruments.

| Table | Coverage window | Instruments | Estimated rows (approx) |
| --- | --- | --- | --- |
| `market_data_bar` | 2023-03-28 -> 2025-12-01 | 95 | ~25,935,000 |
| `ml.microstructure_minute` | 2018-09-04 -> 2026-01-23 | 192 | ~144,293,760 |
| `ml.l2_minute` | 2018-05-01 -> 2026-01-23 | 193 | ~151,819,590 |

### Current DB Snapshot (As of this run)
| Table | Rows | Instruments | ts_event window (UTC) |
| --- | --- | --- | --- |
| `market_data_bar` | 1,696,251 | 5 | 2023-03-28 -> 2025-12-01 |
| `market_data_quote_tick` | 1,355 | 2 | 2024-11-19 -> 2025-05-13 |
| `market_data_tbbo` | 2,003,420 | 95 | 2024-09-30 -> 2024-10-02 |
| `market_data_mbp1` | 6,264,465 | 9 | 2024-09-30 -> 2025-10-27 |
| `market_data_trade_tick` | 31,027 | 2 | 2024-11-19 -> 2025-05-13 |
| `ml.microstructure_minute` | 518,746 | 77 | 2024-08-26 -> 2025-08-29 |
| `ml.l2_minute` | 1,568,073 | 80 | 2025-07-24 -> 2025-09-19 |
| `ml_feature_values` | 1,755 | 2 | 2025-11-03 -> 2025-11-04 |
| `ml.macro_release_calendar` | 2,628 | - | 2022-10-13 -> 2026-01-22 |
| `ml.macro_observations` | 200 | - | 2022-11-01 -> 2025-12-01 |
| `ml.events_calendar` | 77 | 1 | 2022-10-20 -> 2026-01-20 |
| `ml.earnings_actuals` | 560 | - | 2024-01-23 -> 2026-01-23 |
| `ml.earnings_estimates` | 798 | - | 2022-10-27 -> 2026-01-23 |

Definition of Done
- [ ] Coverage targets are defined per schema with explicit windows and
      instrument counts.
- [ ] Expected row estimates are documented for minute-cadence tables.
- [ ] Current DB snapshot is recorded to surface gaps quickly.

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

Definition of Done
- [x] Provider calls reference provider_dataset_id and registry paths use
      dataset_id without collisions.
- [x] Market binding metadata round-trips provider_dataset_id in JSON.

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
- [ ] Add parity checks to flag any schema with Parquet coverage but zero or
      minimal DB rows (gap indicates rehydration or writer issues).
- [ ] Verify rehydration selection includes non-L2 feature datasets
      (`macro_release_calendar`, `macro_observations`, `events_calendar`,
      `earnings_raw`, `feature_values`) and does not silently skip them due to
      dataset type or storage kind.
- [ ] Confirm registry dataset IDs for feature datasets are present before
      rehydration so FK checks do not block writes.

Definition of Done
- [ ] Parquet writes exist for each schema and are consumable for rehydration.
- [ ] Rehydration fills gaps, updates registry state, and parity checks pass.

### 3b) Rehydration Gap Investigation (Non-L2 Features)
Goal: identify why Parquet-backed feature datasets are not reflected in
Postgres without assuming a backfill run will fix the issue.

- [x] Align `ml.feature_values` coverage entities to explicit EQUS.MINI symbols
      derived from Parquet (`ml/config/equs_mini_symbols.txt`).
- [x] Mark sparse feature datasets to use catalog-derived buckets
      (`bucket_mode = "catalog"`) to avoid daily false gaps.
- [ ] Trace catalog rehydration eligibility filters and dataset-type routing in
      `catalog_rehydrator.py` and the entrypoint pipeline for non-L2 feature
      datasets.
- [ ] Validate dataset-type templates (`CATALOG_REHYDRATE_DATASET_TYPE_TEMPLATES`)
      and registry dataset IDs match Parquet identifiers for macro/events/
      earnings/feature_values.
- [ ] Confirm coverage restore flags (`CATALOG_REHYDRATE_ENABLED`,
      `COVERAGE_RESTORE_ENABLED`) are honored for feature datasets.
- [ ] Add a short-run “rehydration dry-run” with logging to prove which
      datasets are included/excluded and why.
- [ ] Validate `COVERAGE_RESTORE_LOOKBACK_DAYS` spans the desired historical
      window (oldest EQUS market data) and plan staged runs if buckets are too
      large to process in one cycle.

Definition of Done
- [ ] Each non-L2 feature dataset with Parquet coverage has a documented
      rehydration eligibility result (included or explicitly excluded).
- [ ] Any exclusion is tied to a concrete config or code guard and resolved or
      documented.

### 4) Docker Stack Verification
- [ ] Ensure docker-compose uses bootstrap migrations for fresh DBs and keeps
      legacy migrations for existing DBs.
- [x] Validate environment flags for migration profile and table routing
      (ML_MARKET_DATA_PROFILE=class_tables when legacy table exists).
- [ ] Run the ML docker stack with ingestion and rehydration enabled.

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
- [ ] Investigate feature store writes if `ml_feature_values` remains sparse
      after rehydration: validate FeatureStore config, storage kind, and write
      path wiring.
- [ ] Investigate earnings parquet read failures or schema mismatches if
      `ml.earnings_actuals` / `ml.earnings_estimates` stay sparse.

Definition of Done
- [ ] All available on-disk schemas have catalog coverage and SQL backfill.

## Verification Checklist
- [x] Postgres tables populated for bars, quotes, trades.
- [ ] Postgres tables populated for tbbo, mbp1.
- [ ] Parquet catalog has matching data for the same schemas.
- [x] Rehydration fills DB gaps from Parquet for bars, quotes, trades.
- [x] Watermarks updated for bars, quotes, trades.
- [ ] New ingestion fills remaining gaps and writes to DB and Parquet.
- [ ] Feature values and feature datasets populate via feature engineering.
- [ ] L2/L3 coverage aligned to the on-disk availability window.
- [ ] Parquet-vs-DB parity checks pass for non-L2 feature schemas.

Definition of Done
- [ ] All checklist items are confirmed against the docker deployment stack.

## Global Definition of Done
- [ ] Rows populate for all market data schemas and feature schemas via
  rehydration, ingestion, or feature computation.
- [ ] Postgres and Parquet stay in sync for covered windows.
- [ ] Registry events and watermarks reflect every successful write.
- [ ] Coverage window matches all available data on disk.
