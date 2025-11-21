# Inference Parity Checklist (Training ⇄ Live)

This checklist enumerates the non‑negotiable requirements to guarantee training/inference parity for the ML Signal Actor and related pipelines, plus how to verify each item in code and ops.

Status: Draft (operationally recommended defaults implemented; verification hooks noted below)

## A. Bar/Input Data Parity

- [ ] BarType string identical (e.g., `SPY.XNAS-1-MINUTE-LAST-EXTERNAL`)
  - Verify: Actor config `bar_type` == model/feature manifest metadata (record during training)
  - Where: Actor startup guard compares strings; fail fast on mismatch
- [ ] Timestamp-on-close identical for bars
  - Verify: Training records `timestamp_on_close=true` in FeatureManifest metadata; actor uses `DatabentoDataClientConfig.bars_timestamp_on_close`
  - Where: Compare manifest metadata vs actor config at startup
- [ ] Venue/Instrument mapping identical (use_exchange_as_venue)
  - Verify: Training metadata captures mapping; actor config `use_exchange_as_venue` matches
  - Where: Actor startup guard
- [ ] Dataset/schema mapping identical (e.g., EQUS.MINI → OHLCV_1M)
  - Verify: Persist training dataset/schema in model metadata (or registry manifest link); compare at startup

## B. Feature Pipeline Parity

- [ ] Single FeatureEngineer code path (batch + online)
  - Verify: FeatureManifest contains `schema_hash` + `pipeline_signature`; actor loads same feature set
  - Where: Registry: `FeatureManifest.schema_hash/pipeline_signature`; Actor validates against ModelManifest
- [ ] Min warm-up bars satisfied (stateful features)
  - Verify: `constraints.min_bars_warmup` in FeatureManifest; actor buffers until satisfied
  - Where: Actor warm-up gate before inference
- [ ] Dtype/precision parity (float32 vs float64)
  - Verify: FeatureManifest `feature_dtypes` and model input dtypes; enforce consistent cast in actor
  - Where: Actor preprocessing step; optional assert
- [ ] Missing data policy parity (fills/drops)
  - Verify: Persist fill/drop rules in FeatureManifest metadata; actor applies same rules
  - Where: FeatureEngineer configuration parity

## C. Data Requirements Parity

- [ ] Live feed meets model `data_requirements` (L1_ONLY, L1_L2, L1_L2_L3)
  - Verify: ModelManifest.data_requirements ≤ live capability; refuse incompatible models
  - Where: Actor startup guard

## D. Preprocessing/Calibration Parity

- [ ] Preprocessors (scalers/calibration) persisted with model
  - Verify: ModelManifest metadata includes scaler params; actor loads and applies identical transforms
  - Where: Model loader in actor; validation that params exist and match expected shapes

## E. Timestamp Policy

- [ ] UNIX nanoseconds throughout; UTC day buckets for coverage/gaps
  - Verify: Writer/ingestor normalize to ns; registries store ns; coverage buckets use UTC
  - Where: `SqlMarketDataWriter`, `DatabentoIngestor`, orchestrator/gap planner

## F. Canonical Store Boundaries

- [ ] Canonical raw store = Postgres `market_data` (004_market_data.sql)
  - Verify: Live/Backfill writes go through `SqlMarketDataWriter`; registry events/watermarks emitted post‑write
  - Where: Ingest Backfill CLI/orchestrator wiring; deployment config
- [ ] Parquet used for training/offline reads/coverage planning only (not authoritative)
  - Verify: Defaults `COVERAGE_MODE=sql`, `WRITE_MODE=sql`; catalog used only for planning or offline client
  - Where: CLI/env defaults and docs; no dual‑write to both DB and Parquet for same dataset

## G. Mapping Semantics to Canonical Schema

- [ ] Writer mapping for Bars/Quotes/Trades is stable
  - Verify: `SqlMarketDataWriter` maps DF columns → canonical columns; idempotent on (instrument_id, ts_event)
  - Where: `ml/stores/coverage_sql.py` implementation and tests

## H. Warm‑Up & Parity Smoke‑Check (optional but recommended)

- [ ] Actor warm‑up ≥ min_bars_warmup before emitting predictions/signals
  - Verify: Actor counters and state gate
- [ ] Parity smoke‑check on startup (small window)
  - Verify: Compute features online for last N bars and re‑compute offline via same FeatureEngineer; compare within tolerance; log metrics
  - Where: Optional actor hook; metric `feature_parity_drift`

## I. Observability

- [ ] Metrics for parity and ingestion
  - Verify: Add `feature_parity_checks_total`, `feature_parity_drift`, ingestion batch metrics; alerts on drift
  - Where: `ml.common.metrics_bootstrap`, actor instrumentation, ingest metrics

---

## Verification Plan (Where/How)

1) Startup Guards (Signal Actor)

- Load ModelManifest + FeatureManifest from registry.
- Verify: bar_type, timestamp_on_close, use_exchange_as_venue, data_requirements, schema_hash, pipeline_signature, min_bars_warmup, dtype list length.
- Fail fast on mismatch with actionable error.

2) Orchestrator/Writer Contracts

- Ensure orchestrator uses `SqlCoverageProvider` (default) and `SqlMarketDataWriter` → Postgres; emits `Stage.DATA_INGESTED` events and updates watermarks.
- Gap planning with `CatalogCoverageProvider` allowed; writes still go to DB.

3) Timestamp Normalization

- Unit tests confirm `DatabentoIngestor` produces ns; writer preserves ns; coverage buckets are UTC days.

4) Mapping Tests

- Writer contract tests for Bars (OHLCV), Quotes (bid/ask sizes), Trades (last/trade_count/vwap); idempotency on PK.

5) Warm‑Up Gate

- Actor counts bars and blocks predictions until warm‑up satisfied; include a unit test.

6) Optional Parity Smoke‑Check

- Implement feature recompute on N recent bars and compare; emit `feature_parity_drift` metric.

7) Config Defaults and Flags

- Ensure `COVERAGE_MODE=sql`, `WRITE_MODE=sql` by default; document flag meanings and risks.

---

## Pointers (Code/Docs)

- Stores: `ml/docs/context/context_stores.md` (Ingestion Strategy; Writer mapping)
- Registry: `ml/docs/context/context_registry.md` (events/watermarks; manifests)
- Orchestrator: `ml/data/ingest/orchestrator.py`
- Ingestors: `ml/data/ingest/resume.py`
- Coverage providers: `ml/stores/coverage_sql.py`, `ml/stores/coverage_catalog.py`
- Writer: `ml/stores/coverage_sql.py` (SqlMarketDataWriter)
- CLI: `ml/cli/ingest_backfill.py`
- Parquet: `nautilus_trader.persistence.catalog.parquet.ParquetDataCatalog`
