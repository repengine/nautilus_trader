# Catalog Rehydration From Parquet Plan

## Objectives

- Restore canonical market data coverage in Postgres by replaying existing Parquet catalogs before any external ingestion.
- Avoid redundant Databento downloads by aligning SQL and Parquet coverage, then running orchestrator ingest for genuine gaps.
- Preserve AGENTS.md requirements: protocol-first design, config-driven tunables, structured logging/metrics, progressive fallbacks, strict typing/testing.

## Existing Building Blocks

- `ParquetDataCatalog` (`nautilus_trader.persistence.catalog.parquet`) exposes `get_intervals`, `bars`, `quote_ticks`, `trade_ticks`.
- SQL coverage/writes already live in `ml/stores/providers.py` (`SqlCoverageProvider`, `SqlMarketDataWriter`).
- Scheduler orchestrator entrypoint (`ml/deployment/entrypoint_pipeline.py`) boots `DataScheduler` and runs ingestion modes.
- Metrics/logging primitives come from `ml.common.metrics_bootstrap` and structured logging conventions.

## Planned Components

1. **Configuration**
   - New frozen dataclass (e.g., `CatalogRehydrationConfig`) under `ml/config/`.
   - Fields: `enabled`, `lookback_days`, `batch_size`, `max_workers` (optional), `rescan_on_schedule`.
   - Environment wiring via `entrypoint_pipeline` (`CATALOG_REHYDRATE_*`).

2. **Rehydration Protocol & Service**
   - Protocol definition in `ml/data/rehydration/protocols.py` for structural typing.
   - Implementation `ParquetCatalogRehydrator` in `ml/data/rehydration/catalog_rehydrator.py`.
   - Responsibilities:
     - Derive symbol/instrument list from scheduler config/universe.
     - Use catalog + `CatalogCoverageProvider` to compute available day buckets.
     - Compare against `SqlCoverageProvider`.
     - Stream parquet slices → pandas frames → write via `SqlMarketDataWriter` (idempotent).
     - Emit metrics (`parquet_rehydrate_rows_total`, `parquet_rehydrate_failures_total`, duration histogram).
     - Structured logging with context (`symbol`, `dataset_id`, `bucket`).
     - Progressive fallback: skip instrument on failure, record telemetry, continue.

3. **Pipeline Integration**
   - Extend `PipelineRunner` in `entrypoint_pipeline.py`:
     - Instantiate rehydrator after catalog initialization when config enabled.
     - Call `rehydrator.rehydrate_missing_data(...)` before `scheduler.run_daily_update()` in each mode.
     - Bubble summary to health endpoint (store last run status/errors).
   - Pass unified config (symbols, dataset/schema from `SchedulerConfig.databento`).
   - Ensure orchestrator flags (`USE_ORCHESTRATOR`, `DUAL_WRITE`) remain functional; rehydration precedes orchestrator backfill.

4. **Testing Strategy**
   - Unit tests: `ml/tests/unit/data/test_catalog_rehydrator.py`
     - Use temp Parquet catalog + SQLite DB to verify selective writes and error telemetry.
   - Deployment tests: extend `ml/tests/unit/deployment/test_entrypoint_pipeline.py` to assert rehydration invocation when env enabled/disabled.
   - Maintain ≥90% coverage for new module.
   - Run `poetry run mypy ml --strict`, `poetry ruff check ml`, targeted pytest, orchestrator ingestion suite if impacted.

5. **Documentation & Ops**
   - Reference new env vars and behaviour in `ml/deployment/README.md` and `ml/docs/tools/ORCHESTRATION_RUNBOOK.md`.
   - Note recovery flow (rehydrate → orchestrator) and failure telemetry.

## Sequencing (TDD-Oriented)

1. Author failing tests for rehydration service and pipeline integration.
2. Implement config dataclass + service/protocol with minimal functionality to satisfy tests.
3. Integrate into entrypoint, update docs, and iterate until tests/mypy/ruff pass.
4. Validate end-to-end by running orchestrator-related test suites.
