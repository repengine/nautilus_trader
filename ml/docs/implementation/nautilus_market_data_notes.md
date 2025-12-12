## Nautilus market data persistence vs ML layer (snapshot)

Purpose: capture current understanding of how vanilla Nautilus handles market data persistence and how the ML layer interacts with it, so we can revisit without re-reading the codebase.

### What Nautilus provides
- Parquet catalog: `nautilus_trader/persistence/catalog/parquet.py` stores Bars/QuoteTicks/TradeTicks/OrderBookDepth10/OrderBookDelta in per-class, per-identifier directories, filenames are `<start_ns>-<end_ns>.parquet`, and interval overlap is prevented.
- Serialization: Arrow schemas are fixed per data class (`nautilus_trader/serialization/arrow/schema.py`), and `ArrowSerializer` enforces those types across writes/reads.
- Query path: Rust-backed fast queries for built-ins when `fs_protocol=file`; `get_intervals` / `get_missing_intervals_for_request` expose coverage and gaps.
- Streaming ingest: `StreamingFeatherWriter` supports live rotation by size/time and per-instrument files; these can be compacted later into the Parquet catalog.
- Identifier rules: identifiers are URIsafe instrument IDs (bars use `bar_type`), resolved via helpers (`urisafe_identifier`, `class_to_filename`); monotonic `ts_init` is required.

### How the ML layer currently uses it
- Catalog reuse: ML wraps `ParquetDataCatalog` for reads/writes; catalog rehydration (`ml/data/rehydration/catalog_rehydrator.py`) replays Parquet into SQL `market_data`.
- Coverage: bucket-based SQL vs catalog comparison (`ml/data/coverage/manager.py`) decides between restore-from-catalog vs re-ingest.
- Writers: `SqlMarketDataWriter` mirrors rows into `market_data`; `ParquetCatalogRawWriter` can fan out to the catalog when ingestion is configured for dual-write.
- Schema mapping: centralized registry (`ml/schema.py`) maps schemas to DatasetType + dataclass + default identifier template (bars/ohlcv → Bar + bar-type identifier; tbbo/quotes/trades/mbp → QuoteTick/TradeTick + raw instrument identifier). Unknown schemas now raise.
- Identifier template: defaults live in the registry; schema/dataset overrides are validated and used by coverage/rehydration (`ml/stores/providers.resolve_catalog_identifier`).

### Gaps / risks if we “just use Nautilus”
- Coverage scope: The pipeline only inspects schemas configured in the scheduler/manifest. If TBBO/Trades/MBP aren’t listed, they will never be restored from catalog and will always be re-ingested.
- Catalog alignment: Data written outside the Parquet catalog layout (e.g., ad-hoc Parquet from DBN) won’t be discoverable by Nautilus interval logic; must ingest via `ParquetDataCatalog.write_data` (or the raw writer) with proper identifiers.
- Interval constraints: Nautilus enforces non-overlapping files; bulk backfills must sort by `ts_init` and write disjoint ranges or writes will be skipped.
- Dual-write expectations: ML coverage/rehydration assumes SQL + catalog mirrors. If we rely solely on Nautilus catalog without SQL mirrors, coverage will think the DB is empty unless rehydration runs every time.

### Can we replace `ml/data` with Nautilus primitives?
- Technical indicators already use Nautilus; for market data IO, most of the heavy lifting (catalog, serialization, interval/gap detection) is in Nautilus and could be reused more directly.
- ML-specific pieces (coverage manager, feature coverage/restorer, dual-write orchestration, dataset manifests) add scheduling, DB mirroring, and telemetry that Nautilus core doesn’t have.
- A full swap would still need: schema-to-dataset mapping for all ML schemas (bars/tbbo/trades/mbp), identifier templates per schema, and maintaining SQL mirrors for coverage/feature builders that expect DB access.
- Pragmatic approach: widen ML’s use of Nautilus catalog APIs (and maybe stream writers) while keeping ML’s coverage/orchestration layer; avoid duplicating serialization/interval logic.

### Follow-ups
- Ensure scheduler/coverage manifests include TBBO/Trades/MBP so they are restored from catalog; validate schema tokens at load time via `ml.schema`.
- Standardize ingestion of DBN → domain objects → `ParquetDataCatalog` so catalog intervals stay valid.
- If pushing more logic to Nautilus: audit any ML-specific transformations/watermarks to ensure they still run (dataset events, metrics, coverage gating).

### Investigation vectors (to finalize an implementation plan)
- DBN → catalog ingest path: define the conversion path from DBN payloads to domain objects and into `ParquetDataCatalog` with sorted, disjoint intervals (chunking/overlap handling).
- Coverage/rehydration scope: list which schemas/instruments the scheduler should classify and restore; update manifests/configs (lookback vs exhaustive, bucket limits) accordingly.
- Dual-write/SQL mirroring: decide which datasets require SQL mirrors (e.g., `market_data`) and how to keep SQL + catalog in sync for coverage and feature builders.
- Config surface alignment: ensure env/config toggles (identifier templates per schema, catalog path, lookback) are aligned across Nautilus and ML layers; add missing toggles if needed.
- Testing/validation plan: outline unit/integration/perf checks (schema mapping, writer overlap tests, restore/coverage integration, large backfill sanity) to validate the end-to-end flow.

### Latest investigation findings (with citations)
- Schema → dataclass/dataset mapping is centralized and explicit (`ml/schema.py`), covering bars/ohlcv, tbbo/bbo/quotes, trades, and mbp-1/mbp-10 with defaults for identifier templates. Unknown schemas now raise to surface misconfigurations.
- Catalog identifier resolution falls back to registry defaults and validated overrides in coverage/rehydration (`ml/stores/providers.py:540-604`, `ml/data/rehydration/catalog_rehydrator.py:64-122`); env parsing validates templates (`ml/deployment/entrypoint_pipeline.py:650-685`).
- Coverage restoration only inspects schemas listed in scheduler inputs/manifests. Dataset coverage configs carry `dataset_id` + `schema` (`ml/data/coverage/manager.py:47-109`), and the entrypoint builds coverage entries from scheduler inputs (`ml/deployment/entrypoint_pipeline.py:580-624`). If TBBO/Trades/MBP aren’t configured, they won’t be restored from catalog and will remain re-ingest targets.
- Parquet catalog raw writer supports bars/quotes/trades conversion; depth/MBP still needs explicit mirroring in scheduler dual-write paths (`ml/stores/io_raw.py:108-220`, `ml/data/scheduler.py:1072-1135`).
- Dual-write orchestration mirrors only bars/trades today. The Databento domain loader toggles `include_trades` and `bars_timestamp_on_close` heuristics; TBBO/MBP aren’t mirrored to catalog in the dual-write path (`ml/data/scheduler.py:1040-1144`, `ml/data/common/orchestrator_collection.py:200-280`).
- DBN archive CLI defaults to SQL (and optional DataStore) writes; catalog mirroring requires the raw writer and is not yet default (`ml/cli/ingest_dbn_archive.py:98-131`).
- Historical migration to catalog exists for tier1 shards but only migrates bars (`ml/data/migration/tier1_to_catalog.py:386-520`); quotes/trades/depth are not covered.

### Decisions/constraints to carry forward
- MBP should align to quote/depth semantics (QuoteTick/OrderBookDepth) per Nautilus adapter behaviour; extend ML schema→type mapping and writers accordingly.
- Dual-write goal is SQL (authoritative) + Parquet catalog (backup) for all schemas; current CLI dual-write for DBN archives mirrors only to SQL (DataStore) and not the catalog.
- Tier1 parquet shards are being deprecated; catalog must gain TBBO/Trades/MBP support (identifier templates, writer conversions, coverage/rehydration mapping) before tier1 removal.
- Identifier templates: defaults are bar-specific (`{instrument_id}-1-MINUTE-LAST-EXTERNAL` via `CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE` in env/docker compose and entrypoint), so we need per-schema templates for TBBO/Trades/MBP to make catalog coverage/rehydration work (`ml/deployment/entrypoint_pipeline.py:593-612`, `ml/data/rehydration/catalog_rehydrator.py:60-101`, `ml/stores/providers.py:540-577`, docker-compose overrides).
- DBN→catalog path: DBN archive CLI writes only to SQL/optional DataStore (not catalog), and tier1→catalog migration only handles bars; catalog population for TBBO/Trades/MBP from DBN archives remains missing (`ml/cli/ingest_dbn_archive.py:98-131`, `ml/data/migration/tier1_to_catalog.py:386-520`).
- Identifier templates: plan to add a schema→template resolver with sensible defaults (bars → `{instrument_id}-1-MINUTE-LAST-EXTERNAL`; TBBO/Trades/MBP → raw `instrument_id` per Nautilus conventions) plus optional overrides (env map or manifest). Validate templates on startup and align docs/runbooks with the per-schema defaults.

### New implementation updates
- Per-schema and per-dataset identifier templates are now part of rehydration config, with URI-safe resolution and defaults of bars → `{instrument_id}-1-MINUTE-LAST-EXTERNAL`, TBBO/Trades/MBP → `{instrument_id}` (`ml/data/rehydration/catalog_rehydrator.py:43`, `ml/data/rehydration/catalog_rehydrator.py:69`, `ml/data/rehydration/catalog_rehydrator.py:104`). Entrypoint parses `CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE_MAP` / `CATALOG_REHYDRATE_DATASET_TYPE_TEMPLATES` and passes the maps into rehydration + coverage providers (`ml/deployment/entrypoint_pipeline.py:394`, `ml/deployment/entrypoint_pipeline.py:656`, `ml/deployment/entrypoint_pipeline.py:733`).
- Catalog coverage now resolves identifiers with the same schema/dataset template maps and URI-safe handling, and schema→dataclass mapping recognizes MBP as QuoteTick (`ml/stores/providers.py:49`, `ml/stores/providers.py:60`, `ml/stores/providers.py:92`, `ml/stores/providers.py:564`).
- Raw catalog writer converts TBBO/MBP data to QuoteTick using best-level bid/ask fallbacks (supports `bid_px_0/bid_px_1` and `bid_sz_0` style columns) so MBP mirrors can be written to the Parquet catalog (`ml/stores/io_raw.py:134`, `ml/stores/io_raw.py:206`).
- DBN archive CLI can now mirror to the Parquet catalog via `--catalog-path`, using a fanout writer that includes the new raw catalog writer alongside SQL/DataStore targets (`ml/cli/ingest_dbn_archive.py:97`, `ml/cli/ingest_dbn_archive.py:118`, `ml/stores/writers.py:281`).
- Schema registry implemented: `ml/schema.py` now centralizes schema → DatasetType/dataclass/default identifier template mappings with validation, identifier resolution falls back to the registry defaults, and discovery/orchestrator helpers delegate to it to avoid drift (`ml/orchestration/discovery_service.py`, `ml/stores/providers.py`, `ml/data/rehydration/catalog_rehydrator.py`). A unit test locks the mapping (`ml/tests/unit/test_schema_registry.py`).
- Dual-write controls: per-dataset-type toggles (bars/trades/tbbo/mbp, default-on) gate catalog mirroring in scheduler/orchestrator paths, and identifier templates for mirrors reuse the catalog rehydration dataset-type map (`DUAL_WRITE_*`, `CATALOG_REHYDRATE_DATASET_TYPE_TEMPLATES`) (`ml/data/scheduler.py`, `ml/data/common/orchestrator_collection.py`, `ml/deployment/entrypoint_pipeline.py`, `ml/stores/io_raw.py`).
- Schema validation: coverage/manifest parsing and ingest adapters now validate schema tokens through `ml.schema` to raise early on unknown schemas and normalize aliases consistently (`ml/config/dataset_coverage.py`, `ml/deployment/entrypoint_pipeline.py`, `ml/orchestration/pipeline_orchestrator.py`, `ml/data/ingest/databento_adapter.py`, `ml/data/ingest/service.py`).
### Identifier template options and best practices (latest notes)
- ML resolves catalog identifiers via `resolve_catalog_identifier`, which now accepts per-schema and per-dataset templates plus URI-safe normalization (`ml/stores/providers.py:564`). Rehydration passes the configured maps from env defaults (`ml/data/rehydration/catalog_rehydrator.py:69`, `ml/deployment/entrypoint_pipeline.py:656`).
- Runbook defaults now document schema/dataset template overrides alongside the bar-style env var (`ml/deployment/README.md:153`).
- Nautilus keeps identifiers URI-safe with `urisafe_identifier`, stripping slashes for any `InstrumentId`/`BarType` (`nautilus_trader/persistence/funcs.py:70-75`) and places Parquet under `kind/identifier/<start>-<end>.parquet` (`nautilus_trader/persistence/catalog/parquet.py:65-69`, `nautilus_trader/persistence/catalog/parquet.py:230-260`). Best practice is to reuse this helper whenever we build per-schema templates.
- Defaults now match Nautilus conventions: Bars use bar-type identifiers; TBBO/Trades/MBP use raw `instrument_id`, with schema-level overrides available via env (`ml/data/rehydration/catalog_rehydrator.py:43`, `ml/deployment/entrypoint_pipeline.py:656`, `ml/stores/providers.py:66`).

### Remaining work after this pass
- Extend live/dual-write scheduler path to mirror TBBO/Trades/MBP into the catalog (today the orchestrator dual-write toggles only bars/trades). Align DataScheduler/DataStore flows with the new raw catalog writer.
- Ensure coverage manifests and scheduler configs include TBBO/Trades/MBP schemas so restoration runs; validate identifier templates in runtime configs (env/docker-compose).
- Add tests: schema→template resolution (per-schema/dataset maps), catalog coverage with MBP identifiers, raw writer MBP conversion, DBN CLI catalog mirror integration, and rehydration with MBP/TBBO templates.
- Update any runbooks/compose defaults that still only expose `CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE` so operators know to set per-schema maps when deviating from defaults.

### Schema registry plan (senior SWE recommendation)
- Create a single, explicit schema registry in a cycle-free module (`ml/schema.py`) mapping schema strings to `(DatasetType, Nautilus data class, default identifier template)`. Make it data-driven (dict), not substring heuristics.
- Align entries with Nautilus behavior: e.g., `mbp-1/mbp-10` → QuoteTick (depth/MBP), `tbbo/bbo` → QuoteTick, `trades` → TradeTick, `ohlcv/bar` → Bar; include Databento schemas, internal aliases, and custom overrides.
- Use this registry everywhere schema resolution is needed (coverage, rehydration, writers, CLI ingest) to avoid drift; add a small unit test to lock the mapping down.
- Keep Nautilus for serialization/catalog; the registry is just a translation layer from schema-string → Nautilus type/identifier template. Optionally derive unknown classes via Nautilus helpers (e.g., `filename_to_class`) but keep Databento schemas explicit to prevent surprises.
- Document defaults and identifier templates alongside the registry and validate at startup; keep identifier templates per schema/dataset in sync with the registry.

### Offline DBN → catalog → Postgres recipe
- Decode archives without SQL by mirroring straight to the Parquet catalog: `poetry run python -m ml.cli.ingest_dbn_archive data/batch --catalog-only --catalog-path ${CATALOG_PATH:-/home/nate/nautilus_data/ml_data/catalog} --dataset EQUS.MINI --source-dataset EQUS.MINI --schema ohlcv-1m --instrument-suffix .EQUS`. Repeat per schema (e.g., `tbbo`, `mbp-1`) or let metadata provide the schema when not overridden.
- When using SQL/DataStore mirrors instead of catalog-only, provide `--db-url` (and optionally `--mirror-data-store`) and the CLI will fan out to the catalog when `--catalog-path` is set.
- After the catalog is populated, run the pipeline with catalog rehydration enabled (`CATALOG_REHYDRATE_ENABLED=1`, template maps aligned to EQUS) so Postgres is replayed from Parquet before any API ingestion. Set `CATALOG_REHYDRATE_STALE_ONLY=0` to force a full replay.
- Verify freshness with `find ${CATALOG_PATH} -type f -printf '%T@ %p\n' | sort -nr | head` and Postgres spot checks on `market_data` before scaling out the universe/lookback.

### EQUS catalog ingest (DBN archives, catalog-only path)
- Bar history (EQUS.MINI, .EQUS suffix):
  - 1m: `data/batch/EQUS-OHLCV1m-20251202-8MXJT7C8WV.zip`
  - 1h: `data/batch/EQUS-OHLCV1H-20251202-PVX3JD5JYX.zip`
  - 1d: `data/batch/EQUS-OHLCV1D-20251202-5LXVKWGLXJ.zip`
  - Older 1m slice: `data/batch/EQUS-20251001-LV5KGQGBGG.zip`
  - Newer 1m slice: `data/batch/EQUS-20251128-S7BGQJWRJ4.zip`
- Quotes/TBBO:
  - Recent: `data/batch/EQUS-20251128-3WJVVFEEE9.zip` (tbbo)
  - Older: `data/batch/EQUS-TBBO-20251002-4EJGGSAJC6.zip` (tbbo)
  - Optional BBO-1m slice: `data/batch/EQUS-BBO-1m-20251002-K8RMF8TWV4.zip`
- Trades:
  - Recent: `data/batch/EQUS-20251128-SLPJ8QUGJL.zip` (trades)
  - Older: `data/batch/EQUS-Trades-20251002-HG76VY476P.zip` (trades)
- MBP-1:
  - Older: `data/batch/EQUS-MBP-1-20251002-R5FTL5LEXY.zip`
  - Newer: `data/batch/EQUS-20251125-KKQ9D3X3EJ.zip`
- Command template for all ingests (catalog-only, suffix enforced):
  ```
  poetry run python -m ml.cli.ingest_dbn_archive <archive> \
    --catalog-only \
    --catalog-path /home/nate/nautilus_data/ml_data/catalog \
    --instrument-suffix .EQUS \
    --schema <schema>
  ```
- Overlap handling: ParquetDataCatalog skips overlapping intervals. For clean rewrites, delete the target instrument directories (e.g., `data/bar/*EQUS*`, `data/quote_tick/*EQUS*`) before re-running. Run long MBP ingests inside tmux/screen.
- Post-ingest: enable catalog rehydration (`CATALOG_REHYDRATE_ENABLED=1`, `CATALOG_REHYDRATE_STALE_ONLY=0` for full replay) to hydrate Postgres from the catalog before API ingestion.
