# LLM Prompt Anchors (Canonical Vocabulary)

Use this file as the first reference for schema/dataset naming and ingestion
policy. It is intentionally short and points to the authoritative sources for
details. Include this file in prompts for any Codex/LLM work on this repo.

## Canonical Sources Of Truth (in priority order)
1. `ml/schema.py`  
   Schema tokens (aliases), DatasetType mapping, dataclass binding, identifier templates.
2. `ml/registry/dataclasses.py`  
   Canonical `DatasetType` enum values.
3. `ml/config/market_feed_descriptors.json`  
   Dataset IDs and provider schema mapping (e.g., `EQUS.MINI_QUOTES` uses provider `tbbo`).
4. `ml/data/dataset_manifest_defaults.py`  
   Manifest defaults + `schema_kind` metadata.
5. `ml/config/market_data.py`  
   Table routing for market data schemas (quotes/tbbo/trades/mbp/mbp10/mbo).
6. `ml/data/common/dataset_registration.py`  
   Dataset ID naming rules (`build_dataset_id_for_schema`).
7. `ml/data/ingest/subscription.py`  
   Lookback policy (L0/L1/L2/L3).
8. `ml/docs/development/SCHEMA_REGISTRY.md`  
   Human-readable schema registry + quick grep targets.

## Current Ingestion Policy (as of 2026-02-05)
- **L0**: `bars` (`ohlcv-1m`)
- **L1**: `quotes` (canonical schema), sourced from Databento `tbbo`
  via `EQUS.MINI_QUOTES` in `ml/config/market_feed_descriptors.json`.
- **L2/L3**: disabled by default; add `mbp-10`/`mbo` when licensing allows.

## Non-Negotiables
- No ad-hoc schema mappings. Always use `ml.schema.schema_spec_for` /
  `ml.schema.map_schema_to_dataset_type`.
- Identifier templates are fixed by the schema registry. Do not introduce
  per-schema or per-dataset overrides.
- Catalog identifiers are canonical (no suffixed directories).

## If Policy Changes
Update all of the following in one pass:
1. `ml/config/market_feed_descriptors.json` (dataset + provider schema).
2. `.env` and `ml/deployment/docker-compose*.yml` (`MARKET_DATASET_INPUTS`).
3. Auto-fill schema selection in orchestration (`ml/orchestration/*`).
4. Tests that assert schema names or auto-fill behavior.
