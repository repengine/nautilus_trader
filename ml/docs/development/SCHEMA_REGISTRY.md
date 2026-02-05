# Schema Registry & Vocabulary

This document centralizes the vocabulary for dataset schema tokens and their mappings.
Use it as the first stop in prompts, reviews, and greps to keep naming consistent
and avoid DRY violations. For a compact LLM prompt anchor, see
`ml/docs/development/LLM_CANONICALS.md`.

## Source Of Truth
- `ml/schema.py`: schema tokens (aliases), dataset types, dataclass bindings, and identifier templates.
- `ml/registry/dataclasses.py`: `DatasetType` enum (canonical dataset type names).
- `ml/data/dataset_manifest_defaults.py`: manifest schemas + `schema_kind` metadata.
- `ml/stores/base.py`: strategy event dataclasses (order events, risk halts, replay summary).

## Programmatic Discovery
Use these helpers from `ml/schema.py` to inspect supported schema tokens:

```python
from ml.schema import list_registered_schemas, schema_registry_snapshot

print(list_registered_schemas())
print(schema_registry_snapshot())
```

## Expectations
- **No ad-hoc schema mappings**: use `ml.schema.map_schema_to_dataset_type` or
  `ml.schema.schema_spec_for`.
- **Explicit unsupported schemas**: if a schema is not registered, treat it as
  unsupported until it is intentionally added.
- **Identifier templates**: schema-specific templates live in `ml/schema.py` and
  dataset-level templates in `DATASET_TYPE_IDENTIFIER_DEFAULTS`.
- **Depth vocab**: MBP-1 (`mbp-1`/`mbp1`), MBP-10 (`mbp-10`/`mbp10`), and MBO (`mbo`)
  are explicit schema tokens with distinct `DatasetType`s.

## Dataset ID Naming
Dataset identifiers are derived from the **canonical schema registry** and should
use `DatasetType.value`, not raw schema tokens:

- `bars_{symbol}_{venue}` for OHLCV schemas (`ohlcv-1m`, etc.)
- `mbp1_{symbol}_{venue}` for `mbp-1`
- `mbp10_{symbol}_{venue}` for `mbp-10`
- `mbo_{symbol}_{venue}` for `mbo`

Use `build_dataset_id_for_schema` to avoid drift:

```python
from ml.data.common.dataset_registration import build_dataset_id_for_schema

dataset_id = build_dataset_id_for_schema(
    schema="mbp-10",
    symbol_code="AAPL",
    venue="XNAS",
)
assert dataset_id == "mbp10_aapl_xnas"
```

Legacy prefixes (`ohlcv_*`, `mbp_*`) should not appear in new registry events.

## Preflight Legacy ID Audit
To check for legacy dataset_id usage in Postgres:

```sql
SELECT DISTINCT dataset_id
FROM ml_dataset_registry
WHERE dataset_id ILIKE 'ohlcv\_%' ESCAPE '\'
   OR dataset_id ILIKE 'mbp\_%' ESCAPE '\'
ORDER BY dataset_id;

SELECT DISTINCT dataset_id
FROM ml_data_events
WHERE dataset_id ILIKE 'ohlcv\_%' ESCAPE '\'
   OR dataset_id ILIKE 'mbp\_%' ESCAPE '\'
ORDER BY dataset_id;

SELECT DISTINCT dataset_id
FROM ml_data_watermarks
WHERE dataset_id ILIKE 'ohlcv\_%' ESCAPE '\'
   OR dataset_id ILIKE 'mbp\_%' ESCAPE '\'
ORDER BY dataset_id;
```

For JSON-backed registries:

```python
from ml.registry.data_registry import DataRegistry

report = registry.list_legacy_dataset_ids()
print(report)
```

## Legacy Catalog Identifier Cleanup
If you previously used suffixed identifier templates (e.g., `AAPL.EQUS-MBP1`),
use the cleanup tool to prune or consolidate those directories. This aligns
catalog identifiers with the schema registry defaults.

```bash
python -m ml.tools.catalog_identifier_cleanup \
  --catalog-path data/catalog \
  --class-dir quote_tick \
  --suffix MBP1 --suffix TBBO \
  --mode prune --apply
```

## Quick Grep Targets
- `ml/schema.py` for schema tokens and aliases.
- `DatasetType` in `ml/registry/dataclasses.py` for canonical dataset type names.
- `schema_kind` in `ml/data/dataset_manifest_defaults.py` for manifest metadata.
