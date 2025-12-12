"""
Typed loader for dataset coverage configuration files.

The coverage manager accepts :class:`~ml.data.coverage.manager.DatasetCoverageConfig`
instances describing which dataset/instrument pairs should be inspected. Feature
families such as earnings or macro data need additional metadata that tells the
system how to inspect SQL tables and where the parquet mirrors live. This module
parses TOML configuration files that capture those details.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.schema import schema_spec_for


try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for alternative runtimes
    import tomli as tomllib  # type: ignore[no-redef]

from ml.config.universes import TIER1_SYMBOL_SETS
from ml.data.coverage.manager import DatasetCoverageConfig
from ml.stores.providers import ParquetCoverageSpec
from ml.stores.providers import SqlCoverageOverride


AliasMap = dict[str, tuple[str, ...]]
_BASE_ALIAS_MAP: AliasMap = {}
for alias, symbols in TIER1_SYMBOL_SETS.items():
    normalized = tuple(str(symbol).strip().upper() for symbol in symbols if symbol)
    if normalized:
        _BASE_ALIAS_MAP[alias.lower()] = normalized

_ALIAS_SYNONYMS = {
    "tier1": "default",
    "tier1_full": "full",
    "tier1_full_95": "full",
    "tier1_core": "core",
    "tier1_core12": "core12",
}
for source, target in _ALIAS_SYNONYMS.items():
    resolved = _BASE_ALIAS_MAP.get(target.lower())
    if resolved is not None:
        _BASE_ALIAS_MAP[source.lower()] = resolved


@dataclass(frozen=True, slots=True)
class CoverageDatasetEntry:
    """
    Fully parsed dataset coverage definition including provider overrides.
    """

    dataset: DatasetCoverageConfig
    sql_override: SqlCoverageOverride | None = None
    parquet_spec: ParquetCoverageSpec | None = None


def load_dataset_coverage_entries(path: str | Path) -> tuple[CoverageDatasetEntry, ...]:
    """
    Load dataset coverage entries from a TOML document.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Coverage dataset config not found: {config_path}")
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    raw_datasets = data.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError(f"No datasets defined in {config_path}")
    entries: list[CoverageDatasetEntry] = []
    for index, raw in enumerate(raw_datasets):
        if not isinstance(raw, dict):
            raise ValueError(f"Dataset entry #{index} must be a table")
        entries.append(_parse_dataset_entry(raw, base_dir=config_path.parent))
    return tuple(entries)


def _parse_dataset_entry(payload: dict[str, Any], *, base_dir: Path) -> CoverageDatasetEntry:
    dataset_id = _require_str(payload.get("dataset_id"), "dataset_id")
    schema = _require_str(payload.get("schema"), "schema")
    schema_spec_for(schema)
    strip_venue = bool(payload.get("strip_venue", False))
    entity_field = payload.get("entity_field", "instrument_id")
    if not isinstance(entity_field, str) or not entity_field.strip():
        raise ValueError(f"dataset {dataset_id} requires a non-empty entity_field")
    entities_value = payload.get("entities") or payload.get("symbols") or payload.get("instruments")
    if entities_value is None:
        raise ValueError(f"dataset {dataset_id} must define 'entities'")
    instruments = _parse_entities(entities_value, strip_venue=strip_venue)
    sql_override = _parse_sql_override(payload.get("sql"), dataset_id=dataset_id)
    parquet_spec = _parse_parquet_spec(payload.get("parquet"), dataset_id=dataset_id, base_dir=base_dir)
    dataset_cfg = DatasetCoverageConfig(
        dataset_id=dataset_id,
        schema=schema,
        instruments=instruments,
        entity_field=entity_field,
    )
    return CoverageDatasetEntry(
        dataset=dataset_cfg,
        sql_override=sql_override,
        parquet_spec=parquet_spec,
    )


def _parse_sql_override(payload: Any, *, dataset_id: str) -> SqlCoverageOverride | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"sql override for {dataset_id} must be a table")
    table = _optional_str(payload.get("table"))
    schema = _optional_str(payload.get("schema"))
    ts_field = _optional_str(payload.get("ts_field"))
    entity_field = _optional_str(payload.get("entity_field"))
    if not any((table, schema, ts_field, entity_field)):
        return None
    return SqlCoverageOverride(
        table_name=table,
        schema=schema,
        ts_field=ts_field,
        entity_field=entity_field,
    )


def _parse_parquet_spec(
    payload: Any,
    *,
    dataset_id: str,
    base_dir: Path,
) -> ParquetCoverageSpec | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"parquet override for {dataset_id} must be a table")
    path_raw = _optional_str(payload.get("path"))
    if not path_raw:
        return None
    base_path = str((base_dir / Path(path_raw)).resolve())
    partition_field = _optional_str(payload.get("partition_field")) or "instrument_id"
    timestamp_field = _optional_str(payload.get("timestamp_field")) or "ts_event"
    template_raw = payload.get("partition_template")
    if template_raw is None:
        partition_template = None
    elif isinstance(template_raw, str):
        partition_template = template_raw.strip()
        if template_raw == "":
            partition_template = ""
    else:
        partition_template = str(template_raw).strip()
    return ParquetCoverageSpec(
        dataset_id=dataset_id,
        base_path=base_path,
        partition_field=partition_field,
        timestamp_field=timestamp_field,
        partition_template=partition_template,
    )


def _parse_entities(value: Any, *, strip_venue: bool) -> tuple[str, ...]:
    tokens: list[str] = []
    if isinstance(value, str):
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple)):
        for entry in value:
            if entry is None:
                continue
            token = str(entry).strip()
            if token:
                tokens.append(token)
    else:
        raise ValueError("entities must be a string or list")
    expanded: list[str] = []
    for token in tokens:
        if token.startswith("@"):
            expanded.extend(_expand_alias(token[1:]))
        else:
            expanded.append(token)
    return _normalize_entities(expanded, strip_venue=strip_venue)


def _expand_alias(alias: str) -> tuple[str, ...]:
    resolved = _BASE_ALIAS_MAP.get(alias.lower())
    if resolved is None:
        raise ValueError(f"Unknown universe alias @{alias}")
    return resolved


def _normalize_entities(values: list[str], *, strip_venue: bool) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = value.strip().upper()
        if not token:
            continue
        if strip_venue and "." in token:
            token = token.split(".", 1)[0]
        if token not in seen:
            seen.add(token)
            normalized.append(token)
    if not normalized:
        raise ValueError("entity list cannot be empty after normalization")
    return tuple(normalized)


def _require_str(payload: Any, field: str) -> str:
    value = _optional_str(payload)
    if not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return str(value).strip() or None


def dump_coverage_entries(entries: tuple[CoverageDatasetEntry, ...]) -> str:
    """
    Helper used in debugging/tests to serialize parsed entries.
    """
    payload = []
    for entry in entries:
        record: dict[str, Any] = {
            "dataset_id": entry.dataset.dataset_id,
            "schema": entry.dataset.schema,
            "entity_field": entry.dataset.entity_field,
            "instruments": entry.dataset.instruments,
        }
        if entry.sql_override:
            record["sql_override"] = entry.sql_override.__dict__
        if entry.parquet_spec:
            record["parquet_spec"] = {
                "base_path": str(entry.parquet_spec.base_path),
                "partition_field": entry.parquet_spec.partition_field,
                "timestamp_field": entry.parquet_spec.timestamp_field,
            }
        payload.append(record)
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "CoverageDatasetEntry",
    "dump_coverage_entries",
    "load_dataset_coverage_entries",
]
