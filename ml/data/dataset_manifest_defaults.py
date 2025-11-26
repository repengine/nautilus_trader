"""Dataset manifest defaults and helpers for auto-registration paths."""

from __future__ import annotations

import time
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from ml.common.timestamps import sanitize_timestamp_ns
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.utils import compute_dataset_schema_hash


@dataclass(slots=True, frozen=True)
class DatasetManifestSpec:
    """Immutable defaults describing a dataset manifest template."""

    schema: Mapping[str, str]
    primary_keys: tuple[str, ...]
    retention_days: int
    partitioning: Mapping[str, Any] = field(
        default_factory=lambda: {"by": "ts_event", "interval": "daily"},
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ts_field: str = "ts_event"
    seq_field: str | None = None
    constraints: Mapping[str, Any] | None = None

    def copy_with(
        self,
        *,
        schema: Mapping[str, str] | None = None,
        primary_keys: tuple[str, ...] | None = None,
        retention_days: int | None = None,
        partitioning: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        ts_field: str | None = None,
        seq_field: str | None = None,
        constraints: Mapping[str, Any] | None = None,
    ) -> DatasetManifestSpec:
        return DatasetManifestSpec(
            schema=dict(schema) if schema is not None else dict(self.schema),
            primary_keys=primary_keys if primary_keys is not None else tuple(self.primary_keys),
            retention_days=retention_days if retention_days is not None else self.retention_days,
            partitioning=dict(partitioning) if partitioning is not None else dict(self.partitioning),
            metadata=dict(metadata) if metadata is not None else dict(self.metadata),
            ts_field=ts_field if ts_field is not None else self.ts_field,
            seq_field=seq_field if seq_field is not None else self.seq_field,
            constraints=dict(constraints) if constraints is not None else (
                dict(self.constraints) if self.constraints is not None else None
            ),
        )


@dataclass(slots=True, frozen=True)
class DatasetManifestOverrides:
    """Override payload for dataset-id specific adjustments."""

    dataset_type: DatasetType | None = None
    spec: DatasetManifestSpec | None = None
    spec_by_type: Mapping[DatasetType, DatasetManifestSpec] | None = None


_DEFAULT_SPEC = DatasetManifestSpec(
    schema={
        "instrument_id": "str",
        "ts_event": "int64",
        "ts_init": "int64",
    },
    primary_keys=("instrument_id", "ts_event"),
    retention_days=365,
)

_DATASET_TYPE_DEFAULTS: dict[DatasetType, DatasetManifestSpec] = {
    DatasetType.BARS: _DEFAULT_SPEC.copy_with(
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "symbol": "str",
            "publisher_id": "str",
            "rtype": "str",
            "source_dataset": "str",
        },
        metadata={
            "schema_kind": "bars",
            "bar_type_template": "{instrument_id}-1-MINUTE-LAST-EXTERNAL",
        },
        retention_days=730,
    ),
    DatasetType.TRADES: _DEFAULT_SPEC.copy_with(
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "price": "float64",
            "size": "float64",
            "sequence": "int64",
            "side": "str",
        },
        primary_keys=("instrument_id", "ts_event", "sequence"),
        metadata={"schema_kind": "trades"},
    ),
    DatasetType.TBBO: _DEFAULT_SPEC.copy_with(
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "bid": "float64",
            "ask": "float64",
            "bid_size": "float64",
            "ask_size": "float64",
        },
        metadata={"schema_kind": "tbbo"},
    ),
    DatasetType.MBP1: _DEFAULT_SPEC.copy_with(
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "bid_px": "float64",
            "ask_px": "float64",
            "bid_sz": "float64",
            "ask_sz": "float64",
            "level": "int32",
            "side": "str",
        },
        primary_keys=("instrument_id", "ts_event", "level", "side"),
        partitioning={"by": "ts_event", "interval": "hourly"},
        retention_days=90,
        metadata={"schema_kind": "mbp1"},
    ),
    DatasetType.QUOTES: _DEFAULT_SPEC.copy_with(
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "bid_px": "float64",
            "ask_px": "float64",
            "bid_sz": "float64",
            "ask_sz": "float64",
        },
        metadata={"schema_kind": "quotes"},
    ),
    DatasetType.FEATURES: DatasetManifestSpec(
        schema={
            "instrument_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "feature_values": "json",
        },
        primary_keys=("instrument_id", "ts_event"),
        retention_days=365,
        metadata={"schema_kind": "features"},
    ),
    DatasetType.PREDICTIONS: DatasetManifestSpec(
        schema={
            "instrument_id": "str",
            "model_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "prediction": "float64",
            "confidence": "float64",
            "metadata": "json",
        },
        primary_keys=("instrument_id", "model_id", "ts_event"),
        retention_days=365,
        metadata={"schema_kind": "predictions"},
    ),
    DatasetType.SIGNALS: DatasetManifestSpec(
        schema={
            "instrument_id": "str",
            "strategy_id": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "signal_type": "str",
            "signal_value": "float64",
            "metadata": "json",
        },
        primary_keys=("instrument_id", "strategy_id", "ts_event"),
        retention_days=365,
        metadata={"schema_kind": "signals"},
    ),
    DatasetType.EARNINGS_ACTUALS: DatasetManifestSpec(
        schema={
            "ticker": "str",
            "period_end": "str",
            "filing_date": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "eps_basic": "float64",
            "eps_diluted": "float64",
            "revenue": "float64",
            "net_income": "float64",
            "operating_income": "float64",
            "shares_outstanding": "int64",
            "filing_type": "str",
            "fiscal_year": "int32",
            "fiscal_quarter": "int32",
            "data_source": "str",
        },
        primary_keys=("ticker", "period_end"),
        retention_days=3650,
        metadata={"schema_kind": "earnings_actuals"},
        partitioning={"by": "filing_date", "interval": "monthly"},
        ts_field="ts_event",
        constraints={
            "nullability": {
                "ticker": False,
                "period_end": False,
                "filing_date": False,
                "ts_event": False,
                "ts_init": False,
            }
        },
    ),
    DatasetType.EARNINGS_ESTIMATES: DatasetManifestSpec(
        schema={
            "ticker": "str",
            "estimate_date": "str",
            "period_end": "str",
            "ts_event": "int64",
            "ts_init": "int64",
            "eps_consensus": "float64",
            "revenue_consensus": "float64",
            "num_analysts": "int32",
            "data_source": "str",
        },
        primary_keys=("ticker", "estimate_date", "period_end"),
        retention_days=1825,
        metadata={"schema_kind": "earnings_estimates"},
        partitioning={"by": "estimate_date", "interval": "monthly"},
        ts_field="ts_event",
        constraints={
            "nullability": {
                "ticker": False,
                "estimate_date": False,
                "period_end": False,
                "ts_event": False,
                "ts_init": False,
            }
        },
    ),
}

_DATASET_ID_OVERRIDES: dict[str, DatasetManifestOverrides] = {
    "DBEQ.BASIC": DatasetManifestOverrides(
        dataset_type=DatasetType.MBP1,
        spec=_DATASET_TYPE_DEFAULTS[DatasetType.MBP1].copy_with(
            metadata={
                "schema_kind": "mbp1",
                "source": "databento",
                "dataset_family": "depth_l1",
            },
        ),
    ),
    "EQUS.MINI": DatasetManifestOverrides(
        dataset_type=DatasetType.TRADES,
        spec_by_type={
            DatasetType.TRADES: _DATASET_TYPE_DEFAULTS[DatasetType.TRADES].copy_with(
                metadata={
                    "schema_kind": "trades",
                    "source": "databento",
                    "dataset_family": "equities_mini",
                    "canonicalization_modes": [
                        "native",
                        "reaggregated_trades",
                        "scaled_volume",
                    ],
                    "fallback_source_dataset": "XNAS.ITCH",
                },
            ),
        },
    ),
    "XNAS.ITCH": DatasetManifestOverrides(
        dataset_type=DatasetType.TRADES,
        spec=_DATASET_TYPE_DEFAULTS[DatasetType.TRADES].copy_with(
            metadata={
                "schema_kind": "trades",
                "source": "nasdaq",
                "dataset_family": "itch",
            },
        ),
    ),
    "ml.earnings_actuals": DatasetManifestOverrides(
        dataset_type=DatasetType.EARNINGS_ACTUALS,
    ),
    "ml.earnings_estimates": DatasetManifestOverrides(
        dataset_type=DatasetType.EARNINGS_ESTIMATES,
    ),
}


def _compute_schema_hash(
    *,
    schema: Mapping[str, str],
    primary_keys: Sequence[str],
    ts_field: str,
    seq_field: str | None,
    pipeline_signature: str,
) -> str:
    return compute_dataset_schema_hash(
        schema=schema,
        primary_keys=primary_keys,
        ts_field=ts_field,
        seq_field=seq_field,
        pipeline_signature=pipeline_signature,
    )


def _merge_metadata(
    base: Mapping[str, Any],
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    if extra:
        merged.update(extra)
    return merged


def resolve_dataset_manifest_spec(
    dataset_id: str,
    *,
    dataset_type: DatasetType | None = None,
) -> tuple[DatasetType, DatasetManifestSpec]:
    """Resolve the manifest spec for the given dataset id/type."""
    override = _DATASET_ID_OVERRIDES.get(dataset_id)
    resolved_type = dataset_type
    if override is not None:
        if override.dataset_type is not None:
            if (
                resolved_type is not None
                and resolved_type != override.dataset_type
                and override.spec_by_type is None
            ):
                raise ValueError(
                    "Dataset type mismatch for override",
                )
            if resolved_type is None:
                resolved_type = override.dataset_type

        if override.spec_by_type is not None:
            if resolved_type is None:
                if len(override.spec_by_type) == 1:
                    resolved_type, spec = next(iter(override.spec_by_type.items()))
                    return resolved_type, spec
            else:
                spec_by_type = override.spec_by_type.get(resolved_type)
                if spec_by_type is not None:
                    return resolved_type, spec_by_type

        if override.spec is not None:
            if resolved_type is None:
                resolved_type = override.dataset_type
            if (
                override.dataset_type is not None
                and resolved_type is not None
                and resolved_type != override.dataset_type
            ):
                raise ValueError(
                    "Dataset type mismatch for override",
                )
            if resolved_type is None:
                raise ValueError(f"Unable to infer dataset type for {dataset_id}")
            return resolved_type, override.spec

    if resolved_type is None:
        raise ValueError(f"Dataset type is required for {dataset_id}")

    spec = _DATASET_TYPE_DEFAULTS.get(resolved_type, _DEFAULT_SPEC)
    return resolved_type, spec


def build_auto_dataset_manifest(
    *,
    dataset_id: str,
    dataset_type: DatasetType | None,
    location: str,
    storage_kind: StorageKind,
    pipeline_signature: str,
    metadata: Mapping[str, Any] | None = None,
    version: str = "1.0.0",
    retention_days: int | None = None,
) -> DatasetManifest:
    """Construct a dataset manifest using defaults and optional overrides."""
    resolved_type, spec = resolve_dataset_manifest_spec(dataset_id, dataset_type=dataset_type)
    schema = dict(spec.schema)
    partitioning = dict(spec.partitioning)
    metadata_payload = _merge_metadata(spec.metadata, metadata)
    metadata_payload.setdefault("dataset_id", dataset_id)
    metadata_payload.setdefault("dataset_type", resolved_type.value)

    non_nullable = {
        field: False
        for field in ("instrument_id", "ts_event", "ts_init")
        if field in schema
    }
    constraints = dict(spec.constraints) if spec.constraints is not None else {}
    if non_nullable:
        constraints.setdefault("nullability", non_nullable)

    now_ns = sanitize_timestamp_ns(time.time_ns(), context="dataset_manifest_defaults:now")

    schema_hash = _compute_schema_hash(
        schema=schema,
        primary_keys=spec.primary_keys,
        ts_field=spec.ts_field,
        seq_field=spec.seq_field,
        pipeline_signature=pipeline_signature,
    )
    location_str = str(Path(location).expanduser()) if storage_kind is StorageKind.PARQUET else location

    return DatasetManifest(
        dataset_id=dataset_id,
        dataset_type=resolved_type,
        storage_kind=storage_kind,
        location=location_str,
        partitioning=partitioning,
        retention_days=retention_days if retention_days is not None else spec.retention_days,
        schema=schema,
        ts_field=spec.ts_field,
        seq_field=spec.seq_field,
        primary_keys=list(spec.primary_keys),
        schema_hash=schema_hash,
        constraints=constraints,
        lineage=[],
        pipeline_signature=pipeline_signature,
        version=version,
        created_at=now_ns,
        last_modified=now_ns,
        metadata=metadata_payload,
    )


__all__ = [
    "DatasetManifestOverrides",
    "DatasetManifestSpec",
    "build_auto_dataset_manifest",
    "resolve_dataset_manifest_spec",
]
