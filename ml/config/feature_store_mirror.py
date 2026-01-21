#!/usr/bin/env python3
"""
Configuration for FeatureStore parquet mirror writes.

The mirror is used to persist computed FeatureStore values as partitioned parquet files
so coverage tooling can restore PostgreSQL without API calls.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ml.config._env_utils import ensure_env
from ml.config._env_utils import env_positive_int
from ml.config._env_utils import env_truthy
from ml.config._env_utils import resolve_db_connection


DEFAULT_FEATURE_PARQUET_MIRROR_DIR: Final[Path] = Path(
    "data/features/store/feature_values",
)


@dataclass(frozen=True)
class FeatureStoreMirrorConfig:
    """
    Configuration for FeatureStore parquet mirroring.

    Attributes
    ----------
    enabled:
        Whether to enable mirror writes.
    base_dir:
        Root directory for partitioned feature parquet files.
    partition_field:
        Column used for partitioning (default: "instrument_id").
    timestamp_field:
        Timestamp column used for day bucketing (default: "ts_event").
    values_field:
        Column containing serialized feature values (default: "values").

    """

    enabled: bool = True
    base_dir: Path = DEFAULT_FEATURE_PARQUET_MIRROR_DIR
    partition_field: str = "instrument_id"
    timestamp_field: str = "ts_event"
    values_field: str = "values"

    def __post_init__(self) -> None:
        """
        Validate configuration values.
        """
        if not str(self.base_dir).strip():
            raise ValueError("base_dir must be provided for FeatureStoreMirrorConfig")
        if not self.partition_field:
            raise ValueError("partition_field must be provided for FeatureStoreMirrorConfig")
        if not self.timestamp_field:
            raise ValueError("timestamp_field must be provided for FeatureStoreMirrorConfig")
        if not self.values_field:
            raise ValueError("values_field must be provided for FeatureStoreMirrorConfig")

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> FeatureStoreMirrorConfig:
        """
        Build a mirror config from environment variables.

        Environment overrides
        ---------------------
        ML_FEATURE_PARQUET_MIRROR_ENABLE:
            Enable mirror writes (default: true).
        ML_FEATURE_PARQUET_MIRROR_DIR:
            Base directory for mirror parquet files.

        """
        source = ensure_env(env)
        enabled = env_truthy(source, "ML_FEATURE_PARQUET_MIRROR_ENABLE", default=True)
        raw_dir = source.get("ML_FEATURE_PARQUET_MIRROR_DIR")
        base_dir = Path(raw_dir).expanduser() if raw_dir else DEFAULT_FEATURE_PARQUET_MIRROR_DIR
        return cls(
            enabled=enabled,
            base_dir=base_dir,
        )


@dataclass(frozen=True)
class FeatureStoreMirrorBackfillConfig:
    """
    Configuration for backfilling feature store parquet mirrors.

    Attributes
    ----------
    db_connection:
        Database connection string for reading feature values.
    batch_size:
        Number of rows to fetch per batch.
    start_ts:
        Optional lower bound (inclusive) for ts_event in nanoseconds.
    end_ts:
        Optional upper bound (exclusive) for ts_event in nanoseconds.
    feature_set_id:
        Optional feature set filter.
    instrument_id:
        Optional instrument filter.
    """

    db_connection: str
    batch_size: int = 5000
    start_ts: int | None = None
    end_ts: int | None = None
    feature_set_id: str | None = None
    instrument_id: str | None = None

    def __post_init__(self) -> None:
        if not self.db_connection:
            raise ValueError("db_connection must be provided for FeatureStoreMirrorBackfillConfig")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.start_ts is not None and self.end_ts is not None:
            if self.start_ts > self.end_ts:
                raise ValueError("start_ts must be <= end_ts")

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> FeatureStoreMirrorBackfillConfig:
        """
        Build a backfill config from environment variables.

        Environment overrides
        ---------------------
        ML_FEATURE_MIRROR_BACKFILL_BATCH_SIZE:
            Batch size for SQL reads.
        ML_FEATURE_MIRROR_BACKFILL_START_TS:
            Start timestamp (ns) filter.
        ML_FEATURE_MIRROR_BACKFILL_END_TS:
            End timestamp (ns) filter.
        ML_FEATURE_MIRROR_BACKFILL_FEATURE_SET_ID:
            Feature set filter.
        ML_FEATURE_MIRROR_BACKFILL_INSTRUMENT_ID:
            Instrument filter.
        """
        source = ensure_env(env)
        db_connection = resolve_db_connection(source)
        if not db_connection:
            raise ValueError("db_connection missing from environment")
        batch_size = env_positive_int(
            source,
            "ML_FEATURE_MIRROR_BACKFILL_BATCH_SIZE",
            default=5000,
        )
        start_ts = _parse_optional_int(source, "ML_FEATURE_MIRROR_BACKFILL_START_TS")
        end_ts = _parse_optional_int(source, "ML_FEATURE_MIRROR_BACKFILL_END_TS")
        feature_set_id = _normalize_optional(source.get("ML_FEATURE_MIRROR_BACKFILL_FEATURE_SET_ID"))
        instrument_id = _normalize_optional(source.get("ML_FEATURE_MIRROR_BACKFILL_INSTRUMENT_ID"))
        return cls(
            db_connection=db_connection,
            batch_size=batch_size,
            start_ts=start_ts,
            end_ts=end_ts,
            feature_set_id=feature_set_id,
            instrument_id=instrument_id,
        )


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _parse_optional_int(source: Mapping[str, str], key: str) -> int | None:
    raw = source.get(key)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


__all__: tuple[str, ...] = (
    "DEFAULT_FEATURE_PARQUET_MIRROR_DIR",
    "FeatureStoreMirrorBackfillConfig",
    "FeatureStoreMirrorConfig",
)
