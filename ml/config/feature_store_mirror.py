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
from ml.config._env_utils import env_truthy


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


__all__: tuple[str, ...] = (
    "DEFAULT_FEATURE_PARQUET_MIRROR_DIR",
    "FeatureStoreMirrorConfig",
)
