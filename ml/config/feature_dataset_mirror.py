"""
Configuration for feature dataset parquet mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml.orchestration.config_types import MacroIngestionConfig
from ml.stores.feature_raw_writer import FeatureDatasetParquetRawWriter


@dataclass(frozen=True, slots=True)
class FeatureDatasetMirrorConfig:
    """
    Configuration for feature dataset parquet mirrors.

    Attributes
    ----------
    events_path : Path
        Destination path for events parquet mirror.
    macro_fred_path : Path
        Destination path for FRED macro observations parquet mirror.
    macro_vintage_dir : Path
        Base directory for ALFRED release calendar parquet mirrors.
    macro_series_path : Path
        Path to the macro series list file.

    Example
    -------
    >>> cfg = FeatureDatasetMirrorConfig.from_env()
    >>> cfg.events_path.name
    'events.parquet'
    """

    events_path: Path
    macro_fred_path: Path
    macro_vintage_dir: Path
    macro_series_path: Path

    @classmethod
    def from_env(cls) -> FeatureDatasetMirrorConfig:
        """
        Build mirror config using standard environment defaults.

        Returns
        -------
        FeatureDatasetMirrorConfig
            Mirror configuration resolved from environment and defaults.
        """
        macro_cfg = MacroIngestionConfig()
        writer = FeatureDatasetParquetRawWriter()
        vintage_dir = macro_cfg.vintage_dir
        if vintage_dir is None:
            raise ValueError("MacroIngestionConfig.vintage_dir must not be None")
        return cls(
            events_path=writer.events_path,
            macro_fred_path=Path(macro_cfg.fred_path),
            macro_vintage_dir=Path(vintage_dir),
            macro_series_path=Path("ml/config/macro_fred_series.txt"),
        )
