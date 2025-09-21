"""
Dataset task helpers consumed by CLI entry points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .report import DatasetReport
from .report import DatasetReportConfig
from .report import generate_dataset_report
from .tft import FeatureRoleName
from .tft import TFTDatasetTaskConfig
from .tft import build_tft_dataset


ProductionDatasetConfig: type[Any] | None = None
build_production_dataset: Callable[..., object] | None = None

try:  # Optional dependency tree
    from .production import ProductionDatasetConfig as _ProductionDatasetConfig
    from .production import build_production_dataset as _build_production_dataset
except ModuleNotFoundError:  # pragma: no cover - optional dependencies missing
    ProductionDatasetConfig = None
    build_production_dataset = None
else:
    ProductionDatasetConfig = _ProductionDatasetConfig
    build_production_dataset = _build_production_dataset


__all__ = [
    "DatasetReport",
    "DatasetReportConfig",
    "FeatureRoleName",
    "ProductionDatasetConfig",
    "TFTDatasetTaskConfig",
    "build_production_dataset",
    "build_tft_dataset",
    "generate_dataset_report",
]
