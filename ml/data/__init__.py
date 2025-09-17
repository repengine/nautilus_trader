"""
Data utilities and infrastructure for ML workflows in Nautilus Trader.

This package provides comprehensive data handling capabilities for ML workflows, including:

- **Data Loading & Conversion**: Utilities for converting Nautilus data to DataFrames
- **Data Collection**: Automated collection from external sources (Databento, FRED)
- **Data Scheduling**: Automated daily data collection and processing pipelines
- **Dataset Building**: TFT-compatible dataset preparation with feature engineering
- **Caching & Performance**: L2 and microstructure minute-level caches for fast training
- **Data Providers**: Pluggable providers for metadata, calendar, and event data
- **Data Sources**: Mock and real data sources for testing and production
- **Ingestion**: Robust data ingestion pipelines with retry and resume capabilities
- **Fixtures**: Test data generation utilities

## Architecture Patterns

This module follows cold path patterns as data loading is never hot path:
- All components are designed for batch/offline processing
- Uses progressive fallback for external dependencies
- Implements comprehensive error handling and retry logic
- Provides extensive metrics and monitoring

## Core Components

### Data Loading & Conversion
- `bars_to_dataframe`: Convert catalog bars to Polars DataFrame
- `quotes_to_dataframe`: Convert catalog quotes to Polars DataFrame
- `trades_to_dataframe`: Convert catalog trades to Polars DataFrame

### High-Level Orchestrators
- `DataCollector`: Rich market microstructure data collection
- `DataScheduler`: Automated daily data collection and processing
- `TFTDatasetBuilder`: Fast TFT-compatible dataset building

### Performance & Caching
- `L2MinuteCache`: L2 market depth per-minute aggregation cache
- `MicroMinuteCache`: Microstructure per-minute feature cache

### Data Providers (Protocol-Based)
- `InstrumentMetadataProvider`: Static instrument metadata
- `MarketCalendarProvider`: Trading calendar and session information
- `EventScheduleProvider`: Economic events and announcements

### Data Sources (Testing & Mocking)
- `MockCalendarSource`: Mock calendar for testing
- `SimpleCalendarSource`: Simple calendar implementation
- `PandasCalendarSource`: Pandas-based calendar
- `MockEventSource`: Mock event source for testing
- `DatabentoMetadataSource`: Databento metadata integration
- `NautilusMetadataSource`: Nautilus-native metadata

### Data Loaders
- `FREDDataLoader`: FRED economic data integration
- `FREDConfig`: FRED loader configuration
- `FREDIndicator`: FRED economic indicators

### Fixtures & Testing
- `FixtureManifest`: Test data manifest management
- `make_mbp10_fixture`: MBP-10 test data generation
- `make_tbbo_fixture`: TBBO test data generation
- `make_trades_fixture`: Trade test data generation

## Example Usage

### Basic Data Loading
```python
from ml.data import bars_to_dataframe
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

catalog = ParquetDataCatalog("./data")
df = bars_to_dataframe(
    catalog=catalog,
    instrument_ids=["SPY.ARCA", "QQQ.NASDAQ"],
    start="2024-01-01",
    end="2024-01-31"
)
```

### TFT Dataset Building
```python
from ml.data import TFTDatasetBuilder

builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=["SPY", "QQQ", "AAPL"],
    include_macro=True,
    include_micro=True
)
dataset = builder.build_training_dataset(
    horizon_minutes=15,
    lookback_periods=30
)
```

### Automated Data Collection
```python
from ml.data import DataCollector

collector = DataCollector(storage_limit_gb=500.0)
collector.run_collection()  # Collect L2, trades, quotes, bars
```

### Scheduled Data Pipeline
```python
from ml.data.scheduler import DataScheduler
from ml.config.scheduler_config import SchedulerConfig

config = SchedulerConfig(
    symbols=["SPY.ARCA", "QQQ.NASDAQ"],
    retention_days=90
)
scheduler = DataScheduler(catalog=catalog, config=config)
scheduler.run_daily_update()
```

## Data Quality & Validation

All data utilities include:
- Schema validation and type checking
- Missing data handling and imputation
- Outlier detection and filtering
- Timestamp validation and timezone handling
- Comprehensive logging and error reporting

## Performance Considerations

- Uses Polars for fast DataFrame operations
- Implements day-partitioned caching for large datasets
- Provides memory-efficient streaming for large time ranges
- Supports parallel processing where applicable

## Dependencies

Optional dependencies are handled gracefully:
- Polars: DataFrame operations (auto-installed)
- Databento: External data collection (optional)
- FRED API: Economic data integration (optional)

See ml/_imports.py for dependency management patterns.

"""

# Enable postponed evaluation of annotations to avoid importing optional deps for types
from __future__ import annotations

# Core data conversion utilities
from ml.data.catalog_utils import bars_to_dataframe
from ml.data.catalog_utils import quotes_to_dataframe
from ml.data.catalog_utils import trades_to_dataframe

# High-level orchestrators and builders
from ml.data.collector import DataCollector

# Fixtures and testing utilities
from ml.data.fixtures import FixtureManifest
from ml.data.fixtures import compute_schema_hash
from ml.data.fixtures import make_mbp10_fixture
from ml.data.fixtures import make_tbbo_fixture
from ml.data.fixtures import make_trades_fixture
from ml.data.ingest.common import BackoffPolicy

# Ingestion utilities
from ml.data.ingest.common import IngestState
from ml.data.ingest.common import RateLimiter
# Note: DomainWindowLoaderProtocol and IngestionOrchestrator moved to avoid circular imports
# Import directly from ml.data.ingest.orchestrator when needed

# Performance and caching
from ml.data.l2_cache import L2MinuteCache

# Data loaders
from ml.data.micro_cache import MicroMinuteCache
from ml.data.providers.base import CacheableProvider

# Provider base classes and protocols
from ml.data.providers.base import DataProvider
from ml.data.providers.base import StaticDataProvider
from ml.data.providers.base import TimeSeriesProvider
from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.providers.events import EventScheduleProvider

# Data providers (protocol-based)
from ml.data.providers.metadata import InstrumentMetadataProvider
# Note: DataScheduler moved to avoid circular imports
# Import directly from ml.data.scheduler when needed

# Data sources (mocks and implementations)
from ml.data.sources.calendar import MockCalendarSource
from ml.data.sources.calendar import PandasCalendarSource
from ml.data.sources.calendar import SimpleCalendarSource
from ml.data.sources.events import MockEventSource
from ml.data.sources.metadata import DatabentoMetadataSource
from ml.data.sources.metadata import NautilusMetadataSource
from ml.data.tft_dataset_builder import TFTDatasetBuilder


__all__ = [
    # Core data conversion utilities
    "BackoffPolicy",
    "BuildResult",
    "CacheableProvider",
    "DataCollector",
    "DataProvider",
    # "DataScheduler",  # Moved to avoid circular imports - import from ml.data.scheduler
    "DatabentoMetadataSource",
    "DatasetBuildConfig",
    "DomainWindowLoaderProtocol",
    "EventScheduleProvider",
    "FREDConfig",
    "FREDDataLoader",
    "FREDIndicator",
    "FixtureManifest",
    "IngestState",
    "IngestionOrchestrator",
    "InstrumentMetadataProvider",
    "L2MinuteCache",
    "MarketCalendarProvider",
    "MicroMinuteCache",
    "MockCalendarSource",
    "MockEventSource",
    "NautilusMetadataSource",
    "PandasCalendarSource",
    "RateLimiter",
    "SimpleCalendarSource",
    "StaticDataProvider",
    "TFTDatasetBuilder",
    "TimeSeriesProvider",
    "bars_to_dataframe",
    "build_tft_dataset",
    "compute_schema_hash",
    "make_mbp10_fixture",
    "make_tbbo_fixture",
    "make_trades_fixture",
    "quotes_to_dataframe",
    "trades_to_dataframe",
]


# Public dataset build facade (cold path)
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np


# (Avoid importing optional deps here; import lazily inside functions)


@dataclass(frozen=True)
class DatasetBuildConfig:
    data_dir: Path
    out_dir: Path
    symbols: list[str]
    # Feature options
    include_macro: bool = True
    macro_lag_days: int = 1
    include_micro: bool = False
    include_l2: bool = False
    # Builder params
    horizon_minutes: int = 15
    threshold: float = 0.001
    lookback_periods: int = 30
    # Optional window
    start: datetime | None = None
    end: datetime | None = None
    chunk_days: int = 0
    # Optional feature registration
    register_features: bool = False
    feature_registry_dir: Path | None = None
    feature_role: str = "teacher"  # teacher|student|inference_support


@dataclass(frozen=True)
class BuildResult:
    dataset_parquet: Path
    dataset_csv: Path
    features_npz: Path
    feature_names: list[str]
    feature_set_id: str | None = None


def _infer_feature_columns(df: Any) -> list[str]:
    """
    Infer numeric feature columns, excluding label/index/meta fields.
    """
    exclude = {"y", "time_index", "timestamp", "instrument_id", "ts_event"}
    from ml._imports import HAS_PANDAS
    from ml._imports import pd
    from ml._imports import pl

    if pl is not None and isinstance(df, pl.DataFrame):
        return [c for c in df.columns if df[c].dtype.is_numeric() and c not in exclude]
    if (
        HAS_PANDAS and pd is not None and isinstance(df, pd.DataFrame)
    ):  # pragma: no cover - alt path
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if c not in exclude]
    return []


def build_tft_dataset(cfg: DatasetBuildConfig) -> BuildResult:
    """
    Build a TFT dataset and persist artifacts under `cfg.out_dir`.
    """
    from ml._imports import check_ml_dependencies
    from ml._imports import pl

    if pl is None:
        check_ml_dependencies(["polars"])  # Guard for cold-path environment

    # Defer ParquetDataCatalog import to avoid import-time heavy deps here
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    catalog = ParquetDataCatalog(path=str(cfg.data_dir))

    builder = TFTDatasetBuilder(
        catalog=catalog,
        symbols=cfg.symbols,
        include_macro=cfg.include_macro,
        macro_lag_days=cfg.macro_lag_days,
        include_micro=cfg.include_micro,
        include_l2=cfg.include_l2,
        micro_base_dir=str(cfg.data_dir),
        l2_base_dir=str(cfg.data_dir),
    )

    # Optional chunked build (date slices)
    from ml._imports import HAS_POLARS
    from ml._imports import pl
    from ml.ml_types import PolarsDF

    assert HAS_POLARS and pl is not None
    df: PolarsDF
    if cfg.chunk_days > 0 and cfg.start and cfg.end:
        from ml.ml_types import PolarsDF

        dfs: list[PolarsDF] = []
        cursor = cfg.start
        while cursor < cfg.end:
            chunk_end = min(cursor + timedelta(days=cfg.chunk_days), cfg.end)
            dchunk = builder.build_training_dataset(
                horizon_minutes=cfg.horizon_minutes,
                min_return_threshold=cfg.threshold,
                lookback_periods=cfg.lookback_periods,
                use_polars=True,
                start=cursor,
                end=chunk_end,
            )
            if isinstance(dchunk, pl.DataFrame) and not dchunk.is_empty():
                dfs.append(dchunk)
            cursor = chunk_end
        df = pl.concat(dfs, how="vertical") if dfs else pl.DataFrame()
    else:
        df_any = builder.build_training_dataset(
            horizon_minutes=cfg.horizon_minutes,
            min_return_threshold=cfg.threshold,
            lookback_periods=cfg.lookback_periods,
            use_polars=True,
            start=cfg.start,
            end=cfg.end,
        )
        # Ensure Polars for consistent artifact writing
        if isinstance(df_any, pl.DataFrame):
            df = df_any
        else:  # pragma: no cover - fallback path
            from ml._imports import HAS_PANDAS
            from ml._imports import pd

            if not HAS_PANDAS:
                check_ml_dependencies(["pandas"])  # pragma: no cover
            assert pd is not None
            df = pl.from_pandas(df_any)

    # Persist dataset artifacts
    dataset_parquet = cfg.out_dir / "dataset.parquet"
    dataset_csv = cfg.out_dir / "dataset.csv"
    df.write_parquet(str(dataset_parquet))
    df.write_csv(str(dataset_csv))

    # Build feature matrix artifacts
    from ml._imports import HAS_PANDAS
    from ml._imports import pd

    if not HAS_PANDAS:
        check_ml_dependencies(["pandas"])  # pragma: no cover
    assert pd is not None
    df_pd = df.to_pandas()
    if "time_index" not in df_pd.columns:
        try:
            df_pd["time_index"] = np.arange(len(df_pd), dtype=np.int64)
        except Exception:  # pragma: no cover - defensive
            pass
    feature_names = _infer_feature_columns(df_pd)
    df_pd_sorted = df_pd.sort_values("time_index")
    cutoff = int(len(df_pd_sorted) * 0.8) if len(df_pd_sorted) > 0 else 0
    X = df_pd_sorted[feature_names].to_numpy(dtype=np.float32)
    X_train = X[:cutoff]
    X_val = X[cutoff:]
    features_npz = cfg.out_dir / "features_npz.npz"
    np.savez_compressed(
        features_npz,
        X_train=X_train,
        X_val=X_val,
        feature_names=np.array(feature_names),
    )

    # Optional feature registration
    feature_set_id: str | None = None
    if cfg.register_features:
        if cfg.feature_registry_dir is None:
            raise ValueError("feature_registry_dir is required when register_features=True")
        from ml.data.feature_manifest_export import FeatureExportConfig
        from ml.data.feature_manifest_export import export_feature_manifest
        from ml.registry.base import DataRequirements
        from ml.registry.feature_registry import FeatureRole

        role_map = {
            "teacher": FeatureRole.TEACHER,
            "student": FeatureRole.STUDENT,
            "inference_support": FeatureRole.INFERENCE_SUPPORT,
        }
        data_req = DataRequirements.L1_ONLY if not cfg.include_l2 else DataRequirements.L1_L2
        reg_cfg = FeatureExportConfig(
            registry_path=Path(cfg.feature_registry_dir),
            role=role_map.get(cfg.feature_role, FeatureRole.TEACHER),
            data_requirements=data_req,
        )
        flags = {
            "include_macro": cfg.include_macro,
            "include_micro": cfg.include_micro,
            "include_l2": cfg.include_l2,
            "horizon_minutes": cfg.horizon_minutes,
            "lookback_periods": cfg.lookback_periods,
        }
        feature_set_id = export_feature_manifest(
            feature_names=feature_names,
            feature_dtypes=["float32"] * len(feature_names),
            flags=flags,
            cfg=reg_cfg,
        )

    return BuildResult(
        dataset_parquet=dataset_parquet,
        dataset_csv=dataset_csv,
        features_npz=features_npz,
        feature_names=feature_names,
        feature_set_id=feature_set_id,
    )


def __getattr__(name: str) -> object:
    """
    Lazy import heavy or cycle-prone symbols.

    This avoids importing FRED loader modules at ml.data import time, preventing cycles
    when stores import ml.data.

    """
    if name in {"FREDConfig", "FREDDataLoader", "FREDIndicator"}:
        from ml.data.loaders import fred_loader as _fred

        return getattr(_fred, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
