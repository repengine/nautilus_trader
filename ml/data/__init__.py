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
from ml.config.targets import BinaryTargetConfig
from ml.config.targets import TargetHorizonSpec
from ml.config.targets import TargetSemanticsConfig
from ml.config.targets import decimal_to_bps
from ml.data import TFTDatasetBuilder

builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=["SPY", "QQQ", "AAPL"],
    include_macro=True,
    include_micro=True
)
dataset = builder.build_training_dataset(
    target_semantics=TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=15),),
        binary=BinaryTargetConfig(
            enabled=True,
            threshold_bps=decimal_to_bps(10.0),
            return_basis="raw",
        ),
    ),
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
from ml.config.targets import TargetSemanticsConfig

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

# Dataset build orchestration (cold path)
from ml.data.build import BuildResult
from ml.data.build import DatasetBuildConfig
from ml.data.build import FeatureRoleName
from ml.data.build import TFTDatasetTaskConfig
from ml.data.build import build_tft_dataset
from ml.data.build import build_tft_dataset_from_task_config
from ml.data.build import compute_dataset_pipeline_signature

# Core data conversion utilities
from ml.data.catalog_utils import bars_to_dataframe
from ml.data.catalog_utils import quotes_to_dataframe
from ml.data.catalog_utils import trades_to_dataframe

# High-level orchestrators and builders
from ml.data.collector import DataCollector
from ml.data.collectors import ProductionDataCollector
from ml.data.collectors import ProductionDatasetConfig
from ml.data.collectors import build_production_dataset

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
from ml.data.ingest.market_bindings import resolve_market_dataset_bindings  # noqa: F401
from ml.data.ingest.service import CostViolationError
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionError
from ml.data.ingest.service import IngestionRequest
from ml.data.ingest.service import IngestionWindow
from ml.data.ingest.service import SymbolIngestionSummary

# Note: DomainWindowLoaderProtocol and IngestionOrchestrator moved to avoid circular imports
# Import directly from ml.data.ingest.orchestrator when needed
# Performance and caching
from ml.data.l2_cache import L2MinuteCache
from ml.data.metadata import DatasetMetadata
from ml.data.metadata import DatasetMetadataExpectations
from ml.data.metadata import MarketBindingMetadata
from ml.data.metadata import build_metadata_expectations
from ml.data.metadata import load_dataset_metadata
from ml.data.metadata import require_reproducibility_metadata
from ml.data.metadata import require_target_column_in_semantics
from ml.data.metadata import require_target_semantics_contract
from ml.data.metadata import require_target_semantics_horizon_mode
from ml.data.metadata import require_target_semantics_metadata
from ml.data.metadata import resolve_target_col_from_metadata
from ml.data.metadata import validate_dataset_metadata_expectations

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
from ml.data.validation import DatasetReport
from ml.data.validation import DatasetReportConfig
from ml.data.validation import DatasetValidationConfig
from ml.data.validation import DatasetValidationError
from ml.data.validation import DatasetValidationResult
from ml.data.validation import generate_dataset_report
from ml.data.validation import validate_dataset
from ml.data.vintage import VintagePolicy  # noqa: F401


__all__ = [
    "ALFREDConfig",
    "ALFREDDataLoader",
    "BackoffPolicy",
    "BuildResult",
    "CacheableProvider",
    "CostViolationError",
    "DataCollector",
    "DataProvider",
    "DatabentoIngestionService",
    "DatabentoMetadataSource",
    "DatasetBuildConfig",
    "DatasetMetadata",
    "DatasetMetadataExpectations",
    "DatasetReport",
    "DatasetReportConfig",
    "DatasetValidationConfig",
    "DatasetValidationError",
    "DatasetValidationResult",
    "EventScheduleProvider",
    "FREDConfig",
    "FREDDataLoader",
    "FREDIndicator",
    "FeatureRoleName",
    "FileEventSource",
    "FixtureManifest",
    "IngestState",
    "IngestionChunk",
    "IngestionError",
    "IngestionRequest",
    "IngestionWindow",
    "InstrumentMetadataProvider",
    "L2MinuteCache",
    "MarketBindingMetadata",
    "MarketCalendarProvider",
    "MicroMinuteCache",
    "MockCalendarSource",
    "MockEventSource",
    "NautilusMetadataSource",
    "PandasCalendarSource",
    "ProductionDataCollector",
    "ProductionDatasetConfig",
    "RateLimiter",
    "SimpleCalendarSource",
    "StaticDataProvider",
    "SymbolIngestionSummary",
    "TFTDatasetBuilder",
    "TFTDatasetTaskConfig",
    "TimeSeriesProvider",
    "bars_to_dataframe",
    "build_metadata_expectations",
    "build_production_dataset",
    "build_tft_dataset",
    "build_tft_dataset_from_task_config",
    "compute_dataset_pipeline_signature",
    "compute_schema_hash",
    "generate_dataset_report",
    "load_dataset_metadata",
    "load_market_feed_descriptors",
    "make_mbp10_fixture",
    "make_tbbo_fixture",
    "make_trades_fixture",
    "quotes_to_dataframe",
    "require_reproducibility_metadata",
    "require_target_column_in_semantics",
    "require_target_semantics_contract",
    "require_target_semantics_horizon_mode",
    "require_target_semantics_metadata",
    "resolve_target_col_from_metadata",
    "trades_to_dataframe",
    "validate_dataset",
    "validate_dataset_metadata_expectations",
]


def __getattr__(name: str) -> object:
    """
    Lazy import heavy or cycle-prone symbols.

    This avoids importing FRED loader modules at ml.data import time, preventing cycles
    when stores import ml.data.

    """
    if name in {"FREDConfig", "FREDDataLoader", "FREDIndicator"}:
        from ml.data.loaders import fred_loader as _fred

        return getattr(_fred, name)
    if name in {"ALFREDConfig", "ALFREDDataLoader"}:
        from ml.data.loaders import alfred_loader as _alfred

        return getattr(_alfred, name)
    if name == "FileEventSource":
        from ml.data.sources import events as _events

        return getattr(_events, name)
    if name == "load_market_feed_descriptors":
        from ml.config import market_data as _market_data

        return getattr(_market_data, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
