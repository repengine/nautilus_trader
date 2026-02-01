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
from ml.config.targets import TargetSemanticsConfig
from ml.config.targets import build_binary_target_column

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

import hashlib
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pyarrow.parquet as pq
import structlog
from numpy.typing import NDArray

from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram
from ml.common.resource_monitor import current_rss_mb
from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import load_market_feed_descriptors
from ml.config.targets import TargetSemanticsConfig
from ml.config.targets import build_binary_target_column

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
from ml.data.ingest.market_bindings import MarketBindingStats
from ml.data.ingest.market_bindings import resolve_market_dataset_bindings
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
from ml.data.validation import DatasetValidationConfig
from ml.data.validation import DatasetValidationError
from ml.data.validation import DatasetValidationResult
from ml.data.validation import validate_dataset
from ml.data.vintage import VintagePolicy
from ml.data.vintage import format_dt
from ml.data.vintage import parse_dt
from ml.ml_types import PolarsDF
from ml.stores.protocols import DataStoreFacadeProtocol


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
    "DatasetValidationConfig",
    "DatasetValidationError",
    "DatasetValidationResult",
    "DomainWindowLoaderProtocol",
    "EventScheduleProvider",
    "FREDConfig",
    "FREDDataLoader",
    "FREDIndicator",
    "FileEventSource",
    "FixtureManifest",
    "IngestState",
    "IngestionChunk",
    "IngestionError",
    "IngestionJob",
    "IngestionOrchestrator",
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
    "RateLimiter",
    "SimpleCalendarSource",
    "StaticDataProvider",
    "SymbolIngestionSummary",
    "TFTDatasetBuilder",
    "TimeSeriesProvider",
    "bars_to_dataframe",
    "build_metadata_expectations",
    "build_tft_dataset",
    "compute_dataset_pipeline_signature",
    "compute_schema_hash",
    "ensure_service",
    "fetch_symbol_data",
    "load_dataset_metadata",
    "make_mbp10_fixture",
    "make_tbbo_fixture",
    "make_trades_fixture",
    "quotes_to_dataframe",
    "run_jobs",
    "trades_to_dataframe",
    "validate_dataset",
    "validate_dataset_metadata_expectations",
]


# Public dataset build facade (cold path)

FloatArray = NDArray[np.float32]


if TYPE_CHECKING:
    from ml.data.tft_dataset_builder import TFTDatasetBuilder

# (Avoid importing optional deps here; import lazily inside functions)

logger = structlog.get_logger(__name__)

_DATASET_BUILD_RSS_GAUGE = get_gauge(
    "ml_dataset_build_rss_mb",
    "Observed RSS (MB) during dataset build chunk stages.",
    labelnames=("stage",),
)
_DATASET_BUILD_CHUNK_SECONDS = get_histogram(
    "ml_dataset_build_chunk_seconds",
    "Elapsed seconds for dataset build chunk stages.",
    labelnames=("stage",),
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

DEFAULT_FRED_PARQUET_PATH = Path("data/features/macro/fred_indicators_ml_format.parquet")



@dataclass(slots=True)
class _ChunkMeta:
    path: Path
    rows: int
    positives: float
    macro_counts: dict[str, int]
    ts_start: datetime | None
    ts_end: datetime | None



def _derive_alfred_range(cfg: DatasetBuildConfig) -> tuple[str | None, str | None]:
    start_dt = getattr(cfg, "start", None)
    end_dt = getattr(cfg, "end", None)
    start_iso = getattr(cfg, "start_iso", None)
    end_iso = getattr(cfg, "end_iso", None)
    if start_dt is None and start_iso:
        start_dt = datetime.fromisoformat(start_iso)
    if end_dt is None and end_iso:
        end_dt = datetime.fromisoformat(end_iso)

    buffer = timedelta(days=30)
    if start_dt is not None:
        start_dt = start_dt - buffer
    if end_dt is not None:
        end_dt = end_dt + buffer

    today_utc = datetime.now(tz=UTC).date()

    def _normalize(dt: datetime | None) -> str | None:
        if dt is None:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC)
        date_value = dt.date()
        if date_value > today_utc:
            date_value = today_utc
        return date_value.isoformat()

    return _normalize(start_dt), _normalize(end_dt)


def _resolve_target_semantics(cfg: DatasetBuildConfig) -> TargetSemanticsConfig:
    """
    Resolve target semantics for a dataset build configuration.

    Args:
        cfg: Dataset build configuration.

    Returns:
        TargetSemanticsConfig instance.
    """
    target_semantics = cast(
        TargetSemanticsConfig | None,
        getattr(cfg, "target_semantics", None),
    )
    if target_semantics is not None:
        return target_semantics
    horizon_minutes = getattr(cfg, "horizon_minutes", 15)
    threshold = getattr(cfg, "threshold", 0.001)
    return TargetSemanticsConfig.from_legacy(
        horizon_minutes=horizon_minutes,
        threshold=threshold,
        legacy_aliases=True,
    )


def _resolve_binary_target_column(target_semantics: TargetSemanticsConfig) -> str | None:
    """
    Resolve the binary target column name for positive-rate checks.

    Args:
        target_semantics: Target semantics configuration.

    Returns:
        Binary target column name if available.
    """
    if not target_semantics.binary.enabled:
        return None
    primary = target_semantics.resolved_primary_target()
    if primary and primary.startswith("target_bin_"):
        return primary
    labels = target_semantics.horizon_labels
    if labels:
        return build_binary_target_column(labels[0])
    return None



@dataclass(frozen=True)
class DatasetBuildConfig:
    data_dir: Path
    out_dir: Path
    symbols: list[str]
    dataset_id: str = "tft_dataset"
    market_dataset_id: str | None = None
    market_inputs: tuple[MarketDatasetInput, ...] | None = None
    instrument_ids: list[str] | None = None
    # Feature options
    include_macro: bool = True
    macro_lag_days: int = 1
    include_micro: bool = False
    include_l2: bool = False
    include_events: bool = False
    include_calendar: bool = False
    include_earnings: bool = False
    earnings_lag_days: int = 1
    include_macro_deltas: bool = False
    include_calendar_lags: bool = False
    include_clustering_tags: bool = False
    include_context_features: bool = False
    fred_vintage_dir: Path | None = None
    events_base_dir: Path | None = None
    student_mode: bool = False
    micro_base_dir: Path | None = None
    l2_base_dir: Path | None = None
    # Builder params
    horizon_minutes: int = 15
    threshold: float = 0.001
    target_semantics: TargetSemanticsConfig | None = None
    lookback_periods: int = 30
    # Optional window
    start: datetime | None = None
    end: datetime | None = None
    chunk_days: int = 0
    # Output controls
    write_csv: bool | None = None
    csv_max_rows: int = 1_000_000
    csv_sample_rows: int = 0
    # Optional feature registration
    register_features: bool = False
    feature_registry_dir: Path | None = None
    feature_role: str = "teacher"  # teacher|student|inference_support
    # Optional lineage/events (cold path only)
    emit_dataset_events: bool = False
    # Macro refresh controls
    auto_refresh_macro: bool = True
    macro_staleness_hours: int = 24
    macro_series_ids: tuple[str, ...] | None = None
    macro_fred_path: Path | None = None
    vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME
    vintage_as_of: datetime | None = None
    include_macro_revisions: bool = False
    macro_revision_mode: Literal["minimal", "core", "full"] = "core"
    macro_revision_windows: tuple[int, ...] | None = None
    # Validation
    validation: DatasetValidationConfig | None = None


@dataclass(frozen=True)
class BuildResult:
    dataset_parquet: Path
    dataset_csv: Path
    features_npz: Path
    feature_names: list[str]
    feature_set_id: str | None = None
    metadata: DatasetMetadata | None = None


@dataclass(frozen=True)
class MarketBindingMetadata:
    binding_id: str
    dataset_id: str
    descriptor_id: str | None
    schema: str | None
    storage_kind: str | None
    symbols: tuple[str, ...]
    instrument_ids: tuple[str, ...]
    source: str
    license_start: str | None
    license_end: str | None
    ts_event_start: str | None
    ts_event_end: str | None
    rows_from_store: int
    rows_from_catalog: int
    source_datasets: tuple[str, ...] | None = None
    provider_dataset_id: str | None = None
    provider_schema: str | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str | None
    vintage_policy: VintagePolicy
    vintage_cutoff: str | None
    build_ts: str
    ts_event_start: str | None
    ts_event_end: str | None
    overall_window: tuple[str, str] | None
    train_window: tuple[str, str] | None
    validation_window: tuple[str, str] | None
    test_window: tuple[str, str] | None
    macro_observation_counts: dict[str, int]
    capability_flags: dict[str, bool] = field(default_factory=dict)
    market_bindings: tuple[MarketBindingMetadata, ...] | None = None
    target_semantics: dict[str, Any] | None = None


@dataclass(frozen=True)
class DatasetMetadataExpectations:
    dataset_id: str | None = None
    vintage_policy: VintagePolicy | None = None
    vintage_cutoff: str | None = None
    ts_event_start: str | None = None
    ts_event_end: str | None = None


def build_metadata_expectations(cfg: DatasetBuildConfig) -> DatasetMetadataExpectations:
    """Create metadata expectations derived from the dataset build configuration."""
    # Support both legacy datetime attributes (`start`/`end`) and the newer ISO
    # string fields (`start_iso`/`end_iso`).
    start_raw = getattr(cfg, "start_iso", None)
    end_raw = getattr(cfg, "end_iso", None)
    if not start_raw and hasattr(cfg, "start"):
        start_raw = getattr(cfg, "start")
    if not end_raw and hasattr(cfg, "end"):
        end_raw = getattr(cfg, "end")

    def _normalize(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        from datetime import datetime  # local import to avoid optional dependency at import time

        if isinstance(value, datetime):
            return format_dt(value)
        return str(value)

    start_iso = _normalize(start_raw)
    end_iso = _normalize(end_raw)
    cutoff_iso = _normalize(getattr(cfg, "vintage_as_of", None))
    return DatasetMetadataExpectations(
        dataset_id=cfg.dataset_id,
        vintage_policy=cfg.vintage_policy,
        vintage_cutoff=cutoff_iso,
        ts_event_start=start_iso,
        ts_event_end=end_iso,
    )


def _ns_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    dt_value = datetime.fromtimestamp(value / 1_000_000_000, tz=UTC)
    return format_dt(dt_value)


def _binding_stats_to_metadata(
    stats: Sequence[MarketBindingStats],
) -> tuple[MarketBindingMetadata, ...]:
    entries: list[MarketBindingMetadata] = []
    for stat in stats:
        storage_kind_value = stat.storage_kind.value if stat.storage_kind else None
        entries.append(
            MarketBindingMetadata(
                binding_id=stat.binding_id,
                dataset_id=stat.dataset_id,
                descriptor_id=stat.descriptor_id,
                schema=stat.schema,
                storage_kind=storage_kind_value,
                symbols=(stat.symbol,),
                instrument_ids=stat.instrument_ids,
                source=stat.source,
                license_start=stat.license_start,
                license_end=stat.license_end,
                ts_event_start=_ns_to_iso(stat.ts_event_start_ns),
                ts_event_end=_ns_to_iso(stat.ts_event_end_ns),
                rows_from_store=stat.rows_from_store,
                rows_from_catalog=stat.rows_from_catalog,
                source_datasets=tuple(sorted(stat.source_datasets)) if stat.source_datasets else None,
                provider_dataset_id=stat.provider_dataset_id,
                provider_schema=stat.provider_schema,
            ),
        )
    return tuple(entries)


def load_dataset_metadata(path: Path) -> DatasetMetadata:
    """Load dataset metadata from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        policy_raw = raw.get("vintage_policy", VintagePolicy.REAL_TIME.value)
        if not policy_raw:
            policy_token = VintagePolicy.REAL_TIME.value
        elif isinstance(policy_raw, VintagePolicy):
            policy_token = policy_raw.value
        else:
            policy_token = str(policy_raw).strip().lower()
        policy = VintagePolicy(policy_token)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid vintage_policy in metadata: {raw.get('vintage_policy')}") from exc

    def _as_tuple(value: object | None) -> tuple[str, str] | None:
        if not value:
            return None
        if isinstance(value, list | tuple) and len(value) == 2:
            return (str(value[0]), str(value[1]))
        raise ValueError(f"Metadata window must be length-2 sequence, got {value!r}")

    macro_counts_raw = raw.get("macro_observation_counts") or {}
    macro_counts: dict[str, int] = {
        str(key): int(value)
        for key, value in macro_counts_raw.items()
    }

    def _normalize_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"1", "true", "yes", "y", "t", "on"}:
                return True
            if token in {"0", "false", "no", "n", "f", "off", ""}:
                return False
        return bool(value)

    capability_raw = raw.get("capability_flags") or {}
    capability_flags: dict[str, bool] = {}
    if isinstance(capability_raw, dict):
        for key, value in capability_raw.items():
            capability_flags[str(key)] = _normalize_bool(value)

    target_semantics_raw = raw.get("target_semantics")
    target_semantics: dict[str, Any] | None = (
        target_semantics_raw if isinstance(target_semantics_raw, dict) else None
    )

    bindings_raw = raw.get("market_bindings")
    market_bindings: tuple[MarketBindingMetadata, ...] | None = None
    if isinstance(bindings_raw, list):
        converted: list[MarketBindingMetadata] = []
        for entry in bindings_raw:
            if not isinstance(entry, dict):
                continue
            symbols_field = entry.get("symbols")
            instrument_field = entry.get("instrument_ids")
            symbols_tuple = (
                tuple(str(item) for item in symbols_field)
                if isinstance(symbols_field, list | tuple)
                else ()
            )
            instruments_tuple = (
                tuple(str(item) for item in instrument_field)
                if isinstance(instrument_field, list | tuple)
                else ()
            )
            converted.append(
                MarketBindingMetadata(
                    binding_id=str(entry.get("binding_id")),
                    dataset_id=str(entry.get("dataset_id")),
                    descriptor_id=(str(entry.get("descriptor_id")) if entry.get("descriptor_id") is not None else None),
                    schema=(str(entry.get("schema")) if entry.get("schema") is not None else None),
                    storage_kind=(str(entry.get("storage_kind")) if entry.get("storage_kind") is not None else None),
                    symbols=symbols_tuple,
                    instrument_ids=instruments_tuple,
                    source=str(entry.get("source", "")),
                    license_start=(str(entry.get("license_start")) if entry.get("license_start") is not None else None),
                    license_end=(str(entry.get("license_end")) if entry.get("license_end") is not None else None),
                    ts_event_start=(str(entry.get("ts_event_start")) if entry.get("ts_event_start") is not None else None),
                    ts_event_end=(str(entry.get("ts_event_end")) if entry.get("ts_event_end") is not None else None),
                    rows_from_store=int(entry.get("rows_from_store", 0) or 0),
                    rows_from_catalog=int(entry.get("rows_from_catalog", 0) or 0),
                    provider_dataset_id=(
                        str(entry.get("provider_dataset_id"))
                        if entry.get("provider_dataset_id") is not None
                        else None
                    ),
                    provider_schema=(
                        str(entry.get("provider_schema"))
                        if entry.get("provider_schema") is not None
                        else None
                    ),
                ),
            )
        market_bindings = tuple(converted)

    return DatasetMetadata(
        dataset_id=str(raw.get("dataset_id")) if raw.get("dataset_id") else None,
        vintage_policy=policy,
        vintage_cutoff=str(raw.get("vintage_cutoff")) if raw.get("vintage_cutoff") else None,
        build_ts=str(raw.get("build_ts", "")),
        ts_event_start=str(raw.get("ts_event_start")) if raw.get("ts_event_start") else None,
        ts_event_end=str(raw.get("ts_event_end")) if raw.get("ts_event_end") else None,
        overall_window=_as_tuple(raw.get("overall_window")),
        train_window=_as_tuple(raw.get("train_window")),
        validation_window=_as_tuple(raw.get("validation_window")),
        test_window=_as_tuple(raw.get("test_window")),
        macro_observation_counts=macro_counts,
        capability_flags=capability_flags,
        market_bindings=market_bindings,
        target_semantics=target_semantics,
    )


def validate_dataset_metadata_expectations(
    metadata: DatasetMetadata,
    expectations: DatasetMetadataExpectations,
    *,
    context: str | None = None,
) -> None:
    """Validate that metadata satisfies the supplied expectations."""
    prefix = f"{context}: " if context else ""

    if expectations.dataset_id and metadata.dataset_id != expectations.dataset_id:
        raise ValueError(
            f"{prefix}dataset_id mismatch (expected {expectations.dataset_id}, got {metadata.dataset_id})",
        )

    if expectations.vintage_policy and metadata.vintage_policy is not expectations.vintage_policy:
        raise ValueError(
            f"{prefix}vintage_policy mismatch (expected {expectations.vintage_policy.value}, got {metadata.vintage_policy.value})",
        )

    if expectations.vintage_cutoff is not None:
        expected_cutoff = expectations.vintage_cutoff
        actual_cutoff = metadata.vintage_cutoff or ""
        if actual_cutoff != expected_cutoff:
            raise ValueError(
                f"{prefix}vintage_cutoff mismatch (expected {expected_cutoff}, got {actual_cutoff or 'None'})",
            )

    def _ensure_bounds(
        label: str,
        expected: str | None,
        actual: str | None,
        *,
        comparator: str,
    ) -> None:
        if not expected or not actual:
            return
        expected_dt = parse_dt(expected)
        actual_dt = parse_dt(actual)
        if expected_dt is None or actual_dt is None:
            return
        if comparator == "gte" and actual_dt < expected_dt:
            raise ValueError(
                f"{prefix}{label} {actual} earlier than expected {expected}",
            )
        if comparator == "lte" and actual_dt > expected_dt:
            raise ValueError(
                f"{prefix}{label} {actual} later than expected {expected}",
            )

    _ensure_bounds("ts_event_start", expectations.ts_event_start, metadata.ts_event_start, comparator="gte")
    _ensure_bounds("ts_event_end", expectations.ts_event_end, metadata.ts_event_end, comparator="lte")


def _sorted_tuple(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(sorted(str(item) for item in values))


def _sorted_market_bindings(
    bindings: Sequence[MarketBindingMetadata] | None,
) -> tuple[tuple[str, ...], ...]:
    if not bindings:
        return ()
    summary: list[tuple[str, ...]] = []
    for binding in bindings:
        summary.append(
            (
                binding.dataset_id,
                binding.descriptor_id or "",
                ",".join(sorted(binding.symbols)),
                ",".join(sorted(binding.instrument_ids)),
                binding.source,
                binding.storage_kind or "",
            ),
        )
    summary.sort()
    return tuple(summary)


def compute_dataset_pipeline_signature(
    *,
    dataset_id: str | None,
    symbols: Sequence[str],
    instrument_ids: Sequence[str] | None,
    macro_series_ids: Sequence[str] | None,
    include_macro: bool,
    macro_lag_days: int,
    vintage_policy: VintagePolicy,
    vintage_cutoff: str | None,
    ts_event_start: str | None,
    ts_event_end: str | None,
    market_bindings: Sequence[MarketBindingMetadata] | None = None,
) -> str:
    """Compute a deterministic pipeline signature describing dataset lineage."""
    payload = {
        "dataset_id": dataset_id,
        "symbols": _sorted_tuple(symbols),
        "instrument_ids": _sorted_tuple(instrument_ids),
        "macro_series_ids": _sorted_tuple(macro_series_ids),
        "include_macro": bool(include_macro),
        "macro_lag_days": int(macro_lag_days),
        "vintage_policy": vintage_policy.value,
        "vintage_cutoff": vintage_cutoff,
        "ts_event_start": ts_event_start,
        "ts_event_end": ts_event_end,
        "market_bindings": _sorted_market_bindings(market_bindings),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"tft_pipeline:{digest[:16]}"


def _infer_feature_columns(df: Any) -> list[str]:
    """
    Infer numeric feature columns, excluding label/index/meta fields.
    """
    from ml.data.feature_columns import infer_numeric_feature_columns

    return infer_numeric_feature_columns(df)


def _resolve_write_csv(cfg: DatasetBuildConfig, row_count: int) -> bool:
    """
    Determine whether to write the full dataset CSV for the given row count.
    """
    if cfg.write_csv is not None:
        return bool(cfg.write_csv)
    max_rows = max(int(cfg.csv_max_rows), 0)
    return row_count <= max_rows


def _write_dataset_csv(
    df_sorted: PolarsDF,
    cfg: DatasetBuildConfig,
    *,
    dataset_csv: Path,
) -> Path | None:
    """
    Write dataset CSV output or a sample CSV when configured.

    Returns the written CSV path, or None when no CSV is emitted.
    """
    row_count = int(df_sorted.height)
    write_full = _resolve_write_csv(cfg, row_count)
    sample_rows = max(int(cfg.csv_sample_rows), 0)

    if write_full:
        df_sorted.write_csv(str(dataset_csv))
        return dataset_csv

    if dataset_csv.exists():
        dataset_csv.unlink()

    if sample_rows <= 0:
        return None

    sample_path = dataset_csv.with_name("dataset_sample.csv")
    sample_df = df_sorted.head(sample_rows)
    sample_df.write_csv(str(sample_path))
    return sample_path


def _write_feature_npz_from_polars(
    df_sorted: PolarsDF,
    feature_names: Sequence[str],
    *,
    out_path: Path,
    cutoff: int,
    chunk_size: int = 200_000,
) -> None:
    """Persist feature matrices to an ``.npz`` without loading all rows into memory."""
    from ml._imports import pl

    if pl is None:
        msg = "Polars is required to write feature matrices"
        raise RuntimeError(msg)
    if not isinstance(df_sorted, pl.DataFrame):
        msg = "df_sorted must be a Polars DataFrame"
        raise TypeError(msg)

    num_rows = df_sorted.height
    num_features = len(feature_names)
    if num_rows == 0 or num_features == 0:
        np.savez(
            out_path,
            X_train=np.empty((0, num_features), dtype=np.float32),
            X_val=np.empty((0, num_features), dtype=np.float32),
            feature_names=np.array(feature_names, dtype=np.str_),
        )
        return

    chunk_size = max(int(chunk_size), 1)
    train_rows = int(min(max(cutoff, 0), num_rows))
    val_rows = int(max(num_rows - train_rows, 0))

    train_tmp = out_path.with_name(out_path.name + ".train.tmp") if train_rows > 0 else None
    val_tmp = out_path.with_name(out_path.name + ".val.tmp") if val_rows > 0 else None

    train_mem: np.memmap[Any, np.dtype[np.float32]] | None = None
    val_mem: np.memmap[Any, np.dtype[np.float32]] | None = None
    if train_tmp is not None and num_features > 0:
        train_mem = np.memmap(
            str(train_tmp),
            dtype=np.float32,
            mode="w+",
            shape=(train_rows, num_features),
        )
    if val_tmp is not None and num_features > 0:
        val_mem = np.memmap(
            str(val_tmp),
            dtype=np.float32,
            mode="w+",
            shape=(val_rows, num_features),
        )

    train_pos = 0
    val_pos = 0
    feature_expr = [pl.col(name).cast(pl.Float32) for name in feature_names]

    row_start = 0
    while row_start < num_rows:
        length = min(chunk_size, num_rows - row_start)
        if feature_expr:
            chunk = df_sorted.slice(row_start, length).select(feature_expr)
            chunk_np = chunk.to_numpy()
        else:
            chunk_np = np.empty((length, 0), dtype=np.float32)

        row_end = row_start + length
        if train_mem is not None and row_start < train_rows:
            train_end = min(row_end, train_rows)
            train_len = train_end - row_start
            if train_len > 0:
                train_mem[train_pos : train_pos + train_len] = chunk_np[:train_len]
                train_pos += train_len
        if val_mem is not None and row_end > train_rows:
            val_chunk_start = max(train_rows - row_start, 0)
            val_len = row_end - max(row_start, train_rows)
            if val_len > 0:
                val_mem[val_pos : val_pos + val_len] = chunk_np[val_chunk_start : val_chunk_start + val_len]
                val_pos += val_len

        row_start = row_end

    X_train: NDArray[np.float32] | np.memmap[Any, np.dtype[np.float32]]
    X_val: NDArray[np.float32] | np.memmap[Any, np.dtype[np.float32]]
    if train_mem is not None:
        assert train_tmp is not None
        train_mem.flush()
        X_train = np.memmap(
            str(train_tmp),
            dtype=np.float32,
            mode="r",
            shape=(train_rows, num_features),
        )
    else:
        X_train = np.empty((0, num_features), dtype=np.float32)
    if val_mem is not None:
        assert val_tmp is not None
        val_mem.flush()
        X_val = np.memmap(
            str(val_tmp),
            dtype=np.float32,
            mode="r",
            shape=(val_rows, num_features),
        )
    else:
        X_val = np.empty((0, num_features), dtype=np.float32)

    try:
        np.savez(
            out_path,
            X_train=X_train,
            X_val=X_val,
            feature_names=np.array(feature_names, dtype=np.str_),
        )
    finally:
        if isinstance(X_train, np.memmap):
            del X_train
        if isinstance(X_val, np.memmap):
            del X_val
        if train_tmp is not None and train_tmp.exists():
            train_tmp.unlink()
        if val_tmp is not None and val_tmp.exists():
            val_tmp.unlink()


@dataclass(slots=True)
class _StreamingFeatureWriter:
    """Incrementally write feature matrices to NPZ using memory-mapped buffers."""

    out_path: Path
    feature_names: list[str]
    total_rows: int
    cutoff: int
    _train_tmp: Path | None = None
    _val_tmp: Path | None = None
    _train_mem: np.memmap[Any, np.dtype[np.float32]] | None = None
    _val_mem: np.memmap[Any, np.dtype[np.float32]] | None = None
    _train_rows: int = 0
    _val_rows: int = 0
    _train_cursor: int = 0
    _val_cursor: int = 0

    def __post_init__(self) -> None:
        if self.total_rows < 0:
            msg = "total_rows must be non-negative"
            raise ValueError(msg)
        feature_dim = len(self.feature_names)
        self._train_rows = max(min(self.cutoff, self.total_rows), 0)
        self._val_rows = max(self.total_rows - self._train_rows, 0)
        if feature_dim == 0:
            return
        if self._train_rows > 0:
            self._train_tmp = self.out_path.with_name(self.out_path.name + ".train.tmp")
            self._train_mem = np.memmap(
                str(self._train_tmp),
                dtype=np.float32,
                mode="w+",
                shape=(self._train_rows, feature_dim),
            )
        if self._val_rows > 0:
            self._val_tmp = self.out_path.with_name(self.out_path.name + ".val.tmp")
            self._val_mem = np.memmap(
                str(self._val_tmp),
                dtype=np.float32,
                mode="w+",
                shape=(self._val_rows, feature_dim),
            )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def append(self, values: FloatArray, *, global_offset: int) -> None:
        if self.feature_dim == 0:
            return
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            msg = "Chunk feature matrix has unexpected shape"
            raise ValueError(msg)
        rows = values.shape[0]
        if rows == 0:
            return
        train_remaining = max(self._train_rows - global_offset, 0)
        train_take = min(train_remaining, rows)
        if train_take > 0 and self._train_mem is not None:
            self._train_mem[self._train_cursor : self._train_cursor + train_take] = values[:train_take]
            self._train_cursor += train_take
        val_take = rows - train_take
        if val_take > 0 and self._val_mem is not None:
            self._val_mem[self._val_cursor : self._val_cursor + val_take] = values[train_take:]
            self._val_cursor += val_take

    def finalize(self) -> None:
        if self.feature_dim == 0:
            empty = cast(FloatArray, np.empty((0, 0), dtype=np.float32))
            np.savez(
                self.out_path,
                X_train=empty,
                X_val=empty,
                feature_names=np.array(self.feature_names, dtype=np.str_),
            )
            return
        if self._train_mem is not None:
            self._train_mem.flush()
        if self._val_mem is not None:
            self._val_mem.flush()
        X_train: FloatArray
        X_val: FloatArray
        if self._train_tmp is not None and self._train_tmp.exists():
            X_train = cast(
                FloatArray,
                np.memmap(
                    str(self._train_tmp),
                    dtype=np.float32,
                    mode="r",
                    shape=(self._train_rows, self.feature_dim),
                ),
            )
        else:
            X_train = cast(FloatArray, np.empty((0, self.feature_dim), dtype=np.float32))
        if self._val_tmp is not None and self._val_tmp.exists():
            X_val = cast(
                FloatArray,
                np.memmap(
                    str(self._val_tmp),
                    dtype=np.float32,
                    mode="r",
                    shape=(self._val_rows, self.feature_dim),
                ),
            )
        else:
            X_val = cast(FloatArray, np.empty((0, self.feature_dim), dtype=np.float32))
        try:
            np.savez(
                self.out_path,
                X_train=X_train,
                X_val=X_val,
                feature_names=np.array(self.feature_names, dtype=np.str_),
            )
        finally:
            if isinstance(X_train, np.memmap):
                del X_train
            if isinstance(X_val, np.memmap):
                del X_val
            if self._train_tmp is not None and self._train_tmp.exists():
                self._train_tmp.unlink()
            if self._val_tmp is not None and self._val_tmp.exists():
                self._val_tmp.unlink()


def _validate_aggregated_dataset(
    *,
    config: DatasetValidationConfig,
    total_rows: int,
    positive_rate: float | None,
    feature_coverage: dict[str, float],
    macro_counts: dict[str, int],
) -> DatasetValidationResult:
    if total_rows < config.min_rows:
        msg = f"Dataset has {total_rows} rows; minimum required is {config.min_rows}"
        raise DatasetValidationError(msg)

    if positive_rate is not None and config.min_positive_rate is not None:
        if positive_rate < config.min_positive_rate:
            msg = (
                f"Target positive rate {positive_rate:.4f} below minimum "
                f"{config.min_positive_rate:.4f}"
            )
            raise DatasetValidationError(msg)
    if positive_rate is not None and config.max_positive_rate is not None:
        if positive_rate > config.max_positive_rate:
            msg = (
                f"Target positive rate {positive_rate:.4f} above maximum "
                f"{config.max_positive_rate:.4f}"
            )
            raise DatasetValidationError(msg)

    if config.min_feature_coverage is not None:
        low_coverage = [
            (name, ratio)
            for name, ratio in feature_coverage.items()
            if ratio < config.min_feature_coverage
        ]
        if low_coverage:
            worst_low_cov = min(low_coverage, key=lambda item: item[1])
            msg = (
                "Feature coverage below acceptance threshold; "
                f"example: {worst_low_cov[0]}={worst_low_cov[1]:.3f} < {config.min_feature_coverage:.3f}"
            )
            raise DatasetValidationError(msg)

    macro_present: tuple[str, ...] = ()
    if config.require_macro_series:
        required = set(config.require_macro_series)
        actual = {name for name, count in macro_counts.items() if count > 0}
        missing = required - actual
        if missing:
            msg = f"Missing macro series: {sorted(missing)}"
            raise DatasetValidationError(msg)
        macro_present = tuple(sorted(actual))
        min_obs = config.macro_min_vintage_observations
        policy = config.expected_vintage_policy or VintagePolicy.REAL_TIME
        if min_obs is not None and policy is VintagePolicy.REAL_TIME:
            failing = [name for name in required if macro_counts.get(name, 0) < min_obs]
            if failing:
                weakest_series = min(failing, key=lambda name: macro_counts.get(name, 0))
                msg = (
                    "Macro vintage coverage below threshold; "
                    f"series {weakest_series} has {macro_counts.get(weakest_series, 0)} observations < {min_obs}"
                )
                raise DatasetValidationError(msg)
    else:
        macro_present = tuple(sorted(macro_counts.keys()))

    return DatasetValidationResult(
        row_count=total_rows,
        positive_rate=positive_rate,
        feature_coverage=feature_coverage,
        macro_columns_present=macro_present,
        macro_observation_counts=macro_counts,
    )


def _build_dataset_chunked(
    *,
    builder: TFTDatasetBuilder,
    cfg: DatasetBuildConfig,
    vintage_as_of: datetime | None,
    build_ts: datetime,
    target_semantics: TargetSemanticsConfig,
) -> tuple[BuildResult, DatasetValidationResult]:
    from ml._imports import HAS_POLARS
    from ml._imports import check_ml_dependencies
    from ml._imports import pl

    polars_module = pl
    if not HAS_POLARS or polars_module is None:
        check_ml_dependencies(["polars"])
        import polars as polars_module_import

        polars_module = polars_module_import
    assert polars_module is not None

    def _update_rss_peak(current_peak: float | None) -> float | None:
        rss_mb = current_rss_mb()
        if rss_mb is None:
            return current_peak
        if current_peak is None or rss_mb > current_peak:
            return rss_mb
        return current_peak

    chunk_dir = cfg.out_dir / ".chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    capability_flags = _capability_flags_from_builder(builder)

    chunk_metas: list[_ChunkMeta] = []
    binary_target_col = _resolve_binary_target_column(target_semantics)
    from ml.training.datasets.target_generator import build_target_semantics_metadata

    target_semantics_metadata = build_target_semantics_metadata(target_semantics)

    cursor = cast(datetime, cfg.start)
    end = cast(datetime, cfg.end)
    chunk_index = 0
    chunk_delta = timedelta(days=cfg.chunk_days)

    while cursor < end:
        chunk_started = time.perf_counter()
        chunk_peak_rss: float | None = None
        chunk_peak_rss = _update_rss_peak(chunk_peak_rss)
        chunk_end = min(cursor + chunk_delta, end)
        df_any = builder.build_training_dataset(
            horizon_minutes=cfg.horizon_minutes,
            min_return_threshold=cfg.threshold,
            target_semantics=target_semantics,
            lookback_periods=cfg.lookback_periods,
            use_polars=True,
            start=cursor,
            end=chunk_end,
        )
        if isinstance(df_any, polars_module.DataFrame):
            df_chunk = df_any
        else:
            from ml._imports import HAS_PANDAS
            from ml._imports import pd

            if not HAS_PANDAS:
                check_ml_dependencies(["pandas"])  # pragma: no cover
            assert pd is not None
            df_chunk = polars_module.from_pandas(df_any)
        chunk_peak_rss = _update_rss_peak(chunk_peak_rss)

        if not df_chunk.is_empty():
            chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.parquet"
            if "timestamp" in df_chunk.columns:
                df_chunk = df_chunk.sort(
                    ["timestamp", "instrument_id"] if "instrument_id" in df_chunk.columns else ["timestamp"],
                )
            elif "time_index" in df_chunk.columns:
                df_chunk = df_chunk.sort("time_index")
            df_chunk.write_parquet(str(chunk_path))

            positives = 0.0
            if binary_target_col and binary_target_col in df_chunk.columns:
                positives = float(df_chunk[binary_target_col].sum())

            macro_counts: dict[str, int] = {}
            if cfg.macro_series_ids:
                for macro in cfg.macro_series_ids:
                    value_col = f"{macro}__value_vintage_ts"
                    if value_col in df_chunk.columns:
                        count_val = int(df_chunk[value_col].is_not_null().sum())
                        macro_counts[macro] = count_val
            ts_start = None
            ts_end = None
            if "timestamp" in df_chunk.columns and df_chunk.height > 0:
                ts_series = df_chunk["timestamp"]
                ts_start = _ensure_datetime(
                    cast(datetime | float | None, ts_series.min()),
                )
                ts_end = _ensure_datetime(
                    cast(datetime | float | None, ts_series.max()),
                )

            chunk_metas.append(
                _ChunkMeta(
                    path=chunk_path,
                    rows=df_chunk.height,
                    positives=positives,
                    macro_counts=macro_counts,
                    ts_start=ts_start,
                    ts_end=ts_end,
                ),
            )

        cursor = chunk_end
        chunk_index += 1
        chunk_peak_rss = _update_rss_peak(chunk_peak_rss)
        _DATASET_BUILD_CHUNK_SECONDS.labels(stage="build").observe(
            time.perf_counter() - chunk_started,
        )
        if chunk_peak_rss is not None:
            _DATASET_BUILD_RSS_GAUGE.labels(stage="build").set(chunk_peak_rss)

    binding_metadata = _binding_stats_to_metadata(builder.get_binding_stats())

    if not chunk_metas:
        dataset_parquet = cfg.out_dir / "dataset.parquet"
        dataset_csv = cfg.out_dir / "dataset.csv"
        features_npz = cfg.out_dir / "features_npz.npz"
        if dataset_parquet.exists():
            dataset_parquet.unlink()
        if dataset_csv.exists():
            dataset_csv.unlink()
        empty_df = polars_module.DataFrame()
        empty_df.write_parquet(str(dataset_parquet))
        _write_dataset_csv(empty_df, cfg, dataset_csv=dataset_csv)
        np.savez(
            features_npz,
            X_train=np.empty((0, 0), dtype=np.float32),
            X_val=np.empty((0, 0), dtype=np.float32),
            feature_names=np.array([], dtype=np.str_),
        )
        metadata = DatasetMetadata(
            dataset_id=cfg.dataset_id,
            vintage_policy=cfg.vintage_policy,
            vintage_cutoff=format_dt(vintage_as_of) if vintage_as_of else None,
            build_ts=format_dt(build_ts) or build_ts.isoformat(),
            ts_event_start=None,
            ts_event_end=None,
            overall_window=None,
            train_window=None,
            validation_window=None,
            test_window=None,
            macro_observation_counts={},
            capability_flags=capability_flags,
            market_bindings=binding_metadata,
            target_semantics=target_semantics_metadata,
        )
        metadata_path = cfg.out_dir / "dataset_metadata.json"
        metadata_path.write_text(json.dumps(_metadata_to_dict(metadata), indent=2), encoding="utf-8")
        validation_result = DatasetValidationResult(
            row_count=0,
            positive_rate=None,
            feature_coverage={},
            macro_columns_present=(),
            macro_observation_counts={},
        )
        shutil.rmtree(chunk_dir, ignore_errors=True)
        return (
            BuildResult(
                dataset_parquet=dataset_parquet,
                dataset_csv=dataset_csv,
                features_npz=features_npz,
                feature_names=[],
                feature_set_id=None,
                metadata=metadata,
            ),
            validation_result,
        )

    total_rows = sum(meta.rows for meta in chunk_metas)
    positive_sum = sum(meta.positives for meta in chunk_metas)
    macro_totals: dict[str, int] = {}
    for meta in chunk_metas:
        for macro, count in meta.macro_counts.items():
            macro_totals[macro] = macro_totals.get(macro, 0) + count
    if cfg.macro_series_ids:
        for macro in cfg.macro_series_ids:
            macro_totals.setdefault(macro, 0)
    overall_start = min((meta.ts_start for meta in chunk_metas if meta.ts_start is not None), default=None)
    overall_end = max((meta.ts_end for meta in chunk_metas if meta.ts_end is not None), default=None)

    dataset_parquet = cfg.out_dir / "dataset.parquet"
    dataset_csv = cfg.out_dir / "dataset.csv"
    features_npz = cfg.out_dir / "features_npz.npz"

    if dataset_parquet.exists():
        dataset_parquet.unlink()
    if dataset_csv.exists():
        dataset_csv.unlink()
    if features_npz.exists():
        features_npz.unlink()
    write_csv = _resolve_write_csv(cfg, total_rows)
    sample_rows = max(int(cfg.csv_sample_rows), 0)
    sample_path = dataset_csv.with_name("dataset_sample.csv")
    if not write_csv and sample_path.exists():
        sample_path.unlink()

    validation_cfg = cfg.validation or DatasetValidationConfig(require_macro_series=cfg.macro_series_ids)
    if validation_cfg.require_macro_series is None and cfg.macro_series_ids:
        validation_cfg = replace(validation_cfg, require_macro_series=cfg.macro_series_ids)
    if validation_cfg.expected_vintage_policy is None:
        validation_cfg = replace(validation_cfg, expected_vintage_policy=cfg.vintage_policy)
    if (
        validation_cfg.macro_min_vintage_observations is None
        and cfg.include_macro
        and cfg.vintage_policy is VintagePolicy.REAL_TIME
        and cfg.macro_series_ids
    ):
        validation_cfg = replace(validation_cfg, macro_min_vintage_observations=1)
    if cfg.vintage_policy is not VintagePolicy.REAL_TIME and validation_cfg.macro_min_vintage_observations is not None:
        validation_cfg = replace(validation_cfg, macro_min_vintage_observations=None)

    pq_writer: pq.ParquetWriter | None = None
    csv_header_written = False
    sample_header_written = False
    sample_remaining = sample_rows
    feature_names: list[str] | None = None
    non_null_counts: dict[str, int] | None = None
    feature_writer: _StreamingFeatureWriter | None = None
    offset = 0
    train_rows = int(total_rows * 0.8)
    train_window_end: datetime | None = None
    validation_window_start: datetime | None = None

    def _column_to_float32(column: Any) -> FloatArray:
        values = column.to_numpy(zero_copy_only=False)
        if isinstance(values, np.ma.MaskedArray):
            filled = np.where(values.mask, np.nan, values.data)
            return cast(FloatArray, np.asarray(filled, dtype=np.float32))
        return cast(FloatArray, np.asarray(values, dtype=np.float32))

    for meta in chunk_metas:
        merge_started = time.perf_counter()
        merge_peak_rss: float | None = None
        merge_peak_rss = _update_rss_peak(merge_peak_rss)
        parquet_file = pq.ParquetFile(str(meta.path))
        for batch in parquet_file.iter_batches():
            batch_rows = batch.num_rows
            if batch_rows == 0:
                continue
            df_batch = polars_module.from_arrow(batch)
            if df_batch.is_empty():
                continue
            df_batch = df_batch.with_columns(
                polars_module.arange(offset, offset + df_batch.height, eager=True).alias("time_index"),
            )
            df_batch = builder._add_known_future_features_polars(df_batch)
            merge_peak_rss = _update_rss_peak(merge_peak_rss)

            if feature_names is None:
                feature_names = _infer_feature_columns(df_batch)
                feature_writer = _StreamingFeatureWriter(
                    out_path=features_npz,
                    feature_names=feature_names,
                    total_rows=total_rows,
                    cutoff=train_rows,
                )
                non_null_counts = dict.fromkeys(feature_names, 0)

            if train_rows > 0 and train_window_end is None and offset <= train_rows - 1 < offset + df_batch.height:
                idx = train_rows - 1 - offset
                if 0 <= idx < df_batch.height and "timestamp" in df_batch.columns:
                    train_window_end = cast(datetime, df_batch[idx, "timestamp"])
            if validation_window_start is None and offset <= train_rows < offset + df_batch.height:
                idx_val = train_rows - offset
                if 0 <= idx_val < df_batch.height and "timestamp" in df_batch.columns:
                    validation_window_start = cast(datetime, df_batch[idx_val, "timestamp"])

            table = df_batch.to_arrow()
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(str(dataset_parquet), table.schema, compression="zstd")

            if feature_names:
                assert non_null_counts is not None
                batch_values: list[FloatArray] = []
                for name in feature_names:
                    try:
                        column = table.column(name)
                    except Exception:
                        column = None
                    if column is None:
                        batch_values.append(cast(FloatArray, np.zeros(batch_rows, dtype=np.float32)))
                        continue
                    non_null_counts[name] += int(batch_rows - column.null_count)
                    batch_values.append(_column_to_float32(column))
                if feature_writer is not None:
                    values = cast(FloatArray, np.column_stack(batch_values))
                    feature_writer.append(
                        values,
                        global_offset=offset,
                    )

            pq_writer.write_table(table)

            if write_csv:
                mode = "w" if not csv_header_written else "a"
                with open(dataset_csv, mode, newline="") as csv_handle:
                    df_batch.write_csv(csv_handle, include_header=not csv_header_written)
                csv_header_written = True
            elif sample_remaining > 0:
                sample_df = df_batch.head(sample_remaining)
                if not sample_df.is_empty():
                    mode = "w" if not sample_header_written else "a"
                    with open(sample_path, mode, newline="") as csv_handle:
                        sample_df.write_csv(csv_handle, include_header=not sample_header_written)
                    sample_remaining -= sample_df.height
                    sample_header_written = True

            offset += df_batch.height
            merge_peak_rss = _update_rss_peak(merge_peak_rss)

        meta.path.unlink(missing_ok=True)
        _DATASET_BUILD_CHUNK_SECONDS.labels(stage="merge").observe(
            time.perf_counter() - merge_started,
        )
        if merge_peak_rss is not None:
            _DATASET_BUILD_RSS_GAUGE.labels(stage="merge").set(merge_peak_rss)

    if pq_writer is not None:
        pq_writer.close()
    if feature_writer is not None:
        feature_writer.finalize()

    shutil.rmtree(chunk_dir, ignore_errors=True)

    positive_rate = (
        positive_sum / total_rows if total_rows and binary_target_col is not None else None
    )
    if feature_names and non_null_counts is not None and total_rows > 0:
        feature_coverage = {
            name: non_null_counts[name] / total_rows
            for name in feature_names
        }
    elif feature_names and non_null_counts is not None:
        feature_coverage = dict.fromkeys(feature_names, 0.0)
    else:
        feature_coverage = {}

    validation_result = _validate_aggregated_dataset(
        config=validation_cfg,
        total_rows=total_rows,
        positive_rate=positive_rate,
        feature_coverage=feature_coverage,
        macro_counts=macro_totals,
    )

    logger.info(
        "Dataset validation succeeded",
        rows=validation_result.row_count,
        positive_rate=validation_result.positive_rate,
    )

    dataset_id = cfg.dataset_id

    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        vintage_policy=cfg.vintage_policy,
        vintage_cutoff=format_dt(vintage_as_of) if vintage_as_of else None,
        build_ts=format_dt(build_ts) or build_ts.isoformat(),
        ts_event_start=format_dt(overall_start) if overall_start else None,
        ts_event_end=format_dt(overall_end) if overall_end else None,
        overall_window=_format_window(overall_start, overall_end),
        train_window=_format_window(overall_start, train_window_end),
        validation_window=_format_window(validation_window_start, overall_end),
        test_window=None,
        macro_observation_counts=macro_totals,
        capability_flags=capability_flags,
        market_bindings=binding_metadata,
        target_semantics=target_semantics_metadata,
    )

    metadata_path = cfg.out_dir / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(_metadata_to_dict(metadata), indent=2), encoding="utf-8")

    return (
        BuildResult(
            dataset_parquet=dataset_parquet,
            dataset_csv=dataset_csv,
            features_npz=features_npz,
            feature_names=list(feature_names or []),
            feature_set_id=None,
            metadata=metadata,
        ),
        validation_result,
    )


def _ensure_datetime(value: datetime | float | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, int | float | np.generic):
        try:
            return datetime.fromtimestamp(float(value) / 1_000_000_000, tz=UTC)
        except (OSError, OverflowError, ValueError):  # pragma: no cover - defensive
            return None
    return None


def _format_window(start: datetime | float | None, end: datetime | float | None) -> tuple[str, str] | None:
    start_dt = _ensure_datetime(start)
    end_dt = _ensure_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    start_iso = format_dt(start_dt)
    end_iso = format_dt(end_dt)
    if start_iso is None or end_iso is None:
        return None
    return (start_iso, end_iso)


def _compute_dataset_metadata(
    df_pd_sorted: Any,
    cutoff: int,
    vintage_policy: VintagePolicy,
    vintage_as_of: datetime | None,
    build_ts: datetime,
    dataset_id: str | None,
    macro_observation_counts: dict[str, int] | None,
    target_semantics: dict[str, Any] | None,
) -> DatasetMetadata:
    from ml._imports import pd
    from ml._imports import pl

    overall_window = None
    train_window = None
    validation_window = None
    ts_start = None
    ts_end = None

    if pl is not None and isinstance(df_pd_sorted, pl.DataFrame):
        if "ts_event" in df_pd_sorted.columns:
            ts_col = "ts_event"
        elif "timestamp" in df_pd_sorted.columns:
            ts_col = "timestamp"
        else:
            ts_col = None

        if ts_col is not None and df_pd_sorted.height > 0:
            ts_series = df_pd_sorted.select(pl.col(ts_col)).to_series()
            start_dt_raw = ts_series[0]
            end_dt_raw = ts_series[len(ts_series) - 1]
            start_dt = _ensure_datetime(start_dt_raw)
            end_dt = _ensure_datetime(end_dt_raw)
            overall_window = _format_window(start_dt, end_dt)
            ts_start = format_dt(start_dt) if start_dt is not None else None
            ts_end = format_dt(end_dt) if end_dt is not None else None

            if cutoff > 0:
                train_start_dt = start_dt
                train_end_dt = _ensure_datetime(ts_series[min(cutoff - 1, len(ts_series) - 1)])
                train_window = _format_window(train_start_dt, train_end_dt)

            if cutoff < len(ts_series):
                val_start_dt = _ensure_datetime(ts_series[cutoff])
                val_end_dt = end_dt
                validation_window = _format_window(val_start_dt, val_end_dt)
    else:
        ts_series = None
        if pd is not None and hasattr(df_pd_sorted, "columns") and "ts_event" in df_pd_sorted.columns:
            try:
                ts_series = pd.to_datetime(df_pd_sorted["ts_event"], utc=True)
            except Exception:
                ts_series = None

        if ts_series is not None and hasattr(ts_series, "iloc") and len(ts_series) > 0:
            start_dt = ts_series.iloc[0].to_pydatetime()
            end_dt = ts_series.iloc[-1].to_pydatetime()
            overall_window = _format_window(start_dt, end_dt)
            ts_start = format_dt(start_dt)
            ts_end = format_dt(end_dt)

            if cutoff > 0:
                train_start = start_dt
                train_end = ts_series.iloc[max(cutoff - 1, 0)].to_pydatetime()
                train_window = _format_window(train_start, train_end)

            if cutoff < len(ts_series):
                val_start = ts_series.iloc[cutoff].to_pydatetime()
                val_end = end_dt
                validation_window = _format_window(val_start, val_end)

    # Placeholder for explicit test split metadata (future extension)
    build_ts_iso = format_dt(build_ts)
    if build_ts_iso is None:
        build_ts_iso = ""

    vintage_iso = format_dt(vintage_as_of) if vintage_as_of else None
    macro_counts = dict(macro_observation_counts or {})

    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        vintage_policy=vintage_policy,
        vintage_cutoff=vintage_iso,
        build_ts=build_ts_iso,
        ts_event_start=ts_start,
        ts_event_end=ts_end,
        overall_window=overall_window,
        train_window=train_window,
        validation_window=validation_window,
        test_window=None,
        macro_observation_counts=macro_counts,
        target_semantics=target_semantics,
    )
    return metadata


def _metadata_to_dict(metadata: DatasetMetadata) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_id": metadata.dataset_id,
        "vintage_policy": metadata.vintage_policy.value,
        "vintage_cutoff": metadata.vintage_cutoff,
        "build_ts": metadata.build_ts,
        "ts_event_start": metadata.ts_event_start,
        "ts_event_end": metadata.ts_event_end,
        "overall_window": list(metadata.overall_window) if metadata.overall_window else None,
        "train_window": list(metadata.train_window) if metadata.train_window else None,
        "validation_window": list(metadata.validation_window) if metadata.validation_window else None,
        "test_window": list(metadata.test_window) if metadata.test_window else None,
        "macro_observation_counts": metadata.macro_observation_counts,
        "capability_flags": metadata.capability_flags,
        "target_semantics": metadata.target_semantics,
    }
    if metadata.market_bindings is not None:
        payload["market_bindings"] = [
            {
                "binding_id": binding.binding_id,
                "dataset_id": binding.dataset_id,
                "descriptor_id": binding.descriptor_id,
                "schema": binding.schema,
                "storage_kind": binding.storage_kind,
                "symbols": list(binding.symbols),
                "instrument_ids": list(binding.instrument_ids),
                "source": binding.source,
                "license_start": binding.license_start,
                "license_end": binding.license_end,
                "ts_event_start": binding.ts_event_start,
                "ts_event_end": binding.ts_event_end,
                "rows_from_store": binding.rows_from_store,
                "rows_from_catalog": binding.rows_from_catalog,
                "provider_dataset_id": binding.provider_dataset_id,
                "provider_schema": binding.provider_schema,
            }
            for binding in metadata.market_bindings
        ]
    else:
        payload["market_bindings"] = None
    return payload


def _validate_dataset_metadata(metadata: DatasetMetadata) -> None:
    """Ensure computed metadata windows are internally consistent."""

    def _parse_window(window: tuple[str, str] | None) -> tuple[datetime | None, datetime | None]:
        if window is None:
            return (None, None)
        start_raw, end_raw = window
        start = parse_dt(start_raw)
        end = parse_dt(end_raw)
        if start is not None and end is not None and start > end:
            msg = f"Window start {start_raw} must be <= end {end_raw}"
            raise ValueError(msg)
        return (start, end)

    overall_start, overall_end = _parse_window(metadata.overall_window)
    ts_start = parse_dt(metadata.ts_event_start) if metadata.ts_event_start else None
    ts_end = parse_dt(metadata.ts_event_end) if metadata.ts_event_end else None

    if ts_start and ts_end and ts_start > ts_end:
        raise ValueError("ts_event_start must be <= ts_event_end")

    if metadata.overall_window and (ts_start or ts_end):
        if ts_start and overall_start and ts_start < overall_start:
            raise ValueError("ts_event_start earlier than overall_window start")
        if ts_end and overall_end and ts_end > overall_end:
            raise ValueError("ts_event_end later than overall_window end")

    for label, window in (
        ("train", metadata.train_window),
        ("validation", metadata.validation_window),
        ("test", metadata.test_window),
    ):
        start, end = _parse_window(window)
        if start and overall_start and start < overall_start:
            raise ValueError(f"{label}_window starts before overall window")
        if end and overall_end and end > overall_end:
            raise ValueError(f"{label}_window ends after overall window")


def _capability_flags_from_builder(builder: TFTDatasetBuilder) -> dict[str, bool]:
    include_l2 = bool(getattr(builder, "include_l2", False))
    include_micro = bool(getattr(builder, "include_micro", False)) or include_l2
    return {
        "include_macro": bool(getattr(builder, "include_macro", False)),
        "include_calendar": bool(getattr(builder, "include_calendar", False)),
        "include_events": bool(getattr(builder, "include_events", False)),
        "include_earnings": bool(getattr(builder, "include_earnings", False)),
        "include_l2": include_l2,
        "include_micro": include_micro,
    }


def build_tft_dataset(
    cfg: DatasetBuildConfig,
    *,
    data_store: DataStoreFacadeProtocol | None = None,
) -> BuildResult:
    """
    Build a TFT dataset and persist artifacts under `cfg.out_dir`.

    Parameters
    ----------
    cfg : DatasetBuildConfig
        Dataset configuration describing the output location and build options.
    data_store : DataStoreFacadeProtocol, optional
        Canonical DataStore for loading raw market data. When provided, the
        builder reads OHLCV data via the store and falls back to the catalog
        only if the store returns no rows.
    """
    from ml._imports import check_ml_dependencies
    from ml._imports import pl

    if pl is None:
        check_ml_dependencies(["polars"])  # Guard for cold-path environment

    # Defer ParquetDataCatalog import to avoid import-time heavy deps here
    from ml.data.ingest.macro_refresh import ensure_macro_ready
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    catalog = ParquetDataCatalog(path=str(cfg.data_dir))

    build_ts = datetime.now(tz=UTC)

    fred_parquet_path = cfg.macro_fred_path or DEFAULT_FRED_PARQUET_PATH
    macro_series_ids = cfg.macro_series_ids
    alfred_start_str, alfred_end_str = _derive_alfred_range(cfg)
    alfred_window_days = 180 if cfg.chunk_days else 365
    if cfg.include_macro and cfg.auto_refresh_macro and not cfg.student_mode:
        refresh_window = timedelta(hours=max(cfg.macro_staleness_hours, 0))
        macro_refresh = ensure_macro_ready(
            fred_path=fred_parquet_path,
            vintage_dir=cfg.fred_vintage_dir,
            max_age=refresh_window,
            data_store=data_store,
            series_ids=macro_series_ids,
            alfred_realtime_start=alfred_start_str,
            alfred_realtime_end=alfred_end_str,
            alfred_window_days=alfred_window_days,
        )
        if macro_refresh.fred_error is not None:
            logger.warning(
                "FRED macro refresh failed; proceeding with existing artifacts",
                error=str(macro_refresh.fred_error),
                path=str(fred_parquet_path),
            )
        if macro_refresh.alfred_error is not None:
            logger.warning(
                "ALFRED macro refresh failed; proceeding with existing artifacts",
                error=str(macro_refresh.alfred_error),
                base_dir=str(macro_refresh.alfred_base_dir),
            )

    fred_path_str = str(fred_parquet_path) if cfg.include_macro else None

    vintage_as_of = cfg.vintage_as_of
    if vintage_as_of is not None:
        if vintage_as_of.tzinfo is None:
            vintage_as_of = vintage_as_of.replace(tzinfo=UTC)
        else:
            vintage_as_of = vintage_as_of.astimezone(UTC)

    descriptor_map = load_market_feed_descriptors().as_mapping()
    resolved_bindings = resolve_market_dataset_bindings(
        symbols=cfg.symbols,
        instrument_ids=cfg.instrument_ids,
        market_dataset_id=cfg.market_dataset_id,
        market_inputs=cfg.market_inputs,
        descriptors=descriptor_map,
    )

    builder = TFTDatasetBuilder(
        catalog=catalog,
        symbols=cfg.symbols,
        instrument_ids=cfg.instrument_ids,
        include_macro=cfg.include_macro,
        macro_lag_days=cfg.macro_lag_days,
        include_micro=cfg.include_micro,
        include_l2=cfg.include_l2,
        include_events=cfg.include_events,
        include_calendar=cfg.include_calendar,
        include_earnings=cfg.include_earnings,
        earnings_lag_days=cfg.earnings_lag_days,
        fred_path=fred_path_str,
        vintage_base_dir=cfg.fred_vintage_dir,
        events_base_dir=cfg.events_base_dir,
        student_mode=cfg.student_mode,
        micro_base_dir=str(cfg.micro_base_dir or cfg.data_dir),
        l2_base_dir=str(cfg.l2_base_dir or cfg.data_dir),
        macro_series_ids=cfg.macro_series_ids,
        vintage_policy=cfg.vintage_policy,
        vintage_as_of=vintage_as_of,
        data_store=data_store,
        market_dataset_id=cfg.market_dataset_id,
        market_bindings=resolved_bindings,
        include_macro_revisions=cfg.include_macro_revisions,
        macro_revision_mode=cfg.macro_revision_mode,
        macro_revision_windows=cfg.macro_revision_windows,
    )
    capability_flags = _capability_flags_from_builder(builder)
    target_semantics = _resolve_target_semantics(cfg)
    from ml.training.datasets.target_generator import build_target_semantics_metadata

    target_semantics_metadata = build_target_semantics_metadata(target_semantics)

    chunk_mode = bool(cfg.chunk_days > 0 and cfg.start and cfg.end)
    if chunk_mode:
        build_result, _ = _build_dataset_chunked(
            builder=builder,
            cfg=cfg,
            vintage_as_of=vintage_as_of,
            build_ts=build_ts,
            target_semantics=target_semantics,
        )
        return build_result

    from ml._imports import HAS_POLARS
    from ml._imports import pl

    assert HAS_POLARS and pl is not None

    df_any = builder.build_training_dataset(
        horizon_minutes=cfg.horizon_minutes,
        min_return_threshold=cfg.threshold,
        target_semantics=target_semantics,
        lookback_periods=cfg.lookback_periods,
        use_polars=True,
        start=cfg.start,
        end=cfg.end,
    )
    if isinstance(df_any, pl.DataFrame):
        dataset_df = df_any
    else:  # pragma: no cover - fallback path
        from ml._imports import HAS_PANDAS
        from ml._imports import pd

        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])  # pragma: no cover
        assert pd is not None
        dataset_df = pl.from_pandas(df_any)

    # Validate dataset before persisting artifacts
    validation_cfg = cfg.validation or DatasetValidationConfig(
        require_macro_series=cfg.macro_series_ids,
    )
    if validation_cfg.require_macro_series is None and cfg.macro_series_ids:
        validation_cfg = replace(validation_cfg, require_macro_series=cfg.macro_series_ids)
    if validation_cfg.expected_vintage_policy is None:
        validation_cfg = replace(validation_cfg, expected_vintage_policy=cfg.vintage_policy)
    if (
        validation_cfg.macro_min_vintage_observations is None
        and cfg.include_macro
        and cfg.vintage_policy is VintagePolicy.REAL_TIME
        and cfg.macro_series_ids
    ):
        validation_cfg = replace(validation_cfg, macro_min_vintage_observations=1)
    if cfg.vintage_policy is not VintagePolicy.REAL_TIME and validation_cfg.macro_min_vintage_observations is not None:
        validation_cfg = replace(validation_cfg, macro_min_vintage_observations=None)
    validation_result = validate_dataset(dataset_df, config=validation_cfg)
    logger.info(
        "Dataset validation succeeded",
        rows=validation_result.row_count,
        positive_rate=validation_result.positive_rate,
    )

    # Persist dataset artifacts
    dataset_parquet = cfg.out_dir / "dataset.parquet"
    dataset_csv = cfg.out_dir / "dataset.csv"
    dataset_df.write_parquet(str(dataset_parquet))
    _write_dataset_csv(dataset_df, cfg, dataset_csv=dataset_csv)

    # Build feature matrix artifacts without materialising the entire dataset in memory
    feature_names = _infer_feature_columns(dataset_df)
    df_sorted = dataset_df.sort("time_index") if "time_index" in dataset_df.columns else dataset_df.clone()
    cutoff = int(df_sorted.height * 0.8) if df_sorted.height > 0 else 0
    features_npz = cfg.out_dir / "features_npz.npz"
    _write_feature_npz_from_polars(
        df_sorted,
        feature_names,
        out_path=features_npz,
        cutoff=cutoff,
    )

    metadata = _compute_dataset_metadata(
        df_sorted,
        cutoff,
        cfg.vintage_policy,
        vintage_as_of,
        build_ts,
        getattr(cfg, "dataset_id", None),
        getattr(validation_result, "macro_observation_counts", {}),
        target_semantics_metadata,
    )
    binding_metadata = _binding_stats_to_metadata(builder.get_binding_stats())
    metadata = replace(metadata, market_bindings=binding_metadata, capability_flags=capability_flags)
    _validate_dataset_metadata(metadata)
    metadata_path = cfg.out_dir / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(_metadata_to_dict(metadata), indent=2), encoding="utf-8")

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
            "include_macro": capability_flags["include_macro"],
            "include_micro": capability_flags["include_micro"],
            "include_l2": capability_flags["include_l2"],
            "include_earnings": capability_flags["include_earnings"],
            "horizon_minutes": cfg.horizon_minutes,
            "lookback_periods": cfg.lookback_periods,
        }
        feature_set_id = export_feature_manifest(
            feature_names=feature_names,
            feature_dtypes=["float32"] * len(feature_names),
            flags=flags,
            cfg=reg_cfg,
        )

    # Optional dataset event emission (best-effort; cold path only)
    if cfg.emit_dataset_events:
        try:
            from ml.common.event_emitter import emit_dataset_event
            from ml.config.events import EventStatus
            from ml.config.events import Source
            from ml.config.events import Stage
            from ml.core.integration import MLIntegrationManager
            from ml.registry.protocols import RegistryProtocol

            mgr = MLIntegrationManager(
                auto_start_postgres=False,
                auto_migrate=False,
                ensure_healthy=False,
            )
            data_registry = cast(RegistryProtocol, mgr.data_registry)
            df_event = df_sorted
            count = int(df_event.height)
            ts_min_ns = 0
            ts_max_ns = 0
            try:
                if "timestamp" in df_event.columns:
                    ts_min_ns = int(
                        df_event.select(pl.col("timestamp").cast(pl.Datetime("ns")).min()).item(),
                    )
                    ts_max_ns = int(
                        df_event.select(pl.col("timestamp").cast(pl.Datetime("ns")).max()).item(),
                    )
                elif "ts_event" in df_event.columns:
                    ts_min_ns = int(df_event.select(pl.col("ts_event").min()).item())
                    ts_max_ns = int(df_event.select(pl.col("ts_event").max()).item())
            except Exception:
                ts_min_ns = 0
                ts_max_ns = 0

            # Emit a single SUCCESS event capturing lineage/flags
            emit_dataset_event(
                data_registry,
                dataset_id=getattr(cfg, "dataset_id", "tft_dataset"),
                instrument_id="GLOBAL",
                stage=Stage.FEATURE_COMPUTED,
                source=Source.HISTORICAL,
                run_id=f"build_tft_{int(__import__('time').time()*1e6):d}",
                ts_min=ts_min_ns,
                ts_max=ts_max_ns,
                count=count,
                status=EventStatus.SUCCESS,
                metadata={
                    "symbols": ",".join(cfg.symbols),
                    "include_macro": bool(cfg.include_macro),
                    "include_micro": bool(cfg.include_micro),
                    "include_l2": bool(cfg.include_l2),
                    "horizon_minutes": int(cfg.horizon_minutes),
                    "lookback_periods": int(cfg.lookback_periods),
                },
                dataset_type="dataset",
                component="dataset_builder",
            )
        except Exception:
            # Best-effort; never fail the dataset build on event emission
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "Emit dataset event failed (ignored)",
                exc_info=True,
            )

    return BuildResult(
        dataset_parquet=dataset_parquet,
        dataset_csv=dataset_csv,
        features_npz=features_npz,
        feature_names=feature_names,
        feature_set_id=feature_set_id,
        metadata=metadata,
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
    if name in {"ALFREDConfig", "ALFREDDataLoader"}:
        from ml.data.loaders import alfred_loader as _alfred

        return getattr(_alfred, name)
    if name == "FileEventSource":
        from ml.data.sources import events as _events

        return getattr(_events, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
