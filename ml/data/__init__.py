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

import structlog

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
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np

from ml.data.vintage import VintagePolicy
from ml.data.vintage import format_dt
from ml.data.vintage import parse_dt


# (Avoid importing optional deps here; import lazily inside functions)

logger = structlog.get_logger(__name__)

DEFAULT_FRED_PARQUET_PATH = Path("data/fred/fred_indicators_ml_format.parquet")




@dataclass(frozen=True)
class DatasetBuildConfig:
    data_dir: Path
    out_dir: Path
    symbols: list[str]
    dataset_id: str = "tft_dataset"
    instrument_ids: list[str] | None = None
    # Feature options
    include_macro: bool = True
    macro_lag_days: int = 1
    include_micro: bool = False
    include_l2: bool = False
    include_events: bool = False
    include_calendar: bool = False
    fred_vintage_dir: Path | None = None
    events_base_dir: Path | None = None
    student_mode: bool = False
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
    # Optional lineage/events (cold path only)
    emit_dataset_events: bool = False
    # Macro refresh controls
    auto_refresh_macro: bool = True
    macro_staleness_hours: int = 24
    macro_series_ids: tuple[str, ...] | None = None
    macro_fred_path: Path | None = None
    vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME
    vintage_as_of: datetime | None = None
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


def load_dataset_metadata(path: Path) -> DatasetMetadata:
    """Load dataset metadata from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        policy = VintagePolicy(raw.get("vintage_policy", VintagePolicy.REAL_TIME.value))
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid vintage_policy in metadata: {raw.get('vintage_policy')}") from exc

    def _as_tuple(value: object | None) -> tuple[str, str] | None:
        if not value:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (str(value[0]), str(value[1]))
        raise ValueError(f"Metadata window must be length-2 sequence, got {value!r}")

    macro_counts_raw = raw.get("macro_observation_counts") or {}
    macro_counts: dict[str, int] = {
        str(key): int(value)
        for key, value in macro_counts_raw.items()
    }

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
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"tft_pipeline:{digest[:16]}"


def _infer_feature_columns(df: Any) -> list[str]:
    """
    Infer numeric feature columns, excluding label/index/meta fields.
    """
    exclude = {
        "y",
        "forward_return",
        "time_index",
        "timestamp",
        "instrument_id",
        "ts_event",
    }
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


def _format_window(start: datetime | None, end: datetime | None) -> tuple[str, str] | None:
    if start is None or end is None:
        return None
    start_iso = format_dt(start)
    end_iso = format_dt(end)
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
) -> DatasetMetadata:
    from ml._imports import pd

    ts_series = None
    if pd is not None and "ts_event" in df_pd_sorted.columns:
        try:
            ts_series = pd.to_datetime(df_pd_sorted["ts_event"], utc=True)
        except Exception:
            ts_series = None

    overall_window = None
    train_window = None
    validation_window = None
    ts_start = None
    ts_end = None

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
    }
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

def build_tft_dataset(cfg: DatasetBuildConfig) -> BuildResult:
    """
    Build a TFT dataset and persist artifacts under `cfg.out_dir`.
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
    if cfg.include_macro and cfg.auto_refresh_macro and not cfg.student_mode:
        refresh_window = timedelta(hours=max(cfg.macro_staleness_hours, 0))
        result = ensure_macro_ready(
            fred_path=fred_parquet_path,
            vintage_dir=cfg.fred_vintage_dir,
            max_age=refresh_window,
            series_ids=macro_series_ids,
        )
        if result.fred_error is not None:
            logger.warning(
                "FRED macro refresh failed; proceeding with existing artifacts",
                error=str(result.fred_error),
                path=str(fred_parquet_path),
            )
        if result.alfred_error is not None:
            logger.warning(
                "ALFRED macro refresh failed; proceeding with existing artifacts",
                error=str(result.alfred_error),
                base_dir=str(result.alfred_base_dir),
            )

    fred_path_str = str(fred_parquet_path) if cfg.include_macro else None

    vintage_as_of = cfg.vintage_as_of
    if vintage_as_of is not None:
        if vintage_as_of.tzinfo is None:
            vintage_as_of = vintage_as_of.replace(tzinfo=UTC)
        else:
            vintage_as_of = vintage_as_of.astimezone(UTC)

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
        fred_path=fred_path_str,
        vintage_base_dir=cfg.fred_vintage_dir,
        events_base_dir=cfg.events_base_dir,
        student_mode=cfg.student_mode,
        micro_base_dir=str(cfg.data_dir),
        l2_base_dir=str(cfg.data_dir),
        macro_series_ids=cfg.macro_series_ids,
        vintage_policy=cfg.vintage_policy,
        vintage_as_of=vintage_as_of,
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
    validation_result = validate_dataset(df, config=validation_cfg)
    logger.info(
        "Dataset validation succeeded",
        rows=validation_result.row_count,
        positive_rate=validation_result.positive_rate,
    )

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

    metadata = _compute_dataset_metadata(
        df_pd_sorted,
        cutoff,
        cfg.vintage_policy,
        vintage_as_of,
        build_ts,
        getattr(cfg, "dataset_id", None),
        getattr(validation_result, "macro_observation_counts", {}),
    )
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
            # Derive basic stats
            count = len(df_pd_sorted) if "df_pd_sorted" in locals() else 0
            # Attempt to extract time bounds (ns) from available columns
            ts_min_ns = 0
            ts_max_ns = 0
            try:
                if "timestamp" in df_pd_sorted.columns:  # pandas path
                    ts_col = df_pd_sorted["timestamp"]
                    # Support ints or pandas datetime
                    if hasattr(ts_col, "dt"):
                        ts_min_ns = int(getattr(ts_col.min(), "value", ts_col.min()))
                        ts_max_ns = int(getattr(ts_col.max(), "value", ts_col.max()))
                    else:
                        ts_min_ns = int(ts_col.min())
                        ts_max_ns = int(ts_col.max())
                elif "ts_event" in df_pd_sorted.columns:
                    ts_min_ns = int(df_pd_sorted["ts_event"].min())
                    ts_max_ns = int(df_pd_sorted["ts_event"].max())
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
