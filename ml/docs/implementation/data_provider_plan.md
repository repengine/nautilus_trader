# Data Provider Implementation Plan

## Overview

This document outlines the comprehensive implementation plan for the data provider architecture that will populate static and known-future features with real data for TFT models.

## Architecture Overview

The implementation follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────┐
│                  Feature Pipeline                    │
│              (Orchestration & Caching)               │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────┐
│               Provider Protocols                     │
│         (Abstract Interfaces - SOLID)                │
└────────────────────┬────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┬────────────────┐
     │               │               │                │
┌────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐ ┌──────▼─────┐
│Metadata   │ │Calendar    │ │Event       │ │Macro       │
│Provider   │ │Provider    │ │Provider    │ │Provider    │
└───────────┘ └────────────┘ └────────────┘ └────────────┘
     │               │               │                │
┌────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐ ┌──────▼─────┐
│External   │ │Market      │ │Economic    │ │Indicator   │
│Sources    │ │Calendars   │ │Calendars   │ │APIs        │
└───────────┘ └────────────┘ └────────────┘ └────────────┘
```

## Design Principles

### SOLID Principles

- **Single Responsibility**: Each provider has one clear responsibility
- **Open/Closed**: New data sources can be added without modifying existing providers
- **Liskov Substitution**: All providers are substitutable via protocols
- **Interface Segregation**: Separate interfaces for static vs time-series data
- **Dependency Inversion**: Depend on protocols, not concrete implementations

### DRY (Don't Repeat Yourself)

- Base classes contain common logic
- Utility functions for shared calculations
- Single source of truth for data schemas

### TDD (Test-Driven Development)

- Write tests before implementation
- Use hypothesis for property-based testing
- Mock external dependencies

### Separation of Concerns

- Data loading separate from transformation
- Caching logic isolated in base classes
- External API access isolated in source classes

## Implementation Phases

### Phase 1: Protocol & Base Infrastructure

#### 1.1 Provider Protocols (`ml/data/providers/base.py`)

```python
from typing import Protocol, Any
import polars as pl
from datetime import datetime

class DataProvider(Protocol):
    """Base protocol all data providers must implement."""

    def load_data(
        self,
        instruments: list[str],
        start: datetime,
        end: datetime
    ) -> pl.DataFrame:
        """Load data for specified instruments and time range."""
        ...

    def validate_data(self, data: pl.DataFrame) -> bool:
        """Validate data meets schema requirements."""
        ...

    def get_schema(self) -> dict[str, type]:
        """Return expected data schema."""
        ...

class CacheableProvider(Protocol):
    """Protocol for providers that support caching."""

    def cache_key(self, params: dict[str, Any]) -> str:
        """Generate cache key from parameters."""
        ...

    def from_cache(self, key: str) -> pl.DataFrame | None:
        """Load data from cache if available."""
        ...

    def to_cache(self, key: str, data: pl.DataFrame) -> None:
        """Save data to cache."""
        ...

class StaticDataProvider(Protocol):
    """Protocol for time-invariant data providers."""

    def load_metadata(self, instruments: list[str]) -> pl.DataFrame:
        """Load static metadata for instruments."""
        ...

class TimeSeriesProvider(Protocol):
    """Protocol for time-varying data providers."""

    def load_timeseries(
        self,
        instruments: list[str],
        timestamps: pl.Series
    ) -> pl.DataFrame:
        """Load time series data."""
        ...
```

#### 1.2 Base Classes (`ml/data/providers/base.py`)

```python
class BaseDataProvider:
    """Base implementation with common functionality."""

    def __init__(self):
        self.logger = self._setup_logging()
        self.metrics = self._setup_metrics()

    def validate_data(self, data: pl.DataFrame) -> bool:
        """Common validation logic."""
        # Check for required columns
        # Validate data types
        # Check for nulls in required fields
        pass

    def _handle_error(self, error: Exception) -> None:
        """Common error handling."""
        self.logger.error(f"Provider error: {error}")
        self.metrics.increment("provider_errors")

class CachedDataProvider(BaseDataProvider):
    """Base class with caching support."""

    def __init__(self, cache_ttl_hours: int = 24):
        super().__init__()
        self.cache_ttl = cache_ttl_hours
        self._cache = {}

    def load_data(self, instruments, start, end):
        """Template method with caching."""
        key = self.cache_key({"instruments": instruments, "start": start, "end": end})

        # Try cache first
        cached = self.from_cache(key)
        if cached is not None:
            return cached

        # Load fresh data
        data = self._load_data_impl(instruments, start, end)

        # Cache for next time
        self.to_cache(key, data)
        return data

    def _load_data_impl(self, instruments, start, end):
        """Override in subclasses."""
        raise NotImplementedError
```

#### 1.3 Utility Functions (`ml/data/providers/utils.py`)

```python
import numpy as np
import polars as pl
from datetime import datetime

def cyclic_encode(value: float, period: float) -> tuple[float, float]:
    """
    Encode a cyclic value as sin/cos pair.

    Parameters
    ----------
    value : float
        The value to encode (e.g., hour of day)
    period : float
        The period of the cycle (e.g., 24 for hours)

    Returns
    -------
    tuple[float, float]
        (sin, cos) encoding

    Examples
    --------
    >>> cyclic_encode(12, 24)  # Noon
    (0.0, -1.0)
    >>> cyclic_encode(0, 24)   # Midnight
    (0.0, 1.0)
    """
    angle = 2 * np.pi * value / period
    return (np.sin(angle), np.cos(angle))

def time_to_event(
    current: datetime,
    event: datetime,
    unit: str = "hours"
) -> float:
    """
    Calculate time until an event.

    Parameters
    ----------
    current : datetime
        Current timestamp
    event : datetime
        Event timestamp
    unit : str
        Time unit for result ('hours', 'days', 'minutes')

    Returns
    -------
    float
        Time to event in specified units
    """
    delta = event - current
    if unit == "hours":
        return delta.total_seconds() / 3600
    elif unit == "days":
        return delta.total_seconds() / 86400
    elif unit == "minutes":
        return delta.total_seconds() / 60
    else:
        raise ValueError(f"Unknown unit: {unit}")

def validate_timestamps(series: pl.Series) -> bool:
    """
    Validate timestamp series.

    Checks:
    - No nulls
    - Monotonically increasing
    - Within reasonable range
    """
    if series.null_count() > 0:
        return False

    if not series.is_sorted():
        return False

    # Check reasonable range (1970 to 2100)
    min_ts = series.min()
    max_ts = series.max()
    if min_ts < 0 or max_ts > 4102444800000000000:  # Year 2100 in nanoseconds
        return False

    return True

def align_timeseries(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    on: str = "timestamp",
    how: str = "inner"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Align two timeseries dataframes.

    Parameters
    ----------
    df1, df2 : pl.DataFrame
        DataFrames to align
    on : str
        Column to align on
    how : str
        Join type ('inner', 'left', 'outer')

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Aligned dataframes
    """
    # Get common timestamps
    if how == "inner":
        common_ts = df1[on].unique().filter(df1[on].is_in(df2[on]))
    elif how == "left":
        common_ts = df1[on].unique()
    else:  # outer
        common_ts = pl.concat([df1[on], df2[on]]).unique().sort()

    # Filter to common timestamps
    df1_aligned = df1.filter(df1[on].is_in(common_ts))
    df2_aligned = df2.filter(df2[on].is_in(common_ts))

    return df1_aligned, df2_aligned
```

### Phase 2: Metadata Provider Implementation

#### 2.1 Instrument Metadata Provider (`ml/data/providers/metadata.py`)

```python
from ml.data.providers.base import BaseStaticProvider
from ml.data.sources.base import MetadataSource
import polars as pl

class InstrumentMetadataProvider(BaseStaticProvider):
    """
    Provider for instrument specifications and metadata.

    Provides static instrument attributes like tick_size, lot_size,
    exchange, asset_class, currency, etc.
    """

    def __init__(self, source: MetadataSource):
        """
        Initialize metadata provider.

        Parameters
        ----------
        source : MetadataSource
            Data source for instrument metadata
        """
        super().__init__()
        self.source = source
        self._cache = {}

    def load_metadata(self, instruments: list[str]) -> pl.DataFrame:
        """
        Load metadata for specified instruments.

        Returns columns:
        - instrument_id: str
        - tick_size: float
        - lot_size: float
        - contract_size: float
        - min_price_increment: float
        - exchange: str
        - asset_class: str
        - currency: str
        - margin_initial: float
        - margin_maintenance: float
        """
        # Check cache first
        cache_key = "_".join(sorted(instruments))
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load from source
        data = self.source.fetch_metadata(instruments)

        # Validate
        if not self.validate_data(data):
            raise ValueError("Invalid metadata format")

        # Cache and return
        self._cache[cache_key] = data
        return data

    def get_schema(self) -> dict[str, type]:
        """Return expected schema."""
        return {
            "instrument_id": str,
            "tick_size": float,
            "lot_size": float,
            "contract_size": float,
            "exchange": str,
            "asset_class": str,
            "currency": str,
        }
```

#### 2.2 Metadata Sources (`ml/data/sources/metadata.py`)

```python
from abc import ABC, abstractmethod
import polars as pl

class MetadataSource(ABC):
    """Abstract base for metadata sources."""

    @abstractmethod
    def fetch_metadata(self, instruments: list[str]) -> pl.DataFrame:
        """Fetch metadata from source."""
        pass

class DatabentoMetadataSource(MetadataSource):
    """Load metadata from Databento."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize Databento client

    def fetch_metadata(self, instruments: list[str]) -> pl.DataFrame:
        # Implementation to fetch from Databento
        pass

class CSVMetadataSource(MetadataSource):
    """Load metadata from CSV (for testing)."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def fetch_metadata(self, instruments: list[str]) -> pl.DataFrame:
        df = pl.read_csv(self.file_path)
        return df.filter(pl.col("instrument_id").is_in(instruments))
```

### Phase 3: Calendar Provider Implementation

#### 3.1 Market Calendar Provider (`ml/data/providers/calendar.py`)

```python
from ml.data.providers.base import BaseTimeSeriesProvider
from ml.data.providers.utils import cyclic_encode
import polars as pl
import numpy as np

class MarketCalendarProvider(BaseTimeSeriesProvider):
    """
    Provider for market calendar features.

    Provides time-based features like trading hours, holidays,
    time-of-day encodings, etc.
    """

    def __init__(self, calendar_source):
        super().__init__()
        self.calendar = calendar_source

    def compute_features(
        self,
        timestamps: pl.Series,
        exchange: str = "NYSE"
    ) -> pl.DataFrame:
        """
        Compute calendar features for timestamps.

        Returns columns:
        - timestamp: int
        - is_trading_day: bool
        - is_pre_market: bool
        - is_after_hours: bool
        - minutes_to_close: float
        - hour_sin: float
        - hour_cos: float
        - dow_sin: float
        - dow_cos: float
        - month_sin: float
        - month_cos: float
        - is_weekend: bool
        - is_month_start: bool
        - is_month_end: bool
        - days_to_month_end: int
        """
        features = []

        for ts in timestamps:
            dt = self._to_datetime(ts)
            schedule = self.calendar.get_schedule(dt, exchange)

            # Time encodings
            hour_sin, hour_cos = cyclic_encode(dt.hour + dt.minute/60, 24)
            dow_sin, dow_cos = cyclic_encode(dt.weekday(), 7)
            month_sin, month_cos = cyclic_encode(dt.month - 1, 12)

            features.append({
                "timestamp": ts,
                "is_trading_day": schedule.is_trading_day,
                "is_pre_market": self._is_pre_market(dt, schedule),
                "is_after_hours": self._is_after_hours(dt, schedule),
                "minutes_to_close": self._minutes_to_close(dt, schedule),
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "dow_sin": dow_sin,
                "dow_cos": dow_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "is_weekend": dt.weekday() >= 5,
                "is_month_start": dt.day <= 3,
                "is_month_end": dt.day >= 28,
                "days_to_month_end": self._days_to_month_end(dt),
            })

        return pl.DataFrame(features)
```

### Phase 4: Event Schedule Provider

#### 4.1 Event Provider (`ml/data/providers/events.py`)

```python
from ml.data.providers.base import BaseTimeSeriesProvider
from ml.data.providers.utils import time_to_event
import polars as pl

class EventScheduleProvider(BaseTimeSeriesProvider):
    """
    Provider for scheduled event features.

    Provides features related to earnings, economic releases,
    Fed meetings, options expiry, etc.
    """

    def __init__(self, event_sources: dict):
        super().__init__()
        self.earnings_source = event_sources.get("earnings")
        self.economic_source = event_sources.get("economic")
        self.fed_source = event_sources.get("fed")

    def compute_features(
        self,
        timestamps: pl.Series,
        instruments: list[str]
    ) -> pl.DataFrame:
        """
        Compute event proximity features.

        Returns columns:
        - timestamp: int
        - instrument_id: str
        - hours_to_earnings: float
        - has_earnings_today: bool
        - hours_to_fed_meeting: float
        - has_fed_meeting_week: bool
        - hours_to_cpi: float
        - hours_to_nfp: float
        - hours_to_options_expiry: float
        - event_density_24h: int
        - event_density_week: int
        """
        features = []

        for ts, inst in zip(timestamps, instruments):
            dt = self._to_datetime(ts)

            # Get upcoming events
            earnings = self.earnings_source.get_next_event(inst, dt)
            fed = self.fed_source.get_next_event(dt)
            cpi = self.economic_source.get_next_event("CPI", dt)
            nfp = self.economic_source.get_next_event("NFP", dt)
            opex = self._get_next_opex(dt)

            features.append({
                "timestamp": ts,
                "instrument_id": inst,
                "hours_to_earnings": time_to_event(dt, earnings) if earnings else 999,
                "has_earnings_today": self._is_same_day(dt, earnings),
                "hours_to_fed_meeting": time_to_event(dt, fed) if fed else 999,
                "has_fed_meeting_week": self._is_same_week(dt, fed),
                "hours_to_cpi": time_to_event(dt, cpi) if cpi else 999,
                "hours_to_nfp": time_to_event(dt, nfp) if nfp else 999,
                "hours_to_options_expiry": time_to_event(dt, opex),
                "event_density_24h": self._count_events_window(dt, 24),
                "event_density_week": self._count_events_window(dt, 168),
            })

        return pl.DataFrame(features)
```

### Phase 5: Integration Layer

#### 5.1 Enhanced Feature Config (`ml/config/features.py`)

```python
from ml.config.base import MLFeatureConfig

class EnhancedFeatureConfig(MLFeatureConfig):
    """Extended configuration with data provider settings."""

    # Data sources
    metadata_source: str = "databento"
    calendar_source: str = "pandas_market_calendars"
    event_source: str = "mock"

    # Feature flags
    include_static_covariates: bool = True
    include_calendar_features: bool = True
    include_event_features: bool = True
    include_macro_indicators: bool = False

    # Caching
    enable_provider_cache: bool = True
    cache_ttl_hours: int = 24
    cache_dir: str = "/tmp/ml_cache"

    # Provider-specific settings
    calendar_exchange: str = "NYSE"
    event_lookback_days: int = 30
    event_lookahead_days: int = 30
```

#### 5.2 Provider Factory (`ml/data/providers/factory.py`)

```python
from ml.data.providers.metadata import InstrumentMetadataProvider
from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.providers.events import EventScheduleProvider

class ProviderFactory:
    """Factory for creating data providers."""

    @staticmethod
    def create_metadata_provider(source: str, **kwargs):
        """Create metadata provider based on source."""
        if source == "databento":
            from ml.data.sources.metadata import DatabentoMetadataSource
            return InstrumentMetadataProvider(
                DatabentoMetadataSource(kwargs.get("api_key"))
            )
        elif source == "csv":
            from ml.data.sources.metadata import CSVMetadataSource
            return InstrumentMetadataProvider(
                CSVMetadataSource(kwargs.get("file_path"))
            )
        elif source == "mock":
            from ml.data.sources.metadata import MockMetadataSource
            return InstrumentMetadataProvider(MockMetadataSource())
        else:
            raise ValueError(f"Unknown metadata source: {source}")

    @staticmethod
    def create_calendar_provider(source: str, **kwargs):
        """Create calendar provider based on source."""
        # Similar pattern for calendar provider
        pass

    @staticmethod
    def create_event_provider(source: str, **kwargs):
        """Create event provider based on source."""
        # Similar pattern for event provider
        pass
```

#### 5.3 Enhanced Feature Engineer (`ml/features/engineering.py`)

```python
class FeatureEngineer:
    """Enhanced with data provider integration."""

    def __init__(self, config: EnhancedFeatureConfig):
        self.config = config
        self._init_providers()
        # ... existing initialization

    def _init_providers(self):
        """Initialize data providers based on config."""
        if self.config.include_static_covariates:
            self.metadata_provider = ProviderFactory.create_metadata_provider(
                self.config.metadata_source,
                cache_ttl=self.config.cache_ttl_hours
            )

        if self.config.include_calendar_features:
            self.calendar_provider = ProviderFactory.create_calendar_provider(
                self.config.calendar_source,
                exchange=self.config.calendar_exchange
            )

        if self.config.include_event_features:
            self.event_provider = ProviderFactory.create_event_provider(
                self.config.event_source,
                lookback_days=self.config.event_lookback_days
            )

    def calculate_features_batch(
        self,
        bars_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Calculate all features including provider-based."""

        # Existing indicator features
        features_df = self._calculate_indicator_features_batch(bars_df)

        # Add static covariates
        if self.config.include_static_covariates and self.metadata_provider:
            instruments = bars_df["instrument_id"].unique().to_list()
            metadata_df = self.metadata_provider.load_metadata(instruments)
            features_df = features_df.join(metadata_df, on="instrument_id")

        # Add calendar features
        if self.config.include_calendar_features and self.calendar_provider:
            calendar_df = self.calendar_provider.compute_features(
                bars_df["timestamp"]
            )
            features_df = features_df.join(calendar_df, on="timestamp")

        # Add event features
        if self.config.include_event_features and self.event_provider:
            event_df = self.event_provider.compute_features(
                bars_df["timestamp"],
                bars_df["instrument_id"].to_list()
            )
            features_df = features_df.join(
                event_df,
                on=["timestamp", "instrument_id"]
            )

        return features_df
```

### Phase 6: Testing Strategy

#### 6.1 Unit Tests with TDD

**Test Provider Protocols** (`ml/tests/unit/data/providers/test_base.py`):

```python
import pytest
from hypothesis import given, strategies as st
from ml.data.providers.base import DataProvider, StaticDataProvider

def test_protocol_enforcement():
    """Test that protocols are properly enforced."""
    class InvalidProvider:
        pass

    with pytest.raises(TypeError):
        # Should fail - doesn't implement protocol
        provider: DataProvider = InvalidProvider()

def test_base_provider_validation():
    """Test base provider validation logic."""
    from ml.data.providers.base import BaseDataProvider

    provider = BaseDataProvider()

    # Test with valid data
    valid_df = pl.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    assert provider.validate_data(valid_df)

    # Test with invalid data (empty)
    invalid_df = pl.DataFrame()
    assert not provider.validate_data(invalid_df)
```

**Test Utility Functions** (`ml/tests/unit/data/providers/test_utils.py`):

```python
from hypothesis import given, strategies as st
import numpy as np
from ml.data.providers.utils import cyclic_encode, time_to_event

@given(
    value=st.floats(min_value=0, max_value=100),
    period=st.floats(min_value=0.1, max_value=100)
)
def test_cyclic_encode_properties(value, period):
    """Test cyclic encoding properties."""
    sin_val, cos_val = cyclic_encode(value, period)

    # Property: sin^2 + cos^2 = 1
    assert np.abs(sin_val**2 + cos_val**2 - 1.0) < 1e-10

    # Property: values in [-1, 1]
    assert -1 <= sin_val <= 1
    assert -1 <= cos_val <= 1

@given(
    hour=st.integers(min_value=0, max_value=23),
    minute=st.integers(min_value=0, max_value=59)
)
def test_time_encoding_continuity(hour, minute):
    """Test that time encoding is continuous."""
    time1 = hour + minute/60
    time2 = time1 + 1/60  # One minute later

    sin1, cos1 = cyclic_encode(time1, 24)
    sin2, cos2 = cyclic_encode(time2 % 24, 24)

    # Small time changes should result in small encoding changes
    assert np.abs(sin2 - sin1) < 0.1
    assert np.abs(cos2 - cos1) < 0.1
```

#### 6.2 Integration Tests

```python
def test_provider_chain_integration():
    """Test that providers work together."""
    config = EnhancedFeatureConfig(
        metadata_source="mock",
        calendar_source="mock",
        event_source="mock",
        include_static_covariates=True,
        include_calendar_features=True,
        include_event_features=True,
    )

    engineer = FeatureEngineer(config)

    # Create sample bars data
    bars_df = create_sample_bars()

    # Calculate features
    features_df = engineer.calculate_features_batch(bars_df)

    # Verify all feature groups present
    assert "tick_size" in features_df.columns  # Static
    assert "hour_sin" in features_df.columns   # Calendar
    assert "hours_to_earnings" in features_df.columns  # Event
```

#### 6.3 Performance Tests

```python
import time
import pytest

@pytest.mark.performance
def test_provider_latency():
    """Test that providers meet latency requirements."""
    provider = create_test_provider()

    start = time.perf_counter()
    data = provider.load_data(["SPY"], start_date, end_date)
    elapsed = time.perf_counter() - start

    # Should complete within 100ms for cached data
    assert elapsed < 0.1

@pytest.mark.performance
def test_cache_effectiveness():
    """Test that caching improves performance."""
    provider = CachedDataProvider()

    # First call - cache miss
    start1 = time.perf_counter()
    data1 = provider.load_data(["SPY"], start_date, end_date)
    time1 = time.perf_counter() - start1

    # Second call - cache hit
    start2 = time.perf_counter()
    data2 = provider.load_data(["SPY"], start_date, end_date)
    time2 = time.perf_counter() - start2

    # Cached should be at least 10x faster
    assert time2 < time1 / 10
```

## Implementation Timeline

1. **Week 1**: Protocols, base classes, and utilities with tests
2. **Week 2**: Metadata and calendar providers with mock sources
3. **Week 3**: Event and macro providers with mock sources
4. **Week 4**: Integration layer and feature engineer updates
5. **Week 5**: Real data source implementations
6. **Week 6**: Performance optimization and comprehensive testing

## Success Metrics

- All tests passing with >95% coverage
- Property-based tests with hypothesis passing
- Latency <10ms for cached data access
- Cache hit rate >80% in production
- Zero data leakage in time-series features
- Full compatibility with existing FeatureEngineer

## Risk Mitigation

- **External API failures**: Implement retry logic and fallbacks
- **Data inconsistency**: Validate all data at provider boundaries
- **Performance issues**: Cache aggressively, optimize hot paths
- **Testing complexity**: Use mocks and property-based testing
- **Integration issues**: Maintain backwards compatibility
