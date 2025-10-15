#!/usr/bin/env python3
"""
FRED (Federal Reserve Economic Data) loader for Nautilus Trader ML.

Provides integration with FRED API to fetch and store key economic indicators for
trading strategies. Supports both real-time updates and historical backfill with proper
caching and rate limiting.

"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from nautilus_trader.model.identifiers import InstrumentId

from ml._imports import HAS_FREDAPI
from ml._imports import HAS_POLARS
from ml._imports import HAS_PROMETHEUS
from ml._imports import check_ml_dependencies
from ml._imports import fredapi as _fredapi
from ml._imports import pl
from ml.ml_types import PolarsDF
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.stores.data_store import DataStore


if TYPE_CHECKING:
    from ml.registry.data_registry import DataRegistry

if not HAS_POLARS:
    check_ml_dependencies(["polars"])  # Ensure clear message

if not HAS_FREDAPI:
    # Defer hard failure until use, so dependent modules can import
    _fredapi = None


logger = logging.getLogger(__name__)

# Prometheus metrics
# Declare metric variables with explicit Any type; assign within branches below
data_fetch_counter: Any
data_fetch_duration_histogram: Any
cache_hit_counter: Any
api_error_counter: Any

if HAS_PROMETHEUS:
    # Create FRED-specific metrics using bootstrap + centralized histogram
    from ml.common.metrics import data_collection_duration
    from ml.common.metrics_bootstrap import get_counter

    fred_fetch_counter = get_counter(
        "nautilus_ml_fred_fetch_total",
        "Total FRED API fetches",
        ["series"],
    )
    fred_cache_hit_counter = get_counter(
        "nautilus_ml_fred_cache_hits_total",
        "FRED cache hits",
        ["series"],
    )
    fred_api_error_counter = get_counter(
        "nautilus_ml_fred_api_errors_total",
        "FRED API errors",
        ["error_type"],
    )

    class _CounterLike(Protocol):
        def labels(self, **kwargs: object) -> object: ...
        def inc(self, _amount: float = 1) -> None: ...

    class _HistogramLike(Protocol):
        def labels(self, **kwargs: object) -> object: ...
        def observe(self, _amount: float) -> None: ...

    # Assign external metric objects (prometheus)
    data_fetch_counter = fred_fetch_counter
    data_fetch_duration_histogram = data_collection_duration
    cache_hit_counter = fred_cache_hit_counter
    api_error_counter = fred_api_error_counter

if not HAS_PROMETHEUS:
    # Create no-op metrics
    class NoOpMetric:
        def labels(self, **kwargs: object) -> object:
            return self

        def inc(self, _amount: float = 1) -> None:
            pass

        def observe(self, _amount: float) -> None:
            pass

    data_fetch_counter = NoOpMetric()
    data_fetch_duration_histogram = NoOpMetric()
    cache_hit_counter = NoOpMetric()
    api_error_counter = NoOpMetric()


@dataclass(frozen=True)
class FREDIndicator:
    """
    Configuration for a FRED economic indicator.

    Attributes
    ----------
    series_id : str
        FRED series identifier (e.g., 'DGS10' for 10-year Treasury)
    name : str
        Human-readable name for the indicator
    category : str
        Category of indicator (interest_rates, volatility, economic, market_breadth, currency)
    frequency : str
        Data frequency (daily, weekly, monthly, quarterly)
    units : str
        Units of measurement
    seasonal_adjustment : str
        Seasonal adjustment type (SA, NSA, SAAR)
    start_date : str | None
        Earliest available data date
    description : str
        Detailed description of the indicator

    """

    series_id: str
    name: str
    category: str
    frequency: str = "daily"
    units: str = "percent"
    seasonal_adjustment: str = "NSA"
    start_date: str | None = None
    description: str = ""


@dataclass(frozen=True)
class FREDConfig:
    """
    Configuration for FRED data loader.

    Attributes
    ----------
    api_key : str | None
        FRED API key (defaults to environment variable FRED_API_KEY)
    cache_dir : Path
        Directory for caching FRED data
    cache_ttl_hours : int
        Cache time-to-live in hours
    rate_limit_calls : int
        Maximum API calls per minute
    backfill_years : int
        Number of years to backfill for historical data
    update_interval_hours : int
        Hours between updates for real-time mode
    max_retries : int
        Maximum retries for failed API calls
    retry_delay_seconds : float
        Delay between retries

    """

    api_key: str | None = None
    cache_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "fred_cache")
    cache_ttl_hours: int = 24
    rate_limit_calls: int = 120  # FRED limit is 120 calls/minute
    backfill_years: int = 10
    update_interval_hours: int = 6
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.
        """
        if self.api_key is None:
            object.__setattr__(self, "api_key", os.getenv("FRED_API_KEY"))
            if not self.api_key:
                raise ValueError(
                    "FRED API key not provided. Set FRED_API_KEY environment variable "
                    "or pass api_key to FREDConfig",
                )

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class FREDDataLoader:
    """
    FRED data loader for economic indicators.

    Fetches and stores key economic indicators from FRED API with support for:
    - Interest rates (Treasury yields, Fed funds rate)
    - Volatility indices (VIX, MOVE)
    - Economic indicators (GDP, CPI, unemployment, consumer sentiment)
    - Market breadth indicators
    - Currency indices (DXY)

    Examples
    --------
    >>> config = FREDConfig(api_key="your_key")
    >>> loader = FREDDataLoader(config)
    >>>
    >>> # Fetch all configured indicators
    >>> data = loader.fetch_all_indicators()
    >>>
    >>> # Fetch specific indicator
    >>> treasury_10y = loader.fetch_indicator("DGS10")
    >>>
    >>> # Store in DataStore with registration
    >>> loader.store_indicators(data_store, data_registry)

    """

    # Default indicators configuration
    DEFAULT_INDICATORS = [
        # Interest Rates
        FREDIndicator(
            series_id="DGS1",
            name="1-Year Treasury Rate",
            category="interest_rates",
            frequency="daily",
            units="percent",
            description="Market yield on U.S. Treasury securities at 1-year constant maturity",
        ),
        FREDIndicator(
            series_id="DGS2",
            name="2-Year Treasury Rate",
            category="interest_rates",
            frequency="daily",
            units="percent",
            description="Market yield on U.S. Treasury securities at 2-year constant maturity",
        ),
        FREDIndicator(
            series_id="DGS10",
            name="10-Year Treasury Rate",
            category="interest_rates",
            frequency="daily",
            units="percent",
            description="Market yield on U.S. Treasury securities at 10-year constant maturity",
        ),
        FREDIndicator(
            series_id="DGS30",
            name="30-Year Treasury Rate",
            category="interest_rates",
            frequency="daily",
            units="percent",
            description="Market yield on U.S. Treasury securities at 30-year constant maturity",
        ),
        FREDIndicator(
            series_id="FEDFUNDS",
            name="Federal Funds Rate",
            category="interest_rates",
            frequency="daily",
            units="percent",
            description="Effective federal funds rate",
        ),
        FREDIndicator(
            series_id="SOFR",
            name="SOFR Rate",
            category="interest_rates",
            frequency="daily",
            units="percent",
            description="Secured Overnight Financing Rate",
        ),
        # Volatility Indices
        FREDIndicator(
            series_id="VIXCLS",
            name="VIX Index",
            category="volatility",
            frequency="daily",
            units="index",
            description="CBOE Volatility Index: VIX",
        ),
        # Economic Indicators - GDP
        FREDIndicator(
            series_id="GDP",
            name="Gross Domestic Product",
            category="economic",
            frequency="quarterly",
            units="billions_dollars",
            seasonal_adjustment="SAAR",
            description="Gross Domestic Product, seasonally adjusted annual rate",
        ),
        FREDIndicator(
            series_id="GDPC1",
            name="Real GDP",
            category="economic",
            frequency="quarterly",
            units="billions_chained_2017_dollars",
            seasonal_adjustment="SAAR",
            description="Real Gross Domestic Product, seasonally adjusted annual rate",
        ),
        # Economic Indicators - Inflation
        FREDIndicator(
            series_id="CPIAUCSL",
            name="Consumer Price Index",
            category="economic",
            frequency="monthly",
            units="index_1982_1984",
            seasonal_adjustment="SA",
            description="Consumer Price Index for All Urban Consumers: All Items",
        ),
        FREDIndicator(
            series_id="CPILFESL",
            name="Core CPI",
            category="economic",
            frequency="monthly",
            units="index_1982_1984",
            seasonal_adjustment="SA",
            description="Consumer Price Index: All Items Less Food and Energy",
        ),
        FREDIndicator(
            series_id="PCEPI",
            name="PCE Price Index",
            category="economic",
            frequency="monthly",
            units="index_2017",
            seasonal_adjustment="SA",
            description="Personal Consumption Expenditures: Chain-type Price Index",
        ),
        # Economic Indicators - Employment
        FREDIndicator(
            series_id="UNRATE",
            name="Unemployment Rate",
            category="economic",
            frequency="monthly",
            units="percent",
            seasonal_adjustment="SA",
            description="Civilian Unemployment Rate",
        ),
        FREDIndicator(
            series_id="PAYEMS",
            name="Nonfarm Payrolls",
            category="economic",
            frequency="monthly",
            units="thousands_persons",
            seasonal_adjustment="SA",
            description="All Employees: Total Nonfarm",
        ),
        FREDIndicator(
            series_id="CIVPART",
            name="Labor Force Participation Rate",
            category="economic",
            frequency="monthly",
            units="percent",
            seasonal_adjustment="SA",
            description="Civilian Labor Force Participation Rate",
        ),
        # Economic Indicators - Consumer
        FREDIndicator(
            series_id="UMCSENT",
            name="Consumer Sentiment",
            category="economic",
            frequency="monthly",
            units="index_1966Q1",
            seasonal_adjustment="NSA",
            description="University of Michigan: Consumer Sentiment",
        ),
        FREDIndicator(
            series_id="RSXFS",
            name="Retail Sales",
            category="economic",
            frequency="monthly",
            units="millions_dollars",
            seasonal_adjustment="SA",
            description="Advance Retail Sales: Retail Trade and Food Services",
        ),
        # Economic Indicators - Housing
        FREDIndicator(
            series_id="HOUST",
            name="Housing Starts",
            category="economic",
            frequency="monthly",
            units="thousands_units",
            seasonal_adjustment="SAAR",
            description="New Privately-Owned Housing Units Started",
        ),
        FREDIndicator(
            series_id="MORTGAGE30US",
            name="30-Year Mortgage Rate",
            category="interest_rates",
            frequency="weekly",
            units="percent",
            description="30-Year Fixed Rate Mortgage Average in the United States",
        ),
        # Currency
        FREDIndicator(
            series_id="DEXUSEU",
            name="USD/EUR Exchange Rate",
            category="currency",
            frequency="daily",
            units="euros_per_dollar",
            description="U.S. Dollars to Euro Spot Exchange Rate",
        ),
        FREDIndicator(
            series_id="DTWEXBGS",
            name="Trade Weighted Dollar Index",
            category="currency",
            frequency="daily",
            units="index_2006",
            description="Trade Weighted U.S. Dollar Index: Broad, Goods and Services",
        ),
        # Credit Spreads
        FREDIndicator(
            series_id="BAMLH0A0HYM2",
            name="High Yield Spread",
            category="interest_rates",
            frequency="daily",
            units="percent",
            description="ICE BofA US High Yield Index Option-Adjusted Spread",
        ),
        FREDIndicator(
            series_id="BAMLC0A0CM",
            name="Investment Grade Spread",
            category="interest_rates",
            frequency="daily",
            units="percent",
            description="ICE BofA US Corporate Index Option-Adjusted Spread",
        ),
    ]

    def __init__(
        self,
        config: FREDConfig | None = None,
        indicators: list[FREDIndicator] | None = None,
    ) -> None:
        """
        Initialize FRED data loader.

        Parameters
        ----------
        config : FREDConfig | None
            Configuration for the loader. If None, uses defaults.
        indicators : list[FREDIndicator] | None
            List of indicators to fetch. If None, uses DEFAULT_INDICATORS.

        Raises
        ------
        ImportError
            If fredapi package is not installed
        ValueError
            If FRED API key is not provided

        """
        self.config = config or FREDConfig()
        self.indicators = indicators or self.DEFAULT_INDICATORS
        # Lazily initialize fred client to ease typing across branches
        if not HAS_FREDAPI or _fredapi is None:  # pragma: no cover - import-time branch
            raise ImportError(
                "fredapi package required for FRED data loading. Install with: pip install fredapi",
            )
        self.fred = _fredapi.Fred(api_key=self.config.api_key)

        # Rate limiting
        self._last_call_time = 0.0
        self._call_count = 0
        self._rate_limit_window = 60.0  # 1 minute window

        # Cache management

        self._cache: dict[str, tuple[PolarsDF, float]] = {}

        logger.info(
            f"Initialized FRED loader with {len(self.indicators)} indicators, "
            f"cache_dir={self.config.cache_dir}",
        )

    def _rate_limit(self) -> None:
        """
        Implement rate limiting for FRED API calls.
        """
        current_time = time.time()

        # Reset counter if window has passed
        if current_time - self._last_call_time > self._rate_limit_window:
            self._call_count = 0
            self._last_call_time = current_time

        # Check if we need to wait
        if self._call_count >= self.config.rate_limit_calls:
            wait_time = self._rate_limit_window - (current_time - self._last_call_time)
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self._call_count = 0
                self._last_call_time = time.time()

        self._call_count += 1

    def _get_cache_path(self, series_id: str) -> Path:
        """
        Get cache file path for a series.
        """
        return self.config.cache_dir / f"{series_id}.parquet"

    def _get_cache_metadata_path(self, series_id: str) -> Path:
        """
        Get cache metadata file path for a series.
        """
        return self.config.cache_dir / f"{series_id}_metadata.json"

    def _is_cache_valid(self, series_id: str) -> bool:
        """
        Check if cached data is still valid.

        Parameters
        ----------
        series_id : str
            FRED series identifier

        Returns
        -------
        bool
            True if cache is valid, False otherwise

        """
        cache_path = self._get_cache_path(series_id)
        metadata_path = self._get_cache_metadata_path(series_id)

        if not cache_path.exists() or not metadata_path.exists():
            return False

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
                if not isinstance(metadata, dict):  # Defensive cast for typing
                    return False

            cache_time = metadata.get("timestamp", 0)
            ttl_seconds = self.config.cache_ttl_hours * 3600

            return (time.time() - float(cache_time)) < float(ttl_seconds)

        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False

    def _load_from_cache(self, series_id: str) -> PolarsDF | None:
        """
        Load data from cache if valid.

        Parameters
        ----------
        series_id : str
            FRED series identifier

        Returns
        -------
        pl.DataFrame | None
            Cached data if valid, None otherwise

        """
        if not self._is_cache_valid(series_id):
            return None

        try:
            cache_path = self._get_cache_path(series_id)
            _pl = pl
            assert _pl is not None
            df = _pl.read_parquet(cache_path)

            cache_hit_counter.labels(series=series_id).inc()
            logger.debug(f"Loaded {series_id} from cache")

            from typing import cast as _cast

            return _cast(PolarsDF, df)

        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
            return None

    def _save_to_cache(self, series_id: str, df: PolarsDF) -> None:
        """
        Save data to cache.

        Parameters
        ----------
        series_id : str
            FRED series identifier
        df : pl.DataFrame
            Data to cache

        """
        try:
            cache_path = self._get_cache_path(series_id)
            metadata_path = self._get_cache_metadata_path(series_id)

            # Save data
            df.write_parquet(cache_path)

            # Save metadata
            metadata = {
                "series_id": series_id,
                "timestamp": time.time(),
                "rows": len(df),
                "columns": list(df.columns),
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Cached {series_id}: {len(df)} rows")

        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def fetch_indicator(
        self,
        series_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        use_cache: bool = True,
    ) -> PolarsDF:
        """
        Fetch a single FRED indicator.

        Parameters
        ----------
        series_id : str
            FRED series identifier
        start_date : datetime | None
            Start date for data (defaults to backfill_years ago)
        end_date : datetime | None
            End date for data (defaults to today)
        use_cache : bool
            Whether to use cached data if available

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: timestamp, series_id, value

        """
        # Check cache first
        if use_cache:
            cached_df = self._load_from_cache(series_id)
            if cached_df is not None:
                return cached_df

        # Set default dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * self.config.backfill_years)

        # Fetch from FRED API with retries (typed helper)
        start_time = time.time()

        def _fetch_once() -> Any:
            self._rate_limit()
            return self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )

        def _on_exc(attempt: int, exc: BaseException) -> None:
            logger.warning(
                f"Error fetching {series_id} (attempt {attempt + 1}/{self.config.max_retries}): {exc}",
            )
            api_error_counter.labels(error_type=type(exc).__name__).inc()

        try:
            from ml.common.retry_utils import retry_with_backoff as _retry

            series_data = _retry(
                _fetch_once,
                max_attempts=int(self.config.max_retries),
                initial_delay=float(self.config.retry_delay_seconds),
                multiplier=2.0,
                max_delay=60.0,
                jitter=0.0,
                sleep_fn=time.sleep,
                on_exception=_on_exc,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch {series_id} after {self.config.max_retries} attempts: {e}",
            ) from e

        # Convert to Polars DataFrame
        _pl = pl
        assert _pl is not None
        df = _pl.DataFrame(
            {
                "timestamp": series_data.index,
                "series_id": series_id,
                "value": series_data.values,
            },
        )

        # Convert timestamp to nanoseconds
        df = df.with_columns(_pl.col("timestamp").dt.timestamp("ns").alias("timestamp_ns"))

        # Remove any null values
        df = df.filter(_pl.col("value").is_not_null())

        # Sort by timestamp
        df = df.sort("timestamp")

        # Record metrics
        duration = time.time() - start_time
        data_fetch_counter.labels(series=series_id).inc()
        data_fetch_duration_histogram.labels(source="fred", schema="economic").observe(
            duration,
        )

        logger.info(
            f"Fetched {series_id}: {len(df)} rows, "
            f"range={df['timestamp'].min()!s} to {df['timestamp'].max()!s}",
        )

        # Save to cache
        if use_cache:
            self._save_to_cache(series_id, df)

        from typing import cast as _cast

        return _cast(PolarsDF, df)

    def fetch_all_indicators(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        use_cache: bool = True,
    ) -> dict[str, PolarsDF]:
        """
        Fetch all configured indicators.

        Parameters
        ----------
        start_date : datetime | None
            Start date for data
        end_date : datetime | None
            End date for data
        use_cache : bool
            Whether to use cached data

        Returns
        -------
        dict[str, pl.DataFrame]
            Dictionary mapping series_id to DataFrame

        """
        results: dict[str, PolarsDF] = {}

        for indicator in self.indicators:
            try:
                df = self.fetch_indicator(
                    indicator.series_id,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache,
                )
                results[indicator.series_id] = df

            except Exception as e:
                logger.error(f"Failed to fetch {indicator.series_id}: {e}")
                # Continue with other indicators

        logger.info(f"Fetched {len(results)}/{len(self.indicators)} indicators")
        return results

    def combine_indicators(self, data: dict[str, PolarsDF]) -> PolarsDF:
        """
        Combine multiple indicators into a single DataFrame.

        Parameters
        ----------
        data : dict[str, pl.DataFrame]
            Dictionary mapping series_id to DataFrame

        Returns
        -------
        pl.DataFrame
            Combined DataFrame with all indicators

        """
        _pl = pl
        assert _pl is not None
        if not data:
            from typing import cast as _cast

            return _cast(PolarsDF, _pl.DataFrame())

        # Start with first indicator
        combined: PolarsDF | None = None

        from typing import cast as _cast

        for series_id, df in data.items():
            # Pivot to wide format
            wide_df = _cast(
                PolarsDF,
                df.select(
                    [
                        "timestamp",
                        _pl.col("value").alias(series_id),
                    ],
                ),
            )

            if combined is None:
                combined = wide_df
            else:
                # Join on timestamp using 'full' instead of deprecated 'outer'
                combined = _cast(
                    PolarsDF,
                    combined.join(
                        wide_df,
                        on="timestamp",
                        how="full",
                        coalesce=True,  # Coalesce duplicate columns
                    ),
                )

        # Sort by timestamp and filter out null timestamps
        if combined is not None:
            # Remove rows with null timestamps (shouldn't happen with coalesce=True)
            combined = combined.filter(_pl.col("timestamp").is_not_null())

            # Sort by timestamp
            combined = combined.sort("timestamp")

            # Add timestamp_ns column
            combined = combined.with_columns(
                _pl.col("timestamp").dt.timestamp("ns").alias("timestamp_ns"),
            )

        if combined is not None:
            return combined
        return _pl.DataFrame()

    def store_indicators(
        self,
        data_store: DataStore,
        data_registry: DataRegistry,
        data: dict[str, PolarsDF] | None = None,
    ) -> None:
        """
        Store indicators in DataStore with proper registration.

        Parameters
        ----------
        data_store : DataStore
            DataStore instance for persistence
        data_registry : DataRegistry
            DataRegistry for dataset registration
        data : dict[str, pl.DataFrame] | None
            Pre-fetched data, or None to fetch all indicators

        """
        # Fetch data if not provided
        if data is None:
            data = self.fetch_all_indicators()

        if not data:
            logger.warning("No data to store")
            return

        # Combine indicators
        combined_df = self.combine_indicators(data)

        if combined_df.is_empty():
            logger.warning("Combined DataFrame is empty")
            return

        # Create dataset manifest
        dataset_id = "fred_economic_indicators"

        # Calculate schema hash
        schema_str = json.dumps(sorted(combined_df.columns), sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]

        # Create data contract
        contract = DataContract(
            contract_id=f"{dataset_id}_contract_v1",
            dataset_id=dataset_id,
            version="1.0.0",
            validation_rules=[
                ValidationRule(
                    rule_type=ValidationRuleType.NULLABILITY,
                    field_name="timestamp",
                    parameters={"allow_null": False},
                    severity=QualityFlag.FAIL,
                    description="Timestamp must not be null",
                ),
                ValidationRule(
                    rule_type=ValidationRuleType.RANGE,
                    field_name="timestamp_ns",
                    parameters={
                        "min": 0,
                        "max": int(2e18),  # Year ~2033
                    },
                    severity=QualityFlag.FAIL,
                    description="Timestamp must be in valid range",
                ),
            ],
            quality_thresholds={
                "null_rate": 0.01,
                "duplicate_rate": 0.0,
            },
            enforcement_mode="lenient",
            metadata={
                "schema_hash": schema_hash,
                "columns": list(combined_df.columns),
            },
        )

        # Create dataset manifest
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            dataset_type=DatasetType.FEATURES,  # Economic indicators are features
            storage_kind=StorageKind.PARQUET,
            location=str(self.config.cache_dir),
            partitioning={},
            retention_days=365,
            schema={
                "instrument_id": "str",  # Required for Nautilus
                "ts_event": "int64",  # Required for Nautilus (use timestamp_ns)
                "ts_init": "int64",  # Required for Nautilus (use timestamp_ns)
                "timestamp": "datetime64[ns]",
                "timestamp_ns": "int64",
                **{
                    col: "float64"
                    for col in combined_df.columns
                    if col not in ["timestamp", "timestamp_ns"]
                },
            },
            ts_field="ts_event",  # Use standard Nautilus field name
            seq_field=None,
            primary_keys=["ts_event"],
            schema_hash=schema_hash,
            constraints={
                "nullability": {
                    "timestamp": False,
                    "timestamp_ns": False,
                },
                "ranges": {
                    "timestamp_ns": {"min": 0, "max": int(2e18)},
                },
            },
            lineage=[],
            pipeline_signature="fred_loader_v1",
            version="1.0.0",
            metadata={
                "description": "FRED economic indicators for ML features",
                "source": "fred",
                "indicators": [ind.series_id for ind in self.indicators],
                "categories": list({ind.category for ind in self.indicators}),
                "last_update": datetime.now().isoformat(),
                "contract": contract.contract_id,
            },
        )

        # Register dataset
        data_registry.register_dataset(manifest)

        # Store in DataStore
        # Convert to numpy for storage
        timestamps = combined_df["timestamp_ns"].to_numpy()

        # Store each indicator as a separate feature
        _pl = pl
        assert _pl is not None
        for column in combined_df.columns:
            if column in ["timestamp", "timestamp_ns"]:
                continue

            values = combined_df[column].to_numpy()

            # Skip if all values are null
            if np.all(np.isnan(values)):
                continue

            # Create pseudo instrument_id for economic indicators
            instrument_id = InstrumentId.from_str(f"FRED.{column}")

            # Store as ingestion data
            data_store.write_ingestion(
                dataset_id=dataset_id,
                records=_pl.DataFrame(
                    {
                        "instrument_id": str(instrument_id),
                        "timestamp_ns": timestamps,
                        "value": values,
                    },
                ),
                source="fred",
                run_id=f"fred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                instrument_id=str(instrument_id),
            )

        logger.info(
            f"Stored {len(combined_df.columns) - 2} indicators to DataStore, "
            f"dataset_id={dataset_id}",
        )

    def update_realtime(
        self,
        data_store: DataStore,
        data_registry: DataRegistry,
    ) -> None:
        """
        Update indicators in real-time mode.

        Fetches only recent data to update existing indicators.

        Parameters
        ----------
        data_store : DataStore
            DataStore instance
        data_registry : DataRegistry
            DataRegistry instance

        """
        # Fetch last 30 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        logger.info(f"Updating FRED indicators from {start_date} to {end_date}")

        data = self.fetch_all_indicators(
            start_date=start_date,
            end_date=end_date,
            use_cache=False,  # Don't use cache for real-time updates
        )

        self.store_indicators(data_store, data_registry, data)

    def export_ml_parquet(
        self,
        *,
        data: dict[str, PolarsDF] | None = None,
        out_path: Path | None = None,
    ) -> Path:
        """
        Export indicators in ML long format parquet for dataset builder joins.

        When ``data`` is None, fetches using current configuration. The output schema is
        columns: ``timestamp`` (datetime[ns]), ``series_id`` (str), ``value`` (float).

        Parameters
        ----------
        data : dict[str, PolarsDF] | None
            Pre-fetched indicator frames keyed by series id.
        out_path : Path | None
            Destination path. Defaults to data/fred/fred_indicators_ml_format.parquet

        Returns
        -------
        Path
            The written parquet file path.

        """
        _pl = pl
        assert _pl is not None

        target = out_path or Path("data/fred/fred_indicators_ml_format.parquet")
        target.parent.mkdir(parents=True, exist_ok=True)

        if data is None:
            data = self.fetch_all_indicators(use_cache=True)

        # Build long format rows
        frames: list[PolarsDF] = []
        for series_id, df in data.items():
            if df.is_empty():
                continue
            # Ensure timestamp exists
            cur = df
            if "timestamp" not in cur.columns and "timestamp_ns" in cur.columns:
                cur = cur.with_columns(_pl.from_epoch("timestamp_ns", unit="ns").alias("timestamp"))
            # Select and rename
            if "value" not in cur.columns:
                # Some sources might use series_id as value column name; skip those
                continue
            cur2 = cur.select(["timestamp", "value"]).with_columns(
                [_pl.lit(series_id).alias("series_id")],
            )
            frames.append(cur2)

        if not frames:
            # Write an empty file with schema to keep downstream happy
            empty = _pl.DataFrame({"timestamp": [], "series_id": [], "value": []})
            empty.write_parquet(target)
            return target

        out = _pl.concat(frames, how="vertical")
        # Sort by time for efficient asof join
        out = out.sort("timestamp")
        out.write_parquet(target)
        return target
