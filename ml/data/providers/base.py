"""
Base classes and protocols for data providers.

This module defines the abstract interfaces and base implementations
for all data providers, following SOLID principles.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl


if TYPE_CHECKING:
    import polars as pl


# ============================================================================
# PROTOCOLS (Interface Segregation Principle)
# ============================================================================


@runtime_checkable
class DataProvider(Protocol):
    """
    Base protocol that all data providers must implement.

    This protocol defines the minimal interface for any data provider,
    ensuring consistency across different implementations.
    """

    def load_data(
        self,
        instruments: list[str],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Load data for specified instruments and time range.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers
        start : datetime
            Start of time range
        end : datetime
            End of time range

        Returns
        -------
        pl.DataFrame
            Loaded data
        """
        ...

    def validate_data(self, data: pl.DataFrame) -> bool:
        """
        Validate that data meets schema requirements.

        Parameters
        ----------
        data : pl.DataFrame
            Data to validate

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        ...

    def get_schema(self) -> dict[str, type]:
        """
        Return the expected data schema.

        Returns
        -------
        dict[str, type]
            Mapping of column names to types
        """
        ...


@runtime_checkable
class CacheableProvider(Protocol):
    """Protocol for providers that support caching."""

    def cache_key(self, params: dict[str, Any]) -> str:
        """
        Generate cache key from parameters.

        Parameters
        ----------
        params : dict
            Parameters to generate key from

        Returns
        -------
        str
            Cache key
        """
        ...

    def from_cache(self, key: str) -> pl.DataFrame | None:
        """
        Load data from cache if available.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        pl.DataFrame | None
            Cached data or None if not found
        """
        ...

    def to_cache(self, key: str, data: pl.DataFrame) -> None:
        """
        Save data to cache.

        Parameters
        ----------
        key : str
            Cache key
        data : pl.DataFrame
            Data to cache
        """
        ...


@runtime_checkable
class StaticDataProvider(Protocol):
    """Protocol for providers of time-invariant data."""

    def load_metadata(self, instruments: list[str]) -> pl.DataFrame:
        """
        Load static metadata for instruments.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers

        Returns
        -------
        pl.DataFrame
            Metadata for instruments
        """
        ...


@runtime_checkable
class TimeSeriesProvider(Protocol):
    """Protocol for providers of time-varying data."""

    def load_timeseries(
        self,
        instruments: list[str],
        timestamps: pl.Series,
    ) -> pl.DataFrame:
        """
        Load time series data.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers
        timestamps : pl.Series
            Timestamps to load data for

        Returns
        -------
        pl.DataFrame
            Time series data
        """
        ...


# ============================================================================
# BASE IMPLEMENTATIONS (DRY Principle)
# ============================================================================


class BaseDataProvider:
    """
    Base implementation with common functionality for all providers.

    This class provides:
    - Logging setup
    - Metrics collection
    - Common validation logic
    - Error handling
    """

    def __init__(self) -> None:
        """Initialize base provider."""
        self.logger = self._setup_logging()
        self.metrics = self._setup_metrics()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the provider."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        return logger

    def _setup_metrics(self) -> dict[str, int]:
        """Set up metrics collection."""
        return defaultdict(int)

    def validate_data(self, data: pl.DataFrame) -> bool:
        """
        Validate data meets common requirements.

        Checks:
        - DataFrame is not empty
        - No nulls in required columns (instrument_id, timestamp if present)
        - Data types are correct

        Parameters
        ----------
        data : pl.DataFrame
            Data to validate

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        # Check not empty
        if len(data) == 0:
            self.logger.warning("Empty dataframe")
            return False

        # Check for required columns and nulls
        if "instrument_id" in data.columns:
            if data["instrument_id"].null_count() > 0:
                self.logger.warning("Nulls in instrument_id column")
                return False

        if "timestamp" in data.columns:
            if data["timestamp"].null_count() > 0:
                self.logger.warning("Nulls in timestamp column")
                return False

        return True

    def _handle_error(self, error: Exception) -> None:
        """
        Handle provider errors uniformly.

        Parameters
        ----------
        error : Exception
            Error to handle
        """
        self.logger.error(f"Provider error: {error}")
        self.metrics["provider_errors"] += 1


class CachedDataProvider(BaseDataProvider):
    """
    Base class with caching support.

    Implements Template Method pattern for caching logic.
    """

    def __init__(self, cache_ttl_hours: int = 24) -> None:
        """
        Initialize cached provider.

        Parameters
        ----------
        cache_ttl_hours : int, default 24
            Cache time-to-live in hours
        """
        super().__init__()
        self.cache_ttl = cache_ttl_hours
        self._cache: dict[str, pl.DataFrame] = {}

    def cache_key(self, params: dict[str, Any]) -> str:
        """
        Generate cache key from parameters.

        Uses SHA256 hash of JSON-serialized parameters.

        Parameters
        ----------
        params : dict
            Parameters to generate key from

        Returns
        -------
        str
            Cache key
        """
        # Convert datetime objects to ISO format for serialization
        serializable_params: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, datetime):
                serializable_params[key] = value.isoformat()
            elif isinstance(value, list):
                # Sort list and convert to string for consistent hashing
                serializable_params[key] = ",".join(sorted(str(v) for v in value))
            else:
                serializable_params[key] = value

        # Generate hash
        params_str = json.dumps(serializable_params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]

    def from_cache(self, key: str) -> pl.DataFrame | None:
        """
        Load data from cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        pl.DataFrame | None
            Cached data or None
        """
        return self._cache.get(key)

    def to_cache(self, key: str, data: pl.DataFrame) -> None:
        """
        Save data to cache.

        Parameters
        ----------
        key : str
            Cache key
        data : pl.DataFrame
            Data to cache
        """
        self._cache[key] = data
        self.metrics["cache_writes"] += 1

    def load_data(
        self,
        instruments: list[str],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Template method implementing caching logic.

        Parameters
        ----------
        instruments : list[str]
            List of instruments
        start : datetime
            Start time
        end : datetime
            End time

        Returns
        -------
        pl.DataFrame
            Loaded data
        """
        # Generate cache key
        params = {"instruments": instruments, "start": start, "end": end}
        key = self.cache_key(params)

        # Try cache first
        cached = self.from_cache(key)
        if cached is not None:
            self.metrics["cache_hits"] += 1
            self.logger.debug(f"Cache hit for key: {key}")
            return cached

        # Cache miss - load fresh data
        self.metrics["cache_misses"] += 1
        self.logger.debug(f"Cache miss for key: {key}")

        try:
            data = self._load_data_impl(instruments, start, end)
            # Validate before caching
            if self.validate_data(data):
                self.to_cache(key, data)
            return data
        except Exception as e:
            self._handle_error(e)
            raise

    @abstractmethod
    def _load_data_impl(
        self,
        instruments: list[str],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Abstract method to be implemented by subclasses.

        This is where the actual data loading logic goes.
        """
        raise NotImplementedError


class BaseStaticProvider(BaseDataProvider):
    """
    Base class for static data providers.

    Static data is cached indefinitely since it doesn't change.
    """

    def __init__(self) -> None:
        """Initialize static provider."""
        super().__init__()
        self._metadata_cache: dict[str, pl.DataFrame] = {}

    def load_metadata(self, instruments: list[str]) -> pl.DataFrame:
        """
        Load metadata with caching.

        Parameters
        ----------
        instruments : list[str]
            List of instruments

        Returns
        -------
        pl.DataFrame
            Metadata
        """
        # Create cache key from sorted instruments
        cache_key = "_".join(sorted(instruments))

        # Check cache
        if cache_key in self._metadata_cache:
            self.metrics["static_cache_hits"] += 1
            return self._metadata_cache[cache_key]

        # Load fresh
        self.metrics["static_cache_misses"] += 1
        try:
            data = self._load_metadata_impl(instruments)
            if self.validate_data(data):
                self._metadata_cache[cache_key] = data
            return data
        except Exception as e:
            self._handle_error(e)
            raise

    @abstractmethod
    def _load_metadata_impl(self, instruments: list[str]) -> pl.DataFrame:
        """To be implemented by subclasses."""
        raise NotImplementedError


class BaseTimeSeriesProvider(BaseDataProvider):
    """
    Base class for time series data providers.

    Provides validation for time series data.
    """

    def load_timeseries(
        self,
        instruments: list[str],
        timestamps: pl.Series,
    ) -> pl.DataFrame:
        """
        Load time series with validation.

        Parameters
        ----------
        instruments : list[str]
            List of instruments
        timestamps : pl.Series
            Timestamps

        Returns
        -------
        pl.DataFrame
            Time series data
        """
        # Validate timestamps
        if not timestamps.is_sorted():
            raise ValueError("Timestamps are not sorted")

        if timestamps.null_count() > 0:
            raise ValueError("Timestamps contain nulls")

        try:
            data = self._load_timeseries_impl(instruments, timestamps)
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            return data
        except Exception as e:
            self._handle_error(e)
            raise

    @abstractmethod
    def _load_timeseries_impl(
        self,
        instruments: list[str],
        timestamps: pl.Series,
    ) -> pl.DataFrame:
        """To be implemented by subclasses."""
        raise NotImplementedError
