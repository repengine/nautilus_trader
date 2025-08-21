"""
Instrument metadata provider for static covariates.

This module provides access to instrument specifications and metadata that remain
constant over time, making them ideal for TFT static covariates.

"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import polars as pl

from ml.data.providers.base import BaseStaticProvider


if TYPE_CHECKING:
    from ml.data.sources.metadata import MetadataSource


logger = logging.getLogger(__name__)


class InstrumentMetadataProvider(BaseStaticProvider):
    """
    Provider for instrument specifications and metadata.

    Provides static instrument attributes like tick_size, lot_size,
    exchange, asset_class, currency, etc. These features are ideal
    for TFT models' static covariate input.

    Attributes
    ----------
    source : MetadataSource
        Data source for instrument metadata
    _cache : dict[str, pl.DataFrame]
        Cache of loaded metadata by instrument set

    """

    def __init__(self, source: MetadataSource) -> None:
        """
        Initialize metadata provider.

        Parameters
        ----------
        source : MetadataSource
            Data source for instrument metadata

        """
        super().__init__()
        self.source = source
        self._cache: dict[str, pl.DataFrame] = {}
        logger.info(f"Initialized InstrumentMetadataProvider with {source.__class__.__name__}")

    def load_metadata(self, instruments: list[str]) -> pl.DataFrame:
        """
        Load metadata for specified instruments.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
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
            - fee_class: str
            - market_segment: str

        Raises
        ------
        ValueError
            If metadata format is invalid

        """
        # Generate cache key from sorted instruments
        cache_key = self._generate_cache_key(instruments)

        # Check cache first
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            logger.debug(f"Cache hit for {len(instruments)} instruments")
            return cached_data

        logger.info(f"Loading metadata for {len(instruments)} instruments")

        # Load from source
        try:
            data = self.source.fetch_metadata(instruments)
        except Exception as e:
            logger.error(f"Failed to fetch metadata: {e}")
            # Return empty DataFrame with schema
            return self._empty_metadata_frame(instruments)

        # Validate data
        if not self.validate_data(data):
            logger.warning("Invalid metadata format, returning default values")
            return self._empty_metadata_frame(instruments)

        # Ensure all requested instruments are present
        data = self._ensure_all_instruments(data, instruments)

        # Cache and return
        self._cache[cache_key] = data
        logger.info(f"Cached metadata for {len(instruments)} instruments")

        return data

    def validate_data(self, data: pl.DataFrame) -> bool:
        """
        Validate metadata DataFrame schema and content.

        Parameters
        ----------
        data : pl.DataFrame
            Data to validate

        Returns
        -------
        bool
            True if valid, False otherwise

        """
        if data.is_empty():
            return False

        # Check required columns
        required_columns = {
            "instrument_id",
            "tick_size",
            "lot_size",
            "exchange",
            "asset_class",
            "currency",
        }

        if not required_columns.issubset(data.columns):
            missing = required_columns - set(data.columns)
            logger.warning(f"Missing required columns: {missing}")
            return False

        # Check for nulls in critical fields
        critical_fields = ["instrument_id", "tick_size", "lot_size"]
        for field in critical_fields:
            if data[field].null_count() > 0:
                logger.warning(f"Null values found in critical field: {field}")
                return False

        # Validate numeric ranges
        if (data["tick_size"] <= 0).any():
            logger.warning("Invalid tick_size values found (must be > 0)")
            return False

        if (data["lot_size"] <= 0).any():
            logger.warning("Invalid lot_size values found (must be > 0)")
            return False

        return True

    def get_schema(self) -> dict[str, type]:
        """
        Return expected metadata schema.

        Returns
        -------
        dict[str, type]
            Column names and their expected types

        """
        return {
            "instrument_id": str,
            "tick_size": float,
            "lot_size": float,
            "contract_size": float,
            "min_price_increment": float,
            "exchange": str,
            "asset_class": str,
            "currency": str,
            "margin_initial": float,
            "margin_maintenance": float,
            "fee_class": str,
            "market_segment": str,
        }

    def _generate_cache_key(self, instruments: list[str]) -> str:
        """
        Generate cache key from instrument list.

        Parameters
        ----------
        instruments : list[str]
            List of instruments

        Returns
        -------
        str
            Cache key

        """
        # Sort for consistency
        sorted_instruments = sorted(instruments)
        key_string = "_".join(sorted_instruments)

        # Hash if too long
        if len(key_string) > 100:
            return hashlib.sha256(key_string.encode()).hexdigest()

        return key_string

    def _empty_metadata_frame(self, instruments: list[str]) -> pl.DataFrame:
        """
        Create empty metadata DataFrame with default values.

        Parameters
        ----------
        instruments : list[str]
            Instrument identifiers

        Returns
        -------
        pl.DataFrame
            DataFrame with default metadata values

        """
        return pl.DataFrame(
            {
                "instrument_id": instruments,
                "tick_size": [0.01] * len(instruments),
                "lot_size": [100.0] * len(instruments),
                "contract_size": [1.0] * len(instruments),
                "min_price_increment": [0.01] * len(instruments),
                "exchange": ["UNKNOWN"] * len(instruments),
                "asset_class": ["EQUITY"] * len(instruments),
                "currency": ["USD"] * len(instruments),
                "margin_initial": [0.0] * len(instruments),
                "margin_maintenance": [0.0] * len(instruments),
                "fee_class": ["DEFAULT"] * len(instruments),
                "market_segment": ["UNKNOWN"] * len(instruments),
            },
        )

    # Implement BaseStaticProvider abstract to satisfy typing/instantiation
    def _load_metadata_impl(self, instruments: list[str]) -> pl.DataFrame:
        """
        Delegate to the underlying source with safe fallback.

        This method exists to satisfy the abstract method on the base class; the
        public `load_metadata` on this provider includes additional validation and
        caching behavior tailored to instrument metadata.

        """
        try:
            return self.source.fetch_metadata(instruments)
        except Exception:
            return self._empty_metadata_frame(instruments)

    def _ensure_all_instruments(
        self,
        data: pl.DataFrame,
        instruments: list[str],
    ) -> pl.DataFrame:
        """
        Ensure all requested instruments are in the data.

        Missing instruments get default values.

        Parameters
        ----------
        data : pl.DataFrame
            Loaded metadata
        instruments : list[str]
            Requested instruments

        Returns
        -------
        pl.DataFrame
            Data with all requested instruments

        """
        existing = set(data["instrument_id"].to_list())
        missing = set(instruments) - existing

        if missing:
            logger.warning(f"Missing metadata for {len(missing)} instruments, using defaults")
            missing_df = self._empty_metadata_frame(list(missing))

            # Ensure schemas match before concatenation
            all_columns = set(data.columns) | set(missing_df.columns)

            # Add missing columns to data
            for col in all_columns - set(data.columns):
                col_type = missing_df[col].dtype
                data = data.with_columns(pl.lit(None).cast(col_type).alias(col))

            # Add missing columns to missing_df
            for col in all_columns - set(missing_df.columns):
                col_type = data[col].dtype
                missing_df = missing_df.with_columns(pl.lit(None).cast(col_type).alias(col))

            # Ensure column order matches
            column_order = sorted(all_columns)
            data = data.select(column_order)
            missing_df = missing_df.select(column_order)

            data = pl.concat([data, missing_df])

        # Filter to only requested instruments
        return data.filter(pl.col("instrument_id").is_in(instruments))
