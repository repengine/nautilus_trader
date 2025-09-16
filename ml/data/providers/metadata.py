"""
Instrument metadata provider for static covariates.

This module provides access to instrument specifications and metadata that remain
constant over time, making them ideal for TFT static covariates.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from typing import cast as _cast

from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.data.providers.base import BaseStaticProvider
from ml.data.sources.metadata import default_metadata


if TYPE_CHECKING:
    from polars import DataFrame as PlDataFrame

    from ml.data.sources.metadata import MetadataSource
else:  # pragma: no cover - typing only
    PlDataFrame = Any  # type: ignore[assignment]


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
    _cache : dict[str, PlDataFrame]
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
        logger.info(f"Initialized InstrumentMetadataProvider with {source.__class__.__name__}")

    def _load_metadata_impl(self, instruments: list[str]) -> PlDataFrame:
        """
        Load metadata for specified instruments (implementation for BaseStaticProvider).

        Returns a DataFrame with expected schema or an empty frame on error.

        """
        if pl is None:
            check_ml_dependencies(["polars"])  # Ensure Polars present when used

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
        return data

    # Backwards compatibility for tests expecting get_metadata
    def get_metadata(self, instruments: list[str]) -> PlDataFrame:
        """
        Alias for load_metadata for backwards compatibility in tests.
        """
        return self.load_metadata(instruments)

    def validate_data(self, data: PlDataFrame) -> bool:
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
        Deprecated: cache key generation now handled by BaseStaticProvider.
        Retained for compatibility where referenced.
        """
        return "_".join(sorted(instruments))

    def _empty_metadata_frame(self, instruments: list[str]) -> PlDataFrame:
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
        if pl is None:
            check_ml_dependencies(["polars"])  # Ensure Polars present when used
        rows = [default_metadata(sym) for sym in instruments]
        return _cast(PlDataFrame, _cast(Any, pl).DataFrame(rows))

    # (Removed legacy duplicate _load_metadata_impl)

    def _ensure_all_instruments(
        self,
        data: PlDataFrame,
        instruments: list[str],
    ) -> PlDataFrame:
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
        existing = set(_cast(Any, data)["instrument_id"].to_list())
        missing = set(instruments) - existing

        if missing:
            logger.warning(f"Missing metadata for {len(missing)} instruments, using defaults")
            missing_df = self._empty_metadata_frame(list(missing))

            # Ensure schemas match before concatenation
            all_columns = set(_cast(Any, data).columns) | set(_cast(Any, missing_df).columns)

            # Add missing columns to data
            PL = _cast(Any, pl)
            for col in all_columns - set(_cast(Any, data).columns):
                col_type = _cast(Any, missing_df)[col].dtype
                data = _cast(Any, data).with_columns(PL.lit(None).cast(col_type).alias(col))

            # Add missing columns to missing_df
            for col in all_columns - set(_cast(Any, missing_df).columns):
                col_type = _cast(Any, data)[col].dtype
                missing_df = _cast(Any, missing_df).with_columns(
                    PL.lit(None).cast(col_type).alias(col),
                )

            # Ensure column order matches
            column_order = sorted(all_columns)
            data = _cast(Any, data).select(column_order)
            missing_df = _cast(Any, missing_df).select(column_order)

            data = PL.concat([_cast(Any, data), _cast(Any, missing_df)])

        # Filter to only requested instruments
        PL = _cast(Any, pl)
        return _cast(
            PlDataFrame,
            _cast(Any, data).filter(PL.col("instrument_id").is_in(instruments)),
        )
