"""
Metadata sources for instrument specifications.

This module provides various sources for loading instrument metadata, including
Databento, Nautilus internal data, CSV files, and mock data.

"""

from __future__ import annotations

import logging
import os
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ml._imports import check_ml_dependencies
from ml._imports import pl as pl_runtime


if TYPE_CHECKING:
    import polars as _pl

    from nautilus_trader.model.instruments import Instrument

# Local runtime alias to avoid Optional[Module] union typing at use sites
PL: Any = cast(Any, pl_runtime)


logger = logging.getLogger(__name__)


class MetadataSource(ABC):
    """
    Abstract base class for metadata sources.
    """

    @abstractmethod
    def fetch_metadata(self, instruments: list[str]) -> _pl.DataFrame:
        """
        Fetch metadata from source.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers

        Returns
        -------
        pl.DataFrame
            Metadata for requested instruments

        """
        ...


class DatabentoMetadataSource(MetadataSource):
    """
    Load metadata from Databento API.

    Uses Databento's instrument definitions to get specifications.

    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize Databento metadata source.

        Parameters
        ----------
        api_key : str, optional
            Databento API key. If None, uses DATABENTO_API_KEY env var

        """
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            logger.warning("No Databento API key found, will return defaults")

    def fetch_metadata(self, instruments: list[str]) -> _pl.DataFrame:
        """
        Fetch metadata from Databento.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers

        Returns
        -------
        pl.DataFrame
            Instrument metadata

        """
        if pl_runtime is None:
            check_ml_dependencies(["polars"])  # Ensure Polars present when used

        if not self.api_key:
            logger.warning("No API key, returning mock data")
            return MockMetadataSource().fetch_metadata(instruments)

        try:
            import databento as db

            _ = db.Historical(self.api_key)

            # Get instrument definitions
            # This is a simplified example - actual implementation would
            # need to handle Databento's specific API format
            metadata_list = []

            for symbol in instruments:
                # In practice, you'd batch these requests
                try:
                    # Get instrument info (this is pseudo-code)
                    # Real implementation would use Databento's actual API
                    info = {
                        "instrument_id": symbol,
                        "tick_size": 0.01,
                        "lot_size": 100.0,
                        "contract_size": 1.0,
                        "min_price_increment": 0.01,
                        "exchange": "XNAS",  # NASDAQ
                        "asset_class": "EQUITY",
                        "currency": "USD",
                        "margin_initial": 0.0,
                        "margin_maintenance": 0.0,
                        "fee_class": "MAKER_TAKER",
                        "market_segment": "MAIN",
                    }
                    metadata_list.append(info)
                except Exception as e:
                    logger.warning(f"Failed to get metadata for {symbol}: {e}")
                    # Add default values
                    metadata_list.append(self._default_metadata(symbol))

            from typing import cast as _cast

            return _cast("_pl.DataFrame", PL.DataFrame(metadata_list))

        except ImportError:
            logger.warning("Databento not installed, using mock data")
            return MockMetadataSource().fetch_metadata(instruments)
        except Exception as e:
            logger.error(f"Failed to fetch from Databento: {e}")
            return MockMetadataSource().fetch_metadata(instruments)

    def _default_metadata(self, symbol: str) -> dict[str, Any]:
        """
        Get default metadata for a symbol (canonical).
        """
        return default_metadata(symbol)


class NautilusMetadataSource(MetadataSource):
    """
    Load metadata from Nautilus instrument definitions.

    Uses Nautilus Trader's internal instrument specifications.

    """

    def __init__(self, instruments: dict[str, Instrument] | None = None) -> None:
        """
        Initialize Nautilus metadata source.

        Parameters
        ----------
        instruments : dict[str, Instrument], optional
            Pre-loaded Nautilus instruments

        """
        self.instruments = instruments or {}

    def fetch_metadata(self, instruments: list[str]) -> _pl.DataFrame:
        """
        Fetch metadata from Nautilus instruments.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers

        Returns
        -------
        pl.DataFrame
            Instrument metadata

        """
        metadata_list = []

        for symbol in instruments:
            if symbol in self.instruments:
                inst = self.instruments[symbol]
                metadata = self._extract_metadata(inst)
            else:
                # Use canonical defaults for unknown instruments
                metadata = default_metadata(symbol)

            metadata_list.append(metadata)

        if pl_runtime is None:
            check_ml_dependencies(["polars"])  # Ensure Polars present when used
        from typing import cast as _cast

        return _cast("_pl.DataFrame", PL.DataFrame(metadata_list))

    def _extract_metadata(self, instrument: Instrument) -> dict[str, Any]:
        """
        Extract metadata from Nautilus instrument.

        Parameters
        ----------
        instrument : Instrument
            Nautilus instrument object

        Returns
        -------
        dict
            Metadata dictionary

        """
        # Extract based on instrument type
        # This is simplified - actual implementation would check instrument type
        return {
            "instrument_id": str(instrument.id.symbol),
            "tick_size": float(instrument.price_increment),
            "lot_size": float(instrument.lot_size),
            "contract_size": float(getattr(instrument, "contract_size", 1.0)),
            "min_price_increment": float(instrument.price_increment),
            "exchange": str(instrument.id.venue),
            "asset_class": str(instrument.asset_class),
            "currency": str(instrument.quote_currency),
            "margin_initial": float(getattr(instrument, "margin_init", 0.0)),
            "margin_maintenance": float(getattr(instrument, "margin_maint", 0.0)),
            "fee_class": "DEFAULT",
            "market_segment": "MAIN",
        }


class CSVMetadataSource(MetadataSource):
    """
    Load metadata from CSV file.

    Useful for testing and offline development.

    """

    def __init__(self, file_path: str | Path) -> None:
        """
        Initialize CSV metadata source.

        Parameters
        ----------
        file_path : str or Path
            Path to CSV file with metadata

        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            logger.warning(f"CSV file not found: {self.file_path}")
            self._data = None
        else:
            if pl_runtime is None:
                check_ml_dependencies(["polars"])  # Ensure Polars present when used
            self._data = PL.read_csv(self.file_path)
            logger.info(f"Loaded metadata for {len(self._data)} instruments from CSV")

    def fetch_metadata(self, instruments: list[str]) -> _pl.DataFrame:
        """
        Fetch metadata from CSV.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers

        Returns
        -------
        pl.DataFrame
            Instrument metadata

        """
        if pl_runtime is None:
            check_ml_dependencies(["polars"])  # Ensure Polars present when used

        if self._data is None:
            logger.warning("No CSV data available, using mock source")
            return MockMetadataSource().fetch_metadata(instruments)

        # Filter to requested instruments
        filtered = self._data.filter(PL.col("instrument_id").is_in(instruments))

        # Add missing instruments with defaults
        existing = set(filtered["instrument_id"].to_list())
        missing = set(instruments) - existing

        if missing:
            logger.info(f"Adding defaults for {len(missing)} missing instruments")
            mock_source = MockMetadataSource()
            missing_df = mock_source.fetch_metadata(list(missing))

            # Ensure schemas match before concatenation
            # Get all columns from both dataframes
            all_columns = set(filtered.columns) | set(missing_df.columns)

            # Add missing columns to filtered with null values
            for col in all_columns - set(filtered.columns):
                # Get the type from missing_df
                col_type = missing_df[col].dtype
                filtered = filtered.with_columns(PL.lit(None).cast(col_type).alias(col))

            # Add missing columns to missing_df with null values
            for col in all_columns - set(missing_df.columns):
                # Get the type from filtered
                col_type = filtered[col].dtype
                missing_df = missing_df.with_columns(PL.lit(None).cast(col_type).alias(col))

            # Ensure column order matches
            column_order = sorted(all_columns)
            filtered = filtered.select(column_order)
            missing_df = missing_df.select(column_order)

            filtered = PL.concat([filtered, missing_df])

        from typing import cast as _cast

        return _cast("_pl.DataFrame", filtered)


class MockMetadataSource(MetadataSource):
    """
    Mock metadata source for testing.

    Generates realistic but synthetic metadata.

    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize mock metadata source.

        Parameters
        ----------
        seed : int, default 42
            Random seed for reproducibility

        """
        self.seed = seed

    def fetch_metadata(self, instruments: list[str]) -> _pl.DataFrame:
        """
        Generate mock metadata.

        Parameters
        ----------
        instruments : list[str]
            List of instrument identifiers

        Returns
        -------
        pl.DataFrame
            Mock instrument metadata

        """
        from random import Random

        rng = Random(self.seed)

        metadata_list = []

        for symbol in instruments:
            # Generate realistic mock data based on symbol characteristics
            is_etf = symbol.endswith("ETF") or symbol in ["SPY", "QQQ", "IWM", "VTI"]
            is_penny = rng.random() < 0.1  # 10% penny stocks

            if is_penny:
                tick_size = 0.0001
                lot_size = 10000.0
            elif is_etf:
                tick_size = 0.01
                lot_size = 100.0
            else:
                tick_size = 0.01
                lot_size = 100.0

            # Randomly assign to exchanges
            exchanges = ["XNAS", "XNYS", "ARCX", "BATS"]
            exchange = rng.choice(exchanges)

            # Asset class
            if is_etf:
                asset_class = "ETF"
            else:
                asset_class = rng.choice(["EQUITY", "ADR"])

            metadata = {
                "instrument_id": symbol,
                "tick_size": tick_size,
                "lot_size": lot_size,
                "contract_size": 1.0,
                "min_price_increment": tick_size,
                "exchange": exchange,
                "asset_class": asset_class,
                "currency": "USD",
                "margin_initial": 0.25 if not is_etf else 0.20,
                "margin_maintenance": 0.25 if not is_etf else 0.20,
                "fee_class": "MAKER_TAKER",
                "market_segment": "MAIN" if not is_penny else "SMALL_CAP",
            }

            metadata_list.append(metadata)

        if pl_runtime is None:
            check_ml_dependencies(["polars"])  # Ensure Polars present when used
        from typing import cast as _cast

        return _cast("_pl.DataFrame", PL.DataFrame(metadata_list))


# ----------------------------------------------------------------------------
# Canonical default metadata (single source of truth)
# ----------------------------------------------------------------------------


def default_metadata(symbol: str) -> dict[str, Any]:
    """
    Canonical default metadata values for instruments.

    This helper prevents drift between different metadata sources/providers.

    """
    return {
        "instrument_id": symbol,
        "tick_size": 0.01,
        "lot_size": 100.0,
        "contract_size": 1.0,
        "min_price_increment": 0.01,
        "exchange": "UNKNOWN",
        "asset_class": "EQUITY",
        "currency": "USD",
        "margin_initial": 0.0,
        "margin_maintenance": 0.0,
        "fee_class": "DEFAULT",
        "market_segment": "UNKNOWN",
    }
