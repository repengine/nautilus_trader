"""
Configuration for data scheduler and collection.

This module provides configuration classes for the data scheduler and Databento
collection.

"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal


@dataclass(frozen=True)
class DatabentoConfig:
    """
    Configuration for Databento data collection.

    Attributes
    ----------
    dataset : str
        Databento dataset to use (e.g., "GLBX.MDP3", "XNAS.ITCH")
    schema : str
        Data schema to fetch (e.g., "ohlcv-1m", "trades", "mbp-1")
    stype_in : str
        Symbol type for input (e.g., "raw_symbol", "instrument_id")
    use_temporary_files : bool
        Whether to use temporary files for DBN data
    temp_data_dir : str
        Directory for temporary DBN files
    price_precision : int | None
        Price precision for instruments (None uses default)

    """

    dataset: str = "GLBX.MDP3"
    schema: str = "ohlcv-1m"
    stype_in: str = "raw_symbol"
    use_temporary_files: bool = True
    temp_data_dir: str = "./temp_databento_data"
    price_precision: int | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class SchedulerConfig:
    """
    Configuration for data scheduler.

    Attributes
    ----------
    symbols : list[str]
        List of symbols to collect (format: "SYMBOL.VENUE")
    collection_time : str
        Time to run daily collection (24-hour format, e.g., "04:00")
    retention_days : int
        Number of days to retain historical data
    databento : DatabentoConfig
        Databento-specific configuration
    enable_l2_depth : bool
        Whether to collect L2 depth data
    enable_trades : bool
        Whether to collect trade tick data
    enable_quotes : bool
        Whether to collect quote tick data
    max_retries : int
        Maximum retries for failed collections
    retry_delay_seconds : float
        Delay between retries in seconds
    feature_store_enabled : bool
        Whether to compute and store features after data collection
    feature_store_connection : str | None
        PostgreSQL connection string for FeatureStore (None uses env var or default)

    """

    symbols: list[str] = field(
        default_factory=lambda: [
            "SPY.XNAS",
            "QQQ.XNAS",
            "IWM.XNAS",
            "AAPL.XNAS",
            "MSFT.XNAS",
            "NVDA.XNAS",
            "AMZN.XNAS",
            "META.XNAS",
            "GOOGL.XNAS",
            "TSLA.XNAS",
        ],
    )
    collection_time: str = "04:00"
    retention_days: int = 90
    databento: DatabentoConfig = field(default_factory=DatabentoConfig)
    enable_l2_depth: bool = False
    enable_trades: bool = False
    enable_quotes: bool = False
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    feature_store_enabled: bool = True
    feature_store_connection: str | None = None


@dataclass(frozen=True)
class UniverseConfig:
    """
    Configuration for symbol universe management.

    Attributes
    ----------
    priority_symbols : list[str]
        High-priority symbols for deep historical data
    sector_etfs : list[str]
        Sector ETF symbols
    volatility_symbols : list[str]
        Volatility-related symbols
    commodity_symbols : list[str]
        Commodity and bond symbols
    expansion_mode : Literal["conservative", "moderate", "aggressive"]
        How aggressively to expand universe

    """

    priority_symbols: list[str] = field(
        default_factory=lambda: [
            "SPY.XNAS",
            "QQQ.XNAS",
            "IWM.XNAS",
            "DIA.XNAS",
            "VTI.XNAS",
        ],
    )

    sector_etfs: list[str] = field(
        default_factory=lambda: [
            "XLF.XNAS",  # Financials
            "XLK.XNAS",  # Technology
            "XLE.XNAS",  # Energy
            "XLV.XNAS",  # Healthcare
            "XLI.XNAS",  # Industrials
            "XLU.XNAS",  # Utilities
            "XLB.XNAS",  # Materials
            "XLP.XNAS",  # Consumer Staples
            "XLY.XNAS",  # Consumer Discretionary
            "XLRE.XNAS",  # Real Estate
        ],
    )

    volatility_symbols: list[str] = field(
        default_factory=lambda: [
            "VXX.XNAS",
            "UVXY.XNAS",
            "SVXY.XNAS",
        ],
    )

    commodity_symbols: list[str] = field(
        default_factory=lambda: [
            "TLT.XNAS",  # Bonds
            "GLD.XNAS",  # Gold
            "SLV.XNAS",  # Silver
            "USO.XNAS",  # Oil
        ],
    )

    expansion_mode: Literal["conservative", "moderate", "aggressive"] = "moderate"

    def get_full_universe(self) -> list[str]:
        """
        Get the complete universe based on expansion mode.

        Returns
        -------
        list[str]
            Complete list of symbols to track

        """
        base = self.priority_symbols.copy()

        if self.expansion_mode in ["moderate", "aggressive"]:
            base.extend(self.sector_etfs)
            base.extend(self.volatility_symbols)

        if self.expansion_mode == "aggressive":
            base.extend(self.commodity_symbols)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for symbol in base:
            if symbol not in seen:
                seen.add(symbol)
                result.append(symbol)

        return result
