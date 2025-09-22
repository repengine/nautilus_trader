"""
Data loaders for Nautilus Trader ML.

This module provides various data loaders for different data sources.

"""

from ml.data.loaders.alfred_loader import ALFREDConfig
from ml.data.loaders.alfred_loader import ALFREDDataLoader
from ml.data.loaders.alternative import AlternativeDataConfig
from ml.data.loaders.alternative import AlternativeDataResult
from ml.data.loaders.alternative import AlternativeSource
from ml.data.loaders.alternative import load_tier1_symbols
from ml.data.loaders.alternative import populate_alternative_data
from ml.data.loaders.alternative import save_alternative_data
from ml.data.loaders.fred_loader import FREDConfig
from ml.data.loaders.fred_loader import FREDDataLoader
from ml.data.loaders.fred_loader import FREDIndicator
from ml.data.loaders.ohlcv_recent import OhlcvRecentBackfillConfig
from ml.data.loaders.ohlcv_recent import OhlcvRecentBackfillResult
from ml.data.loaders.ohlcv_recent import SymbolBackfillStatus
from ml.data.loaders.ohlcv_recent import SymbolBackfillSummary
from ml.data.loaders.ohlcv_recent import backfill_recent_ohlcv
from ml.data.loaders.supplementary import DEFAULT_BASE_SYMBOLS
from ml.data.loaders.supplementary import DEFAULT_SPREADS
from ml.data.loaders.supplementary import SUPPLEMENTARY_SYMBOLS
from ml.data.loaders.supplementary import SpreadDefinition
from ml.data.loaders.supplementary import SupplementaryDataConfig
from ml.data.loaders.supplementary import SupplementaryOutputs
from ml.data.loaders.supplementary import calculate_correlations
from ml.data.loaders.supplementary import calculate_spreads
from ml.data.loaders.supplementary import create_synthetic_supplementary_data
from ml.data.loaders.supplementary import write_supplementary_outputs


__all__ = [
    "DEFAULT_BASE_SYMBOLS",
    "DEFAULT_SPREADS",
    "SUPPLEMENTARY_SYMBOLS",
    "ALFREDConfig",
    "ALFREDDataLoader",
    "AlternativeDataConfig",
    "AlternativeDataResult",
    "AlternativeSource",
    "FREDConfig",
    "FREDDataLoader",
    "FREDIndicator",
    "OhlcvRecentBackfillConfig",
    "OhlcvRecentBackfillResult",
    "SpreadDefinition",
    "SupplementaryDataConfig",
    "SupplementaryOutputs",
    "SymbolBackfillStatus",
    "SymbolBackfillSummary",
    "backfill_recent_ohlcv",
    "calculate_correlations",
    "calculate_spreads",
    "create_synthetic_supplementary_data",
    "load_tier1_symbols",
    "populate_alternative_data",
    "save_alternative_data",
    "write_supplementary_outputs",
]
