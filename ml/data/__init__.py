"""
ML data utilities for Nautilus Trader.

This package provides utilities for working with Nautilus Trader's data infrastructure
for ML workflows. It uses ParquetDataCatalog directly instead of custom loaders,
following the principle of using Nautilus native components.

"""

from ml.data.catalog_utils import bars_to_dataframe
from ml.data.catalog_utils import quotes_to_dataframe
from ml.data.catalog_utils import trades_to_dataframe

# Curated public API to reduce import churn for common tasks
from ml.data.collector import DataCollector
from ml.data.scheduler import DataScheduler
from ml.data.tft_dataset_builder import TFTDatasetBuilder

from ml.data.providers.metadata import InstrumentMetadataProvider
from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.providers.events import EventScheduleProvider

from ml.data.sources.calendar import MockCalendarSource
from ml.data.sources.calendar import SimpleCalendarSource
from ml.data.sources.calendar import PandasCalendarSource
from ml.data.sources.events import MockEventSource
from ml.data.sources.metadata import DatabentoMetadataSource
from ml.data.sources.metadata import NautilusMetadataSource

from ml.data.l2_cache import L2MinuteCache
from ml.data.micro_cache import MicroMinuteCache


__all__ = [
    "bars_to_dataframe",
    "quotes_to_dataframe",
    "trades_to_dataframe",
    # High-level orchestrators
    "DataCollector",
    "DataScheduler",
    "TFTDatasetBuilder",
    # Providers
    "InstrumentMetadataProvider",
    "MarketCalendarProvider",
    "EventScheduleProvider",
    # Sources (mocks/simple for quick starts and tests)
    "MockCalendarSource",
    "SimpleCalendarSource",
    "PandasCalendarSource",
    "MockEventSource",
    "DatabentoMetadataSource",
    "NautilusMetadataSource",
    # Caches
    "L2MinuteCache",
    "MicroMinuteCache",
]
