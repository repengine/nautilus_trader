"""
ML Stores: Pattern 1 Compliant Store Architecture
================================================

This module provides the 4 mandatory stores required by Universal ML Architecture Pattern 1:
- FeatureStore: Feature computation and storage for ML pipeline
- ModelStore: Model predictions with performance tracking
- StrategyStore: Strategy signals and decisions
- DataStore: Unified facade with contract validation and event emission

All ML actors MUST use these 4 stores via BaseMLInferenceActor inheritance to ensure:
- Consistent data lifecycle management
- Automatic component initialization
- Progressive fallback to DummyStore when PostgreSQL unavailable
- Health monitoring across all components

Pattern 1 Integration Example:
-----------------------------
```python
# Import stores from ml.stores, actors from ml.actors
# (Avoids circular dependency - actors depend on stores, not vice versa)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml.actors.base import BaseMLInferenceActor

class YourCustomActor(BaseMLInferenceActor):
    def __init__(self, config: YourConfig):
        # REQUIRED: Call super().__init__ first
        super().__init__(config)

        # Stores are now automatically available:
        # - self.feature_store
        # - self.model_store
        # - self.strategy_store
        # - self.data_store

        # Your custom initialization here
        self.custom_logic = self._initialize_custom_logic()

    def on_bar(self, bar: Bar) -> None:
        # ✅ CORRECT: Use pre-initialized stores
        features = self.feature_store.compute_realtime(bar)
        prediction = self.model.predict(features)

        self.model_store.write_prediction(
            model_id=self.config.model_id,
            prediction=prediction,
            confidence=0.95,
            features=dict(zip(self.feature_names, features)),
            inference_time_ms=1.2,
            ts_event=bar.ts_event,
            instrument_id=str(bar.instrument_id)
        )
```

Pattern 2 Protocol-First Design:
-------------------------------
All stores implement strict protocols for structural typing:
- FeatureStoreProtocol / FeatureStoreStrictProtocol
- ModelStoreProtocol / ModelStoreStrictProtocol
- StrategyStoreProtocol / StrategyStoreStrictProtocol
- DataStoreFacadeProtocol

Pattern 4 Progressive Fallback:
------------------------------
Stores automatically fallback when PostgreSQL unavailable:
PostgreSQL → DummyStore (no persistence, warnings logged)

See ml/docs/architecture/universal_patterns_guide.md for complete documentation.
"""

from ml.stores.base import BaseStore
from ml.stores.base import DummyStore
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.data_processor import DataProcessor
from ml.stores.data_store_facade import DataStore
from ml.stores.data_store_facade import DataStoreFacade
from ml.stores.earnings_store import DummyEarningsStore
from ml.stores.earnings_store import EarningsStore
from ml.stores.feature_store_facade import FeatureStore
from ml.stores.feature_store_facade import FeatureStoreFacade
from ml.stores.file_backed import FileDataStore
from ml.stores.file_backed import FileEarningsStore  # noqa: F401 - re-export for Pattern 4 fallback
from ml.stores.file_backed import FileFeatureStore
from ml.stores.file_backed import FileModelStore
from ml.stores.file_backed import FileStrategyStore
from ml.stores.infrastructure import PartitionManager
from ml.stores.infrastructure import check_db_prereqs
from ml.stores.infrastructure import run_partition_maintenance
from ml.stores.instrument_metadata_store import DummyInstrumentMetadataStore
from ml.stores.instrument_metadata_store import InstrumentMetadataStore
from ml.stores.io_raw import ParquetCatalogRawReader
from ml.stores.io_raw import ParquetCatalogRawWriter
from ml.stores.io_raw import RawIngestionWriterProtocol
from ml.stores.io_raw import RawReaderProtocol
from ml.stores.mixins import BufferedStoreMixin
from ml.stores.mixins import DataRegistryMixin
from ml.stores.mixins import EngineInitMixin
from ml.stores.mixins import HealthMixin
from ml.stores.mixins import ReadQueryMixin
from ml.stores.mixins import SQLUpsertMixin
from ml.stores.mixins import StoreInitMixin
from ml.stores.mixins import publish_batch_and_rows
from ml.stores.mixins import sanitize_and_dedup
from ml.stores.model_store import ModelStore
from ml.stores.protocols import BaseStoreProtocol
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import DataStoreFacadeProtocol
from ml.stores.protocols import FeatureStoreProtocol
from ml.stores.protocols import FeatureStoreStrictProtocol
from ml.stores.protocols import InstrumentMetadataStoreProtocol
from ml.stores.protocols import MarketDataWriterProtocol
from ml.stores.protocols import ModelStoreProtocol
from ml.stores.protocols import ModelStoreStrictProtocol
from ml.stores.protocols import PredictionRecord
from ml.stores.protocols import ReadFrame
from ml.stores.protocols import SignalRecord
from ml.stores.protocols import StrategyStoreProtocol
from ml.stores.protocols import StrategyStoreStrictProtocol
from ml.stores.protocols import WriteRecords
from ml.stores.providers import CatalogCoverageProvider
from ml.stores.providers import SqlCoverageProvider
from ml.stores.providers import SqlMarketDataWriter
from ml.stores.strategy_store import StrategyStore
from ml.stores.table_factory import build_instrument_id_column
from ml.stores.table_factory import build_nautilus_timestamp_columns
from ml.stores.table_factory import build_standard_indexes
from ml.stores.table_factory import create_ml_table
from ml.stores.table_factory import get_schema_name
from ml.stores.writers import DataStoreMarketDataWriter
from ml.stores.writers import LiveDataRecorder
from ml.stores.writers import ParquetCatalogMarketDataWriter


# Component-based alias (legacy removed, facade is canonical).
ComponentFeatureStore = FeatureStoreFacade


# =============================================================================
# Public API Definition
# =============================================================================

__all__ = [
    "BaseStore",
    "BaseStoreProtocol",
    "BufferedStoreMixin",
    "CatalogCoverageProvider",
    "ComponentFeatureStore",
    "CoverageProviderProtocol",
    "DataProcessor",
    "DataRegistryMixin",
    "DataStore",
    "DataStoreFacade",
    "DataStoreFacadeProtocol",
    "DataStoreMarketDataWriter",
    "DummyEarningsStore",
    "DummyInstrumentMetadataStore",
    "DummyStore",
    "EarningsStore",
    "EngineInitMixin",
    "FeatureData",
    "FeatureStore",
    "FeatureStoreFacade",
    "FeatureStoreProtocol",
    "FeatureStoreStrictProtocol",
    "FileDataStore",
    "FileFeatureStore",
    "FileModelStore",
    "FileStrategyStore",
    "HealthMixin",
    "InstrumentMetadataStore",
    "InstrumentMetadataStoreProtocol",
    "LiveDataRecorder",
    "MarketDataWriterProtocol",
    "ModelPrediction",
    "ModelStore",
    "ModelStoreProtocol",
    "ModelStoreStrictProtocol",
    "ParquetCatalogMarketDataWriter",
    "ParquetCatalogRawReader",
    "ParquetCatalogRawWriter",
    "PartitionManager",
    "PredictionRecord",
    "RawIngestionWriterProtocol",
    "RawReaderProtocol",
    "ReadFrame",
    "ReadQueryMixin",
    "SQLUpsertMixin",
    "SignalRecord",
    "SqlCoverageProvider",
    "SqlMarketDataWriter",
    "StoreInitMixin",
    "StrategySignal",
    "StrategyStore",
    "StrategyStoreProtocol",
    "StrategyStoreStrictProtocol",
    "WriteRecords",
    "build_instrument_id_column",
    "build_nautilus_timestamp_columns",
    "build_standard_indexes",
    "check_db_prereqs",
    "create_ml_table",
    "get_schema_name",
    "publish_batch_and_rows",
    "run_partition_maintenance",
    "sanitize_and_dedup",
]

# =============================================================================
# Pattern Compliance Notes
# =============================================================================

# Pattern 1: All 4 stores are exposed and MUST be used via BaseMLInferenceActor
# Pattern 2: All major interfaces use typing.Protocol for structural typing
# Pattern 3: Stores separate hot path (real-time inference) from cold path (training)
# Pattern 4: DummyStore provides progressive fallback when PostgreSQL unavailable
# Pattern 5: All stores use ml.common.metrics_bootstrap (never direct prometheus_client)

# See ml/docs/architecture/universal_patterns_guide.md for complete implementation guide
