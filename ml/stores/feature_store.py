"""
Compatibility shim for FeatureStore.

Redirects legacy imports to the facade-only implementation.

"""

from __future__ import annotations

from ml.common.db_utils import get_or_create_engine
from ml.core.db_engine import EngineManager
from ml.stores.common.feature_computation import FeatureComputationComponent
from ml.stores.common.feature_event import FeatureEventComponent
from ml.stores.common.feature_health import FeatureHealthComponent
from ml.stores.common.feature_reader import FeatureReaderComponent
from ml.stores.common.feature_schema import FeatureSchemaComponent
from ml.stores.common.feature_writer import FeatureWriterComponent
from ml.stores.feature_store_facade import FeatureStore
from ml.stores.feature_store_facade import FeatureStoreFacade
from ml.stores.feature_store_facade import create_engine


__all__ = [
    "EngineManager",
    "FeatureComputationComponent",
    "FeatureEventComponent",
    "FeatureHealthComponent",
    "FeatureReaderComponent",
    "FeatureSchemaComponent",
    "FeatureStore",
    "FeatureStoreFacade",
    "FeatureWriterComponent",
    "create_engine",
    "get_or_create_engine",
]
