"""
Compatibility shim for FeatureStore.

Redirects legacy imports to the facade-only implementation.

"""

from __future__ import annotations

from ml.stores.feature_store_facade import FeatureStore
from ml.stores.feature_store_facade import FeatureStoreFacade
from ml.stores.feature_store_facade import create_engine


__all__ = [
    "FeatureStore",
    "FeatureStoreFacade",
    "create_engine",
]
