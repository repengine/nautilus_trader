"""
Compatibility shim for DataStore.

Redirects legacy imports to the facade-only implementation.

"""

from __future__ import annotations

import time

from ml.stores.data_store_facade import DataStore
from ml.stores.data_store_facade import DataStoreConfig
from ml.stores.data_store_facade import DataStoreFacade
from ml.stores.data_store_facade import logger


__all__ = [
    "DataStore",
    "DataStoreConfig",
    "DataStoreFacade",
    "logger",
    "time",
]
