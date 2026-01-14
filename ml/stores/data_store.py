"""
Compatibility shim for DataStore.

Redirects legacy imports to the facade-only implementation.

"""

from __future__ import annotations

from ml.stores.data_store_facade import DataStore
from ml.stores.data_store_facade import DataStoreFacade


__all__ = [
    "DataStore",
    "DataStoreFacade",
]
