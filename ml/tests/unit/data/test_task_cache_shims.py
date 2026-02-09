from __future__ import annotations

import importlib

import pytest

from ml.data.rehydration.cache_hydration import CacheHydrationResult as CanonicalCacheHydrationResult
from ml.data.rehydration.cache_hydration import L2CacheHydrationConfig as CanonicalL2CacheHydrationConfig
from ml.data.rehydration.cache_hydration import MicroCacheHydrationConfig as CanonicalMicroCacheHydrationConfig
from ml.data.rehydration.cache_hydration import SymbolHydrationResult as CanonicalSymbolHydrationResult
from ml.data.rehydration.cache_hydration import hydrate_l2_caches as canonical_hydrate_l2_caches
from ml.data.rehydration.cache_hydration import hydrate_micro_caches as canonical_hydrate_micro_caches
from ml.data.rehydration.cache_hydration import ingest_l2_cache_partitions as canonical_ingest_l2_cache_partitions
from ml.data.rehydration.cache_hydration import ingest_micro_cache_partitions as canonical_ingest_micro_cache_partitions


def test_cache_hydration_canonical_symbols_are_available() -> None:
    assert CanonicalCacheHydrationResult.__name__ == "CacheHydrationResult"
    assert CanonicalSymbolHydrationResult.__name__ == "SymbolHydrationResult"
    assert CanonicalMicroCacheHydrationConfig.__name__ == "MicroCacheHydrationConfig"
    assert CanonicalL2CacheHydrationConfig.__name__ == "L2CacheHydrationConfig"
    assert callable(canonical_hydrate_micro_caches)
    assert callable(canonical_hydrate_l2_caches)
    assert callable(canonical_ingest_micro_cache_partitions)
    assert callable(canonical_ingest_l2_cache_partitions)


def test_task_cache_shim_module_is_retired() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ml.tasks.caches.hydration")
