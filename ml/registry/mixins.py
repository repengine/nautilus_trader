#!/usr/bin/env python3

"""
Optional mixins for registry concerns.

These helpers are advisory and can be adopted incrementally by concrete
registries. They are intentionally minimal and safe.
"""

from __future__ import annotations

import logging
import time
from typing import Any


class StageLifecycleMixin:
    """
    Helper for registries that manage a simple stage/state enum on entries.

    Provides a single utility to update ``stage`` and ``last_modified`` fields
    on a manifest-like object. Subclasses should hold external locks as needed
    around mutations.
    """

    @staticmethod
    def _set_stage(entry: Any, stage: Any) -> None:
        if hasattr(entry, "stage"):
            setattr(entry, "stage", stage)
        if hasattr(entry, "last_modified"):
            try:
                setattr(entry, "last_modified", float(time.time()))
            except Exception as exc:
                # Best effort; keep mutation safe
                logging.getLogger(__name__).debug(
                    "Setting last_modified failed: %s", exc, exc_info=True
                )


class ArtifactMixin:
    """
    Helper for registries that store a mapping of artifact name -> path/URI.
    """

    @staticmethod
    def _attach_artifacts(container: Any, artifacts: dict[str, str]) -> None:
        current: dict[str, str] | None = getattr(container, "artifacts", None)
        if current is None:
            setattr(container, "artifacts", {str(k): str(v) for k, v in artifacts.items()})
            return
        current.update({str(k): str(v) for k, v in artifacts.items()})


class CacheMixin:
    """
    Simple LRU cache helper keyed by ``str``.

    Intended for use in ModelRegistry to cache loaded ONNX sessions.
    """

    def __init__(self, cache_size: int = 10) -> None:
        self._cache_size = int(cache_size)
        self._cache: dict[str, object] = {}
        self._cache_access: dict[str, float] = {}

    def cache_get(self, key: str) -> object | None:
        if key in self._cache:
            self._cache_access[key] = time.time()
            return self._cache[key]
        return None

    def cache_put(self, key: str, value: object) -> None:
        if key not in self._cache and len(self._cache) >= self._cache_size:
            self._evict_lru()
        self._cache[key] = value
        self._cache_access[key] = time.time()

    def cache_pop(self, key: str) -> None:
        self._cache.pop(key, None)
        self._cache_access.pop(key, None)

    def _evict_lru(self) -> None:
        if not self._cache_access:
            return
        lru_key = min(self._cache_access.items(), key=lambda kv: kv[1])[0]
        self._cache.pop(lru_key, None)
        self._cache_access.pop(lru_key, None)


__all__ = [
    "ArtifactMixin",
    "CacheMixin",
    "StageLifecycleMixin",
]
