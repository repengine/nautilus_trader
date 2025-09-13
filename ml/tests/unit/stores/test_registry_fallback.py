"""
DataRegistry progressive fallback tests for stores.

Ensures that when PostgreSQL initialization fails, the store falls back to a
JSON-backed DataRegistry instead of raising.
"""

from __future__ import annotations

import pytest

from ml.registry.persistence import BackendType
from ml.stores.strategy_store import StrategyStore


@pytest.mark.serial
def test_data_registry_fallback_to_json(monkeypatch, tmp_path, postgres_connection: str) -> None:
    # Simulate failure when initializing a POSTGRES-backed DataRegistry,
    # forcing the mixin to fall back to JSON.
    import ml.registry.data_registry as _dr
    from ml.registry.persistence import BackendType as _BT

    _orig = _dr.DataRegistry

    def _factory(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        pc = kwargs.get("persistence_config")
        if getattr(pc, "backend", None) == _BT.POSTGRES:
            raise RuntimeError("PG backend unavailable")
        return _orig(*args, **kwargs)

    monkeypatch.setattr(_dr, "DataRegistry", _factory)

    # Isolate registry under a temp HOME to avoid existing malformed files
    import ml.stores._registry_mixin as _rm
    monkeypatch.setattr(_rm.Path, "home", lambda: tmp_path)

    store = StrategyStore(connection_string=postgres_connection)
    registry = store._get_data_registry()
    assert registry is not None
    assert getattr(registry, "backend", None) == BackendType.JSON
