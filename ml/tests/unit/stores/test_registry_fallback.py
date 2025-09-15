"""
DataRegistry progressive fallback tests for stores.

Ensures that when PostgreSQL initialization fails, the store falls back to a
JSON-backed DataRegistry instead of raising.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from ml.registry.persistence import BackendType
from ml.stores.data_store import DataStore


@pytest.mark.unit
def test_data_registry_fallback_to_json(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
    import pathlib as _pl
    monkeypatch.setattr(_pl.Path, "home", lambda: tmp_path)

    # Provide mock stores to avoid DB initialization
    mock_feat = MagicMock()
    mock_model = MagicMock()
    mock_strat = MagicMock()

    # Use a POSTGRES-looking DSN to exercise registry fallback, without DB I/O
    dsn = "postgresql://postgres:postgres@localhost:5432/nautilus"
    # Ensure DataProcessor does not establish a real DB connection in unit tests
    import ml.stores.data_processor as _dp
    from ml.core.db_engine import EngineManager as _EM
    monkeypatch.setattr(_dp, "create_engine", lambda *a, **k: _EM.get_engine("sqlite:///:memory:"))
    store = DataStore(
        connection_string=dsn,
        feature_store=mock_feat,  # type: ignore[arg-type]
        model_store=mock_model,  # type: ignore[arg-type]
        strategy_store=mock_strat,  # type: ignore[arg-type]
        fail_on_validation_error=False,
    )
    registry = store._get_data_registry()
    assert registry is not None
    assert getattr(registry, "backend", None) == BackendType.JSON
