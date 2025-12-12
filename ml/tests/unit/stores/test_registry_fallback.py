"""
DataRegistry progressive fallback tests for stores.

Ensures that when PostgreSQL initialization fails, the store falls back to a JSON-backed
DataRegistry instead of raising.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from unittest.mock import MagicMock

from ml.registry.persistence import BackendType, PersistenceConfig
from ml.stores.data_store import DataStore
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.tests.utils.db import build_postgres_url
from ml.tests.utils.stubs import FeatureStoreNoOp, ModelStoreNoOp, StrategyStoreNoOp


@pytest.mark.unit
def test_data_registry_fallback_to_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Simulate failure when initializing a POSTGRES-backed DataRegistry,
    # forcing the mixin to fall back to JSON.
    import ml.registry.data_registry as _dr
    from ml.registry.persistence import BackendType as _BT

    _orig = _dr.DataRegistry

    def _factory(
        registry_path: Path,
        batch_save_interval: float = 0.1,
        persistence_config: PersistenceConfig | None = None,
    ) -> _dr.DataRegistry:
        if persistence_config is not None and persistence_config.backend == _BT.POSTGRES:
            raise RuntimeError("PG backend unavailable")
        return _orig(
            registry_path=registry_path,
            batch_save_interval=batch_save_interval,
            persistence_config=persistence_config,
        )

    monkeypatch.setattr(_dr, "DataRegistry", _factory)

    # Isolate registry under a temp HOME to avoid existing malformed files
    import pathlib as _pl

    monkeypatch.setattr(_pl.Path, "home", lambda: tmp_path)

    # Provide mock stores to avoid DB initialization
    mock_feat = cast(FeatureStore, FeatureStoreNoOp())
    mock_model = cast(ModelStore, ModelStoreNoOp())
    mock_strat = cast(StrategyStore, StrategyStoreNoOp())

    # Use a POSTGRES-looking DSN to exercise registry fallback, without DB I/O
    dsn = build_postgres_url()
    # Ensure DataProcessor does not establish a real DB connection in unit tests
    import ml.common.db_utils as _db
    from ml.core.db_engine import EngineManager as _EM

    def _fake_get_or_create_engine(*a: object, **k: object) -> object:
        return _EM.get_engine("sqlite:///:memory:")

    monkeypatch.setattr(_db, "get_or_create_engine", _fake_get_or_create_engine)
    store = DataStore(
        connection_string=dsn,
        feature_store=mock_feat,
        model_store=mock_model,
        strategy_store=mock_strat,
        fail_on_validation_error=False,
    )
    registry = store._get_data_registry()
    assert registry is not None
    assert getattr(registry, "backend", None) == BackendType.JSON
