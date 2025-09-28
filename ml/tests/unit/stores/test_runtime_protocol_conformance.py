from __future__ import annotations

import types

import pytest

from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.stores.protocols import (
    FeatureStoreStrictProtocol,
    ModelStoreStrictProtocol,
    StrategyStoreStrictProtocol,
)


def test_runtime_protocol_conformance_isinstance(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid real database connections by patching engine/table initialization.
    monkeypatch.setattr(
        "ml.stores.model_store.ModelStore._init_engine_and_tables", lambda self: None
    )
    monkeypatch.setattr(
        "ml.stores.strategy_store.StrategyStore._init_engine_and_tables", lambda self: None
    )
    monkeypatch.setattr("ml.stores.feature_store.FeatureStore._setup_tables", lambda self: None)

    dummy_engine = types.SimpleNamespace(connect=lambda: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        "ml.stores.feature_store.create_engine", lambda *_args, **_kwargs: dummy_engine
    )

    fs = FeatureStore(connection_string="postgresql://postgres:postgres@localhost:5432/nautilus")
    ms = ModelStore(connection_string=None, persistence_config=None)
    ss = StrategyStore(connection_string=None, persistence_config=None)

    assert isinstance(fs, FeatureStoreStrictProtocol)
    assert isinstance(ms, ModelStoreStrictProtocol)
    assert isinstance(ss, StrategyStoreStrictProtocol)
