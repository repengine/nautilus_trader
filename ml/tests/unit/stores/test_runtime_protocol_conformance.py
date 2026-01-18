from __future__ import annotations

from typing import Callable, ContextManager
from unittest.mock import MagicMock

import pytest

from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.stores.protocols import (
    FeatureStoreStrictProtocol,
    ModelStoreStrictProtocol,
    StrategyStoreStrictProtocol,
)
from ml.tests.utils.db import build_postgres_url


PatchEngineManager = Callable[..., ContextManager[MagicMock]]


def test_runtime_protocol_conformance_isinstance(
    monkeypatch: pytest.MonkeyPatch,
    patch_engine_manager: PatchEngineManager,
) -> None:
    # Avoid real database connections by patching engine/table initialization.
    monkeypatch.setattr(
        "ml.stores.model_store.ModelStore._init_engine_and_tables", lambda self: None
    )
    monkeypatch.setattr(
        "ml.stores.strategy_store.StrategyStore._init_engine_and_tables", lambda self: None
    )
    monkeypatch.setattr("ml.stores.feature_store.FeatureStore._setup_tables", lambda self: None)

    with patch_engine_manager():
        fs = FeatureStore(connection_string=build_postgres_url())
    ms = ModelStore(connection_string=None, persistence_config=None)
    ss = StrategyStore(connection_string=None, persistence_config=None)

    assert isinstance(fs, FeatureStoreStrictProtocol)
    assert isinstance(ms, ModelStoreStrictProtocol)
    assert isinstance(ss, StrategyStoreStrictProtocol)
