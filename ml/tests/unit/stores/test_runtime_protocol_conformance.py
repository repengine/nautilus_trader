from __future__ import annotations

from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.stores.protocols import (
    FeatureStoreStrictProtocol,
    ModelStoreStrictProtocol,
    StrategyStoreStrictProtocol,
)


def test_runtime_protocol_conformance_isinstance() -> None:
    # Use minimal init to avoid hitting DB in this unit smoke
    fs = FeatureStore(connection_string="postgresql://postgres:postgres@localhost:5432/nautilus")
    ms = ModelStore(connection_string=None, persistence_config=None)
    ss = StrategyStore(connection_string=None, persistence_config=None)

    assert isinstance(fs, FeatureStoreStrictProtocol)
    assert isinstance(ms, ModelStoreStrictProtocol)
    assert isinstance(ss, StrategyStoreStrictProtocol)

