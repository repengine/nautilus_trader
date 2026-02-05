from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId

from ml.actors.actor_services import ActorServices
from ml.actors.enhanced import EnhancedMLInferenceActor
from ml.config.base import MLActorConfig
from ml.stores.base import DummyStore


def _make_services() -> ActorServices:
    dummy = DummyStore()
    return ActorServices(
        feature_store=dummy,
        model_store=dummy,
        strategy_store=dummy,
        data_store=dummy,
        feature_registry=object(),
        model_registry=object(),
        strategy_registry=object(),
        data_registry=object(),
    )


def test_enhanced_actor_compute_features_returns_buffer_view(monkeypatch: pytest.MonkeyPatch) -> None:
    services = _make_services()
    monkeypatch.setattr(
        "ml.actors.actor_services.init_actor_services",
        lambda _config: services,
    )

    config = MLActorConfig(
        component_id="enhanced",
        model_path="model.onnx",
        model_id="model-v1",
        bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-LAST-EXTERNAL"),
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        enable_async_persistence=False,
        enable_health_monitoring=False,
    )
    actor = EnhancedMLInferenceActor(config)

    bar = SimpleNamespace(open=1.0, high=1.2, low=0.8, close=1.1, volume=10.0)
    features = actor._compute_features(bar)

    assert features is not None
    assert np.shares_memory(features, actor._feature_buffer)
    assert features.shape[0] == actor._feature_buffer.shape[0]

    actor._load_model()
    assert actor._model is None
    assert actor._model_metadata == {}
    assert actor._predict(features) == (0.0, 0.0)
