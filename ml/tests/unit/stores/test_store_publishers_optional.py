from __future__ import annotations

from typing import Any, cast
from pathlib import Path

from ml.common.message_bus import MessagePublisherProtocol
from ml.config.events import Stage
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.feature_store_facade import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


def test_model_store_optional_publishing(tmp_path: Path) -> None:
    pub = CapturePublisher()
    store = ModelStore(
        connection_string=f"sqlite:///{tmp_path}/model.db",
        enable_publishing=True,
        publisher=pub,
    )
    mp = ModelPrediction(
        model_id="m1",
        instrument_id="EURUSD.SIM",
        prediction=0.1,
        confidence=0.9,
        features_used={},
        inference_time_ms=0.2,
        _ts_event=1,
        _ts_init=1,
    )
    store.write_batch([mp])
    assert pub.calls, "Publisher should be called when enabled"
    topic, payload = pub.calls[-1]
    assert topic == "ml.models.created.EURUSD.SIM"
    assert payload["stage"] == Stage.PREDICTION_EMITTED.value


def test_strategy_store_optional_publishing(tmp_path: Path) -> None:
    pub = CapturePublisher()
    store = StrategyStore(
        connection_string=f"sqlite:///{tmp_path}/strategy.db",
        enable_publishing=True,
        publisher=pub,
    )
    ss = StrategySignal(
        strategy_id="s1",
        instrument_id="EURUSD.SIM",
        signal_type="BUY",
        strength=0.5,
        model_predictions={},
        risk_metrics={},
        execution_params={},
        _ts_event=2,
        _ts_init=2,
    )
    store.write_batch([ss])
    assert pub.calls, "Publisher should be called when enabled"
    topic, payload = pub.calls[-1]
    assert topic == "ml.strategies.created.EURUSD.SIM"
    assert payload["stage"] == Stage.SIGNAL_EMITTED.value


def test_feature_store_optional_publishing(tmp_path: Path) -> None:
    pub = CapturePublisher()
    store = FeatureStore(
        connection_string=f"sqlite:///{tmp_path}/feature.db",
        enable_publishing=True,
        publisher=pub,
    )
    fd = FeatureData(
        feature_set_id="fs1",
        instrument_id="EURUSD.SIM",
        values={"a": 1.0},
        _ts_event=3,
        _ts_init=3,
    )
    store.write_batch([cast(Any, fd)])
    assert pub.calls, "Publisher should be called when enabled"
    topic, payload = pub.calls[-1]
    assert topic == "ml.features.updated.EURUSD.SIM"
    assert payload["stage"] == Stage.FEATURE_COMPUTED.value
