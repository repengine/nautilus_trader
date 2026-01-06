from __future__ import annotations

from typing import Any

from ml.common.message_bus import MessagePublisherProtocol
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


from pathlib import Path


def test_feature_store_per_row_publishing(tmp_path: Path) -> None:
    pub = CapturePublisher()
    store = FeatureStore(
        connection_string=f"sqlite:///{tmp_path}/feature.db",
        enable_publishing=True,
        publisher=pub,
        publish_mode="row",
    )
    fd1 = FeatureData("fs1", "EURUSD.SIM", {"a": 1.0}, _ts_event=3, _ts_init=3)
    fd2 = FeatureData("fs1", "EURUSD.SIM", {"a": 2.0}, _ts_event=4, _ts_init=4)
    store.write_batch([fd1, fd2])
    # Expect two per-row publishes
    topics = [t for t, _ in pub.calls]
    assert topics.count("ml.features.updated.EURUSD.SIM") >= 2


def test_model_store_per_row_publishing(tmp_path: Path) -> None:
    pub = CapturePublisher()
    store = ModelStore(
        connection_string=f"sqlite:///{tmp_path}/model.db",
        enable_publishing=True,
        publisher=pub,
        publish_mode="row",
    )
    mp1 = ModelPrediction("m1", "EURUSD.SIM", 0.1, 0.9, {}, 0.2, _ts_event=1, _ts_init=1)
    mp2 = ModelPrediction("m1", "EURUSD.SIM", 0.2, 0.8, {}, 0.2, _ts_event=2, _ts_init=2)
    store.write_batch([mp1, mp2])
    topics = [t for t, _ in pub.calls]
    assert topics.count("ml.models.created.EURUSD.SIM") >= 2


def test_strategy_store_per_row_publishing(tmp_path: Path) -> None:
    pub = CapturePublisher()
    store = StrategyStore(
        connection_string=f"sqlite:///{tmp_path}/strategy.db",
        enable_publishing=True,
        publisher=pub,
        publish_mode="row",
    )
    ss1 = StrategySignal("s1", "EURUSD.SIM", "BUY", 0.5, {}, {}, {}, _ts_event=2, _ts_init=2)
    ss2 = StrategySignal("s1", "EURUSD.SIM", "SELL", 0.4, {}, {}, {}, _ts_event=3, _ts_init=3)
    store.write_batch([ss1, ss2])
    topics = [t for t, _ in pub.calls]
    assert topics.count("ml.strategies.created.EURUSD.SIM") >= 2
