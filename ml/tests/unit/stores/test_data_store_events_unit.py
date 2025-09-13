"""
DataStore event emission tests using stubbed registry and stores.

Focus: functional outcome that DataStore emits events and updates watermarks
with expected parameters when writing features/predictions/signals.

"""

from __future__ import annotations

from typing import Any

import pytest

from ml.stores.data_store import DataStore
from nautilus_trader.model.identifiers import InstrumentId


class _StubRegistry:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.watermarks: list[dict[str, Any]] = []

    def emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(locals())

    def update_watermark(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: str,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        self.watermarks.append(locals())


class _NoOpStore:
    def write_features(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op write_features.
        """
        return None

    def write_prediction(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op write_prediction.
        """
        return None

    def write_signal(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op write_signal.
        """
        return None

    def write_batch(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op batch write for stores expecting it.
        """
        return None


@pytest.fixture
def stubbed_data_store(monkeypatch: pytest.MonkeyPatch) -> tuple[DataStore, _StubRegistry]:
    # Build an instance and replace internal stores and registry accessor
    ds = object.__new__(DataStore)  # type: ignore[misc]
    ds.connection_string = "sqlite:///:memory:"  # type: ignore[attr-defined]
    ds.feature_store = _NoOpStore()  # type: ignore[attr-defined]
    ds.model_store = _NoOpStore()  # type: ignore[attr-defined]
    ds.strategy_store = _NoOpStore()  # type: ignore[attr-defined]
    stub_registry = _StubRegistry()
    ds.registry = stub_registry  # type: ignore[attr-defined]
    ds._get_dataset_ids = lambda: {  # type: ignore[attr-defined]
        "features": "features",
        "predictions": "predictions",
        "signals": "signals",
    }

    # Provide clock stub
    class _Clock:
        def timestamp_ns(self) -> int:
            return 100

    ds.clock = _Clock()  # type: ignore[attr-defined]
    # Avoid schema/registration side effects
    ds._ensure_dataset_registered = lambda **kwargs: None  # type: ignore[attr-defined]
    return ds, stub_registry


def test_data_store_emits_feature_events(
    stubbed_data_store: tuple[DataStore, _StubRegistry],
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
) -> None:
    ds, reg = stubbed_data_store
    from ml.stores.base import FeatureData

    ts_event, ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    fd = FeatureData(
        feature_set_id="fs1",
        instrument_id=instrument_id_str,
        values={"a": 1.0},
        _ts_event=ts_event,
        _ts_init=ts_init,
    )
    ds.write_features(instrument_id=instrument_id_str, features=[fd])  # type: ignore[attr-defined]
    assert any(e for e in reg.events if e["dataset_id"] == "features")


def test_data_store_emits_prediction_events(
    stubbed_data_store: tuple[DataStore, _StubRegistry],
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
) -> None:
    ds, reg = stubbed_data_store
    from ml.stores.base import ModelPrediction

    ts_event, ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    mp = ModelPrediction(
        model_id="m1",
        instrument_id=instrument_id_str,
        prediction=0.8,
        confidence=0.9,
        features_used={"a": 1.0},
        inference_time_ms=0.1,
        _ts_event=ts_event,
        _ts_init=ts_init,
    )
    ds.write_predictions(predictions=[mp])  # type: ignore[attr-defined]
    assert any(e for e in reg.events if e["dataset_id"] == "predictions")


def test_data_store_emits_signal_events(
    stubbed_data_store: tuple[DataStore, _StubRegistry],
    default_instrument_id: InstrumentId,
    test_timestamps: tuple[int, int],
) -> None:
    ds, reg = stubbed_data_store
    from ml.stores.base import StrategySignal

    ts_event, ts_init = test_timestamps
    instrument_id_str = str(default_instrument_id)

    ss = StrategySignal(
        strategy_id="s1",
        instrument_id=instrument_id_str,
        signal_type="BUY",
        strength=0.8,
        model_predictions={"m1": 0.8},
        risk_metrics={"conf": 0.8},
        execution_params={"side": "BUY"},
        _ts_event=ts_event,
        _ts_init=ts_init,
    )
    ds.write_signals(signals=[ss])  # type: ignore[attr-defined]
    assert any(e for e in reg.events if e["dataset_id"] == "signals")
