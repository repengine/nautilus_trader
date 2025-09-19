from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pytest

from ml.actors.multi_signal import MultiInstrumentSignalActor
from ml.actors.multi_signal import MultiInstrumentSignalActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


class _FakeBarType:
    def __init__(self, instrument_id: str) -> None:
        self.instrument_id = instrument_id


class _FakeBar:
    def __init__(self, instrument_id: str) -> None:
        self.bar_type = _FakeBarType(instrument_id)
        self.ts_event = 0
        self.ts_init = 0


class _Actor(MultiInstrumentSignalActor):
    """Testable actor overriding compute/infer/pipeline hooks."""

    def __init__(self, config: MultiInstrumentSignalActorConfig) -> None:
        super().__init__(config)
        self.last_batch: npt.NDArray[np.float32] | None = None
        self.pred_calls: int = 0

    def _compute_features(self, bar: Any) -> npt.NDArray[np.float32]:  # type: ignore[override]
        # Deterministic 1..feature_dim sequence as features
        dim = self._cfg.feature_dim
        return np.arange(1, dim + 1, dtype=np.float32)

    def _generate_prediction_protected(self, bar: Any, features: npt.NDArray[np.float32]) -> None:  # type: ignore[override]
        # Count calls without invoking stores
        self.pred_calls += 1

    def _infer_batch(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:  # type: ignore[override]
        self.last_batch = features.copy()
        return super()._infer_batch(features)


class _ORTModel:
    """Simple fake ONNXRuntime-like model for tests."""

    def __init__(self, fn: Callable[[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]]]) -> None:
        self._fn = fn

    def run(
        self, _: Any, inputs: dict[str, npt.NDArray[np.float32]]
    ) -> list[npt.NDArray[np.float32]]:
        # Take the first input tensor in dict order
        arr = next(iter(inputs.values()))
        # If 2D, keep batch dimension; tests will provide already batched arrays
        return self._fn(arr)


def test_multi_actor_batches_to_capacity_and_flush() -> None:
    inst = InstrumentId.from_str("A.TEST")
    bar_type = BarType.from_str("A.TEST-1-MINUTE-LAST-EXTERNAL")
    cfg = MultiInstrumentSignalActorConfig(
        actor_id="test-actor",
        max_batch_size=2,
        feature_dim=4,
        use_dummy_stores=True,
        model_path="dummy.onnx",
        model_id="dummy_model",
        instrument_id=inst,
        bar_type=bar_type,
    )
    actor = _Actor(cfg)

    # Add instruments to universe
    actor.add_instrument("A")
    actor.add_instrument("B")

    # Feed two instruments; batch should flush on capacity
    actor.on_bar(_FakeBar("A"))
    actor.on_bar(_FakeBar("B"))

    # Force flush in case the actor is configured with larger capacity
    actor._flush_batch()

    assert actor.last_batch is not None
    assert actor.last_batch.shape == (2, 4)
    # Prediction pipeline invoked per instrument
    assert actor.pred_calls == 2


def test_multi_actor_universe_filtering() -> None:
    inst = InstrumentId.from_str("X.TEST")
    bar_type = BarType.from_str("X.TEST-1-MINUTE-LAST-EXTERNAL")
    cfg = MultiInstrumentSignalActorConfig(
        actor_id="test-actor",
        max_batch_size=4,
        feature_dim=3,
        use_dummy_stores=True,
        initial_universe=["X"],
        model_path="dummy.onnx",
        model_id="dummy_model",
        instrument_id=inst,
        bar_type=bar_type,
    )
    actor = _Actor(cfg)

    # Bars for non-universe instrument should be ignored
    actor.on_bar(_FakeBar("Y"))
    actor._flush_batch()
    assert actor.last_batch is None
    assert actor.pred_calls == 0


def test_infer_batch_vectorized_ort_path_returns_expected() -> None:
    cfg = MultiInstrumentSignalActorConfig(
        actor_id="test-actor",
        max_batch_size=8,
        feature_dim=3,
        use_dummy_stores=True,
        model_path="dummy.onnx",
        model_id="dummy_model",
        instrument_id=InstrumentId.from_str("Z.TEST"),
        bar_type=BarType.from_str("Z.TEST-1-MINUTE-LAST-EXTERNAL"),
    )
    actor = _Actor(cfg)

    # Fake ORT model that returns two outputs: pred = sum(row), conf = 0.9
    def _ort_fn(x: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        preds = np.sum(x, axis=1, dtype=np.float32).astype(np.float32)
        confs = np.full((x.shape[0],), 0.9, dtype=np.float32)
        return [preds, confs]

    actor._model = _ORTModel(_ort_fn)  # type: ignore[attr-defined]
    actor._model_metadata = {  # type: ignore[attr-defined]
        "input_names": ["features"],
        "output_names": ["prediction", "confidence"],
    }

    batch = np.array(
        [[1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [2.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    preds, confs = actor._infer_batch(batch)
    assert preds.shape == (3,)
    assert confs.shape == (3,)
    np.testing.assert_allclose(preds, np.array([6.0, 1.5, 3.0], dtype=np.float32))
    np.testing.assert_allclose(confs, np.array([0.9, 0.9, 0.9], dtype=np.float32))


def test_infer_batch_falls_back_to_per_row_on_ort_error(monkeypatch: Any) -> None:
    cfg = MultiInstrumentSignalActorConfig(
        actor_id="test-actor",
        max_batch_size=8,
        feature_dim=2,
        use_dummy_stores=True,
        model_path="dummy.onnx",
        model_id="dummy_model",
        instrument_id=InstrumentId.from_str("W.TEST"),
        bar_type=BarType.from_str("W.TEST-1-MINUTE-LAST-EXTERNAL"),
    )
    actor = _Actor(cfg)

    # Model that raises on run to trigger fallback
    class _BadORT:
        def run(self, *_: Any, **__: Any) -> list[npt.NDArray[np.float32]]:  # noqa: D401
            raise RuntimeError("boom")

    actor._model = _BadORT()  # type: ignore[attr-defined]
    actor._model_metadata = {"input_names": ["features"]}  # type: ignore[attr-defined]

    # Patch the base class _predict to a simple function of features for easy assertion
    def _fake_predict(self: Any, features: npt.NDArray[np.float32]) -> tuple[float, float]:
        return float(features[0] + features[1]), 0.7

    from ml.actors.signal import MLSignalActor as _Base

    monkeypatch.setattr(_Base, "_predict", _fake_predict, raising=True)

    batch = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    preds, confs = actor._infer_batch(batch)
    np.testing.assert_allclose(preds, np.array([3.0, 7.0], dtype=np.float32))
    np.testing.assert_allclose(confs, np.array([0.7, 0.7], dtype=np.float32))


def test_time_based_flush_triggers(monkeypatch: Any) -> None:
    inst = InstrumentId.from_str("T.TEST")
    bar_type = BarType.from_str("T.TEST-1-MINUTE-LAST-EXTERNAL")
    cfg = MultiInstrumentSignalActorConfig(
        actor_id="test-actor",
        max_batch_size=8,
        feature_dim=3,
        use_dummy_stores=True,
        initial_universe=["T"],
        model_path="dummy.onnx",
        model_id="dummy_model",
        instrument_id=inst,
        bar_type=bar_type,
        flush_max_latency_ms=1,
    )
    actor = _Actor(cfg)

    # Monkeypatch time_ns to simulate elapsed time exceeding threshold in a single call
    calls: list[int] = []

    def _fake_time_ns() -> int:
        if not calls:
            calls.append(1)
            return 1_000_000  # start
        return 3_000_000  # +2ms, exceeds 1ms threshold

    import time as _time

    monkeypatch.setattr(_time, "time_ns", _fake_time_ns, raising=False)

    # Feed one bar for member instrument; flush should trigger via timer
    actor.on_bar(_FakeBar("T"))
    # Ensure batch was flushed (pred pipeline called once)
    assert actor.pred_calls == 1
