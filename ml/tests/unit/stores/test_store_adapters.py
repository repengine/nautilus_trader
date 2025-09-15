import typing as t

import pytest

from ml.stores.adapters import FeatureStoreStrictAdapter
from ml.stores.adapters import ModelStoreStrictAdapter
from ml.stores.adapters import StrategyStoreStrictAdapter
from ml.stores.protocols import FeatureStoreStrictProtocol
from ml.stores.protocols import ModelStoreStrictProtocol
from ml.stores.protocols import StrategyStoreStrictProtocol


class _DummyFeatureStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, t.Any]]] = []

    def write_features(self, **kwargs: t.Any) -> None:
        self.calls.append(("write_features", kwargs))

    def flush(self) -> None:  # pragma: no cover - called but trivial
        self.calls.append(("flush", {}))


class _DummyModelStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, t.Any]]] = []

    def write_prediction(self, **kwargs: t.Any) -> None:
        self.calls.append(("write_prediction", kwargs))

    def write_batch(self, data: list[t.Any], *, emit_events: bool = True) -> None:
        self.calls.append(("write_batch", {"data_len": len(data), "emit_events": emit_events}))

    def flush(self) -> None:  # pragma: no cover - called but trivial
        self.calls.append(("flush", {}))


class _DummyStrategyStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, t.Any]]] = []

    def write_signal(self, **kwargs: t.Any) -> None:
        self.calls.append(("write_signal", kwargs))

    def write_batch(self, data: list[t.Any]) -> None:
        self.calls.append(("write_batch", {"data_len": len(data)}))

    def flush(self) -> None:  # pragma: no cover - called but trivial
        self.calls.append(("flush", {}))


def test_feature_store_adapter_protocol_and_delegation() -> None:
    base = _DummyFeatureStore()
    adapter: FeatureStoreStrictProtocol = FeatureStoreStrictAdapter(base)

    adapter.write_features(
        feature_set_id="fs",
        instrument_id="EURUSD.SIM",
        features={"x": 1.0},
        ts_event=1,
        ts_init=1,
    )
    adapter.flush()

    assert any(call[0] == "write_features" for call in base.calls)
    assert any(call[0] == "flush" for call in base.calls)


def test_model_store_adapter_protocol_and_delegation() -> None:
    base = _DummyModelStore()
    adapter: ModelStoreStrictProtocol = ModelStoreStrictAdapter(base)

    adapter.write_prediction(
        model_id="m",
        instrument_id="EURUSD.SIM",
        prediction=0.1,
        confidence=0.9,
        features={"x": 1.0},
        inference_time_ms=0.2,
        ts_event=1,
        is_live=True,
    )
    adapter.write_batch([{"ok": True}], emit_events=False)
    adapter.flush()

    assert any(call[0] == "write_prediction" for call in base.calls)
    assert ("write_batch", {"data_len": 1, "emit_events": False}) in base.calls
    assert any(call[0] == "flush" for call in base.calls)


def test_strategy_store_adapter_protocol_and_delegation() -> None:
    base = _DummyStrategyStore()
    adapter: StrategyStoreStrictProtocol = StrategyStoreStrictAdapter(base)

    adapter.write_signal(
        strategy_id="s",
        instrument_id="EURUSD.SIM",
        signal_type="buy",
        strength=0.5,
        model_predictions={"m": 0.2},
        risk_metrics={"dd": 0.1},
        execution_params={"limit": 1},
        ts_event=1,
        is_live=False,
    )
    adapter.write_batch([{"ok": True}])
    adapter.flush()

    assert any(call[0] == "write_signal" for call in base.calls)
    assert ("write_batch", {"data_len": 1}) in base.calls
    assert any(call[0] == "flush" for call in base.calls)

