from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId

from ml.actors import signal as signal_module
from ml.actors.multi_signal import MultiInstrumentSignalActor
from ml.actors.multi_signal import MultiInstrumentSignalActorConfig
from ml.actors.multi_signal import _UniverseManager


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


class _Logger:
    def info(self, *_: object, **__: object) -> None:
        return None

    def warning(self, *_: object, **__: object) -> None:
        return None

    def exception(self, *_: object, **__: object) -> None:
        return None

    def debug(self, *_: object, **__: object) -> None:
        return None


class _Metric:
    def labels(self, *_: object, **__: object) -> _Metric:
        return self

    def inc(self, *_: object, **__: object) -> None:
        return None

    def observe(self, *_: object, **__: object) -> None:
        return None

    def set(self, *_: object, **__: object) -> None:
        return None


class _MetricsManager:
    def counter(self, *_: object, **__: object) -> _Metric:
        return _Metric()

    def histogram(self, *_: object, **__: object) -> _Metric:
        return _Metric()

    def gauge(self, *_: object, **__: object) -> _Metric:
        return _Metric()


def _patch_base_init(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_init(self: MultiInstrumentSignalActor, config: object) -> None:
        from nautilus_trader.common.actor import Actor as NautilusActor
        from nautilus_trader.common.config import ActorConfig

        actor_config = ActorConfig(
            component_id=getattr(config, "component_id", "actor-1"),
            log_events=False,
            log_commands=False,
        )
        NautilusActor.__init__(self, actor_config)
        self._config = config
        self._circuit_breaker = None

    monkeypatch.setattr(signal_module.MLSignalActor, "__init__", _fake_init)
    monkeypatch.setattr(
        "ml.common.metrics_manager.MetricsManager.default",
        lambda: _MetricsManager(),
    )


def _make_config(
    *,
    max_batch_size: int = 2,
    feature_dim: int = 4,
    initial_universe: list[str] | None = None,
) -> MultiInstrumentSignalActorConfig:
    return MultiInstrumentSignalActorConfig(
        component_id="multi-signal",
        model_path="model.onnx",
        model_id="model-v1",
        bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-LAST-EXTERNAL"),
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        enable_async_persistence=False,
        enable_health_monitoring=False,
        max_batch_size=max_batch_size,
        feature_dim=feature_dim,
        initial_universe=initial_universe,
        flush_max_latency_ms=0,
    )


class _ModelInfo:
    def __init__(self, metadata: dict[str, object]) -> None:
        self.metadata = metadata


class _ModelRegistry:
    def __init__(self, metadata: dict[str, object]) -> None:
        self.metadata = metadata
        self.calls: list[str] = []

    def get_model(self, model_id: str) -> _ModelInfo:
        self.calls.append(model_id)
        return _ModelInfo(self.metadata)


def test_universe_manager_round_trip() -> None:
    manager = _UniverseManager(["EUR/USD.SIM", "SPY.EQUS"])

    assert manager.contains("EUR/USD.SIM")
    manager.add("BTC.USD")
    manager.remove("SPY.EQUS")

    assert set(manager.snapshot()) == {"EUR/USD.SIM", "BTC.USD"}

    manager.set_all(["ETH.USD"])
    assert manager.size() == 1

    manager.clear()
    assert manager.size() == 0


def test_multi_signal_init_preallocates_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_base_init(monkeypatch)
    config = _make_config(initial_universe=["EUR/USD.SIM"])

    actor = MultiInstrumentSignalActor(config)

    assert actor._batch_features.shape == (config.max_batch_size, config.feature_dim)
    assert actor._batch_features.dtype == np.float32
    assert actor._batch_size == 0
    assert actor._batch_instruments == []
    assert actor._batch_bars == []
    assert actor._prepared_preds == []
    assert actor._universe.size() == 1


def test_multi_signal_skips_out_of_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_base_init(monkeypatch)
    config = _make_config(initial_universe=["EUR/USD.SIM"])
    actor = MultiInstrumentSignalActor(config)

    actor._compute_features = lambda _bar: np.ones(4, dtype=np.float32)

    bar = SimpleNamespace(
        bar_type=SimpleNamespace(instrument_id="OTHER.SIM"),
    )
    actor.on_bar(bar)

    assert actor._batch_size == 0


def test_multi_signal_on_bar_flushes_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_base_init(monkeypatch)
    config = _make_config(max_batch_size=2, feature_dim=4)
    actor = MultiInstrumentSignalActor(config)

    actor._compute_features = lambda _bar: np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    preds = np.array([0.2, 0.8], dtype=np.float32)
    confs = np.array([0.6, 0.9], dtype=np.float32)
    actor._infer_batch = lambda _features: (preds, confs)

    generated: list[tuple[float, float]] = []

    def _generate(bar: object, features: np.ndarray) -> None:
        generated.append(actor._predict(features))

    actor._generate_prediction_protected = _generate

    bar_a = SimpleNamespace(bar_type=SimpleNamespace(instrument_id="EUR/USD.SIM"))
    bar_b = SimpleNamespace(bar_type=SimpleNamespace(instrument_id="SPY.EQUS"))

    actor.on_bar(bar_a)
    actor.on_bar(bar_b)

    assert generated == [
        (pytest.approx(0.2), pytest.approx(0.6)),
        (pytest.approx(0.8), pytest.approx(0.9)),
    ]
    assert actor._batch_size == 0
    assert actor._batch_instruments == []
    assert actor._batch_bars == []
    assert actor._prepared_preds == []


def test_multi_signal_infer_batch_onnx_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_base_init(monkeypatch)
    config = _make_config(max_batch_size=2, feature_dim=4)
    actor = MultiInstrumentSignalActor(config)

    class _Model:
        def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            data = next(iter(inputs.values()))
            preds = np.full((data.shape[0],), 0.25, dtype=np.float32)
            confs = np.full((data.shape[0],), 0.75, dtype=np.float32)
            return [preds, confs]

    actor._model = _Model()
    actor._model_metadata = {
        "input_names": ["input"],
        "output_is_logits": False,
        "positive_class_index": 0,
    }

    features = np.zeros((2, 4), dtype=np.float32)
    preds, confs = actor._infer_batch(features)

    assert preds.shape == (2,)
    assert confs.shape == (2,)
    assert np.allclose(preds, 0.25)
    assert np.allclose(confs, 0.75)


def test_universe_management_updates_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_base_init(monkeypatch)
    config = _make_config(initial_universe=[])
    actor = MultiInstrumentSignalActor(config)

    actor.add_instrument("EUR/USD.SIM")
    assert actor._universe.contains("EUR/USD.SIM")

    actor.remove_instrument("EUR/USD.SIM")
    assert actor._universe.size() == 0

    actor.set_universe(["SPY.EQUS", "AAPL.EQUS"])
    assert actor._universe.size() == 2

    actor.clear_universe()
    assert actor._universe.size() == 0


def test_on_start_adjusts_feature_dim_from_engineer(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_base_init(monkeypatch)
    monkeypatch.setattr(signal_module.MLSignalActor, "on_start", lambda _self: None)
    config = _make_config(feature_dim=4)
    actor = MultiInstrumentSignalActor(config)

    actor._feature_engineer = SimpleNamespace(n_features=6)
    actor._manifest_feature_names = []

    actor.on_start()

    assert actor._batch_features.shape == (config.max_batch_size, 6)


def test_on_start_adjusts_feature_dim_from_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_base_init(monkeypatch)
    monkeypatch.setattr(signal_module.MLSignalActor, "on_start", lambda _self: None)
    config = _make_config(feature_dim=4)
    actor = MultiInstrumentSignalActor(config)

    actor._feature_engineer = None
    actor._manifest_feature_names = ["f1", "f2", "f3"]

    actor.on_start()

    assert actor._batch_features.shape == (config.max_batch_size, 3)


def test_on_start_loads_universe_from_registry_instrument_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_base_init(monkeypatch)
    monkeypatch.setattr(signal_module.MLSignalActor, "on_start", lambda _self: None)
    config = _make_config(feature_dim=4)
    actor = MultiInstrumentSignalActor(config)

    actor._model_id = "model-x"
    setattr(
        actor,
        "_BaseMLInferenceActor__model_registry_instance",
        _ModelRegistry({"universe_instrument_ids": ["EUR/USD.SIM", "SPY.EQUS"]}),
    )

    actor.on_start()

    assert actor._universe.contains("EUR/USD.SIM")
    assert actor._universe.contains("SPY.EQUS")


def test_on_start_loads_universe_from_registry_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_base_init(monkeypatch)
    monkeypatch.setattr(signal_module.MLSignalActor, "on_start", lambda _self: None)
    config = _make_config(feature_dim=4)
    actor = MultiInstrumentSignalActor(config)

    actor._model_id = "model-y"
    setattr(
        actor,
        "_BaseMLInferenceActor__model_registry_instance",
        _ModelRegistry({"universe_symbols": ["AAPL", "MSFT"]}),
    )

    actor.on_start()

    assert actor._universe.contains("AAPL.SIM")
    assert actor._universe.contains("MSFT.SIM")


def test_on_start_skips_registry_when_universe_preseeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_base_init(monkeypatch)
    monkeypatch.setattr(signal_module.MLSignalActor, "on_start", lambda _self: None)
    config = _make_config(feature_dim=4, initial_universe=["EUR/USD.SIM"])
    actor = MultiInstrumentSignalActor(config)

    actor._model_id = "model-z"
    setattr(
        actor,
        "_BaseMLInferenceActor__model_registry_instance",
        _ModelRegistry({"universe_instrument_ids": ["SPY.EQUS"]}),
    )

    actor.on_start()

    assert actor._universe.contains("EUR/USD.SIM")
    assert not actor._universe.contains("SPY.EQUS")
