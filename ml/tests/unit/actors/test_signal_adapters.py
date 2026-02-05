from __future__ import annotations

from types import SimpleNamespace

import pytest

from ml.actors.adapters import DynamicThresholdAdapter, build_strategy_from_policy
from ml.actors.signal import ThresholdSignalStrategy


class _DummyStrategy:
    def __init__(self, label: str) -> None:
        self.label = label

    def generate_signal(self, *_args: object, **_kwargs: object) -> None:
        return None


def adapter_fn(_actor: object) -> _DummyStrategy:
    return _DummyStrategy("fn")


def adapter_fn_invalid(_actor: object) -> object:
    return object()


class AdapterClass:
    def __init__(self, label: str = "class") -> None:
        self.label = label

    def make(self, _actor: object) -> _DummyStrategy:
        return _DummyStrategy(self.label)


adapter_instance = AdapterClass("instance")


class PolicyClass:
    def __init__(self, _actor: object | None = None, label: str = "policy") -> None:
        self.label = label

    def generate_signal(self, *_args: object, **_kwargs: object) -> None:
        return None


def test_dynamic_threshold_adapter_uses_actor_threshold() -> None:
    actor = SimpleNamespace(_adaptive_threshold=0.9, _config=SimpleNamespace(prediction_threshold=0.1))

    strategy = DynamicThresholdAdapter().make(actor)

    assert isinstance(strategy, ThresholdSignalStrategy)
    assert strategy.threshold == pytest.approx(0.9)


def test_dynamic_threshold_adapter_falls_back_to_config() -> None:
    actor = SimpleNamespace(_config=SimpleNamespace(prediction_threshold=0.4))

    strategy = DynamicThresholdAdapter().make(actor)

    assert isinstance(strategy, ThresholdSignalStrategy)
    assert strategy.threshold == pytest.approx(0.4)


def test_build_strategy_from_function_adapter() -> None:
    strategy = build_strategy_from_policy(
        policy_path=f"{__name__}.adapter_fn",
        actor=SimpleNamespace(),
    )

    assert isinstance(strategy, _DummyStrategy)
    assert strategy.label == "fn"


def test_build_strategy_from_function_adapter_invalid() -> None:
    with pytest.raises(RuntimeError):
        build_strategy_from_policy(
            policy_path=f"{__name__}.adapter_fn_invalid",
            actor=SimpleNamespace(),
        )


def test_build_strategy_from_adapter_instance() -> None:
    strategy = build_strategy_from_policy(
        policy_path=f"{__name__}.adapter_instance",
        actor=SimpleNamespace(),
    )

    assert isinstance(strategy, _DummyStrategy)
    assert strategy.label == "instance"


def test_build_strategy_from_adapter_class_make() -> None:
    strategy = build_strategy_from_policy(
        policy_path=f"{__name__}.AdapterClass",
        actor=SimpleNamespace(),
        config={"label": "custom"},
    )

    assert isinstance(strategy, _DummyStrategy)
    assert strategy.label == "custom"


def test_build_strategy_from_policy_class() -> None:
    strategy = build_strategy_from_policy(
        policy_path=f"{__name__}.PolicyClass",
        actor=SimpleNamespace(),
        config={"label": "policy-custom"},
    )

    assert isinstance(strategy, PolicyClass)
    assert strategy.label == "policy-custom"


def test_build_strategy_from_policy_invalid_path() -> None:
    with pytest.raises(RuntimeError):
        build_strategy_from_policy(
            policy_path="InvalidPath",
            actor=SimpleNamespace(),
        )
