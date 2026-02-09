"""
Unit tests for model-driven exit policy helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from ml.config.base import ModelExitConfig
from ml.strategies.common.model_exit_policy import ExitDecision
from ml.strategies.common.model_exit_policy import evaluate_model_exit


@dataclass
class _SignalStub:
    prediction: float
    confidence: float


@dataclass
class _PositionStub:
    side: SimpleNamespace
    ts_opened: int | None


def test_model_exit_returns_exit_on_flip_for_long_position() -> None:
    config = ModelExitConfig(exit_on_flip=True, reverse_on_flip=False)
    signal = _SignalStub(prediction=0.4, confidence=0.8)
    position = _PositionStub(SimpleNamespace(name="LONG"), ts_opened=500_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=1_000_000_000,
        trigger_price=101.0,
    )

    assert decision is not None
    assert decision.action == "exit"
    assert decision.reason == "model_flip"
    assert decision.trigger_price == 101.0
    assert decision.time_in_trade_ns == 500_000_000
    assert decision.confidence == 0.8


def test_model_exit_returns_reverse_on_flip_when_configured() -> None:
    config = ModelExitConfig(exit_on_flip=True, reverse_on_flip=True)
    signal = _SignalStub(prediction=0.7, confidence=0.9)
    position = _PositionStub(SimpleNamespace(name="SHORT"), ts_opened=100_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=400_000_000,
        trigger_price=99.5,
    )

    assert decision is not None
    assert decision.action == "reverse"
    assert decision.reason == "model_flip"
    assert decision.trigger_price == 99.5
    assert decision.time_in_trade_ns == 300_000_000
    assert decision.confidence == 0.9


def test_model_exit_returns_exit_on_low_confidence() -> None:
    config = ModelExitConfig(exit_confidence_threshold=0.6)
    signal = _SignalStub(prediction=0.9, confidence=0.4)
    position = _PositionStub(SimpleNamespace(name="LONG"), ts_opened=100_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=300_000_000,
        trigger_price=100.0,
    )

    assert decision is not None
    assert decision.action == "exit"
    assert decision.reason == "model_confidence"


def test_model_exit_returns_exit_in_neutral_zone() -> None:
    config = ModelExitConfig(exit_prediction_band=0.1)
    signal = _SignalStub(prediction=0.55, confidence=0.8)
    position = _PositionStub(SimpleNamespace(name="LONG"), ts_opened=1_000_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=1_500_000_000,
        trigger_price=100.5,
    )

    assert decision is not None
    assert decision.action == "exit"
    assert decision.reason == "model_neutral"


def test_model_exit_respects_min_hold_time() -> None:
    config = ModelExitConfig(exit_on_flip=True, min_hold_ms=1_000)
    signal = _SignalStub(prediction=0.1, confidence=0.8)
    position = _PositionStub(SimpleNamespace(name="LONG"), ts_opened=1_000_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=1_500_000_000,
        trigger_price=100.0,
    )

    assert decision is None


def test_model_exit_noop_when_prediction_aligned() -> None:
    config = ModelExitConfig(exit_prediction_band=0.1, exit_confidence_threshold=0.6)
    signal = _SignalStub(prediction=0.85, confidence=0.9)
    position = _PositionStub(SimpleNamespace(name="LONG"), ts_opened=1_000_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=1_800_000_000,
        trigger_price=102.0,
    )

    assert decision is None


def test_model_exit_returns_none_for_unknown_position_side() -> None:
    config = ModelExitConfig(exit_on_flip=True)
    signal = _SignalStub(prediction=0.1, confidence=0.8)
    position = _PositionStub(SimpleNamespace(name="FLAT"), ts_opened=100_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=300_000_000,
        trigger_price=100.0,
    )

    assert decision is None


def test_model_exit_band_flip_returns_none_when_flip_exits_disabled() -> None:
    config = ModelExitConfig(
        exit_on_flip=False,
        exit_prediction_band=0.1,
        reverse_on_flip=False,
    )
    signal = _SignalStub(prediction=0.30, confidence=0.7)
    position = _PositionStub(SimpleNamespace(name="LONG"), ts_opened=100_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=300_000_000,
        trigger_price=99.0,
    )

    assert decision is None


def test_model_exit_ignores_min_hold_when_open_time_is_invalid() -> None:
    config = ModelExitConfig(
        exit_on_flip=True,
        reverse_on_flip=False,
        min_hold_ms=10_000,
    )
    signal = _SignalStub(prediction=0.1, confidence=0.8)
    position = _PositionStub(SimpleNamespace(name="LONG"), ts_opened=2_000_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=1_000_000_000,
        trigger_price=98.5,
    )

    assert decision is not None
    assert decision.reason == "model_flip"
    assert decision.time_in_trade_ns is None


def test_model_exit_resolves_string_side_and_short_neutral_band() -> None:
    config = ModelExitConfig(
        exit_prediction_band=0.1,
        exit_on_flip=True,
        reverse_on_flip=False,
    )
    signal = _SignalStub(prediction=0.45, confidence=0.7)
    position = _PositionStub(side="SHORT", ts_opened=100_000_000)  # type: ignore[arg-type]

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=300_000_000,
        trigger_price=100.0,
    )

    assert decision is not None
    assert decision.action == "exit"
    assert decision.reason == "model_neutral"


def test_model_exit_returns_none_when_open_time_is_not_numeric() -> None:
    config = ModelExitConfig(exit_on_flip=True, min_hold_ms=1)
    signal = _SignalStub(prediction=0.2, confidence=0.8)
    position = _PositionStub(SimpleNamespace(name="LONG"), ts_opened="bad")  # type: ignore[arg-type]

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=500_000_000,
        trigger_price=95.0,
    )

    assert decision is not None
    assert decision.reason == "model_flip"
    assert decision.time_in_trade_ns is None


def test_exit_decision_metadata_omits_confidence_when_missing() -> None:
    decision = ExitDecision(
        action="exit",
        reason="manual",
        trigger_price=101.0,
        time_in_trade_ns=10,
        confidence=None,
    )

    metadata = decision.to_metadata()

    assert metadata["action"] == "exit"
    assert metadata["reason"] == "manual"
    assert "confidence" not in metadata
