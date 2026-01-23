"""
Property tests for model-driven exit policy invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.config.base import ModelExitConfig
from ml.strategies.common.model_exit_policy import evaluate_model_exit


@dataclass
class _SignalStub:
    prediction: float
    confidence: float


@dataclass
class _PositionStub:
    side: SimpleNamespace
    ts_opened: int | None


@settings(max_examples=50)
@given(
    prediction=st.floats(min_value=0.61, max_value=1.0, allow_nan=False, allow_infinity=False),
    confidence=st.floats(min_value=0.7, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_model_exit_noop_when_confidence_stable_property(
    prediction: float,
    confidence: float,
) -> None:
    config = ModelExitConfig(exit_prediction_band=0.1, exit_confidence_threshold=0.7)
    signal = _SignalStub(prediction=prediction, confidence=confidence)
    position = _PositionStub(SimpleNamespace(name="LONG"), ts_opened=1_000_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=1_500_000_000,
        trigger_price=None,
    )

    assert decision is None


@settings(max_examples=50)
@given(
    prediction=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
)
def test_model_exit_never_reverses_when_exit_to_flat_on_flip_property(
    prediction: float,
) -> None:
    config = ModelExitConfig(exit_on_flip=True, reverse_on_flip=False)
    signal = _SignalStub(prediction=prediction, confidence=0.9)
    position = _PositionStub(SimpleNamespace(name="LONG"), ts_opened=1_000_000_000)

    decision = evaluate_model_exit(
        position=position,
        signal=signal,
        config=config,
        now_ns=2_000_000_000,
        trigger_price=100.0,
    )

    assert decision is not None
    assert decision.action == "exit"
