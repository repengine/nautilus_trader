"""
Model-driven exit policy helpers for ML strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ml.config.base import ModelExitConfig


@runtime_checkable
class ModelSignalProtocol(Protocol):
    """
    Minimal signal interface required for model-driven exits.
    """

    prediction: float
    confidence: float


@runtime_checkable
class PositionSideProtocol(Protocol):
    """
    Minimal position interface required for model-driven exits.
    """

    side: object
    ts_opened: int | None


@dataclass(frozen=True, slots=True)
class ExitDecision:
    """
    Model-driven exit decision.
    """

    action: Literal["exit", "reverse"]
    reason: str
    trigger_price: float | None
    time_in_trade_ns: int | None
    confidence: float | None

    def to_metadata(self) -> dict[str, object]:
        """
        Serialize the decision to metadata for persistence and intents.
        """
        payload: dict[str, object] = {
            "action": self.action,
            "reason": self.reason,
            "trigger_price": self.trigger_price,
            "time_in_trade_ns": self.time_in_trade_ns,
        }
        if self.confidence is not None:
            payload["confidence"] = float(self.confidence)
        return payload


def resolve_time_in_trade_ns(position: PositionSideProtocol, now_ns: int) -> int | None:
    """
    Compute time-in-trade in nanoseconds for the position, if available.
    """
    opened = getattr(position, "ts_opened", None)
    if opened is None:
        return None
    try:
        opened_ns = int(opened)
    except (TypeError, ValueError):
        return None
    if opened_ns <= 0 or now_ns < opened_ns:
        return None
    return now_ns - opened_ns


def _side_name(position: PositionSideProtocol) -> str:
    side = getattr(position, "side", None)
    name = getattr(side, "name", None)
    if name:
        return str(name).upper()
    if isinstance(side, str):
        return side.upper()
    return ""


def _flip_decision(
    *,
    config: ModelExitConfig,
    trigger_price: float | None,
    time_in_trade_ns: int | None,
    confidence: float,
) -> ExitDecision | None:
    if not config.exit_on_flip:
        return None
    action: Literal["exit", "reverse"] = "reverse" if config.reverse_on_flip else "exit"
    return ExitDecision(
        action=action,
        reason="model_flip",
        trigger_price=trigger_price,
        time_in_trade_ns=time_in_trade_ns,
        confidence=confidence,
    )


def evaluate_model_exit(
    *,
    position: PositionSideProtocol,
    signal: ModelSignalProtocol,
    config: ModelExitConfig,
    now_ns: int,
    trigger_price: float | None,
) -> ExitDecision | None:
    """
    Evaluate model-driven exit conditions for a position.
    """
    side_name = _side_name(position)
    if side_name not in {"LONG", "SHORT"}:
        return None

    prediction = float(signal.prediction)
    confidence = float(signal.confidence)
    time_in_trade_ns = resolve_time_in_trade_ns(position, now_ns)

    if config.min_hold_ms is not None and time_in_trade_ns is not None:
        min_hold_ns = int(config.min_hold_ms) * 1_000_000
        if time_in_trade_ns < min_hold_ns:
            return None

    if config.exit_confidence_threshold is not None and config.exit_confidence_threshold > 0.0:
        if confidence < float(config.exit_confidence_threshold):
            return ExitDecision(
                action="exit",
                reason="model_confidence",
                trigger_price=trigger_price,
                time_in_trade_ns=time_in_trade_ns,
                confidence=confidence,
            )

    band = float(config.exit_prediction_band)
    threshold = 0.5
    if band > 0.0:
        lower = threshold - band
        upper = threshold + band
        if side_name == "LONG":
            if prediction <= lower:
                return _flip_decision(
                    config=config,
                    trigger_price=trigger_price,
                    time_in_trade_ns=time_in_trade_ns,
                    confidence=confidence,
                )
            if prediction < upper:
                return ExitDecision(
                    action="exit",
                    reason="model_neutral",
                    trigger_price=trigger_price,
                    time_in_trade_ns=time_in_trade_ns,
                    confidence=confidence,
                )
        else:
            if prediction > upper:
                return _flip_decision(
                    config=config,
                    trigger_price=trigger_price,
                    time_in_trade_ns=time_in_trade_ns,
                    confidence=confidence,
                )
            if prediction > lower:
                return ExitDecision(
                    action="exit",
                    reason="model_neutral",
                    trigger_price=trigger_price,
                    time_in_trade_ns=time_in_trade_ns,
                    confidence=confidence,
                )
        return None

    if side_name == "LONG":
        if prediction <= threshold:
            return _flip_decision(
                config=config,
                trigger_price=trigger_price,
                time_in_trade_ns=time_in_trade_ns,
                confidence=confidence,
            )
    else:
        if prediction > threshold:
            return _flip_decision(
                config=config,
                trigger_price=trigger_price,
                time_in_trade_ns=time_in_trade_ns,
                confidence=confidence,
            )
    return None


__all__ = [
    "ExitDecision",
    "ModelSignalProtocol",
    "PositionSideProtocol",
    "evaluate_model_exit",
    "resolve_time_in_trade_ns",
]
