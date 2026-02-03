"""
Strategy decision publisher service and DTO.

This module provides a typed msgspec DTO for strategy decision events and a
service to publish them via a MessagePublisherProtocol. Topics are built using
ml.common.message_topics.build_topic_for_stage and honor MessageBusConfig
scheme/prefix. Publishing is best‑effort and non‑blocking.
"""

from __future__ import annotations

from typing import Any, Final, cast

import msgspec

from ml.common.message_bus import MessagePublisherProtocol
from ml.common.message_bus import publisher_from_config
from ml.common.message_topics import build_topic_for_stage
from ml.config.bus import MessageBusConfig
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


class DecisionEvent(msgspec.Struct, frozen=True):
    """
    Typed payload for a strategy decision event.

    All enums are stored as their `.value` to align with DB/contract expectations.
    """

    dataset_id: str
    stage: str
    status: str
    source: str
    strategy_id: str
    instrument_id: str
    signal_type: str
    strength: float
    model_predictions: dict[str, float]
    risk_metrics: dict[str, float]
    execution_params: dict[str, Any]
    decision_metadata: dict[str, Any]
    ts_event: int

    def to_payload(self) -> dict[str, Any]:
        """
        Convert to a plain dict suitable for publishers.
        """
        # msgspec can convert to builtins efficiently
        return cast(dict[str, Any], msgspec.to_builtins(self))


class StrategyDecisionPublisher:
    """
    Publisher for strategy decision events.

    - Builds topics via build_topic_for_stage
    - Honors MessageBusConfig scheme/prefix
    - Publishes best‑effort (swallows exceptions and returns False)
    """

    def __init__(
        self,
        publisher: MessagePublisherProtocol | None = None,
        *,
        scheme: str | None = None,
        prefix: str | None = None,
    ) -> None:
        cfg = MessageBusConfig.from_env()
        self._publisher: MessagePublisherProtocol = publisher or publisher_from_config(cfg)
        self._scheme: str = (scheme or cfg.scheme)
        self._prefix: str = (prefix or cfg.topic_prefix)

    def publish(
        self,
        *,
        strategy_id: str,
        instrument_id: str,
        signal_type: str,
        strength: float,
        model_predictions: dict[str, float],
        risk_metrics: dict[str, float] | None,
        execution_params: dict[str, Any] | None,
        decision_metadata: dict[str, Any],
        ts_event: int,
        is_live: bool,
        status: EventStatus = EventStatus.SUCCESS,
    ) -> bool:
        """
        Build and publish a strategy decision event.

        Returns True on publisher success, False otherwise.
        """
        try:
            topic = build_topic_for_stage(
                Stage.SIGNAL_EMITTED,
                instrument_id,
                scheme=self._scheme,
                prefix=self._prefix,
            )
            payload = DecisionEvent(
                dataset_id="signals",
                stage=Stage.SIGNAL_EMITTED.value,
                status=status.value,
                source=Source.LIVE.value if is_live else Source.HISTORICAL.value,
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                signal_type=signal_type,
                strength=float(strength),
                model_predictions=dict(model_predictions),
                risk_metrics=dict(risk_metrics or {}),
                execution_params=dict(execution_params or {}),
                decision_metadata=dict(decision_metadata),
                ts_event=int(ts_event),
            ).to_payload()
            try:
                return bool(self._publisher.publish(topic, payload))
            except Exception:
                return False
        except Exception:
            return False


__all__: Final[list[str]] = ["DecisionEvent", "StrategyDecisionPublisher"]
