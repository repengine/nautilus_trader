"""
Actor-side domain events bridge (non-blocking enqueue + background flusher).

This module wraps the shared bus bridge implementation with actor-specific
initialization and mutual-exclusion safeguards against store-level publishing.
"""

from __future__ import annotations

import logging
from typing import Any

from ml.common.bus_bridge import DomainEventBridge
from ml.common.bus_bridge import TopicThrottleConfig
from ml.common.bus_bridge import parse_topic_throttles_from_env
from ml.common.message_bus import publisher_from_config
from ml.common.throttler import Throttler
from ml.config.bus import MessageBusConfig


def _parse_per_topic_throttles() -> dict[str, TopicThrottleConfig]:
    """
    Parse per-topic throttling configuration from environment variables.

    Returns
    -------
    dict[str, TopicThrottleConfig]
        Mapping of topic patterns to their throttling configurations.
    """
    return parse_topic_throttles_from_env()


def init_actor_bus_bridge(actor: Any) -> tuple[DomainEventBridge | None, str, str]:
    """
    Initialize an actor-side domain event bridge from environment configuration.

    Returns
    -------
    tuple[DomainEventBridge | None, str, str]
        (bridge, topic_scheme, topic_prefix). Bridge is None when disabled.
    """
    # Default topic configuration
    topic_scheme = "domain_op"
    topic_prefix = "events.ml"

    try:
        from ml.config.actor_bus import ActorBusConfig

        actor_bus_cfg = ActorBusConfig.from_env()
        bus_cfg = MessageBusConfig.from_env()
        if not (actor_bus_cfg.from_actor and bus_cfg.enabled):
            return None, topic_scheme, topic_prefix

        publisher = publisher_from_config(bus_cfg)
        throttler = (
            Throttler(
                rate_per_sec=float(actor_bus_cfg.throttle_rate_per_sec),
                burst=int(actor_bus_cfg.throttle_burst),
            )
            if actor_bus_cfg.throttle_enabled
            else None
        )
        per_topic_throttles = _parse_per_topic_throttles()

        bridge = DomainEventBridge(
            publisher,
            max_queue=int(actor_bus_cfg.max_queue),
            throttler=throttler,
            per_topic_throttles=per_topic_throttles,
            component_id="ml_actor",
        )
        bridge.start()

        # Update topic config from actor bus settings
        topic_scheme = str(actor_bus_cfg.scheme)
        topic_prefix = str(actor_bus_cfg.prefix)

        # Mutual exclusion: disable store-path publishers to avoid duplicates
        try:
            stores = [
                getattr(actor, "_feature_store", None),
                getattr(actor, "_model_store", None),
                getattr(actor, "_strategy_store", None),
                getattr(actor, "_data_store", None),
            ]
            for st in stores:
                if st is None:
                    continue
                if hasattr(st, "publisher"):
                    setattr(st, "publisher", None)
                if hasattr(st, "_enable_publishing"):
                    try:
                        setattr(st, "_enable_publishing", False)
                    except Exception as set_exc:
                        logging.getLogger(__name__).debug(
                            "Failed to disable store-level publishing: %s",
                            set_exc,
                        )
        except Exception as exc:
            # Never impact initialization on optional convenience — log debug
            logging.getLogger(__name__).debug(
                "Actor bus mutual exclusion setup failed: %s",
                exc,
            )

        return bridge, topic_scheme, topic_prefix
    except Exception as exc:
        # Best-effort helper; keep actor hot path clean — record warning metric
        try:
            from ml.common.metrics_manager import MetricsManager as _MM

            _MM.default().inc(
                "ml_pipeline_warnings_total",
                "Pipeline warnings",
                labels={
                    "component": "domain_events",
                    "op": "init_actor_bus_bridge",
                    "error_type": "exception",
                },
                labelnames=("component", "op", "error_type"),
            )
        except Exception as metric_exc:
            logging.getLogger(__name__).debug(
                "Warning metric emit failed (init_actor_bus_bridge): %s",
                metric_exc,
            )
        logging.getLogger(__name__).debug(
            "init_actor_bus_bridge failed: %s",
            exc,
        )
        return None, topic_scheme, topic_prefix


__all__ = [
    "DomainEventBridge",
    "TopicThrottleConfig",
    "init_actor_bus_bridge",
]
