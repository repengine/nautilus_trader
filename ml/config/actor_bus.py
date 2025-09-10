"""
Actor-side message bus configuration.

Resolves whether to publish from the actor thread (via DomainEventBridge) or from the
store path, topic scheme/prefix, and optional throttling parameters.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal


TopicScheme = Literal["domain_op", "stage_first"]


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass(frozen=True)
class ActorBusConfig:
    """
    Resolved actor-side bus configuration.

    Exactly one of `from_actor` or `from_store` should be enabled.

    """

    from_actor: bool
    from_store: bool
    scheme: TopicScheme
    prefix: str
    throttle_enabled: bool
    throttle_rate_per_sec: float
    throttle_burst: int

    @staticmethod
    def from_env() -> ActorBusConfig:
        """
        Create configuration from environment variables.
        """
        from_actor = _truthy("ML_BUS_FROM_ACTOR", default=False)
        from_store = _truthy("ML_BUS_FROM_STORE", default=False)

        scheme_raw = (os.getenv("ML_BUS_SCHEME") or "domain_op").strip().lower()
        scheme: TopicScheme = "stage_first" if scheme_raw == "stage_first" else "domain_op"
        prefix = os.getenv("ML_BUS_TOPIC_PREFIX", "events.ml").strip() or "events.ml"

        throttle_enabled = _truthy("ML_BUS_THROTTLE_ENABLE", default=False)
        try:
            throttle_rate = float(os.getenv("ML_BUS_THROTTLE_RATE", "100.0"))
        except ValueError:
            throttle_rate = 100.0
        try:
            throttle_burst = int(os.getenv("ML_BUS_THROTTLE_BURST", "100"))
        except ValueError:
            throttle_burst = 100

        # Exclusive path validation (not fatal): prefer actor if both set
        if from_actor and from_store:
            # Prefer actor to avoid store hot-path I/O; disable store path
            from_store = False

        return ActorBusConfig(
            from_actor=from_actor,
            from_store=from_store,
            scheme=scheme,
            prefix=prefix,
            throttle_enabled=throttle_enabled,
            throttle_rate_per_sec=throttle_rate,
            throttle_burst=throttle_burst,
        )


__all__: Final[list[str]] = ["ActorBusConfig", "TopicScheme"]
