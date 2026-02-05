from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterator

from ml.config.actor_bus import ActorBusConfig


@contextmanager
def env(vars: dict[str, str]) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in vars}
    try:
        os.environ.update(vars)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_actor_bus_config_defaults() -> None:
    cfg = ActorBusConfig.from_env()
    assert cfg.from_actor is False and cfg.from_strategy is False and cfg.from_store is False
    assert cfg.scheme in {"domain_op", "stage_first"}
    assert isinstance(cfg.prefix, str) and cfg.prefix
    assert cfg.max_queue > 0


def test_actor_bus_exclusive_prefer_actor() -> None:
    with env(
        {
            "ML_BUS_FROM_ACTOR": "1",
            "ML_BUS_FROM_STRATEGY": "1",
            "ML_BUS_FROM_STORE": "1",
            "ML_BUS_SCHEME": "stage_first",
            "ML_BUS_TOPIC_PREFIX": "events.ml",
        },
    ):
        cfg = ActorBusConfig.from_env()
        assert cfg.from_actor is True and cfg.from_strategy is False and cfg.from_store is False
        assert cfg.scheme == "stage_first" and cfg.prefix == "events.ml"


def test_actor_bus_throttle_parsing() -> None:
    with env(
        {
            "ML_BUS_THROTTLE_ENABLE": "true",
            "ML_BUS_THROTTLE_RATE": "25.5",
            "ML_BUS_THROTTLE_BURST": "10",
        },
    ):
        cfg = ActorBusConfig.from_env()
        assert cfg.throttle_enabled is True
        assert cfg.throttle_rate_per_sec == 25.5
        assert cfg.throttle_burst == 10


def test_actor_bus_prefers_strategy_over_store() -> None:
    with env(
        {
            "ML_BUS_FROM_STRATEGY": "1",
            "ML_BUS_FROM_STORE": "1",
        },
    ):
        cfg = ActorBusConfig.from_env()
        assert cfg.from_strategy is True and cfg.from_store is False
