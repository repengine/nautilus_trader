from __future__ import annotations

from ml.common.message_bus import NoopPublisher, RedisStreamsPublisher, publisher_from_config
from ml.config.bus import MessageBusConfig


def test_publisher_from_config_disabled_returns_noop() -> None:
    cfg = MessageBusConfig(enabled=False, backend="noop")
    pub = publisher_from_config(cfg)
    assert isinstance(pub, NoopPublisher)


def test_publisher_from_config_redis_backend() -> None:
    cfg = MessageBusConfig(
        enabled=True,
        backend="redis",
        scheme="domain_op",
        topic_prefix="events.ml",
        redis_url="redis://localhost:6379/0",
        redis_stream="ml-events",
    )
    pub = publisher_from_config(cfg)
    # Does not require real Redis client; constructor handles absence gracefully
    assert isinstance(pub, RedisStreamsPublisher)

