from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from ml import _imports as ml_imports
from ml.common.message_bus import FilePublisher, NoopPublisher, RedisStreamsPublisher, publisher_from_config
from ml.config.bus import MessageBusConfig


def test_publisher_from_config_disabled_returns_noop() -> None:
    cfg = MessageBusConfig(enabled=False, backend="noop")
    pub = publisher_from_config(cfg)
    assert isinstance(pub, NoopPublisher)


def test_publisher_from_config_redis_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_module = ModuleType("redis")

    class _DummyRedisClient:
        @classmethod
        def from_url(cls, url: str, decode_responses: bool = False) -> _DummyRedisClient:  # noqa: ARG003
            return cls()

        def xadd(self, *args: object, **kwargs: object) -> str:  # noqa: ARG002
            return "0-0"

    setattr(dummy_module, "Redis", _DummyRedisClient)
    monkeypatch.setattr(ml_imports, "redis", dummy_module, raising=False)
    monkeypatch.setattr(ml_imports, "HAS_REDIS", True, raising=False)

    cfg = MessageBusConfig(
        enabled=True,
        backend="redis",
        scheme="domain_op",
        topic_prefix="events.ml",
        redis_url="redis://localhost:6379/0",
        redis_stream="ml-events",
    )
    pub = publisher_from_config(cfg)
    assert isinstance(pub, RedisStreamsPublisher)


def test_publisher_from_config_file_backend(tmp_path: Path) -> None:
    cfg = MessageBusConfig(
        enabled=True,
        backend="file",
        scheme="domain_op",
        topic_prefix="events.ml",
        file_path=str(tmp_path / "bus.jsonl"),
    )
    pub = publisher_from_config(cfg)
    assert isinstance(pub, FilePublisher)
