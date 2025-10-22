from __future__ import annotations

# ruff: noqa: I001

from typing import Any
from types import ModuleType

from ml import _imports as ml_imports
from ml.common.message_bus import NoopPublisher
from ml.common.message_bus import publisher_from_config
from ml.common.message_bus import RedisStreamsPublisher
from ml.config.bus import MessageBusConfig


class DummyRedis:
    class Redis:
        """
        Dummy stand-in for redis.Redis with an xadd method.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs
            self.xadd_calls: list[tuple[str, dict[str, str]]] = []

        @classmethod
        def from_url(cls, url: str, decode_responses: bool = False) -> DummyRedis.Redis:
            return cls(url, decode_responses=decode_responses)

        def xadd(
            self,
            stream: str,
            fields: dict[str, str],
            maxlen: int | None = None,
            approximate: bool | None = None,
        ) -> str:
            self.xadd_calls.append((stream, fields))
            return "0-0"


def test_noop_publisher_returns_false() -> None:
    pub = NoopPublisher()
    ok = pub.publish("topic", {"a": 1})
    assert ok is False


def test_redis_streams_publisher_uses_client(monkeypatch: Any) -> None:
    dummy_module = ModuleType("redis")
    setattr(dummy_module, "Redis", DummyRedis.Redis)
    monkeypatch.setattr(ml_imports, "redis", dummy_module, raising=False)
    monkeypatch.setattr(ml_imports, "HAS_REDIS", True, raising=False)
    pub = RedisStreamsPublisher(url="redis://localhost:6379/0", stream="ml-events", maxlen=10)
    ok = pub.publish("ml.data.created.EURUSD.SIM", {"k": 1})
    assert ok is True


def test_publisher_from_config_factory(monkeypatch: Any) -> None:
    # When disabled -> Noop
    cfg = MessageBusConfig(enabled=False, backend="noop")
    pub = publisher_from_config(cfg)
    assert isinstance(pub, NoopPublisher)

    # When enabled with redis -> RedisStreamsPublisher (with dummy redis module)
    dummy_module = ModuleType("redis")
    setattr(dummy_module, "Redis", DummyRedis.Redis)
    monkeypatch.setattr(ml_imports, "redis", dummy_module, raising=False)
    monkeypatch.setattr(ml_imports, "HAS_REDIS", True, raising=False)
    cfg2 = MessageBusConfig(enabled=True, backend="redis")
    pub2 = publisher_from_config(cfg2)
    assert isinstance(pub2, RedisStreamsPublisher)


def test_redis_publisher_falls_back_without_dependency(monkeypatch: Any) -> None:
    monkeypatch.setattr(ml_imports, "redis", None, raising=False)
    monkeypatch.setattr(ml_imports, "HAS_REDIS", False, raising=False)
    cfg = MessageBusConfig(enabled=True, backend="redis")
    pub = publisher_from_config(cfg)
    assert isinstance(pub, NoopPublisher)


def test_redis_streams_publisher_no_client_when_missing_dependency(monkeypatch: Any) -> None:
    monkeypatch.setattr(ml_imports, "redis", None, raising=False)
    monkeypatch.setattr(ml_imports, "HAS_REDIS", False, raising=False)
    pub = RedisStreamsPublisher(url="redis://localhost:6379/0", stream="ml-events", maxlen=10)
    ok = pub.publish("topic", {"payload": 1})
    assert ok is False
