from __future__ import annotations

import os
from contextlib import contextmanager

from ml.config.bus import MessageBusConfig


@contextmanager
def temp_env(env: dict[str, str]) -> None:
    old = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestMessageBusConfig:
    def test_defaults_disabled(self) -> None:
        cfg = MessageBusConfig.from_env()
        assert cfg.enabled is False
        assert cfg.backend == "noop"
        assert cfg.scheme == "domain_op"
        assert cfg.topic_prefix == "events.ml"

    def test_env_parsing(self) -> None:
        with temp_env(
            {
                "ML_BUS_ENABLE": "1",
                "ML_BUS_BACKEND": "redis",
                "ML_BUS_SCHEME": "stage_first",
                "ML_BUS_TOPIC_PREFIX": "events.custom",
                "ML_BUS_REDIS_URL": "redis://localhost:6379/1",
                "ML_BUS_REDIS_STREAM": "events-stream",
                "ML_BUS_REDIS_MAXLEN": "5000",
            },
        ):
            cfg = MessageBusConfig.from_env()
            assert cfg.enabled is True
            assert cfg.backend == "redis"
            assert cfg.scheme == "stage_first"
            assert cfg.topic_prefix == "events.custom"
            assert cfg.redis_url.endswith("/1")
            assert cfg.redis_stream == "events-stream"
            assert cfg.redis_maxlen == 5000
