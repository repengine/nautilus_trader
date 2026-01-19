#!/usr/bin/env python3
from __future__ import annotations

import os

from pytest import MonkeyPatch

from ml.config.bus import MessageBusConfig


def test_message_bus_config_from_env_defaults(monkeypatch: MonkeyPatch) -> None:
    # Clear related env
    for name in (
        "ML_BUS_ENABLE",
        "ML_BUS_BACKEND",
        "ML_BUS_SCHEME",
        "ML_BUS_TOPIC_PREFIX",
        "ML_BUS_REDIS_URL",
        "ML_BUS_REDIS_STREAM",
        "ML_BUS_REDIS_MAXLEN",
        "ML_BUS_FILE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = MessageBusConfig.from_env()
    assert cfg.enabled is False
    assert cfg.backend == "noop"
    assert cfg.scheme == "domain_op"
    assert cfg.topic_prefix == "events.ml"
    assert cfg.redis_url.startswith("redis://")
    assert cfg.redis_stream == "ml-events"
    assert cfg.redis_maxlen is None
    assert cfg.file_path is None


def test_message_bus_config_from_env_overrides(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("ML_BUS_ENABLE", "true")
    monkeypatch.setenv("ML_BUS_BACKEND", "redis")
    monkeypatch.setenv("ML_BUS_SCHEME", "stage_first")
    monkeypatch.setenv("ML_BUS_TOPIC_PREFIX", "events.custom")
    monkeypatch.setenv("ML_BUS_REDIS_URL", "redis://example:6380/1")
    monkeypatch.setenv("ML_BUS_REDIS_STREAM", "custom-stream")
    monkeypatch.setenv("ML_BUS_REDIS_MAXLEN", "1000")
    monkeypatch.setenv("ML_BUS_FILE_PATH", "/tmp/ml-events.jsonl")

    cfg = MessageBusConfig.from_env()
    assert cfg.enabled is True
    assert cfg.backend == "redis"
    assert cfg.scheme == "stage_first"
    assert cfg.topic_prefix == "events.custom"
    assert cfg.redis_url == "redis://example:6380/1"
    assert cfg.redis_stream == "custom-stream"
    assert cfg.redis_maxlen == 1000
    assert cfg.file_path == "/tmp/ml-events.jsonl"
