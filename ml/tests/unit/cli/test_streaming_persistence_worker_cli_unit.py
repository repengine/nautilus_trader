from __future__ import annotations

import argparse
from typing import Any

import pytest

from ml.cli import streaming_persistence_worker as cli
from ml.config.streaming_pipeline import StreamingPersistenceConfig


def test_build_config_merges_overrides() -> None:
    args = argparse.Namespace(
        state_path="/tmp/override.json",
        batch_size=32,
        block_ms=200,
        poll_interval_seconds=0.25,
        enable=True,
        disable=False,
    )
    env = {
        "ML_STREAM_PERSIST_ENABLE": "0",
        "ML_STREAM_PERSIST_STATE_PATH": "/tmp/default.json",
        "ML_STREAM_PERSIST_BATCH_SIZE": "16",
    }
    cfg = cli.build_config(args, env=env)
    assert cfg.enabled is True
    assert cfg.state_path == "/tmp/override.json"
    assert cfg.batch_size == 32
    assert cfg.block_ms == 200
    assert cfg.poll_interval_seconds == 0.25


def test_streaming_persistence_cli_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: list[Any] = []

    class DummyWorker:
        def __init__(
            self,
            config: StreamingPersistenceConfig,
            message_bus_config: Any,
            *,
            observability: Any = None,
            state_store: Any = None,
            consumer_factory: Any = None,
        ) -> None:
            self.config = config
            self.message_bus_config = message_bus_config
            self.observability = observability
            self.state_store = state_store
            self.consumer_factory = consumer_factory
            self.ran = False
            self.stop_called = False
            instances.append(self)

        def run_forever(self) -> None:
            self.ran = True

        def stop(self) -> None:
            self.stop_called = True

    monkeypatch.setattr(cli, "StreamingTrainingPersistenceWorker", DummyWorker)
    rc = cli.main(["--disable", "--batch-size", "8"])
    assert rc == 0
    assert len(instances) == 1
    worker = instances[0]
    assert worker.ran is True
    assert worker.config.batch_size == 8
    assert worker.config.enabled is False
