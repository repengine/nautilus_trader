from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml.cli import streaming_persistence_worker as cli
from ml.consumers.streaming_training_worker import StreamingTrainingPersistenceWorker
from ml.tests.fixtures.streaming_events import build_streaming_test_payloads


@pytest.mark.integration
def test_streaming_persistence_worker_cli_disable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("ML_STREAM_PERSIST_STATE_PATH", str(state_path))
    monkeypatch.setenv("ML_STREAM_PERSIST_ENABLE", "0")

    rc = cli.main(["--disable"])
    assert rc == 0


@pytest.mark.integration
def test_streaming_persistence_worker_cli_persists_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_id = "cli-dataset"
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("ML_STREAM_PERSIST_STATE_PATH", str(state_path))
    monkeypatch.setenv("ML_STREAM_PERSIST_ENABLE", "1")

    payloads = build_streaming_test_payloads(dataset_id=dataset_id, plan_id="plan-cli", parquet_path=tmp_path / "dataset.parquet")
    messages: list[tuple[str, dict[str, Any]]] = [
        (f"events.ml.DATASET_PLANNED.{dataset_id}", payloads.plan_message()),
        (f"events.ml.MODEL_TRAINING_COMPLETED.{dataset_id}", payloads.result_message()),
        (f"events.ml.WORKER_HEARTBEAT.{dataset_id}", payloads.heartbeat_message()),
    ]

    class _RecordingWorker(StreamingTrainingPersistenceWorker):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._messages = messages

        def run_forever(self) -> None:
            for topic, payload in self._messages:
                self.service.handle(topic, payload)

    monkeypatch.setattr(cli, "StreamingTrainingPersistenceWorker", _RecordingWorker)
    monkeypatch.setattr(cli, "_install_signal_handlers", lambda worker: None)

    rc = cli.main([])
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert state_path.exists()

    snapshot = json.loads(state_path.read_text(encoding="utf-8"))
    plan_entry = snapshot["plans"][payloads.plan_event.plan_id]
    result_entry = snapshot["results"][payloads.result_event.plan_id]
    assert plan_entry["metadata_summary"]["total_shards"] == payloads.plan_event.metadata_summary.total_shards
    assert result_entry["artifact_paths"]["logits"].endswith("tft_cli_logits.npz")
    assert snapshot["heartbeats"]
