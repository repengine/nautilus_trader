from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.common.events_util import build_bus_payload
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.config.replay import LiveReplayConfig
from ml.deployment.live_paced_replay import ReplayEvent
from ml.deployment.live_paced_replay import load_replay_events
from ml.deployment.live_paced_replay import replay_events


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _CapturePublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def publish(self, topic: str, payload: dict[str, object]) -> bool:
        self.calls.append((topic, payload))
        return True


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )


def test_load_replay_events_filters_invalid_payloads(tmp_path: Path) -> None:
    valid_payload = build_bus_payload(
        dataset_id="dataset",
        instrument_id="EURUSD",
        stage=Stage.FEATURE_COMPUTED,
        source=Source.LIVE,
        run_id="run",
        ts_min=1,
        ts_max=1,
        count=1,
        status=EventStatus.SUCCESS,
        metadata={"note": "ok"},
    )
    invalid_payload = {"ts_min": 1}
    records = [
        {"topic": "events.ml.FEATURE_COMPUTED.EURUSD", "payload": valid_payload},
        {"topic": "events.ml.FEATURE_COMPUTED.EURUSD", "payload": invalid_payload},
    ]

    path = tmp_path / "events.jsonl"
    _write_jsonl(path, records)

    result = load_replay_events(path, config=LiveReplayConfig())

    assert result.skipped == 1
    assert len(result.events) == 1
    assert result.events[0].timestamp_ns == 1


def test_replay_events_paces_by_timestamp() -> None:
    events = (
        ReplayEvent(topic="t1", payload={"ts_min": 1, "ts_max": 1}, timestamp_ns=1_000_000_000),
        ReplayEvent(topic="t2", payload={"ts_min": 3, "ts_max": 3}, timestamp_ns=3_000_000_000),
    )
    publisher = _CapturePublisher()
    clock = _FakeClock()
    config = LiveReplayConfig(speed_multiplier=2.0)

    summary = replay_events(events, publisher, config=config, clock=clock)

    assert summary.published == 2
    assert np.isclose(clock.sleeps[0], 1.0)
    assert len(publisher.calls) == 2
