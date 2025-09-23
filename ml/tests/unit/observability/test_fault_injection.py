from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from ml.observability.scheduler import ObservabilityFlusher
from ml.observability.service import ObservabilityService


def test_background_flusher_handles_persist_errors(monkeypatch: Any, tmp_path: Path) -> None:
    from ml import observability as obs_pkg
    from ml.observability import persistence as _persistence

    # Force ObservabilityPersistor.persist to raise
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(_persistence.ObservabilityPersistor, "persist", _boom)

    svc = ObservabilityService()
    svc.add_latency_stage(
        correlation_id="c1",
        instrument_id="inst",
        pipeline_stage="stage",
        ts_stage_start=1,
        ts_stage_end=2,
    )
    flusher = ObservabilityFlusher(
        service=svc,
        base_path=tmp_path,
        file_format="jsonl",
        interval_seconds=0.05,
    )
    stop = threading.Event()
    t = flusher.start_background(stop)
    # Let it run a couple of ticks
    time.sleep(0.2)
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_background_flusher_handles_db_errors(tmp_path: Path) -> None:
    svc = ObservabilityService()
    svc.add_latency_stage(
        correlation_id="c1",
        instrument_id="inst",
        pipeline_stage="stage",
        ts_stage_start=1,
        ts_stage_end=2,
    )
    flusher = ObservabilityFlusher(
        service=svc,
        base_path=tmp_path,
        file_format="jsonl",
        interval_seconds=0.05,
        sink="db",
        db_connection_string="postgresql://invalid:invalid@localhost:1/doesnotexist",
    )
    stop = threading.Event()
    t = flusher.start_background(stop)
    time.sleep(0.2)
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()
