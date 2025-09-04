from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ml.observability.scheduler import ObservabilityFlusher


class DummyService:
    def latency_watermarks_df(self) -> pd.DataFrame:
        return pd.DataFrame([{"a": 1}])

    def metrics_collection_df(self) -> pd.DataFrame:
        return pd.DataFrame([{"a": 1}])

    def event_correlation_df(self) -> pd.DataFrame:
        return pd.DataFrame([{"a": 1}])

    def health_scores_df(self) -> pd.DataFrame:
        return pd.DataFrame([{"a": 1}])


def test_background_flusher_handles_persist_errors(monkeypatch: Any, tmp_path: Path) -> None:
    from ml import observability as obs_pkg  # type: ignore[import-not-found]
    from ml.observability import persistence as _persistence

    # Force ObservabilityPersistor.persist to raise
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(_persistence.ObservabilityPersistor, "persist", _boom)

    flusher = ObservabilityFlusher(
        service=DummyService(),
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
    flusher = ObservabilityFlusher(
        service=DummyService(),
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
