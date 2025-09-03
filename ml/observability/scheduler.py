"""
Background flusher for observability (off hot-path).

This scheduler can be driven via ticks (deterministic tests) or started in a
background thread for periodic persistence of observability tables.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ml.observability.persistence import ObservabilityPersistor
from ml.observability.service import ObservabilityService


NowFunc = Callable[[], float]


@dataclass(slots=True)
class ObservabilityFlusher:
    """Flush observability tables to disk periodically.

    Prefer using ``tick`` in tests for determinism. For background operation,
    use ``start_background`` with a caller-managed ``threading.Event`` to stop.
    """

    service: ObservabilityService
    base_path: Path
    file_format: str = "jsonl"
    interval_seconds: float = 60.0
    now: NowFunc = field(default=lambda: time.time())
    _last_flush: float = field(default=0.0, init=False)

    def flush_once(self) -> dict[str, Path]:
        tables = {
            "latency": self.service.latency_watermarks_df(),
            "metrics": self.service.metrics_collection_df(),
            "correlation": self.service.event_correlation_df(),
            "health": self.service.health_scores_df(),
        }
        sink = ObservabilityPersistor(base_path=self.base_path, file_format=self.file_format)
        out = sink.persist(tables)
        self._last_flush = self.now()
        return out

    def tick(self) -> dict[str, Path]:
        """Flush if interval has elapsed; return mapping of written files."""
        if self.interval_seconds <= 0:
            return self.flush_once()
        if self.now() - self._last_flush >= self.interval_seconds:
            return self.flush_once()
        return {}

    def start_background(self, stop_event: threading.Event) -> threading.Thread:
        """
        Start a background thread that ticks until ``stop_event`` is set.
        Caller is responsible for setting the event and joining the thread.
        """

        def _run() -> None:
            while not stop_event.is_set():
                try:
                    self.tick()
                except Exception:
                    # Keep background resilient
                    pass
                # Sleep a small fraction to avoid busy spinning; cap by interval
                time.sleep(max(0.01, min(0.5, self.interval_seconds)))

        t = threading.Thread(target=_run, name="ObservabilityFlusher", daemon=True)
        t.start()
        return t


__all__ = ["ObservabilityFlusher"]

