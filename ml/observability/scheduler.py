"""
Background flusher for observability (off hot-path).

This scheduler can be driven via ticks (deterministic tests) or started in a
background thread for periodic persistence of observability tables.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from ml.observability.persistence import ObservabilityPersistor
from ml.observability.service import ObservabilityService


NowFunc = Callable[[], float]


@dataclass(slots=True)
class ObservabilityFlusher:
    """
    Flush observability tables to disk periodically.

    Prefer using ``tick`` in tests for determinism. For background operation,
    use ``start_background`` with a caller-managed ``threading.Event`` to stop.
    """

    service: ObservabilityService
    base_path: Path
    file_format: str = "jsonl"
    interval_seconds: float = 60.0
    now: NowFunc = field(default=lambda: time.time())
    _last_flush: float = field(default=0.0, init=False)
    sink: str = "file"  # one of {"file", "db"}
    db_connection_string: str | None = None

    def flush_once(self) -> dict[str, Path] | dict[str, int]:
        tables = {
            "latency": self.service.latency_watermarks_df(),
            "metrics": self.service.metrics_collection_df(),
            "correlation": self.service.event_correlation_df(),
            "health": self.service.health_scores_df(),
        }
        if self.sink == "db":
            from ml.observability.db_persistence import ObservabilityDBPersistor

            if not self.db_connection_string:
                return {}
            per = ObservabilityDBPersistor(connection_string=self.db_connection_string)
            written = per.persist(tables)
            self._last_flush = self.now()
            # Return a normalized dict[str, Path] | dict[str, int]; choose int map for DB sink
            return written
        else:
            sink = ObservabilityPersistor(base_path=self.base_path, file_format=self.file_format)
            out = sink.persist(tables)
            self._last_flush = self.now()
            return out

    def tick(self) -> dict[str, Path] | dict[str, int]:
        """Flush if interval has elapsed; return mapping of written files."""
        if self.interval_seconds <= 0:
            return self.flush_once()
        if self.now() - self._last_flush >= self.interval_seconds:
            return self.flush_once()
        # Return consistent type depending on sink
        return {} if self.sink == "db" else {}

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
