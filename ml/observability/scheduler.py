"""
Background flusher for observability (off hot-path).

This scheduler can be driven via ticks (deterministic tests) or started in a background
thread for periodic persistence of observability tables.

"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Protocol, cast

from ml.config.observability import ObservabilityConfig
from ml.observability.persistence import ObservabilityPersistor
from ml.observability.service import ObservabilityService


NowFunc = Callable[[], float]
logger = logging.getLogger(__name__)


class ObservabilityRuntimeProtocol(Protocol):
    """
    Runtime surface required to execute observability start commands.
    """

    def start_observability_from_config(self, cfg: object) -> None:
        """Start observability from config."""
        ...

    def start_observability_flush(
        self,
        *,
        base_path: Path,
        interval_seconds: float | None = 60.0,
        file_format: str = "jsonl",
        sink: str = "file",
        db_connection_string: str | None = None,
    ) -> dict[str, Path] | None:
        """Start sync background flush."""
        ...

    def stop_observability_flush(self) -> None:
        """Stop sync background flush."""
        ...


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
        """
        Flush if interval has elapsed; return mapping of written files.
        """
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
                except Exception as exc:
                    # Keep background resilient — log and record a counter
                    try:
                        import logging as _logging

                        from ml.common.metrics_manager import MetricsManager as _MM

                        _logging.getLogger(__name__).debug(
                            "Observability flusher tick failed: %s",
                            exc,
                            exc_info=True,
                        )
                        _MM.default().inc(
                            "nautilus_ml_observability_errors_total",
                            "Total errors observed in observability components",
                            labels={"component": "observability_flusher", "kind": "tick"},
                            labelnames=("component", "kind"),
                        )
                    except Exception as log_exc:
                        # Never impact control flow on metrics/logging failures
                        import logging as _logging

                        _logging.getLogger(__name__).debug(
                            "Logging/metrics for flusher tick also failed: %s",
                            log_exc,
                            exc_info=True,
                        )
                # Sleep a small fraction to avoid busy spinning; cap by interval
                time.sleep(max(0.01, min(0.5, self.interval_seconds)))

        t = threading.Thread(target=_run, name="ObservabilityFlusher", daemon=True)
        t.start()
        return t


@dataclass(slots=True, frozen=True)
class ObservabilityStartConfig:
    """
    Runtime config for `start` observability CLI command.

    Args:
        sink: Output sink ("file" or "db").
        base_path: Base path for file sink.
        file_format: File sink format ("jsonl" or "csv").
        db_url: Optional DB URL for DB sink.
        interval_seconds: Flush interval seconds.
        duration_seconds: Optional bounded run duration seconds.
        async_enabled: Whether async worker mode is enabled.
        async_queue_maxsize: Async queue capacity.
        async_component_label: Metrics component label for async worker.
    """

    sink: str = "file"
    base_path: Path = Path("./observability")
    file_format: str = "jsonl"
    db_url: str | None = None
    interval_seconds: float = 60.0
    duration_seconds: float = 0.0
    async_enabled: bool = False
    async_queue_maxsize: int = 4096
    async_component_label: str = "obs_async_worker"

    def __post_init__(self) -> None:
        if self.sink not in {"file", "db"}:
            raise ValueError("sink must be 'file' or 'db'")
        if self.file_format not in {"jsonl", "csv"}:
            raise ValueError("file_format must be 'jsonl' or 'csv'")
        if self.interval_seconds < 0.0:
            raise ValueError("interval_seconds must be >= 0")
        if self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be >= 0")
        if self.async_queue_maxsize < 1:
            raise ValueError("async_queue_maxsize must be >= 1")
        if not self.async_component_label.strip():
            raise ValueError("async_component_label must be non-empty")


async def _stop_async_worker(runtime: ObservabilityRuntimeProtocol) -> None:
    worker = getattr(runtime, "_obs_async_worker", None)
    if worker is None:
        return
    stop_method = getattr(worker, "stop", None)
    if not callable(stop_method):
        return
    try:
        await cast(Callable[..., Awaitable[object]], stop_method)(drain=True, timeout=1.0)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Async worker stop failed: %s", exc, exc_info=True)


def run_observability_start(
    runtime: ObservabilityRuntimeProtocol,
    config: ObservabilityStartConfig,
) -> int:
    """
    Run observability background start flow in sync or async mode.

    Args:
        runtime: Runtime object exposing observability start/stop operations.
        config: Start configuration.

    Returns:
        Process exit code.
    """
    if config.async_enabled:

        async def _run_async() -> int:
            runtime.start_observability_from_config(
                ObservabilityConfig(
                    sink="db" if config.sink == "db" else "file",
                    base_path=str(config.base_path),
                    file_format=config.file_format,
                    db_connection_string=config.db_url,
                    interval_seconds=config.interval_seconds,
                    async_enabled=True,
                    async_queue_maxsize=config.async_queue_maxsize,
                    async_component_label=config.async_component_label,
                ),
            )
            if config.duration_seconds > 0.0:
                await asyncio.sleep(config.duration_seconds)
                await _stop_async_worker(runtime)
            return 0

        return asyncio.run(_run_async())

    runtime.start_observability_flush(
        base_path=config.base_path,
        interval_seconds=config.interval_seconds,
        file_format=config.file_format,
        sink=config.sink,
        db_connection_string=config.db_url,
    )
    if config.duration_seconds > 0.0:
        time.sleep(config.duration_seconds)
        runtime.stop_observability_flush()
    return 0


__all__ = [
    "ObservabilityFlusher",
    "ObservabilityStartConfig",
    "run_observability_start",
]
