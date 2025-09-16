"""
Async observability worker (off hot-path) with bounded queue.

This component provides a minimal asyncio-based background worker that accepts
lightweight row items (latency, metrics, correlation, health) via non-blocking
enqueue methods, aggregates them into an ObservabilityService, and periodically
persists tables to disk/DB using existing sinks. Producers use non-blocking
``enqueue_*`` methods; when the queue is full, the worker drops the item and
increments backpressure metrics (no allocations in hot paths).

Notes
-----
- Keep hot loops free of I/O. Only enqueue from hot paths; persistence occurs
  off-path inside the worker task.
- All timestamps are nanoseconds since epoch.
- Metrics are acquired via ml.common.metrics_bootstrap.

"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal, TypedDict

from ml.common.metrics_manager import MetricsManager
from ml.observability.service import ObservabilityService


SinkKind = Literal["file", "db"]


class _LatencyItem(TypedDict):
    kind: Literal["latency"]
    correlation_id: str
    instrument_id: str
    pipeline_stage: str
    ts_stage_start: int
    ts_stage_end: int


class _MetricItem(TypedDict):
    kind: Literal["metrics"]
    metric_name: str
    metric_type: str
    value: float
    timestamp: int
    labels: dict[str, object] | str | None


class _CorrelationItem(TypedDict):
    kind: Literal["correlation"]
    correlation_id: str
    event_id: str
    parent_event_id: str | None
    instrument_id: str
    domain: str
    lineage_depth: int
    ts_event: int
    propagation_path: list[str] | str


class _HealthItem(TypedDict):
    kind: Literal["health"]
    component_id: str
    health_score: float
    subsystem_scores: dict[str, float] | str
    timestamp: int
    measurement_window_ms: int


QueueItem = _LatencyItem | _MetricItem | _CorrelationItem | _HealthItem


@dataclass(slots=True)
class ObservabilityAsyncWorker:
    """
    Async worker to aggregate observability rows and persist off hot-path.

    Parameters
    ----------
    service : ObservabilityService
        The in-memory service to collect rows and materialize tables.
    sink : {"file", "db"}
        Persistence sink kind. "file" uses JSONL/CSV; "db" uses SQLAlchemy.
    base_path : Path | None
        Base path for file sink (required when sink == "file").
    db_connection_string : str | None
        Database URL for DB sink (required when sink == "db").
    flush_interval_seconds : float
        Periodic flush interval in seconds.
    queue_maxsize : int
        Bounded queue capacity; enqueue drops when full (backpressure).
    component_label : str
        Component label for metrics (default: "obs_async_worker").

    """

    service: ObservabilityService
    sink: SinkKind = "file"
    base_path: Path | None = None
    db_connection_string: str | None = None
    flush_interval_seconds: float = 5.0
    queue_maxsize: int = 4096
    component_label: str = "obs_async_worker"
    use_async_db: bool = False

    _queue: asyncio.Queue[QueueItem] = field(init=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False)
    _stop: asyncio.Event = field(init=False)
    _last_flush: float = field(default=0.0, init=False)

    # Metrics via manager (uses bootstrap under the hood)
    _MM = MetricsManager.default()
    _ENQUEUED = _MM.counter(
        "nautilus_ml_observability_enqueued_total",
        "Total observability items enqueued",
        ["kind"],
    )
    _FLUSH_SEC = _MM.histogram(
        "nautilus_ml_observability_async_flush_duration_seconds",
        "Async observability flush duration",
        ["sink"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    )
    _Q_DEPTH = _MM.gauge(
        "nautilus_ml_observability_queue_depth",
        "Current depth of the observability async queue",
        ["component"],
    )
    _ERRORS = _MM.counter(
        "nautilus_ml_observability_errors_total",
        "Total errors observed in observability async worker",
        ["component", "kind"],
    )

    _LOGGER = logging.getLogger(__name__)

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue(maxsize=int(self.queue_maxsize))
        self._stop = asyncio.Event()

    # ------------------------------ API ----------------------------------

    def start(self) -> None:
        """
        Start background worker task (idempotent).
        """
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="ObservabilityAsyncWorker")

    async def stop(self, *, drain: bool = True, timeout: float | None = 2.0) -> None:
        """
        Stop worker; optionally drain queue before exit.
        """
        if drain:
            # Best-effort drain with time-bound
            start = time.perf_counter()
            while not self._queue.empty() and (
                timeout is None or time.perf_counter() - start < timeout
            ):
                await asyncio.sleep(0.01)
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except Exception:
                # Ensure task cancelled on timeout or error
                self._task.cancel()
                with contextlib.suppress(Exception):
                    await self._task

    def queue_size(self) -> int:
        """
        Return current queue size.

        This is a cheap call and safe for status polling.

        """
        return int(self._queue.qsize())

    # --------------------------- Enqueue API -----------------------------

    def enqueue_latency(
        self,
        *,
        correlation_id: str,
        instrument_id: str,
        pipeline_stage: str,
        ts_stage_start: int,
        ts_stage_end: int,
    ) -> bool:
        return self._try_put(
            _LatencyItem(
                kind="latency",
                correlation_id=correlation_id,
                instrument_id=instrument_id,
                pipeline_stage=pipeline_stage,
                ts_stage_start=int(ts_stage_start),
                ts_stage_end=int(ts_stage_end),
            ),
        )

    def enqueue_metric(
        self,
        *,
        metric_name: str,
        metric_type: str,
        value: float,
        timestamp: int,
        labels: dict[str, object] | str | None = None,
    ) -> bool:
        return self._try_put(
            _MetricItem(
                kind="metrics",
                metric_name=metric_name,
                metric_type=metric_type,
                value=float(value),
                timestamp=int(timestamp),
                labels=labels,
            ),
        )

    def enqueue_correlation(
        self,
        *,
        correlation_id: str,
        event_id: str,
        parent_event_id: str | None,
        instrument_id: str,
        domain: str,
        lineage_depth: int,
        ts_event: int,
        propagation_path: list[str] | str,
    ) -> bool:
        return self._try_put(
            _CorrelationItem(
                kind="correlation",
                correlation_id=correlation_id,
                event_id=event_id,
                parent_event_id=parent_event_id,
                instrument_id=instrument_id,
                domain=domain,
                lineage_depth=int(lineage_depth),
                ts_event=int(ts_event),
                propagation_path=propagation_path,
            ),
        )

    def enqueue_health(
        self,
        *,
        component_id: str,
        health_score: float,
        subsystem_scores: dict[str, float] | str,
        timestamp: int,
        measurement_window_ms: int,
    ) -> bool:
        return self._try_put(
            _HealthItem(
                kind="health",
                component_id=component_id,
                health_score=float(health_score),
                subsystem_scores=subsystem_scores,
                timestamp=int(timestamp),
                measurement_window_ms=int(measurement_window_ms),
            ),
        )

    # --------------------------- Internals --------------------------------

    def _try_put(self, item: QueueItem) -> bool:
        try:
            self._queue.put_nowait(item)
            self._ENQUEUED.labels(kind=item["kind"]).inc()
            # Update queue depth gauge cheaply
            self._Q_DEPTH.labels(component=self.component_label).set(self._queue.qsize())
            return True
        except asyncio.QueueFull:
            # Record backpressure drop via MetricsManager (off hot path)
            try:
                mm = MetricsManager.default()
                mm.inc(
                    "nautilus_ml_backpressure_drops_total",
                    "Total events dropped due to backpressure",
                    labels={"component": self.component_label, "reason": "queue_full"},
                    labelnames=("component", "reason"),
                )
            except Exception as exc:
                # Best-effort metrics; never raise from hot path
                self._LOGGER.debug("Backpressure metric emit failed: %s", exc, exc_info=True)
            return False

    async def _run(self) -> None:
        import contextlib
        from typing import Any as _Any

        # Lazy init of sinks to avoid overhead until needed
        file_sink: _Any = None
        db_sink: _Any = None
        db_async: _Any = None

        def _flush_sync() -> None:
            nonlocal file_sink, db_sink
            start = time.perf_counter()
            tables = {
                "latency": self.service.latency_watermarks_df(),
                "metrics": self.service.metrics_collection_df(),
                "correlation": self.service.event_correlation_df(),
                "health": self.service.health_scores_df(),
            }
            if self.sink == "db":
                if db_sink is None:
                    from ml.observability.db_persistence import ObservabilityDBPersistor

                    db_sink = ObservabilityDBPersistor(
                        connection_string=str(self.db_connection_string or ""),
                    )
                db_sink.persist(tables)
            else:
                if file_sink is None:
                    from ml.observability.persistence import ObservabilityPersistor

                    file_sink = ObservabilityPersistor(
                        base_path=Path(self.base_path or Path("./observability")),
                        file_format="jsonl",
                    )
                file_sink.persist(tables)
            dur = time.perf_counter() - start
            self._FLUSH_SEC.labels(sink=self.sink).observe(dur)
            self._last_flush = time.time()

        async def _flush_async_db() -> None:
            nonlocal db_async
            if db_async is None:
                from ml.observability.async_db_persistence import ObservabilityAsyncDBPersistor

                db_async = ObservabilityAsyncDBPersistor(
                    connection_string=str(self.db_connection_string or ""),
                )
            tables = {
                "latency": self.service.latency_watermarks_df(),
                "metrics": self.service.metrics_collection_df(),
                "correlation": self.service.event_correlation_df(),
                "health": self.service.health_scores_df(),
            }
            start = time.perf_counter()
            await db_async.persist_async(tables)
            dur = time.perf_counter() - start
            self._FLUSH_SEC.labels(sink=self.sink).observe(dur)
            self._last_flush = time.time()

        # Main loop: drain queue with small sleeps, flush periodically
        while not self._stop.is_set():
            try:
                # Pull at most a small batch to avoid starving flush
                for _ in range(256):
                    item = await asyncio.wait_for(self._queue.get(), timeout=0.05)
                    kind = item["kind"]
                    if kind == "latency":
                        from typing import Any, cast

                        i_lat = cast(_LatencyItem, item)
                        self.service.add_latency_stage(
                            correlation_id=i_lat["correlation_id"],
                            instrument_id=i_lat["instrument_id"],
                            pipeline_stage=i_lat["pipeline_stage"],
                            ts_stage_start=i_lat["ts_stage_start"],
                            ts_stage_end=i_lat["ts_stage_end"],
                        )
                    elif kind == "metrics":
                        from typing import Any, cast

                        i_met = cast(_MetricItem, item)
                        labels_val = i_met.get("labels")
                        labels_cast = cast("dict[str, Any] | str | None", labels_val)
                        self.service.add_metric(
                            metric_name=i_met["metric_name"],
                            metric_type=i_met["metric_type"],
                            value=i_met["value"],
                            timestamp=i_met["timestamp"],
                            labels=labels_cast,
                        )
                    elif kind == "correlation":
                        from typing import cast

                        i_cor = cast(_CorrelationItem, item)
                        self.service.add_correlation(
                            correlation_id=i_cor["correlation_id"],
                            event_id=i_cor["event_id"],
                            parent_event_id=i_cor["parent_event_id"],
                            instrument_id=i_cor["instrument_id"],
                            domain=i_cor["domain"],
                            lineage_depth=i_cor["lineage_depth"],
                            ts_event=i_cor["ts_event"],
                            propagation_path=i_cor["propagation_path"],
                        )
                    else:  # health
                        from typing import cast

                        i_hea = cast(_HealthItem, item)
                        self.service.add_health(
                            component_id=i_hea["component_id"],
                            health_score=i_hea["health_score"],
                            subsystem_scores=i_hea["subsystem_scores"],
                            timestamp=i_hea["timestamp"],
                            measurement_window_ms=i_hea["measurement_window_ms"],
                        )
                    self._queue.task_done()
            except TimeoutError:
                # Normal during idle periods; intentionally ignore
                pass
            except Exception as proc_exc:
                # Swallow and continue to keep background robust — record and log
                try:
                    self._ERRORS.labels(component=self.component_label, kind="process").inc()
                except Exception:
                    self._LOGGER.debug(
                        "Observability error counter emit failed",
                        exc_info=True,
                    )
                self._LOGGER.debug(
                    "Observability async worker encountered an error: %s",
                    proc_exc,
                    exc_info=True,
                )

            # Update queue depth gauge
            self._Q_DEPTH.labels(component=self.component_label).set(self._queue.qsize())

            # Periodic flush
            now = time.time()
            if now - self._last_flush >= float(self.flush_interval_seconds):
                with contextlib.suppress(Exception):
                    if self.sink == "db" and self.use_async_db:
                        await _flush_async_db()
                    else:
                        await asyncio.to_thread(_flush_sync)
