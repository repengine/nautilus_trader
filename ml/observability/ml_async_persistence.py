"""
ML async persistence worker (off hot-path).

Provides non-blocking feature and prediction persistence for ML actors, using the same
async worker pattern as ObservabilityAsyncWorker. Integrates with FeatureStore and
ModelStore to batch writes and avoid blocking the inference hot path.

Notes
-----
- Hot-path actors call non-blocking ``enqueue_*`` methods
- Background asyncio task batches and flushes to stores periodically
- Queue backpressure is tracked via metrics; full queue drops writes gracefully
- Supports both direct store calls and async DB persistence

Example
-------
>>> from ml.observability.ml_async_persistence import MLPersistenceWorker
>>> worker = MLPersistenceWorker(
...     feature_store=feature_store,
...     model_store=model_store,
...     flush_interval_seconds=1.0,
... )
>>> worker.start()
>>> # In hot path:
>>> worker.enqueue_features(feature_set_id="default", instrument_id="EUR/USD.SIM", ...)
>>> # Clean shutdown:
>>> await worker.stop(drain=True)

"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from dataclasses import field
from typing import Literal, TypedDict

from ml.common.metrics_manager import MetricsManager
from ml.stores.protocols import FeatureStoreStrictProtocol
from ml.stores.protocols import ModelStoreStrictProtocol


class _FeatureItem(TypedDict):
    """Feature data item for async persistence queue."""

    kind: Literal["feature"]
    feature_set_id: str
    instrument_id: str
    features: dict[str, float]
    ts_event: int
    ts_init: int


class _PredictionItem(TypedDict):
    """Prediction data item for async persistence queue."""

    kind: Literal["prediction"]
    model_id: str
    instrument_id: str
    prediction: float
    confidence: float
    features: dict[str, float]
    inference_time_ms: float
    ts_event: int


MLPersistenceItem = _FeatureItem | _PredictionItem


@dataclass(slots=True)
class MLPersistenceWorker:
    """
    Async worker for ML feature/prediction persistence.

    Provides non-blocking enqueue methods for hot-path actors and batched background
    writes to FeatureStore and ModelStore. Tracks queue depth and dropped writes.

    Parameters
    ----------
    feature_store : object
        Feature store instance (FeatureStore or DummyStore).
    model_store : object
        Model store instance (ModelStore or DummyStore).
    queue_maxsize : int
        Bounded queue capacity; enqueue drops when full (default: 10000).
    flush_interval_seconds : float
        Periodic flush interval in seconds (default: 1.0).
    batch_size : int
        Max items to process per flush cycle (default: 100).
    component_label : str
        Component label for metrics (default: "ml_persistence_worker").

    """

    feature_store: FeatureStoreStrictProtocol
    model_store: ModelStoreStrictProtocol
    queue_maxsize: int = 10000
    flush_interval_seconds: float = 1.0
    batch_size: int = 100
    component_label: str = "ml_persistence_worker"

    _queue: asyncio.Queue[MLPersistenceItem] = field(init=False)
    _task: asyncio.Task[None] | Future[None] | None = field(default=None, init=False)
    _stop: asyncio.Event = field(init=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False)
    _loop_thread: threading.Thread | None = field(default=None, init=False)
    _last_flush: float = field(default=0.0, init=False)

    # Metrics via MetricsManager
    _MM = MetricsManager.default()
    _ENQUEUED = _MM.counter(
        "nautilus_ml_persistence_enqueued_total",
        "Total ML persistence items enqueued",
        ["kind"],
    )
    _FLUSH_SEC = _MM.histogram(
        "nautilus_ml_persistence_flush_duration_seconds",
        "ML persistence flush duration",
        ["store_type"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    )
    _Q_DEPTH = _MM.gauge(
        "nautilus_ml_persistence_queue_depth",
        "Current depth of ML persistence queue",
        ["component"],
    )
    _ERRORS = _MM.counter(
        "nautilus_ml_persistence_errors_total",
        "Total errors in ML persistence worker",
        ["component", "kind"],
    )
    _DROPS = _MM.counter(
        "nautilus_ml_persistence_drops_total",
        "Total ML persistence items dropped due to backpressure",
        ["kind"],
    )

    _LOGGER = logging.getLogger(__name__)

    def __post_init__(self) -> None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._loop = loop
        self._queue = asyncio.Queue(maxsize=int(self.queue_maxsize))
        self._stop = asyncio.Event()

    # ------------------------------ API ----------------------------------

    def start(self) -> None:
        """
        Start background worker task (idempotent).
        """
        if self._task is not None and not self._task_done():
            return

        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            self._loop = loop

        assert loop is not None

        self._stop.clear()

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            if not loop.is_running():
                def _loop_runner() -> None:
                    assert loop is not None
                    asyncio.set_event_loop(loop)
                    loop.run_forever()

                thread = threading.Thread(
                    target=_loop_runner,
                    name="MLPersistenceWorkerLoop",
                    daemon=True,
                )
                thread.start()
                self._loop_thread = thread

            future = asyncio.run_coroutine_threadsafe(self._run(), loop)
            self._task = future
        else:
            if running_loop is not loop:
                self._loop = running_loop
                loop = running_loop
            self._task = loop.create_task(self._run(), name="MLPersistenceWorker")

    async def stop(self, *, drain: bool = True, timeout: float | None = 5.0) -> None:
        """
        Stop worker; optionally drain queue before exit.

        Parameters
        ----------
        drain : bool
            If True, process remaining queue items before stopping (default: True).
        timeout : float | None
            Maximum time to wait for drain/shutdown in seconds (default: 5.0).

        """
        if drain:
            # Best-effort drain with time-bound
            start = time.perf_counter()
            while not self._queue.empty() and (
                timeout is None or time.perf_counter() - start < timeout
            ):
                await asyncio.sleep(0.01)
        self._stop.set()
        task = self._task
        loop = self._loop
        if task is not None:
            try:
                if isinstance(task, asyncio.Task):
                    await asyncio.wait_for(task, timeout=timeout)
                else:
                    if timeout is None:
                        task.result()
                    else:
                        task.result(timeout=timeout)
            except Exception:
                if isinstance(task, asyncio.Task):
                    task.cancel()
                    with contextlib.suppress(Exception):
                        await task
                else:
                    task.cancel()
            finally:
                self._task = None

        if self._loop_thread is not None and loop is not None:
            loop.call_soon_threadsafe(loop.stop)
            self._loop_thread.join(timeout=timeout)
            self._loop_thread = None
            loop.close()
            self._loop = None

    def queue_size(self) -> int:
        """
        Return current queue size.

        This is a cheap call and safe for status polling.

        """
        return int(self._queue.qsize())

    def _task_done(self) -> bool:
        task = self._task
        if task is None:
            return True
        if isinstance(task, asyncio.Task):
            return task.done()
        return task.done()

    # --------------------------- Enqueue API -----------------------------

    def enqueue_features(
        self,
        *,
        feature_set_id: str,
        instrument_id: str,
        features: dict[str, float],
        ts_event: int,
        ts_init: int,
    ) -> bool:
        """
        Enqueue feature data for async persistence.

        Non-blocking call suitable for hot-path usage. Returns False if queue is full.

        Parameters
        ----------
        feature_set_id : str
            Feature set identifier.
        instrument_id : str
            Instrument identifier.
        features : dict[str, float]
            Feature name-value pairs.
        ts_event : int
            Event timestamp (nanoseconds).
        ts_init : int
            Initialization timestamp (nanoseconds).

        Returns
        -------
        bool
            True if enqueued successfully, False if queue full (dropped).

        """
        return self._try_put(
            _FeatureItem(
                kind="feature",
                feature_set_id=feature_set_id,
                instrument_id=instrument_id,
                features=features,
                ts_event=int(ts_event),
                ts_init=int(ts_init),
            ),
        )

    def enqueue_prediction(
        self,
        *,
        model_id: str,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        inference_time_ms: float,
        ts_event: int,
    ) -> bool:
        """
        Enqueue prediction data for async persistence.

        Non-blocking call suitable for hot-path usage. Returns False if queue is full.

        Parameters
        ----------
        model_id : str
            Model identifier.
        instrument_id : str
            Instrument identifier.
        prediction : float
            Model prediction value.
        confidence : float
            Model confidence score.
        features : dict[str, float]
            Feature name-value pairs used for prediction.
        inference_time_ms : float
            Inference latency in milliseconds.
        ts_event : int
            Event timestamp (nanoseconds).

        Returns
        -------
        bool
            True if enqueued successfully, False if queue full (dropped).

        """
        return self._try_put(
            _PredictionItem(
                kind="prediction",
                model_id=model_id,
                instrument_id=instrument_id,
                prediction=float(prediction),
                confidence=float(confidence),
                features=features,
                inference_time_ms=float(inference_time_ms),
                ts_event=int(ts_event),
            ),
        )

    # --------------------------- Internals --------------------------------

    def _try_put(self, item: MLPersistenceItem) -> bool:
        try:
            self._queue.put_nowait(item)
            self._ENQUEUED.labels(kind=item["kind"]).inc()
            # Update queue depth gauge cheaply
            self._Q_DEPTH.labels(component=self.component_label).set(self._queue.qsize())
            return True
        except asyncio.QueueFull:
            # Record backpressure drop
            self._DROPS.labels(kind=item["kind"]).inc()
            self._LOGGER.debug(
                "ML persistence queue full - dropped %s item",
                item["kind"],
            )
            return False

    async def _run(self) -> None:
        """Main worker loop: drain queue and flush periodically."""
        feature_batch: list[_FeatureItem] = []
        prediction_batch: list[_PredictionItem] = []

        while not self._stop.is_set():
            try:
                # Pull at most batch_size items to avoid starving flush
                for _ in range(self.batch_size):
                    item = await asyncio.wait_for(self._queue.get(), timeout=0.05)
                    if item["kind"] == "feature":
                        feature_batch.append(item)
                    else:
                        prediction_batch.append(item)
                    self._queue.task_done()
            except TimeoutError:
                # Normal during idle periods; intentionally ignore
                pass
            except Exception as proc_exc:
                # Swallow and continue to keep background robust
                self._ERRORS.labels(component=self.component_label, kind="process").inc()
                self._LOGGER.debug(
                    "ML persistence worker encountered an error: %s",
                    proc_exc,
                    exc_info=True,
                )

            # Update queue depth gauge
            self._Q_DEPTH.labels(component=self.component_label).set(self._queue.qsize())

            # Periodic flush
            now = time.time()
            if now - self._last_flush >= float(self.flush_interval_seconds):
                if feature_batch:
                    await self._flush_features(feature_batch)
                    feature_batch.clear()
                if prediction_batch:
                    await self._flush_predictions(prediction_batch)
                    prediction_batch.clear()
                self._last_flush = now

    async def _flush_features(self, batch: list[_FeatureItem]) -> None:
        """Flush feature batch to store (off event loop via thread)."""
        start = time.perf_counter()
        try:
            await asyncio.to_thread(self._flush_features_sync, batch)
            dur = time.perf_counter() - start
            self._FLUSH_SEC.labels(store_type="feature").observe(dur)
        except Exception as exc:
            self._ERRORS.labels(component=self.component_label, kind="flush_features").inc()
            self._LOGGER.warning("Feature batch flush failed: %s", exc, exc_info=True)

    def _flush_features_sync(self, batch: list[_FeatureItem]) -> None:
        """Synchronous batch write to feature store."""
        for item in batch:
            try:
                self.feature_store.write_features(
                    feature_set_id=item["feature_set_id"],
                    instrument_id=item["instrument_id"],
                    features=item["features"],
                    ts_event=item["ts_event"],
                    ts_init=item["ts_init"],
                )
            except Exception as exc:
                # Log individual write failures but continue processing batch
                self._LOGGER.debug(
                    "Feature write failed for %s: %s",
                    item["instrument_id"],
                    exc,
                    exc_info=True,
                )

    async def _flush_predictions(self, batch: list[_PredictionItem]) -> None:
        """Flush prediction batch to store (off event loop via thread)."""
        start = time.perf_counter()
        try:
            await asyncio.to_thread(self._flush_predictions_sync, batch)
            dur = time.perf_counter() - start
            self._FLUSH_SEC.labels(store_type="prediction").observe(dur)
        except Exception as exc:
            self._ERRORS.labels(
                component=self.component_label,
                kind="flush_predictions",
            ).inc()
            self._LOGGER.warning("Prediction batch flush failed: %s", exc, exc_info=True)

    def _flush_predictions_sync(self, batch: list[_PredictionItem]) -> None:
        """Synchronous batch write to model store."""
        for item in batch:
            try:
                self.model_store.write_prediction(
                    model_id=item["model_id"],
                    instrument_id=item["instrument_id"],
                    prediction=item["prediction"],
                    confidence=item["confidence"],
                    features=item["features"],
                    inference_time_ms=item["inference_time_ms"],
                    ts_event=item["ts_event"],
                )
            except Exception as exc:
                # Log individual write failures but continue processing batch
                self._LOGGER.debug(
                    "Prediction write failed for %s: %s",
                    item["instrument_id"],
                    exc,
                    exc_info=True,
                )


__all__ = ["MLPersistenceWorker"]
