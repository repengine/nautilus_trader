"""Unit tests for the ML async persistence worker."""

from __future__ import annotations

import asyncio
import time

import pytest

from ml.observability.ml_async_persistence import MLPersistenceWorker

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


class DummyFeatureStore:
    """Feature store stub for persistence worker tests."""

    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []

    def write_features(
        self,
        *,
        feature_set_id: str,
        instrument_id: str,
        features: dict[str, float],
        ts_event: int,
        ts_init: int,
    ) -> None:
        """Record a feature write for verification."""
        self.writes.append(
            {
                "feature_set_id": feature_set_id,
                "instrument_id": instrument_id,
                "features": features,
                "ts_event": ts_event,
                "ts_init": ts_init,
            },
        )


class DummyModelStore:
    """Model store stub for persistence worker tests."""

    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []

    def write_prediction(
        self,
        *,
        model_id: str,
        instrument_id: str,
        prediction: float,
        confidence: float,
        features: dict[str, float],
        inference_time_ms: float,
        ts_event: int,
    ) -> None:
        """Record a prediction write for verification."""
        self.writes.append(
            {
                "model_id": model_id,
                "instrument_id": instrument_id,
                "prediction": prediction,
                "confidence": confidence,
                "features": features,
                "inference_time_ms": inference_time_ms,
                "ts_event": ts_event,
            },
        )


def test_worker_stop_drains_queue_and_joins_thread() -> None:
    """Ensure the worker stops cleanly and the background thread exits."""
    feature_store = DummyFeatureStore()
    model_store = DummyModelStore()
    worker = MLPersistenceWorker(
        feature_store=feature_store,
        model_store=model_store,
        queue_maxsize=10,
        flush_interval_seconds=0.01,
        batch_size=5,
    )

    worker.start()
    loop_thread = worker._loop_thread
    assert loop_thread is not None
    assert loop_thread.is_alive()

    assert worker.enqueue_features(
        feature_set_id="default",
        instrument_id="EUR/USD.SIM",
        features={"feature_0": 1.0},
        ts_event=1,
        ts_init=1,
    )
    assert worker.enqueue_prediction(
        model_id="model-1",
        instrument_id="EUR/USD.SIM",
        prediction=0.5,
        confidence=0.9,
        features={"feature_0": 1.0},
        inference_time_ms=0.1,
        ts_event=2,
    )

    time.sleep(0.05)
    asyncio.run(worker.stop(drain=True, timeout=2.0))

    assert worker.queue_size() == 0
    assert loop_thread is not None
    assert not loop_thread.is_alive()


def test_enqueue_returns_false_when_queue_full() -> None:
    """Ensure enqueue returns False when queue capacity is exceeded."""
    feature_store = DummyFeatureStore()
    model_store = DummyModelStore()
    worker = MLPersistenceWorker(
        feature_store=feature_store,
        model_store=model_store,
        queue_maxsize=1,
        flush_interval_seconds=1.0,
        batch_size=1,
    )

    assert worker.enqueue_features(
        feature_set_id="default",
        instrument_id="EUR/USD.SIM",
        features={"feature_0": 1.0},
        ts_event=1,
        ts_init=1,
    )

    assert not worker.enqueue_prediction(
        model_id="model-1",
        instrument_id="EUR/USD.SIM",
        prediction=0.5,
        confidence=0.9,
        features={"feature_0": 1.0},
        inference_time_ms=0.1,
        ts_event=2,
    )
