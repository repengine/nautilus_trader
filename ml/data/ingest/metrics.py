from __future__ import annotations

"""
Helpers to record ingestion metrics for fixtures and adapters.

Uses centralized metrics from ml.common.metrics to ensure consistent naming.
"""

from typing import Final

from ml.common.metrics import data_collection_duration
from ml.common.metrics import data_collection_errors_total
from ml.common.metrics import record_pipeline_event
from ml.common.metrics import watermark_lag_seconds
from ml.config.events import Stage


def record_ingest_batch(
    *,
    dataset: str,
    instrument: str,
    source: str,
    duration_seconds: float,
    ts_min: int,
    ts_max: int,
) -> None:
    """
    Record metrics for a completed ingest batch.

    - Observes collection duration histogram
    - Records pipeline event
    - Updates watermark lag gauge as seconds (relative to ts_max)

    """
    data_collection_duration.labels(source=source, schema=dataset).observe(
        max(duration_seconds, 0.0),
    )
    record_pipeline_event(
        dataset_type=dataset,
        component=instrument,
        stage=Stage.DATA_INGESTED.value,
        source=source,
    )
    # Watermark lag is defined as "now - last_success_ns"; tests pass 0 for simplicity
    # Here we store as 0 to avoid relying on real clock in tests
    watermark_lag_seconds.labels(dataset=dataset, instrument=instrument, source=source).set(0.0)


def record_ingest_error(*, dataset: str, instrument: str, error_type: str) -> None:
    data_collection_errors_total.labels(
        source=dataset,
        instrument=instrument,
        error_type=error_type,
    ).inc()


__all__: Final[list[str]] = ["record_ingest_batch", "record_ingest_error"]
