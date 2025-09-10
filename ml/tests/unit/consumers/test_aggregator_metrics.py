from __future__ import annotations

from prometheus_client import CollectorRegistry

from ml.consumers.aggregator import AggregatingConsumer
from ml.consumers.protocols import Envelope
from ml.common import metrics as m


def _has_sample(metric, name: str) -> bool:  # type: ignore[no-untyped-def]
    for fam in metric.collect():
        for s in fam.samples:
            if s.name == name:
                return True
    return False


def test_aggregator_emits_metrics_on_duplicate_and_flush() -> None:
    agg = AggregatingConsumer()
    # Two identical envelopes (duplicate): same id
    e: Envelope = {
        "id": "e1",
        "parent_id": None,
        "instrument_id": "EURUSD.SIM",
        "ts_event": 10,
        "stage": "FEATURE_COMPUTED",
        "correlation_id": "c1",
        "payload": {},
    }
    agg.handle("events.ml.FEATURE_COMPUTED", e)
    # duplicate
    agg.handle("events.ml.FEATURE_COMPUTED", e)
    # watermark flush should produce flushed_total and watermark lag metrics
    flushed = agg.advance_watermark("EURUSD.SIM", watermark_ns=100)
    assert len(flushed) == 1

    # We don't assert specific label values, just presence of metric families
    assert _has_sample(
        m.aggregator_duplicates_total, "nautilus_ml_aggregator_duplicates_total_total"
    ) or _has_sample(
        m.aggregator_duplicates_total,
        "nautilus_ml_aggregator_duplicates_total",
    )
    assert _has_sample(
        m.aggregator_flushed_total, "nautilus_ml_aggregator_flushed_total_total"
    ) or _has_sample(
        m.aggregator_flushed_total,
        "nautilus_ml_aggregator_flushed_total",
    )
    assert _has_sample(m.aggregator_buffer_size, "nautilus_ml_aggregator_buffer_size")
    assert _has_sample(
        m.aggregator_watermark_lag_seconds, "nautilus_ml_aggregator_watermark_lag_seconds"
    )
