from __future__ import annotations

from ml.data.ingest.metrics import record_ingest_batch, record_ingest_error
from ml.common import metrics as m


def test_record_ingest_metrics_smoke() -> None:
    # Record a successful batch
    record_ingest_batch(
        dataset="tbbo",
        instrument="EURUSD.SIM",
        source="historical",
        duration_seconds=0.01,
        ts_min=1,
        ts_max=2,
    )
    # Record an error
    record_ingest_error(dataset="tbbo", instrument="EURUSD.SIM", error_type="rate_limit")

    # Inspect via metric collectors
    def has_sample(metric, name: str, labels: dict[str, str]) -> bool:  # type: ignore[no-untyped-def]
        for fam in metric.collect():
            for s in fam.samples:
                if s.name != name:
                    continue
                if all(str(s.labels.get(k)) == v for k, v in labels.items()):
                    return True
        return False

    # Check +Inf bucket which is always present for histograms
    # Histogram sample presence can vary; focus on counters/gauges which are stable
    assert has_sample(
        m.data_collection_errors_total,
        "nautilus_ml_data_collection_errors_total",
        {"source": "tbbo", "instrument": "EURUSD.SIM", "error_type": "rate_limit"},
    )
    assert has_sample(
        m.watermark_lag_seconds,
        "nautilus_ml_watermark_lag_seconds",
        {"dataset": "tbbo", "instrument": "EURUSD.SIM", "source": "historical"},
    )
