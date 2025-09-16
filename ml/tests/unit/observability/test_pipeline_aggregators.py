from __future__ import annotations

from ml.observability.pipeline import aggregate_metrics_by_window
from ml.observability.pipeline import scale_health_scores


def test_aggregate_metrics_by_window_preserves_totals() -> None:
    rows = [
        {
            "metric_name": "ml_predictions_total",
            "domain": "models",
            "instrument_id": "EURUSD",
            "value": 1.0,
            "timestamp": 1000,
        },
        {
            "metric_name": "ml_predictions_total",
            "domain": "models",
            "instrument_id": "EURUSD",
            "value": 2.0,
            "timestamp": 1001,
        },
    ]
    out = aggregate_metrics_by_window(rows, window_ns=100)
    assert out["total_value"].sum() == 3.0


def test_scale_health_scores_clips_and_scales() -> None:
    rows = [
        {"component_id": "data_store", "health_score": 0.9},
        {"component_id": "model_store", "health_score": 0.6},
    ]
    out = scale_health_scores(rows, factor=1.2)
    from typing import Any, cast

    assert float(cast(Any, out.loc[0, "health_score"])) <= 1.0
    assert float(cast(Any, out.loc[1, "health_score"])) > 0.6
