"""
Tests for Phase 4 factor outlier detection utilities.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl

from playground.risk_model.outlier_detection import evaluate_factor_outliers


def _sample_factor_frame(with_outlier: bool) -> pl.DataFrame:
    """Build a small factor dataframe for testing."""
    timestamps = [
        datetime(2024, 1, 1 + idx, tzinfo=UTC)
        for idx in range(6)
    ]
    duration = [0.01, 0.012, 0.009, 0.011, 0.010, 0.013]
    credit = [0.005, 0.006, 0.004, 0.005, 0.006, 0.005]
    if with_outlier:
        # Inject a large duration outlier on the final observation.
        duration[-1] = 0.15
    returns = [0.008, 0.009, 0.007, 0.0085, 0.0092, 0.0078]
    frame = pl.DataFrame({
        "timestamp": timestamps,
        "return": returns,
        "factor_duration": duration,
        "factor_credit": credit,
    })
    return frame


def test_evaluate_factor_outliers_detects_anomalies(tmp_path: Path) -> None:
    """Outlier detection should flag >3σ samples and evaluate treatments."""
    frame = _sample_factor_frame(with_outlier=True)
    dataset_path = tmp_path / "factor_returns.parquet"
    frame.write_parquet(dataset_path)

    report = evaluate_factor_outliers(dataset_path=dataset_path, threshold=2.0)

    assert report.total_rows == 6
    assert report.outlier_rows == 1
    assert report.recommended_treatment in {"winsorize", "exclude"}

    summary_map = {summary.factor: summary for summary in report.factor_summaries}
    assert summary_map["factor_duration"].outlier_count == 1
    assert summary_map["factor_credit"].outlier_count == 0

    assert report.treatment_impacts
    for impact in report.treatment_impacts:
        assert impact.treatment in {"winsorize", "exclude"}
        assert impact.retained_rows >= 5

    payload = report.to_dict()
    assert payload["outlier_rows"] == 1
    assert payload["baseline_betas"]


def test_evaluate_factor_outliers_returns_none_when_clean(tmp_path: Path) -> None:
    """Datasets without outliers should emit a 'none' recommendation."""
    frame = _sample_factor_frame(with_outlier=False)
    dataset_path = tmp_path / "factor_returns.parquet"
    frame.write_parquet(dataset_path)

    report = evaluate_factor_outliers(dataset_path=dataset_path, threshold=2.5)

    assert report.outlier_rows == 0
    assert report.recommended_treatment == "none"
    assert not report.treatment_impacts
