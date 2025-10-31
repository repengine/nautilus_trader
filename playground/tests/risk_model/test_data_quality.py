"""
Tests for Phase 4 missing data audit utilities.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from playground.risk_model.data_quality import audit_missing_data


@pytest.fixture()
def dataset_with_gaps(tmp_path: Path) -> Path:
    """Create a small dataset with intentional missing values."""
    timestamps = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC),
    ]
    data = pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["XLK", "XLK", "XLK", "XLK"],
        "return": [0.01, None, 0.02, None],
        "volume": [1_000, 1_050, None, 1_020],
    })
    dataset_path = tmp_path / "gaps.parquet"
    data.write_parquet(dataset_path)
    return dataset_path


def test_audit_missing_data_reports_ratios(dataset_with_gaps: Path) -> None:
    """Audit should record missing ratio and imputation summaries."""
    result = audit_missing_data(dataset_path=dataset_with_gaps)
    assert result.missing_ratio > 0.0
    assert "return" in result.missing_by_column
    summaries = {summary.method: summary for summary in result.imputation_summaries}
    forward_fill = summaries["forward_fill"]
    assert forward_fill.filled_ratio > 0.0
    assert forward_fill.remaining_missing_ratio < result.missing_ratio
    assert "kalman" in summaries
    assert summaries["kalman"].note is not None


def test_audit_missing_data_supports_method_filter(dataset_with_gaps: Path) -> None:
    """Custom method filters should narrow evaluation set."""
    result = audit_missing_data(
        dataset_path=dataset_with_gaps,
        methods=("linear",),
    )
    assert [summary.method for summary in result.imputation_summaries] == ["linear"]
