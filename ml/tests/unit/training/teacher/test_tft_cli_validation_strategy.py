"""Tests for TFT CLI validation strategy handling."""

from __future__ import annotations

import pytest

from ml._imports import HAS_PANDAS
from ml._imports import pd
from ml.training.teacher.tft_cli import _resolve_validation_split


def _require_pandas() -> None:
    if not HAS_PANDAS or pd is None:
        pytest.skip("pandas not installed")


def _sample_df() -> pd.DataFrame:
    _require_pandas()
    assert pd is not None
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": list(range(10)),
        },
    )


def test_time_window_split_requires_val_days() -> None:
    _require_pandas()
    df = _sample_df()
    with pytest.raises(ValueError, match="val_days"):
        _resolve_validation_split(
            df,
            validation_strategy="time_window",
            val_days=0,
            timestamp_col="timestamp",
            test_fraction=0.2,
            cv_splits=2,
            purge_gap=0,
            embargo_hours=24.0,
            embargo_pct=None,
        )


def test_time_window_split_returns_partitions() -> None:
    _require_pandas()
    df = _sample_df()
    train_df, val_df = _resolve_validation_split(
        df,
        validation_strategy="time_window",
        val_days=3,
        timestamp_col="timestamp",
        test_fraction=0.2,
        cv_splits=2,
        purge_gap=0,
        embargo_hours=24.0,
        embargo_pct=None,
    )
    assert not train_df.empty
    assert not val_df.empty


def test_purged_split_returns_partitions() -> None:
    _require_pandas()
    df = _sample_df()
    train_df, val_df = _resolve_validation_split(
        df,
        validation_strategy="purged",
        val_days=0,
        timestamp_col="timestamp",
        test_fraction=0.2,
        cv_splits=2,
        purge_gap=0,
        embargo_hours=24.0,
        embargo_pct=0.0,
    )
    assert not train_df.empty
    assert not val_df.empty
