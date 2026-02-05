"""
Unit tests for Chronos evaluation utilities.

These tests validate time-based splits and baseline metric calculations.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.config.autogluon import ChronosEvaluationConfig
from ml.training.autogluon.chronos_evaluation import evaluate_baseline
from ml.training.autogluon.chronos_evaluation import run_chronos_time_split_evaluation
from ml.training.autogluon.chronos_evaluation import sanitize_chronos_frame
from ml.training.autogluon.chronos_evaluation import split_time_series_frame


if TYPE_CHECKING:
    import polars as pl


def _require_polars() -> None:
    """
    Skip tests when Polars is unavailable.
    """
    if not HAS_POLARS or pl is None:
        pytest.skip("Polars not installed")


def _sample_frame() -> pl.DataFrame:
    """
    Build a deterministic dataset for split testing.
    """
    if pl is None:
        raise RuntimeError("Polars not available for test setup")
    return pl.DataFrame(
        {
            "instrument_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "ts_event": [1, 2, 3, 4, 1, 2, 3, 4],
            "forward_return": [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0],
        },
    )


def test_chronos_evaluation_config_invalid_fractions() -> None:
    """
    Ensure invalid split fractions raise a ValueError.
    """
    with pytest.raises(ValueError, match=r"must equal 1\.0"):
        ChronosEvaluationConfig(
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.1,
        )


def test_split_time_series_frame_boundaries() -> None:
    """
    Verify time-based splits respect expected boundaries.
    """
    _require_polars()
    df = _sample_frame()
    config = ChronosEvaluationConfig(
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        min_rows_per_split=1,
    )
    split = split_time_series_frame(df, config)

    assert split.boundaries.train_end_ts == 2
    assert split.boundaries.val_end_ts == 3
    assert split.row_counts == {"train": 4, "val": 2, "test": 2}


def test_split_time_series_frame_purged_strategy() -> None:
    """
    Verify purged validation splits respect timestamp boundaries.
    """
    _require_polars()
    df = _sample_frame()
    config = ChronosEvaluationConfig(
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        min_rows_per_split=1,
        validation_strategy="purged",
        cv_splits=2,
        purge_gap=0,
        embargo_pct=0.0,
    )
    split = split_time_series_frame(df, config)

    assert split.row_counts == {"train": 2, "val": 4, "test": 2}


def test_chronos_evaluation_config_invalid_validation_strategy() -> None:
    """
    Invalid validation strategies should raise.
    """
    with pytest.raises(ValueError, match="validation_strategy"):
        ChronosEvaluationConfig(validation_strategy="unsupported")


def test_evaluate_baseline_per_item_mean() -> None:
    """
    Baseline metrics should match expected per-item means.
    """
    _require_polars()
    df = _sample_frame()
    config = ChronosEvaluationConfig(
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        min_rows_per_split=1,
    )
    split = split_time_series_frame(df, config)

    metrics = evaluate_baseline(split.train, split.val, config)
    expected_mse = 1.125
    expected_rmse = np.sqrt(expected_mse)
    expected_mae = 0.75

    assert np.isclose(metrics["mse"], expected_mse)
    assert np.isclose(metrics["rmse"], expected_rmse)
    assert np.isclose(metrics["mae"], expected_mae)


def test_sanitize_chronos_frame_drops_non_numeric_and_constant() -> None:
    """
    Sanitization should drop non-numeric and constant columns.
    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars not available for test setup")

    df = pl.DataFrame(
        {
            "instrument_id": ["A", "A", "A"],
            "ts_event": [1, 2, 3],
            "forward_return": [0.1, 0.2, 0.3],
            "feature_one": [1.0, 2.0, 3.0],
            "constant_feature": [1.0, 1.0, 1.0],
            "note": ["x", "y", "z"],
            "GDP__value_vintage_ts": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "is_weekend": [False, False, False],
        },
    )

    config = ChronosEvaluationConfig(min_rows_per_split=1)
    result = sanitize_chronos_frame(df, config)

    assert "note" not in result.frame.columns
    assert "GDP__value_vintage_ts" not in result.frame.columns
    assert "constant_feature" not in result.frame.columns
    assert "feature_one" in result.frame.columns
    assert "is_weekend" not in result.frame.columns
    assert "note" in result.dropped_non_numeric
    assert "GDP__value_vintage_ts" in result.dropped_excluded
    assert "constant_feature" in result.dropped_constant


def test_run_chronos_time_split_evaluation_filters_market_hours() -> None:
    """
    Ensure market hours filtering removes non-regular rows before splitting.
    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars not available for test setup")

    df = pl.DataFrame(
        {
            "instrument_id": ["A", "A", "A", "A", "A", "A"],
            "ts_event": [1, 2, 3, 4, 5, 6],
            "forward_return": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_one": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "is_market_hours": [1, 1, 0, 1, 0, 1],
        },
    )

    config = ChronosEvaluationConfig(
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        min_rows_per_split=1,
    )

    report = run_chronos_time_split_evaluation(df, eval_config=config)

    assert report["filtering"]["enabled"] is True
    assert report["filtering"]["rows_before"] == 6
    assert report["filtering"]["rows_after"] == 4
    assert report["filtering"]["filtered_rows"] == 2
    assert report["split"]["row_counts"] == {"train": 2, "val": 1, "test": 1}


def test_run_chronos_time_split_evaluation_requires_market_hours_column() -> None:
    """
    Ensure filtering raises when the market hours column is missing.
    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars not available for test setup")

    df = pl.DataFrame(
        {
            "instrument_id": ["A", "A", "A"],
            "ts_event": [1, 2, 3],
            "forward_return": [0.1, 0.2, 0.3],
        },
    )

    config = ChronosEvaluationConfig(min_rows_per_split=1)

    with pytest.raises(ValueError, match="market hours column"):
        run_chronos_time_split_evaluation(df, eval_config=config)


def test_run_chronos_time_split_evaluation_filters_series_without_full_split() -> None:
    """
    Series missing split coverage should be removed before evaluation.
    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars not available for test setup")

    df = pl.DataFrame(
        {
            "instrument_id": ["A", "A", "A", "A", "B", "B"],
            "ts_event": [1, 2, 3, 4, 1, 2],
            "forward_return": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        },
    )

    config = ChronosEvaluationConfig(
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
        min_rows_per_split=1,
        min_rows_per_series_split=1,
        filter_market_hours=False,
    )

    report = run_chronos_time_split_evaluation(df, eval_config=config)

    assert report["series_filtering"]["enabled"] is True
    assert report["series_filtering"]["total_series_before"] == 2
    assert report["series_filtering"]["total_series_after"] == 1
    assert report["series_filtering"]["dropped_series"] == ("B",)
    assert report["split"]["row_counts"] == {"train": 2, "val": 1, "test": 1}
