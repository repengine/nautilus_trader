"""
Unit tests for Chronos soft label generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml._imports import HAS_POLARS
from ml.config.autogluon import AutoGluonDataConfig
from ml.config.autogluon import ChronosDistillationConfig
from ml.config.autogluon import ChronosTrainingConfig


class DummyTimeSeriesFrame:
    """Lightweight stand-in for AutoGluon TimeSeriesDataFrame."""

    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        *,
        id_column: str,
        timestamp_column: str,
    ) -> pd.DataFrame:
        ordered = [id_column, timestamp_column, "target"]
        ordered += [
            col for col in df.columns if col not in {id_column, timestamp_column, "target"}
        ]
        return df[ordered]


class DummyPredictor:
    """Predictor stub that emits deterministic mean forecasts."""

    def __init__(self, prediction_length: int) -> None:
        self._prediction_length = prediction_length

    def make_future_data_frame(self, train_data: Any) -> pd.DataFrame:
        if hasattr(train_data, "reset_index"):
            history = train_data.reset_index()
        else:
            history = pd.DataFrame(train_data)
        if history.empty:
            raise ValueError("History data is empty")
        if "timestamp" not in history.columns or "item_id" not in history.columns:
            raise ValueError("History data missing item_id/timestamp columns")
        last_ts = history["timestamp"].iloc[-1]
        if len(history) > 1:
            freq = history["timestamp"].iloc[-1] - history["timestamp"].iloc[-2]
        else:
            freq = pd.Timedelta(minutes=1)
        future_ts = [last_ts + freq * (step + 1) for step in range(self._prediction_length)]
        return pd.DataFrame(
            {
                "item_id": [str(history["item_id"].iloc[-1])] * self._prediction_length,
                "timestamp": future_ts,
            }
        )

    def predict(self, train_data: Any, *, known_covariates: pd.DataFrame | None = None) -> pd.DataFrame:
        if known_covariates is None:
            raise ValueError("known_covariates required for this dummy predictor")
        output = known_covariates.copy()
        output["mean"] = np.arange(len(output), dtype=float)
        return output


def _build_sample_df() -> pd.DataFrame:
    base_ts = 1704067200_000_000_000
    timestamps = [base_ts + i * 60_000_000_000 for i in range(6)]
    return pd.DataFrame(
        {
            "instrument_id": ["SPY"] * 6,
            "ts_event": timestamps,
            "forward_return": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "hour": [9, 10, 11, 12, 13, 14],
        }
    )


def test_generate_rolling_soft_labels_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generate rolling labels and verify alignment/shape."""
    from ml.training.autogluon import soft_label_generator as slg

    monkeypatch.setattr(slg, "HAS_AUTOGLUON", True)
    monkeypatch.setattr(slg, "TimeSeriesDataFrame", DummyTimeSeriesFrame)

    df = _build_sample_df()
    teacher_config = ChronosTrainingConfig(
        prediction_length=2,
        data_config=AutoGluonDataConfig(
            known_covariates=("hour",),
        ),
    )
    distill_config = ChronosDistillationConfig(
        teacher_config=teacher_config,
        student_config=ChronosTrainingConfig(preset="bolt_small"),
        min_history=1,
        stride=1,
        forecast_step=1,
        max_windows_per_series=None,
        sample_fraction=None,
    )

    labels, stats = slg.generate_rolling_soft_labels(
        df,
        DummyPredictor(teacher_config.prediction_length),
        teacher_config=teacher_config,
        distillation_config=distill_config,
    )

    assert not labels.empty
    assert distill_config.soft_target_column in labels.columns
    assert labels[distill_config.soft_target_column].eq(0.0).all()
    assert stats.generated == len(labels)


def test_build_distillation_dataset_blend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify blended target construction and filtering."""
    from ml.training.autogluon import soft_label_generator as slg

    monkeypatch.setattr(slg, "HAS_AUTOGLUON", True)
    monkeypatch.setattr(slg, "TimeSeriesDataFrame", DummyTimeSeriesFrame)

    df = _build_sample_df()
    teacher_config = ChronosTrainingConfig(
        prediction_length=2,
        data_config=AutoGluonDataConfig(known_covariates=("hour",)),
    )
    distill_config = ChronosDistillationConfig(
        teacher_config=teacher_config,
        student_config=ChronosTrainingConfig(preset="bolt_small"),
        min_history=1,
        stride=1,
        forecast_step=1,
        distillation_alpha=0.5,
        label_strategy="blend",
        max_windows_per_series=None,
        sample_fraction=None,
    )

    distilled = slg.build_distillation_dataset(
        df,
        DummyPredictor(teacher_config.prediction_length),
        teacher_config=teacher_config,
        distillation_config=distill_config,
    )

    data = distilled.data
    assert distill_config.distilled_target_column in data.columns
    assert len(data) == 4
    first_val = data[distill_config.distilled_target_column].iloc[0]
    expected = df["forward_return"].iloc[1] * 0.5
    assert np.isclose(first_val, expected)


def test_generate_rolling_soft_labels_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply temperature scaling to soft labels."""
    from ml.training.autogluon import soft_label_generator as slg

    monkeypatch.setattr(slg, "HAS_AUTOGLUON", True)
    monkeypatch.setattr(slg, "TimeSeriesDataFrame", DummyTimeSeriesFrame)

    df = _build_sample_df()
    teacher_config = ChronosTrainingConfig(
        prediction_length=2,
        data_config=AutoGluonDataConfig(
            known_covariates=("hour",),
        ),
    )
    distill_config = ChronosDistillationConfig(
        teacher_config=teacher_config,
        student_config=ChronosTrainingConfig(preset="bolt_small"),
        min_history=1,
        stride=1,
        forecast_step=2,
        soft_label_temperature=2.0,
        max_windows_per_series=None,
        sample_fraction=None,
    )

    labels, _stats = slg.generate_rolling_soft_labels(
        df,
        DummyPredictor(teacher_config.prediction_length),
        teacher_config=teacher_config,
        distillation_config=distill_config,
    )

    assert np.allclose(labels[distill_config.soft_target_column], 0.5)


def test_generate_rolling_soft_labels_fills_future_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure missing future covariates are synthesized for the forecast horizon."""
    from ml.training.autogluon import soft_label_generator as slg

    monkeypatch.setattr(slg, "HAS_AUTOGLUON", True)
    monkeypatch.setattr(slg, "TimeSeriesDataFrame", DummyTimeSeriesFrame)

    base_ts = 1704067200_000_000_000
    minutes = [0, 1, 2, 3, 5, 6]
    df = pd.DataFrame(
        {
            "instrument_id": ["SPY"] * len(minutes),
            "ts_event": [base_ts + m * 60_000_000_000 for m in minutes],
            "forward_return": np.linspace(0.0, 0.5, len(minutes)),
            "hour": [9] * len(minutes),
            "time_index": list(range(len(minutes))),
        }
    )
    teacher_config = ChronosTrainingConfig(
        prediction_length=3,
        data_config=AutoGluonDataConfig(known_covariates=("hour",)),
    )
    distill_config = ChronosDistillationConfig(
        teacher_config=teacher_config,
        student_config=ChronosTrainingConfig(preset="bolt_small"),
        min_history=2,
        stride=1,
        forecast_step=1,
        max_windows_per_series=None,
        sample_fraction=None,
    )

    labels, stats = slg.generate_rolling_soft_labels(
        df,
        DummyPredictor(teacher_config.prediction_length),
        teacher_config=teacher_config,
        distillation_config=distill_config,
    )

    assert not labels.empty
    assert stats.eligible_candidates > 0


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
def test_merge_soft_labels_polars_timezone() -> None:
    """Ensure timezone-aware timestamps can be merged with soft labels."""
    import polars as pl

    from ml.training.autogluon import soft_label_generator as slg

    timestamps = pd.date_range("2025-01-01", periods=3, freq="min", tz="UTC")
    df = pl.from_pandas(
        pd.DataFrame(
            {
                "instrument_id": ["SPY"] * 3,
                "ts_event": timestamps,
                "forward_return": [0.1, 0.2, 0.3],
            }
        )
    )
    labels = pd.DataFrame(
        {
            "item_id": ["SPY"],
            "timestamp": [timestamps[1].tz_localize(None)],
            "soft_target": [0.5],
        }
    )

    merged = slg._merge_soft_labels(
        df,
        labels,
        data_config=AutoGluonDataConfig(),
        soft_target_column="soft_target",
    )

    assert "soft_target" in merged.columns
    assert merged.filter(pl.col("soft_target").is_not_null()).height == 1
