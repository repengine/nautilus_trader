from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import ml.training.event_driven.economic_metrics as economic_metrics_module
import ml.training.event_driven.wave_planner as wave_planner_module
from ml.training.event_driven.economic_metrics import compute_economic_and_stability_metrics
from ml.training.event_driven.wave_planner import recommend_next_wave, summarize_samples, WaveBounds, WaveSample


def _sample(time_offset: int, roc: float, gpu_mb: float) -> WaveSample:
    completed_at = datetime.now(tz=UTC) - timedelta(minutes=time_offset)
    return WaveSample(
        completed_at=completed_at,
        roc_auc=roc,
        pr_auc=None,
        max_gpu_memory_mb=gpu_mb,
    )


def test_recommend_next_wave_increments_bounds() -> None:
    samples = [
        _sample(1, 0.66, 1_700.0),
        _sample(2, 0.65, 1_650.0),
        _sample(3, 0.64, 1_600.0),
    ]
    current = WaveBounds(
        shard_row_budget=120_000,
        max_total_rows=120_000,
        max_total_sequences=90_000,
        max_shards=32,
    )
    recommendation = recommend_next_wave(
        samples,
        current,
        row_increment=30_000,
        shard_increment=8,
        device_memory_mb=6_144.0,
        regression_delta=0.02,
    )

    assert recommendation.proposed.max_total_rows == 150_000
    assert recommendation.proposed.max_total_sequences >= 112_500
    assert recommendation.proposed.max_shards == 40
    assert not recommendation.warnings


def test_recommend_next_wave_flags_gpu_pressure() -> None:
    samples = [
        _sample(1, 0.64, 5_600.0),
        _sample(2, 0.63, 5_400.0),
    ]
    current = WaveBounds(
        shard_row_budget=120_000,
        max_total_rows=120_000,
        max_total_sequences=90_000,
        max_shards=32,
    )
    recommendation = recommend_next_wave(
        samples,
        current,
        device_memory_mb=6_144.0,
        gpu_threshold_ratio=0.85,
    )

    assert any("GPU consumption exceeds threshold" in warning for warning in recommendation.warnings)


def test_recommend_next_wave_flags_regression() -> None:
    samples = [
        _sample(1, 0.50, 1_500.0),
        _sample(2, 0.51, 1_480.0),
        _sample(3, 0.67, 1_450.0),
    ]
    current = WaveBounds(
        shard_row_budget=120_000,
        max_total_rows=120_000,
        max_total_sequences=90_000,
        max_shards=32,
    )
    recommendation = recommend_next_wave(
        samples,
        current,
        regression_delta=0.01,
    )

    assert any("Recent ROC-AUC mean" in warning for warning in recommendation.warnings)


def test_recommend_next_wave_validates_inputs() -> None:
    current = WaveBounds(
        shard_row_budget=120_000,
        max_total_rows=120_000,
        max_total_sequences=90_000,
        max_shards=32,
    )
    with pytest.raises(ValueError):
        recommend_next_wave([], current, row_increment=0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"shard_row_budget": 0}, "shard_row_budget must be positive"),
        ({"max_total_rows": 0}, "max_total_rows must be positive"),
        ({"max_total_sequences": 0}, "max_total_sequences must be positive"),
        ({"max_shards": 0}, "max_shards must be positive"),
        ({"max_total_rows": 10}, "max_total_rows must be >= shard_row_budget"),
        ({"max_total_sequences": 150_000}, "max_total_sequences must be <= max_total_rows"),
    ],
)
def test_wave_bounds_reject_invalid_limits(
    overrides: dict[str, int],
    message: str,
) -> None:
    bounds = {
        "shard_row_budget": 120_000,
        "max_total_rows": 120_000,
        "max_total_sequences": 90_000,
        "max_shards": 32,
    }
    with pytest.raises(ValueError, match=message):
        WaveBounds(**(bounds | overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"shard_increment": 0}, "shard_increment must be positive"),
        ({"sequence_ratio_floor": 0.0}, "sequence_ratio_floor must be in \\(0, 1\\]"),
        ({"regression_delta": -0.1}, "regression_delta must be non-negative"),
        ({"device_memory_mb": 0.0}, "device_memory_mb must be positive"),
        ({"gpu_threshold_ratio": 1.1}, "gpu_threshold_ratio must be in \\(0, 1\\]"),
    ],
)
def test_recommend_next_wave_rejects_invalid_optional_arguments(
    overrides: dict[str, float | int],
    message: str,
) -> None:
    current = WaveBounds(
        shard_row_budget=120_000,
        max_total_rows=120_000,
        max_total_sequences=90_000,
        max_shards=32,
    )
    with pytest.raises(ValueError, match=message):
        recommend_next_wave([], current, **overrides)


def test_recommend_next_wave_defaults_for_empty_samples() -> None:
    current = WaveBounds(
        shard_row_budget=120_000,
        max_total_rows=120_000,
        max_total_sequences=90_000,
        max_shards=32,
    )
    recommendation = recommend_next_wave([], current, row_increment=10_000, shard_increment=2)

    assert recommendation.notes == ()
    assert recommendation.warnings == ("No manifests available; defaulting to incremental increase.",)
    assert summarize_samples([]) == (None, None)
    assert wave_planner_module._max_gpu([]) is None
    with pytest.raises(ValueError, match="window must be positive"):
        wave_planner_module._take_recent([], window=0)


def test_compute_economic_and_stability_metrics_handles_empty_inputs() -> None:
    economic, stability, flattened = compute_economic_and_stability_metrics(
        validation_probabilities=np.asarray([], dtype=np.float64),
        validation_labels=np.asarray([], dtype=np.float64),
    )

    assert economic.as_dict() == {}
    assert stability.as_dict() == {}
    assert flattened == {}


def test_compute_economic_and_stability_metrics_aligns_returns_and_handles_bad_calibration() -> None:
    economic, stability, flattened = compute_economic_and_stability_metrics(
        validation_probabilities=np.asarray([0.9, 0.2, 0.8], dtype=np.float64),
        validation_labels=np.asarray([1.0, 0.0, 1.0], dtype=np.float64),
        training_probabilities=np.asarray([0.3, 0.6], dtype=np.float64),
        validation_returns=np.asarray([0.01], dtype=np.float64),
        slippage_bps=5.0,
        calibration_metrics={"calibration_ece_20": "bad"},
        baseline_calibration_ece=0.1,
    )

    assert economic.slippage_adjusted_sharpe is None
    assert economic.hit_rate == pytest.approx(1.0)
    assert economic.turnover is None
    assert stability.ks_statistic is not None
    assert stability.calibration_drift is None
    assert "stability_ks_statistic" in flattened


def test_compute_economic_and_stability_metrics_calibration_drift_uses_baseline() -> None:
    _, stability, flattened = compute_economic_and_stability_metrics(
        validation_probabilities=np.asarray([0.6, 0.4], dtype=np.float64),
        validation_labels=np.asarray([1.0, 0.0], dtype=np.float64),
        training_probabilities=np.asarray([0.5, 0.5], dtype=np.float64),
        calibration_metrics={"calibration_ece_20": 0.25},
        baseline_calibration_ece=0.1,
    )

    assert stability.calibration_drift == pytest.approx(0.15)
    assert flattened["stability_calibration_drift"] == pytest.approx(0.15)


def test_economic_metric_helpers_handle_empty_inputs() -> None:
    assert economic_metrics_module._compute_turnover(np.asarray([1.0], dtype=np.float64)) is None
    assert economic_metrics_module._compute_drawdown(np.asarray([], dtype=np.float64)) is None
    assert (
        economic_metrics_module._two_sample_ks_statistic(
            np.asarray([], dtype=np.float64),
            np.asarray([0.1], dtype=np.float64),
        )
        is None
    )
