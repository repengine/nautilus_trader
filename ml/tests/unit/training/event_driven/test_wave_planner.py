from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ml.training.event_driven.wave_planner import recommend_next_wave, WaveBounds, WaveSample


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
