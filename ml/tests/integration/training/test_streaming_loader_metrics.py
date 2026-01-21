"""Integration tests for streaming loader RSS metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml._imports import HAS_PANDAS
from ml._imports import HAS_TORCH
from ml._imports import check_ml_dependencies
from ml._imports import pd
import ml.training.teacher.streaming_loader as streaming_loader_module
from ml.training.teacher.streaming_loader import (
    TFTStreamingConfig,
    build_streaming_dataloader,
    collect_streaming_metadata,
)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_PANDAS or not HAS_TORCH, reason="pandas/torch dependency required")
def test_streaming_loader_rss_metrics_when_iterating_emits_scan_and_batch_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_prometheus_registry: Any,
) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")
    if not getattr(streaming_loader_module, "HAS_PYARROW", False):
        pytest.skip("pyarrow dependency required")

    monkeypatch.setattr(streaming_loader_module, "current_rss_mb", lambda: 256.0)

    frame = pd.DataFrame(
        {
            "time_index": np.arange(6, dtype=np.int64),
            "instrument_id": ["AAPL"] * 6,
            "feature_a": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
            "y": np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.0], dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "dataset.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = collect_streaming_metadata(
        parquet_path,
        feature_names=("feature_a",),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature_a", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        target_col="y",
        shard_row_budget=10,
    )

    config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=(),
        static_reals=(),
        time_varying_known_reals=("feature_a",),
        time_varying_unknown_reals=(),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=1,
        num_workers=0,
    )

    loader = build_streaming_dataloader(parquet_path, metadata, config)
    next(iter(loader))

    registry = isolated_prometheus_registry.registry
    scan_value = registry.get_sample_value(
        "ml_tft_streaming_rss_mb",
        labels={"stage": "scan"},
    )
    assembly_value = registry.get_sample_value(
        "ml_tft_streaming_rss_mb",
        labels={"stage": "batch_assembly"},
    )

    assert scan_value == pytest.approx(256.0)
    assert assembly_value == pytest.approx(256.0)
