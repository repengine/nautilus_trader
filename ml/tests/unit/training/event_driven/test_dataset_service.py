from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml._imports import HAS_PANDAS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.training.event_driven.dataset_service import StreamingDatasetPlanner
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.teacher.streaming_loader import TFTStreamingConfig


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_streaming_dataset_planner_limits_rows(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")
    dataset_path = tmp_path / "dataset.parquet"
    frame = pd.DataFrame(
        {
            "time_index": np.arange(10, dtype=np.int64),
            "instrument_id": ["AAPL"] * 5 + ["MSFT"] * 5,
            "feature": np.linspace(0.0, 1.0, num=10, dtype=np.float32),
            "y": np.linspace(0.0, 1.0, num=10, dtype=np.float32),
        },
    )
    frame.to_parquet(dataset_path, index=False)

    service_config = DatasetServiceConfig(
        parquet_root=str(tmp_path),
        shard_row_budget=4,
        max_total_rows=6,
    )
    streaming_config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=None,
        num_workers=0,
        max_total_rows=None,
        max_total_sequences=None,
        max_shards=None,
    )
    request = DatasetPlanRequest(
        dataset_id="dataset.parquet",
        streaming_config=streaming_config,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature",),
    )

    planner = StreamingDatasetPlanner(service_config)
    event = planner.plan(request)

    assert event.dataset_id == "dataset.parquet"
    assert event.parquet_path == dataset_path
    assert event.metadata_summary.total_rows <= service_config.max_total_rows
    assert event.caps["max_total_rows"] == service_config.max_total_rows
    assert event.limits.skipped_rows >= 0


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_streaming_dataset_planner_merges_feature_flags(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    dataset_path = tmp_path / "dataset.parquet"
    frame = pd.DataFrame(
        {
            "time_index": np.arange(4, dtype=np.int64),
            "instrument_id": ["SPY"] * 4,
            "feature": np.linspace(0.0, 1.0, num=4, dtype=np.float32),
            "y": np.linspace(0.0, 1.0, num=4, dtype=np.float32),
        },
    )
    frame.to_parquet(dataset_path, index=False)

    service_config = DatasetServiceConfig(
        parquet_root=str(tmp_path),
        include_macro=True,
        include_events=True,
        include_micro=True,
        include_l2=True,
        include_earnings=True,
    )
    streaming_config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=None,
        num_workers=0,
        include_macro=False,
        include_events=False,
        include_micro=False,
        include_l2=False,
        include_earnings=False,
    )
    request = DatasetPlanRequest(
        dataset_id="dataset.parquet",
        streaming_config=streaming_config,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature",),
    )

    planner = StreamingDatasetPlanner(service_config)
    event = planner.plan(request)

    assert event.streaming_config.include_macro is True
    assert event.streaming_config.include_events is True
    assert event.streaming_config.include_micro is True
    assert event.streaming_config.include_l2 is True
    assert event.streaming_config.include_earnings is True
    assert event.streaming_config.include_micro is True


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_streaming_dataset_planner_enables_micro_when_l2_requested(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    dataset_path = tmp_path / "dataset.parquet"
    frame = pd.DataFrame(
        {
            "time_index": np.arange(4, dtype=np.int64),
            "instrument_id": ["SPY"] * 4,
            "feature": np.linspace(0.0, 1.0, num=4, dtype=np.float32),
            "y": np.linspace(0.0, 1.0, num=4, dtype=np.float32),
        },
    )
    frame.to_parquet(dataset_path, index=False)

    service_config = DatasetServiceConfig(
        parquet_root=str(tmp_path),
        include_l2=True,
        include_micro=False,
    )
    streaming_config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=None,
        num_workers=0,
        include_micro=False,
        include_l2=True,
    )
    request = DatasetPlanRequest(
        dataset_id="dataset.parquet",
        streaming_config=streaming_config,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature",),
    )

    planner = StreamingDatasetPlanner(service_config)
    event = planner.plan(request)

    assert event.streaming_config.include_l2 is True
    assert event.streaming_config.include_micro is True


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_streaming_dataset_planner_raises_for_missing_path(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")
    service_config = DatasetServiceConfig(parquet_root=str(tmp_path))
    streaming_config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        drop_last=False,
        shuffle_shards=False,
        seed=None,
        num_workers=0,
    )
    request = DatasetPlanRequest(
        dataset_id="missing.parquet",
        streaming_config=streaming_config,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature",),
    )

    planner = StreamingDatasetPlanner(service_config)
    with pytest.raises(FileNotFoundError):
        planner.plan(request)
