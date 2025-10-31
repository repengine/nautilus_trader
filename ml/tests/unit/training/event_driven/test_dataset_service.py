from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml._imports import HAS_PANDAS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.training.event_driven.dataset_service import StreamingDatasetPlanner
from ml.training.event_driven.guardrails import DatasetGuardrailError
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
        seed=11,
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
    assert event.streaming_config.seed == 11
    assert event.caps["dataset_seed"] == 11


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
        include_macro_deltas=True,
        include_calendar_lags=True,
        include_clustering_tags=True,
        include_context_features=True,
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
        include_macro_deltas=False,
        include_calendar_lags=False,
        include_clustering_tags=False,
        include_context_features=False,
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
    assert event.streaming_config.include_macro_deltas is True
    assert event.streaming_config.include_calendar_lags is True
    assert event.streaming_config.include_clustering_tags is True
    assert event.streaming_config.include_context_features is True


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


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_streaming_dataset_planner_positive_rate_guardrails(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    dataset_path = tmp_path / "dataset.parquet"
    frame = pd.DataFrame(
        {
            "time_index": np.arange(6, dtype=np.int64),
            "timestamp": np.arange(200, 206, dtype=np.int64),
            "instrument_id": ["SPY"] * 6,
            "feature": np.linspace(0.0, 1.0, num=6, dtype=np.float32),
            "y": np.array([0, 1, 0, 1, 0, 1], dtype=np.float32),
        },
    )
    frame.to_parquet(dataset_path, index=False)

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
        parquet_path=dataset_path,
    )

    service_ok = DatasetServiceConfig(
        parquet_root=str(tmp_path),
        min_positive_rate=0.3,
        max_positive_rate=0.7,
        positive_rate_baseline=0.5,
        positive_rate_drift_tolerance=0.3,
    )
    planner_ok = StreamingDatasetPlanner(service_ok)
    event = planner_ok.plan(request)
    assert event.metadata.numeric_stats["y"].mean == pytest.approx(0.5)

    service_fail = DatasetServiceConfig(
        parquet_root=str(tmp_path),
        min_positive_rate=0.6,
    )
    planner_fail = StreamingDatasetPlanner(service_fail)
    with pytest.raises(DatasetGuardrailError):
        planner_fail.plan(request)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_streaming_dataset_planner_schema_guardrails(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    dataset_path = tmp_path / "dataset.parquet"
    frame = pd.DataFrame(
        {
            "time_index": np.arange(5, dtype=np.int64),
            "instrument_id": ["QQQ"] * 5,
            "feature": np.linspace(0.0, 1.0, num=5, dtype=np.float32),
            "y": np.linspace(0.0, 1.0, num=5, dtype=np.float32),
        },
    )
    frame.to_parquet(dataset_path, index=False)

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
        parquet_path=dataset_path,
    )

    service_config = DatasetServiceConfig(
        parquet_root=str(tmp_path),
        schema_reference_columns=("feature", "y", "missing_column"),
    )
    planner = StreamingDatasetPlanner(service_config)
    with pytest.raises(DatasetGuardrailError):
        planner.plan(request)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas dependency required")
def test_streaming_dataset_planner_known_future_guardrails(tmp_path: Path) -> None:
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")
    pytest.importorskip("pyarrow.dataset")

    dataset_path = tmp_path / "dataset.parquet"
    timestamp = np.arange(300, 308, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "time_index": np.arange(8, dtype=np.int64),
            "timestamp": timestamp,
            "macro_effective_ns": timestamp - 5,
            "instrument_id": ["IWM"] * 8,
            "feature": np.linspace(0.0, 1.0, num=8, dtype=np.float32),
            "y": np.array([0, 1, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        },
    )
    frame.to_parquet(dataset_path, index=False)

    streaming_config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=("timestamp",),
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
        feature_names=("timestamp", "feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("timestamp", "feature"),
        parquet_path=dataset_path,
    )

    service_config = DatasetServiceConfig(
        parquet_root=str(tmp_path),
        known_future_pairs=("timestamp:macro_effective_ns",),
        known_future_sample_rows=16,
    )
    planner = StreamingDatasetPlanner(service_config)
    planner.plan(request)

    frame_invalid = frame.copy()
    frame_invalid["macro_effective_ns"] = frame_invalid["timestamp"] + 10
    invalid_path = tmp_path / "invalid.parquet"
    frame_invalid.to_parquet(invalid_path, index=False)
    request_invalid = DatasetPlanRequest(
        dataset_id="invalid.parquet",
        streaming_config=streaming_config,
        feature_names=("timestamp", "feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("timestamp", "feature"),
        parquet_path=invalid_path,
    )
    planner_invalid = StreamingDatasetPlanner(service_config)
    with pytest.raises(DatasetGuardrailError):
        planner_invalid.plan(request_invalid)
