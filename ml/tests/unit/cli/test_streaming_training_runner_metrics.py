from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.cli.streaming_training_runner import REQUIRED_MANIFEST_METRICS
from ml.cli.streaming_training_runner import DatasetSpecification
from ml.cli.streaming_training_runner import FeatureLayout
from ml.cli.streaming_training_runner import PromotionMetricCheck
from ml.cli.streaming_training_runner import _build_manifest_payload
from ml.cli.streaming_training_runner import _normalize_metrics
from ml.cli.streaming_training_runner import _parse_metric_check
from ml.config.events import EventStatus
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import TrainingResultEvent
from ml.training.teacher.streaming_loader import PhaseOneFeatureSignals
from ml.training.teacher.streaming_loader import StreamingLimitSummary
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry


def test_normalize_metrics_backfills_missing_values(tmp_path: Path) -> None:
    logits = np.array([0.1, -0.2, 0.9, 1.5], dtype=np.float64)
    targets = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    artifact = tmp_path / "cohort_logits.npz"
    np.savez_compressed(artifact, z_val=logits, y_val=targets)

    metrics = _normalize_metrics({}, artifact)

    for metric_name in REQUIRED_MANIFEST_METRICS:
        assert metric_name in metrics
        assert isinstance(metrics[metric_name], float)


def test_promotion_metric_check_parsing_and_evaluation() -> None:
    check_min = _parse_metric_check("pr_auc>=0.55")
    assert check_min.metric == "pr_auc"
    assert check_min.comparator == "ge"
    assert check_min.threshold == 0.55
    assert check_min.evaluate({"pr_auc": 0.60})
    assert not check_min.evaluate({"pr_auc": 0.50})

    check_max = _parse_metric_check("calibration_ece_20<=0.05")
    assert check_max.metric == "calibration_ece_20"
    assert check_max.comparator == "le"
    assert check_max.threshold == 0.05
    assert check_max.evaluate({"calibration_ece_20": 0.04})
    assert not check_max.evaluate({"calibration_ece_20": 0.06})

    check_abs = _parse_metric_check("stability_calibration_drift|abs<=0.05")
    assert check_abs.metric == "stability_calibration_drift"
    assert check_abs.comparator == "le"
    assert check_abs.threshold == 0.05
    assert check_abs.absolute is True
    assert check_abs.evaluate({"stability_calibration_drift": -0.04})
    assert not check_abs.evaluate({"stability_calibration_drift": 0.08})


def test_manifest_payload_includes_seeds(tmp_path: Path) -> None:
    feature_layout = FeatureLayout(
        feature_names=("y",),
        numeric_columns=("y",),
        categorical_columns=(),
        feature_schema={"y": "float32"},
        phase_one_signals=PhaseOneFeatureSignals(),
    )
    streaming_config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=(),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
        seed=11,
    )
    spec = DatasetSpecification(
        dataset_id="dataset",
        dataset_dir=tmp_path,
        metadata={"build_ts": "2025-10-30T00:00:00Z", "column_info": {}},
        report={},
        streaming_config=streaming_config,
        feature_layout=feature_layout,
        phase_one_signals=feature_layout.phase_one_signals,
    )
    summary = TFTStreamingSummary(total_shards=1, total_rows=10, max_shard_rows=10)
    loader_stats = StreamingLoaderTelemetry(
        loader="train",
        total_shards=1,
        selected_shards=1,
        skipped_shards=0,
        total_rows=10,
        selected_rows=8,
        skipped_rows=2,
        total_sequences=4,
        selected_sequences=4,
        skipped_sequences=0,
    )
    telemetry = StreamingRunTelemetry(
        metadata_summary=summary,
        caps={
            "dataset_seed": 11,
            "worker_seed": 7,
            "worker_loss_name": "bce",
            "worker_loss_pos_weight": 2.0,
        },
        train=loader_stats,
        validation=loader_stats,
    )
    result = TrainingResultEvent(
        plan_id="plan",
        dataset_id="dataset",
        model_id="model",
        telemetry=telemetry,
        artifact_paths={"logits": "logits.npz"},
        metrics={"roc_auc": 0.5},
        status=EventStatus.SUCCESS,
    )
    metadata = TFTStreamingMetadata(
        shard_indices=(),
        numeric_stats={},
        categorical_vocab={},
        instrument_row_counts={},
        phase_one_signals=feature_layout.phase_one_signals,
    )
    plan = DatasetPlanEvent(
        plan_id="plan",
        dataset_id="dataset",
        parquet_path=tmp_path / "dataset.parquet",
        metadata=metadata,
        metadata_summary=summary,
        limits=StreamingLimitSummary(),
        streaming_config=streaming_config,
        caps={"dataset_seed": 11},
        phase_one_signals=feature_layout.phase_one_signals,
    )
    worker_config = StreamingWorkerConfig(worker_seed=7, dataset_seed=11, loss_pos_weight=2.0)
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    artifact_path = registry_root / "staging" / "artifact.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.touch()
    payload = _build_manifest_payload(
        spec=spec,
        plan=plan,
        result=result,
        worker_config=worker_config,
        registry_path=registry_root,
        artifact_path=artifact_path,
        artifact_digest="abc123",
        state_path=tmp_path / "state.json",
        dataset_shape=(10, 1),
        target_stats={},
        model_id="model",
    )
    training_config = payload["cohort_run"]["training_config"]
    assert training_config["dataset_seed"] == 11
    assert training_config["worker_seed"] == 7
    assert training_config["loss_name"] == "bce"
    assert training_config["loss_pos_weight"] == pytest.approx(2.0)
    telemetry_caps = payload["cohort_run"]["telemetry"]["caps"]
    assert telemetry_caps["dataset_seed"] == 11
    assert telemetry_caps["worker_seed"] == 7
    assert telemetry_caps["worker_loss_name"] == "bce"
    assert telemetry_caps["worker_loss_pos_weight"] == pytest.approx(2.0)
