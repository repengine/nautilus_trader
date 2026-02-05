"""
Chronos -> ONNX distillation pipeline.

This module trains a Chronos teacher, generates rolling soft labels, aligns them with
feature matrices, and distills a LightGBM student exported to ONNX for production
inference.

"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_PANDAS
from ml._imports import HAS_POLARS
from ml._imports import TimeSeriesPredictor
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml._imports import pl
from ml.common.model_sidecar import extract_inference_metadata
from ml.common.model_sidecar import load_sidecar_metadata
from ml.config.autogluon import ChronosOnnxDistillationConfig
from ml.data.build import _write_feature_npz_from_polars
from ml.ml_types import PolarsDF
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.model_registry_facade import ModelRegistry
from ml.registry.utils import build_feature_schema
from ml.registry.utils import build_student_manifest
from ml.training.autogluon.chronos_trainer import ChronosTrainer
from ml.training.autogluon.soft_label_generator import SoftLabelStats
from ml.training.autogluon.soft_label_generator import build_distillation_dataset
from ml.training.student.lightgbm import LightGBMStudentDistiller
from ml.training.student.lightgbm import build_student_decision_config


if TYPE_CHECKING:
    import pandas as _pd


__all__ = [
    "ChronosOnnxDistillationArtifacts",
    "ChronosOnnxDistillationResult",
    "prepare_chronos_onnx_distillation_artifacts",
    "run_chronos_onnx_distillation",
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChronosOnnxDistillationArtifacts:
    """
    Artifacts emitted by Chronos -> ONNX distillation preparation.

    Attributes
    ----------
    distilled_frame : pl.DataFrame
        Distilled dataset filtered to rows with soft labels.
    feature_names : list[str]
        Feature names aligned with the FeatureRegistry manifest.
    features_npz_path : Path
        Path to the filtered feature matrix NPZ.
    teacher_preds_path : Path
        Path to the teacher predictions NPZ.
    cutoff : int
        Training split cutoff index.
    stats : SoftLabelStats
        Coverage statistics for rolling soft labels.

    """

    distilled_frame: PolarsDF
    feature_names: list[str]
    features_npz_path: Path
    teacher_preds_path: Path
    cutoff: int
    stats: SoftLabelStats


@dataclass(frozen=True)
class ChronosOnnxDistillationResult:
    """
    Result of Chronos -> ONNX distillation pipeline.

    Attributes
    ----------
    teacher_artifacts : ChronosOnnxDistillationArtifacts
        Teacher artifacts including soft labels and feature matrices.
    student_onnx_path : Path
        Path to the exported ONNX student model.
    student_meta_path : Path
        Path to the exported ONNX student metadata sidecar.
    performance_metrics : dict[str, float]
        Optional student performance metrics.

    """

    teacher_artifacts: ChronosOnnxDistillationArtifacts
    student_onnx_path: Path
    student_meta_path: Path
    performance_metrics: dict[str, float]


def _require_polars() -> None:
    if not HAS_POLARS or pl is None:
        check_ml_dependencies(["polars"])
        raise ImportError("Polars not available")


def _ensure_polars_frame(frame: Any) -> PolarsDF:
    _require_polars()
    assert pl is not None
    if isinstance(frame, pl.DataFrame):
        return cast(PolarsDF, frame)
    if HAS_PANDAS and pd is not None and isinstance(frame, pd.DataFrame):
        return cast(PolarsDF, pl.from_pandas(frame))
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _load_feature_manifest(config: ChronosOnnxDistillationConfig) -> FeatureManifest:
    registry = FeatureRegistry(Path(config.feature_registry_dir))
    info = registry.get_feature_set(config.feature_set_id)
    if info is None:
        raise ValueError(f"Unknown feature_set_id: {config.feature_set_id}")
    return info.manifest


def _validate_feature_columns(frame: PolarsDF, feature_names: list[str]) -> None:
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Distillation frame missing feature columns: {missing}")

    non_numeric: list[str] = []
    if pl is None:
        return
    for name in feature_names:
        dtype = frame[name].dtype
        if hasattr(dtype, "is_numeric") and dtype.is_numeric():
            continue
        if dtype == pl.Boolean:
            continue
        non_numeric.append(name)
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")


def _select_sort_column(
    frame: PolarsDF,
    *,
    fallback: str,
) -> str:
    if "time_index" in frame.columns:
        return "time_index"
    if fallback in frame.columns:
        return fallback
    raise ValueError("Distillation frame missing time_index and timestamp column")


def _apply_output_transform(
    values: npt.NDArray[np.float64],
    *,
    transform: str,
) -> npt.NDArray[np.float32]:
    if transform == "identity":
        result = values.astype(np.float32)
    elif transform == "sigmoid":
        result = (1.0 / (1.0 + np.exp(-values))).astype(np.float32)
    else:
        raise ValueError(f"Unsupported output_transform: {transform}")

    if not np.all(np.isfinite(result)):
        raise ValueError("Transformed teacher outputs contain non-finite values")
    return result


def _ensure_registry_artifacts(
    onnx_path: Path,
    meta_path: Path,
    *,
    registry_dir: Path,
    model_id: str,
) -> tuple[Path, Path]:
    registry_root = registry_dir.resolve()
    registry_model_dir = registry_root / model_id
    registry_model_dir.mkdir(parents=True, exist_ok=True)

    resolved_onnx = onnx_path.resolve()
    resolved_meta = meta_path.resolve()
    registry_onnx = registry_model_dir / resolved_onnx.name
    registry_meta = registry_model_dir / resolved_meta.name

    if not str(resolved_onnx).startswith(str(registry_root)):
        shutil.copy2(resolved_onnx, registry_onnx)
    else:
        registry_onnx = resolved_onnx

    if not str(resolved_meta).startswith(str(registry_root)):
        shutil.copy2(resolved_meta, registry_meta)
    else:
        registry_meta = resolved_meta

    return registry_onnx, registry_meta


def _export_soft_labels(
    labels: _pd.DataFrame,
    *,
    config: ChronosOnnxDistillationConfig,
) -> None:
    if not config.distillation_config.export_soft_labels:
        return

    base_dir = Path(config.output_dir)
    soft_labels_path = (
        Path(config.distillation_config.soft_labels_path)
        if config.distillation_config.soft_labels_path
        else base_dir / "soft_labels.parquet"
    )
    soft_labels_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if soft_labels_path.name.endswith(".csv") or soft_labels_path.name.endswith(".csv.gz"):
            labels.to_csv(soft_labels_path, index=False)
        else:
            labels.to_parquet(soft_labels_path, index=False)
        logger.info("Soft labels saved to %s", soft_labels_path)
    except Exception as exc:
        logger.error(
            "Failed to export soft labels to %s: %s",
            soft_labels_path,
            exc,
            exc_info=True,
        )
        raise


def prepare_chronos_onnx_distillation_artifacts(
    df: PolarsDF,
    predictor: TimeSeriesPredictor,
    *,
    config: ChronosOnnxDistillationConfig,
) -> ChronosOnnxDistillationArtifacts:
    """
    Build aligned teacher predictions and feature matrices for distillation.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset containing features and targets.
    predictor : TimeSeriesPredictor
        Trained Chronos predictor used for rolling forecasts.
    config : ChronosOnnxDistillationConfig
        Distillation configuration.

    Returns
    -------
    ChronosOnnxDistillationArtifacts
        Artifacts for student distillation.

    """
    _require_polars()
    assert pl is not None
    distill_cfg = config.distillation_config
    teacher_cfg = distill_cfg.teacher_config

    distilled = build_distillation_dataset(
        df,
        predictor,
        teacher_config=teacher_cfg,
        distillation_config=distill_cfg,
    )
    _export_soft_labels(distilled.labels, config=config)

    if distilled.stats.coverage < distill_cfg.min_soft_label_coverage:
        raise ValueError(
            "Soft label coverage below threshold: "
            f"{distilled.stats.coverage:.3f} < {distill_cfg.min_soft_label_coverage:.3f}",
        )

    frame = _ensure_polars_frame(distilled.data)
    if frame.height == 0:
        raise ValueError("No rows available after distillation filtering")

    label_column = distill_cfg.distilled_target_column
    if label_column not in frame.columns:
        raise ValueError(f"Missing distilled target column: {label_column}")

    data_config = teacher_cfg.get_data_config()
    sort_column = _select_sort_column(frame, fallback=data_config.timestamp_column)
    frame_sorted = frame.sort(sort_column)

    manifest = _load_feature_manifest(config)
    feature_names = list(manifest.feature_names)
    _validate_feature_columns(frame_sorted, feature_names)

    q_raw = frame_sorted.select(pl.col(label_column)).to_numpy().reshape(-1)
    q_values = _apply_output_transform(q_raw.astype(np.float64), transform=config.output_transform)

    total_rows = int(frame_sorted.height)
    cutoff = int(total_rows * float(config.train_fraction))
    if cutoff <= 0 or cutoff >= total_rows:
        raise ValueError(
            "train_fraction yields empty train/val split; " f"rows={total_rows}, cutoff={cutoff}",
        )
    q_train = q_values[:cutoff].astype(np.float32)
    q_val = q_values[cutoff:].astype(np.float32)

    y_val_true: npt.NDArray[np.float32] | None = None
    if config.hard_label_column and config.hard_label_column in frame_sorted.columns:
        y_all = frame_sorted.select(pl.col(config.hard_label_column)).to_numpy().reshape(-1)
        y_val_true = y_all[cutoff:].astype(np.float32)
    elif config.require_hard_labels:
        raise ValueError(
            f"Missing required hard label column: {config.hard_label_column}",
        )

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_npz_path = out_dir / config.filtered_features_filename
    _write_feature_npz_from_polars(
        frame_sorted,
        feature_names,
        out_path=features_npz_path,
        cutoff=cutoff,
    )

    teacher_preds_path = out_dir / config.teacher_preds_filename
    payload: dict[str, npt.NDArray[np.float32]] = {
        "q_train": q_train,
        "q_val": q_val,
    }
    if y_val_true is not None:
        payload["y_val_true"] = y_val_true
    np.savez_compressed(teacher_preds_path, **payload)

    return ChronosOnnxDistillationArtifacts(
        distilled_frame=frame_sorted,
        feature_names=feature_names,
        features_npz_path=features_npz_path,
        teacher_preds_path=teacher_preds_path,
        cutoff=cutoff,
        stats=distilled.stats,
    )


def _train_chronos_teacher(
    df: PolarsDF,
    *,
    config: ChronosOnnxDistillationConfig,
) -> ChronosTrainer:
    trainer = ChronosTrainer(config.distillation_config.teacher_config)
    trainer.train(df)
    if trainer.predictor is None:
        raise RuntimeError("Chronos trainer produced no predictor")
    return trainer


def _train_lightgbm_student(
    artifacts: ChronosOnnxDistillationArtifacts,
    *,
    config: ChronosOnnxDistillationConfig,
    feature_manifest: FeatureManifest,
) -> tuple[Path, Path, dict[str, float]]:
    if list(feature_manifest.feature_names) != artifacts.feature_names:
        raise ValueError(
            "Feature manifest mismatch with distillation artifacts: "
            f"expected={list(feature_manifest.feature_names)}, "
            f"actual={artifacts.feature_names}",
        )
    X_npz = np.load(artifacts.features_npz_path, allow_pickle=True)
    X_train = X_npz["X_train"].astype(np.float32)
    X_val = X_npz["X_val"].astype(np.float32)

    teacher_npz = np.load(artifacts.teacher_preds_path, allow_pickle=True)
    q_train = teacher_npz["q_train"].astype(np.float32) if "q_train" in teacher_npz else None
    q_val = teacher_npz["q_val"].astype(np.float32) if "q_val" in teacher_npz else None
    y_val_true = (
        teacher_npz["y_val_true"].astype(np.float32) if "y_val_true" in teacher_npz else None
    )

    if config.use_val_for_distill:
        if q_val is None:
            raise ValueError("use_val_for_distill set but teacher_preds missing q_val")
        X_train_sel = X_val
        q_train_sel = q_val.reshape(-1)
    else:
        if q_train is None:
            raise ValueError("teacher_preds missing q_train")
        X_train_sel = X_train
        q_train_sel = q_train.reshape(-1)

    if X_train_sel.shape[0] != q_train_sel.shape[0]:
        raise ValueError(
            "Training shape mismatch: " f"X ({X_train_sel.shape[0]}) vs q ({q_train_sel.shape[0]})",
        )

    distiller = LightGBMStudentDistiller(
        objective=config.student_objective,
        kd_lambda=float(config.kd_lambda),
        early_stopping=int(config.early_stopping),
        opset=int(config.opset),
    )
    distiller.fit(X_train_sel, q_train_sel, X_val, y_val_true)

    student_out_dir = Path(config.output_dir) / config.student_output_subdir
    onnx_path, meta_path = distiller.export_onnx(
        feature_names=artifacts.feature_names,
        out_dir=str(student_out_dir),
        model_id=config.model_id,
        flags={
            "distilled_from": "chronos",
            "objective": config.student_objective,
        },
    )

    performance_metrics: dict[str, float] = {}
    if y_val_true is not None:
        try:
            from sklearn.metrics import average_precision_score
            from sklearn.metrics import brier_score_loss
            from sklearn.metrics import log_loss
            from sklearn.metrics import roc_auc_score

            p_val = distiller.predict_proba(X_val).reshape(-1)
            yv = y_val_true.reshape(-1).astype(np.int32)
            p_val = np.clip(p_val, 1e-6, 1.0 - 1e-6)
            performance_metrics = {
                "auc": float(roc_auc_score(yv, p_val)),
                "pr_auc": float(average_precision_score(yv, p_val)),
                "brier": float(brier_score_loss(yv, p_val)),
                "logloss": float(log_loss(yv, p_val)),
            }
        except Exception:
            performance_metrics = {}

    registry_onnx, registry_meta = _ensure_registry_artifacts(
        Path(onnx_path),
        Path(meta_path),
        registry_dir=Path(config.registry_dir),
        model_id=config.model_id,
    )
    registry = ModelRegistry(Path(config.registry_dir))
    dtypes = ["float32"] * len(artifacts.feature_names)
    feature_schema = build_feature_schema(artifacts.feature_names, dtypes)

    decision_cfg = build_student_decision_config()
    sidecar = load_sidecar_metadata(registry_meta)
    output_schema, calibration = (
        extract_inference_metadata(sidecar) if sidecar is not None else (None, None)
    )

    student_manifest = build_student_manifest(
        model_id=config.model_id,
        architecture="LightGBM",
        feature_schema=feature_schema,
        feature_schema_hash=feature_manifest.schema_hash,
        parent_id=config.parent_id,
        performance_metrics=performance_metrics or {"inference_latency_ms": 1.0},
        feature_set_id=feature_manifest.feature_set_id,
        pipeline_signature=feature_manifest.pipeline_signature,
        pipeline_version=feature_manifest.pipeline_version,
        decision_config=decision_cfg,
        output_schema=output_schema,
        calibration=calibration,
    )
    registry.register_model(registry_onnx, student_manifest, auto_deploy=True)

    return registry_onnx, registry_meta, performance_metrics


def run_chronos_onnx_distillation(
    df: PolarsDF,
    *,
    config: ChronosOnnxDistillationConfig,
    predictor: TimeSeriesPredictor | None = None,
) -> ChronosOnnxDistillationResult:
    """
    Train a Chronos teacher and distill a LightGBM ONNX student.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataset containing features and targets.
    config : ChronosOnnxDistillationConfig
        Distillation configuration.
    predictor : TimeSeriesPredictor | None
        Optional pre-trained Chronos predictor; if None, a teacher is trained.

    Returns
    -------
    ChronosOnnxDistillationResult
        Distillation result with student artifacts.

    """
    if predictor is None:
        teacher = _train_chronos_teacher(df, config=config)
        predictor = teacher.predictor
    if predictor is None:
        raise RuntimeError("No predictor available for distillation")

    artifacts = prepare_chronos_onnx_distillation_artifacts(
        df,
        predictor,
        config=config,
    )
    feature_manifest = _load_feature_manifest(config)
    onnx_path, meta_path, metrics = _train_lightgbm_student(
        artifacts,
        config=config,
        feature_manifest=feature_manifest,
    )

    return ChronosOnnxDistillationResult(
        teacher_artifacts=artifacts,
        student_onnx_path=onnx_path,
        student_meta_path=meta_path,
        performance_metrics=metrics,
    )
