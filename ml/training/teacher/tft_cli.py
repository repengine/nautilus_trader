from __future__ import annotations


# ruff: noqa: E402 - allow module docstring before imports in CLI script

"""
CLI to calibrate a TFT teacher and emit soft labels for distillation, with registry integration.

This CLI integrates with:
- Local Feature Registry: resolve feature schema and enforce feature parity
- Local Model Registry: optionally load a teacher ONNX to produce raw logits

Inputs (NPZ conventions)
- If passing precomputed logits: keys {"z_val", "y_val_true"}
- If using a teacher ONNX: keys {"X_val", "y_val_true"}; ensure X_val columns
  are ordered to match the feature manifest (or provide the same order in file)

Outputs
- teacher_preds.npz: contains q_train (calibrated probabilities) and y_val_true
- teacher_meta.json: includes model_id, feature_set_id, teacher_model_id, schema hash
"""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_PANDAS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.config.names import ONNX_INPUT_NAME
from ml.registry.feature_registry import FeatureRegistry
from ml.training.teacher.base import BaseTeacher
from ml.training.teacher.base import TeacherConfig


class CalibratingTeacher(BaseTeacher):
    def fit(self, dataset: object) -> CalibratingTeacher:
        self._is_fitted = True
        return self

    def predict_logits(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return X.astype(np.float64)

    def feature_schema(self) -> dict[str, str]:
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    # Data/outputs
    ap.add_argument(
        "--student_window_npz",
        required=False,
        help="NPZ with either {z_val,y_val_true} or {X_val,y_val_true}",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", required=True)
    # Feature registry integration
    ap.add_argument("--feature_registry_dir", required=True)
    ap.add_argument("--feature_set_id", required=True)
    # Model registry integration (optional ONNX teacher)
    ap.add_argument("--model_registry_dir", required=False)
    ap.add_argument("--teacher_model_id", required=False)
    ap.add_argument(
        "--onnx_output_is_logits",
        action="store_true",
        help="Interpret ONNX model output as logits (else as probabilities)",
    )
    # Optional training mode
    ap.add_argument("--train_data_csv", required=False, help="CSV with training data")
    ap.add_argument("--target_col", required=False, default="y")
    ap.add_argument("--time_index_col", required=False, default="time_index")
    ap.add_argument("--group_id_col", required=False, default="instrument_id")
    ap.add_argument("--max_encoder_length", required=False, type=int, default=30)
    ap.add_argument("--max_prediction_length", required=False, type=int, default=1)
    ap.add_argument("--max_epochs", required=False, type=int, default=1)
    ap.add_argument("--hidden_size", required=False, type=int, default=16)
    ap.add_argument("--lstm_layers", required=False, type=int, default=1)
    ap.add_argument("--attention_head_size", required=False, type=int, default=2)
    ap.add_argument("--dropout", required=False, type=float, default=0.1)
    ap.add_argument(
        "--dataloader_workers",
        required=False,
        type=int,
        default=0,
        help="Number of DataLoader workers for train/val (default: 0)",
    )
    ap.add_argument(
        "--loss",
        required=False,
        choices=["poisson", "bce"],
        default="poisson",
        help="Loss function for TFT teacher (default: poisson)",
    )
    ap.add_argument("--seed", required=False, type=int, default=None)
    ap.add_argument(
        "--static_categoricals",
        required=False,
        help="Comma-separated static categorical column names",
        default=None,
    )
    ap.add_argument(
        "--static_reals",
        required=False,
        help="Comma-separated static real column names",
        default=None,
    )
    ap.add_argument(
        "--known_future_reals",
        required=False,
        help="Comma-separated known-future real column names",
        default=None,
    )
    ap.add_argument(
        "--save_interpretability",
        action="store_true",
        help="Attempt to save TFT interpretability artifacts (feature relevance/attention)",
    )
    ap.add_argument(
        "--export_torchscript",
        action="store_true",
        help="Export a TorchScript artifact (.pt) for the teacher if available",
    )
    ap.add_argument(
        "--export_safetensors",
        action="store_true",
        help="Export teacher weights as .safetensors with sidecar metadata",
    )
    ap.add_argument(
        "--register_teacher",
        action="store_true",
        help="Register the trained teacher as a non-serveable model",
    )
    args = ap.parse_args(argv)

    # Resolve feature manifest and enforce schema
    freg = FeatureRegistry(Path(args.feature_registry_dir))
    finfo = freg.get_feature_set(args.feature_set_id)
    if finfo is None:
        raise SystemExit(f"Unknown feature_set_id: {args.feature_set_id}")
    fman = finfo.manifest
    feature_names = list(fman.feature_names)
    n_features = len(feature_names)

    # Ensure either training CSV or NPZ path provided
    if not args.train_data_csv and not args.student_window_npz:
        raise SystemExit(
            "Provide either --train_data_csv for training or --student_window_npz for calibration",
        )

    # Initialize outputs (populated below depending on mode)
    q_train: npt.NDArray[np.float32] | None = None
    q_val: npt.NDArray[np.float32] | None = None

    # If training data is provided, run training mode
    if args.train_data_csv:
        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])  # pragma: no cover - import guard
        assert pd is not None
        df = pd.read_csv(args.train_data_csv)
        # Set seeds if provided
        if args.seed is not None:
            import random

            import numpy as _np

            try:
                import torch as _torch

                _torch.manual_seed(args.seed)
                _torch.cuda.manual_seed_all(args.seed)
            except Exception:
                import logging as _logging

                _logging.getLogger(__name__).debug(
                    "Torch seeding failed; continuing",
                    exc_info=True,
                )
            random.seed(args.seed)
            _np.random.seed(args.seed)
        # Enforce feature column order
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise SystemExit(f"Training CSV missing required feature columns: {missing}")
        # Use last 20% as validation; keep train/val slices explicit
        df_sorted = df.sort_values(args.time_index_col)
        cutoff = int(len(df_sorted) * 0.8)
        _df_train = df_sorted.iloc[:cutoff]
        df_val = df_sorted.iloc[cutoff:]
        y_val_true = np.asarray(df_val[args.target_col], dtype=np.float64).reshape(-1)

        # Try TFT teacher; if unavailable or fails, fall back to a simple linear model producing logits
        z_val_vec: npt.NDArray[np.float64]
        z_train_vec: npt.NDArray[np.float64]
        used_tft = False
        try:  # pragma: no cover - exercised in integration path when dependencies ok
            from ml.training.teacher.tft_teacher import TFTTeacher
            from ml.training.teacher.tft_teacher import TFTTeacherConfig

            teacher_tft = TFTTeacher(
                TFTTeacherConfig(architecture="TFT", loss_name=str(args.loss)),
                max_encoder_length=args.max_encoder_length,
                max_prediction_length=args.max_prediction_length,
                time_varying_unknown_reals=feature_names,
                static_categoricals=(
                    [s for s in (args.static_categoricals or "").split(",") if s]
                    if args.static_categoricals
                    else None
                ),
                static_reals=(
                    [s for s in (args.static_reals or "").split(",") if s]
                    if args.static_reals
                    else None
                ),
                time_varying_known_reals=(
                    [s for s in (args.known_future_reals or "").split(",") if s]
                    if args.known_future_reals
                    else None
                ),
                time_idx_col=args.time_index_col,
                group_id_col=args.group_id_col,
                target_col=args.target_col,
                max_epochs=args.max_epochs,
                hidden_size=args.hidden_size,
                lstm_layers=args.lstm_layers,
                attention_head_size=args.attention_head_size,
                dropout=args.dropout,
                dataloader_workers=args.dataloader_workers,
            )
            teacher_tft.fit(df)
            z_all = teacher_tft.predict_logits(df_sorted)
            # Split logits into train/val according to cutoff
            z_train_vec = z_all[:cutoff]
            z_val_vec = z_all[cutoff:]
            used_tft = True
        except Exception:
            # Fallback: scikit-learn logistic regression as a simple teacher proxy
            from ml._imports import HAS_SKLEARN

            if not HAS_SKLEARN:
                raise SystemExit("Training requires TFT dependencies or scikit-learn as fallback")
            from sklearn.linear_model import LogisticRegression

            X = np.asarray(df_sorted[feature_names].to_numpy(), dtype=np.float64)
            y = np.asarray(df_sorted[args.target_col].to_numpy(), dtype=int)
            X_train, X_val_arr = X[:cutoff], X[cutoff:]
            y_train = y[:cutoff]
            # Impute NaNs with training column means for logistic regression fallback
            if np.isnan(X_train).any() or np.isnan(X_val_arr).any():
                col_means = np.nanmean(X_train, axis=0)
                # Replace NaNs in training set
                inds_tr = np.where(np.isnan(X_train))
                if inds_tr[0].size > 0:
                    X_train[inds_tr] = np.take(col_means, inds_tr[1])
                # Replace NaNs in validation set using training means
                inds_va = np.where(np.isnan(X_val_arr))
                if inds_va[0].size > 0:
                    X_val_arr[inds_va] = np.take(col_means, inds_va[1])
            lr = LogisticRegression(max_iter=200)
            lr.fit(X_train, y_train)
            # decision_function gives logits for binary classifier
            z_train_vec = lr.decision_function(X_train).astype(np.float64)
            z_val_vec = lr.decision_function(X_val_arr).astype(np.float64)

        # Calibrate and produce calibrated probabilities
        teacher = CalibratingTeacher(TeacherConfig(architecture="TFT"))
        teacher.calibrate(z_val_vec.reshape(-1, 1), y_val_true)
        q_val = teacher.predict_proba(z_val_vec.reshape(-1, 1)).astype(np.float32)
        q_train = teacher.predict_proba(z_train_vec.reshape(-1, 1)).astype(np.float32)

        # Optional interpretability save
        if used_tft and args.save_interpretability:
            try:
                # Build val dataset/loader similar to predict path
                from pytorch_forecasting import TimeSeriesDataSet

                training_ds = getattr(teacher_tft, "_training_dataset", None)
                assert training_ds is not None
                val_ds = TimeSeriesDataSet.from_dataset(
                    training_ds,
                    df_val,
                    predict=True,
                    stop_randomization=True,
                )
                val_loader = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
                # calculate_feature_relevance may not be available in all versions
                tft_model = getattr(teacher_tft, "_tft", None)
                if tft_model is not None and hasattr(tft_model, "calculate_feature_relevance"):
                    relevance = tft_model.calculate_feature_relevance(val_loader)
                    interp_path = Path(args.out_dir) / "interpretability.npz"
                    np.savez_compressed(interp_path, feature_relevance=relevance)
            except Exception:
                import logging as _logging

                _logging.getLogger(__name__).debug(
                    "Interpretability save failed; continuing",
                    exc_info=True,
                )

        # Optionally register teacher as non-serveable with artifact saved under registry
        if args.register_teacher and args.model_registry_dir:
            from ml.registry.base import DataRequirements
            from ml.registry.base import ModelManifest
            from ml.registry.base import ModelRole
            from ml.registry.model_registry import ModelRegistry
            from ml.training.export import save_model_with_metadata

            reg_dir = Path(args.model_registry_dir)
            artifacts_dir = reg_dir / "artifacts" / "teachers"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Choose an artifact format and persist (TorchScript > safetensors > pickle)
            artifact_format = "pkl"
            if used_tft and getattr(args, "export_torchscript", False):
                try:
                    from pytorch_forecasting import TimeSeriesDataSet

                    from ml.training.teacher.tft_torchscript import export_tft_to_torchscript_from_batch

                    training_ds = getattr(teacher_tft, "_training_dataset", None)
                    assert training_ds is not None
                    val_ds = TimeSeriesDataSet.from_dataset(
                        training_ds,
                        df_val,
                        predict=True,
                        stop_randomization=True,
                    )
                    val_loader = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
                    batch = next(iter(val_loader))
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    import torch.nn as nn

                    tft_model = getattr(teacher_tft, "_tft", None)
                    if tft_model is None:
                        raise RuntimeError("No TFT model available for TorchScript export")
                    ts_path = export_tft_to_torchscript_from_batch(
                        cast(nn.Module, tft_model),
                        x,
                        artifacts_dir / args.model_id,
                    )
                    model_path = ts_path
                    artifact_format = "pt"
                except Exception as exc:
                    raise RuntimeError(
                        f"TorchScript export failed and pickle is unsupported: {exc}",
                    )
            elif used_tft and getattr(args, "export_safetensors", False):
                try:
                    import json as _json

                    from safetensors.torch import save_file as _save_safetensors

                    tft = getattr(teacher_tft, "_tft", None)
                    if tft is None:
                        raise RuntimeError("No TFT model available for safetensors export")
                    state = {k: v.detach().cpu() for k, v in tft.state_dict().items()}
                    st_path = (artifacts_dir / args.model_id).with_suffix(".safetensors")
                    _save_safetensors(state, str(st_path))
                    meta = {
                        "feature_names": feature_names,
                        "time_index_col": args.time_index_col,
                        "group_id_col": args.group_id_col,
                        "target_col": args.target_col,
                        "used_tft": used_tft,
                        "format": "safetensors",
                    }
                    with open(
                        st_path.with_suffix(".safetensors.meta.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        _json.dump(meta, f, indent=2)
                    model_path = st_path
                    artifact_format = "safetensors"
                except Exception as exc:
                    raise RuntimeError(
                        f"Safetensors export failed and pickle is unsupported: {exc}",
                    )
            else:
                model_obj = getattr(teacher_tft, "_tft", None) if used_tft else lr
                model_path = save_model_with_metadata(
                    model=model_obj,
                    path=artifacts_dir / args.model_id,
                    input_shape=(1, n_features),
                    training_metadata={
                        "feature_names": feature_names,
                        "time_index_col": args.time_index_col,
                        "group_id_col": args.group_id_col,
                        "target_col": args.target_col,
                        "used_tft": used_tft,
                    },
                )
                artifact_format = model_path.suffix.lstrip(".")

            # Build and register manifest
            feature_schema = dict.fromkeys(feature_names, "float32")
            # Compute basic validation metrics
            perf_metrics: dict[str, float] = {}
            try:
                from ml._imports import HAS_SKLEARN

                if HAS_SKLEARN:
                    from sklearn.metrics import accuracy_score
                    from sklearn.metrics import brier_score_loss
                    from sklearn.metrics import roc_auc_score

                    p_val = 1.0 / (1.0 + np.exp(-z_val_vec))
                    perf_metrics = {
                        "auc": float(roc_auc_score(y_val_true.astype(int), p_val)),
                        "accuracy": float(
                            accuracy_score(y_val_true.astype(int), (p_val >= 0.5).astype(int)),
                        ),
                        "brier": float(brier_score_loss(y_val_true.astype(int), p_val)),
                    }
            except Exception:
                perf_metrics = {}

            assert y_val_true is not None
            manifest = ModelManifest(
                model_id=args.model_id,
                role=ModelRole.TEACHER,
                data_requirements=DataRequirements.HISTORICAL,
                architecture="TFT",
                feature_schema=feature_schema,
                feature_schema_hash=fman.schema_hash,
                parent_id=None,
                version="1.0.0",
                serveable=False,
                artifact_format=artifact_format,
                training_config={
                    "max_encoder_length": args.max_encoder_length,
                    "max_prediction_length": args.max_prediction_length,
                    "max_epochs": args.max_epochs,
                    "hidden_size": args.hidden_size,
                    "lstm_layers": args.lstm_layers,
                    "attention_head_size": args.attention_head_size,
                    "dropout": args.dropout,
                },
                performance_metrics=perf_metrics,
                feature_set_id=args.feature_set_id,
                pipeline_signature=fman.pipeline_signature,
                pipeline_version=fman.pipeline_version,
            )
            mreg = ModelRegistry(reg_dir)
            reg_id = mreg.register_model(model_path, manifest, auto_deploy=False)
            mreg.flush()
            print(f"Registered teacher model {reg_id} at {model_path}")
    else:
        # Load arrays for calibration-only/ONNX modes
        npz = np.load(args.student_window_npz, allow_pickle=True)
        y_val_true = None
        if "y_val_true" in npz:
            y_val_true = npz["y_val_true"].astype(np.float64)
        else:
            raise SystemExit("NPZ missing required key 'y_val_true'")

        z_val: npt.NDArray[np.float64] | None = None
        X_val: npt.NDArray[np.float32] | None = None
        if "z_val" in npz:
            z_val = np.asarray(npz["z_val"], dtype=np.float64)
        elif "X_val" in npz:
            X_val = np.asarray(npz["X_val"], dtype=np.float32)
            if X_val.ndim != 2 or X_val.shape[1] != n_features:
                raise SystemExit(
                    f"X_val shape {X_val.shape} does not match feature manifest width {n_features}",
                )
        else:
            raise SystemExit("NPZ must contain either 'z_val' or 'X_val' with 'y_val_true'")

        # Optionally load teacher ONNX to produce logits if X_val provided
        if X_val is not None:
            if not args.model_registry_dir or not args.teacher_model_id:
                raise SystemExit(
                    "Provide --model_registry_dir and --teacher_model_id to run ONNX teacher on X_val",
                )
            from ml.registry.model_registry import ModelRegistry

            mreg = ModelRegistry(Path(args.model_registry_dir))
            session = mreg.load_model(args.teacher_model_id)
            if session is None:
                raise SystemExit(
                    f"Failed to load teacher model {args.teacher_model_id} from registry",
                )
            # Run inference
            try:
                session_any: Any = session
                input_name = (
                    session_any.get_inputs()[0].name
                    if hasattr(session_any, "get_inputs")
                    else ONNX_INPUT_NAME
                )
                outputs: list[Any] = session_any.run(None, {input_name: X_val})
                raw_out = outputs[0]
                raw = np.asarray(raw_out, dtype=np.float64).reshape(-1)
                if args.onnx_output_is_logits:
                    z_val = raw
                else:
                    eps = 1e-6
                    p = np.clip(raw, eps, 1.0 - eps)
                    z_val = np.log(p / (1.0 - p))
            except Exception as exc:  # pragma: no cover - runtime dependency
                raise SystemExit(f"ONNX inference failed: {exc}")

        assert z_val is not None

        # Calibrate and produce calibrated probabilities
        teacher = CalibratingTeacher(TeacherConfig(architecture="TFT"))
        teacher.calibrate(z_val.reshape(-1, 1), y_val_true)
        q_cal = teacher.predict_proba(z_val.reshape(-1, 1)).astype(np.float32)
        q_val = q_cal

    # Persist outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "teacher_preds.npz"
    # Save q_train (if available) and q_val for student distillation
    assert y_val_true is not None
    if q_train is not None:
        np.savez_compressed(
            preds_path,
            q_train=q_train.squeeze(),
            q_val=(q_val.squeeze() if q_val is not None else np.array([], dtype=np.float32)),
            y_val_true=y_val_true.astype(np.float32),
        )
    else:
        # Calibration-only/ONNX path — save validation predictions only
        np.savez_compressed(
            preds_path,
            q_val=(q_val.squeeze() if q_val is not None else np.array([], dtype=np.float32)),
            y_val_true=y_val_true.astype(np.float32),
        )
    meta_path = out_dir / "teacher_meta.json"
    meta = {
        "model_id": args.model_id,
        "feature_set_id": args.feature_set_id,
        "feature_schema_hash": fman.schema_hash,
        "teacher_model_id": args.teacher_model_id,
        "calibrator": True,
        "onnx_output_is_logits": bool(args.onnx_output_is_logits),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {preds_path}\nMeta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
