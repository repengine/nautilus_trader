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
- validation_returns.npy: validation forward returns aligned to q_val (when available)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_PANDAS
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.data import DatasetMetadataExpectations
from ml.data import load_dataset_metadata
from ml.data import validate_dataset_metadata_expectations
from ml.data.vintage import VintagePolicy
from ml.registry.feature_registry import FeatureRegistry
from ml.tasks.datasets.splits import create_purged_splits
from ml.training.teacher.base import BaseTeacher
from ml.training.teacher.base import TeacherConfig


logger = logging.getLogger(__name__)


def _resolve_tft_feature_columns(
    df: object,
    *,
    feature_names: list[str],
    group_id_col: str,
    static_categoricals: list[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Resolve numeric/static/encoded feature columns for TFT training.

    Returns numeric feature names, static categorical feature names, and
    dynamically encoded categorical feature names.
    """
    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])
        raise ImportError("pandas is required for TFT feature resolution")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if group_id_col not in df.columns:
        raise ValueError(f"Missing group_id_col in DataFrame: {group_id_col}")

    static_set = set(static_categoricals or [])
    numeric_cols: list[str] = []
    static_cols: list[str] = []
    encoded_cols: list[str] = []

    for name in feature_names:
        if name not in df.columns:
            continue
        series = df[name]

        if name in static_set:
            df[name] = series.fillna("UNKNOWN").astype("category")
            static_cols.append(name)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(name)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            converted = pd.to_datetime(series, errors="coerce")
            values = converted.view("int64").astype("float64")
            values[converted.isna()] = np.nan
            df[name] = values
            numeric_cols.append(name)
            continue

        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().any():
            df[name] = coerced
            numeric_cols.append(name)
            continue

        if static_categoricals is None:
            per_group = (
                df[[group_id_col, name]]
                .dropna(subset=[group_id_col])
                .groupby(group_id_col)[name]
                .nunique(dropna=True)
            )
            if not per_group.empty and (per_group <= 1).all():
                df[name] = series.fillna("UNKNOWN").astype("category")
                static_cols.append(name)
                continue

        categorical = series.astype("category")
        df[name] = categorical.cat.codes.astype("int64")
        numeric_cols.append(name)
        encoded_cols.append(name)

    return numeric_cols, static_cols, encoded_cols


def _compute_sharpe_ratio(
    probabilities: npt.NDArray[np.float64],
    returns: npt.NDArray[np.float64],
) -> float:
    """Compute a simple Sharpe ratio using binary signals derived from probabilities."""
    if probabilities.size == 0 or returns.size == 0:
        return 0.0

    n = int(min(probabilities.size, returns.size))
    if n <= 0:
        return 0.0

    aligned_probs = probabilities[:n]
    aligned_returns = returns[:n]

    signals = (aligned_probs >= 0.5).astype(np.float64)
    strategy_returns = aligned_returns * signals
    strategy_returns = strategy_returns[np.isfinite(strategy_returns)]
    if strategy_returns.size == 0:
        return 0.0

    std = float(np.std(strategy_returns))
    if std == 0.0:
        return 0.0
    mean = float(np.mean(strategy_returns))
    return float(mean / std)


def _persist_teacher_outputs(
    out_dir: Path,
    *,
    q_train: npt.NDArray[np.float32] | None,
    q_val: npt.NDArray[np.float32] | None,
    y_val_true: npt.NDArray[np.float64] | None,
    meta: dict[str, Any],
    streaming_telemetry: Any | None = None,
) -> dict[str, Path]:
    """
    Persist teacher outputs for downstream student distillation.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    q_train : np.ndarray | None
        Training predictions.
    q_val : np.ndarray | None
        Validation predictions.
    y_val_true : np.ndarray | None
        Validation true labels.
    meta : dict
        Metadata dictionary.
    streaming_telemetry : Any | None
        Optional streaming telemetry to persist.

    Returns
    -------
    dict[str, Path]
        Dictionary with keys preds_path, meta_path, and optionally streaming_summary_path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    # Save predictions
    preds_path = out_dir / "teacher_preds.npz"
    y_true = y_val_true.astype(np.float32) if y_val_true is not None else np.array([], dtype=np.float32)
    if q_train is not None:
        np.savez_compressed(
            preds_path,
            q_train=q_train.squeeze(),
            q_val=(q_val.squeeze() if q_val is not None else np.array([], dtype=np.float32)),
            y_val_true=y_true,
        )
    else:
        np.savez_compressed(
            preds_path,
            q_val=(q_val.squeeze() if q_val is not None else np.array([], dtype=np.float32)),
            y_val_true=y_true,
        )
    result["preds_path"] = preds_path

    # Save metadata
    meta_path = out_dir / "teacher_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    result["meta_path"] = meta_path

    # Optionally save streaming telemetry summary
    if streaming_telemetry is not None:
        summary_path = out_dir / "streaming_summary.json"
        if hasattr(streaming_telemetry, "to_dict"):
            summary_dict = streaming_telemetry.to_dict()
        elif hasattr(streaming_telemetry, "as_dict"):
            summary_dict = streaming_telemetry.as_dict()
        else:
            summary_dict = streaming_telemetry
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
        result["streaming_summary_path"] = summary_path

    return result


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
    ap.add_argument(
        "--dataset_metadata",
        required=False,
        help="Path to dataset_metadata.json (required when training on dataset CSV/Parquet)",
    )
    ap.add_argument("--expected_dataset_id", required=False)
    ap.add_argument(
        "--expected_vintage_policy",
        required=False,
        choices=[policy.value for policy in VintagePolicy],
    )
    ap.add_argument("--expected_vintage_cutoff", required=False)
    # Optional decision policy adapter for inference actor
    ap.add_argument(
        "--decision_policy",
        required=False,
        default=None,
        help="Fully-qualified adapter or strategy class for prediction→signal policy",
    )
    ap.add_argument(
        "--decision_config",
        required=False,
        default=None,
        help="JSON dict of parameters passed to the decision policy adapter",
    )
    # Optional training mode
    ap.add_argument("--train_data_csv", required=False, help="CSV with training data")
    ap.add_argument("--train_data_parquet", required=False, help="Parquet with training data")
    ap.add_argument("--target_col", required=False, default="y")
    ap.add_argument("--time_index_col", required=False, default="time_index")
    ap.add_argument("--timestamp_col", required=False, default="timestamp")
    ap.add_argument("--group_id_col", required=False, default="instrument_id")
    ap.add_argument(
        "--limit_groups",
        required=False,
        type=int,
        default=0,
        help="If > 0, keep only the top-N groups by row count before training",
    )
    ap.add_argument("--max_encoder_length", required=False, type=int, default=30)
    ap.add_argument("--max_prediction_length", required=False, type=int, default=1)
    ap.add_argument("--max_epochs", required=False, type=int, default=1)
    ap.add_argument(
        "--val_days",
        required=False,
        type=int,
        default=0,
        help="If >0 and timestamp column exists, use last N days for validation",
    )
    ap.add_argument(
        "--embargo_hours",
        required=False,
        type=float,
        default=24.0,
        help="Embargo window in hours for purged splits",
    )
    ap.add_argument(
        "--purge_gap",
        required=False,
        type=int,
        default=0,
        help="Gap in samples between train and validation folds",
    )
    ap.add_argument(
        "--cv_splits",
        required=False,
        type=int,
        default=5,
        help="Number of purged cross-validation splits",
    )
    ap.add_argument(
        "--test_fraction",
        required=False,
        type=float,
        default=0.2,
        help="Fraction of data reserved as hold-out test",
    )
    ap.add_argument("--hidden_size", required=False, type=int, default=16)
    ap.add_argument("--lstm_layers", required=False, type=int, default=1)
    ap.add_argument("--attention_head_size", required=False, type=int, default=2)
    ap.add_argument("--dropout", required=False, type=float, default=0.1)
    ap.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=64,
        help="Batch size for train/val DataLoaders (default: 64)",
    )

    ap.add_argument(
        "--accelerator",
        required=False,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Lightning accelerator to use (auto/cpu/gpu)",
    )
    ap.add_argument(
        "--devices",
        required=False,
        type=int,
        default=1,
        help="Number of devices to use when accelerator is gpu (default: 1)",
    )
    ap.add_argument(
        "--dataloader_workers",
        required=False,
        type=int,
        default=0,
        help="Number of DataLoader workers for train/val (default: 0)",
    )
    ap.add_argument(
        "--precision",
        required=False,
        default="32",
        help="Training precision for Lightning Trainer (e.g., 32, 16, 16-mixed, bf16)",
    )
    ap.add_argument(
        "--learning_rate",
        required=False,
        type=float,
        default=3e-4,
        help="Optimizer learning rate for TFT (default: 3e-4)",
    )
    ap.add_argument(
        "--loss",
        required=False,
        choices=["poisson", "bce"],
        default="poisson",
        help="Loss function for TFT teacher (default: poisson)",
    )
    ap.add_argument(
        "--pos_weight",
        required=False,
        default=None,
        help="Positive class weight for BCE; 'auto' or a float (e.g., 3.0)",
    )
    ap.add_argument("--seed", required=False, type=int, default=None)
    ap.add_argument(
        "--tail_rows",
        required=False,
        type=int,
        default=0,
        help=(
            "If > 0, keep only the last N rows per group after sorting by time_index to cap memory"
        ),
    )
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
        "--pretrained_state_path",
        required=False,
        default=None,
        help="Optional path to a pretrained state dict for warm-start (e.g., MTM)",
    )
    ap.add_argument(
        "--register_teacher",
        action="store_true",
        help="Register the trained teacher as a non-serveable model",
    )
    args = ap.parse_args(argv)

    metadata_path: Path | None = Path(args.dataset_metadata) if args.dataset_metadata else None
    metadata_required = bool(args.train_data_csv or args.train_data_parquet)
    metadata = None
    if metadata_required:
        if metadata_path is None:
            candidate = args.train_data_parquet or args.train_data_csv
            if candidate:
                metadata_path = Path(candidate).with_name("dataset_metadata.json")
        if metadata_path is None:
            metadata_path = Path(args.out_dir) / "dataset_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata is required at {metadata_path}")
        metadata = load_dataset_metadata(metadata_path)
        expected_policy = (
            VintagePolicy(args.expected_vintage_policy)
            if args.expected_vintage_policy
            else None
        )
        expectations = DatasetMetadataExpectations(
            dataset_id=args.expected_dataset_id,
            vintage_policy=expected_policy,
            vintage_cutoff=args.expected_vintage_cutoff,
        )
        validate_dataset_metadata_expectations(
            metadata,
            expectations,
            context="tft_cli",
        )
        if metadata.dataset_id is None:
            raise ValueError("dataset_metadata.json must include dataset_id when training a teacher")
    elif metadata_path is not None:
        metadata = load_dataset_metadata(metadata_path)
        expected_policy = (
            VintagePolicy(args.expected_vintage_policy)
            if args.expected_vintage_policy
            else None
        )
        expectations = DatasetMetadataExpectations(
            dataset_id=args.expected_dataset_id,
            vintage_policy=expected_policy,
            vintage_cutoff=args.expected_vintage_cutoff,
        )
        validate_dataset_metadata_expectations(
            metadata,
            expectations,
            context="tft_cli",
        )

    if metadata_path is not None and args.dataset_metadata is None:
        args.dataset_metadata = str(metadata_path)

    # Resolve feature manifest and enforce schema
    freg = FeatureRegistry(Path(args.feature_registry_dir))
    finfo = freg.get_feature_set(args.feature_set_id)
    if finfo is None:
        raise SystemExit(f"Unknown feature_set_id: {args.feature_set_id}")
    fman = finfo.manifest
    feature_names = list(fman.feature_names)
    n_features = len(feature_names)

    # Ensure either training CSV or NPZ path provided
    if not args.train_data_csv and not args.train_data_parquet and not args.student_window_npz:
        raise SystemExit(
            "Provide either --train_data_csv/--train_data_parquet for training or --student_window_npz for calibration",
        )

    # Initialize outputs (populated below depending on mode)
    q_train: npt.NDArray[np.float32] | None = None
    q_val: npt.NDArray[np.float32] | None = None
    y_val_true: npt.NDArray[np.float64] | None = None
    validation_returns: npt.NDArray[np.float64] | None = None

    # If training data is provided, run training mode
    if args.train_data_csv or args.train_data_parquet:
        if not HAS_PANDAS:
            check_ml_dependencies(["pandas"])  # pragma: no cover - import guard
        if pd is None:
            raise SystemExit("pandas is required to load training data")
        df = (
            pd.read_parquet(args.train_data_parquet)
            if args.train_data_parquet
            else pd.read_csv(args.train_data_csv)
        )
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
        # Optionally restrict groups to top-N by size
        if int(args.limit_groups or 0) > 0:
            counts = df_sorted[args.group_id_col].value_counts()
            keep = set(counts.head(int(args.limit_groups)).index.tolist())
            df_sorted = df_sorted[df_sorted[args.group_id_col].isin(keep)]
        # Optionally cap rows per group to tail N (recent data)
        if int(args.tail_rows or 0) > 0:
            try:
                df_sorted = (
                    df_sorted.groupby(args.group_id_col, group_keys=False)
                    .tail(int(args.tail_rows))
                    .reset_index(drop=True)
                )
            except Exception:
                # Fallback: global tail if groupby-tail not available
                df_sorted = df_sorted.tail(int(args.tail_rows)).reset_index(drop=True)
        # Prefer time-based validation window if requested and timestamp is available
        _df_train = None
        df_val = None
        use_time_window = (
            int(getattr(args, "val_days", 0) or 0) > 0 and args.timestamp_col in df_sorted.columns
        )
        if use_time_window:
            try:
                ts = pd.to_datetime(df_sorted[args.timestamp_col], errors="coerce")
                df_sorted = df_sorted.assign(_ts=ts)
                max_ts = df_sorted["_ts"].max()
                if pd.notna(max_ts):
                    cutoff_ts = max_ts - pd.Timedelta(days=int(args.val_days))
                    mask_val = df_sorted["_ts"] > cutoff_ts
                    df_val = df_sorted.loc[mask_val].drop(columns=["_ts"], errors="ignore")
                    _df_train = df_sorted.loc[~mask_val].drop(columns=["_ts"], errors="ignore")
            except Exception:
                _df_train = None
                df_val = None

        if (_df_train is None or df_val is None or len(df_val) == 0) and not use_time_window:
            try:
                split_info = create_purged_splits(
                    df_sorted,
                    timestamp_col=args.timestamp_col,
                    test_fraction=float(args.test_fraction),
                    n_splits=int(args.cv_splits),
                    purge_gap=int(args.purge_gap),
                    embargo_hours=float(args.embargo_hours),
                )
                if split_info["cv_splits"]:
                    train_idx, val_idx = split_info["cv_splits"][-1]
                    _df_train = df_sorted.iloc[train_idx]
                    df_val = df_sorted.iloc[val_idx]
            except Exception:
                _df_train = None
                df_val = None

        if _df_train is None or df_val is None or len(df_val) == 0:
            n_total = len(df_sorted)
            min_val_len = max(int(n_total * 0.2), 1)
            cutoff = max(int(n_total * 0.8), n_total - min_val_len)
            _df_train = df_sorted.iloc[:cutoff]
            df_val = df_sorted.iloc[cutoff:]
        y_val_true = np.asarray(df_val[args.target_col], dtype=np.float64).reshape(-1)
        if "forward_return" in df_val.columns:
            try:
                validation_returns = np.asarray(
                    df_val["forward_return"],
                    dtype=np.float64,
                ).reshape(-1)
            except Exception:
                validation_returns = None

        # Try TFT teacher; if unavailable or fails, fall back to a simple linear model producing logits
        z_val_vec: npt.NDArray[np.float64] | None
        z_train_vec: npt.NDArray[np.float64] | None
        z_all: npt.NDArray[np.float64] | None = None
        used_tft = False
        # Determine BCE pos_weight (class imbalance) if requested and applicable
        pos_weight_val = None
        if str(args.loss).lower() == "bce" and args.pos_weight:
            try:
                if str(args.pos_weight).strip().lower() == "auto":
                    # prevalence on training slice
                    y_tr = np.asarray(_df_train[args.target_col], dtype=np.float64)
                    prev = float(y_tr.mean()) if y_tr.size > 0 else 0.0
                    if prev > 0.0 and prev < 1.0:
                        pos_weight_val = float((1.0 - prev) / prev)
                else:
                    pos_weight_val = float(args.pos_weight)
            except Exception:
                pos_weight_val = None
        try:  # pragma: no cover - exercised in integration path when dependencies ok
            from ml.training.teacher.tft_teacher import TFTTeacher
            from ml.training.teacher.tft_teacher import TFTTeacherConfig

            teacher_tft = TFTTeacher(
                TFTTeacherConfig(
                    architecture="TFT",
                    loss_name=str(args.loss),
                    pos_weight=pos_weight_val,
                ),
                max_encoder_length=args.max_encoder_length,
                max_prediction_length=args.max_prediction_length,
                time_varying_unknown_reals=feature_names,
                batch_size=int(args.batch_size),
                precision=str(args.precision),
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
                pretrained_state_path=(args.pretrained_state_path or None),
                learning_rate=float(args.learning_rate),
                accelerator=str(args.accelerator),
                devices=int(args.devices),
            )
            teacher_tft.fit(df)
            # Prefer aligned PF targets for validation to ensure q_val matches y_val_true
            z_val_vec = None
            z_train_vec = None
            try:
                # Prefer PF return_x alignment when it yields a sufficiently large validation set
                z_val_vec, y_val_true_pf = teacher_tft.predict_logits_with_targets(df_val)
                min_required = 100
                if (
                    z_val_vec is None
                    or (hasattr(z_val_vec, "size") and int(z_val_vec.size) < min_required)
                    or (
                        hasattr(y_val_true_pf, "size")
                        and int(getattr(y_val_true_pf, "size", 0)) < min_required
                    )
                    or (np.unique(y_val_true_pf).size < 2)
                ):
                    raise RuntimeError(
                        "Insufficient validation samples or label variance from PF alignment",
                    )
                # Override y_val_true with PF-aligned decoder targets
                y_val_true = y_val_true_pf
            except Exception:
                # Fallback path 1: predict on the full sorted frame and slice
                try:
                    z_all = teacher_tft.predict_logits(df_sorted)
                    z_val_vec = z_all[cutoff:]
                    # use original df_val labels already assigned above
                except Exception:
                    z_val_vec = None
            # Fallback path 2: predict directly on validation frame
            if z_val_vec is None or (hasattr(z_val_vec, "size") and z_val_vec.size == 0):
                try:
                    z_val_vec = teacher_tft.predict_logits(df_val)
                except Exception:
                    z_val_vec = None
            # For q_train, compute logits on the training slice directly; fallback to slicing if needed
            try:
                z_train_vec = teacher_tft.predict_logits(_df_train)
            except Exception:
                try:
                    if z_all is not None:
                        z_train_vec = z_all[:cutoff]
                    else:
                        z_train_vec = None
                except Exception:
                    z_train_vec = None
            # Guard: ensure we have non-empty validation logits for calibration
            if z_val_vec is None or (hasattr(z_val_vec, "size") and z_val_vec.size == 0):
                raise RuntimeError("Empty validation logits after TFT prediction")
            # Align y_val_true length to z_val_vec if needed (fallback modes may produce shorter sequences)
            try:
                import numpy as _np

                z_len = int(z_val_vec.shape[0])
                y_len = int(_np.asarray(y_val_true).shape[0]) if y_val_true is not None else 0
                if z_len > 0 and y_len > 0 and z_len != y_len:
                    if z_len < y_len:
                        # Use most recent labels to match predicted horizon
                        y_val_true = _np.asarray(y_val_true, dtype=_np.float64)[-z_len:]
                        if validation_returns is not None:
                            validation_returns = _np.asarray(
                                validation_returns,
                                dtype=_np.float64,
                            )[-z_len:]
                    else:
                        z_val_vec = _np.asarray(z_val_vec, dtype=_np.float64)[-y_len:]
                        if validation_returns is not None and validation_returns.size >= y_len:
                            validation_returns = _np.asarray(
                                validation_returns,
                                dtype=_np.float64,
                            )[-y_len:]
                # If training logits exist, enforce 80/20 split consistency (optional)
                if z_train_vec is not None:
                    zt = int(z_train_vec.shape[0])
                    if zt == 0:
                        z_train_vec = None
            except Exception as exc:
                logging.getLogger(__name__).debug("TFT post-processing alignment failed: %s", exc)
            used_tft = True
        except Exception:
            import logging as _logging

            _logging.getLogger(__name__).exception(
                "TFT training failed; falling back to logistic regression",
            )
            # Fallback: scikit-learn logistic regression as a simple teacher proxy
            from ml._imports import HAS_SKLEARN

            if not HAS_SKLEARN:
                raise SystemExit("Training requires TFT dependencies or scikit-learn as fallback")
            from sklearn.linear_model import LogisticRegression

            X = np.asarray(df_sorted[feature_names].to_numpy(), dtype=np.float64)
            y = np.asarray(df_sorted[args.target_col].to_numpy(), dtype=int)
            # Derive cutoff from previously prepared splits when available; else 80/20
            try:
                # _df_train is defined earlier in this scope; use its length when available
                cut_idx = len(_df_train) if _df_train is not None else None
            except Exception:
                cut_idx = None
            if not cut_idx:
                n_total = int(X.shape[0])
                min_val_len = 5000
                cut_idx = max(int(n_total * 0.8), n_total - min_val_len)
            X_train, X_val_arr = X[:cut_idx], X[cut_idx:]
            y_train = y[:cut_idx]
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
        if z_val_vec is None:
            raise SystemExit("Calibration requires z_val_vec logits; got None")
        teacher.calibrate(z_val_vec.reshape(-1, 1), y_val_true)
        q_val = teacher.predict_proba(z_val_vec.reshape(-1, 1)).astype(np.float32)
        if z_train_vec is not None:
            q_train = teacher.predict_proba(z_train_vec.reshape(-1, 1)).astype(np.float32)

        # Optional interpretability save
        if used_tft and args.save_interpretability:
            try:
                # Build val dataset/loader similar to predict path
                from pytorch_forecasting import TimeSeriesDataSet

                training_ds = getattr(teacher_tft, "_training_dataset", None)
                if training_ds is None:
                    raise RuntimeError("TFT training dataset missing for interpretability save")
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
        if args.register_teacher:
            from sqlalchemy.exc import OperationalError as _OpErr

            from ml.core.db_engine import EngineManager as _EM
            from ml.registry.base import DataRequirements
            from ml.registry.base import ModelManifest
            from ml.registry.base import ModelRole
            from ml.registry.model_registry import ModelRegistry

            # Persistence preference: Database → JSON
            from ml.registry.persistence import BackendType as _BT
            from ml.registry.persistence import PersistenceConfig as _PC
            from ml.training.export import save_model_with_metadata

            # Resolve registry path (used for artifacts); default if not provided
            reg_dir = (
                Path(args.model_registry_dir)
                if args.model_registry_dir
                else Path.home() / ".nautilus" / "ml_registry"
            )
            artifacts_dir = reg_dir / "artifacts" / "teachers"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Choose an artifact format and persist (TorchScript > safetensors > pickle)
            artifact_format = "pkl"
            if used_tft and getattr(args, "export_torchscript", False):
                try:
                    from pytorch_forecasting import TimeSeriesDataSet

                    from ml.training.teacher.tft_torchscript import export_tft_to_torchscript_from_batch

                    training_ds = getattr(teacher_tft, "_training_dataset", None)
                    if training_ds is None:
                        raise RuntimeError("TFT training dataset missing for TorchScript export")
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

            if y_val_true is None:
                raise SystemExit("y_val_true is required to build manifest metrics")
            # Parse optional decision adapter config
            decision_cfg: dict[str, Any] = {}
            if args.decision_config:
                try:
                    decision_cfg = cast(dict[str, Any], json.loads(args.decision_config))
                except Exception as exc:
                    raise SystemExit(f"Invalid --decision_config JSON: {exc}")

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
                decision_policy=(args.decision_policy or None),
                decision_config=decision_cfg,
            )
            # Prefer DB when available; fallback to JSON
            db_url = None
            try:
                import os as _os

                db_url = (
                    _os.getenv("NAUTILUS_DB")
                    or _os.getenv("DB_CONNECTION")
                    or "postgresql://postgres:postgres@localhost:5432/nautilus"
                )
                engine = _EM.get_engine(db_url)
                with engine.connect() as conn:
                    conn.exec_driver_sql("SELECT 1")
                persistence = _PC(backend=_BT.POSTGRES, connection_string=str(db_url))
            except (_OpErr, Exception):
                persistence = _PC(backend=_BT.JSON, json_path=reg_dir)

            mreg = ModelRegistry(reg_dir, persistence_config=persistence)
            reg_id = mreg.register_model(model_path, manifest, auto_deploy=False)
            mreg.flush()
            print(f"Registered teacher model {reg_id} at {model_path}")
    else:
        # Load arrays for calibration-only/ONNX modes
        npz = np.load(args.student_window_npz, allow_pickle=True)
        if "y_val_true" not in npz:
            raise SystemExit("NPZ missing required key 'y_val_true'")
        y_val_true = npz["y_val_true"].astype(np.float64)

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
                try:
                    from ml.config.names import ONNX_INPUT_NAME as _ONNX_INPUT
                except Exception:
                    _ONNX_INPUT = "input"
                input_name = (
                    session_any.get_inputs()[0].name
                    if hasattr(session_any, "get_inputs")
                    else _ONNX_INPUT
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

        if z_val is None:
            raise SystemExit("Calibration requires z_val logits; got None")

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
    if y_val_true is None:
        raise SystemExit("Final outputs require y_val_true; got None")
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

    # ---------------------------------------------------------------------
    # Predictive metrics summary (Phase-1 gating inputs)
    # ---------------------------------------------------------------------
    # Compute core predictive metrics from q_val/y_val_true and, when available,
    # basic stability diagnostics across instruments and calendar weeks.
    try:
        # Ensure arrays available
        q = (
            (q_val if q_val is not None else np.array([], dtype=np.float32))
            .astype(np.float64)
            .reshape(-1)
        )
        y = (
            (y_val_true if y_val_true is not None else np.array([], dtype=np.float64))
            .astype(np.float64)
            .reshape(-1)
        )
        if q.size and y.size and q.size == y.size:
            # Clip for numeric stability
            p = np.clip(q, 1e-6, 1.0 - 1e-6)
            returns_arr: npt.NDArray[np.float64] | None = None
            if validation_returns is not None:
                try:
                    returns_arr = np.asarray(validation_returns, dtype=np.float64).reshape(-1)
                    if returns_arr.size != p.size and returns_arr.size > 0:
                        returns_arr = returns_arr[-p.size :]
                except Exception:
                    returns_arr = None
            # Lightweight metrics (avoid heavy deps by default)
            try:
                from ml.evaluation.metrics import binary_logloss as _ll
                from ml.evaluation.metrics import pr_auc as _pra
                from ml.evaluation.metrics import roc_auc as _ra

                roc_auc = float(_ra(y, p))
                pr_auc = float(_pra(y, p))
                logloss = float(_ll(y, p))
            except Exception:
                # Fallback via sklearn if available in training env
                try:
                    from sklearn.metrics import average_precision_score
                    from sklearn.metrics import log_loss
                    from sklearn.metrics import roc_auc_score

                    roc_auc = float(roc_auc_score(y.astype(int), p))
                    pr_auc = float(average_precision_score(y.astype(int), p))
                    logloss = float(log_loss(y.astype(int), p))
                except Exception:
                    roc_auc = 0.0
                    pr_auc = 0.0
                    logloss = float("nan")
            # Brier
            try:
                brier = float(np.mean((p - y) ** 2))
            except Exception:
                brier = float("nan")
            # ECE (10-bin)
            try:
                bins = np.linspace(0.0, 1.0, 11)
                inds = np.digitize(p, bins) - 1
                ece = 0.0
                n = len(p)
                for b in range(10):
                    mask = inds == b
                    if np.any(mask):
                        conf = float(np.mean(p[mask]))
                        acc = float(np.mean(y[mask]))
                        ece += (np.sum(mask) / n) * abs(acc - conf)
                ece = float(ece)
            except Exception:
                ece = float("nan")

            prev = float(y.mean()) if y.size > 0 else 0.0
            prx = float(pr_auc / prev) if prev > 0.0 else 0.0
            sharpe_ratio = float("nan")
            if returns_arr is not None and returns_arr.size:
                sharpe_ratio = _compute_sharpe_ratio(p, returns_arr)

            # Stability across instruments/weeks when validation DataFrame is available
            stability_inst_std = float("nan")
            stability_week_std = float("nan")
            uniq_instruments: list[str] | None = None
            if "df_val" in locals() and df_val is not None:
                try:
                    dfv = df_val.reset_index(drop=True)
                    if "instrument_id" in dfv.columns:
                        uniq_instruments = sorted(
                            {str(x) for x in dfv["instrument_id"].astype(str).tolist()},
                        )
                        # align lengths defensively
                        m = min(len(dfv), len(p))
                        dfp = dfv.iloc[:m]
                        pv = p[:m]
                        yv = y[:m]
                        aucs: list[float] = []
                        for inst, sub in dfp.groupby("instrument_id"):
                            idx = sub.index.to_numpy()
                            yy = yv[idx]
                            pp = pv[idx]
                            if (yy.sum() > 0) and (len(yy) - yy.sum() > 0):
                                try:
                                    from ml.evaluation.metrics import roc_auc as _ra2

                                    aucs.append(float(_ra2(yy, pp)))
                                except Exception as auc_exc:
                                    logger.debug(
                                        "tft_cli.instrument_auc_failed instrument=%s error=%s",
                                        inst,
                                        auc_exc,
                                        exc_info=True,
                                    )
                        if aucs:
                            stability_inst_std = float(np.std(np.asarray(aucs, dtype=np.float64)))
                    # Weekly grouping if timestamp available
                    if "timestamp" in dfv.columns:
                        import pandas as _pd

                        ts = _pd.to_datetime(dfv["timestamp"], errors="coerce")
                        dfp2 = dfv.assign(_ts=ts)
                        m2 = min(len(dfp2), len(p))
                        pv2 = p[:m2]
                        yv2 = y[:m2]
                        aucs_w: list[float] = []
                        for _, sub in dfp2.iloc[:m2].groupby(_pd.Grouper(key="_ts", freq="W")):
                            if len(sub) == 0:
                                continue
                            idx = sub.index.to_numpy()
                            yy = yv2[idx]
                            pp = pv2[idx]
                            if (yy.sum() > 0) and (len(yy) - yy.sum() > 0):
                                try:
                                    from ml.evaluation.metrics import roc_auc as _ra3

                                    aucs_w.append(float(_ra3(yy, pp)))
                                except Exception as auc_exc:
                                    logger.debug(
                                        "tft_cli.weekly_auc_failed error=%s",
                                        auc_exc,
                                        exc_info=True,
                                    )
                        if aucs_w:
                            stability_week_std = float(np.std(np.asarray(aucs_w, dtype=np.float64)))
                except Exception as stability_exc:
                    logger.warning(
                        "tft_cli.stability_metrics_failed error=%s",
                        stability_exc,
                        exc_info=True,
                    )
                    stability_inst_std = float("nan")
                    stability_week_std = float("nan")

            # Universe derivation (best-effort): prefer instrument_ids from validation frame
            universe_instrument_ids: list[str] | None = None
            if uniq_instruments is not None and len(uniq_instruments) > 0:
                universe_instrument_ids = uniq_instruments
            else:
                # Fallback: try training DataFrame if available
                if (
                    "df_sorted" in locals()
                    and isinstance(df_sorted, object)
                    and hasattr(df_sorted, "columns")
                ):
                    try:
                        from typing import Any as _Any
                        from typing import cast as _cast

                        df_any = _cast(_Any, df_sorted)
                        if "instrument_id" in df_any.columns:
                            universe_instrument_ids = sorted(
                                {str(x) for x in df_any["instrument_id"].astype(str).tolist()},
                            )
                    except Exception as universe_exc:
                        logger.debug(
                            "tft_cli.universe_build_failed error=%s",
                            universe_exc,
                            exc_info=True,
                        )
                        universe_instrument_ids = None

            # Build JSON payload for promotions wiring
            metrics_out: dict[str, Any] = {
                "model_id": args.model_id,
                "architecture": "TFT",
                "feature_set_id": args.feature_set_id,
                "feature_schema_hash": fman.schema_hash,
                "serveable": False,
                "version": "1.0.0",
                # Metrics (lowercase to align with gates examples)
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "logloss": logloss,
                "brier": brier,
                "ece": ece,
                "prx": prx,
                "prevalence": prev,
                "sharpe_ratio": sharpe_ratio,
                # Stability diagnostics (stddev across groups)
                "stability_auc_by_instrument_std": stability_inst_std,
                "stability_auc_by_week_std": stability_week_std,
                # Advisory identifiers for actor auto-universe
                "training_dataset_id": "tft_dataset",
            }
            if returns_arr is not None and returns_arr.size:
                try:
                    returns_path = out_dir / "validation_returns.npy"
                    np.save(returns_path, returns_arr.astype(np.float32))
                    metrics_out["validation_returns_path"] = str(returns_path)
                except Exception as save_exc:
                    logger.debug(
                        "tft_cli.validation_returns_save_failed error=%s",
                        save_exc,
                        exc_info=True,
                    )
            if universe_instrument_ids:
                metrics_out["universe_instrument_ids"] = universe_instrument_ids

            # Include artifact path if available from training branch
            try:
                if "model_path" in locals() and model_path is not None:
                    metrics_out["model_path"] = str(model_path)
            except Exception as path_exc:
                logger.debug(
                    "tft_cli.metrics_model_path_attach_failed error=%s",
                    path_exc,
                    exc_info=True,
                )

            (out_dir / "model_metrics.json").write_text(
                json.dumps(metrics_out, indent=2),
                encoding="utf-8",
            )
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Writing model_metrics.json failed: %s",
            exc,
            exc_info=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
