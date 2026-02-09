#!/usr/bin/env python3
"""
Task helper for TFT HPO runs.

Shared by CLI entrypoints and orchestration layers to keep `ml/cli` thin.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from ml._imports import HAS_OPTUNA
from ml.common.subprocess_utils import SubprocessExecutionError
from ml.common.subprocess_utils import run_command
from ml.common.validation_strategies import HOLDOUT_STRATEGIES
from ml.common.validation_strategies import require_holdout_strategy
from ml.data import load_dataset_metadata
from ml.data import require_target_column_in_semantics
from ml.data import require_target_semantics_metadata
from ml.data import resolve_target_col_from_metadata


logger = logging.getLogger(__name__)

TeacherMain = Callable[[list[str] | None], int]


def _score(npz_path: Path) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    q = data["q_val"].astype(float).reshape(-1)
    y = data["y_val_true"].astype(int).reshape(-1)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(y, q))
    pr = float(average_precision_score(y, q))
    prev = float(y.mean())
    prx = float(pr / prev) if prev > 0 else 0.0
    ll = float(log_loss(y, q))
    br = float(brier_score_loss(y, q))
    # ECE (Expected Calibration Error)
    try:
        bins = np.linspace(0.0, 1.0, 11)
        inds = np.digitize(q, bins) - 1
        ece = 0.0
        n = len(q)
        for b in range(10):
            mask = inds == b
            if np.any(mask):
                conf = float(np.mean(q[mask]))
                acc = float(np.mean(y[mask]))
                ece += (np.sum(mask) / n) * abs(acc - conf)
        ece = float(ece)
    except Exception:
        ece = float("nan")
    return {
        "AUC": auc,
        "PR_AUC": pr,
        "PRx": prx,
        "LogLoss": ll,
        "Brier": br,
        "ECE": ece,
        "Prevalence": prev,
    }


def _infer_direction(metric: str) -> str:
    metric_lower = metric.strip().lower()
    minimize_aliases = {
        "logloss",
        "loss",
        "brier",
        "rmse",
        "mae",
        "mse",
        "error",
    }
    if metric_lower.endswith("loss") or metric_lower in minimize_aliases:
        return "minimize"
    return "maximize"


def _resolve_metric_value(metrics: Mapping[str, float], metric: str) -> float:
    target = metric.strip().lower()
    for key, value in metrics.items():
        if key.lower() == target:
            return float(value)
    fallback = metrics.get("PRx")
    return float(fallback) if fallback is not None else 0.0


def _merge_error(existing: str | None, new: str) -> str:
    """Append a new error message, preserving prior context."""
    if existing:
        return f"{existing}; {new}"
    return new


def _require_teacher_main(teacher_main: TeacherMain | None) -> TeacherMain:
    if teacher_main is None:
        raise RuntimeError("teacher_main is required for HPO runs")
    return teacher_main


def main(
    argv: list[str] | None = None,
    *,
    teacher_main: TeacherMain | None = None,
    has_optuna: bool | None = None,
) -> int:
    """
    Run an HPO sweep for the TFT teacher model.

    Args:
        argv: Optional CLI-style arguments list.
        teacher_main: Callable used to execute teacher training runs.
        has_optuna: Override for Optuna availability (for tests or feature gating).

    Returns:
        Exit code (0 for success).
    """
    ap = argparse.ArgumentParser(description="HPO sweep for TFT teacher (BCE)")
    ap.add_argument("--dataset_csv", required=False)
    ap.add_argument(
        "--dataset_parquet",
        required=False,
        help="Path to a Parquet dataset; if provided, supersedes --dataset_csv",
    )
    ap.add_argument(
        "--dataset_metadata",
        required=False,
        help="Path to dataset_metadata.json (required for target semantics alignment)",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_registry_dir", required=False, default=None)
    ap.add_argument("--feature_set_id", required=False, default=None)
    ap.add_argument(
        "--backend",
        choices=["grid", "optuna"],
        default="grid",
        help="Optimization backend to use (default: grid)",
    )
    ap.add_argument(
        "--metric",
        default="prx",
        help="Metric name to optimize (default: prx)",
    )
    ap.add_argument(
        "--direction",
        choices=["maximize", "minimize"],
        default=None,
        help="Optimization direction; inferred when omitted",
    )
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--tail_rows", type=int, default=0)
    ap.add_argument("--limit_groups", type=int, default=0)
    ap.add_argument(
        "--val_days",
        type=int,
        default=14,
        help="If >0 and timestamp exists, use last N days for validation (default: 14)",
    )
    ap.add_argument(
        "--cv_strategy",
        "--validation_strategy",
        dest="validation_strategy",
        choices=list(HOLDOUT_STRATEGIES),
        default="time_window",
        help="Validation strategy for HPO runs (default: time_window)",
    )
    ap.add_argument(
        "--purge_gap",
        type=int,
        default=0,
        help="Gap in samples between train and validation folds for purged CV",
    )
    ap.add_argument(
        "--embargo_pct",
        type=float,
        default=None,
        help="Embargo percentage for purged CV (overrides teacher default)",
    )
    # Hardware / Accelerator passthrough to teacher CLI
    ap.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Lightning accelerator to use for teacher runs (default: auto)",
    )
    ap.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use when accelerator is gpu (default: 1)",
    )
    ap.add_argument(
        "--subprocess",
        action="store_true",
        help="Run each config in a isolated subprocess (default: in-process for tests/debug)",
    )
    ap.add_argument(
        "--subprocess-timeout",
        type=float,
        default=None,
        help="Optional timeout (seconds) for each teacher subprocess run.",
    )
    ap.add_argument(
        "--precision",
        default="32",
        help="Training precision to pass to teacher CLI (e.g., 32, 16, 16-mixed, bf16)",
    )
    ap.add_argument(
        "--target_col",
        required=False,
        default=None,
        help="Target column name declared in target_semantics labels (or legacy_aliases when enabled).",
    )
    # Search space controls (comma-separated lists)
    ap.add_argument(
        "--hidden_sizes",
        type=str,
        default="32,64",
        help="Comma-separated hidden sizes (e.g., 32,64,128)",
    )
    ap.add_argument(
        "--lstm_layers_list",
        type=str,
        default="2,3",
        help="Comma-separated LSTM layer counts (e.g., 1,2,3)",
    )
    ap.add_argument(
        "--attention_heads",
        type=str,
        default="2,4",
        help="Comma-separated attention head sizes (e.g., 2,4,8)",
    )
    ap.add_argument(
        "--dropouts",
        type=str,
        default="0.1,0.2",
        help="Comma-separated dropout values (e.g., 0.1,0.2,0.3)",
    )
    ap.add_argument(
        "--learning_rates",
        type=str,
        default="0.0003,0.001",
        help="Comma-separated learning rates (e.g., 0.0003,0.001)",
    )
    ap.add_argument(
        "--max_encoder_lengths",
        type=str,
        default="30,60",
        help="Comma-separated encoder lengths (e.g., 30,60,120)",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated seeds (e.g., 1,2,3) to repeat runs",
    )
    args = ap.parse_args(argv)

    teacher_main_fn = _require_teacher_main(teacher_main)
    optuna_available = HAS_OPTUNA if has_optuna is None else bool(has_optuna)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    feature_registry_dir = (
        Path(args.feature_registry_dir)
        if args.feature_registry_dir is not None
        else out / "feature_registry"
    )
    feature_registry_dir.mkdir(parents=True, exist_ok=True)
    feature_set_id_value = args.feature_set_id or "default_feature_set"

    # Choose dataset source
    train_flag: list[str]
    train_path: str
    if args.dataset_parquet:
        train_flag = ["--train_data_parquet"]
        train_path = str(args.dataset_parquet)
    elif args.dataset_csv:
        train_flag = ["--train_data_csv"]
        train_path = str(args.dataset_csv)
    else:  # pragma: no cover - CLI guard
        print("ERROR: one of --dataset_parquet or --dataset_csv is required", file=sys.stderr)
        return 2

    metadata_path = (
        Path(args.dataset_metadata)
        if args.dataset_metadata
        else Path(train_path).with_name("dataset_metadata.json")
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata is required at {metadata_path}")
    metadata = load_dataset_metadata(metadata_path)
    require_target_semantics_metadata(metadata, context="hpo_tft")
    resolved_target_col = (
        str(args.target_col)
        if args.target_col is not None
        else resolve_target_col_from_metadata(metadata_path, context="hpo_tft")
    )
    require_target_column_in_semantics(metadata, resolved_target_col, context="hpo_tft")

    # Parse grids
    def _parse_ints(s: str) -> list[int]:
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    def _parse_floats(s: str) -> list[float]:
        vals: list[float] = []
        for tok in [t for t in s.split(",") if t.strip()]:
            vals.append(float(tok))
        return vals

    hidden_sizes = _parse_ints(args.hidden_sizes)
    lstm_layers = _parse_ints(args.lstm_layers_list)
    attn_heads = _parse_ints(args.attention_heads)
    dropouts = _parse_floats(args.dropouts)
    lrs = _parse_floats(args.learning_rates)
    enc_lengths = _parse_ints(args.max_encoder_lengths)
    seeds_int: tuple[int, ...] = tuple(_parse_ints(args.seeds)) if args.seeds else ()
    seeds: tuple[int | None, ...] = seeds_int if seeds_int else (None,)

    metric_name = args.metric
    direction = args.direction or _infer_direction(metric_name)
    effective_backend = args.backend
    if effective_backend == "optuna" and not optuna_available:
        print(
            "Optuna backend requested but Optuna is unavailable; falling back to grid.",
            file=sys.stderr,
        )
        effective_backend = "grid"
    elif effective_backend == "optuna":  # pragma: no cover - placeholder until Optuna backend lands
        print(
            "Optuna backend requested but integration is not yet implemented; using grid backend.",
            file=sys.stderr,
        )
        effective_backend = "grid"

    results: list[dict[str, Any]] = []
    validation_strategy = require_holdout_strategy(str(args.validation_strategy))
    val_days = int(args.val_days)
    if validation_strategy == "purged":
        val_days = 0
    elif val_days <= 0:
        raise ValueError("validation_strategy=time_window requires --val_days > 0")

    teacher_common_args = [
        *train_flag,
        train_path,
        "--max_epochs",
        str(args.epochs),
        "--dataset_metadata",
        str(metadata_path),
        "--val_days",
        str(val_days),
        "--loss",
        "bce",
        "--dataloader_workers",
        str(args.workers),
        "--pos_weight",
        "auto",
        "--batch_size",
        str(args.batch_size),
        "--accelerator",
        str(args.accelerator),
        "--devices",
        str(int(args.devices)),
        "--precision",
        str(args.precision),
        "--target_col",
        resolved_target_col,
        "--feature_registry_dir",
        str(feature_registry_dir),
        "--feature_set_id",
        feature_set_id_value,
    ]
    teacher_common_args += ["--validation_strategy", validation_strategy]
    if validation_strategy == "purged":
        teacher_common_args += ["--purge_gap", str(int(args.purge_gap))]
        if args.embargo_pct is not None:
            teacher_common_args += ["--embargo_pct", str(float(args.embargo_pct))]
    # Optional dataset capping
    if int(args.tail_rows or 0) > 0:
        teacher_common_args += ["--tail_rows", str(int(args.tail_rows))]
    if int(args.limit_groups or 0) > 0:
        teacher_common_args += ["--limit_groups", str(int(args.limit_groups))]

    train_args_common = [
        "-m",
        "ml.training.teacher.tft_cli",
        *teacher_common_args,
    ]

    run_id = 0
    for hs in hidden_sizes:
        for ll in lstm_layers:
            for ah in attn_heads:
                for dr in dropouts:
                    for lr in lrs:
                        for mel in enc_lengths:
                            for seed in seeds:
                                run_id += 1
                                seed_tag = f"_s{seed}" if seed is not None else ""
                                model_id = (
                                    f"tft_hpo_h{hs}_l{ll}_a{ah}_d{str(dr).replace('.', '')}"
                                    f"_lr{str(lr).replace('.', '')}_e{mel}_r{run_id}{seed_tag}"
                                )
                                run_dir = out / model_id
                                run_dir.mkdir(parents=True, exist_ok=True)
                                t0 = time.perf_counter()
                                rc: int
                                err_msg: str | None = None
                                if not args.subprocess:
                                    # In-process path (tests/debug): call provided teacher_main
                                    inproc_args = [
                                        *teacher_common_args,
                                        "--out_dir",
                                        str(run_dir),
                                        "--model_id",
                                        model_id,
                                        "--max_encoder_length",
                                        str(mel),
                                        "--hidden_size",
                                        str(hs),
                                        "--lstm_layers",
                                        str(ll),
                                        "--attention_head_size",
                                        str(ah),
                                        "--dropout",
                                        str(dr),
                                        "--learning_rate",
                                        str(lr),
                                    ]
                                    if seed is not None:
                                        inproc_args += ["--seed", str(int(seed))]
                                    rc = teacher_main_fn(inproc_args)
                                else:
                                    # Subprocess isolation to prevent memory from accumulating across runs
                                    cmd = [
                                        sys.executable,
                                        *train_args_common,
                                        "--out_dir",
                                        str(run_dir),
                                        "--model_id",
                                        model_id,
                                        "--max_encoder_length",
                                        str(mel),
                                        "--hidden_size",
                                        str(hs),
                                        "--lstm_layers",
                                        str(ll),
                                        "--attention_head_size",
                                        str(ah),
                                        "--dropout",
                                        str(dr),
                                        "--learning_rate",
                                        str(lr),
                                    ]
                                    if seed is not None:
                                        cmd += ["--seed", str(int(seed))]
                                    log_path = run_dir / "train.log"
                                    try:
                                        with open(log_path, "wb") as lf:
                                            try:
                                                proc = run_command(
                                                    cmd,
                                                    stdout=lf,
                                                    merge_stderr=True,
                                                    text=False,
                                                    timeout=args.subprocess_timeout,
                                                    log=logger,
                                                )
                                                rc = int(proc.returncode)
                                            except SubprocessExecutionError as exc:
                                                rc = int(exc.returncode)
                                                err_msg = _merge_error(err_msg, f"train_failed: {exc}")
                                    except OSError as file_error:
                                        logger.debug(
                                            "teacher_subprocess_log_open_failed model_id=%s path=%s",
                                            model_id,
                                            log_path,
                                            exc_info=True,
                                        )
                                        try:
                                            fallback_proc = run_command(
                                                cmd,
                                                capture_output=True,
                                                merge_stderr=True,
                                                text=True,
                                                timeout=args.subprocess_timeout,
                                                log=logger,
                                            )
                                            rc = int(fallback_proc.returncode)
                                            stdout_value = fallback_proc.stdout
                                            if isinstance(stdout_value, bytes):
                                                stdout_text = stdout_value.decode("utf-8", errors="ignore")
                                            else:
                                                stdout_text = stdout_value or ""
                                            log_path.write_text(stdout_text, encoding="utf-8")
                                        except SubprocessExecutionError as exc:
                                            rc = int(exc.returncode)
                                            error_stdout = ""
                                            if isinstance(exc.stdout, bytes):
                                                error_stdout = exc.stdout.decode("utf-8", errors="ignore")
                                            elif isinstance(exc.stdout, str):
                                                error_stdout = exc.stdout
                                            log_path.write_text(error_stdout, encoding="utf-8")
                                            err_msg = _merge_error(err_msg, f"train_failed: {exc}")
                                        err_msg = _merge_error(err_msg, f"log_file_open_failed: {file_error}")
                                    # Best-effort: free any buffers
                                    gc.collect()
                                # If process was killed by OOM, annotate in metrics later
                                dur = time.perf_counter() - t0
                                npz = run_dir / "teacher_preds.npz"
                                metrics_path = run_dir / "model_metrics.json"
                                metrics: dict[str, float] = {}
                                if metrics_path.exists():
                                    try:
                                        metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
                                        metrics = {key: float(value) for key, value in metrics_json.items()}
                                    except Exception as exc:  # pragma: no cover - metrics JSON optional
                                        err_msg = _merge_error(err_msg, f"metrics_json_error: {exc}")
                                if npz.exists():
                                    try:
                                        np_metrics = _score(npz)
                                        metrics.update(np_metrics)
                                    except Exception as exc:  # pragma: no cover
                                        err_msg = _merge_error(err_msg, str(exc))
                                rec: dict[str, Any] = {
                                    "model_id": model_id,
                                    "rc": rc,
                                    "duration_sec": dur,
                                    "hidden_size": hs,
                                    "lstm_layers": ll,
                                    "attention_heads": ah,
                                    "dropout": dr,
                                    "max_encoder_length": mel,
                                    "seed": seed,
                                    "metrics": metrics,
                                    "params": {
                                        "hidden_size": hs,
                                        "lstm_layers": ll,
                                        "attention_heads": ah,
                                        "dropout": dr,
                                        "learning_rate": lr,
                                        "max_encoder_length": mel,
                                        "seed": seed,
                                    },
                                }
                                if err_msg is not None:
                                    rec["error"] = err_msg
                                results.append(rec)

    def metric_value(record: dict[str, Any]) -> float:
        metrics_obj = record.get("metrics", {})
        if isinstance(metrics_obj, Mapping):
            return _resolve_metric_value(metrics_obj, metric_name)
        return 0.0

    if not results:
        summary = {
            "metric": metric_name,
            "direction": direction,
            "backend": effective_backend,
            "best": None,
            "all": results,
        }
    else:
        if direction == "minimize":
            best = min(results, key=metric_value)
        else:
            best = max(results, key=metric_value)
        summary = {
            "metric": metric_name,
            "direction": direction,
            "backend": effective_backend,
            "best": best,
            "all": results,
        }
    print(json.dumps(summary))
    (out / "hpo_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


__all__ = ["HAS_OPTUNA", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
