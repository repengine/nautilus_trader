#!/usr/bin/env python3
"""
Simple HPO sweep for TFT teacher (BCE).

Runs a small grid over model hyperparameters using the existing teacher CLI,
evaluates validation metrics from teacher_preds.npz, and prints a JSON summary
with the best configuration.

Example:
    python -m ml.scripts.hpo_tft \
      --dataset_csv /tmp/tft_universe_60d/merged/dataset.csv \
      --out_dir /tmp/tft_universe_60d/hpo \
      --feature_registry_dir ~/.nautilus/ml/features \
      --feature_set_id <fid> \
      --epochs 2 \
      --workers 4

"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HPO sweep for TFT teacher (BCE)")
    ap.add_argument("--dataset_csv", required=False)
    ap.add_argument(
        "--dataset_parquet",
        required=False,
        help="Path to a Parquet dataset; if provided, supersedes --dataset_csv",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_registry_dir", required=True)
    ap.add_argument("--feature_set_id", required=True)
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
        "--inproc",
        action="store_true",
        help="Run training in-process (default runs each config in a subprocess for memory isolation)",
    )
    ap.add_argument(
        "--precision",
        default="32",
        help="Training precision to pass to teacher CLI (e.g., 32, 16, 16-mixed, bf16)",
    )
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    # Small grid
    hidden_sizes = [32, 64]
    lstm_layers = [2, 3]
    attn_heads = [2, 4]
    dropouts = [0.1, 0.2]
    lrs = [3e-4, 1e-3]

    results: list[dict[str, Any]] = []
    train_args_common = [
        "-m",
        "ml.training.teacher.tft_cli",
        *train_flag,
        train_path,
        "--feature_registry_dir",
        args.feature_registry_dir,
        "--feature_set_id",
        args.feature_set_id,
        "--max_epochs",
        str(args.epochs),
        "--val_days",
        str(int(args.val_days)),
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
    ]
    # Optional dataset capping
    if int(args.tail_rows or 0) > 0:
        train_args_common += ["--tail_rows", str(int(args.tail_rows))]
    if int(args.limit_groups or 0) > 0:
        train_args_common += ["--limit_groups", str(int(args.limit_groups))]

    run_id = 0
    for hs in hidden_sizes:
        for ll in lstm_layers:
            for ah in attn_heads:
                for dr in dropouts:
                    for lr in lrs:
                        run_id += 1
                        model_id = f"tft_hpo_h{hs}_l{ll}_a{ah}_d{str(dr).replace('.', '')}_lr{str(lr).replace('.', '')}_r{run_id}"
                        run_dir = out / model_id
                        run_dir.mkdir(parents=True, exist_ok=True)
                        t0 = time.perf_counter()
                        rc: int
                        if args.inproc:
                            # In-process path (tests/debug): import main and call directly
                            from ml.training.teacher.tft_cli import main as train_main

                            rc = train_main(
                                [
                                    *train_flag,
                                    train_path,
                                    "--out_dir",
                                    str(run_dir),
                                    "--model_id",
                                    model_id,
                                    "--feature_registry_dir",
                                    args.feature_registry_dir,
                                    "--feature_set_id",
                                    args.feature_set_id,
                                    "--max_epochs",
                                    str(args.epochs),
                                    "--val_days",
                                    str(int(args.val_days)),
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
                                + (
                                    []
                                    if int(args.tail_rows or 0) <= 0
                                    else ["--tail_rows", str(int(args.tail_rows))]
                                )
                                + (
                                    []
                                    if int(args.limit_groups or 0) <= 0
                                    else [
                                        "--limit_groups",
                                        str(int(args.limit_groups)),
                                    ]
                                ),
                            )
                        else:
                            # Subprocess isolation to prevent memory from accumulating across runs
                            cmd = [
                                sys.executable,
                                *train_args_common,
                                "--out_dir",
                                str(run_dir),
                                "--model_id",
                                model_id,
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
                            log_path = run_dir / "train.log"
                            try:
                                with open(log_path, "wb") as lf:
                                    proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
                                rc = int(proc.returncode)
                            except Exception:
                                # Fallback to default capture when file open fails (should be rare)
                                proc = subprocess.run(cmd, capture_output=True)
                                rc = int(proc.returncode)
                            # Best-effort: free any buffers
                            gc.collect()
                            # If process was killed by OOM, annotate in metrics later
                        dur = time.perf_counter() - t0
                        npz = run_dir / "teacher_preds.npz"
                        err_msg: str | None = None
                        try:
                            metrics = _score(npz)
                        except Exception as exc:  # pragma: no cover
                            metrics = {}
                            err_msg = str(exc)
                        rec: dict[str, Any] = {
                            "model_id": model_id,
                            "rc": rc,
                            "duration_sec": dur,
                            "hidden_size": hs,
                            "lstm_layers": ll,
                            "attention_heads": ah,
                            "dropout": dr,
                            "metrics": metrics,
                        }
                        if err_msg is not None:
                            rec["error"] = err_msg
                        results.append(rec)

    # Pick best by PRx then AUC
    def keyfn(r: dict[str, Any]) -> tuple[float, float]:
        m = r.get("metrics", {})
        return float(m.get("PRx", 0.0)), float(m.get("AUC", 0.0))

    best = max(results, key=keyfn)
    summary = {"best": best, "all": results}
    print(json.dumps(summary))
    (out / "hpo_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
