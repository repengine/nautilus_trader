#!/usr/bin/env python3
"""
Ensemble multiple teacher_preds.npz files by averaging calibrated probabilities.

Reads subdirectories under an input directory (each expected to contain
teacher_preds.npz produced by the TFT teacher CLI), averages q_val (and q_train
when present), verifies label alignment, and writes an aggregated teacher_preds.npz
to the output path. Also writes a small JSON summary.

Usage:
    python -m ml.cli.ensemble_teacher_preds \
      --runs_dir /path/to/hpo_runs \
      --out_path /path/to/ensemble/teacher_preds.npz
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def _load_preds(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out: dict[str, Any] = {}
    for key in ("q_val", "q_train", "y_val_true"):
        if key in data:
            out[key] = data[key]
    return out


def _collect_runs(runs_dir: Path) -> list[Path]:
    return [p / "teacher_preds.npz" for p in runs_dir.iterdir() if (p / "teacher_preds.npz").exists()]


def _avg_safe(arrs: Sequence[npt.NDArray[np.float64] | npt.NDArray[np.float32]]) -> npt.NDArray[np.float64]:
    stack: npt.NDArray[np.float64] = np.stack([np.asarray(a, dtype=np.float64).reshape(-1) for a in arrs], axis=0)
    return np.asarray(stack.mean(axis=0), dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Average ensemble for teacher predictions")
    ap.add_argument("--runs_dir", required=True, help="Directory containing subdirs with teacher_preds.npz")
    ap.add_argument("--out_path", required=True, help="Path to write ensembled teacher_preds.npz")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_paths = _collect_runs(runs_dir)
    if not pred_paths:
        print("No teacher_preds.npz found under runs_dir", flush=True)
        return 2

    q_val_list: list[npt.NDArray[np.float64]] = []
    q_train_list: list[npt.NDArray[np.float64]] = []
    y_ref: npt.NDArray[np.float64] | None = None
    lens: list[int] = []

    for p in pred_paths:
        d = _load_preds(p)
        if "q_val" not in d or "y_val_true" not in d:
            continue
        qv = np.asarray(d["q_val"], dtype=np.float64).reshape(-1)
        yv = np.asarray(d["y_val_true"], dtype=np.float64).reshape(-1)
        # Align lengths conservatively by truncating to min length across runs
        if y_ref is None:
            y_ref = yv
        lens.append(min(len(qv), len(yv)))
        q_val_list.append(qv)
        if "q_train" in d:
            q_train_list.append(np.asarray(d["q_train"], dtype=np.float64).reshape(-1))

    if not q_val_list or y_ref is None:
        print("No valid q_val/y_val_true found to ensemble", flush=True)
        return 3

    min_len = int(min(lens))
    q_val_list = [a[-min_len:] for a in q_val_list]
    y_val = y_ref[-min_len:]
    q_val_avg = _avg_safe(q_val_list)

    payload: dict[str, Any] = {"q_val": q_val_avg.astype(np.float32), "y_val_true": y_val.astype(np.float64)}
    if q_train_list:
        min_len_tr = int(min(len(a) for a in q_train_list))
        payload["q_train"] = _avg_safe([a[-min_len_tr:] for a in q_train_list]).astype(np.float32)

    np.savez_compressed(out_path, **payload)
    summary = {
        "runs": len(q_val_list),
        "min_len": min_len,
        "out_path": str(out_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
