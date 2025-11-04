#!/usr/bin/env python3
"""Produce replay and sweep templates for collapsed streaming cohorts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml.scripts.find_collapsed_streaming_cohorts import run_inspection


def _coerce_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    return value.expanduser().resolve()


def _build_replay_command(
    *,
    dataset_dir: Path | None,
    output_dir: Path | None,
    state_path: Path | None,
    dataset_seed: int | None,
    worker_seed: int | None,
    shard_row_budget: int | None,
) -> list[str] | None:
    if dataset_dir is None or output_dir is None or state_path is None:
        return None
    command = [
        "poetry",
        "run",
        "python",
        "-m",
        "ml.cli.streaming_training_runner",
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_dir),
        "--state-path",
        str(state_path),
        "--max-plans",
        "1",
    ]
    if dataset_seed is not None:
        command.extend(["--dataset-seed", str(dataset_seed)])
    if worker_seed is not None:
        command.extend(["--worker-seed", str(worker_seed)])
    if shard_row_budget is not None and shard_row_budget > 0:
        command.extend(["--shard-row-budget", str(shard_row_budget)])
    return command


def _build_sweep_command(
    *,
    include_sweep: bool,
    dataset_dir: Path | None,
    output_dir: Path | None,
    state_path: Path | None,
    dataset_seed: int | None,
    worker_seed: int | None,
    sweep_max_trials: int,
) -> list[str] | None:
    if not include_sweep:
        return None
    if dataset_dir is None or output_dir is None or state_path is None:
        return None
    command = [
        "poetry",
        "run",
        "python",
        "-m",
        "ml.cli.streaming_training_runner",
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_dir),
        "--state-path",
        str(state_path),
        "--run-worker-sweep",
        "--sweep-max-trials",
        str(sweep_max_trials),
    ]
    if dataset_seed is not None:
        command.extend(["--dataset-seed", str(dataset_seed)])
    if worker_seed is not None:
        command.extend(["--worker-seed", str(worker_seed)])
    return command


def _sort_collapsed(collapsed: list[Mapping[str, Any]], *, sort_key: str) -> list[Mapping[str, Any]]:
    def _key(entry: Mapping[str, Any]) -> tuple[float, float]:
        sharpe = float(entry.get("sharpe") or float("inf"))
        z_std = float(entry.get("z_val_std") or float("inf"))
        if sort_key == "z_val_std":
            primary = z_std
            secondary = sharpe
        else:
            primary = sharpe
            secondary = z_std
        return primary, secondary

    return sorted(collapsed, key=_key)


def _build_plan_entry(
    entry: Mapping[str, Any],
    *,
    replay_command: list[str] | None,
    sweep_command: list[str] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "plan_id": entry.get("plan_id"),
        "manifest_path": entry.get("manifest_path"),
        "logits_path": entry.get("logits_path"),
        "z_val_std": entry.get("z_val_std"),
        "sharpe": entry.get("sharpe"),
        "hit_rate": entry.get("hit_rate"),
        "worker_curriculum_enabled": entry.get("worker_curriculum_enabled"),
        "worker_amp_enabled": entry.get("worker_amp_enabled"),
    }
    if replay_command is not None:
        payload["replay_command"] = replay_command
    if sweep_command is not None:
        payload["sweep_command"] = sweep_command
    return payload


def plan_collapsed_replays(
    *,
    manifest_dir: Path,
    dataset_dir: Path | None,
    output_dir: Path | None,
    state_path: Path | None,
    dataset_seed: int | None,
    worker_seed: int | None,
    shard_row_budget: int | None,
    sweep_max_trials: int,
    include_sweep: bool,
    top_n: int,
    z_threshold: float,
    sharpe_threshold: float,
    sort_by: str,
) -> dict[str, Any]:
    inspection = run_inspection(manifest_dir, z_threshold=z_threshold, sharpe_threshold=sharpe_threshold)
    collapsed = inspection.get("collapsed", [])
    sorted_collapsed = _sort_collapsed(collapsed, sort_key=sort_by)
    selected = sorted_collapsed[:top_n] if top_n > 0 else sorted_collapsed

    dataset_dir_resolved = _coerce_path(dataset_dir)
    output_dir_resolved = _coerce_path(output_dir)
    state_path_resolved = _coerce_path(state_path)

    entries = []
    for entry in selected:
        replay_command = _build_replay_command(
            dataset_dir=dataset_dir_resolved,
            output_dir=output_dir_resolved,
            state_path=state_path_resolved,
            dataset_seed=dataset_seed,
            worker_seed=worker_seed,
            shard_row_budget=shard_row_budget,
        )
        sweep_command = _build_sweep_command(
            include_sweep=include_sweep,
            dataset_dir=dataset_dir_resolved,
            output_dir=output_dir_resolved,
            state_path=state_path_resolved,
            dataset_seed=dataset_seed,
            worker_seed=worker_seed,
            sweep_max_trials=sweep_max_trials,
        )
        entries.append(
            _build_plan_entry(entry, replay_command=replay_command, sweep_command=sweep_command),
        )

    return {
        "manifest_dir": str(manifest_dir),
        "collapsed_count": inspection.get("collapsed_count", 0),
        "total_count": inspection.get("total_count", 0),
        "z_val_std_threshold": z_threshold,
        "sharpe_threshold": sharpe_threshold,
        "sort_by": sort_by,
        "top_n": top_n,
        "tasks": entries,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate replay/sweep templates for collapsed streaming cohorts.")
    parser.add_argument("--manifest-dir", type=Path, required=True, help="Directory containing streaming manifests.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset directory passed to streaming runner.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for replay artifacts.")
    parser.add_argument("--state-path", type=Path, default=None, help="Streaming state snapshot used by the runner.")
    parser.add_argument("--dataset-seed", type=int, default=None, help="Dataset seed supplied to replay commands.")
    parser.add_argument("--worker-seed", type=int, default=None, help="Worker seed supplied to replay commands.")
    parser.add_argument("--shard-row-budget", type=int, default=None, help="Optional shard row budget override.")
    parser.add_argument("--sweep-max-trials", type=int, default=10, help="Sweep trials when --include-sweep is set.")
    parser.add_argument("--include-sweep", action="store_true", help="Emit worker sweep commands alongside replays.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of collapsed cohorts to include (<=0 for all).")
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=0.05,
        help="Std-dev threshold for z_val below which a cohort is considered collapsed.",
    )
    parser.add_argument(
        "--sharpe-threshold",
        type=float,
        default=0.0,
        help="Sharpe threshold below which a cohort is considered collapsed.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("sharpe", "z_val_std"),
        default="sharpe",
        help="Ranking criteria for collapsed cohorts (default: sharpe).",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plan = plan_collapsed_replays(
        manifest_dir=args.manifest_dir.resolve(),
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        state_path=args.state_path,
        dataset_seed=args.dataset_seed,
        worker_seed=args.worker_seed,
        shard_row_budget=args.shard_row_budget,
        sweep_max_trials=int(args.sweep_max_trials),
        include_sweep=bool(args.include_sweep),
        top_n=int(args.top_n),
        z_threshold=float(args.z_threshold),
        sharpe_threshold=float(args.sharpe_threshold),
        sort_by=str(args.sort_by),
    )
    indent = 2 if args.pretty else None
    print(json.dumps(plan, indent=indent, sort_keys=bool(args.pretty)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
