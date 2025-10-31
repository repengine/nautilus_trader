#!/usr/bin/env python3
"""Recommend the next streaming training wave based on recent manifests."""
from __future__ import annotations

import argparse
import json
import logging
import shlex
from collections.abc import Sequence
from pathlib import Path

from ml.common.subprocess_utils import SubprocessExecutionError
from ml.common.subprocess_utils import run_command
from ml.scripts.summarize_streaming_manifests import ManifestSummary
from ml.scripts.summarize_streaming_manifests import summarize_manifests
from ml.training.event_driven.wave_planner import WaveBounds
from ml.training.event_driven.wave_planner import WaveRecommendation
from ml.training.event_driven.wave_planner import WaveSample
from ml.training.event_driven.wave_planner import recommend_next_wave


logger = logging.getLogger(__name__)


def _to_wave_samples(manifests: Sequence[ManifestSummary]) -> list[WaveSample]:
    samples: list[WaveSample] = []
    for item in manifests:
        samples.append(
            WaveSample(
                completed_at=item.completed_at,
                roc_auc=item.roc_auc,
                pr_auc=item.pr_auc,
                max_gpu_memory_mb=item.peak_gpu_mb,
            ),
        )
    return samples


def _run_validate_wave(manifest_dir: Path, extra_args: Sequence[str]) -> None:
    """Invoke the validation bundle prior to recommending a new wave."""
    command: list[str] = [
        "poetry",
        "run",
        "python",
        "-m",
        "ml.scripts.validate_wave",
        "--manifest-dir",
        str(manifest_dir),
    ]
    command.extend(extra_args)
    logger.info("running validate_wave bundle", extra={"command": shlex.join(command)})
    run_command(command)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("ml_out/tft_streaming_artifacts/full_tft_95"),
        help="Directory containing manifest JSON files.",
    )
    parser.add_argument(
        "--shard-row-budget",
        type=int,
        default=120_000,
        help="Current shard row budget.",
    )
    parser.add_argument(
        "--max-total-rows",
        type=int,
        default=120_000,
        help="Current max_total_rows value.",
    )
    parser.add_argument(
        "--max-total-sequences",
        type=int,
        default=90_000,
        help="Current max_total_sequences value.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=32,
        help="Current max_shards value.",
    )
    parser.add_argument(
        "--row-increment",
        type=int,
        default=30_000,
        help="Row increment for the next wave.",
    )
    parser.add_argument(
        "--shard-increment",
        type=int,
        default=8,
        help="Shard increment for the next wave.",
    )
    parser.add_argument(
        "--device-memory-mb",
        type=float,
        default=6_144.0,
        help="Available GPU memory (MiB).",
    )
    parser.add_argument(
        "--gpu-threshold-ratio",
        type=float,
        default=0.85,
        help="GPU utilisation ratio threshold that triggers a warning.",
    )
    parser.add_argument(
        "--regression-delta",
        type=float,
        default=0.01,
        help="ROC-AUC delta that triggers a regression warning.",
    )
    parser.add_argument(
        "--run-validate-wave",
        action="store_true",
        help="Run ml.scripts.validate_wave before generating a recommendation.",
    )
    parser.add_argument(
        "--validate-wave-args",
        type=str,
        default=None,
        help="Additional arguments forwarded to validate_wave (quote the string).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry-point for the CLI."""
    args = _parse_args(argv)
    extra_args: Sequence[str] = ()
    if args.validate_wave_args:
        extra_args = tuple(shlex.split(args.validate_wave_args))
    if args.run_validate_wave:
        try:
            _run_validate_wave(args.manifest_dir, extra_args)
        except SubprocessExecutionError as exc:
            logger.error(
                "validate_wave_failed",
                extra={
                    "command": shlex.join(exc.command),
                    "returncode": exc.returncode,
                },
                exc_info=True,
            )
            return exc.returncode or 1
    manifests = summarize_manifests(args.manifest_dir, limit=None)
    samples = _to_wave_samples(manifests)
    current_bounds = WaveBounds(
        shard_row_budget=args.shard_row_budget,
        max_total_rows=args.max_total_rows,
        max_total_sequences=args.max_total_sequences,
        max_shards=args.max_shards,
    )
    recommendation = recommend_next_wave(
        samples,
        current_bounds,
        row_increment=args.row_increment,
        shard_increment=args.shard_increment,
        device_memory_mb=args.device_memory_mb,
        gpu_threshold_ratio=args.gpu_threshold_ratio,
        regression_delta=args.regression_delta,
    )
    print(_serialize(recommendation))
    return 0


def _serialize(recommendation: WaveRecommendation) -> str:
    payload = {
        "current": {
            "shard_row_budget": recommendation.current.shard_row_budget,
            "max_total_rows": recommendation.current.max_total_rows,
            "max_total_sequences": recommendation.current.max_total_sequences,
            "max_shards": recommendation.current.max_shards,
        },
        "proposed": {
            "shard_row_budget": recommendation.proposed.shard_row_budget,
            "max_total_rows": recommendation.proposed.max_total_rows,
            "max_total_sequences": recommendation.proposed.max_total_sequences,
            "max_shards": recommendation.proposed.max_shards,
        },
        "notes": recommendation.notes,
        "warnings": recommendation.warnings,
    }
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
