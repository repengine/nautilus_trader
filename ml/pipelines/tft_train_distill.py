#!/usr/bin/env python3
"""
Compatibility wrapper around the unified pipeline orchestrator for TFT teacher + student runs.

The legacy pipeline staged dataset build, teacher training, and student distillation
manually. This wrapper now forwards arguments to ``ml.orchestration.pipeline_orchestrator``
so that a single orchestrator path handles registration, promotions, and validation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml.orchestration.pipeline_orchestrator import main as orchestrator_main


LOGGER = logging.getLogger(__name__)


def build_orchestrator_args(args: argparse.Namespace) -> list[str]:
    cli_args: list[str] = [
        "--data_dir",
        args.data_dir,
        "--symbols",
        args.symbols,
        "--out_dir",
        args.out_dir,
        "--catalog_path",
        args.data_dir,
        "--dataset_id",
        "tft_dataset",
        "--horizon_minutes",
        str(args.horizon_minutes),
        "--threshold",
        str(args.threshold),
        "--lookback_periods",
        str(args.lookback_periods),
    ]
    if args.include_macro:
        cli_args += ["--include_macro", "--macro_lag_days", str(args.macro_lag_days)]
    if args.include_micro:
        cli_args += ["--include_micro"]
    if args.include_l2:
        cli_args += ["--include_l2", "--skip_l2_ingest", "--auto_fill_skip_l2"]
    if args.include_events:
        cli_args += ["--include_events"]
    if args.include_calendar:
        cli_args += ["--include_calendar"]
    if args.feature_registry_dir:
        cli_args += ["--feature_registry_dir", args.feature_registry_dir]
    if args.feature_set_id:
        cli_args += ["--feature_set_id", args.feature_set_id]
    if args.register_features:
        cli_args += ["--dataset_register_features"]

    if args.train_teacher:
        cli_args += [
            "--train",
            "--teacher_model_id",
            args.teacher_model_id,
        ]
    else:
        LOGGER.info(
            "Skipping teacher training; existing predictions at %s must be present",
            Path(args.out_dir) / "teacher_preds.npz",
        )

    cli_args += [
        "--distill_student",
        "--student_model_id",
        args.student_model_id,
        "--student_model_registry_dir",
        args.model_registry_dir,
    ]
    if args.student_parent_model_id:
        cli_args += ["--student_parent_model_id", args.student_parent_model_id]
    else:
        cli_args += ["--student_parent_model_id", args.teacher_model_id]
    if args.student_feature_registry_dir:
        cli_args += ["--student_feature_registry_dir", args.student_feature_registry_dir]
    if args.student_feature_set_id:
        cli_args += ["--student_feature_set_id", args.student_feature_set_id]
    if args.student_objective != "logit_mse":
        cli_args += ["--student_objective", args.student_objective]
    if args.student_kd_lambda != 0.5:
        cli_args += ["--student_kd_lambda", str(args.student_kd_lambda)]
    if args.student_use_val_for_distill:
        cli_args += ["--student_use_val_for_distill"]
    cli_args += [
        "--student_early_stopping",
        str(args.student_early_stopping),
    ]
    if args.student_opset is not None:
        cli_args += ["--student_opset", str(args.student_opset)]

    return [str(value) for value in cli_args]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/tier1")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--include_macro", action="store_true")
    parser.add_argument("--macro_lag_days", type=int, default=1)
    parser.add_argument("--include_micro", action="store_true")
    parser.add_argument("--include_l2", action="store_true")
    parser.add_argument("--include_events", action="store_true")
    parser.add_argument("--include_calendar", action="store_true")
    parser.add_argument("--horizon_minutes", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--lookback_periods", type=int, default=30)
    parser.add_argument("--feature_registry_dir", default=None)
    parser.add_argument("--feature_set_id", default=None)
    parser.add_argument("--register_features", action="store_true")
    parser.add_argument("--train_teacher", action="store_true")
    parser.add_argument("--teacher_model_id", required=True)
    parser.add_argument("--model_registry_dir", required=True)
    parser.add_argument("--student_model_id", required=True)
    parser.add_argument("--student_parent_model_id", default=None)
    parser.add_argument("--student_feature_registry_dir", default=None)
    parser.add_argument("--student_feature_set_id", default=None)
    parser.add_argument(
        "--student_objective",
        default="logit_mse",
        choices=["logit_mse", "soft_ce", "hybrid"],
    )
    parser.add_argument("--student_kd_lambda", type=float, default=0.5)
    parser.add_argument("--student_early_stopping", type=int, default=200)
    parser.add_argument("--student_opset", type=int, default=None)
    parser.add_argument("--student_use_val_for_distill", action="store_true")
    parser.add_argument(
        "--emit_teacher_predictions",
        action="store_true",
        help="Deprecated: orchestrator no longer generates placeholder logits",
    )
    args = parser.parse_args(argv)

    if args.emit_teacher_predictions:
        LOGGER.warning("emit_teacher_predictions is deprecated; ignoring flag")

    cli_args = build_orchestrator_args(args)
    return orchestrator_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
