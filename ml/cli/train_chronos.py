#!/usr/bin/env python3
"""
Thin CLI wrapper for Chronos foundation model training.

This CLI provides a convenient interface to train Chronos models
for time series forecasting using AutoGluon TimeSeries.

Usage:
    python -m ml.cli.train_chronos --symbols SPY,AAPL --preset chronos2
    python -m ml.cli.train_chronos --symbols SPY --preset bolt_small --time-limit 600
    python -m ml.cli.train_chronos --symbols SPY,QQQ --distill

"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from ml.common.logging_config import configure_logging


__all__ = ["main"]


# Ensure parquet fallback is enabled
os.environ.setdefault("ML_TFT_ALLOW_PARQUET_FALLBACK", "1")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command-line arguments. Uses sys.argv if None.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.

    """
    parser = argparse.ArgumentParser(
        description="Train Chronos foundation models for time series forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated list of symbols (e.g., SPY,AAPL,MSFT)",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tier1"),
        help="Directory containing parquet data (default: data/tier1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for models and results",
    )

    # Training arguments
    parser.add_argument(
        "--preset",
        type=str,
        default="chronos2",
        choices=["chronos2", "bolt_small", "bolt_tiny", "bolt_mini", "chronos_small", "chronos_base"],
        help="Chronos model preset (default: chronos2)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=1800,
        help="Training time limit in seconds (default: 1800)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=15,
        help="Forecast horizon in minutes (default: 15)",
    )
    parser.add_argument(
        "--target-semantics",
        "--target_semantics",
        required=True,
        help="Target semantics JSON (string or path to .json file).",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default="RMSE",
        choices=["RMSE", "MAE", "MASE", "MAPE"],
        help="Evaluation metric (default: RMSE)",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensembling during model selection",
    )
    parser.add_argument(
        "--num-val-windows",
        type=int,
        default=1,
        help="Number of validation windows for tuning (default: 1)",
    )
    parser.add_argument(
        "--refit-every-n-windows",
        type=int,
        default=1,
        help="Refit cadence for rolling windows (default: 1)",
    )
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help="Refit best model on full dataset after tuning",
    )
    parser.add_argument(
        "--skip-model-selection",
        action="store_true",
        help="Skip model selection/tuning (train a single model)",
    )
    parser.add_argument(
        "--tune-num-trials",
        type=int,
        default=None,
        help="Enable AutoGluon HPO with the specified number of trials",
    )
    parser.add_argument(
        "--tune-scheduler",
        type=str,
        default=None,
        help="AutoGluon scheduler for HPO (e.g., local, ray)",
    )
    parser.add_argument(
        "--tune-searcher",
        type=str,
        default=None,
        help="AutoGluon searcher for HPO (e.g., random, bayes)",
    )

    # Feature arguments
    parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Lookback periods for feature computation (default: 120)",
    )
    parser.add_argument("--no-macro", action="store_true", help="Disable macro features")
    parser.add_argument("--no-calendar", action="store_true", help="Disable calendar features")
    parser.add_argument("--no-earnings", action="store_true", help="Disable earnings features")
    parser.add_argument("--include-micro", action="store_true", help="Include microstructure features")

    # Distillation arguments
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Enable teacher-student distillation (chronos2 -> bolt_small)",
    )
    parser.add_argument(
        "--teacher-preset",
        type=str,
        default="chronos2",
        help="Teacher model preset for distillation (default: chronos2)",
    )
    parser.add_argument(
        "--student-preset",
        type=str,
        default="bolt_small",
        help="Student model preset for distillation (default: bolt_small)",
    )
    parser.add_argument(
        "--teacher-time-limit",
        type=int,
        default=3600,
        help="Teacher training time limit (default: 3600)",
    )
    parser.add_argument(
        "--student-time-limit",
        type=int,
        default=1800,
        help="Student training time limit (default: 1800)",
    )
    parser.add_argument(
        "--distill-forecast-step",
        type=int,
        default=None,
        help="Forecast step for soft label alignment (1=next step)",
    )
    parser.add_argument(
        "--distill-min-history",
        type=int,
        default=None,
        help="Minimum history length before generating soft labels",
    )
    parser.add_argument(
        "--distill-stride",
        type=int,
        default=None,
        help="Stride between rolling forecast cutoffs",
    )
    parser.add_argument(
        "--distill-max-windows",
        type=int,
        default=None,
        help="Max rolling windows per series for soft labels",
    )
    parser.add_argument(
        "--distill-max-series",
        type=int,
        default=None,
        help="Max number of series to include in distillation",
    )
    parser.add_argument(
        "--distill-sample-fraction",
        type=float,
        default=None,
        help="Sample fraction of windows per series (0 < f <= 1)",
    )
    parser.add_argument(
        "--distill-window-strategy",
        type=str,
        default=None,
        help="Window sampling strategy for distillation (uniform or contiguous)",
    )
    parser.add_argument(
        "--distill-min-coverage",
        type=float,
        default=None,
        help="Minimum coverage fraction required for soft labels",
    )

    # Hardware arguments
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")

    # Output arguments
    parser.add_argument("--no-soft-labels", action="store_true", help="Skip soft label export")
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -vv for debug)",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_symbols(value: str) -> list[str]:
    """
    Parse a comma-separated symbol list.

    Parameters
    ----------
    value : str
        Comma-separated symbol list.

    Returns
    -------
    list[str]
        Normalized symbols.

    """
    symbols = [tok.strip().upper() for tok in value.split(",") if tok.strip()]
    if not symbols:
        raise ValueError("At least one symbol must be provided")
    return symbols


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point for Chronos training CLI.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command-line arguments. Uses sys.argv if None.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).

    """
    args = parse_args(argv)

    # Configure logging
    verbosity = min(args.verbose + 2, 3)  # 2=INFO, 3=DEBUG
    configure_logging(level="DEBUG" if verbosity >= 3 else "INFO")

    # Import here to avoid slow startup
    from ml.experiments.chronos_training_experiment import main as run_experiment

    # Validate symbols
    _parse_symbols(args.symbols)  # Will raise if invalid

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        preset_name = "distill" if args.distill else args.preset
        output_dir = Path(f"reports/experiments/chronos_{preset_name}")

    # Build experiment argv for the experiment main()
    # Note: We pass args directly to avoid redundant config creation
    experiment_argv = [
        f"--symbols={args.symbols}",
        f"--data_dir={args.data_dir}",
        f"--out_dir={output_dir}",
        f"--preset={args.preset}",
        f"--time_limit={args.time_limit}",
        f"--horizon={args.horizon}",
        f"--target-semantics={args.target_semantics}",
        f"--eval_metric={args.eval_metric}",
        f"--lookback={args.lookback}",
        f"--num_gpus={args.num_gpus}",
        f"--verbosity={verbosity}",
        f"--num_val_windows={args.num_val_windows}",
        f"--refit_every_n_windows={args.refit_every_n_windows}",
    ]
    if args.tune_num_trials is not None:
        experiment_argv.append(f"--tune_num_trials={args.tune_num_trials}")
    if args.tune_scheduler is not None:
        experiment_argv.append(f"--tune_scheduler={args.tune_scheduler}")
    if args.tune_searcher is not None:
        experiment_argv.append(f"--tune_searcher={args.tune_searcher}")

    if args.distill:
        experiment_argv.append("--distill")
        experiment_argv.append(f"--teacher_preset={args.teacher_preset}")
        experiment_argv.append(f"--student_preset={args.student_preset}")
        experiment_argv.append(f"--teacher_time_limit={args.teacher_time_limit}")
        experiment_argv.append(f"--student_time_limit={args.student_time_limit}")
        if args.distill_forecast_step is not None:
            experiment_argv.append(f"--distill_forecast_step={args.distill_forecast_step}")
        if args.distill_min_history is not None:
            experiment_argv.append(f"--distill_min_history={args.distill_min_history}")
        if args.distill_stride is not None:
            experiment_argv.append(f"--distill_stride={args.distill_stride}")
        if args.distill_max_windows is not None:
            experiment_argv.append(
                f"--distill_max_windows_per_series={args.distill_max_windows}"
            )
        if args.distill_max_series is not None:
            experiment_argv.append(f"--distill_max_series={args.distill_max_series}")
        if args.distill_sample_fraction is not None:
            experiment_argv.append(
                f"--distill_sample_fraction={args.distill_sample_fraction}"
            )
        if args.distill_window_strategy is not None:
            experiment_argv.append(
                f"--distill_window_sampling_strategy={args.distill_window_strategy}"
            )
        if args.distill_min_coverage is not None:
            experiment_argv.append(
                f"--distill_min_soft_label_coverage={args.distill_min_coverage}"
            )

    if args.no_macro:
        experiment_argv.append("--no_macro")
    if args.no_calendar:
        experiment_argv.append("--no_calendar")
    if args.no_earnings:
        experiment_argv.append("--no_earnings")
    if args.include_micro:
        experiment_argv.append("--include_micro")
    if args.no_ensemble:
        experiment_argv.append("--no_ensemble")
    if args.refit_full:
        experiment_argv.append("--refit_full")
    if args.skip_model_selection:
        experiment_argv.append("--skip_model_selection")
    if args.cpu_only:
        experiment_argv.append("--cpu_only")
    if args.no_soft_labels:
        experiment_argv.append("--no_soft_labels")

    return run_experiment(experiment_argv)


if __name__ == "__main__":
    sys.exit(main())
