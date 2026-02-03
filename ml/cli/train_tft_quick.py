#!/usr/bin/env python3
"""
Thin CLI wrapper for quick TFT training via :mod:`ml.tasks.training`.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid as _uuid
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.config.targets import TargetSemanticsConfig
from ml.tasks.training import QuickTFTTrainConfig
from ml.tasks.training import train_tft_quick
from ml.tasks.training.quick import _DEFAULT_SYMBOLS


__all__ = ["main"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick TFT dataset build and training")
    parser.add_argument(
        "--data-dir",
        action="append",
        type=Path,
        help="Candidate data directories (can be specified multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where CSV and Parquet outputs are written",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to include",
    )
    parser.add_argument(
        "--target-semantics",
        "--target_semantics",
        required=True,
        help="Target semantics JSON (string or path to .json file).",
    )
    parser.add_argument("--lookback-periods", type=int, default=50)
    parser.add_argument("--sample-predictions", type=int, default=10)
    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_symbols(value: str | None) -> Sequence[str]:
    if value is None:
        return _DEFAULT_SYMBOLS
    symbols = [tok.strip().upper() for tok in value.split(",") if tok.strip()]
    if not symbols:
        raise ValueError("At least one symbol must be provided when --symbols is used")
    return symbols


def _parse_target_semantics(value: str) -> TargetSemanticsConfig:
    try:
        return TargetSemanticsConfig.from_json(value)
    except Exception as exc:
        parse_exc = exc
    try:
        path = Path(value)
        if path.exists():
            payload = path.read_text(encoding="utf-8")
            return TargetSemanticsConfig.from_json(payload)
    except OSError as exc:  # pragma: no cover - invalid path payloads
        raise ValueError(f"Invalid target_semantics payload: {exc}") from exc
    raise ValueError(f"Invalid target_semantics payload: {parse_exc}") from parse_exc


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    bind_log_context(
        run_id=f"cli_train_tft_quick_{_uuid.uuid4().hex[:8]}",
        component="ml.cli.train_tft_quick",
    )

    default_fields = QuickTFTTrainConfig.__dataclass_fields__
    default_data_dirs = cast(Sequence[Path], default_fields["data_dirs"].default)
    default_output_dir = cast(Path, default_fields["output_dir"].default)
    candidate_dirs = tuple(args.data_dir) if args.data_dir else default_data_dirs
    try:
        symbols = _parse_symbols(args.symbols)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        target_semantics = _parse_target_semantics(args.target_semantics)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    config = QuickTFTTrainConfig(
        data_dirs=candidate_dirs,
        output_dir=args.output_dir or default_output_dir,
        symbols=symbols,
        target_semantics=target_semantics,
        lookback_periods=args.lookback_periods,
        sample_prediction_count=args.sample_predictions,
    )

    result = train_tft_quick(config)
    summary = {
        "dataset_parquet": str(result.dataset_parquet),
        "dataset_csv": str(result.dataset_csv),
        "dataset_shape": result.dataset_shape,
        "target_distribution": json.loads(result.target_distribution_json),
        "trained": result.trained,
        "sample_predictions": list(result.sample_predictions or []),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
