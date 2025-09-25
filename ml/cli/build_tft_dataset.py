#!/usr/bin/env python3
"""
CLI wrapper for building TFT datasets via :mod:`ml.tasks.datasets`.
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid as _uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.data.vintage import VintagePolicy
from ml.tasks.datasets import TFTDatasetTaskConfig
from ml.tasks.datasets import build_tft_dataset


LOGGER = logging.getLogger(__name__)


def _parse_symbols(value: str) -> list[str]:
    symbols = [item.strip() for item in value.split(",") if item.strip()]
    if not symbols:
        msg = "At least one symbol is required"
        raise argparse.ArgumentTypeError(msg)
    return symbols


def _parse_optional_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse ensures format
        msg = f"Invalid date '{value}'. Expected YYYY-MM-DD."
        raise argparse.ArgumentTypeError(msg) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build TFT dataset artifacts")
    parser.add_argument("--data_dir", default="data/tier1")
    parser.add_argument("--symbols", required=True, type=_parse_symbols)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--horizon_minutes", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--lookback_periods", type=int, default=30)
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--chunk_days", type=int, default=0)
    parser.add_argument("--macro_lag_days", type=int, default=1)
    parser.add_argument("--include_micro", action="store_true")
    parser.add_argument("--include_l2", action="store_true")
    parser.add_argument("--include_events", action="store_true")
    parser.add_argument("--include_calendar", action="store_true")
    parser.add_argument("--student_mode", action="store_true")
    parser.add_argument("--emit_dataset_events", action="store_true")
    parser.add_argument("--fred_vintage_dir")
    parser.add_argument("--events_dir")
    parser.add_argument("--register_features", action="store_true")
    parser.add_argument("--feature_registry_dir")
    parser.add_argument(
        "--feature_role",
        choices=["teacher", "student", "inference_support"],
        default="teacher",
    )
    parser.add_argument("--market_dataset_id")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--no_macro", action="store_true", help="Disable FRED macro join")
    parser.add_argument(
        "--vintage_policy",
        default=VintagePolicy.REAL_TIME.value,
        choices=[policy.value for policy in VintagePolicy],
        help="Vintage policy for macro features (real_time or final)",
    )
    parser.add_argument(
        "--vintage_as_of",
        help="ISO8601 timestamp limiting macro revisions (optional)",
        default=None,
    )
    args = parser.parse_args(argv)

    if args.verbose or os.environ.get("ML_DEBUG"):
        configure_logging(level="DEBUG")
    else:
        configure_logging()
    run_id = f"cli_build_tft_dataset_{_uuid.uuid4().hex[:8]}"
    bind_log_context(run_id=run_id, component="ml.cli.build_tft_dataset")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        vintage_policy = VintagePolicy(args.vintage_policy)
    except ValueError as exc:  # pragma: no cover - guarded by argparse choices but defensive
        raise SystemExit(f"Invalid vintage_policy: {args.vintage_policy}") from exc

    cfg = TFTDatasetTaskConfig(
        data_dir=Path(args.data_dir),
        out_dir=out_dir,
        symbols=args.symbols,
        horizon_minutes=args.horizon_minutes,
        threshold=args.threshold,
        lookback_periods=args.lookback_periods,
        include_macro=not bool(args.no_macro),
        macro_lag_days=args.macro_lag_days,
        include_micro=args.include_micro,
        include_l2=args.include_l2,
        include_events=args.include_events,
        include_calendar=args.include_calendar,
        chunk_days=args.chunk_days,
        start=_parse_optional_date(args.start),
        end=_parse_optional_date(args.end),
        register_features=args.register_features,
        feature_registry_dir=Path(args.feature_registry_dir) if args.feature_registry_dir else None,
        feature_role=args.feature_role,
        emit_dataset_events=args.emit_dataset_events,
        fred_vintage_dir=Path(args.fred_vintage_dir) if args.fred_vintage_dir else None,
        events_base_dir=Path(args.events_dir) if args.events_dir else None,
        student_mode=args.student_mode,
        market_dataset_id=args.market_dataset_id,
        vintage_policy=vintage_policy,
        vintage_as_of=_parse_optional_date(args.vintage_as_of),
    )

    LOGGER.info("Building TFT dataset %s", cfg)
    result = build_tft_dataset(cfg)

    print(
        "Saved dataset to"
        f" {result.dataset_parquet} and {result.dataset_csv}\nSaved features to {result.features_npz}",
    )
    if result.feature_set_id:
        print(f"Registered feature set: {result.feature_set_id} in {cfg.feature_registry_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
