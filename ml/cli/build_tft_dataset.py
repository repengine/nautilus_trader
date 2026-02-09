#!/usr/bin/env python3
"""
CLI entrypoint for building TFT datasets.
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid as _uuid
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import fields
from datetime import datetime
from pathlib import Path

from ml.common.cli_parsers import parse_market_inputs_json
from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.config.market_data import MarketDatasetInput
from ml.config.targets import TargetSemanticsConfig
from ml.data import BuildResult
from ml.data import TFTDatasetTaskConfig
from ml.data import build_tft_dataset_from_task_config as build_tft_dataset
from ml.data.vintage import VintagePolicy
from ml.stores.data_store import DataStore
from ml.stores.feature_raw_writer import FeatureDatasetParquetRawWriter


LOGGER = logging.getLogger(__name__)

BuildFn = Callable[..., BuildResult]


def _parse_symbols(value: str) -> list[str]:
    """
    Parse comma-separated symbols into a list.
    """
    symbols = [item.strip() for item in value.split(",") if item.strip()]
    if not symbols:
        msg = "At least one symbol is required"
        raise argparse.ArgumentTypeError(msg)
    return symbols


def _parse_optional_date(value: str | None) -> datetime | None:
    """
    Parse ISO date strings when provided.
    """
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse ensures format
        msg = f"Invalid date '{value}'. Expected YYYY-MM-DD."
        raise argparse.ArgumentTypeError(msg) from exc


def _parse_market_inputs(value: str | None) -> tuple[MarketDatasetInput, ...] | None:
    """
    Parse market input JSON payloads into `MarketDatasetInput` entries.
    """
    return parse_market_inputs_json(value)


def _resolve_dsn(args: argparse.Namespace) -> str:
    """
    Resolve DSN from CLI args and environment.
    """
    candidates = (
        getattr(args, "dsn", None),
        os.getenv("FEATURE_STORE_CONNECTION"),
        os.getenv("DB_CONNECTION"),
        os.getenv("NAUTILUS_DB"),
        os.getenv("DATABASE_URL"),
    )
    for candidate in candidates:
        if candidate:
            return candidate
    msg = (
        "Database connection string required (use --dsn or set FEATURE_STORE_CONNECTION,"
        " DB_CONNECTION, or DATABASE_URL)."
    )
    raise SystemExit(msg)


def _parse_target_semantics(value: str) -> TargetSemanticsConfig:
    """
    Parse target semantics JSON from string or file path.
    """
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
        raise SystemExit(f"Invalid target_semantics payload: {exc}") from exc
    raise SystemExit(f"Invalid target_semantics payload: {parse_exc}") from parse_exc


def _main_with_dependencies(
    argv: Sequence[str] | None = None,
    *,
    build_fn: BuildFn = build_tft_dataset,
    data_store_cls: type[DataStore] = DataStore,
) -> int:
    """
    CLI entrypoint for building TFT datasets.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv`` parsing).
        build_fn: Injectable build function for tests or alternate flows.
        data_store_cls: DataStore factory/class for dependency injection.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="Build TFT dataset artifacts")
    parser.add_argument("--data_dir", default="data/tier1")
    parser.add_argument("--symbols", required=True, type=_parse_symbols)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--target-semantics",
        "--target_semantics",
        required=True,
        help="Target semantics JSON (string or path to .json file).",
    )
    parser.add_argument("--lookback_periods", type=int, default=30)
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--chunk_days", type=int, default=0)
    parser.add_argument(
        "--write_csv",
        action="store_true",
        help="Always write dataset.csv (overrides size-based defaults)",
    )
    parser.add_argument(
        "--skip_csv",
        action="store_true",
        help="Skip writing dataset.csv (optional dataset_sample.csv still possible)",
    )
    parser.add_argument(
        "--csv_max_rows",
        type=int,
        default=None,
        help="Row threshold for auto CSV writing (ignored when --write_csv/--skip_csv set)",
    )
    parser.add_argument(
        "--csv_sample_rows",
        type=int,
        default=0,
        help="Write dataset_sample.csv with N rows when full CSV is skipped",
    )
    parser.add_argument("--macro_lag_days", type=int, default=1)
    parser.add_argument("--include_micro", action="store_true")
    parser.add_argument("--include_l2", action="store_true")
    parser.add_argument("--include_events", action="store_true")
    parser.add_argument("--include_calendar", action="store_true")
    parser.add_argument("--include-macro-deltas", "--include_macro_deltas", action="store_true")
    parser.add_argument("--include-calendar-lags", "--include_calendar_lags", action="store_true")
    parser.add_argument("--include-clustering-tags", "--include_clustering_tags", action="store_true")
    parser.add_argument("--include-context-features", "--include_context_features", action="store_true")
    parser.add_argument("--include-earnings", "--include_earnings", action="store_true")
    parser.add_argument("--earnings-lag-days", "--earnings_lag_days", type=int, default=1)
    parser.add_argument(
        "--micro-base-dir",
        default=os.environ.get("ML_STREAMING_MICRO_BASE_DIR"),
        help="Override directory for microstructure parquet cache (default data_dir or env).",
    )
    parser.add_argument(
        "--l2-base-dir",
        default=os.environ.get("ML_STREAMING_L2_BASE_DIR"),
        help="Override directory for L2 cache (default data_dir or env).",
    )
    parser.add_argument("--student_mode", action="store_true")
    parser.add_argument("--emit_dataset_events", action="store_true")
    parser.add_argument("--fred_vintage_dir")
    parser.add_argument("--events_dir")
    parser.add_argument("--register_features", action="store_true")
    parser.add_argument("--feature_registry_dir")
    parser.add_argument(
        "--dsn",
        help="PostgreSQL DSN for SQL ingestion (default: FEATURE_STORE_CONNECTION / DB_CONNECTION / DATABASE_URL)",
    )
    parser.add_argument(
        "--convert-vintage-age",
        action="store_true",
        help="Convert *_value_vintage_ts columns into *_vintage_age_minutes after build.",
    )
    parser.add_argument(
        "--feature_role",
        choices=["teacher", "student", "inference_support"],
        default="teacher",
    )
    parser.add_argument("--market_dataset_id")
    parser.add_argument(
        "--market_inputs_json",
        help="JSON payload describing market feed inputs",
    )
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
    events_dir = Path(args.events_dir) if args.events_dir else None
    micro_base_dir = Path(args.micro_base_dir) if args.micro_base_dir else None
    l2_base_dir = Path(args.l2_base_dir) if args.l2_base_dir else None
    raw_writer = FeatureDatasetParquetRawWriter(
        events_path=events_dir,
        micro_base_dir=micro_base_dir,
        l2_base_dir=l2_base_dir,
    )
    dsn = _resolve_dsn(args)
    data_store = data_store_cls(
        connection_string=dsn,
        raw_writer=raw_writer,
    )

    try:
        vintage_policy = VintagePolicy(args.vintage_policy)
    except ValueError as exc:  # pragma: no cover - guarded by argparse choices but defensive
        raise SystemExit(f"Invalid vintage_policy: {args.vintage_policy}") from exc
    if args.write_csv and args.skip_csv:
        raise SystemExit("--write_csv and --skip_csv are mutually exclusive")
    if args.write_csv:
        write_csv: bool | None = True
    elif args.skip_csv:
        write_csv = False
    else:
        write_csv = None
    default_csv_max_rows = None
    for field in fields(TFTDatasetTaskConfig):
        if field.name == "csv_max_rows":
            if isinstance(field.default, int):
                default_csv_max_rows = field.default
            break
    if default_csv_max_rows is None or not isinstance(default_csv_max_rows, int):
        default_csv_max_rows = 1_000_000
    csv_max_rows = (
        int(args.csv_max_rows)
        if args.csv_max_rows is not None
        else int(default_csv_max_rows)
    )

    target_semantics = _parse_target_semantics(args.target_semantics)
    cfg = TFTDatasetTaskConfig(
        data_dir=Path(args.data_dir),
        out_dir=out_dir,
        symbols=args.symbols,
        target_semantics=target_semantics,
        lookback_periods=args.lookback_periods,
        include_macro=not bool(args.no_macro),
        macro_lag_days=args.macro_lag_days,
        include_micro=args.include_micro,
        include_l2=args.include_l2,
        include_events=args.include_events,
        include_calendar=args.include_calendar,
        include_macro_deltas=args.include_macro_deltas,
        include_calendar_lags=args.include_calendar_lags,
        include_clustering_tags=args.include_clustering_tags,
        include_context_features=args.include_context_features,
        include_earnings=args.include_earnings,
        earnings_lag_days=args.earnings_lag_days,
        micro_base_dir=micro_base_dir,
        l2_base_dir=l2_base_dir,
        chunk_days=args.chunk_days,
        start=_parse_optional_date(args.start),
        end=_parse_optional_date(args.end),
        register_features=args.register_features,
        feature_registry_dir=Path(args.feature_registry_dir) if args.feature_registry_dir else None,
        feature_role=args.feature_role,
        emit_dataset_events=args.emit_dataset_events,
        fred_vintage_dir=Path(args.fred_vintage_dir) if args.fred_vintage_dir else None,
        events_base_dir=events_dir,
        student_mode=args.student_mode,
        market_dataset_id=args.market_dataset_id,
        market_inputs=_parse_market_inputs(args.market_inputs_json),
        vintage_policy=vintage_policy,
        vintage_as_of=_parse_optional_date(args.vintage_as_of),
        convert_vintage_to_age=args.convert_vintage_age,
        write_csv=write_csv,
        csv_max_rows=csv_max_rows,
        csv_sample_rows=args.csv_sample_rows,
    )

    LOGGER.info("Building TFT dataset %s", cfg)
    result = build_fn(cfg, data_store=data_store)

    print(
        "Saved dataset to"
        f" {result.dataset_parquet} and {result.dataset_csv}\nSaved features to {result.features_npz}",
    )
    if result.feature_set_id:
        print(f"Registered feature set: {result.feature_set_id} in {cfg.feature_registry_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint for building TFT datasets.
    """
    return _main_with_dependencies(
        argv,
        build_fn=build_tft_dataset,
        data_store_cls=DataStore,
    )


__all__ = ["DataStore", "TFTDatasetTaskConfig", "build_tft_dataset", "main"]


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
