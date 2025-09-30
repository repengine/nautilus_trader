#!/usr/bin/env python3
"""Generate EQUS fallback calibration artefacts from overlapping ITCH data."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from pathlib import Path

import structlog

from ml.data.ingest.calibration import calibration_bundle_to_mapping
from ml.data.ingest.calibration_capture import CalibrationCaptureConfig
from ml.data.ingest.calibration_capture import CalibrationCaptureService
from ml.data.ingest.service import DatabentoIngestionService


logger = structlog.get_logger(__name__)


def _parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate calibration JSON compatible with SymbolCalibration",
    )
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        required=True,
        help="Symbol to calibrate (repeat for multiple).",
    )
    parser.add_argument("--start", required=True, help="Calibration window start (ISO-8601)")
    parser.add_argument("--end", required=True, help="Calibration window end (ISO-8601)")
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to ML_EQUS_CALIBRATION_PATH when set.",
    )
    parser.add_argument("--eq-dataset", default="EQUS.MINI", help="EQ dataset to sample")
    parser.add_argument("--eq-schema", default="ohlcv-1m", help="EQ schema to sample")
    parser.add_argument("--fallback-dataset", default="XNAS.ITCH", help="Fallback dataset")
    parser.add_argument("--trades-schema", default="trades", help="Fallback trades schema")
    parser.add_argument("--depth-schema", default="mbp-1", help="Fallback depth schema")
    parser.add_argument(
        "--chunk-days",
        default=7,
        type=int,
        help="Chunk size in days for ingestion requests",
    )
    parser.add_argument(
        "--min-ratio-minutes",
        default=10,
        type=int,
        help="Minimum overlapping minutes required per minute-of-day scaler",
    )
    parser.add_argument(
        "--price-clip",
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.05, 20.0),
        type=float,
        help="Clamp bounds for price scaling ratios",
    )
    parser.add_argument(
        "--volume-clip",
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.01, 100.0),
        type=float,
        help="Clamp bounds for volume scaling ratios",
    )
    parser.add_argument(
        "--disallow-cost",
        action="store_true",
        help="Enforce ingestion cost safety limits (default allows cost for calibration)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_output_path(args: argparse.Namespace) -> Path | None:
    if args.output:
        return Path(args.output)
    env_path = os.getenv("ML_EQUS_CALIBRATION_PATH")
    return Path(env_path) if env_path else None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = _resolve_output_path(args)
    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    symbols = tuple(symbol.strip() for symbol in args.symbols if symbol.strip())
    if not symbols:
        raise SystemExit("No valid symbols provided")

    ingestion_service = DatabentoIngestionService.from_env()
    capture_service = CalibrationCaptureService(ingestion_service)
    config = CalibrationCaptureConfig(
        symbols=symbols,
        start=start,
        end=end,
        output_path=output_path,
        eq_dataset=args.eq_dataset,
        eq_schema=args.eq_schema,
        fallback_dataset=args.fallback_dataset,
        trades_schema=args.trades_schema,
        depth_schema=args.depth_schema,
        allow_cost=not args.disallow_cost,
        chunk_days=args.chunk_days,
        min_ratio_minutes=args.min_ratio_minutes,
        price_scale_clip=(float(args.price_clip[0]), float(args.price_clip[1])),
        volume_scale_clip=(float(args.volume_clip[0]), float(args.volume_clip[1])),
    )

    try:
        result = capture_service.capture(config)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("calibration.capture.failed", error=str(exc))
        raise SystemExit(1) from exc

    if result.output_path is None:
        payload = calibration_bundle_to_mapping(result.bundle)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Calibration bundle written to {result.output_path}")
    return 0


__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

