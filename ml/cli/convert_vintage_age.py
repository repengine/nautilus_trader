#!/usr/bin/env python3
r"""
CLI for converting macro vintage timestamps into age features.

This tool streams a parquet dataset, replaces every ``*_value_vintage_ts`` column with
its numeric age-in-minutes counterpart, and updates the accompanying metadata JSON.

Example:
    $ python -m ml.cli.convert_vintage_age \\
        --source ml_out/full_tft_95/dataset.parquet \\
        --metadata ml_out/full_tft_95/dataset_metadata.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import structlog

from ml.preprocessing.vintage_age import convert_vintage_timestamps_to_age
from ml.preprocessing.vintage_age import update_metadata_with_vintage_age
from ml.preprocessing.vintage_age import write_metadata


LOGGER = structlog.get_logger(__name__)


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        msg = f"Metadata file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"Metadata JSON must be an object (received {type(payload).__name__})"
        raise ValueError(msg)
    return {str(key): value for key, value in payload.items()}


def _default_destination(source: Path) -> Path:
    return source.with_name(f"{source.stem}_with_vintage_age{source.suffix}")


def _default_metadata_path(source: Path) -> Path:
    return source.with_name("dataset_metadata.json")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = f"Batch size must be positive (received {parsed})."
        raise argparse.ArgumentTypeError(msg)
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert *_value_vintage_ts columns to *_vintage_age_minutes features.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the parquet dataset containing vintage timestamp columns.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Output parquet path (defaults to <stem>_with_vintage_age.parquet).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata JSON path (defaults to sibling dataset_metadata.json).",
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Column containing event timestamps stored as int64 nanoseconds.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=32_768,
        help="Maximum rows read per batch while streaming (default: 32768).",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        help="Compression codec for the output parquet (default: snappy).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination parquet if it already exists.",
    )
    args = parser.parse_args(argv)

    source = args.source.resolve()
    destination = (args.destination or _default_destination(source)).resolve()
    metadata_path = (args.metadata or _default_metadata_path(source)).resolve()

    if destination.exists() and not args.overwrite:
        parser.error(
            f"Destination parquet already exists: {destination}. "
            "Run with --overwrite to replace it.",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    result = convert_vintage_timestamps_to_age(
        source,
        destination,
        timestamp_column=args.timestamp_column,
        batch_size=args.batch_size,
        compression=args.compression,
    )
    metadata = _load_metadata(metadata_path)
    updated_metadata = update_metadata_with_vintage_age(
        metadata,
        vintage_columns=result.vintage_columns,
        age_columns=result.age_columns,
    )
    write_metadata(metadata_path, updated_metadata)

    LOGGER.info(
        "Converted vintage timestamps to age features",
        source=str(source),
        destination=str(destination),
        new_columns=list(result.age_columns),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
