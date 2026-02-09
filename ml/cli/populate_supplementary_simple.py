#!/usr/bin/env python3
"""
Thin CLI wrapper for synthetic supplementary data generation.
"""

from __future__ import annotations

import argparse
import sys
import uuid as _uuid
from pathlib import Path

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.data.ingest import PopulateSupplementaryTaskConfig
from ml.data.ingest import populate_supplementary_data
from ml.data.loaders.supplementary import SupplementaryOutputs


__all__ = ["main"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate supplementary data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/supplementary"),
        help="Output directory",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Synthetic data horizon in years (default: 2)",
    )
    return parser.parse_args(argv)


def _print_summary(outputs: SupplementaryOutputs) -> None:
    print("Supplementary data generation complete")
    print(f"Output directory: {outputs.ohlcv_path.parent}")
    print(f"Records: {outputs.record_count:,}")
    print(f"Symbols: {outputs.symbol_count}")
    print(f"Date range: {outputs.start.date()} -> {outputs.end.date()}")
    for label, path in (
        ("OHLCV", outputs.ohlcv_path),
        ("Correlations", outputs.correlations_path),
        ("Spreads", outputs.spreads_path),
        ("Metadata", outputs.metadata_path),
    ):
        if path is None:
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {label}: {path.name} ({size_mb:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    bind_log_context(
        run_id=f"cli_populate_supplementary_simple_{_uuid.uuid4().hex[:8]}",
        component="ml.cli.populate_supplementary_simple",
    )

    config = PopulateSupplementaryTaskConfig(
        output_dir=args.output_dir,
        synthetic_years=max(args.years, 1),
    )
    try:
        outputs = populate_supplementary_data(config)
    except Exception as exc:  # pragma: no cover - surface CLI errors
        print(str(exc), file=sys.stderr)
        return 1

    _print_summary(outputs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
