#!/usr/bin/env python3
"""
CLI wrapper for dataset quality report generation.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ml.data import DatasetReportConfig
from ml.data import generate_dataset_report


__all__ = ["main"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset quality report")
    parser.add_argument("--dataset", required=True, help="Path to dataset parquet or CSV")
    parser.add_argument("--out_json", required=False, help="Optional JSON output path")
    parser.add_argument("--out_md", required=False, help="Optional Markdown output path")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = DatasetReportConfig(
        dataset_path=Path(args.dataset),
        output_json=Path(args.out_json) if args.out_json else None,
        output_markdown=Path(args.out_md) if args.out_md else None,
    )
    report = generate_dataset_report(config)
    print(report.to_json())
    if report.markdown is not None and config.output_markdown is None:
        print(report.markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
