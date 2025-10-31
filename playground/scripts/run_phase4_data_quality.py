"""
Utility to audit missing data coverage and imputation strategies (Phase 4).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import structlog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playground.risk_model.data_quality import audit_missing_data  # noqa: E402


DEFAULT_DATASET_PATH = Path("playground/data/sector_dataset/sector_returns.parquet")
DEFAULT_OUTPUT_DIR = Path("playground/reports/backtesting")
LOGGER = structlog.get_logger(__name__)


def _parse_methods(value: str) -> tuple[str, ...]:
    """Parse comma separated imputation methods."""
    entries: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        trimmed = part.strip().lower()
        if not trimmed or trimmed in seen:
            continue
        entries.append(trimmed)
        seen.add(trimmed)
    return tuple(entries)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Phase 4 missing data audit.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the sector dataset file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store audit artefacts (default: %(default)s)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma separated imputation methods to evaluate (default: forward_fill,linear,kalman).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for Phase 4 data quality audits."""
    args = parse_args(argv)
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    methods = _parse_methods(args.methods)

    audit_result = audit_missing_data(
        dataset_path=dataset_path,
        methods=methods if methods else None,
    )
    report_dir = output_dir / "data_quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "missing_data_audit.json"
    summary_path.write_text(json.dumps(audit_result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    LOGGER.info(
        "phase4_data_quality_audit_completed",
        dataset=str(dataset_path.resolve()),
        summary_path=str(summary_path.resolve()),
        missing_ratio=audit_result.missing_ratio,
    )


if __name__ == "__main__":
    main()
