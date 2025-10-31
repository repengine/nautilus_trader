"""
CLI utility for Phase 4 factor outlier detection and treatment analysis.
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

from playground.risk_model.outlier_detection import evaluate_factor_outliers  # noqa: E402


DEFAULT_DATASET_PATH = Path("playground/data/sector_dataset/factor_returns.parquet")
DEFAULT_OUTPUT_DIR = Path("playground/reports/backtesting")
LOGGER = structlog.get_logger(__name__)


def _parse_columns(value: str) -> tuple[str, ...]:
    """Parse a comma-separated list of factor columns."""
    entries: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        trimmed = part.strip()
        if not trimmed or trimmed in seen:
            continue
        entries.append(trimmed)
        seen.add(trimmed)
    return tuple(entries)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the outlier detection CLI."""
    parser = argparse.ArgumentParser(description="Run Phase 4 factor outlier detection.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the factor return dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store outlier detection artefacts (default: %(default)s)",
    )
    parser.add_argument(
        "--factor-columns",
        type=str,
        default="",
        help="Comma separated factor column names (default: infer columns prefixed with 'factor_').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for outlier detection (default: %(default)s).",
    )
    parser.add_argument(
        "--treatments",
        type=str,
        default="winsorize,exclude",
        help="Comma separated treatment strategies to evaluate (default: winsorize,exclude).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Execute the outlier detection workflow."""
    args = parse_args(argv)
    dataset_path: Path = args.dataset_path
    output_dir: Path = args.output_dir
    factor_columns = _parse_columns(args.factor_columns)
    treatments = _parse_columns(args.treatments)

    report = evaluate_factor_outliers(
        dataset_path=dataset_path,
        factor_columns=factor_columns if factor_columns else None,
        threshold=args.threshold,
        treatments=treatments if treatments else None,
    )

    report_dir = output_dir / "outliers"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "factor_outlier_report.json"
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    LOGGER.info(
        "phase4_factor_outlier_report_written",
        dataset=str(dataset_path.resolve()),
        output_path=str(output_path.resolve()),
        outlier_ratio=report.outlier_ratio,
        recommended_treatment=report.recommended_treatment,
    )


if __name__ == "__main__":
    main()
