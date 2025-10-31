"""
Convert the Phase 4 sensitivity summary CSV into a PDF report.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import structlog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playground.backtest.reporting import generate_sensitivity_summary_pdf  # noqa: E402


DEFAULT_SUMMARY_PATH = Path("playground/reports/backtesting/sensitivity/summary.csv")
DEFAULT_OUTPUT_PATH = Path("playground/reports/backtesting/sensitivity/sensitivity_analysis.pdf")
LOGGER = structlog.get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the report generator."""
    parser = argparse.ArgumentParser(description="Generate the Phase 4 sensitivity PDF report.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to the sensitivity summary CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination PDF path (default: %(default)s)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Override report title (default: Phase 4 Parameter Sensitivity Summary).",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=None,
        help="Optional note to include under the title (can be specified multiple times).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for the CLI."""
    args = parse_args(argv)
    title = args.title or None
    notes = tuple(args.note) if args.note else None

    generate_sensitivity_summary_pdf(
        summary_path=args.summary_path,
        output_path=args.output_path,
        title=title,
        notes=notes,
    )

    LOGGER.info(
        "phase4_sensitivity_report_generated",
        summary_path=str(args.summary_path.resolve()),
        output_path=str(args.output_path.resolve()),
        notes=len(notes) if notes else 0,
    )


if __name__ == "__main__":
    main()
