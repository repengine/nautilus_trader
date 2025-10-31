"""
Utility to execute Phase 4 parameter sensitivity sweeps.

This CLI orchestrates the robustness-focused parameter sensitivity specifications
defined in :class:`ml.config.playground.ThreeDRiskBacktestDefaults`. Results are
persisted under ``playground/reports/backtesting/sensitivity`` with per-spec
artefacts and an aggregate summary.
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
from playground.backtest.runner import run_parameter_sensitivity_suite  # noqa: E402


DEFAULT_DATASET_PATH = Path("playground/data/sector_dataset")
DEFAULT_OUTPUT_DIR = Path("playground/reports/backtesting")
LOGGER = structlog.get_logger(__name__)


def _parse_comma_separated(value: str) -> tuple[str, ...]:
    """Parse comma-separated CLI input into a tuple of unique entries preserving order."""
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Execute Phase 4 parameter sensitivity sweeps.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the sector dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store sensitivity artefacts (default: %(default)s)",
    )
    parser.add_argument(
        "--spec-slugs",
        type=str,
        default="",
        help="Comma separated sensitivity spec slugs to execute (default: all configured).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for Phase 4 sensitivity sweeps."""
    args = parse_args(argv)
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    spec_slugs = _parse_comma_separated(args.spec_slugs)

    suite_result = run_parameter_sensitivity_suite(
        dataset_path=dataset_path,
        output_dir=output_dir,
        spec_slugs=spec_slugs if spec_slugs else None,
    )
    tolerance_breaches = [
        (
            run.spec.slug,
            run.metadata.get("metric_spread"),
            run.metadata.get("metric_spread_tolerance"),
        )
        for run in suite_result.runs
        if run.metadata.get("metric_spread_ok") is False
    ]
    for slug, spread, tolerance in tolerance_breaches:
        LOGGER.warning(
            "phase4_sensitivity_sharpe_delta_violation",
            spec=slug,
            spread=spread,
            tolerance=tolerance,
        )
    summary = suite_result.summary_frame()
    summary_path = output_dir / suite_result.output_dirname / "summary.csv"
    if summary_path.exists():
        report_path = summary_path.with_name("sensitivity_analysis.pdf")
        try:
            generate_sensitivity_summary_pdf(
                summary_path=summary_path,
                output_path=report_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.warning(
                "phase4_sensitivity_pdf_generation_failed",
                summary_path=str(summary_path.resolve()),
                report_path=str(report_path.resolve()),
                error=str(exc),
                exc_info=True,
            )
        else:
            LOGGER.info(
                "phase4_sensitivity_pdf_generated",
                summary_path=str(summary_path.resolve()),
                report_path=str(report_path.resolve()),
            )
    else:
        LOGGER.warning(
            "phase4_sensitivity_summary_missing",
            summary_path=str(summary_path.resolve()),
        )

    LOGGER.info(
        "phase4_parameter_sensitivity_completed",
        summary_path=str(summary_path.resolve()),
        specs_evaluated=len(suite_result.runs),
        total_combinations=int(summary.get_column("evaluated_combinations").sum()) if not summary.is_empty() else 0,
    )


if __name__ == "__main__":
    main()
