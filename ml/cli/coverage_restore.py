#!/usr/bin/env python3
"""
Run the coverage restoration workflow using the pipeline environment settings.

This CLI mirrors the automation executed inside the pipeline container: it
builds scheduler configuration from environment variables, bootstraps the
catalog + scheduler, and invokes ``PipelineRunner.run_coverage_restoration_once``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ml.deployment.entrypoint_pipeline import CoverageStatus
from ml.deployment.entrypoint_pipeline import PipelineRunner


__all__ = ["main"]


def _format_summary(summary: CoverageStatus) -> str:
    """
    Produce a human-readable summary string for terminal output.
    """
    lines = [
        "Coverage restoration completed:",
        f"  last_run: {summary['last_run']}",
        f"  last_success: {summary['last_success']}",
        f"  buckets_total: {summary['buckets_total']}",
        f"  buckets_restore_catalog: {summary['buckets_restore_catalog']}",
        f"  buckets_reingest_source: {summary['buckets_reingest_source']}",
        f"  buckets_healthy: {summary['buckets_healthy']}",
        f"  last_error: {summary['last_error']}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point for the coverage restoration CLI.
    """
    parser = argparse.ArgumentParser(description="Run coverage restoration once using pipeline config.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the coverage summary as JSON (default: human-readable text).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify coverage buckets without restoring or re-ingesting data.",
    )
    args = parser.parse_args(argv)

    runner = PipelineRunner()
    summary = runner.run_coverage_restoration_once(dry_run=args.dry_run)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_format_summary(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
