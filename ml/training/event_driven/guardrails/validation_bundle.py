"""Validation bundle for streaming wave rollouts."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ml.common.subprocess_utils import run_command


logger = logging.getLogger(__name__)

# Default paths and targets
DEFAULT_DOC_PATHS = (
    Path("ml/docs/architecture/event_driven_streaming_plan.md"),
    Path("ml/docs/ops/streaming_scaling_experiments.md"),
)
DEFAULT_STATE_PATH = Path("ml_out/streaming_training_state_snapshot.json")
ALERTS_PATH = Path("ml_out/streaming_alerts.json")
DEFAULT_PYTEST_TARGETS = (
    "ml/tests/unit/config/test_streaming_pipeline_config.py",
    "ml/tests/unit/training/event_driven/",
    "ml/tests/integration/training/event_driven/test_plan_to_result.py",
)


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for validation bundle.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Run validation bundle for streaming wave rollouts"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing streaming manifest JSON files",
    )
    parser.add_argument(
        "--manifest-limit",
        type=int,
        default=None,
        help="Limit number of manifests to validate (newest first)",
    )
    parser.add_argument(
        "--max-doc-age-hours",
        type=float,
        default=24.0,
        help="Maximum age in hours for documentation files",
    )
    parser.add_argument(
        "--alerts-only",
        action="store_true",
        help="Only check for alert files, don't run full validation",
    )
    return parser


def run_alerts_only() -> list[str]:
    """
    Check for presence of alert files.

    Returns:
        List of alert rule names that have files present
    """
    alerts = []
    if ALERTS_PATH.exists():
        alerts.append("streaming_alerts")
    return alerts


def validate_manifest_coverage(manifest_dir: Path, limit: int | None = None) -> None:
    """
    Validate that recent manifests have adequate coverage.

    Args:
        manifest_dir: Directory containing manifest files
        limit: Optional limit on number of manifests to check

    Raises:
        RuntimeError: If manifests are missing required fields
    """
    # TODO: Implement actual manifest validation
    # For now, just check directory exists
    if not manifest_dir.exists():
        raise RuntimeError(f"Manifest directory does not exist: {manifest_dir}")


def _check_doc_staleness(max_age_hours: float) -> None:
    """
    Check that documentation files are not stale.

    Args:
        max_age_hours: Maximum age in hours

    Raises:
        RuntimeError: If any docs are stale
    """
    now = time.time()
    max_age_seconds = max_age_hours * 3600

    for doc_path in DEFAULT_DOC_PATHS:
        if not doc_path.exists():
            continue

        age_seconds = now - doc_path.stat().st_mtime
        if age_seconds > max_age_seconds:
            age_hours = age_seconds / 3600
            raise RuntimeError(
                f"Documentation file {doc_path} is stale "
                f"({age_hours:.1f}h old, max {max_age_hours}h)"
            )


def run_validation(args: argparse.Namespace) -> None:
    """
    Run full validation bundle.

    Args:
        args: Parsed command-line arguments

    Raises:
        SubprocessExecutionError: If any validation command fails
        RuntimeError: If validation checks fail
    """
    # Check doc staleness first
    _check_doc_staleness(args.max_doc_age_hours)

    # Validate manifests
    validate_manifest_coverage(args.manifest_dir, args.manifest_limit)

    # Run mypy
    logger.info("Running mypy type checking...")
    run_command(["poetry", "run", "mypy", "ml", "--strict"])

    # Run ruff
    logger.info("Running ruff linter...")
    run_command(["poetry", "run", "ruff", "check", "ml"])

    # Run pytest on targeted test suites
    logger.info("Running pytest on streaming tests...")
    pytest_cmd = ["poetry", "run", "pytest"] + list(DEFAULT_PYTEST_TARGETS) + ["-q"]
    run_command(pytest_cmd)

    logger.info("✓ All validation checks passed")


__all__ = [
    "ALERTS_PATH",
    "DEFAULT_DOC_PATHS",
    "DEFAULT_PYTEST_TARGETS",
    "DEFAULT_STATE_PATH",
    "build_parser",
    "run_alerts_only",
    "run_validation",
    "validate_manifest_coverage",
]
