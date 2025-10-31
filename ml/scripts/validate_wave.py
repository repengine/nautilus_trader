#!/usr/bin/env python3
"""Validation bundle for streaming wave rollouts."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from pathlib import Path

from ml.common.subprocess_utils import SubprocessExecutionError
from ml.common.subprocess_utils import run_command


logger = logging.getLogger(__name__)


DEFAULT_DOC_PATHS: tuple[Path, ...] = (
    Path("ml/docs/architecture/event_driven_streaming_plan.md"),
    Path("ml/docs/architecture/model_training_enhancements_plan.md"),
    Path("ml/docs/ops/streaming_scaling_experiments.md"),
)

DEFAULT_PYTEST_TARGETS: tuple[str, ...] = (
    "ml/tests/unit/config/test_streaming_pipeline_config.py",
    "ml/tests/unit/training/event_driven/test_worker.py",
    "ml/dashboard/tests/test_streaming_monitor.py",
    "ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry",
    "ml/tests/integration/consumers/test_streaming_persistence_integration.py",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the streaming validation wave bundle.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("ml_out/tft_streaming_artifacts/full_tft_95"),
        help="Directory containing streaming manifests for summarisation.",
    )
    parser.add_argument(
        "--manifest-limit",
        type=int,
        default=5,
        help="Number of manifests to include in the summary output.",
    )
    parser.add_argument(
        "--max-doc-age-hours",
        type=float,
        default=48.0,
        help="Fail when required docs are older than this many hours.",
    )
    parser.add_argument(
        "--pytest-target",
        action="append",
        dest="pytest_targets",
        default=None,
        help="Additional pytest target to execute (repeatable).",
    )
    parser.add_argument(
        "--doc",
        action="append",
        dest="docs",
        type=Path,
        default=None,
        help="Documentation path that must be fresh (repeatable).",
    )
    return parser


def _run_commands(commands: Iterable[Sequence[str]]) -> None:
    for command in commands:
        logger.info("running command", extra={"command": " ".join(command)})
        run_command(command)


def _ensure_docs_fresh(paths: Iterable[Path], max_age_hours: float) -> None:
    cutoff_seconds = max_age_hours * 3600.0
    now = datetime.now(tz=UTC)
    stale: list[str] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists():
            stale.append(f"{resolved} (missing)")
            continue
        mtime = datetime.fromtimestamp(resolved.stat().st_mtime, tz=UTC)
        age_seconds = (now - mtime).total_seconds()
        if age_seconds > cutoff_seconds:
            hours = age_seconds / 3600.0
            stale.append(f"{resolved} ({hours:.1f}h old)")
    if stale:
        raise RuntimeError("stale documentation detected: " + ", ".join(stale))


def _build_pytest_commands(targets: Sequence[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    for target in targets:
        commands.append(["poetry", "run", "pytest", target])
    return commands


def run_validation(args: argparse.Namespace) -> None:
    pytest_targets = tuple(args.pytest_targets or DEFAULT_PYTEST_TARGETS)
    commands: list[list[str]] = [
        ["poetry", "run", "mypy", "ml", "--strict"],
        ["poetry", "run", "ruff", "check", "ml"],
    ]
    commands.extend(_build_pytest_commands(pytest_targets))
    commands.extend(
        [
            ["make", "validate-metrics"],
            ["make", "validate-events"],
            ["poetry", "run", "coverage", "report"],
            [
                "poetry",
                "run",
                "python",
                "-m",
                "ml.scripts.summarize_streaming_manifests",
                "--manifest-dir",
                str(args.manifest_dir),
                "--limit",
                str(args.manifest_limit),
            ],
        ],
    )
    _run_commands(commands)
    doc_paths = tuple(args.docs or DEFAULT_DOC_PATHS)
    _ensure_docs_fresh(doc_paths, float(args.max_doc_age_hours))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run_validation(args)
    except SubprocessExecutionError as exc:  # pragma: no cover - exercised via tests
        logger.error("validation command failed", extra={"command": exc.command, "returncode": exc.returncode})
        return 1
    except RuntimeError as exc:
        logger.error("validation checks failed", extra={"error": str(exc)})
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
