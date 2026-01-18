#!/usr/bin/env python3
"""Validation bundle for streaming wave rollouts."""

from __future__ import annotations

import argparse
import logging
from typing import Any, cast

from ml.common.subprocess_utils import SubprocessExecutionError
from ml.common.subprocess_utils import run_command as _run_command
from ml.training.event_driven.guardrails import validation_bundle


_VALIDATION_BUNDLE = cast(Any, validation_bundle)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    return validation_bundle.build_parser()


def _sync_validation_bundle_defaults() -> None:
    """
    Mirror module-level defaults into the shared validation bundle.

    This keeps CLI tests free to monkeypatch defaults on this module while
    ensuring the underlying validation bundle sees the updated values.
    """
    _VALIDATION_BUNDLE.DEFAULT_DOC_PATHS = DEFAULT_DOC_PATHS
    _VALIDATION_BUNDLE.DEFAULT_STATE_PATH = DEFAULT_STATE_PATH
    _VALIDATION_BUNDLE.ALERTS_PATH = ALERTS_PATH
    _VALIDATION_BUNDLE.DEFAULT_PYTEST_TARGETS = DEFAULT_PYTEST_TARGETS


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _sync_validation_bundle_defaults()
    _VALIDATION_BUNDLE.run_command = run_command
    try:
        if args.alerts_only:
            rules = validation_bundle.run_alerts_only()
            for name in rules:
                print(f"[alerts] {name}: present")
        else:
            validation_bundle.run_validation(args)
    except SubprocessExecutionError as exc:  # pragma: no cover - exercised via tests
        logger.error("validation command failed", extra={"command": exc.command, "returncode": exc.returncode})
        return 1
    except RuntimeError as exc:
        logger.error("validation checks failed", extra={"error": str(exc)})
        return 2
    return 0


DEFAULT_DOC_PATHS = validation_bundle.DEFAULT_DOC_PATHS
DEFAULT_STATE_PATH = validation_bundle.DEFAULT_STATE_PATH
ALERTS_PATH = validation_bundle.ALERTS_PATH
DEFAULT_PYTEST_TARGETS = validation_bundle.DEFAULT_PYTEST_TARGETS
run_validation = validation_bundle.run_validation
_validate_manifest_coverage = validation_bundle.validate_manifest_coverage
_run_alerts_only = validation_bundle.run_alerts_only
run_command = _run_command
_VALIDATION_BUNDLE.run_command = run_command

__all__ = [
    "ALERTS_PATH",
    "DEFAULT_DOC_PATHS",
    "DEFAULT_PYTEST_TARGETS",
    "DEFAULT_STATE_PATH",
    "_validate_manifest_coverage",
    "main",
    "run_command",
    "run_validation",
]


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
