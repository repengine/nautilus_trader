#!/usr/bin/env python3
"""Guard streaming manifests for validation-return join regressions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ml.training.event_driven.guardrails import check_validation_joins


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assert validation-return joins remain healthy for recent streaming cohorts.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing streaming manifest JSON files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of manifests to inspect (newest first).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    issues = check_validation_joins(args.manifest_dir, limit=args.limit)
    if issues:
        for issue in issues:
            print(issue)
        return 1
    return 0


__all__ = ["check_validation_joins", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
