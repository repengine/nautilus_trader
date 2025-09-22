#!/usr/bin/env python3
"""
Thin wrapper delegating observability backfill tasks.
"""

from __future__ import annotations

from collections.abc import Sequence

from ml.tasks.observability.backfill import main as backfill_main


__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    return backfill_main(list(argv) if argv is not None else None)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
