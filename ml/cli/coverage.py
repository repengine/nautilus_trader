#!/usr/bin/env python3
"""
Thin wrapper delegating to :mod:`ml.tasks.monitoring.coverage`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime as _datetime

from ml.tasks.monitoring.coverage import main as coverage_main
from ml.tasks.monitoring.coverage import plan_backfill as _plan_backfill


datetime = _datetime  # Re-exported for legacy tests which monkeypatch this symbol.

__all__ = ["main", "plan_backfill"]


def main(argv: Sequence[str] | None = None) -> int:
    return coverage_main(list(argv) if argv is not None else None)


def plan_backfill(*args: object, **kwargs: object) -> None:
    kwargs.setdefault("now_fn", datetime.now)
    _plan_backfill(*args, **kwargs)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
