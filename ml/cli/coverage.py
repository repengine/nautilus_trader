#!/usr/bin/env python3
"""
Thin wrapper delegating to :mod:`ml.tasks.monitoring.coverage`.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from datetime import datetime as _datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ml.tasks.monitoring.coverage import main as coverage_main
from ml.tasks.monitoring.coverage import plan_backfill as _plan_backfill


datetime = _datetime  # Re-exported for legacy tests which monkeypatch this symbol.

__all__ = ["main", "plan_backfill"]

if TYPE_CHECKING:
    from ml.registry.persistence import PersistenceConfig


def main(argv: Sequence[str] | None = None) -> int:
    return coverage_main(list(argv) if argv is not None else None)


def plan_backfill(
    from_dataset: str,
    to_dataset: str,
    date: str,
    *,
    instruments: list[str] | None = None,
    registry_path: Path | None = None,
    persistence_config: PersistenceConfig | None = None,
    output_file: Path | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> None:
    if now_fn is None:
        now_fn = datetime.now
    _plan_backfill(
        from_dataset=from_dataset,
        to_dataset=to_dataset,
        date=date,
        instruments=instruments,
        registry_path=registry_path,
        persistence_config=persistence_config,
        output_file=output_file,
        now_fn=now_fn,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
