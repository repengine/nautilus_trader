#!/usr/bin/env python3
"""
Thin wrapper delegating to :mod:`ml.tasks.monitoring.health`.
"""

from __future__ import annotations

from collections.abc import Sequence

from ml.tasks.monitoring.health import main as health_main


__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    return health_main(list(argv) if argv is not None else None)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
