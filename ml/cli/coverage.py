#!/usr/bin/env python3
"""
Thin wrapper delegating to :mod:`ml.tasks.monitoring.coverage`.
"""

from __future__ import annotations

from collections.abc import Sequence

from ml.tasks.monitoring.coverage import main as coverage_main


__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError(
            "This CLI entrypoint does not accept argv override; use subprocess invocation instead",
        )
    return coverage_main()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
