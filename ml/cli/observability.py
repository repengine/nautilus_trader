#!/usr/bin/env python3
"""
Thin wrapper delegating observability flush tasks.
"""

from __future__ import annotations

from collections.abc import Sequence

from ml.tasks.observability.flush import main as flush_main


__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    return flush_main(list(argv) if argv is not None else None)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
