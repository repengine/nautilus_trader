#!/usr/bin/env python3
"""
Thin wrapper delegating to :mod:`ml.tasks.dev.sanity_check`.
"""

from __future__ import annotations

from collections.abc import Sequence

from ml.tasks.dev.sanity_check import main as sanity_main


__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    sanity_main()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
