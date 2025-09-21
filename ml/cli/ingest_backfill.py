#!/usr/bin/env python3
"""
Thin wrapper delegating ingestion backfill tasks.
"""

from __future__ import annotations

from collections.abc import Sequence

from ml.tasks.ingest.backfill import main as ingest_backfill_main


__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError(
            "This CLI entrypoint does not accept argv override; use subprocess invocation instead",
        )
    return ingest_backfill_main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
