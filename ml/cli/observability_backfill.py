#!/usr/bin/env python3
"""
CLI entrypoint for observability JSONL backfill.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ml.observability.backfill import ObservabilityBackfillConfig
from ml.observability.backfill import backfill_observability_tables


__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill observability JSONL into DB")
    parser.add_argument("--src", required=True, help="Base directory of observability files")
    parser.add_argument("--db-url", required=True, help="SQLAlchemy DB URL")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        config = ObservabilityBackfillConfig(
            src=Path(str(args.src)),
            db_url=str(args.db_url),
        )
        backfill_observability_tables(config, emit=print)
    except ValueError as exc:
        parser.error(str(exc))
    except OSError as exc:
        raise SystemExit(str(exc)) from exc

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
