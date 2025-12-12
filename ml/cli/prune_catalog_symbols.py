#!/usr/bin/env python3
"""
Prune non-matching instrument directories from a Parquet catalog.

Use this to enforce a suffix (e.g., ``.EQUS``) across all catalog kinds
(``bar``, ``quote_tick``, ``trade_tick``, ``order_book_depth10``). By default
the command runs in dry-run mode and only reports what would be removed; pass
``--apply`` to delete directories.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import structlog


__all__ = ["main"]


logger = structlog.get_logger(__name__)

_DEFAULT_KINDS: Final[tuple[str, ...]] = ("bar", "quote_tick", "trade_tick", "order_book_depth10")


def _instrument_token(path: Path, kind: str) -> str:
    if kind == "bar":
        return path.name.split("-", 1)[0]
    return path.name


def _iter_candidate_dirs(catalog_path: Path, kinds: Iterable[str]) -> Iterable[tuple[str, Path, str]]:
    for kind in kinds:
        base = catalog_path / "data" / kind
        if not base.exists():
            continue
        for entry in base.iterdir():
            if entry.is_dir():
                yield (kind, entry, _instrument_token(entry, kind))


def _prune_catalog(
    *,
    catalog_path: Path,
    suffix: str,
    kinds: Iterable[str],
    apply: bool,
) -> list[Path]:
    removed: list[Path] = []
    for kind, entry, token in _iter_candidate_dirs(catalog_path, kinds):
        if token.endswith(suffix):
            continue
        removed.append(entry)
        if apply:
            shutil.rmtree(entry, ignore_errors=False)
            logger.info(
                "catalog.prune.removed",
                kind=kind,
                path=str(entry),
                suffix=suffix,
            )
    return removed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove catalog instrument directories that do not match the required suffix.",
    )
    parser.add_argument(
        "catalog_path",
        help="Path to the Parquet catalog root.",
    )
    parser.add_argument(
        "--suffix",
        default=".EQUS",
        help="Required instrument suffix (default: .EQUS).",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=_DEFAULT_KINDS,
        default=_DEFAULT_KINDS,
        help="Catalog kinds to prune (default: all).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletions. Without this flag the command only reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    catalog_path = Path(args.catalog_path).expanduser()
    if not catalog_path.exists():
        parser.error(f"catalog path does not exist: {catalog_path}")

    removed = _prune_catalog(
        catalog_path=catalog_path,
        suffix=args.suffix,
        kinds=tuple(args.kinds),
        apply=args.apply,
    )

    logger.info(
        "catalog.prune.summary",
        removed=len(removed),
        suffix=args.suffix,
        applied=bool(args.apply),
    )
    if not args.apply and removed:
        logger.info(
            "catalog.prune.dry_run",
            hint="Re-run with --apply to delete the listed directories",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
