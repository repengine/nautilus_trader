#!/usr/bin/env python3
"""
Clean up legacy suffixed catalog identifiers.

This tool scans ParquetDataCatalog directories (e.g., quote_tick) for identifier
folders that include deprecated suffixes like ``-MBP1`` or ``-TBBO`` and either
prunes them or consolidates them into the canonical unsuffixed identifier.

Defaults are conservative: dry-run + prune mode. Use ``--apply`` to execute.

Examples:
    python -m ml.tools.catalog_identifier_cleanup \\
        --catalog-path data/catalog \\
        --class-dir quote_tick \\
        --suffix MBP1 --suffix TBBO \\
        --mode prune --apply

    python -m ml.tools.catalog_identifier_cleanup \\
        --catalog-path data/catalog \\
        --class-dir quote_tick \\
        --suffix MBP1 --suffix TBBO \\
        --mode consolidate --on-conflict skip --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SUFFIXES = ("MBP1", "TBBO")
DEFAULT_CLASS_DIRS = ("quote_tick",)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CleanupCandidate:
    """
    A single suffixed identifier directory discovered in the catalog.
    """

    class_dir: str
    canonical_id: str
    suffix: str
    canonical_path: Path
    suffixed_path: Path


@dataclass(frozen=True, slots=True)
class CleanupStats:
    """
    Summary counters for a cleanup operation.
    """

    total_candidates: int
    planned: int
    skipped: int
    orphans: int
    conflicts: int


def _default_catalog_path() -> Path | None:
    env_path = os.environ.get("CATALOG_PATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate
    local = Path.cwd() / "data" / "catalog"
    if local.exists():
        return local
    return None


def _split_suffix(name: str, suffixes: set[str]) -> tuple[str, str] | None:
    if "-" not in name:
        return None
    base, suffix = name.rsplit("-", 1)
    if not base:
        return None
    normalized = suffix.upper()
    if normalized not in suffixes:
        return None
    return base, normalized


def _iter_identifier_dirs(class_dir: Path) -> Iterable[Path]:
    if not class_dir.exists():
        return ()
    return (child for child in class_dir.iterdir() if child.is_dir())


def _discover_candidates(
    *,
    catalog_root: Path,
    class_dirs: Iterable[str],
    suffixes: set[str],
) -> list[CleanupCandidate]:
    candidates: list[CleanupCandidate] = []
    for class_dir in class_dirs:
        data_root = catalog_root / "data" / class_dir
        for child in _iter_identifier_dirs(data_root):
            parsed = _split_suffix(child.name, suffixes)
            if parsed is None:
                continue
            canonical_id, suffix = parsed
            canonical_path = data_root / canonical_id
            candidates.append(
                CleanupCandidate(
                    class_dir=class_dir,
                    canonical_id=canonical_id,
                    suffix=suffix,
                    canonical_path=canonical_path,
                    suffixed_path=child,
                ),
            )
    return candidates


def _prune_candidates(
    candidates: list[CleanupCandidate],
    *,
    apply: bool,
    allow_orphans: bool,
    verbose: bool,
) -> CleanupStats:
    planned = 0
    skipped = 0
    orphans = 0
    for candidate in candidates:
        canonical_exists = candidate.canonical_path.exists()
        if not canonical_exists and not allow_orphans:
            skipped += 1
            orphans += 1
            if verbose:
                logger.info(
                    "Skipping orphaned suffixed directory (no canonical target)",
                    extra={"path": str(candidate.suffixed_path)},
                )
            continue
        planned += 1
        if verbose:
            logger.info(
                "Pruning suffixed directory",
                extra={"path": str(candidate.suffixed_path)},
            )
        if apply:
            shutil.rmtree(candidate.suffixed_path)
    return CleanupStats(
        total_candidates=len(candidates),
        planned=planned,
        skipped=skipped,
        orphans=orphans,
        conflicts=0,
    )


def _move_file(
    *,
    source: Path,
    target: Path,
    on_conflict: str,
    apply: bool,
    verbose: bool,
) -> bool:
    if target.exists():
        if on_conflict == "skip":
            if verbose:
                logger.info(
                    "Skipping file due to conflict",
                    extra={"source": str(source), "target": str(target)},
                )
            return False
        if on_conflict == "overwrite":
            if verbose:
                logger.info(
                    "Overwriting file due to conflict",
                    extra={"source": str(source), "target": str(target)},
                )
            if apply:
                target.unlink()
        else:
            msg = f"Conflict for {source} -> {target}"
            raise RuntimeError(msg)
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    return True


def _consolidate_candidates(
    candidates: list[CleanupCandidate],
    *,
    apply: bool,
    allow_orphans: bool,
    on_conflict: str,
    verbose: bool,
) -> CleanupStats:
    planned = 0
    skipped = 0
    orphans = 0
    conflicts = 0
    for candidate in candidates:
        canonical_exists = candidate.canonical_path.exists()
        if not canonical_exists and not allow_orphans:
            skipped += 1
            orphans += 1
            if verbose:
                logger.info(
                    "Skipping orphaned suffixed directory (no canonical target)",
                    extra={"path": str(candidate.suffixed_path)},
                )
            continue
        planned += 1
        if not canonical_exists:
            if verbose:
                logger.info(
                    "Renaming suffixed directory to canonical identifier",
                    extra={
                        "source": str(candidate.suffixed_path),
                        "target": str(candidate.canonical_path),
                    },
                )
            if apply:
                candidate.canonical_path.parent.mkdir(parents=True, exist_ok=True)
                candidate.suffixed_path.rename(candidate.canonical_path)
            continue
        for source in candidate.suffixed_path.rglob("*"):
            if source.is_dir():
                continue
            rel = source.relative_to(candidate.suffixed_path)
            target = candidate.canonical_path / rel
            try:
                moved = _move_file(
                    source=source,
                    target=target,
                    on_conflict=on_conflict,
                    apply=apply,
                    verbose=verbose,
                )
                if not moved:
                    conflicts += 1
            except RuntimeError as exc:
                logger.error(str(exc))
                raise
        if apply:
            shutil.rmtree(candidate.suffixed_path)
    return CleanupStats(
        total_candidates=len(candidates),
        planned=planned,
        skipped=skipped,
        orphans=orphans,
        conflicts=conflicts,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean up legacy suffixed catalog identifiers.",
    )
    parser.add_argument(
        "--catalog-path",
        dest="catalog_path",
        help="Parquet catalog root (defaults to $CATALOG_PATH or ./data/catalog).",
    )
    parser.add_argument(
        "--class-dir",
        dest="class_dirs",
        action="append",
        help="Catalog class directory to scan (default: quote_tick).",
    )
    parser.add_argument(
        "--suffix",
        dest="suffixes",
        action="append",
        help="Suffix to remove or consolidate (default: MBP1, TBBO).",
    )
    parser.add_argument(
        "--mode",
        choices=("prune", "consolidate"),
        default="prune",
        help="Cleanup mode: prune removes suffixed dirs; consolidate merges into canonical.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute changes (default: dry-run).",
    )
    parser.add_argument(
        "--allow-orphans",
        action="store_true",
        help="Allow removal/consolidation when no canonical identifier exists.",
    )
    parser.add_argument(
        "--on-conflict",
        choices=("skip", "overwrite", "error"),
        default="skip",
        help="Conflict policy when consolidating into existing canonical directories.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    catalog_path = (
        Path(args.catalog_path).expanduser()
        if args.catalog_path
        else _default_catalog_path()
    )
    if catalog_path is None:
        parser.error("catalog path not found (set --catalog-path or $CATALOG_PATH)")
    if not catalog_path.exists():
        parser.error(f"catalog path does not exist: {catalog_path}")

    class_dirs = tuple(args.class_dirs or DEFAULT_CLASS_DIRS)
    suffixes = {suffix.upper() for suffix in (args.suffixes or DEFAULT_SUFFIXES)}

    candidates = _discover_candidates(
        catalog_root=catalog_path,
        class_dirs=class_dirs,
        suffixes=suffixes,
    )
    if not candidates:
        logger.info("No suffixed identifiers found.")
        return 0

    logger.info(
        "Discovered %d suffixed identifiers under %s",
        len(candidates),
        catalog_path,
    )

    if args.mode == "prune":
        stats = _prune_candidates(
            candidates,
            apply=args.apply,
            allow_orphans=args.allow_orphans,
            verbose=args.verbose,
        )
    else:
        stats = _consolidate_candidates(
            candidates,
            apply=args.apply,
            allow_orphans=args.allow_orphans,
            on_conflict=args.on_conflict,
            verbose=args.verbose,
        )

    logger.info(
        "Cleanup summary: total=%d planned=%d skipped=%d orphans=%d conflicts=%d apply=%s",
        stats.total_candidates,
        stats.planned,
        stats.skipped,
        stats.orphans,
        stats.conflicts,
        args.apply,
    )
    if not args.apply:
        logger.info("Dry-run only (use --apply to execute changes).")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
