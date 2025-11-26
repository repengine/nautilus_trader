"""
Catalog hygiene utilities for ParquetDataCatalog locations.

These helpers keep orchestrator runs deterministic by relocating stale catalog
trees (which may contain overlapping intervals) into timestamped archives
before creating fresh directories for future writes.
"""

from __future__ import annotations

import shutil
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Final

import structlog


logger = structlog.get_logger(__name__)

_ARCHIVE_PREFIX: Final[str] = "archive"


def prepare_clean_catalog_path(
    *,
    catalog_path: Path,
    backup_root: Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """
    Ensure ``catalog_path`` exists and contains no legacy files.

    The function moves any pre-existing catalog directory into a timestamped
    archive folder (under ``backup_root`` when provided) before recreating an
    empty directory at ``catalog_path``.  When the directory is missing or
    already empty the function simply creates the path if necessary.
    """
    resolved = catalog_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not resolved.exists():
        resolved.mkdir(parents=True, exist_ok=True)
        logger.info(
            "catalog_hygiene.created_empty_catalog",
            catalog=str(resolved),
        )
        return resolved
    if not any(resolved.iterdir()):
        return resolved
    archived = archive_catalog(
        catalog_path=resolved,
        backup_root=backup_root,
        timestamp=timestamp,
    )
    resolved.mkdir(parents=True, exist_ok=True)
    logger.info(
        "catalog_hygiene.catalog_archived",
        catalog=str(resolved),
        archive=str(archived),
    )
    return resolved


def archive_catalog(
    *,
    catalog_path: Path,
    backup_root: Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """
    Move ``catalog_path`` into a timestamped archive directory.

    Returns the destination path for observability.  The caller is responsible
    for recreating ``catalog_path`` afterwards if future writes depend on it.
    """
    resolved = catalog_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Catalog path does not exist: {resolved}")
    backup_base = _resolve_backup_root(backup_root, resolved)
    backup_base.mkdir(parents=True, exist_ok=True)
    stamp = (timestamp or datetime.now(tz=UTC)).strftime("%Y%m%d%H%M%S")
    archive_name = f"{resolved.name}_{_ARCHIVE_PREFIX}_{stamp}"
    destination = backup_base / archive_name
    shutil.move(str(resolved), destination)
    return destination


def _resolve_backup_root(candidate: Path | None, catalog_path: Path) -> Path:
    if candidate is None:
        return catalog_path.parent
    return candidate.expanduser().resolve()


__all__ = [
    "archive_catalog",
    "prepare_clean_catalog_path",
]
