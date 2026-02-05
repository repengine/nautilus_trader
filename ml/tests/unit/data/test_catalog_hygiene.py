from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ml.data.catalog_hygiene import archive_catalog
from ml.data.catalog_hygiene import prepare_clean_catalog_path


def test_prepare_clean_catalog_path_creates_missing_directory(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog"

    resolved = prepare_clean_catalog_path(catalog_path=catalog_path)

    assert resolved.exists()
    assert resolved.is_dir()
    assert list(resolved.iterdir()) == []


def test_prepare_clean_catalog_path_archives_existing_catalog(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog"
    catalog_path.mkdir()
    (catalog_path / "old.txt").write_text("stale", encoding="utf-8")

    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    resolved = prepare_clean_catalog_path(catalog_path=catalog_path, timestamp=stamp)

    assert resolved.exists()
    assert list(resolved.iterdir()) == []

    archive_name = f"{catalog_path.name}_archive_{stamp.strftime('%Y%m%d%H%M%S')}"
    archive_path = tmp_path / archive_name
    assert archive_path.exists()
    assert (archive_path / "old.txt").exists()


def test_archive_catalog_uses_backup_root(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog"
    catalog_path.mkdir()
    (catalog_path / "file.txt").write_text("data", encoding="utf-8")
    backup_root = tmp_path / "backups"

    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    destination = archive_catalog(
        catalog_path=catalog_path,
        backup_root=backup_root,
        timestamp=stamp,
    )

    assert destination.parent == backup_root.resolve()
    assert destination.exists()
    assert not catalog_path.exists()


def test_archive_catalog_raises_when_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        archive_catalog(catalog_path=missing_path)
