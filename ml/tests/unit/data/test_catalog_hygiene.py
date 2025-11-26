from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ml.data.catalog_hygiene import archive_catalog
from ml.data.catalog_hygiene import prepare_clean_catalog_path


def test_prepare_clean_catalog_path_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "catalog"
    prepare_clean_catalog_path(catalog_path=target)
    assert target.exists()
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_prepare_clean_catalog_path_archives_existing_contents(tmp_path: Path) -> None:
    target = tmp_path / "catalog"
    target.mkdir()
    payload = target / "sample.txt"
    payload.write_text("payload", encoding="utf-8")
    backup_root = tmp_path / "archives"
    ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    prepare_clean_catalog_path(
        catalog_path=target,
        backup_root=backup_root,
        timestamp=ts,
    )
    assert target.exists()
    assert target.is_dir()
    assert list(target.iterdir()) == []
    archives = list(backup_root.iterdir())
    assert len(archives) == 1
    archived = archives[0]
    assert archived.name.startswith("catalog_archive_20250102030405")
    archived_payload = archived / "sample.txt"
    assert archived_payload.exists()
    assert archived_payload.read_text(encoding="utf-8") == "payload"


def test_archive_catalog_missing(tmp_path: Path) -> None:
    target = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        archive_catalog(catalog_path=target)
