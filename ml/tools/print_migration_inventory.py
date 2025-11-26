"""Emit a deterministic inventory of SQL migrations for review."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from typing import Sequence


@dataclass(frozen=True, slots=True)
class InventoryRecord:
    """Summary of a migration file."""

    migration_id: str
    relative_path: Path
    checksum: str


def _compute_checksum(path: Path, *, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _migration_id_from_name(path: Path) -> str:
    name = path.name
    stem = name.split(".", 1)[0]
    prefix = stem.split("_", 1)[0]
    return prefix


def _iter_sql_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return ()
    return (
        path
        for path in sorted(directory.glob("*.sql"), key=lambda p: p.name)
        if path.is_file()
    )


def build_inventory(*, base_dir: Path, include_archive: bool = True) -> tuple[InventoryRecord, ...]:
    """Collect inventory records for ``base_dir`` (and archive when enabled)."""
    if not base_dir.exists():
        msg = f"Migrations directory not found: {base_dir}"
        raise FileNotFoundError(msg)
    records: list[InventoryRecord] = []
    base_dir = base_dir.resolve()

    def _append_records(root: Path) -> None:
        for path in _iter_sql_files(root):
            relative_path = path.relative_to(base_dir)
            records.append(
                InventoryRecord(
                    migration_id=_migration_id_from_name(path),
                    relative_path=relative_path,
                    checksum=_compute_checksum(path),
                ),
            )

    _append_records(base_dir)
    if include_archive:
        archive_dir = base_dir / "archive"
        if archive_dir.exists():
            _append_records(archive_dir)

    records.sort(key=lambda record: record.relative_path.as_posix())
    return tuple(records)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print migration inventory (id::path::checksum).")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("ml/stores/migrations"),
        help="Path to the migrations directory (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="Do not include the archive directory.",
    )
    parser.add_argument(
        "--algorithm",
        default="sha256",
        help="Checksum algorithm (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m ml.tools.print_migration_inventory``."""
    args = _parse_args(argv)
    inventory = build_inventory(base_dir=args.base, include_archive=not args.skip_archive)
    for record in inventory:
        digest = record.checksum if args.algorithm == "sha256" else _compute_checksum(
            args.base / record.relative_path,
            algorithm=args.algorithm,
        )
        print(f"{record.migration_id}::{record.relative_path.as_posix()}::{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
