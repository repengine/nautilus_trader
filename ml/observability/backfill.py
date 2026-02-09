"""Observability JSONL backfill service (cold path)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.observability.db_persistence import ObservabilityDBPersistor


logger = logging.getLogger(__name__)

DEFAULT_OBSERVABILITY_TABLES: tuple[str, ...] = ("latency", "metrics", "correlation", "health")


@dataclass(slots=True, frozen=True)
class ObservabilityBackfillConfig:
    """
    Runtime configuration for observability JSONL backfill.

    Args:
        src: Base directory containing day-partitioned and/or flat JSONL observability files.
        db_url: SQLAlchemy database URL consumed by ``ObservabilityDBPersistor``.
        table_names: Ordered table stems to backfill from ``<name>*.jsonl`` shards.
    """

    src: Path
    db_url: str
    table_names: tuple[str, ...] = DEFAULT_OBSERVABILITY_TABLES

    def __post_init__(self) -> None:
        if not self.db_url.strip():
            raise ValueError("db_url must not be empty")
        if not self.table_names:
            raise ValueError("table_names must contain at least one table")
        if any(not name.strip() for name in self.table_names):
            raise ValueError("table_names must not contain empty values")


def _load_name_shards(dir_path: Path, name: str) -> list[pd.DataFrame]:
    """
    Load JSONL shards matching ``{name}*.jsonl`` under a directory.

    Args:
        dir_path: Directory containing JSONL shards.
        name: Logical table name/stem (e.g., ``metrics``).

    Returns:
        Parsed DataFrames for each readable shard in lexical filename order.
    """
    frames: list[pd.DataFrame] = []
    for file_path in sorted(dir_path.glob(f"{name}*.jsonl")):
        try:
            frames.append(pd.read_json(file_path, orient="records", lines=True))
        except Exception as exc:  # pragma: no cover - best effort backfill
            logger.debug("Failed reading %s: %s", file_path, exc, exc_info=True)
    return frames


def _load_jsonl_files(base: Path, name: str) -> list[pd.DataFrame]:
    """
    Load day-partitioned and root-level JSONL shards for a table.

    Args:
        base: Base directory that may contain ``YYYY-MM-DD`` subdirectories and flat shards.
        name: Logical table name/stem (e.g., ``latency``).

    Returns:
        Parsed shard DataFrames in deterministic order.
    """
    frames: list[pd.DataFrame] = []
    for day_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        frames.extend(_load_name_shards(day_dir, name))
    frames.extend(_load_name_shards(base, name))
    return frames


def collect_observability_backfill_tables(
    base: Path,
    *,
    table_names: tuple[str, ...] = DEFAULT_OBSERVABILITY_TABLES,
) -> dict[str, pd.DataFrame]:
    """
    Collect concatenated DataFrames per observability table from JSONL shards.

    Args:
        base: Source directory containing JSONL files.
        table_names: Table stems to scan and concatenate.

    Returns:
        Mapping of table name to concatenated DataFrame for non-empty tables.
    """
    tables: dict[str, pd.DataFrame] = {}
    for name in table_names:
        frames = _load_jsonl_files(base, name)
        if frames:
            tables[name] = pd.concat(frames, ignore_index=True)
    return tables


def backfill_observability_tables(
    config: ObservabilityBackfillConfig,
    *,
    emit: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """
    Backfill observability tables from persisted JSONL files.

    Args:
        config: Backfill source and DB persistence configuration.
        emit: Optional sink for user-visible status lines.

    Returns:
        Row-count mapping emitted by ``ObservabilityDBPersistor.persist``.
    """
    tables = collect_observability_backfill_tables(
        config.src,
        table_names=config.table_names,
    )
    if not tables:
        if emit is not None:
            emit("No observability files found")
        return {}

    persistor = ObservabilityDBPersistor(connection_string=config.db_url)
    out = persistor.persist(tables)
    if emit is not None:
        for table_name, row_count in out.items():
            emit(f"{table_name}: {row_count}")
    return out


__all__ = [
    "DEFAULT_OBSERVABILITY_TABLES",
    "ObservabilityBackfillConfig",
    "backfill_observability_tables",
    "collect_observability_backfill_tables",
]
