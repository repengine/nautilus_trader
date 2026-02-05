"""
Dataset CSV output helpers for dataset builds.

Centralizes CSV sampling and write controls so build pipelines share
one implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ml.ml_types import PolarsDF


class DatasetCsvConfig(Protocol):
    """
    Protocol describing the CSV-related dataset build configuration.

    Attributes:
        write_csv: Force writing full CSV output when True/False; None for auto.
        csv_max_rows: Maximum rows for auto CSV emission.
        csv_sample_rows: Number of sample rows to emit when full CSV is skipped.
    """

    @property
    def write_csv(self) -> bool | None:  # pragma: no cover - protocol declaration
        ...

    @property
    def csv_max_rows(self) -> int:  # pragma: no cover - protocol declaration
        ...

    @property
    def csv_sample_rows(self) -> int:  # pragma: no cover - protocol declaration
        ...


def resolve_write_csv(cfg: DatasetCsvConfig, row_count: int) -> bool:
    """
    Determine whether to write the full dataset CSV for the given row count.

    Args:
        cfg: Dataset build config carrying CSV settings.
        row_count: Number of rows in the dataset.

    Returns:
        True when a full CSV should be emitted.

    Example:
        >>> class _Cfg:
        ...     write_csv = None
        ...     csv_max_rows = 10
        ...     csv_sample_rows = 5
        >>> resolve_write_csv(_Cfg(), row_count=3)
        True
    """
    if cfg.write_csv is not None:
        return bool(cfg.write_csv)
    max_rows = max(int(cfg.csv_max_rows), 0)
    return row_count <= max_rows


def write_dataset_csv(
    df_sorted: PolarsDF,
    cfg: DatasetCsvConfig,
    *,
    dataset_csv: Path,
) -> Path | None:
    """
    Write dataset CSV output or a sample CSV when configured.

    Args:
        df_sorted: Polars DataFrame sorted by timestamp.
        cfg: Dataset build config carrying CSV settings.
        dataset_csv: Path for the full dataset CSV.

    Returns:
        The written CSV path, or None when no CSV is emitted.

    Example:
        >>> class _Cfg:
        ...     write_csv = False
        ...     csv_max_rows = 0
        ...     csv_sample_rows = 10
        >>> _ = write_dataset_csv  # doctest: +ELLIPSIS
    """
    row_count = int(df_sorted.height)
    write_full = resolve_write_csv(cfg, row_count)
    sample_rows = max(int(cfg.csv_sample_rows), 0)

    if write_full:
        df_sorted.write_csv(str(dataset_csv))
        return dataset_csv

    if dataset_csv.exists():
        dataset_csv.unlink()

    if sample_rows <= 0:
        return None

    sample_path = dataset_csv.with_name("dataset_sample.csv")
    sample_df = df_sorted.head(sample_rows)
    sample_df.write_csv(str(sample_path))
    return sample_path
