"""
Utilities for converting macro vintage timestamps into age features.

This module provides batch-aware helpers which operate directly on Parquet files using
PyArrow while keeping memory usage bounded. The primary entry point,
``convert_vintage_timestamps_to_age``, scans an input parquet file, replaces every
``*_value_vintage_ts`` timestamp column with a numeric ``*_vintage_age_minutes`` column,
and streams the transformed data to an output parquet file.

All functions are typed and safe for use in cold paths such as dataset build steps.

Example:
    >>> from pathlib import Path
    >>> from ml.preprocessing.vintage_age import convert_vintage_timestamps_to_age
    >>> src = Path("ml_out/full_tft_95/dataset.parquet")
    >>> dst = src.with_name("dataset_with_vintage_age.parquet")
    >>> result = convert_vintage_timestamps_to_age(src, dst)
    >>> result.age_columns[:2]
    ('BAMLC0A0CM__vintage_age_minutes', 'BAMLH0A0HYM2__vintage_age_minutes')
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ml.data.phase_one_signals import derive_phase_one_signals


_TIMESTAMP_SUFFIX = "__value_vintage_ts"
_AGE_SUFFIX = "__vintage_age_minutes"
_NANOSECONDS_PER_MINUTE = 60_000_000_000


@dataclass(frozen=True)
class VintageConversionResult:
    """Summary of the conversion step."""

    vintage_columns: tuple[str, ...]
    age_columns: tuple[str, ...]


def _ensure_parquet_file(path: Path) -> None:
    if not path.exists():
        msg = f"Parquet file not found: {path}"
        raise FileNotFoundError(msg)
    if path.suffix.lower() != ".parquet":
        msg = f"Expected .parquet file, received: {path}"
        raise ValueError(msg)


def _validate_timestamp_column(schema: pa.Schema, timestamp_column: str) -> None:
    field = schema.field(timestamp_column)
    if not (pa.types.is_integer(field.type) or pa.types.is_timestamp(field.type)):
        msg = (
            f"Timestamp column '{timestamp_column}' must be stored as int64 nanoseconds or "
            "pyarrow timestamp."
        )
        raise ValueError(msg)


def _derive_vintage_columns(schema: pa.Schema, suffix: str) -> tuple[str, ...]:
    return tuple(name for name in schema.names if name.endswith(suffix))


def _compute_age_minutes_array(
    timestamp_ns: pa.Array,
    vintage_ts: pa.Array,
    *,
    divisor: pa.Scalar,
) -> pa.Array:
    timestamp_int = pc.cast(timestamp_ns, pa.int64())
    vintage_int = pc.cast(vintage_ts, pa.int64())
    delta = pc.subtract(timestamp_int, vintage_int)
    minutes = pc.divide(delta, divisor)
    return pc.cast(minutes, pa.float32())


def convert_vintage_timestamps_to_age(
    source: Path,
    destination: Path,
    *,
    timestamp_column: str = "timestamp",
    batch_size: int = 32768,
    compression: str = "snappy",
    vintage_suffix: str = _TIMESTAMP_SUFFIX,
    age_suffix: str = _AGE_SUFFIX,
) -> VintageConversionResult:
    """
    Convert vintage timestamp columns to age-in-minutes features.

    Args:
        source: Path to the parquet file containing ``*_value_vintage_ts`` columns.
        destination: Output parquet file that will receive the transformed table.
        timestamp_column: Column providing event timestamps stored as int64 nanoseconds.
        batch_size: Maximum number of rows read per batch during streaming.
        compression: Compression codec applied to the output parquet file.
        vintage_suffix: Suffix identifying columns that contain vintage timestamps.
        age_suffix: Suffix applied to the derived age feature columns.

    Returns:
        VintageConversionResult: Names of original timestamp columns and their
            corresponding age feature replacements.

    Raises:
        FileNotFoundError: If the source parquet file does not exist.
        ValueError: When timestamp column types or suffix parameters are invalid.

    Example:
        >>> from pathlib import Path
        >>> src = Path("dataset.parquet")
        >>> dst = Path("dataset_with_age.parquet")
        >>> convert_vintage_timestamps_to_age(src, dst)
        VintageConversionResult(...)
    """
    _ensure_parquet_file(source)
    if destination == source:
        raise ValueError("Destination parquet path must differ from the source path.")

    parquet = pq.ParquetFile(source)
    _validate_timestamp_column(parquet.schema_arrow, timestamp_column)
    vintage_columns = _derive_vintage_columns(parquet.schema_arrow, vintage_suffix)
    if not vintage_columns:
        msg = f"No columns matching '*{vintage_suffix}' found in {source}"
        raise ValueError(msg)

    age_columns = tuple(column.replace(vintage_suffix, age_suffix) for column in vintage_columns)
    divisor = pa.scalar(_NANOSECONDS_PER_MINUTE, pa.int64())
    writer: pq.ParquetWriter | None = None

    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            timestamp_array = table[timestamp_column]
            replacements: dict[str, tuple[str, pa.Array]] = {}
            for column, age_column in zip(vintage_columns, age_columns):
                replacements[column] = (
                    age_column,
                    _compute_age_minutes_array(timestamp_array, table[column], divisor=divisor),
                )

            new_arrays: list[pa.Array] = []
            new_names: list[str] = []
            for name in table.column_names:
                if name in replacements:
                    replacement_name, replacement_array = replacements[name]
                    new_arrays.append(replacement_array)
                    new_names.append(replacement_name)
                else:
                    new_arrays.append(table[name])
                    new_names.append(name)

            transformed = pa.Table.from_arrays(new_arrays, names=new_names)
            if writer is None:
                writer = pq.ParquetWriter(destination, transformed.schema, compression=compression)
            writer.write_table(transformed)
    finally:
        if writer is not None:
            writer.close()

    return VintageConversionResult(vintage_columns=vintage_columns, age_columns=age_columns)


def update_metadata_with_vintage_age(
    metadata: dict[str, object],
    *,
    vintage_columns: Sequence[str],
    age_columns: Sequence[str],
) -> dict[str, object]:
    """
    Return metadata with time-varying known reals updated for age features.

    Args:
        metadata: Parsed dataset metadata to mutate.
        vintage_columns: Original timestamp column names.
        age_columns: Replacement age feature column names.

    Returns:
        A deep copy of the metadata dictionary with updated column listings.

    Example:
        >>> updated = update_metadata_with_vintage_age(metadata, ...)
        >>> updated["column_info"]["time_varying_known_reals"][-1]
        'GDP__vintage_age_minutes'
    """
    cloned = deepcopy(metadata)
    column_info = cloned.get("column_info")
    if not isinstance(column_info, dict):
        msg = "Metadata missing 'column_info' dictionary."
        raise ValueError(msg)

    known_reals = [
        name
        for name in column_info.get("time_varying_known_reals", [])
        if isinstance(name, str) and name not in set(vintage_columns)
    ]
    known_reals.extend(name for name in age_columns if name not in known_reals)

    drop_columns = list(column_info.get("drop_columns", []))
    for column in vintage_columns:
        if column not in drop_columns:
            drop_columns.append(column)

    column_info["time_varying_known_reals"] = known_reals
    column_info["drop_columns"] = drop_columns
    column_info["vintage_handling"] = {
        "strategy": "age_features",
        "reason": "Converted *_value_vintage_ts columns into *_vintage_age_minutes features.",
    }
    column_info["vintage_timestamp_columns"] = list(vintage_columns)
    column_info["vintage_age_columns"] = list(age_columns)

    cloned["column_info"] = column_info
    column_names: list[str] = []
    for key in ("static_reals", "time_varying_known_reals", "time_varying_unknown_reals"):
        values = column_info.get(key, [])
        if isinstance(values, list | tuple):
            column_names.extend(str(name) for name in values if isinstance(name, str))
    phase_one = derive_phase_one_signals(column_names)
    cloned["phase_one_signals"] = {key: list(values) for key, values in phase_one.items()}
    return cloned


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    """Persist metadata dictionary to disk as JSON with newline termination."""
    text = json.dumps(metadata, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")


__all__ = [
    "VintageConversionResult",
    "convert_vintage_timestamps_to_age",
    "update_metadata_with_vintage_age",
    "write_metadata",
]
