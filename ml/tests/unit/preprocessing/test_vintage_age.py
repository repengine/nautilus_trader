from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ml.preprocessing.vintage_age import convert_vintage_timestamps_to_age
from ml.preprocessing.vintage_age import update_metadata_with_vintage_age


def _write_sample_dataset(path: Path) -> None:
    base_ns = 1_000_000_000_000
    table = pa.table(
        {
            "timestamp": pa.array(
                [
                    base_ns + 600_000_000_000,
                    base_ns + 900_000_000_000,
                    base_ns + 1_200_000_000_000,
                ],
                type=pa.int64(),
            ),
            "time_index": pa.array([0, 1, 2], type=pa.int64()),
            "instrument_id": pa.array(["ABC", "ABC", "ABC"]),
            "foo__value_real_time": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            "foo__value_vintage_ts": pa.array(
                [
                    base_ns + 300_000_000_000,
                    base_ns + 600_000_000_000,
                    None,
                ],
                type=pa.timestamp("ns"),
            ),
        },
    )
    pq.write_table(table, path, compression="snappy")


def _write_metadata(path: Path) -> dict[str, object]:
    data = {
        "dataset_id": "test",
        "column_info": {
            "time_varying_known_reals": [
                "timestamp",
                "time_index",
                "foo__value_vintage_ts",
                "foo__value_real_time",
            ],
            "drop_columns": [],
            "vintage_timestamp_columns": ["foo__value_vintage_ts"],
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def test_convert_vintage_columns(tmp_path: Path) -> None:
    source = tmp_path / "dataset.parquet"
    destination = tmp_path / "dataset_with_vintage_age.parquet"
    metadata_path = tmp_path / "dataset_metadata.json"

    _write_sample_dataset(source)
    original_metadata = _write_metadata(metadata_path)

    result = convert_vintage_timestamps_to_age(source, destination)
    assert destination.exists()
    assert result.vintage_columns == ("foo__value_vintage_ts",)
    assert result.age_columns == ("foo__vintage_age_minutes",)

    read_back = pq.read_table(destination)
    assert "foo__value_vintage_ts" not in read_back.column_names
    assert "foo__vintage_age_minutes" in read_back.column_names
    age_values = read_back.column("foo__vintage_age_minutes").to_pylist()
    assert age_values[0] == pytest.approx(5.0)
    assert age_values[1] == pytest.approx(5.0)
    assert age_values[2] is None

    updated = update_metadata_with_vintage_age(
        original_metadata,
        vintage_columns=result.vintage_columns,
        age_columns=result.age_columns,
    )

    assert updated is not original_metadata  # ensures immutability
    assert "foo__value_vintage_ts" in original_metadata["column_info"]["time_varying_known_reals"]  # type: ignore[index]
    assert original_metadata["column_info"]["drop_columns"] == []  # type: ignore[index]

    known_reals = updated["column_info"]["time_varying_known_reals"]  # type: ignore[index]
    assert "foo__value_vintage_ts" not in known_reals
    assert "foo__vintage_age_minutes" in known_reals

    drop_columns = updated["column_info"]["drop_columns"]  # type: ignore[index]
    assert "foo__value_vintage_ts" in drop_columns
    assert updated["column_info"]["vintage_age_columns"] == ["foo__vintage_age_minutes"]  # type: ignore[index]
