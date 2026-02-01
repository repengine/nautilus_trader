from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ml.data import _compute_dataset_metadata
from ml.data import _write_feature_npz_from_polars
from ml.data import DatasetMetadata
from ml.data.vintage import VintagePolicy


def _make_frame(rows: int, *, start: datetime) -> pl.DataFrame:
    timestamps = pl.Series(
        "timestamp",
        [start + timedelta(minutes=idx) for idx in range(rows)],
        dtype=pl.Datetime(time_zone="UTC"),
    )
    return pl.DataFrame(
        {
            "time_index": list(range(rows)),
            "timestamp": timestamps,
            "feature_a": [float(idx) for idx in range(rows)],
            "feature_b": [float(idx * 2) for idx in range(rows)],
            "y": [0] * rows,
        },
    )


def test_write_feature_npz_from_polars_chunked(tmp_path: Path) -> None:
    frame = _make_frame(6, start=datetime(2024, 1, 1, tzinfo=UTC))
    out_path = tmp_path / "features.npz"

    _write_feature_npz_from_polars(
        frame,
        ["feature_a", "feature_b"],
        out_path=out_path,
        cutoff=4,
        chunk_size=2,
    )

    with np.load(out_path) as data:
        assert data["X_train"].shape == (4, 2)
        assert data["X_val"].shape == (2, 2)
        assert list(data["feature_names"]) == ["feature_a", "feature_b"]


def test_compute_metadata_from_polars(tmp_path: Path) -> None:
    frame = _make_frame(5, start=datetime(2024, 1, 1, tzinfo=UTC))
    frame = frame.sort("time_index")

    metadata = _compute_dataset_metadata(
        frame,
        cutoff=3,
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_as_of=None,
        build_ts=datetime(2024, 1, 1, tzinfo=UTC),
        dataset_id="test_dataset",
        macro_observation_counts={"CPI": 5},
        target_semantics=None,
    )

    assert isinstance(metadata, DatasetMetadata)
    assert metadata.dataset_id == "test_dataset"
    assert metadata.train_window is not None
    assert metadata.validation_window is not None
    assert metadata.macro_observation_counts["CPI"] == 5
    assert metadata.capability_flags == {}
