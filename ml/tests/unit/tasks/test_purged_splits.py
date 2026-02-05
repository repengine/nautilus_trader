from __future__ import annotations

import polars as pl
import pytest

from ml.tasks.datasets.splits import create_purged_splits


def test_create_purged_splits_basic() -> None:
    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 1),
        end=pl.datetime(2024, 2, 9),
        interval="12h",
        eager=True,
    )
    df = pl.DataFrame({"timestamp": timestamps, "value": list(range(len(timestamps)))})

    splits = create_purged_splits(
        df,
        timestamp_col="timestamp",
        test_fraction=0.2,
        n_splits=4,
        purge_gap=1,
        embargo_hours=24,
    )

    train_indices = splits["train_indices"]
    test_indices = splits["test_indices"]
    cv_splits = splits["cv_splits"]

    assert len(test_indices) == max(int(len(df) * 0.2), 1)
    assert train_indices.max() < test_indices.min()
    assert len(cv_splits) == 4

    for train_idx, val_idx in cv_splits:
        assert len(set(train_idx).intersection(val_idx)) == 0


def test_create_purged_splits_embargo_ratio() -> None:
    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 1),
        end=pl.datetime(2024, 1, 31),
        interval="1d",
        eager=True,
    )
    df = pl.DataFrame({"timestamp": timestamps})

    splits = create_purged_splits(df, embargo_hours=48)
    assert 0 <= splits["embargo_pct"] < 0.5


def test_create_purged_splits_embargo_pct_override() -> None:
    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 1),
        end=pl.datetime(2024, 1, 20),
        interval="1d",
        eager=True,
    )
    df = pl.DataFrame({"timestamp": timestamps, "value": list(range(len(timestamps)))})

    splits = create_purged_splits(
        df,
        embargo_hours=0.0,
        embargo_pct=0.2,
        n_splits=3,
    )

    assert splits["embargo_pct"] == pytest.approx(0.2)
    assert len(splits["cv_splits"]) == 3


def test_create_purged_splits_invalid_embargo_pct() -> None:
    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 1),
        end=pl.datetime(2024, 1, 10),
        interval="1d",
        eager=True,
    )
    df = pl.DataFrame({"timestamp": timestamps})

    with pytest.raises(ValueError, match="embargo_pct must be in"):
        create_purged_splits(df, embargo_pct=1.2)
