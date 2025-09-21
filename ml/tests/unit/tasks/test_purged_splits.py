from __future__ import annotations

import polars as pl

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
