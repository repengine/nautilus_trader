"""
Dataset split helpers supporting purged/embargoed cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from ml._imports import pd as _pd
from ml._imports import pl as _pl
from ml.preprocessing.stationarity import PurgedCrossValidator


if TYPE_CHECKING:
    from pandas import DataFrame as PandasFrame
else:
    PandasFrame = Any


@dataclass(slots=True, frozen=True)
class PurgedSplitResult:
    train_indices: npt.NDArray[np.int64]
    validation_indices: npt.NDArray[np.int64]


def _to_pandas(df: Any) -> PandasFrame:
    if _pd is None:
        raise RuntimeError("pandas is required for purged split helpers")
    if _pl is not None and isinstance(df, _pl.DataFrame):
        return cast(PandasFrame, df.to_pandas())
    if isinstance(df, _pd.DataFrame):
        return cast(PandasFrame, df)
    raise TypeError("Expected pandas or polars DataFrame")


def create_purged_splits(
    df: Any,
    *,
    timestamp_col: str = "timestamp",
    test_fraction: float = 0.2,
    n_splits: int = 5,
    purge_gap: int = 0,
    embargo_hours: float = 24.0,
) -> dict[str, Any]:
    """
    Create purged cross-validation splits with configurable embargo.
    """
    pdf = _to_pandas(df).copy()
    if _pd is None:
        raise RuntimeError("pandas is required for purged split helpers")
    pdf[timestamp_col] = (
        _pd.to_datetime(pdf[timestamp_col], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    )
    pdf = pdf.sort_values(timestamp_col).reset_index(drop=True)

    n_samples = len(pdf)
    if n_samples < 2:
        raise ValueError("Dataset too small for purged splits")

    test_size = max(int(n_samples * test_fraction), 1)
    train_len = n_samples - test_size
    if train_len < n_splits:
        raise ValueError("Not enough samples for requested splits")

    train_df = pdf.iloc[:train_len]
    train_indices = np.arange(train_len)
    test_indices = np.arange(train_len, n_samples)

    span = train_df[timestamp_col].iloc[-1] - train_df[timestamp_col].iloc[0]
    total_hours = max(span.total_seconds() / 3600.0, 1.0)
    embargo_pct = min(max(embargo_hours / total_hours, 0.0), 0.5)

    cv = PurgedCrossValidator(
        n_splits=n_splits,
        purge_gap=purge_gap,
        embargo_pct=embargo_pct,
    )
    cv_splits = cv.split(train_indices.reshape(-1, 1))

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "cv_splits": cv_splits,
        "embargo_pct": embargo_pct,
    }
