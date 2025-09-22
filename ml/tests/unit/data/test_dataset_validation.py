from __future__ import annotations

import polars as pl
import pytest

from ml.data.validation import DatasetValidationConfig
from ml.data.validation import DatasetValidationError
from ml.data.validation import validate_dataset


def test_validate_dataset_enforces_min_rows() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [0],
            "instrument_id": ["SPY.NYSE"],
            "feature_one": [1.0],
            "y": [1],
        },
    )
    cfg = DatasetValidationConfig(min_rows=2)
    with pytest.raises(DatasetValidationError):
        validate_dataset(df, config=cfg)


def test_validate_dataset_requires_macro_series() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "instrument_id": ["SPY.NYSE"] * 3,
            "DGS10": [1.0, 1.1, 1.2],
            "feature_one": [0.1, 0.2, 0.3],
            "y": [0, 1, 0],
        },
    )
    cfg = DatasetValidationConfig(require_macro_series=("DGS10",))
    result = validate_dataset(df, config=cfg)
    assert result.row_count == 3
    assert result.macro_columns_present == ("DGS10",)
