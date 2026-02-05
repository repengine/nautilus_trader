from __future__ import annotations

import polars as pl
import pytest

from ml.data.validation import DatasetValidationConfig
from ml.data.validation import DatasetValidationError
from ml.data.validation import validate_dataset


pytestmark = [
    pytest.mark.contracts,
    pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend"),
]


def test_validate_dataset_rejects_non_numeric_features() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [0, 1],
            "instrument_id": ["SPY", "SPY"],
            "feature_text": ["alpha", "beta"],
            "feature_num": [1.0, 2.0],
        },
    )

    with pytest.raises(DatasetValidationError, match="Non-numeric feature columns"):
        validate_dataset(df, config=DatasetValidationConfig())


def test_validate_dataset_rejects_timestamp_reversal() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [1, 0, 2],
            "instrument_id": ["SPY", "SPY", "SPY"],
            "feature_one": [1.0, 1.1, 1.2],
        },
    )

    with pytest.raises(DatasetValidationError, match="Timestamp reversals"):
        validate_dataset(df, config=DatasetValidationConfig())


def test_validate_dataset_rejects_forward_return_misalignment() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "instrument_id": ["SPY", "SPY", "SPY"],
            "close": [100.0, 101.0, 103.0],
            "forward_return": [0.0, 0.0, 0.0],
        },
    )

    cfg = DatasetValidationConfig(
        min_positive_rate=None,
        max_positive_rate=None,
        forward_return_horizon=1,
        forward_return_column="forward_return",
        forward_return_price_column="close",
    )

    with pytest.raises(DatasetValidationError, match="forward_return misaligned"):
        validate_dataset(df, config=cfg)
