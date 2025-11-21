from __future__ import annotations

import polars as pl
import pytest

from ml.data.validation import DatasetValidationConfig
from ml.data.validation import DatasetValidationError
from ml.data.validation import validate_dataset
from ml.data.vintage import VintagePolicy

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")

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
    assert result.macro_observation_counts == {"DGS10": 0}

def test_validate_dataset_enforces_macro_observations() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "instrument_id": ["SPY.NYSE"] * 3,
            "DGS10": [1.0, 1.1, 1.2],
            "DGS10__value_vintage_ts": [None, None, None],
        },
    )
    cfg = DatasetValidationConfig(
        require_macro_series=("DGS10",),
        expected_vintage_policy=VintagePolicy.REAL_TIME,
        macro_min_vintage_observations=1,
        min_feature_coverage=None,
    )
    with pytest.raises(DatasetValidationError):
        validate_dataset(df, config=cfg)

def test_validate_dataset_macro_observations_pass() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "instrument_id": ["SPY.NYSE"] * 3,
            "DGS10": [1.0, 1.1, 1.2],
            "DGS10__value_vintage_ts": [None, 0, None],
        },
    )
    cfg = DatasetValidationConfig(
        require_macro_series=("DGS10",),
        expected_vintage_policy=VintagePolicy.REAL_TIME,
        macro_min_vintage_observations=1,
        min_feature_coverage=None,
    )
    result = validate_dataset(df, config=cfg)
    assert result.macro_observation_counts == {"DGS10": 1}
