"""Tests for FeatureAlignmentComponent canonical delegation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml.data.common.feature_alignment import FeatureAlignmentComponent
from ml.data.common.feature_alignment import _OHLCV_TRANSFORMS
from ml.features.config import build_pipeline_spec_from_feature_config
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec


@pytest.fixture
def component() -> FeatureAlignmentComponent:
    """Provide a FeatureAlignmentComponent instance."""
    return FeatureAlignmentComponent()


@pytest.fixture
def sample_ohlcv_polars_df() -> pl.DataFrame:
    """Create a sample Polars DataFrame with OHLCV + timestamp columns."""
    base_ts = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    timestamps = [base_ts + timedelta(minutes=i) for i in range(64)]

    rng = np.random.default_rng(42)
    base_price = 100.0
    prices = base_price + np.cumsum(rng.standard_normal(64) * 0.1)
    prices = np.maximum(prices, 1.0)

    high_adj = rng.uniform(0, 1, 64)
    low_adj = rng.uniform(0, 1, 64)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices + rng.uniform(-0.3, 0.3, 64),
            "high": prices + high_adj,
            "low": prices - low_adj,
            "close": prices,
            "volume": rng.integers(1000, 10000, 64).astype(float),
            "instrument_id": ["SPY"] * 64,
        }
    )


@pytest.fixture
def sample_ohlcv_pandas_df(sample_ohlcv_polars_df: pl.DataFrame) -> pd.DataFrame:
    """Create a Pandas DataFrame matching the Polars fixture."""
    return sample_ohlcv_polars_df.to_pandas()


def _expected_feature_names(component: FeatureAlignmentComponent) -> list[str]:
    full_spec = build_pipeline_spec_from_feature_config(component._feature_config)
    ohlcv_transforms = [ts for ts in full_spec.transforms if ts.name in _OHLCV_TRANSFORMS]
    runner = PipelineRunner(
        PipelineSpec(transforms=ohlcv_transforms),
        allowable=component._feature_config.resolved_data_requirements(),
    )
    return runner.compute_feature_names()


def test_compute_features_polars_when_sample_data_returns_canonical_columns(
    component: FeatureAlignmentComponent,
    sample_ohlcv_polars_df: pl.DataFrame,
) -> None:
    """Compute features via Polars and verify canonical columns."""
    features = component.compute_features_polars(sample_ohlcv_polars_df)
    expected = _expected_feature_names(component)

    assert features.columns == expected
    assert len(features) == len(sample_ohlcv_polars_df)


def test_compute_features_pandas_when_sample_data_returns_canonical_columns(
    component: FeatureAlignmentComponent,
    sample_ohlcv_pandas_df: pd.DataFrame,
) -> None:
    """Compute features via Pandas and verify canonical columns."""
    features = component.compute_features_pandas(sample_ohlcv_pandas_df)
    expected = _expected_feature_names(component)

    assert list(features.columns) == expected
    assert len(features) == len(sample_ohlcv_pandas_df)


def test_compute_features_polars_when_empty_frame_returns_empty_columns(
    component: FeatureAlignmentComponent,
) -> None:
    """Empty Polars input should return empty canonical columns."""
    empty = pl.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    features = component.compute_features_polars(empty)
    expected = _expected_feature_names(component)

    assert features.columns == expected
    assert len(features) == 0


def test_compute_features_pandas_when_empty_frame_returns_empty_columns(
    component: FeatureAlignmentComponent,
) -> None:
    """Empty Pandas input should return empty canonical columns."""
    empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    features = component.compute_features_pandas(empty)
    expected = _expected_feature_names(component)

    assert list(features.columns) == expected
    assert len(features) == 0


def test_compute_features_polars_when_called_matches_canonical_output(
    component: FeatureAlignmentComponent,
    sample_ohlcv_polars_df: pl.DataFrame,
) -> None:
    """Legacy entrypoint should delegate to canonical computation."""
    features = component.compute_features_polars(sample_ohlcv_polars_df)
    canonical = component.compute_features_canonical_polars(sample_ohlcv_polars_df)

    assert features.columns == canonical.columns
    assert features.rows() == canonical.rows()


def test_add_static_features_polars_when_instrument_present_returns_expected_values(
    component: FeatureAlignmentComponent,
) -> None:
    """Static features should map known instruments in Polars."""
    df = pl.DataFrame({"instrument_id": ["SPY"], "close": [450.0]})
    result = component.add_static_features_polars(df)

    assert result["asset_class"][0] == "ETF"
    assert result["exchange"][0] == "ARCA"


def test_add_static_features_pandas_when_missing_instrument_id_raises(
    component: FeatureAlignmentComponent,
) -> None:
    """Missing instrument_id should raise for Pandas static features."""
    df = pd.DataFrame({"close": [450.0]})

    with pytest.raises(ValueError, match="instrument_id"):
        component.add_static_features_pandas(df)


def test_append_macro_delta_features_polars_when_enabled_appends_delta(
    component: FeatureAlignmentComponent,
) -> None:
    """Macro delta helper should append delta columns when enabled."""
    df = pl.DataFrame(
        {
            "timestamp": [1, 2],
            "instrument_id": ["SPY", "SPY"],
            "PAYEMS": [100.0, 101.0],
        }
    )

    result = component.append_macro_delta_features_polars(
        df,
        include_macro=True,
        include_macro_deltas=True,
        macro_series_ids=("PAYEMS",),
    )

    assert "PAYEMS_delta_1d" in result.columns
    assert result["PAYEMS_delta_1d"].to_list() == [0.0, 1.0]
