"""Tests for KnownFutureFeatureComponent canonical outputs."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import polars as pl
from ml.data.common.known_future_features import KnownFutureFeatureComponent
from ml.features.pipeline import PipelineRunner


def _expected_feature_names(component: KnownFutureFeatureComponent) -> list[str]:
    spec = component._build_canonical_spec()
    runner = PipelineRunner(
        spec,
        allowable=component._feature_config.resolved_data_requirements(),
    )
    return runner.compute_feature_names()


def test_known_future_features_polars_when_calendar_enabled_returns_canonical_columns() -> None:
    """Polars calendar features should align to the canonical pipeline spec."""
    component = KnownFutureFeatureComponent(include_calendar=True)
    ts = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
    df = pl.DataFrame({"timestamp": [ts]})

    out = component.add_known_future_features_canonical_polars(df)
    expected = _expected_feature_names(component)

    for name in expected:
        assert name in out.columns
    assert "tod_sin" not in out.columns
    assert "is_market_open" not in out.columns


def test_known_future_features_pandas_when_calendar_enabled_returns_canonical_columns() -> None:
    """Pandas calendar features should align to the canonical pipeline spec."""
    component = KnownFutureFeatureComponent(include_calendar=True)
    ts = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
    df = pd.DataFrame({"timestamp": [ts]})

    out = component.add_known_future_features_canonical_pandas(df)
    expected = _expected_feature_names(component)

    for name in expected:
        assert name in out.columns
    assert "tod_cos" not in out.columns
    assert "is_premarket" not in out.columns


def test_known_future_features_polars_when_event_schedule_enabled_includes_events() -> None:
    """Event schedule features should be appended when enabled."""
    component = KnownFutureFeatureComponent(include_calendar=True, include_event_schedule=True)
    ts = datetime(2024, 1, 15, 15, 0, tzinfo=UTC)
    df = pl.DataFrame({"timestamp": [ts], "instrument_id": ["SPY"]})

    out = component.add_known_future_features_canonical_polars(df)

    assert "hours_to_earnings" in out.columns
    assert "event_density_week" in out.columns
