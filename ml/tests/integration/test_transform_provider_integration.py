"""
Integration tests for transform-provider connections.

Tests the end-to-end integration of feature transforms with data providers, ensuring
that transforms can properly load data through providers.

"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta

import polars as pl
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.data.providers.factory import ProviderFactory
from ml.data.providers.factory import TransformProviderAdapter
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec


@pytest.mark.property
@pytest.mark.parallel_safe
@pytest.mark.integration
class TestTransformProviderIntegration:
    """
    Test integration between transforms and providers.
    """

    def test_calendar_transform_integration(self) -> None:
        """
        Test calendar transform loads data through provider.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Create calendar transform spec
        calendar_spec = TransformSpec(
            name="calendar",
            params={"encoding": "cyclic", "granularity": "hour"},
        )

        # Create timestamps
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 9, 30).timestamp() * 1e9),  # Monday morning
                int(datetime(2024, 1, 15, 15, 30).timestamp() * 1e9),  # Monday afternoon
                int(datetime(2024, 1, 16, 12, 0).timestamp() * 1e9),  # Tuesday noon
                int(datetime(2024, 1, 20, 10, 0).timestamp() * 1e9),  # Saturday morning
            ],
        )

        # Load data through adapter
        df = adapter.load_transform_data(
            transform=calendar_spec,
            timestamps=timestamps,
            instruments=["AAPL"],
        )

        # Verify calendar features
        assert len(df) == 4
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        assert "dow_sin" in df.columns
        assert "dow_cos" in df.columns
        assert "is_weekend" in df.columns

        # Check specific values
        assert df["is_weekend"][0] is False  # Monday
        assert df["is_weekend"][3] is True  # Saturday

    def test_metadata_transform_integration(self) -> None:
        """
        Test static covariates transform loads metadata.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Create static covariates transform spec
        static_spec = TransformSpec(
            name="static_covariates",
            params={
                "numeric_features": ["tick_size", "lot_size"],
                "categorical_features": ["exchange", "asset_class"],
            },
        )

        # Load metadata for instruments
        df = adapter.load_transform_data(
            transform=static_spec,
            timestamps=None,  # Static data doesn't need timestamps
            instruments=["AAPL", "MSFT", "GOOGL"],
        )

        # Verify metadata features
        assert len(df) == 3
        assert "tick_size" in df.columns
        assert "lot_size" in df.columns
        assert "exchange" in df.columns
        assert "asset_class" in df.columns

        # Check all instruments present
        assert set(df["instrument_id"].to_list()) == {"AAPL", "MSFT", "GOOGL"}

    def test_event_transform_integration(self) -> None:
        """
        Test event schedule transform loads event data.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Create event schedule transform spec
        event_spec = TransformSpec(
            name="event_schedule",
            params={
                "lookback_days": 7,
                "lookahead_days": 7,
            },
        )

        # Create timestamps around potential events
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 10, 12, 0).timestamp() * 1e9),
                int(datetime(2024, 1, 15, 12, 0).timestamp() * 1e9),
                int(datetime(2024, 1, 31, 14, 0).timestamp() * 1e9),  # Fed meeting day
            ],
        )

        # Load event data
        df = adapter.load_transform_data(
            transform=event_spec,
            timestamps=timestamps,
            instruments=["AAPL", "MSFT"],
        )

        # Verify event features
        assert len(df) == 3
        assert "has_fed_event_today" in df.columns
        assert "has_earnings_today" in df.columns
        assert "days_to_next_fed" in df.columns
        assert "event_importance_score" in df.columns

    def test_pipeline_with_providers(self) -> None:
        """
        Test full pipeline with multiple transforms and providers.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Create pipeline with multiple transforms
        pipeline = PipelineSpec(
            transforms=[
                TransformSpec(name="calendar", params={"encoding": "cyclic"}),
                TransformSpec(name="event_schedule", params={}),
            ],
        )

        # Create timestamps
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 10, 0).timestamp() * 1e9),
                int(datetime(2024, 1, 16, 10, 0).timestamp() * 1e9),
            ],
        )

        # Load data for each transform
        all_features: list[pl.DataFrame] = []

        for transform_spec in pipeline.transforms:
            df = adapter.load_transform_data(
                transform=transform_spec,
                timestamps=timestamps,
                instruments=["AAPL"],
            )

            if not df.is_empty():
                # Remove duplicate timestamp column if present
                if "timestamp" in df.columns and len(all_features) > 0:
                    df = df.drop("timestamp")
                all_features.append(df)

        # Combine all features
        if all_features:
            if len(all_features) == 1:
                combined = all_features[0]
            else:
                # Join on index (assuming same order)
                combined = pl.concat(all_features, how="horizontal")

            # Verify combined features
            assert len(combined) == 2
            # Calendar features
            assert "hour_sin" in combined.columns
            # Event features
            assert "has_fed_event_today" in combined.columns

    def test_transform_feature_consistency(self) -> None:
        """
        Test that transform feature names match loaded data columns.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Test calendar transform
        from ml.features.pipeline import _CalendarTransform

        calendar_transform = _CalendarTransform()
        params = {"encoding": "cyclic", "granularity": "hour"}
        expected_features = calendar_transform.feature_names(params)

        # Load actual data
        calendar_spec = TransformSpec(name="calendar", params=params)
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 10, 0).timestamp() * 1e9),
            ],
        )

        df = adapter.load_transform_data(
            transform=calendar_spec,
            timestamps=timestamps,
            instruments=["AAPL"],
        )

        # Check that expected features are in loaded data
        # (Note: Provider may include additional features)
        for feature in expected_features:
            if feature not in [
                "minute_sin",
                "minute_cos",
            ]:  # These are optional based on granularity
                assert feature in df.columns, f"Expected feature {feature} not in loaded data"

    @given(
        n_timestamps=st.integers(min_value=1, max_value=50),
        n_instruments=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=5)
    def test_provider_scalability(self, n_timestamps: int, n_instruments: int) -> None:
        """Property test: providers handle various data scales."""
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Generate timestamps
        base = datetime(2024, 1, 1, 10, 0)
        timestamps = []
        for i in range(n_timestamps):
            dt = base + timedelta(hours=i)
            timestamps.append(int(dt.timestamp() * 1e9))

        ts_series = pl.Series("timestamp", timestamps)

        # Generate instruments
        instruments = [f"INST_{i}" for i in range(n_instruments)]

        # Load calendar data
        calendar_spec = TransformSpec(name="calendar", params={})
        df = adapter.load_transform_data(
            transform=calendar_spec,
            timestamps=ts_series,
            instruments=instruments,
        )

        # Should handle any scale
        assert len(df) == n_timestamps
        assert not df.is_empty()

    def test_provider_caching_efficiency(self) -> None:
        """
        Test that providers efficiently cache data.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Create timestamps
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 10, 0).timestamp() * 1e9),
            ],
        )

        # First load
        event_spec = TransformSpec(name="event_schedule", params={})
        df1 = adapter.load_transform_data(
            transform=event_spec,
            timestamps=timestamps,
            instruments=["AAPL"],
        )

        # Second load with same parameters
        df2 = adapter.load_transform_data(
            transform=event_spec,
            timestamps=timestamps,
            instruments=["AAPL"],
        )

        # Should return same data (cached)
        assert df1.equals(df2)

        # Provider should be cached in adapter
        provider1 = adapter.get_provider_for_transform(event_spec)
        provider2 = adapter.get_provider_for_transform(event_spec)
        assert provider1 is provider2
