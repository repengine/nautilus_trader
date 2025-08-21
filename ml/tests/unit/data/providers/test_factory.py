"""
Unit tests for provider factory and integration.

Tests the factory pattern for creating and managing data providers, and the integration
with feature transforms.

"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import polars as pl
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.data.providers.base import BaseStaticProvider
from ml.data.providers.base import BaseTimeSeriesProvider
from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.providers.events import EventScheduleProvider
from ml.data.providers.factory import ProviderFactory
from ml.data.providers.factory import TransformProviderAdapter
from ml.data.providers.metadata import InstrumentMetadataProvider
from ml.data.sources.calendar import MockCalendarSource
from ml.data.sources.events import MockEventSource
from ml.data.sources.metadata import MockMetadataSource
from ml.features.pipeline import TransformSpec


class TestProviderFactory:
    """
    Test provider factory pattern.
    """

    def test_factory_creates_default_providers(self) -> None:
        """
        Test factory creates providers with default sources.
        """
        factory = ProviderFactory()

        # Should create default providers
        metadata_provider = factory.get_metadata_provider()
        assert isinstance(metadata_provider, InstrumentMetadataProvider)

        calendar_provider = factory.get_calendar_provider()
        assert isinstance(calendar_provider, MarketCalendarProvider)

        event_provider = factory.get_event_provider()
        assert isinstance(event_provider, EventScheduleProvider)

    def test_factory_uses_custom_sources(self) -> None:
        """
        Test factory accepts custom data sources.
        """
        custom_metadata = MockMetadataSource(seed=123)
        custom_calendar = MockCalendarSource()
        custom_events = MockEventSource(seed=456)

        factory = ProviderFactory(
            metadata_source=custom_metadata,
            calendar_source=custom_calendar,
            event_source=custom_events,
        )

        # Providers should use custom sources
        metadata_provider = factory.get_metadata_provider()
        assert metadata_provider.source == custom_metadata

        calendar_provider = factory.get_calendar_provider()
        assert calendar_provider.calendar == custom_calendar

        event_provider = factory.get_event_provider()
        assert event_provider.event_source == custom_events

    def test_factory_singleton_pattern(self) -> None:
        """
        Test factory returns same provider instances.
        """
        factory = ProviderFactory()

        # Should return same instances
        provider1 = factory.get_metadata_provider()
        provider2 = factory.get_metadata_provider()
        assert provider1 is provider2

        cal1 = factory.get_calendar_provider()
        cal2 = factory.get_calendar_provider()
        assert cal1 is cal2

    def test_factory_get_provider_by_name(self) -> None:
        """
        Test getting provider by string name.
        """
        factory = ProviderFactory()

        # Should support string lookup
        metadata = factory.get_provider("metadata")
        assert isinstance(metadata, InstrumentMetadataProvider)

        calendar = factory.get_provider("calendar")
        assert isinstance(calendar, MarketCalendarProvider)

        events = factory.get_provider("events")
        assert isinstance(events, EventScheduleProvider)

        # Invalid name should raise
        with pytest.raises(ValueError, match="Unknown provider"):
            factory.get_provider("invalid")

    def test_factory_register_custom_provider(self) -> None:
        """
        Test registering custom providers.
        """
        factory = ProviderFactory()

        # Create mock custom provider
        custom_provider = MagicMock(spec=BaseStaticProvider)

        # Register it
        factory.register_provider("custom", custom_provider)

        # Should be retrievable
        retrieved = factory.get_provider("custom")
        assert retrieved is custom_provider

    @given(
        n_providers=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=5)
    def test_factory_handles_multiple_custom_providers(self, n_providers: int) -> None:
        """Property test: factory handles arbitrary number of custom providers."""
        factory = ProviderFactory()

        # Register multiple providers
        providers = {}
        for i in range(n_providers):
            provider = MagicMock(spec=BaseTimeSeriesProvider)
            name = f"custom_{i}"
            factory.register_provider(name, provider)
            providers[name] = provider

        # All should be retrievable
        for name, provider in providers.items():
            retrieved = factory.get_provider(name)
            assert retrieved is provider


class TestTransformProviderAdapter:
    """
    Test adapter between transforms and providers.
    """

    def test_adapter_maps_transform_to_provider(self) -> None:
        """
        Test adapter correctly maps transforms to providers.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Calendar transform should map to calendar provider
        calendar_spec = TransformSpec(name="calendar", params={"encoding": "cyclic"})
        provider = adapter.get_provider_for_transform(calendar_spec)
        assert isinstance(provider, MarketCalendarProvider)

        # Static covariates should map to metadata provider
        static_spec = TransformSpec(name="static_covariates", params={})
        provider = adapter.get_provider_for_transform(static_spec)
        assert isinstance(provider, InstrumentMetadataProvider)

        # Event schedule should map to event provider
        event_spec = TransformSpec(name="event_schedule", params={})
        provider = adapter.get_provider_for_transform(event_spec)
        assert isinstance(provider, EventScheduleProvider)

    def test_adapter_loads_transform_data(self) -> None:
        """
        Test adapter loads data for transforms.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Test calendar transform
        calendar_spec = TransformSpec(name="calendar", params={"encoding": "cyclic"})
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 10, 0).timestamp() * 1e9),
                int(datetime(2024, 1, 15, 14, 0).timestamp() * 1e9),
            ],
        )

        df = adapter.load_transform_data(
            transform=calendar_spec,
            timestamps=timestamps,
            instruments=["AAPL"],
        )

        # Should have calendar features
        assert len(df) == 2
        assert "hour_sin" in df.columns
        assert "dow_sin" in df.columns

    def test_adapter_handles_static_data(self) -> None:
        """
        Test adapter handles static covariate data.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        static_spec = TransformSpec(name="static_covariates", params={})

        df = adapter.load_transform_data(
            transform=static_spec,
            timestamps=None,  # Static data doesn't need timestamps
            instruments=["AAPL", "MSFT"],
        )

        # Should have metadata for instruments
        assert len(df) == 2
        assert "tick_size" in df.columns
        assert "exchange" in df.columns

    def test_adapter_caches_providers(self) -> None:
        """
        Test adapter caches provider instances.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        calendar_spec = TransformSpec(name="calendar", params={})

        # Should reuse same provider instance
        provider1 = adapter.get_provider_for_transform(calendar_spec)
        provider2 = adapter.get_provider_for_transform(calendar_spec)
        assert provider1 is provider2

    def test_adapter_handles_unknown_transform(self) -> None:
        """
        Test adapter handles unknown transforms gracefully.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        unknown_spec = TransformSpec(name="unknown_transform", params={})

        # Should return None for unknown transforms
        provider = adapter.get_provider_for_transform(unknown_spec)
        assert provider is None

        # Loading data should return empty DataFrame
        df = adapter.load_transform_data(
            transform=unknown_spec,
            timestamps=pl.Series("timestamp", []),
            instruments=[],
        )
        assert df.is_empty()

    def test_adapter_merges_multi_instrument_data(self) -> None:
        """
        Test adapter correctly merges data for multiple instruments.
        """
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Test with event data
        event_spec = TransformSpec(name="event_schedule", params={})
        timestamps = pl.Series(
            "timestamp",
            [
                int(datetime(2024, 1, 15, 10, 0).timestamp() * 1e9),
            ],
        )

        df = adapter.load_transform_data(
            transform=event_spec,
            timestamps=timestamps,
            instruments=["AAPL", "MSFT", "GOOGL"],
        )

        # Should have features computed
        assert len(df) >= 1
        assert "has_earnings_today" in df.columns

    @given(
        n_timestamps=st.integers(min_value=1, max_value=20),
        n_instruments=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=5)
    def test_adapter_handles_arbitrary_data_sizes(
        self,
        n_timestamps: int,
        n_instruments: int,
    ) -> None:
        """Property test: adapter handles various data sizes."""
        factory = ProviderFactory()
        adapter = TransformProviderAdapter(factory)

        # Generate timestamps
        base = datetime(2024, 1, 1, 10, 0)
        timestamps = []
        for i in range(n_timestamps):
            dt = datetime(2024, 1, 1 + i, 10, 0)
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

        # Should handle any size
        assert len(df) == n_timestamps

    def test_adapter_with_custom_provider(self) -> None:
        """
        Test adapter works with custom registered providers.
        """
        factory = ProviderFactory()

        # Register custom provider - using spec_set=False to allow adding methods
        custom_provider = MagicMock()
        custom_provider.compute_features = MagicMock(
            return_value=pl.DataFrame(
                {
                    "timestamp": [1705312800000000000],
                    "custom_feature": [42.0],
                },
            ),
        )
        factory.register_provider("custom_provider", custom_provider)

        # Register mapping
        adapter = TransformProviderAdapter(factory)
        adapter.register_transform_mapping("custom_transform", "custom_provider")

        # Should use custom provider
        custom_spec = TransformSpec(name="custom_transform", params={})
        provider = adapter.get_provider_for_transform(custom_spec)
        assert provider is custom_provider

        # Should load custom data
        timestamps = pl.Series("timestamp", [1705312800000000000])
        df = adapter.load_transform_data(
            transform=custom_spec,
            timestamps=timestamps,
            instruments=["TEST"],
        )
        assert "custom_feature" in df.columns
        assert df["custom_feature"][0] == 42.0
