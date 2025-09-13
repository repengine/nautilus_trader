"""
Unit tests for data provider base classes and protocols.

Following TDD approach - tests written before implementation.

"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.tests.builders import DataBuilder


if TYPE_CHECKING:
    import polars as pl


@pytest.mark.property
@pytest.mark.parallel_safe
@pytest.mark.unit
class TestProviderProtocols:
    """
    Test that provider protocols are properly enforced.
    """

    def test_data_provider_protocol_enforcement(self) -> None:
        """
        Test that DataProvider protocol is enforced.
        """
        from ml.data.providers.base import DataProvider

        # This should fail at runtime if protocol not satisfied
        class InvalidProvider:
            pass

        class ValidProvider:
            def load_data(
                self,
                instruments: list[str],
                start: datetime,
                end: datetime,
            ) -> pl.DataFrame:
                return pl.DataFrame()

            def validate_data(self, data: pl.DataFrame) -> bool:
                return True

            def get_schema(self) -> dict[str, type]:
                return {}

        # Valid provider should work
        provider: DataProvider = ValidProvider()
        assert provider.load_data(["SPY"], datetime.now(), datetime.now()) is not None

        # Invalid provider should be detectable
        invalid = InvalidProvider()
        assert not hasattr(invalid, "load_data")

    def test_static_provider_protocol(self) -> None:
        """
        Test StaticDataProvider protocol.
        """
        from ml.data.providers.base import StaticDataProvider

        class ValidStaticProvider:
            def load_metadata(self, instruments: list[str]) -> pl.DataFrame:
                return pl.DataFrame({"instrument_id": instruments})

        provider: StaticDataProvider = ValidStaticProvider()
        result = provider.load_metadata(["SPY"])
        assert len(result) == 1

    def test_cacheable_provider_protocol(self) -> None:
        """
        Test CacheableProvider protocol.
        """
        from ml.data.providers.base import CacheableProvider

        class ValidCacheableProvider:
            def cache_key(self, params: dict[str, Any]) -> str:
                return "test_key"

            def from_cache(self, key: str) -> pl.DataFrame | None:
                return None

            def to_cache(self, key: str, data: pl.DataFrame) -> None:
                pass

        provider: CacheableProvider = ValidCacheableProvider()
        assert provider.cache_key({}) == "test_key"


class TestBaseDataProvider:
    """
    Test base data provider implementation.
    """

    def test_base_provider_initialization(self) -> None:
        """
        Test that base provider initializes correctly.
        """
        from ml.data.providers.base import BaseDataProvider

        provider = BaseDataProvider()
        assert provider.logger is not None
        assert provider.metrics is not None

    def test_base_provider_validation(self) -> None:
        """
        Test data validation in base provider.
        """
        from ml.data.providers.base import BaseDataProvider

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        provider = BaseDataProvider()

        # Valid data with required columns - using DataBuilder for consistent test data
        timestamps = DataBuilder.time_series(n_points=2, start_time=100, interval_ns=100)
        valid_df = pl.DataFrame(
            {
                "instrument_id": ["SPY", "QQQ"],
                "timestamp": timestamps,
                "value": [1.0, 2.0],
            },
        )
        assert provider.validate_data(valid_df)

        # Empty dataframe should fail
        empty_df = pl.DataFrame()
        assert not provider.validate_data(empty_df)

        # Data with nulls in required columns should fail
        timestamps_null = DataBuilder.time_series(n_points=2, start_time=100, interval_ns=100)
        null_df = pl.DataFrame(
            {
                "instrument_id": ["SPY", None],
                "timestamp": timestamps_null,
            },
        )
        assert not provider.validate_data(null_df)

    def test_base_provider_error_handling(self) -> None:
        """
        Test error handling in base provider.
        """
        from ml.data.providers.base import BaseDataProvider

        provider = BaseDataProvider()

        # Should handle errors gracefully
        test_error = ValueError("Test error")
        provider._handle_error(test_error)

        # Check that metrics were updated
        assert provider.metrics.get("provider_errors", 0) > 0


class TestCachedDataProvider:
    """
    Test cached data provider implementation.
    """

    def test_cached_provider_initialization(self) -> None:
        """
        Test cached provider initialization.
        """
        from ml.data.providers.base import CachedDataProvider

        class DummyCachedProvider(CachedDataProvider):
            def _load_data_impl(
                self,
                instruments: list[str],
                start: datetime,
                end: datetime,
            ) -> pl.DataFrame:
                return pl.DataFrame()

        provider = DummyCachedProvider(cache_ttl_hours=24)
        assert provider.cache_ttl == 24
        assert provider._cache == {}

    def test_cache_key_generation(self) -> None:
        """
        Test cache key generation.
        """
        from ml.data.providers.base import CachedDataProvider

        class DummyCachedProvider(CachedDataProvider):
            def _load_data_impl(
                self,
                instruments: list[str],
                start: datetime,
                end: datetime,
            ) -> pl.DataFrame:
                return pl.DataFrame()

        provider = DummyCachedProvider()

        params = {
            "instruments": ["SPY", "QQQ"],
            "start": datetime(2024, 1, 1),
            "end": datetime(2024, 1, 31),
        }

        key = provider.cache_key(params)
        assert isinstance(key, str)
        assert len(key) > 0

        # Same params should give same key
        key2 = provider.cache_key(params)
        assert key == key2

        # Different params should give different key
        params2 = params.copy()
        params2["instruments"] = ["AAPL"]
        key3 = provider.cache_key(params2)
        assert key != key3

    @given(
        ttl_hours=st.integers(min_value=1, max_value=168),
        num_instruments=st.integers(min_value=1, max_value=10),
    )
    def test_cache_ttl_property(self, ttl_hours: int, num_instruments: int) -> None:
        """Property: cache TTL should be respected."""
        from ml.data.providers.base import CachedDataProvider

        class DummyCachedProvider(CachedDataProvider):
            def _load_data_impl(
                self,
                instruments: list[str],
                start: datetime,
                end: datetime,
            ) -> pl.DataFrame:
                return pl.DataFrame()

        provider = DummyCachedProvider(cache_ttl_hours=ttl_hours)
        assert provider.cache_ttl == ttl_hours

    def test_cache_hit_and_miss(self) -> None:
        """
        Test cache hit and miss scenarios.
        """
        from ml.data.providers.base import CachedDataProvider

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        class TestProvider(CachedDataProvider):
            def __init__(self) -> None:
                super().__init__()
                self.load_count = 0

            def _load_data_impl(
                self,
                instruments: list[str],
                start: datetime,
                end: datetime,
            ) -> pl.DataFrame:
                self.load_count += 1
                return pl.DataFrame({"instrument_id": instruments})

        provider = TestProvider()

        # Use fixed datetime values for consistent cache keys
        start_dt = datetime(2024, 1, 1)
        end_dt = datetime(2024, 1, 31)

        # First call - cache miss
        provider.load_data(["SPY"], start_dt, end_dt)
        assert provider.load_count == 1

        # Second call with same params - cache hit
        provider.load_data(["SPY"], start_dt, end_dt)
        assert provider.load_count == 1  # Should not increment

        # Call with different params - cache miss
        provider.load_data(["QQQ"], start_dt, end_dt)
        assert provider.load_count == 2


class TestBaseStaticProvider:
    """
    Test base static data provider.
    """

    def test_static_provider_caching(self) -> None:
        """
        Test that static providers cache by default.
        """
        from ml.data.providers.base import BaseStaticProvider

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        class TestStaticProvider(BaseStaticProvider):
            def __init__(self) -> None:
                super().__init__()
                self.load_count = 0

            def _load_metadata_impl(self, instruments: list[str]) -> pl.DataFrame:
                self.load_count += 1
                return pl.DataFrame(
                    {
                        "instrument_id": instruments,
                        "tick_size": [0.01] * len(instruments),
                    },
                )

        provider = TestStaticProvider()

        # First call
        provider.load_metadata(["SPY", "QQQ"])
        assert provider.load_count == 1

        # Second call with same instruments - should use cache
        provider.load_metadata(["SPY", "QQQ"])
        assert provider.load_count == 1

        # Different instruments - new load
        provider.load_metadata(["AAPL"])
        assert provider.load_count == 2


class TestBaseTimeSeriesProvider:
    """
    Test base time series provider.
    """

    def test_timeseries_provider_validation(self) -> None:
        """
        Test that time series providers validate timestamps.
        """
        from ml.data.providers.base import BaseTimeSeriesProvider

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        class TestTimeSeriesProvider(BaseTimeSeriesProvider):
            def _load_timeseries_impl(
                self,
                instruments: list[str],
                timestamps: pl.Series,
            ) -> pl.DataFrame:
                return pl.DataFrame(
                    {
                        "timestamp": timestamps,
                        "instrument_id": instruments[0],
                    },
                )

        provider = TestTimeSeriesProvider()

        # Valid timestamps
        valid_ts = pl.Series([100, 200, 300])
        result = provider.load_timeseries(["SPY"], valid_ts)
        assert len(result) == 3

        # Invalid timestamps (not sorted)
        invalid_ts = pl.Series([300, 100, 200])
        with pytest.raises(ValueError, match="not sorted"):
            provider.load_timeseries(["SPY"], invalid_ts)

    @given(
        num_timestamps=st.integers(min_value=1, max_value=100),
        num_instruments=st.integers(min_value=1, max_value=10),
    )
    def test_timeseries_shape_property(
        self,
        num_timestamps: int,
        num_instruments: int,
    ) -> None:
        """Property: output shape should match input shape."""
        from ml.data.providers.base import BaseTimeSeriesProvider

        if not HAS_POLARS:
            check_ml_dependencies(["polars"])

        class TestProvider(BaseTimeSeriesProvider):
            def _load_timeseries_impl(
                self,
                instruments: list[str],
                timestamps: pl.Series,
            ) -> pl.DataFrame:
                rows = []
                for ts in timestamps:
                    for inst in instruments:
                        rows.append({"timestamp": ts, "instrument_id": inst})
                return pl.DataFrame(rows)

        provider = TestProvider()

        timestamps = pl.Series(list(range(num_timestamps)))
        instruments = [f"INST{i}" for i in range(num_instruments)]

        result = provider.load_timeseries(instruments, timestamps)

        # Should have one row per timestamp-instrument combination
        expected_rows = num_timestamps * num_instruments
        assert len(result) == expected_rows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
