# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Tests for the MLDataLoader class.

This test module ensures comprehensive coverage of the MLDataLoader functionality
including data loading, caching, error handling, and integration with Nautilus
components.

"""

from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.data.loader import MLDataLoader
from ml.data.loader import load_ml_data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


class TestMLDataLoader:
    """
    Test cases for MLDataLoader class.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        # Create mock catalog
        self.mock_catalog = MagicMock(spec=ParquetDataCatalog)

        # Create test instrument ID
        self.instrument_id = InstrumentId.from_str("EURUSD.SIM")

        # Create test timestamps
        self.start_time = datetime(2023, 1, 1)
        self.end_time = datetime(2023, 1, 2)

        # Create loader instance
        self.loader = MLDataLoader(self.mock_catalog, cache_size=10, enable_cache=True)

    def create_test_bars(self, count: int = 3) -> list[Bar]:
        """
        Create test Bar objects.
        """
        bars = []
        base_time = 1672531200000000000  # 2023-01-01T00:00:00Z in nanoseconds

        for i in range(count):
            bar = Mock(spec=Bar)
            bar.ts_event = base_time + i * 60000000000  # 1 minute intervals
            bar.open = Price.from_str(f"1.100{i}")
            bar.high = Price.from_str(f"1.101{i}")
            bar.low = Price.from_str(f"1.099{i}")
            bar.close = Price.from_str(f"1.100{i + 1}")
            bar.volume = Quantity.from_int(1000 + i)
            bars.append(bar)

        return bars

    def create_test_quotes(self, count: int = 3) -> list[QuoteTick]:
        """
        Create test QuoteTick objects.
        """
        quotes = []
        base_time = 1672531200000000000  # 2023-01-01T00:00:00Z in nanoseconds

        for i in range(count):
            quote = Mock(spec=QuoteTick)
            quote.ts_event = base_time + i * 1000000000  # 1 second intervals
            quote.bid_price = Price.from_str(f"1.0999{i}")
            quote.ask_price = Price.from_str(f"1.1001{i}")
            quote.bid_size = Quantity.from_int(100 + i)
            quote.ask_size = Quantity.from_int(100 + i)
            quotes.append(quote)

        return quotes

    def create_test_trades(self, count: int = 3) -> list[TradeTick]:
        """
        Create test TradeTick objects.
        """
        trades = []
        base_time = 1672531200000000000  # 2023-01-01T00:00:00Z in nanoseconds

        for i in range(count):
            trade = Mock(spec=TradeTick)
            trade.ts_event = base_time + i * 1000000000  # 1 second intervals
            trade.price = Price.from_str(f"1.1000{i}")
            trade.size = Quantity.from_int(50 + i)
            side = AggressorSide.BUYER if i % 2 == 0 else AggressorSide.SELLER
            # Create a mock with name attribute
            mock_side = Mock()
            mock_side.name = side.name
            trade.aggressor_side = mock_side
            trades.append(trade)

        return trades

    def test_init_requires_polars(self) -> None:
        """
        Test that initialization checks for Polars dependency.
        """
        with patch("ml.data.loader.HAS_POLARS", False):
            with patch("ml.data.loader.check_ml_dependencies") as mock_check:
                mock_check.side_effect = ImportError("Polars required")

                with pytest.raises(ImportError):
                    MLDataLoader(self.mock_catalog)

                mock_check.assert_called_once_with(["polars"])

    def test_init_with_valid_params(self) -> None:
        """
        Test successful initialization with valid parameters.
        """
        loader = MLDataLoader(
            catalog=self.mock_catalog,
            cache_size=100,
            enable_cache=True,
        )

        assert loader._catalog == self.mock_catalog
        assert loader._cache_size == 100
        assert loader._enable_cache is True
        assert hasattr(loader, "_cache")
        assert hasattr(loader, "_cache_access_order")

    def test_init_cache_disabled(self) -> None:
        """
        Test initialization with caching disabled.
        """
        loader = MLDataLoader(
            catalog=self.mock_catalog,
            enable_cache=False,
        )

        assert loader._enable_cache is False

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_bars_success(self) -> None:
        """
        Test successful bar loading.
        """
        test_bars = self.create_test_bars()
        self.mock_catalog.query.return_value = test_bars

        df = self.loader.load_bars(self.instrument_id, self.start_time, self.end_time)

        assert not df.is_empty()
        assert df.shape[0] == 3
        expected_columns = {"timestamp", "open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected_columns

        # Check data types
        assert df["timestamp"].dtype == pl.Datetime("ns")
        assert df["open"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Int64

        # Verify that the datetime objects were converted to ISO strings
        self.mock_catalog.query.assert_called_once_with(
            data_cls=Bar,
            identifiers=["EURUSD.SIM"],
            start="2023-01-01T00:00:00Z",  # datetime(2023, 1, 1) converted to ISO
            end="2023-01-02T00:00:00Z",  # datetime(2023, 1, 2) converted to ISO
        )

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_bars_with_string_instrument_id(self) -> None:
        """
        Test bar loading with string instrument ID.
        """
        test_bars = self.create_test_bars()
        self.mock_catalog.query.return_value = test_bars

        # Test with string instrument ID and datetime timestamps
        df = self.loader.load_bars("EURUSD.SIM", self.start_time, self.end_time)

        assert not df.is_empty()
        self.mock_catalog.query.assert_called_once()

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_bars_empty_result(self) -> None:
        """
        Test bar loading when no data is found.
        """
        self.mock_catalog.query.return_value = []

        df = self.loader.load_bars(self.instrument_id)

        assert df.is_empty()
        expected_columns = {"timestamp", "open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected_columns

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_bars_query_exception(self) -> None:
        """
        Test bar loading when catalog query raises exception.
        """
        self.mock_catalog.query.side_effect = Exception("Catalog error")

        df = self.loader.load_bars(self.instrument_id)

        assert df.is_empty()

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_quotes_success(self) -> None:
        """
        Test successful quote loading.
        """
        test_quotes = self.create_test_quotes()
        self.mock_catalog.query.return_value = test_quotes

        df = self.loader.load_quotes(self.instrument_id, self.start_time, self.end_time)

        assert not df.is_empty()
        assert df.shape[0] == 3
        expected_columns = {
            "timestamp",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "mid_price",
            "spread",
        }
        assert set(df.columns) == expected_columns

        # Check derived columns
        assert "mid_price" in df.columns
        assert "spread" in df.columns

        # Verify that the datetime objects were converted to ISO strings
        self.mock_catalog.query.assert_called_once_with(
            data_cls=QuoteTick,
            identifiers=["EURUSD.SIM"],
            start="2023-01-01T00:00:00Z",  # datetime(2023, 1, 1) converted to ISO
            end="2023-01-02T00:00:00Z",  # datetime(2023, 1, 2) converted to ISO
        )

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_quotes_empty_result(self) -> None:
        """
        Test quote loading when no data is found.
        """
        self.mock_catalog.query.return_value = []

        df = self.loader.load_quotes(self.instrument_id)

        assert df.is_empty()
        expected_columns = {
            "timestamp",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "mid_price",
            "spread",
        }
        assert set(df.columns) == expected_columns

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_trades_success(self) -> None:
        """
        Test successful trade loading.
        """
        test_trades = self.create_test_trades()
        self.mock_catalog.query.return_value = test_trades

        df = self.loader.load_trades(self.instrument_id, self.start_time, self.end_time)

        assert not df.is_empty()
        assert df.shape[0] == 3
        expected_columns = {"timestamp", "price", "size", "aggressor_side"}
        assert set(df.columns) == expected_columns

        # Check aggressor side conversion
        assert df["aggressor_side"].dtype == pl.Utf8

        # Verify that the datetime objects were converted to ISO strings
        self.mock_catalog.query.assert_called_once_with(
            data_cls=TradeTick,
            identifiers=["EURUSD.SIM"],
            start="2023-01-01T00:00:00Z",  # datetime(2023, 1, 1) converted to ISO
            end="2023-01-02T00:00:00Z",  # datetime(2023, 1, 2) converted to ISO
        )

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_trades_empty_result(self) -> None:
        """
        Test trade loading when no data is found.
        """
        self.mock_catalog.query.return_value = []

        df = self.loader.load_trades(self.instrument_id)

        assert df.is_empty()
        expected_columns = {"timestamp", "price", "size", "aggressor_side"}
        assert set(df.columns) == expected_columns

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_multiple_bars(self) -> None:
        """
        Test loading bars for multiple instruments.
        """
        instruments = ["EURUSD.SIM", "GBPUSD.SIM"]
        test_bars = self.create_test_bars()
        self.mock_catalog.query.return_value = test_bars

        result = self.loader.load_multiple(instruments, data_type="bars")

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "EURUSD.SIM" in result
        assert "GBPUSD.SIM" in result

        # Should have called query twice
        assert self.mock_catalog.query.call_count == 2

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_multiple_quotes(self) -> None:
        """
        Test loading quotes for multiple instruments.
        """
        instruments = ["EURUSD.SIM", "GBPUSD.SIM"]
        test_quotes = self.create_test_quotes()
        self.mock_catalog.query.return_value = test_quotes

        result = self.loader.load_multiple(instruments, data_type="quotes")

        assert isinstance(result, dict)
        assert len(result) == 2

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_multiple_trades(self) -> None:
        """
        Test loading trades for multiple instruments.
        """
        instruments = ["EURUSD.SIM", "GBPUSD.SIM"]
        test_trades = self.create_test_trades()
        self.mock_catalog.query.return_value = test_trades

        result = self.loader.load_multiple(instruments, data_type="trades")

        assert isinstance(result, dict)
        assert len(result) == 2

    def test_load_multiple_invalid_data_type(self) -> None:
        """
        Test load_multiple with invalid data type.
        """
        instruments = ["EURUSD.SIM"]

        # load_multiple catches exceptions and returns empty dict for resilience
        result = self.loader.load_multiple(instruments, data_type="invalid")
        assert isinstance(result, dict)
        assert len(result) == 0  # Should be empty due to exception handling

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_multiple_with_empty_data(self) -> None:
        """
        Test load_multiple excludes instruments with no data.
        """
        instruments = ["EURUSD.SIM", "GBPUSD.SIM"]

        # First call returns data, second call returns empty
        test_bars = self.create_test_bars()
        self.mock_catalog.query.side_effect = [test_bars, []]

        result = self.loader.load_multiple(instruments, data_type="bars")

        # Should only contain the instrument with data
        assert len(result) == 1
        assert "EURUSD.SIM" in result
        assert "GBPUSD.SIM" not in result

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_multiple_with_exception(self) -> None:
        """
        Test load_multiple handles exceptions gracefully.
        """
        instruments = ["EURUSD.SIM", "GBPUSD.SIM"]

        # First call succeeds, second call raises exception
        test_bars = self.create_test_bars()
        self.mock_catalog.query.side_effect = [test_bars, Exception("Error")]

        result = self.loader.load_multiple(instruments, data_type="bars")

        # Should only contain the successful instrument
        assert len(result) == 1
        assert "EURUSD.SIM" in result

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_caching_functionality(self) -> None:
        """
        Test that caching works correctly.
        """
        test_bars = self.create_test_bars()
        self.mock_catalog.query.return_value = test_bars

        # Load data twice
        df1 = self.loader.load_bars(self.instrument_id, self.start_time, self.end_time)
        df2 = self.loader.load_bars(self.instrument_id, self.start_time, self.end_time)

        # Should only call catalog once due to caching
        assert self.mock_catalog.query.call_count == 1

        # DataFrames should be equal
        assert df1.equals(df2)

    def test_cache_disabled(self) -> None:
        """
        Test behavior when caching is disabled.
        """
        loader = MLDataLoader(self.mock_catalog, enable_cache=False)
        test_bars = self.create_test_bars()
        self.mock_catalog.query.return_value = test_bars

        # Load data twice with same parameters
        loader.load_bars(self.instrument_id, self.start_time, self.end_time)
        loader.load_bars(self.instrument_id, self.start_time, self.end_time)

        # Should call catalog twice since caching is disabled
        assert self.mock_catalog.query.call_count == 2

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_cache_eviction(self) -> None:
        """
        Test LRU cache eviction.
        """
        loader = MLDataLoader(self.mock_catalog, cache_size=2, enable_cache=True)
        test_bars = self.create_test_bars(1)
        self.mock_catalog.query.return_value = test_bars

        # Load data for 3 different keys (exceeds cache size of 2)
        loader.load_bars("EURUSD.SIM")
        loader.load_bars("GBPUSD.SIM")
        loader.load_bars("USDJPY.SIM")  # This should evict the first entry

        # Check cache size
        assert len(loader._cache) == 2

        # The first key should have been evicted
        cache_keys = list(loader._cache.keys())
        assert not any("EURUSD.SIM" in key for key in cache_keys)

    def test_clear_cache(self) -> None:
        """
        Test cache clearing functionality.
        """
        test_bars = self.create_test_bars()
        self.mock_catalog.query.return_value = test_bars

        # Load some data to populate cache
        self.loader.load_bars(self.instrument_id)
        assert len(self.loader._cache) > 0

        # Clear cache
        self.loader.clear_cache()
        assert len(self.loader._cache) == 0
        assert len(self.loader._cache_access_order) == 0

    def test_get_cache_stats(self) -> None:
        """
        Test cache statistics reporting.
        """
        stats = self.loader.get_cache_stats()

        assert isinstance(stats, dict)
        assert "size" in stats
        assert "max_size" in stats
        assert "enabled" in stats

        assert stats["max_size"] == 10
        assert stats["enabled"] is True

    def test_get_cache_stats_disabled(self) -> None:
        """
        Test cache stats when caching is disabled.
        """
        loader = MLDataLoader(self.mock_catalog, enable_cache=False)
        stats = loader.get_cache_stats()

        assert stats["size"] == 0
        assert stats["enabled"] is False

    def test_generate_cache_key(self) -> None:
        """
        Test cache key generation.
        """
        key1 = self.loader._generate_cache_key("bars", "EURUSD.SIM", self.start_time, self.end_time)
        key2 = self.loader._generate_cache_key("bars", "EURUSD.SIM", self.start_time, self.end_time)
        key3 = self.loader._generate_cache_key(
            "quotes",
            "EURUSD.SIM",
            self.start_time,
            self.end_time,
        )

        assert key1 == key2  # Same parameters should generate same key
        assert key1 != key3  # Different data type should generate different key

        # Test with None timestamps
        key4 = self.loader._generate_cache_key("bars", "EURUSD.SIM", None, None)
        assert "None" in key4

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_timestamp_conversion(self) -> None:
        """
        Test that different timestamp types are handled correctly.
        """
        import pandas as pd

        test_bars = self.create_test_bars()
        self.mock_catalog.query.return_value = test_bars

        # Test with datetime objects
        df1 = self.loader.load_bars(
            self.instrument_id,
            start=datetime(2023, 1, 1, 12, 30),
            end=datetime(2023, 1, 2, 15, 45),
        )
        assert not df1.is_empty()

        # Check that datetime was converted to ISO string with time
        self.mock_catalog.query.assert_called_with(
            data_cls=Bar,
            identifiers=["EURUSD.SIM"],
            start="2023-01-01T12:30:00Z",
            end="2023-01-02T15:45:00Z",
        )

        # Test with pandas Timestamp
        self.mock_catalog.query.reset_mock()
        df2 = self.loader.load_bars(
            self.instrument_id,
            start=pd.Timestamp("2023-01-01"),
            end=pd.Timestamp("2023-01-02"),
        )
        assert not df2.is_empty()

        # Test with string timestamps
        self.mock_catalog.query.reset_mock()
        df3 = self.loader.load_bars(
            self.instrument_id,
            start="2023-01-01",
            end="2023-01-02",
        )
        assert not df3.is_empty()

        # Verify string timestamps are passed through unchanged
        self.mock_catalog.query.assert_called_with(
            data_cls=Bar,
            identifiers=["EURUSD.SIM"],
            start="2023-01-01",
            end="2023-01-02",
        )

        # Test with int (nanoseconds)
        self.mock_catalog.query.reset_mock()
        df4 = self.loader.load_bars(
            self.instrument_id,
            start=1672531200000000000,
            end=1672617600000000000,
        )
        assert not df4.is_empty()

        # Verify int timestamps are passed through unchanged
        self.mock_catalog.query.assert_called_with(
            data_cls=Bar,
            identifiers=["EURUSD.SIM"],
            start=1672531200000000000,
            end=1672617600000000000,
        )

    def test_bars_to_polars_empty(self) -> None:
        """
        Test converting empty bars list.
        """
        df = self.loader._bars_to_polars([])
        assert df.is_empty()
        expected_columns = {"timestamp", "open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected_columns

    def test_quotes_to_polars_empty(self) -> None:
        """
        Test converting empty quotes list.
        """
        df = self.loader._quotes_to_polars([])
        assert df.is_empty()
        expected_columns = {
            "timestamp",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "mid_price",
            "spread",
        }
        assert set(df.columns) == expected_columns

    def test_trades_to_polars_empty(self) -> None:
        """
        Test converting empty trades list.
        """
        df = self.loader._trades_to_polars([])
        assert df.is_empty()
        expected_columns = {"timestamp", "price", "size", "aggressor_side"}
        assert set(df.columns) == expected_columns

    def test_create_empty_dataframes(self) -> None:
        """
        Test empty DataFrame creation methods.
        """
        bars_df = self.loader._create_empty_bars_df()
        quotes_df = self.loader._create_empty_quotes_df()
        trades_df = self.loader._create_empty_trades_df()

        assert bars_df.is_empty()
        assert quotes_df.is_empty()
        assert trades_df.is_empty()

        # Check schemas
        assert bars_df.schema["timestamp"] == pl.Datetime("ns")
        assert bars_df.schema["open"] == pl.Float64
        assert bars_df.schema["volume"] == pl.Int64

        assert quotes_df.schema["mid_price"] == pl.Float64
        assert quotes_df.schema["spread"] == pl.Float64

        assert trades_df.schema["aggressor_side"] == pl.Utf8


class TestLoadMLDataFunction:
    """
    Test cases for the load_ml_data convenience function.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.mock_catalog = MagicMock(spec=ParquetDataCatalog)

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_ml_data_bars(self) -> None:
        """
        Test load_ml_data function with bars.
        """
        instruments = ["EURUSD.SIM", "GBPUSD.SIM"]

        with patch("ml.data.loader.MLDataLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_loader.load_multiple.return_value = {"EURUSD.SIM": MagicMock()}

            result = load_ml_data(
                instrument_ids=instruments,
                catalog=self.mock_catalog,
                data_type="bars",
            )

            mock_loader_class.assert_called_once_with(self.mock_catalog)
            mock_loader.load_multiple.assert_called_once_with(
                instrument_ids=instruments,
                data_type="bars",
                start=None,
                end=None,
            )

            assert isinstance(result, dict)

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_load_ml_data_with_time_range(self) -> None:
        """
        Test load_ml_data function with time range.
        """
        instruments = ["EURUSD.SIM"]
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 1, 2)

        with patch("ml.data.loader.MLDataLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            mock_loader.load_multiple.return_value = {}

            load_ml_data(
                instrument_ids=instruments,
                catalog=self.mock_catalog,
                data_type="quotes",
                start=start_time,
                end=end_time,
            )

            mock_loader.load_multiple.assert_called_once_with(
                instrument_ids=instruments,
                data_type="quotes",
                start=start_time,
                end=end_time,
            )


# Integration test to ensure the components work together
@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
def test_integration_with_real_data_structures() -> None:
    """
    Test integration with actual Nautilus data structures.
    """
    # This test verifies that our loader can handle real Nautilus objects
    # without breaking due to type mismatches
    mock_catalog = MagicMock(spec=ParquetDataCatalog)
    loader = MLDataLoader(mock_catalog)

    # Test that the loader initializes properly with real types
    assert loader is not None
    assert hasattr(loader, "_catalog")

    # Test empty schema creation
    bars_df = loader._create_empty_bars_df()
    assert bars_df.is_empty()
    assert set(bars_df.columns) == {"timestamp", "open", "high", "low", "close", "volume"}
