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
Simplified integration test to verify MLDataLoader works with real usage patterns.
"""

import time
from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import Mock

import pandas as pd
import pytest

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.data.loader import MLDataLoader
from ml.data.loader import load_ml_data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if not HAS_POLARS:
    pytest.skip("Polars required for ML tests", allow_module_level=True)


def create_mock_bars(count: int = 100) -> list[Bar]:
    """
    Create mock Bar objects for testing.
    """
    bars = []
    base_time = 1672531200000000000  # 2023-01-01T00:00:00Z in nanoseconds

    for i in range(count):
        bar = Mock(spec=Bar)
        bar.ts_event = base_time + i * 60000000000  # 1 minute intervals
        bar.open = Price.from_str(f"1.100{i % 10}")
        bar.high = Price.from_str(f"1.101{i % 10}")
        bar.low = Price.from_str(f"1.099{i % 10}")
        bar.close = Price.from_str(f"1.100{(i + 1) % 10}")
        bar.volume = Quantity.from_int(100000 + i * 100)
        bars.append(bar)

    return bars


def create_mock_quotes(count: int = 50) -> list[QuoteTick]:
    """
    Create mock QuoteTick objects for testing.
    """
    quotes = []
    base_time = 1672531200000000000  # 2023-01-01T00:00:00Z in nanoseconds

    for i in range(count):
        quote = Mock(spec=QuoteTick)
        quote.ts_event = base_time + i * 1000000000  # 1 second intervals
        quote.bid_price = Price.from_str(f"1.0999{i % 10}")
        quote.ask_price = Price.from_str(f"1.1001{i % 10}")
        quote.bid_size = Quantity.from_int(50000 + i * 50)
        quote.ask_size = Quantity.from_int(50000 + i * 50)
        quotes.append(quote)

    return quotes


def create_mock_trades(count: int = 30) -> list[TradeTick]:
    """
    Create mock TradeTick objects for testing.
    """
    trades = []
    base_time = 1672531200000000000  # 2023-01-01T00:00:00Z in nanoseconds

    for i in range(count):
        trade = Mock(spec=TradeTick)
        trade.ts_event = base_time + i * 1000000000  # 1 second intervals
        trade.price = Price.from_str(f"1.1000{i % 10}")
        trade.size = Quantity.from_int(10000 + i * 10)
        side = AggressorSide.BUYER if i % 2 == 0 else AggressorSide.SELLER
        mock_side = Mock()
        mock_side.name = side.name
        trade.aggressor_side = mock_side
        trades.append(trade)

    return trades


# Configure module logger
logger = logging.getLogger(__name__)


class TestMLDataLoaderRealUsage:
    """
    Test MLDataLoader in realistic usage scenarios.
    """

    def test_typical_ml_workflow(self) -> None:
        """
        Test a typical ML workflow with data loading and feature engineering.
        """
        # Setup mock catalog
        mock_catalog = MagicMock(spec=ParquetDataCatalog)
        mock_catalog.query.return_value = create_mock_bars(1000)

        # Initialize loader
        loader = MLDataLoader(mock_catalog, cache_size=100, enable_cache=True)

        # Load data for multiple instruments (simulating real ML workflow)
        instruments = ["EURUSD.SIM", "GBPUSD.SIM", "USDJPY.SIM"]

        all_data = {}
        for instrument in instruments:
            # Load bars
            bars_df = loader.load_bars(instrument)
            assert not bars_df.is_empty()
            assert bars_df.shape[0] == 1000

            # Verify data can be used for feature engineering
            # Calculate simple features
            returns = (bars_df["close"] - bars_df["open"]) / bars_df["open"]
            assert len(returns) == 1000

            # Calculate rolling statistics
            sma_20 = bars_df["close"].rolling_mean(20)
            assert sma_20 is not None

            all_data[instrument] = bars_df

        # Verify cache is working
        stats = loader.get_cache_stats()
        assert stats["size"] == 3  # Three instruments cached
        assert stats["enabled"] is True

    def test_performance_benchmarks(self) -> None:
        """
        Test performance meets requirements.
        """
        mock_catalog = MagicMock(spec=ParquetDataCatalog)
        mock_catalog.query.return_value = create_mock_bars(10000)

        loader = MLDataLoader(mock_catalog, cache_size=10, enable_cache=True)

        # Benchmark initial load
        start = time.perf_counter()
        df1 = loader.load_bars("EURUSD.SIM")
        initial_load_time = time.perf_counter() - start

        # Benchmark cached load
        start = time.perf_counter()
        df2 = loader.load_bars("EURUSD.SIM")
        cached_load_time = time.perf_counter() - start

        # Performance assertions
        assert initial_load_time < 1.0  # Should load 10k bars in < 1 second
        assert cached_load_time < 0.01  # Cache hit should be < 10ms
        assert cached_load_time < initial_load_time * 0.1  # Cache should be 10x faster

        logger.info(f"Initial load: {initial_load_time*1000:.2f}ms")
        logger.info(f"Cached load: {cached_load_time*1000:.2f}ms")
        logger.info(f"Speedup: {initial_load_time/cached_load_time:.1f}x")

    def test_memory_efficiency(self) -> None:
        """
        Test memory usage is efficient and bounded.
        """
        mock_catalog = MagicMock(spec=ParquetDataCatalog)
        mock_catalog.query.return_value = create_mock_bars(1000)

        # Small cache size to test eviction
        loader = MLDataLoader(mock_catalog, cache_size=2, enable_cache=True)

        # Load multiple datasets
        for i in range(5):
            df = loader.load_bars(f"INST{i}.SIM")
            assert not df.is_empty()

        # Cache should be bounded
        stats = loader.get_cache_stats()
        assert stats["size"] <= 2  # Cache size is respected

        # Clear cache should free memory
        loader.clear_cache()
        stats = loader.get_cache_stats()
        assert stats["size"] == 0

    def test_error_handling_production(self) -> None:
        """
        Test error handling in production scenarios.
        """
        mock_catalog = MagicMock(spec=ParquetDataCatalog)

        # Simulate various error conditions
        loader = MLDataLoader(mock_catalog, cache_size=100, enable_cache=True)

        # 1. Empty result from catalog
        mock_catalog.query.return_value = []
        df = loader.load_bars("EMPTY.SIM")
        assert df.is_empty()
        assert set(df.columns) == {"timestamp", "open", "high", "low", "close", "volume"}

        # 2. Exception from catalog
        mock_catalog.query.side_effect = Exception("Network error")
        df = loader.load_bars("ERROR.SIM")
        assert df.is_empty()  # Should return empty DataFrame, not crash

        # 3. Mixed success/failure in load_multiple
        mock_catalog.query.side_effect = [
            create_mock_bars(100),  # Success
            Exception("Error"),  # Failure
            create_mock_quotes(50),  # Success
        ]

        result = loader.load_multiple(
            ["GOOD1.SIM", "BAD.SIM", "GOOD2.SIM"],
            data_type="bars",
        )

        # Should only contain successful loads
        assert len(result) == 1
        assert "GOOD1.SIM" in result
        assert "BAD.SIM" not in result

    def test_data_type_conversions(self) -> None:
        """
        Test all data type conversions work correctly.
        """
        mock_catalog = MagicMock(spec=ParquetDataCatalog)
        loader = MLDataLoader(mock_catalog, cache_size=100, enable_cache=True)

        # Test bars conversion
        mock_catalog.query.return_value = create_mock_bars(10)
        bars_df = loader.load_bars("TEST.SIM")
        assert bars_df["timestamp"].dtype == pl.Datetime("ns")
        assert bars_df["open"].dtype == pl.Float64
        assert bars_df["volume"].dtype == pl.Int64

        # Test quotes conversion with derived columns
        mock_catalog.query.return_value = create_mock_quotes(10)
        quotes_df = loader.load_quotes("TEST.SIM")
        assert "mid_price" in quotes_df.columns
        assert "spread" in quotes_df.columns

        # Verify derived columns are calculated correctly
        expected_mid = (quotes_df["bid_price"] + quotes_df["ask_price"]) / 2
        assert (quotes_df["mid_price"] - expected_mid).abs().max() < 1e-10

        # Test trades conversion
        mock_catalog.query.return_value = create_mock_trades(10)
        trades_df = loader.load_trades("TEST.SIM")
        assert trades_df["aggressor_side"].dtype == pl.Utf8
        assert set(trades_df["aggressor_side"].unique()) == {"BUYER", "SELLER"}

    def test_timestamp_flexibility(self) -> None:
        """
        Test that various timestamp formats are handled correctly.
        """
        mock_catalog = MagicMock(spec=ParquetDataCatalog)
        mock_catalog.query.return_value = create_mock_bars(100)
        loader = MLDataLoader(mock_catalog, cache_size=100, enable_cache=True)

        # Test different timestamp formats
        test_cases = [
            ("2023-01-01", "2023-01-02"),  # String dates
            (datetime(2023, 1, 1), datetime(2023, 1, 2)),  # datetime objects
            (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")),  # pandas timestamps
            (1672531200000000000, 1672617600000000000),  # nanoseconds
            (None, None),  # No filtering
        ]

        for start, end in test_cases:
            df = loader.load_bars("TEST.SIM", start=start, end=end)
            assert not df.is_empty()
            assert df.shape[0] == 100  # Mock always returns 100 bars

    def test_concurrent_usage(self) -> None:
        """
        Test that loader can handle concurrent access safely.
        """
        from concurrent.futures import ThreadPoolExecutor

        mock_catalog = MagicMock(spec=ParquetDataCatalog)
        mock_catalog.query.return_value = create_mock_bars(100)
        loader = MLDataLoader(mock_catalog, cache_size=100, enable_cache=True)

        results = []
        errors = []

        def load_data(instrument: str, data_type: str) -> None:
            try:
                if data_type == "bars":
                    df = loader.load_bars(instrument)
                elif data_type == "quotes":
                    df = loader.load_quotes(instrument)
                else:
                    df = loader.load_trades(instrument)
                results.append((instrument, df.shape[0] if not df.is_empty() else 0))
            except Exception as e:
                errors.append((instrument, str(e)))

        # Simulate concurrent access from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(10):
                instrument = f"INST{i}.SIM"
                data_type = ["bars", "quotes", "trades"][i % 3]
                futures.append(executor.submit(load_data, instrument, data_type))

            # Wait for all to complete
            for future in futures:
                future.result()

        # Should have no errors
        assert len(errors) == 0
        assert len(results) == 10


def test_load_ml_data_convenience_function() -> None:
    """
    Test the convenience function for loading ML data.
    """
    mock_catalog = MagicMock(spec=ParquetDataCatalog)
    mock_catalog.query.return_value = create_mock_bars(500)

    # Test basic usage
    instruments = ["EURUSD.SIM", "GBPUSD.SIM"]
    data = load_ml_data(instruments, mock_catalog, data_type="bars")

    assert isinstance(data, dict)
    assert len(data) == 2
    for instrument in instruments:
        assert instrument in data
        assert not data[instrument].is_empty()
        assert data[instrument].shape[0] == 500

    # Test with date filtering
    data = load_ml_data(
        instruments,
        mock_catalog,
        start="2023-01-01",
        end="2023-01-31",
    )
    assert len(data) == 2


if __name__ == "__main__":
    # Run performance benchmark when executed directly
    logger.info("Running MLDataLoader Performance Benchmark...")
    test = TestMLDataLoaderRealUsage()
    test.test_performance_benchmarks()
    logger.info("\nAll benchmarks passed! ")
