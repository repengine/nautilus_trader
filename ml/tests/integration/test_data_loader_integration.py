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
Integration tests for MLDataLoader with real ParquetDataCatalog.

These tests verify that the MLDataLoader works correctly with actual Nautilus data types
and ParquetDataCatalog implementation, not just mocks.

"""

import tempfile
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.data.loader import MLDataLoader
from ml.data.loader import load_ml_data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggregationSource
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestMLDataLoaderIntegration:
    """
    Integration tests for MLDataLoader with real ParquetDataCatalog.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures with temporary directory for catalog.
        """
        self.temp_dir = tempfile.mkdtemp(prefix="nautilus_ml_test_")
        self.catalog = ParquetDataCatalog(self.temp_dir)
        self.loader = MLDataLoader(self.catalog, cache_size=100, enable_cache=True)

        # Create test instrument IDs
        self.instrument_id_1 = InstrumentId.from_str("EURUSD.SIM")
        self.instrument_id_2 = InstrumentId.from_str("GBPUSD.SIM")

        # Create bar type for testing
        self.bar_type = BarType(
            instrument_id=self.instrument_id_1,
            bar_spec=BarSpecification(
                step=1,
                aggregation=BarAggregation.MINUTE,
                price_type=PriceType.LAST,
            ),
            aggregation_source=AggregationSource.EXTERNAL,
        )

    def teardown_method(self) -> None:
        """
        Clean up temporary directory.
        """
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_sample_bars(self, instrument_id: InstrumentId, count: int = 100) -> list[Bar]:
        """
        Create sample bar data for testing.
        """
        bars = []
        base_time = pd.Timestamp("2023-01-01", tz="UTC").value

        for i in range(count):
            bar = Bar(
                bar_type=BarType(
                    instrument_id=instrument_id,
                    bar_spec=BarSpecification(
                        step=1,
                        aggregation=BarAggregation.MINUTE,
                        price_type=PriceType.LAST,
                    ),
                    aggregation_source=AggregationSource.EXTERNAL,
                ),
                open=Price.from_str(f"1.{1000 + i:04d}"),
                high=Price.from_str(f"1.{1010 + i:04d}"),
                low=Price.from_str(f"1.{990 + i:04d}"),
                close=Price.from_str(f"1.{1005 + i:04d}"),
                volume=Quantity.from_int(1000 + i * 10),
                ts_event=base_time + i * 60_000_000_000,  # 1 minute intervals
                ts_init=base_time + i * 60_000_000_000,
            )
            bars.append(bar)
        return bars

    def create_sample_quotes(
        self,
        instrument_id: InstrumentId,
        count: int = 100,
    ) -> list[QuoteTick]:
        """
        Create sample quote tick data for testing.
        """
        quotes = []
        base_time = pd.Timestamp("2023-01-01", tz="UTC").value

        for i in range(count):
            quote = QuoteTick(
                instrument_id=instrument_id,
                bid_price=Price.from_str(f"1.{999 + i:04d}"),
                ask_price=Price.from_str(f"1.{1001 + i:04d}"),
                bid_size=Quantity.from_int(100 + i),
                ask_size=Quantity.from_int(100 + i),
                ts_event=base_time + i * 1_000_000_000,  # 1 second intervals
                ts_init=base_time + i * 1_000_000_000,
            )
            quotes.append(quote)
        return quotes

    def create_sample_trades(
        self,
        instrument_id: InstrumentId,
        count: int = 100,
    ) -> list[TradeTick]:
        """
        Create sample trade tick data for testing.
        """
        trades = []
        base_time = pd.Timestamp("2023-01-01", tz="UTC").value

        for i in range(count):
            trade = TradeTick(
                instrument_id=instrument_id,
                price=Price.from_str(f"1.{1000 + i:04d}"),
                size=Quantity.from_int(50 + i),
                aggressor_side=AggressorSide.BUYER if i % 2 == 0 else AggressorSide.SELLER,
                trade_id=TradeId(f"T{i:06d}"),
                ts_event=base_time + i * 1_000_000_000,  # 1 second intervals
                ts_init=base_time + i * 1_000_000_000,
            )
            trades.append(trade)
        return trades

    def test_load_bars_from_real_catalog(self) -> None:
        """
        Test loading bars from a real ParquetDataCatalog.
        """
        # Write test data to catalog
        bars = self.create_sample_bars(self.instrument_id_1, 50)
        self.catalog.write_data(bars)

        # Load data using MLDataLoader
        df = self.loader.load_bars(self.instrument_id_1)

        # Verify data
        assert not df.is_empty()
        assert df.shape[0] == 50
        assert set(df.columns) == {"timestamp", "open", "high", "low", "close", "volume"}

        # Verify data types
        assert df["timestamp"].dtype == pl.Datetime("ns")
        assert df["open"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Int64

        # Verify values match
        assert df["open"][0] == 1.1000
        assert df["close"][0] == 1.1005
        assert df["volume"][0] == 1000

    def test_load_quotes_from_real_catalog(self) -> None:
        """
        Test loading quotes from a real ParquetDataCatalog.
        """
        # Write test data to catalog
        quotes = self.create_sample_quotes(self.instrument_id_1, 30)
        self.catalog.write_data(quotes)

        # Load data using MLDataLoader
        df = self.loader.load_quotes(self.instrument_id_1)

        # Verify data
        assert not df.is_empty()
        assert df.shape[0] == 30
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

        # Verify derived columns
        mid_prices = df["mid_price"].to_numpy()
        spreads = df["spread"].to_numpy()
        assert np.allclose(mid_prices[0], 1.1000, rtol=1e-6)
        assert np.allclose(spreads[0], 0.0002, rtol=1e-6)

    def test_load_trades_from_real_catalog(self) -> None:
        """
        Test loading trades from a real ParquetDataCatalog.
        """
        # Write test data to catalog
        trades = self.create_sample_trades(self.instrument_id_1, 25)
        self.catalog.write_data(trades)

        # Load data using MLDataLoader
        df = self.loader.load_trades(self.instrument_id_1)

        # Verify data
        assert not df.is_empty()
        assert df.shape[0] == 25
        assert set(df.columns) == {"timestamp", "price", "size", "aggressor_side"}

        # Verify aggressor side conversion
        assert df["aggressor_side"][0] == "BUYER"
        assert df["aggressor_side"][1] == "SELLER"

    def test_date_range_filtering(self) -> None:
        """
        Test date range filtering with real data.
        """
        # Write test data
        bars = self.create_sample_bars(self.instrument_id_1, 100)
        self.catalog.write_data(bars)

        # Test with datetime objects
        start = datetime(2023, 1, 1, 0, 30)  # 30 minutes in
        end = datetime(2023, 1, 1, 1, 0)  # 60 minutes in

        df = self.loader.load_bars(self.instrument_id_1, start=start, end=end)

        # Should have approximately 30 bars (30-60 minute range)
        assert not df.is_empty()
        assert df.shape[0] <= 31  # Allow for boundary conditions

    def test_load_multiple_instruments(self) -> None:
        """
        Test loading data for multiple instruments.
        """
        # Write data for multiple instruments
        bars1 = self.create_sample_bars(self.instrument_id_1, 20)
        bars2 = self.create_sample_bars(self.instrument_id_2, 30)
        self.catalog.write_data(bars1)
        self.catalog.write_data(bars2)

        # Load multiple
        result = self.loader.load_multiple(
            [self.instrument_id_1, self.instrument_id_2],
            data_type="bars",
        )

        assert len(result) == 2
        assert "EURUSD.SIM" in result
        assert "GBPUSD.SIM" in result
        assert result["EURUSD.SIM"].shape[0] == 20
        assert result["GBPUSD.SIM"].shape[0] == 30

    def test_cache_performance(self) -> None:
        """
        Test that caching improves performance.
        """
        # Write substantial data
        bars = self.create_sample_bars(self.instrument_id_1, 1000)
        self.catalog.write_data(bars)

        # First load (uncached)
        start_time = time.time()
        df1 = self.loader.load_bars(self.instrument_id_1)
        first_load_time = time.time() - start_time

        # Second load (cached)
        start_time = time.time()
        df2 = self.loader.load_bars(self.instrument_id_1)
        cached_load_time = time.time() - start_time

        # Verify cached load is faster
        assert cached_load_time < first_load_time
        assert df1.equals(df2)

        # Verify cache stats
        stats = self.loader.get_cache_stats()
        assert stats["size"] == 1
        assert stats["enabled"] is True

    def test_cache_eviction_lru(self) -> None:
        """
        Test LRU cache eviction behavior.
        """
        # Create loader with small cache
        small_loader = MLDataLoader(self.catalog, cache_size=2, enable_cache=True)

        # Write data for 3 instruments
        for i, inst_id in enumerate(
            [
                InstrumentId.from_str("EUR/USD.SIM"),
                InstrumentId.from_str("GBP/USD.SIM"),
                InstrumentId.from_str("USD/JPY.SIM"),
            ],
        ):
            bars = self.create_sample_bars(inst_id, 10)
            self.catalog.write_data(bars)

            # Load each instrument
            small_loader.load_bars(inst_id)

        # Cache should only have 2 entries (last 2 loaded)
        stats = small_loader.get_cache_stats()
        assert stats["size"] == 2

    def test_memory_stability(self) -> None:
        """
        Test memory stability with repeated operations.
        """
        # Write test data
        bars = self.create_sample_bars(self.instrument_id_1, 100)
        self.catalog.write_data(bars)

        # Perform multiple loads
        memory_samples = []
        for _ in range(10):
            df = self.loader.load_bars(self.instrument_id_1)
            # Clear cache periodically
            if _ % 3 == 0:
                self.loader.clear_cache()

            # Track memory (simplified - in production use memory_profiler)
            import sys

            memory_samples.append(sys.getsizeof(df))

        # Memory should be stable (not growing unbounded)
        assert max(memory_samples) - min(memory_samples) < 1000  # Reasonable variance

    def test_error_handling_missing_instrument(self) -> None:
        """
        Test handling of missing instrument data.
        """
        # Try to load non-existent instrument
        df = self.loader.load_bars(InstrumentId.from_str("INVALID.SIM"))

        # Should return empty DataFrame with correct schema
        assert df.is_empty()
        assert set(df.columns) == {"timestamp", "open", "high", "low", "close", "volume"}

    def test_concurrent_access_simulation(self) -> None:
        """
        Simulate concurrent access patterns (single-threaded).
        """
        # Write test data
        bars = self.create_sample_bars(self.instrument_id_1, 50)
        self.catalog.write_data(bars)

        # Simulate multiple "concurrent" requests
        results = []
        for _ in range(5):
            df = self.loader.load_bars(self.instrument_id_1)
            results.append(df)

        # All results should be identical
        for df in results[1:]:
            assert df.equals(results[0])

    def test_load_ml_data_convenience_function(self) -> None:
        """
        Test the convenience function with real catalog.
        """
        # Write test data
        for inst_id in [self.instrument_id_1, self.instrument_id_2]:
            bars = self.create_sample_bars(inst_id, 20)
            self.catalog.write_data(bars)

        # Use convenience function
        data = load_ml_data(
            instrument_ids=["EURUSD.SIM", "GBPUSD.SIM"],
            catalog=self.catalog,
            data_type="bars",
        )

        assert len(data) == 2
        assert all(not df.is_empty() for df in data.values())


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestMLDataLoaderPerformance:
    """
    Performance-focused tests for MLDataLoader.
    """

    def setup_method(self) -> None:
        """
        Set up test fixtures.
        """
        self.temp_dir = tempfile.mkdtemp(prefix="nautilus_ml_perf_")
        self.catalog = ParquetDataCatalog(self.temp_dir)
        self.loader = MLDataLoader(self.catalog)

    def teardown_method(self) -> None:
        """
        Clean up.
        """
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_large_dataset_loading(self) -> None:
        """
        Test loading large datasets efficiently.
        """
        instrument_id = InstrumentId.from_str("EURUSD.SIM")

        # Create large dataset (10,000 bars)
        bars = []
        base_time = pd.Timestamp("2023-01-01", tz="UTC").value

        for i in range(10000):
            bar = Bar(
                bar_type=BarType(
                    instrument_id=instrument_id,
                    bar_spec=BarSpecification(
                        step=1,
                        aggregation=BarAggregation.MINUTE,
                        price_type=PriceType.LAST,
                    ),
                    aggregation_source=AggregationSource.EXTERNAL,
                ),
                open=Price.from_str("1.1000"),
                high=Price.from_str("1.1010"),
                low=Price.from_str("1.0990"),
                close=Price.from_str("1.1005"),
                volume=Quantity.from_int(1000),
                ts_event=base_time + i * 60_000_000_000,
                ts_init=base_time + i * 60_000_000_000,
            )
            bars.append(bar)

        # Write to catalog
        self.catalog.write_data(bars)

        # Measure loading time
        start_time = time.time()
        df = self.loader.load_bars(instrument_id)
        load_time = time.time() - start_time

        # Verify data
        assert df.shape[0] == 10000
        # Performance requirement: Load 10k bars in under 1 second
        assert load_time < 1.0, f"Loading took {load_time:.2f}s, expected < 1s"

    def test_vectorized_conversion_performance(self) -> None:
        """
        Test that vectorized conversion is efficient.
        """
        instrument_id = InstrumentId.from_str("EURUSD.SIM")

        # Create test quotes (5000)
        quotes = []
        base_time = pd.Timestamp("2023-01-01", tz="UTC").value

        for i in range(5000):
            quote = QuoteTick(
                instrument_id=instrument_id,
                bid_price=Price.from_str("1.0999"),
                ask_price=Price.from_str("1.1001"),
                bid_size=Quantity.from_int(100),
                ask_size=Quantity.from_int(100),
                ts_event=base_time + i * 1_000_000_000,
                ts_init=base_time + i * 1_000_000_000,
            )
            quotes.append(quote)

        # Measure conversion time directly
        start_time = time.time()
        df = self.loader._quotes_to_polars(quotes)
        conversion_time = time.time() - start_time

        # Verify data and performance
        assert df.shape[0] == 5000
        assert "mid_price" in df.columns
        assert "spread" in df.columns
        # Conversion should be fast
        assert conversion_time < 0.5, f"Conversion took {conversion_time:.2f}s"


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])
