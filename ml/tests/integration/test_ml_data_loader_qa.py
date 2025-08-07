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
Comprehensive QA test suite for MLDataLoader.

This module performs thorough integration, performance, and stress testing of the
MLDataLoader to ensure production readiness with ZERO technical debt.

"""

import gc
import tempfile
import time
import tracemalloc
from datetime import datetime
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.data.loader import MLDataLoader
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if not HAS_POLARS:
    pytest.skip("Polars required for ML tests", allow_module_level=True)


class TestMLDataLoaderIntegration:
    """
    Integration tests for MLDataLoader with real ParquetDataCatalog.
    """

    @classmethod
    def setup_class(cls) -> None:
        """
        Set up test data directory and catalog.
        """
        cls.temp_dir = tempfile.mkdtemp(prefix="nautilus_ml_test_")
        cls.catalog = ParquetDataCatalog(cls.temp_dir)

        # Create test instruments
        cls.eurusd = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
        cls.gbpusd = InstrumentId(Symbol("GBPUSD"), Venue("SIM"))
        cls.usdjpy = InstrumentId(Symbol("USDJPY"), Venue("SIM"))

        # Generate and write test data
        cls._generate_test_data()

    @classmethod
    def _generate_test_data(cls) -> None:
        """
        Generate comprehensive test data for all scenarios.
        """
        base_time = pd.Timestamp("2023-01-01", tz="UTC")

        # Generate bars for EURUSD (1000 bars, 1-minute intervals)
        bars_eurusd = []
        bar_type = BarType.from_str(f"{cls.eurusd}-1-MINUTE-BID-EXTERNAL")
        for i in range(1000):
            timestamp = base_time + timedelta(minutes=i)
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"1.{1000 + i:04d}"),
                high=Price.from_str(f"1.{1005 + i:04d}"),
                low=Price.from_str(f"1.{995 + i:04d}"),
                close=Price.from_str(f"1.{1002 + i:04d}"),
                volume=Quantity.from_int(100000 + i * 100),
                ts_event=int(timestamp.timestamp() * 1e9),
                ts_init=int(timestamp.timestamp() * 1e9),
            )
            bars_eurusd.append(bar)

        # Generate quotes for GBPUSD (500 quotes, 2-second intervals)
        quotes_gbpusd = []
        for i in range(500):
            timestamp = base_time + timedelta(seconds=i * 2)
            quote = QuoteTick(
                instrument_id=cls.gbpusd,
                bid_price=Price.from_str(f"1.{2500 + i:04d}"),
                ask_price=Price.from_str(f"1.{2502 + i:04d}"),
                bid_size=Quantity.from_int(50000 + i * 50),
                ask_size=Quantity.from_int(50000 + i * 50),
                ts_event=int(timestamp.timestamp() * 1e9),
                ts_init=int(timestamp.timestamp() * 1e9),
            )
            quotes_gbpusd.append(quote)

        # Generate trades for USDJPY (300 trades, 5-second intervals)
        trades_usdjpy = []
        for i in range(300):
            timestamp = base_time + timedelta(seconds=i * 5)
            trade = TradeTick(
                instrument_id=cls.usdjpy,
                price=Price.from_str(f"145.{i:03d}"),
                size=Quantity.from_int(10000 + i * 10),
                aggressor_side=AggressorSide.BUYER if i % 2 == 0 else AggressorSide.SELLER,
                trade_id=f"T{i:06d}",
                ts_event=int(timestamp.timestamp() * 1e9),
                ts_init=int(timestamp.timestamp() * 1e9),
            )
            trades_usdjpy.append(trade)

        # Write data to catalog
        cls.catalog.write_data(bars_eurusd)
        cls.catalog.write_data(quotes_gbpusd)
        cls.catalog.write_data(trades_usdjpy)

    @classmethod
    def teardown_class(cls) -> None:
        """
        Clean up test data.
        """
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_load_bars_from_real_catalog(self) -> None:
        """
        Test loading bars from actual ParquetDataCatalog.
        """
        loader = MLDataLoader(self.catalog)

        # Load bars
        df = loader.load_bars(self.eurusd)

        # Validate results
        assert not df.is_empty()
        assert df.shape[0] == 1000
        assert set(df.columns) == {"timestamp", "open", "high", "low", "close", "volume"}

        # Check data types
        assert df["timestamp"].dtype == pl.Datetime("ns")
        assert df["open"].dtype == pl.Float64
        assert df["volume"].dtype == pl.Int64

        # Verify data integrity
        assert df["high"].min() >= df["low"].min()
        assert df["high"].max() >= df["open"].max()
        assert df["high"].max() >= df["close"].max()

    def test_load_quotes_from_real_catalog(self) -> None:
        """
        Test loading quotes from actual ParquetDataCatalog.
        """
        loader = MLDataLoader(self.catalog)

        # Load quotes
        df = loader.load_quotes(self.gbpusd)

        # Validate results
        assert not df.is_empty()
        assert df.shape[0] == 500
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
        mid_prices = (df["bid_price"] + df["ask_price"]) / 2
        assert (df["mid_price"] - mid_prices).abs().max() < 1e-10

        spreads = df["ask_price"] - df["bid_price"]
        assert (df["spread"] - spreads).abs().max() < 1e-10

    def test_load_trades_from_real_catalog(self) -> None:
        """
        Test loading trades from actual ParquetDataCatalog.
        """
        loader = MLDataLoader(self.catalog)

        # Load trades
        df = loader.load_trades(self.usdjpy)

        # Validate results
        assert not df.is_empty()
        assert df.shape[0] == 300
        assert set(df.columns) == {"timestamp", "price", "size", "aggressor_side"}

        # Check aggressor side values
        unique_sides = df["aggressor_side"].unique()
        assert set(unique_sides) == {"BUYER", "SELLER"}

    def test_date_range_filtering(self) -> None:
        """
        Test date range filtering with various timestamp formats.
        """
        loader = MLDataLoader(self.catalog)

        # Test with datetime objects
        start_dt = datetime(2023, 1, 1, 0, 0, 0)
        end_dt = datetime(2023, 1, 1, 0, 30, 0)

        df1 = loader.load_bars(self.eurusd, start=start_dt, end=end_dt)
        assert df1.shape[0] <= 31  # ~30 minutes of 1-minute bars

        # Test with string timestamps
        df2 = loader.load_bars(
            self.eurusd,
            start="2023-01-01T00:00:00",
            end="2023-01-01T00:30:00",
        )
        assert df1.shape[0] == df2.shape[0]

        # Test with pandas timestamps
        start_pd = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
        end_pd = pd.Timestamp("2023-01-01 00:30:00", tz="UTC")

        df3 = loader.load_bars(self.eurusd, start=start_pd, end=end_pd)
        assert df1.shape[0] == df3.shape[0]

    def test_load_multiple_instruments(self) -> None:
        """
        Test loading data for multiple instruments efficiently.
        """
        loader = MLDataLoader(self.catalog)

        # Load bars for all instruments (only EURUSD has bars)
        result = loader.load_multiple(
            [self.eurusd, self.gbpusd, self.usdjpy],
            data_type="bars",
        )

        # Should only contain EURUSD
        assert len(result) == 1
        assert str(self.eurusd) in result
        assert not result[str(self.eurusd)].is_empty()

        # Load quotes (only GBPUSD has quotes)
        result = loader.load_multiple(
            [self.eurusd, self.gbpusd, self.usdjpy],
            data_type="quotes",
        )

        assert len(result) == 1
        assert str(self.gbpusd) in result

        # Load trades (only USDJPY has trades)
        result = loader.load_multiple(
            [self.eurusd, self.gbpusd, self.usdjpy],
            data_type="trades",
        )

        assert len(result) == 1
        assert str(self.usdjpy) in result

    def test_cache_performance(self) -> None:
        """
        Test caching improves performance significantly.
        """
        loader = MLDataLoader(self.catalog, cache_size=10, enable_cache=True)

        # First load (cold cache)
        start_time = time.perf_counter()
        df1 = loader.load_bars(self.eurusd)
        cold_time = time.perf_counter() - start_time

        # Second load (warm cache)
        start_time = time.perf_counter()
        df2 = loader.load_bars(self.eurusd)
        warm_time = time.perf_counter() - start_time

        # Cache should be significantly faster
        assert warm_time < cold_time * 0.5  # At least 2x faster
        assert df1.equals(df2)

        # Verify cache statistics
        stats = loader.get_cache_stats()
        assert stats["size"] == 1
        assert stats["enabled"] is True

    def test_memory_management(self) -> None:
        """
        Test memory management and leak prevention.
        """
        loader = MLDataLoader(self.catalog, cache_size=5, enable_cache=True)

        # Start memory tracking
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Load data multiple times to test cache eviction
        for i in range(10):
            loader.load_bars(self.eurusd, start=f"2023-01-01T00:{i:02d}:00")
            gc.collect()

        # Check memory usage
        current_memory = tracemalloc.get_traced_memory()[0]
        memory_growth = current_memory - initial_memory
        tracemalloc.stop()

        # Memory growth should be bounded by cache size
        # Allow some overhead but ensure no unbounded growth
        assert memory_growth < 10 * 1024 * 1024  # Less than 10MB growth

        # Verify cache size is bounded
        assert len(loader._cache) <= 5

    def test_error_handling_resilience(self) -> None:
        """
        Test error handling and resilience.
        """
        loader = MLDataLoader(self.catalog)

        # Test with non-existent instrument
        df = loader.load_bars("INVALID.INST")
        assert df.is_empty()

        # Test with invalid date range (end before start)
        df = loader.load_bars(
            self.eurusd,
            start="2023-01-02",
            end="2023-01-01",
        )
        # Should return empty or handle gracefully
        assert df.is_empty() or df.shape[0] == 0

        # Test load_multiple with mixed valid/invalid instruments
        result = loader.load_multiple(
            [self.eurusd, "INVALID.INST", self.gbpusd],
            data_type="bars",
        )
        # Should only contain valid instruments with data
        assert "INVALID.INST" not in result
        assert len(result) == 1  # Only EURUSD has bars

    def test_concurrent_access(self) -> None:
        """
        Test concurrent access patterns.
        """
        from concurrent.futures import ThreadPoolExecutor

        loader = MLDataLoader(self.catalog, cache_size=100, enable_cache=True)
        results = []
        errors = []

        def load_data(instrument_id: InstrumentId, data_type: str) -> None:
            try:
                if data_type == "bars":
                    df = loader.load_bars(instrument_id)
                elif data_type == "quotes":
                    df = loader.load_quotes(instrument_id)
                else:
                    df = loader.load_trades(instrument_id)
                results.append((instrument_id, df.shape[0]))
            except Exception as e:
                errors.append((instrument_id, str(e)))

        # Simulate concurrent access
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            futures.append(executor.submit(load_data, self.eurusd, "bars"))
            futures.append(executor.submit(load_data, self.gbpusd, "quotes"))
            futures.append(executor.submit(load_data, self.usdjpy, "trades"))

            # Wait for completion
            for future in futures:
                future.result()

        # Verify no errors occurred
        assert len(errors) == 0
        assert len(results) == 3

        # Verify correct data was loaded
        for instrument_id, count in results:
            if instrument_id == self.eurusd:
                assert count == 1000
            elif instrument_id == self.gbpusd:
                assert count == 500
            elif instrument_id == self.usdjpy:
                assert count == 300


class TestMLDataLoaderPerformance:
    """
    Performance benchmarks for MLDataLoader.
    """

    @classmethod
    def setup_class(cls) -> None:
        """
        Set up large test dataset for performance testing.
        """
        cls.temp_dir = tempfile.mkdtemp(prefix="nautilus_ml_perf_")
        cls.catalog = ParquetDataCatalog(cls.temp_dir)

        # Generate large dataset
        cls.instrument = InstrumentId(Symbol("PERF"), Venue("TEST"))
        cls._generate_large_dataset()

    @classmethod
    def _generate_large_dataset(cls) -> None:
        """
        Generate large dataset for performance testing.
        """
        base_time = pd.Timestamp("2023-01-01", tz="UTC")

        # Generate 100,000 bars in batches
        batch_size = 10000
        for batch in range(10):
            bars = []
            for i in range(batch_size):
                idx = batch * batch_size + i
                timestamp = base_time + timedelta(seconds=idx)
                bar = Bar(
                    bar_type=BarType.from_str(f"{cls.instrument}-1-SECOND-MID-EXTERNAL"),
                    open=Price.from_str(f"1.{1000 + idx % 1000:04d}"),
                    high=Price.from_str(f"1.{1002 + idx % 1000:04d}"),
                    low=Price.from_str(f"1.{998 + idx % 1000:04d}"),
                    close=Price.from_str(f"1.{1001 + idx % 1000:04d}"),
                    volume=Quantity.from_int(100000),
                    ts_event=int(timestamp.timestamp() * 1e9),
                    ts_init=int(timestamp.timestamp() * 1e9),
                )
                bars.append(bar)

            cls.catalog.write_data(bars)

    @classmethod
    def teardown_class(cls) -> None:
        """
        Clean up test data.
        """
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_load_time_large_dataset(self) -> None:
        """
        Benchmark load time for large datasets.
        """
        loader = MLDataLoader(self.catalog, enable_cache=False)

        # Measure load time
        start_time = time.perf_counter()
        df = loader.load_bars(self.instrument)
        load_time = time.perf_counter() - start_time

        # Verify data
        assert df.shape[0] == 100000

        # Performance requirement: < 2 seconds for 100k bars
        assert load_time < 2.0, f"Load time {load_time:.2f}s exceeds 2s limit"

        print(f"Loaded 100,000 bars in {load_time:.3f} seconds")
        print(f"Throughput: {100000 / load_time:.0f} bars/second")

    def test_cache_hit_performance(self) -> None:
        """
        Test cache hit performance meets requirements.
        """
        loader = MLDataLoader(self.catalog, cache_size=10, enable_cache=True)

        # Prime cache
        df1 = loader.load_bars(self.instrument)

        # Measure cache hit time
        times = []
        for _ in range(100):
            start_time = time.perf_counter()
            df2 = loader.load_bars(self.instrument)
            times.append(time.perf_counter() - start_time)

        avg_time = np.mean(times)
        p99_time = np.percentile(times, 99)

        # Performance requirements
        assert avg_time < 0.001, f"Avg cache hit time {avg_time:.4f}s exceeds 1ms"
        assert p99_time < 0.005, f"P99 cache hit time {p99_time:.4f}s exceeds 5ms"

        print(f"Cache hit performance: avg={avg_time*1000:.2f}ms, p99={p99_time*1000:.2f}ms")

    def test_memory_efficiency(self) -> None:
        """
        Test memory efficiency for large datasets.
        """
        loader = MLDataLoader(self.catalog, cache_size=3, enable_cache=True)

        # Load data and measure memory
        import psutil

        process = psutil.Process()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Load multiple large datasets
        for i in range(5):
            df = loader.load_bars(
                self.instrument,
                start=f"2023-01-01T00:{i*10:02d}:00",
                end=f"2023-01-01T00:{(i+1)*10:02d}:00",
            )
            del df
            gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory should be bounded by cache size
        assert memory_growth < 500, f"Memory growth {memory_growth:.1f}MB exceeds 500MB limit"

        print(f"Memory efficiency: {memory_growth:.1f}MB growth for 5 loads with cache_size=3")


class TestMLDataLoaderCompliance:
    """
    Test compliance with Nautilus Trader conventions and requirements.
    """

    def test_american_english_spelling(self) -> None:
        """
        Verify American English spelling in code and docstrings.
        """
        import inspect

        from ml.data.loader import MLDataLoader

        # Check docstrings for British spelling
        british_words = ["colour", "serialise", "optimise", "analyse", "behaviour"]

        source = inspect.getsource(MLDataLoader)
        source_lower = source.lower()

        for word in british_words:
            assert word not in source_lower, f"Found British spelling: {word}"

    def test_no_tabs_in_source(self) -> None:
        """
        Verify no tabs in source code (4 spaces only).
        """
        import inspect

        from ml.data.loader import MLDataLoader

        source = inspect.getsource(MLDataLoader)
        assert "\t" not in source, "Found tabs in source code (should use 4 spaces)"

    def test_line_length_compliance(self) -> None:
        """
        Verify line length <= 100 characters.
        """
        import inspect

        from ml.data.loader import MLDataLoader

        source = inspect.getsource(MLDataLoader)
        lines = source.split("\n")

        long_lines = []
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                long_lines.append((i, len(line), line[:50]))

        assert len(long_lines) == 0, f"Found {len(long_lines)} lines > 100 chars"

    def test_type_hints_complete(self) -> None:
        """
        Verify all public methods have complete type hints.
        """
        import inspect

        from ml.data.loader import MLDataLoader

        for name, method in inspect.getmembers(MLDataLoader, inspect.isfunction):
            if not name.startswith("_"):  # Public methods
                sig = inspect.signature(method)
                for param_name, param in sig.parameters.items():
                    if param_name != "self":
                        assert (
                            param.annotation != inspect.Parameter.empty
                        ), f"Missing type hint for {name}.{param_name}"

                # Check return type hint
                if name != "__init__":
                    assert (
                        sig.return_annotation != inspect.Signature.empty
                    ), f"Missing return type hint for {name}"

    def test_none_checks_optimized(self) -> None:
        """
        Verify using 'is None' instead of '== None' for Cython optimization.
        """
        import inspect

        from ml.data.loader import MLDataLoader

        source = inspect.getsource(MLDataLoader)

        # Check for non-optimized None comparisons
        assert "== None" not in source, "Found '== None' (should use 'is None')"
        assert "!= None" not in source, "Found '!= None' (should use 'is not None')"

        # Verify correct patterns are used
        assert "is None" in source or "is not None" in source


def run_qa_suite() -> dict[str, Any]:
    """
    Run complete QA suite and return summary report.

    Returns
    -------
    dict[str, Any]
        QA test results and metrics.

    """
    import subprocess
    import sys

    results = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "tests_passed": 0,
        "tests_failed": 0,
        "coverage": 0.0,
        "performance_metrics": {},
        "issues": [],
        "recommendations": [],
    }

    # Run pytest with coverage
    try:
        result = subprocess.run(
            ["pytest", __file__, "-v", "--cov=ml.data.loader", "--cov-report=term-missing"],
            capture_output=True,
            text=True,
        )

        # Parse coverage from output
        for line in result.stdout.split("\n"):
            if "ml/data/loader.py" in line and "%" in line:
                coverage_str = line.split()[-1].replace("%", "")
                results["coverage"] = float(coverage_str)

        # Count passed/failed tests
        if "passed" in result.stdout:
            import re

            match = re.search(r"(\d+) passed", result.stdout)
            if match:
                results["tests_passed"] = int(match.group(1))

        if result.returncode != 0:
            results["tests_failed"] = 1
            results["issues"].append("Some tests failed")

    except Exception as e:
        results["issues"].append(f"Failed to run tests: {e}")

    # Check for technical debt indicators
    import inspect

    from ml.data.loader import MLDataLoader

    source = inspect.getsource(MLDataLoader)

    # Check for stub implementations
    if "NotImplementedError" in source:
        results["issues"].append("Found NotImplementedError (stub implementation)")

    if "TODO" in source or "FIXME" in source:
        results["issues"].append("Found TODO/FIXME comments")

    # Generate recommendations
    if results["coverage"] < 90:
        results["recommendations"].append(
            f"Increase test coverage from {results['coverage']:.1f}% to ≥90%",
        )

    if results["tests_failed"] > 0:
        results["recommendations"].append("Fix failing tests before deployment")

    return results


if __name__ == "__main__":
    # Run QA suite when executed directly
    print("=" * 80)
    print("ML DATA LOADER QA TEST SUITE")
    print("=" * 80)

    report = run_qa_suite()

    print("\nQA REPORT:")
    print("-" * 40)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Tests Passed: {report['tests_passed']}")
    print(f"Tests Failed: {report['tests_failed']}")
    print(f"Code Coverage: {report['coverage']:.1f}%")

    if report["issues"]:
        print("\nISSUES FOUND:")
        for issue in report["issues"]:
            print(f"  - {issue}")

    if report["recommendations"]:
        print("\nRECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")

    print("\nSTATUS:", "✅ PASSED" if report["tests_failed"] == 0 else "❌ FAILED")
    print("=" * 80)
