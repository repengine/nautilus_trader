#!/usr/bin/env python
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
Comprehensive QA test suite for FeatureEngineer implementation.

This module performs thorough testing of the enhanced FeatureEngineer including:
- Performance benchmarking
- Feature parity verification
- Memory usage testing
- Edge case handling
- Production readiness assessment

"""

import gc
import time
import tracemalloc

import numpy as np
import pandas as pd

from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class FeatureEngineerQATester:
    """
    Comprehensive QA testing for FeatureEngineer.
    """

    def __init__(self):
        self.test_results = {
            "static_analysis": {},
            "performance": {},
            "feature_parity": {},
            "memory": {},
            "edge_cases": {},
            "integration": {},
        }
        self.failures = []

    def run_all_tests(self) -> dict:
        """
        Run all QA tests and return results.
        """
        print("=" * 80)
        print("FEATURE ENGINEER QA TEST SUITE")
        print("=" * 80)

        # 1. Performance Testing
        print("\n1. PERFORMANCE TESTING")
        print("-" * 40)
        self._test_performance()

        # 2. Feature Parity Testing
        print("\n2. FEATURE PARITY TESTING")
        print("-" * 40)
        self._test_feature_parity()

        # 3. Memory Usage Testing
        print("\n3. MEMORY USAGE TESTING")
        print("-" * 40)
        self._test_memory_usage()

        # 4. Edge Cases Testing
        print("\n4. EDGE CASES TESTING")
        print("-" * 40)
        self._test_edge_cases()

        # 5. Integration Testing
        print("\n5. INTEGRATION TESTING")
        print("-" * 40)
        self._test_integration()

        # Generate report
        self._generate_report()

        return self.test_results

    def _test_performance(self):
        """
        Test performance requirements.
        """
        print("Testing feature calculation performance...")

        # Create test data
        df = self._create_test_data(1000)
        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=True,
        )
        engineer = FeatureEngineer(config)

        # Test batch processing performance
        start = time.perf_counter()
        features_batch, _ = engineer.calculate_features_batch(df)
        batch_time = (time.perf_counter() - start) * 1000

        print(f"  Batch processing (1000 bars): {batch_time:.2f}ms")
        self.test_results["performance"]["batch_time_ms"] = batch_time
        self.test_results["performance"]["batch_bars_per_second"] = 1000 / (batch_time / 1000)

        # Test online processing performance (hot path)
        indicator_mgr = IndicatorManager(config)

        # Warm up indicators
        for i in range(50):
            bar = self._create_bar(100.0 + i * 0.1, 1000 + i * 10)
            indicator_mgr.update_from_bar(bar)

        # Measure online performance
        online_times = []
        for i in range(100):
            current_bar = {
                "close": 105.0 + i * 0.01,
                "high": 105.5 + i * 0.01,
                "low": 104.5 + i * 0.01,
                "volume": 2000.0,
            }

            start = time.perf_counter()
            features = engineer.calculate_features_online(current_bar, indicator_mgr)
            online_time = (time.perf_counter() - start) * 1_000_000  # Convert to microseconds
            online_times.append(online_time)

        avg_online_time = np.mean(online_times)
        p99_online_time = np.percentile(online_times, 99)

        print(f"  Online processing (avg): {avg_online_time:.2f}μs")
        print(f"  Online processing (P99): {p99_online_time:.2f}μs")

        self.test_results["performance"]["online_avg_us"] = avg_online_time
        self.test_results["performance"]["online_p99_us"] = p99_online_time

        # Check against requirements
        if avg_online_time > 500:
            self.failures.append(
                f"Online avg time {avg_online_time:.2f}μs exceeds 500μs requirement",
            )
            print("  ❌ FAILED: Average online time exceeds 500μs requirement")
        else:
            print("  ✅ PASSED: Average online time meets <500μs requirement")

        if p99_online_time > 2000:
            self.failures.append(f"Online P99 time {p99_online_time:.2f}μs exceeds 2ms requirement")
            print("  ❌ FAILED: P99 online time exceeds 2ms requirement")
        else:
            print("  ✅ PASSED: P99 online time meets <2ms requirement")

    def _test_feature_parity(self):
        """
        Test feature parity between batch and online calculation.
        """
        print("Testing batch vs online feature parity...")

        # Create test data with various market conditions
        df = self._create_test_data(100)
        config = FeatureConfig(
            include_microstructure=False,  # Test basic features first
            include_trade_flow=False,
        )
        engineer = FeatureEngineer(config)

        # Calculate batch features
        features_batch, _ = engineer.calculate_features_batch(df)

        # Calculate online features sequentially
        indicator_mgr = IndicatorManager(config)
        online_features = []

        for i in range(len(df)):
            # Create bar from row
            bar = self._create_bar(
                float(df.iloc[i]["close"]),
                float(df.iloc[i]["volume"]),
                float(df.iloc[i]["high"]),
                float(df.iloc[i]["low"]),
            )
            indicator_mgr.update_from_bar(bar)

            # Calculate online features
            current_bar = {
                "close": float(df.iloc[i]["close"]),
                "high": float(df.iloc[i]["high"]),
                "low": float(df.iloc[i]["low"]),
                "volume": float(df.iloc[i]["volume"]),
            }

            if i >= 30:  # Skip warm-up period
                features = engineer.calculate_features_online(current_bar, indicator_mgr)
                online_features.append(features)

        # Compare features (skip warm-up rows)
        batch_array = features_batch.iloc[30:].to_numpy()
        online_array = np.array(online_features)

        # Calculate differences
        max_diff = np.max(np.abs(batch_array - online_array))
        mean_diff = np.mean(np.abs(batch_array - online_array))

        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")

        self.test_results["feature_parity"]["max_difference"] = max_diff
        self.test_results["feature_parity"]["mean_difference"] = mean_diff

        # Check tolerance
        if max_diff > 1e-10:
            self.failures.append(f"Feature parity max diff {max_diff:.2e} exceeds 1e-10 tolerance")
            print("  ❌ FAILED: Maximum difference exceeds 1e-10 tolerance")

            # Find which features have largest differences
            for j in range(batch_array.shape[1]):
                col_diff = np.max(np.abs(batch_array[:, j] - online_array[:, j]))
                if col_diff > 1e-10:
                    feature_name = features_batch.columns[j]
                    print(f"    Feature '{feature_name}': max diff = {col_diff:.2e}")
        else:
            print("  ✅ PASSED: Feature parity within 1e-10 tolerance")

    def _test_memory_usage(self):
        """
        Test memory usage and leaks.
        """
        print("Testing memory usage...")

        config = FeatureConfig(
            include_microstructure=True,
            include_trade_flow=True,
        )
        engineer = FeatureEngineer(config)
        indicator_mgr = IndicatorManager(config)

        # Start memory tracking
        tracemalloc.start()
        gc.collect()

        # Get baseline memory
        baseline = tracemalloc.get_traced_memory()[0]

        # Process many bars
        for i in range(10000):
            bar = self._create_bar(100.0 + (i % 100) * 0.1, 1000 + (i % 100))
            indicator_mgr.update_from_bar(bar)

            if i % 100 == 0:
                current_bar = {
                    "close": 100.0 + (i % 100) * 0.1,
                    "high": 101.0,
                    "low": 99.0,
                    "volume": 1000.0,
                }
                _ = engineer.calculate_features_online(current_bar, indicator_mgr)

        # Get final memory
        gc.collect()
        final_memory = tracemalloc.get_traced_memory()[0]
        memory_growth = (final_memory - baseline) / (1024 * 1024)  # Convert to MB

        tracemalloc.stop()

        print(f"  Memory growth after 10k bars: {memory_growth:.2f} MB")

        self.test_results["memory"]["growth_mb"] = memory_growth

        # Check for memory leaks
        if memory_growth > 10:
            self.failures.append(f"Memory growth {memory_growth:.2f}MB exceeds 10MB limit")
            print("  ❌ FAILED: Excessive memory growth detected")
        else:
            print("  ✅ PASSED: Memory usage is stable")

        # Test buffer limits
        print(f"  Price history maxlen: {indicator_mgr.price_history['closes'].__class__.__name__}")
        if len(indicator_mgr.price_history["closes"]) > 1000:
            self.failures.append("Price history not bounded")
            print("  ❌ FAILED: Price history not properly bounded")
        else:
            print("  ✅ PASSED: Price history properly bounded")

    def _test_edge_cases(self):
        """
        Test edge cases and error handling.
        """
        print("Testing edge cases...")

        config = FeatureConfig()
        engineer = FeatureEngineer(config)

        # Test 1: Empty DataFrame
        print("  Testing empty DataFrame...")
        try:
            empty_df = pd.DataFrame()
            features, _ = engineer.calculate_features_batch(empty_df)
            if len(features) == 0:
                print("    ✅ PASSED: Empty DataFrame handled correctly")
            else:
                self.failures.append("Empty DataFrame not handled correctly")
                print("    ❌ FAILED: Empty DataFrame produced non-empty features")
        except Exception as e:
            self.failures.append(f"Empty DataFrame caused exception: {e}")
            print(f"    ❌ FAILED: Exception on empty DataFrame: {e}")

        # Test 2: Single row DataFrame
        print("  Testing single row DataFrame...")
        try:
            single_df = pd.DataFrame(
                {
                    "close": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "volume": [1000.0],
                },
            )
            features, _ = engineer.calculate_features_batch(single_df)
            if len(features) == 1:
                print("    ✅ PASSED: Single row handled correctly")
            else:
                self.failures.append("Single row DataFrame not handled correctly")
                print("    ❌ FAILED: Single row produced incorrect number of features")
        except Exception as e:
            self.failures.append(f"Single row DataFrame caused exception: {e}")
            print(f"    ❌ FAILED: Exception on single row: {e}")

        # Test 3: Zero prices
        print("  Testing zero prices...")
        try:
            zero_df = pd.DataFrame(
                {
                    "close": [0.0, 0.0, 0.0],
                    "high": [0.0, 0.0, 0.0],
                    "low": [0.0, 0.0, 0.0],
                    "volume": [0.0, 0.0, 0.0],
                },
            )
            features, _ = engineer.calculate_features_batch(zero_df)
            if not features.isnull().all().all() and not np.isinf(features.values).any():
                print("    ✅ PASSED: Zero prices handled without NaN/Inf")
            else:
                self.failures.append("Zero prices produced NaN or Inf values")
                print("    ❌ FAILED: Zero prices produced NaN or Inf")
        except Exception as e:
            self.failures.append(f"Zero prices caused exception: {e}")
            print(f"    ❌ FAILED: Exception on zero prices: {e}")

        # Test 4: Extreme volatility
        print("  Testing extreme volatility...")
        try:
            volatile_df = pd.DataFrame(
                {
                    "close": [100.0, 200.0, 50.0, 300.0, 10.0],
                    "high": [250.0, 250.0, 100.0, 350.0, 50.0],
                    "low": [50.0, 150.0, 25.0, 250.0, 5.0],
                    "volume": [10000.0, 20000.0, 5000.0, 30000.0, 1000.0],
                },
            )
            features, _ = engineer.calculate_features_batch(volatile_df)
            if not np.isinf(features.values).any():
                print("    ✅ PASSED: Extreme volatility handled correctly")
            else:
                self.failures.append("Extreme volatility produced Inf values")
                print("    ❌ FAILED: Extreme volatility produced Inf")
        except Exception as e:
            self.failures.append(f"Extreme volatility caused exception: {e}")
            print(f"    ❌ FAILED: Exception on extreme volatility: {e}")

    def _test_integration(self):
        """
        Test integration with other components.
        """
        print("Testing component integration...")

        # Test with different configurations
        configs = [
            FeatureConfig(),  # Default
            FeatureConfig(include_microstructure=True),
            FeatureConfig(include_trade_flow=True),
            FeatureConfig(include_microstructure=True, include_trade_flow=True),
        ]

        for i, config in enumerate(configs):
            print(f"  Testing config {i+1}/{len(configs)}...")
            try:
                engineer = FeatureEngineer(config)
                df = self._create_test_data(50)

                # Test batch processing
                features_batch, _ = engineer.calculate_features_batch(df)

                # Test online processing
                indicator_mgr = IndicatorManager(config)
                for j in range(10):
                    bar = self._create_bar(100.0 + j * 0.1, 1000 + j)
                    indicator_mgr.update_from_bar(bar)

                current_bar = {
                    "close": 100.5,
                    "high": 101.0,
                    "low": 100.0,
                    "volume": 1500.0,
                }
                features_online = engineer.calculate_features_online(current_bar, indicator_mgr)

                # Verify feature counts match
                expected_features = len(config.get_feature_names())
                actual_batch = len(features_batch.columns)
                actual_online = len(features_online)

                if actual_batch == expected_features and actual_online == expected_features:
                    print(
                        f"    ✅ Config {i+1}: Feature counts match ({expected_features} features)",
                    )
                else:
                    self.failures.append(f"Config {i+1}: Feature count mismatch")
                    print(f"    ❌ Config {i+1}: Feature count mismatch")
                    print(
                        f"       Expected: {expected_features}, Batch: {actual_batch}, Online: {actual_online}",
                    )

            except Exception as e:
                self.failures.append(f"Config {i+1} integration failed: {e}")
                print(f"    ❌ Config {i+1}: Integration failed: {e}")

    def _create_test_data(self, n_rows: int) -> pd.DataFrame:
        """
        Create realistic test data.
        """
        np.random.seed(42)

        # Generate realistic price series
        price = 100.0
        prices = []
        volumes = []

        for _ in range(n_rows):
            # Random walk with mean reversion
            change = np.random.normal(0, 0.5)
            price = price * (1 + change / 100)
            price = max(price, 10.0)  # Minimum price

            volume = np.random.lognormal(7, 1)

            prices.append(price)
            volumes.append(volume)

        # Create OHLC from close prices
        df = pd.DataFrame(
            {
                "close": prices,
                "volume": volumes,
            },
        )

        # Generate high/low with realistic spreads
        df["high"] = df["close"] * (1 + np.random.uniform(0.001, 0.01, n_rows))
        df["low"] = df["close"] * (1 - np.random.uniform(0.001, 0.01, n_rows))
        df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])

        # Ensure OHLC consistency (low <= open <= high, low <= close <= high)
        df["low"] = df[["low", "open", "close"]].min(axis=1)
        df["high"] = df[["high", "open", "close"]].max(axis=1)

        return df

    def _create_bar(
        self,
        close: float,
        volume: float,
        high: float = None,
        low: float = None,
    ) -> Bar:
        """
        Create a Bar object for testing.
        """
        if high is None:
            high = close * 1.005
        if low is None:
            low = close * 0.995

        instrument_id = InstrumentId.from_str("TEST.VENUE")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        return Bar(
            bar_type=bar_type,
            open=Price.from_str(str(close)),
            high=Price.from_str(str(high)),
            low=Price.from_str(str(low)),
            close=Price.from_str(str(close)),
            volume=Quantity.from_str(str(volume)),
            ts_event=0,
            ts_init=0,
        )

    def _generate_report(self):
        """
        Generate final QA report.
        """
        print("\n" + "=" * 80)
        print("QA TEST REPORT SUMMARY")
        print("=" * 80)

        # Performance Summary
        print("\nPERFORMANCE METRICS:")
        print("-" * 40)
        perf = self.test_results.get("performance", {})
        if perf:
            print(f"  Batch Processing: {perf.get('batch_time_ms', 0):.2f}ms for 1000 bars")
            print(f"  Online Processing (avg): {perf.get('online_avg_us', 0):.2f}μs")
            print(f"  Online Processing (P99): {perf.get('online_p99_us', 0):.2f}μs")
            print(f"  Throughput: {perf.get('batch_bars_per_second', 0):.0f} bars/second")

        # Feature Parity Summary
        print("\nFEATURE PARITY:")
        print("-" * 40)
        parity = self.test_results.get("feature_parity", {})
        if parity:
            print(f"  Max Difference: {parity.get('max_difference', 0):.2e}")
            print(f"  Mean Difference: {parity.get('mean_difference', 0):.2e}")

        # Memory Summary
        print("\nMEMORY USAGE:")
        print("-" * 40)
        memory = self.test_results.get("memory", {})
        if memory:
            print(f"  Growth after 10k bars: {memory.get('growth_mb', 0):.2f} MB")

        # Overall Status
        print("\nOVERALL STATUS:")
        print("-" * 40)
        if not self.failures:
            print("✅ ALL TESTS PASSED - PRODUCTION READY")
            self.test_results["status"] = "PASSED"
        else:
            print(f"❌ {len(self.failures)} TESTS FAILED:")
            for failure in self.failures:
                print(f"  - {failure}")
            self.test_results["status"] = "FAILED"
            self.test_results["failures"] = self.failures

        print("\n" + "=" * 80)


if __name__ == "__main__":
    tester = FeatureEngineerQATester()
    results = tester.run_all_tests()
