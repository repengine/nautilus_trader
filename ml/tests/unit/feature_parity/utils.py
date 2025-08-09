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
Core utilities for feature parity validation tests.

This module provides utilities for comparing batch and online feature computations to
ensure perfect parity with < 1e-10 tolerance.

"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.config.constants import MLConstants
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from ml.tests.unit.test_fixtures import MockPolarsModule


if TYPE_CHECKING:
    pass


class ParityTestUtils:
    """
    Utilities for comparing batch vs online feature computation parity.

    This class provides methods to validate that feature engineering produces identical
    results between batch processing (training) and online processing (inference) within
    the required tolerance of < 1e-10.

    """

    TOLERANCE = MLConstants.FEATURE_PARITY_TOLERANCE

    @staticmethod
    def assert_features_equal(
        batch_features: npt.NDArray[np.float32],
        online_features: npt.NDArray[np.float32],
        feature_names: list[str] | None = None,
        tolerance: float | None = None,
    ) -> None:
        """
        Assert that batch and online features are equal within tolerance.

        Parameters
        ----------
        batch_features : npt.NDArray[np.float32]
            Features computed in batch mode.
        online_features : npt.NDArray[np.float32]
            Features computed in online mode.
        feature_names : list[str], optional
            Names of features for better error reporting.
        tolerance : float, optional
            Tolerance for comparison. Defaults to MLConstants.FEATURE_PARITY_TOLERANCE.

        Raises
        ------
        AssertionError
            If features do not match within tolerance.

        """
        tolerance = tolerance or ParityTestUtils.TOLERANCE

        # Ensure both are numpy arrays
        batch_features = np.asarray(batch_features)
        online_features = np.asarray(online_features)

        # Check shapes match
        if batch_features.shape != online_features.shape:
            msg = (
                f"Feature shapes do not match: "
                f"batch={batch_features.shape}, online={online_features.shape}"
            )
            raise AssertionError(msg)

        # Handle empty arrays
        if batch_features.size == 0:
            return

        # Check for exact parity within tolerance
        try:
            np.testing.assert_allclose(
                batch_features,
                online_features,
                rtol=tolerance,
                atol=tolerance,
                err_msg="Feature parity violation detected",
            )
        except AssertionError as e:
            # Provide detailed error information
            diff = np.abs(batch_features - online_features)
            max_diff = np.max(diff)
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)

            error_msg = (
                f"Feature parity violation:\n"
                f"  Maximum difference: {max_diff:.2e} (tolerance: {tolerance:.2e})\n"
                f"  At index: {max_diff_idx}\n"
                f"  Batch value: {batch_features[max_diff_idx]}\n"
                f"  Online value: {online_features[max_diff_idx]}\n"
            )

            if feature_names is not None:
                feature_idx = max_diff_idx[1] if len(max_diff_idx) > 1 else max_diff_idx[0]
                if feature_idx < len(feature_names):
                    error_msg += f"  Feature name: {feature_names[feature_idx]}\n"

            raise AssertionError(error_msg) from e

    @staticmethod
    def compare_feature_vectors(
        batch_df: Any,  # pl.DataFrame or pd.DataFrame
        online_features: list[npt.NDArray[np.float32]],
        feature_names: list[str],
        tolerance: float | None = None,
    ) -> None:
        """
        Compare batch DataFrame with list of online feature vectors.

        Parameters
        ----------
        batch_df : pl.DataFrame or pd.DataFrame
            Batch computed features.
        online_features : list[npt.NDArray[np.float32]]
            List of online feature vectors.
        feature_names : list[str]
            Names of features.
        tolerance : float, optional
            Tolerance for comparison.

        """
        tolerance = tolerance or ParityTestUtils.TOLERANCE

        # Convert batch to numpy
        if HAS_POLARS and hasattr(batch_df, "select"):
            batch_array = batch_df.select(feature_names).to_numpy()
        else:
            batch_array = batch_df[feature_names].to_numpy()

        # Convert online features to 2D array
        if not online_features:
            msg = "No online features provided for comparison"
            raise ValueError(msg)

        online_array = np.array(online_features)

        # Compare
        ParityTestUtils.assert_features_equal(
            batch_array,
            online_array,
            feature_names,
            tolerance,
        )

    @staticmethod
    def measure_computation_time(
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, float]:
        """
        Measure computation time for a function.

        Parameters
        ----------
        func : Callable[..., Any]
            Function to measure.
        args : Any
            Function arguments.
        kwargs : Any
            Function keyword arguments.

        Returns
        -------
        tuple[Any, float]
            Function result and computation time in milliseconds.

        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        computation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, computation_time

    @staticmethod
    def validate_performance(
        online_time: float,
        max_latency_ms: float = MLConstants.MAX_INFERENCE_LATENCY_MS,
    ) -> None:
        """
        Validate that online computation meets performance requirements.

        Parameters
        ----------
        online_time : float
            Online computation time in milliseconds.
        max_latency_ms : float, default 5.0
            Maximum allowed latency in milliseconds.

        Raises
        ------
        AssertionError
            If performance requirement is not met.

        """
        if online_time > max_latency_ms:
            msg = (
                f"Performance requirement violated: "
                f"{online_time:.2f}ms > {max_latency_ms:.2f}ms"
            )
            raise AssertionError(msg)


class TestDataGenerators:
    """
    Generators for test market data scenarios.

    This class provides various market data scenarios for testing feature parity under
    different conditions.

    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize data generator with fixed seed for reproducibility.

        Parameters
        ----------
        seed : int, default 42
            Random seed for deterministic test data.

        """
        self.rng = np.random.default_rng(seed)

    def generate_normal_ohlcv(
        self,
        n_bars: int = 100,
        base_price: float = 100.0,
        volatility: float = 0.02,
    ) -> Any:
        """
        Generate normal OHLCV market data.

        Parameters
        ----------
        n_bars : int, default 100
            Number of bars to generate.
        base_price : float, default 100.0
            Base price level.
        volatility : float, default 0.02
            Price volatility (standard deviation of returns).

        Returns
        -------
        pl.DataFrame or MockPolarsModule.DataFrame
            DataFrame with OHLCV columns.

        """
        # Generate price series with realistic relationships
        returns = self.rng.normal(0, volatility, n_bars)
        closes = np.zeros(n_bars)
        closes[0] = base_price

        for i in range(1, n_bars):
            closes[i] = closes[i - 1] * (1 + returns[i])

        # Generate OHLC from closes with realistic intrabar patterns
        opens = np.roll(closes, 1)
        opens[0] = base_price

        # Generate realistic high/low spreads
        hl_spreads = self.rng.exponential(closes * 0.005)  # 0.5% average spread
        highs = np.maximum(opens, closes) + hl_spreads * self.rng.uniform(0, 1, n_bars)
        lows = np.minimum(opens, closes) - hl_spreads * self.rng.uniform(0, 1, n_bars)

        # Ensure high >= max(open, close) and low <= min(open, close)
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        # Generate volumes with some correlation to price movements
        base_volume = 10000
        volume_multipliers = 1 + 0.5 * np.abs(returns) + self.rng.uniform(-0.2, 0.2, n_bars)
        volumes = base_volume * volume_multipliers
        volumes = np.maximum(volumes, 100)  # Minimum volume

        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }

        if HAS_POLARS:
            return pl.DataFrame(data)
        else:
            return MockPolarsModule.DataFrame(data)

    def generate_trending_data(
        self,
        n_bars: int = 100,
        base_price: float = 100.0,
        trend_strength: float = 0.001,
        noise: float = 0.02,
    ) -> Any:
        """
        Generate trending market data.

        Parameters
        ----------
        n_bars : int, default 100
            Number of bars to generate.
        base_price : float, default 100.0
            Starting price.
        trend_strength : float, default 0.001
            Strength of trend (0.1% per bar).
        noise : float, default 0.02
            Random noise level.

        Returns
        -------
        pl.DataFrame or MockPolarsModule.DataFrame
            DataFrame with trending OHLCV data.

        """
        # Generate trending prices
        trend = np.arange(n_bars) * trend_strength
        noise_component = self.rng.normal(0, noise, n_bars)

        closes = np.zeros(n_bars)
        closes[0] = base_price

        for i in range(1, n_bars):
            closes[i] = closes[i - 1] * (1 + trend[i] + noise_component[i])

        # Generate OHLC from trending closes
        opens = np.roll(closes, 1)
        opens[0] = base_price

        # Smaller spreads for trending markets
        hl_spreads = self.rng.exponential(closes * 0.003)
        highs = np.maximum(opens, closes) + hl_spreads * 0.7
        lows = np.minimum(opens, closes) - hl_spreads * 0.3

        # Ensure relationships hold
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        # Volume increases with stronger moves
        volume_base = 8000
        volume_multipliers = 1 + 2 * np.abs(trend + noise_component)
        volumes = volume_base * volume_multipliers

        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }

        if HAS_POLARS:
            return pl.DataFrame(data)
        else:
            return MockPolarsModule.DataFrame(data)

    def generate_volatile_data(
        self,
        n_bars: int = 100,
        base_price: float = 100.0,
        volatility: float = 0.05,
    ) -> Any:
        """
        Generate high-volatility market data.

        Parameters
        ----------
        n_bars : int, default 100
            Number of bars.
        base_price : float, default 100.0
            Base price.
        volatility : float, default 0.05
            High volatility level (5%).

        Returns
        -------
        pl.DataFrame or MockPolarsModule.DataFrame
            DataFrame with volatile OHLCV data.

        """
        # High volatility with occasional large moves
        returns = self.rng.normal(0, volatility, n_bars)

        # Add occasional large moves (regime changes)
        large_move_prob = 0.05
        large_moves = self.rng.uniform(0, 1, n_bars) < large_move_prob
        large_move_indices = np.where(large_moves)[0]
        if len(large_move_indices) > 0:
            returns[large_moves] *= self.rng.choice([-3, 3], size=len(large_move_indices))

        closes = np.zeros(n_bars)
        closes[0] = base_price

        for i in range(1, n_bars):
            closes[i] = closes[i - 1] * (1 + returns[i])

        opens = np.roll(closes, 1)
        opens[0] = base_price

        # Wider spreads for volatile markets
        hl_spreads = self.rng.exponential(closes * 0.01)  # 1% average spread
        highs = np.maximum(opens, closes) + hl_spreads * self.rng.uniform(0, 1.5, n_bars)
        lows = np.minimum(opens, closes) - hl_spreads * self.rng.uniform(0, 1.5, n_bars)

        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        # Higher volume during volatile periods
        volume_base = 15000
        volume_multipliers = 1 + 3 * np.abs(returns) + self.rng.uniform(0, 1, n_bars)
        volumes = volume_base * volume_multipliers

        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }

        if HAS_POLARS:
            return pl.DataFrame(data)
        else:
            return MockPolarsModule.DataFrame(data)

    def generate_gapped_data(
        self,
        n_bars: int = 100,
        base_price: float = 100.0,
        gap_probability: float = 0.05,
        gap_size_range: tuple[float, float] = (0.02, 0.05),
    ) -> Any:
        """
        Generate market data with price gaps.

        Parameters
        ----------
        n_bars : int, default 100
            Number of bars.
        base_price : float, default 100.0
            Base price.
        gap_probability : float, default 0.05
            Probability of gaps.
        gap_size_range : tuple[float, float], default (0.02, 0.05)
            Range of gap sizes as fraction of price.

        Returns
        -------
        pl.DataFrame or MockPolarsModule.DataFrame
            DataFrame with gapped OHLCV data.

        """
        closes = np.zeros(n_bars)
        closes[0] = base_price

        # Normal returns
        normal_returns = self.rng.normal(0, 0.015, n_bars)

        # Add gaps
        gap_mask = self.rng.uniform(0, 1, n_bars) < gap_probability
        gap_sizes = self.rng.uniform(*gap_size_range, n_bars)
        gap_directions = self.rng.choice([-1, 1], n_bars)
        gap_returns = gap_sizes * gap_directions

        returns = normal_returns.copy()
        returns[gap_mask] += gap_returns[gap_mask]

        for i in range(1, n_bars):
            closes[i] = closes[i - 1] * (1 + returns[i])

        # For gapped bars, open should reflect the gap
        opens = np.roll(closes, 1)
        opens[0] = base_price

        # Adjust opens for gaps
        gap_indices = np.where(gap_mask)[0]
        for idx in gap_indices:
            if idx > 0:
                gap_open = closes[idx - 1] * (1 + gap_returns[idx])
                opens[idx] = gap_open

        # Generate high/low respecting gaps
        hl_spreads = self.rng.exponential(closes * 0.004)
        highs = np.maximum(opens, closes) + hl_spreads * 0.6
        lows = np.minimum(opens, closes) - hl_spreads * 0.4

        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        # Volume spikes during gaps
        volume_base = 12000
        volume_multipliers = np.ones(n_bars)
        volume_multipliers[gap_mask] *= 2.5  # Volume spike on gaps
        volume_multipliers *= 1 + 0.8 * np.abs(returns)
        volumes = volume_base * volume_multipliers

        data = {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }

        if HAS_POLARS:
            return pl.DataFrame(data)
        else:
            return MockPolarsModule.DataFrame(data)

    def generate_with_microstructure_data(
        self,
        n_bars: int = 100,
        base_price: float = 100.0,
    ) -> Any:
        """
        Generate OHLCV data with additional microstructure columns.

        Parameters
        ----------
        n_bars : int, default 100
            Number of bars.
        base_price : float, default 100.0
            Base price.

        Returns
        -------
        pl.DataFrame or MockPolarsModule.DataFrame
            DataFrame with OHLCV and microstructure columns.

        """
        # Start with normal OHLCV
        df = self.generate_normal_ohlcv(n_bars, base_price)

        # Add microstructure data
        if HAS_POLARS:
            closes = df["close"].to_numpy()
        else:
            closes = df["close"].to_numpy()

        # Generate bid/ask based on close prices
        spreads = self.rng.exponential(closes * 0.001)  # 0.1% average spread
        bid_prices = closes - spreads / 2
        ask_prices = closes + spreads / 2

        # Generate sizes
        base_size = 1000
        bid_sizes = base_size * self.rng.exponential(1, n_bars)
        ask_sizes = base_size * self.rng.exponential(1, n_bars)

        microstructure_data = {
            "bid_price": bid_prices,
            "ask_price": ask_prices,
            "bid_size": bid_sizes,
            "ask_size": ask_sizes,
        }

        if HAS_POLARS:
            for col, data in microstructure_data.items():
                df = df.with_columns(pl.Series(col, data))
        else:
            # For MockPolarsModule, add to the data dict
            df.data.update(microstructure_data)
            df._columns.extend(microstructure_data.keys())

        return df

    def generate_with_trade_data(
        self,
        n_bars: int = 100,
        base_price: float = 100.0,
    ) -> Any:
        """
        Generate OHLCV data with additional trade flow columns.

        Parameters
        ----------
        n_bars : int, default 100
            Number of bars.
        base_price : float, default 100.0
            Base price.

        Returns
        -------
        pl.DataFrame or MockPolarsModule.DataFrame
            DataFrame with OHLCV and trade data columns.

        """
        # Start with normal OHLCV
        df = self.generate_normal_ohlcv(n_bars, base_price)

        if HAS_POLARS:
            closes = df["close"].to_numpy()
            volumes = df["volume"].to_numpy()
        else:
            closes = df["close"].to_numpy()
            volumes = df["volume"].to_numpy()

        # Generate trade prices (close to actual close)
        trade_prices = closes + self.rng.normal(0, closes * 0.0005, n_bars)

        # Trade volumes (fraction of bar volume)
        trade_volumes = volumes * self.rng.uniform(0.8, 1.0, n_bars)

        # Trade sides (1 = buy, -1 = sell)
        trade_sides = self.rng.choice([-1, 1], n_bars, p=[0.48, 0.52])  # Slight buy bias

        trade_data = {
            "trade_price": trade_prices,
            "trade_volume": trade_volumes,
            "trade_side": trade_sides,
        }

        if HAS_POLARS:
            for col, data in trade_data.items():
                df = df.with_columns(pl.Series(col, data))
        else:
            # For MockPolarsModule, add to the data dict
            df.data.update(trade_data)
            df._columns.extend(trade_data.keys())

        return df


class PerformanceProfiler:
    """
    Performance profiling utilities for feature computation.

    Measures latency and validates performance requirements for hot path operations.

    """

    def __init__(self) -> None:
        self.measurements: list[dict[str, Any]] = []

    def profile_feature_computation(
        self,
        feature_engineer: FeatureEngineer,
        indicator_manager: IndicatorManager,
        test_bars: list[dict[str, float]],
        description: str = "",
    ) -> dict[str, float]:
        """
        Profile online feature computation performance.

        Parameters
        ----------
        feature_engineer : FeatureEngineer
            Feature engineer instance.
        indicator_manager : IndicatorManager
            Indicator manager instance.
        test_bars : list[dict[str, float]]
            List of OHLCV bars for testing.
        description : str, default ""
            Description of the test scenario.

        Returns
        -------
        dict[str, float]
            Performance metrics including P50, P95, P99 latencies.

        """
        latencies = []

        for bar_data in test_bars:
            # Measure single computation
            start_time = time.perf_counter()
            feature_engineer.calculate_features_online(bar_data, indicator_manager)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate percentiles
        latencies_array = np.array(latencies)
        metrics = {
            "mean_latency_ms": float(np.mean(latencies_array)),
            "p50_latency_ms": float(np.percentile(latencies_array, 50)),
            "p95_latency_ms": float(np.percentile(latencies_array, 95)),
            "p99_latency_ms": float(np.percentile(latencies_array, 99)),
            "max_latency_ms": float(np.max(latencies_array)),
            "n_measurements": len(latencies),
        }

        # Store measurement
        self.measurements.append(
            {
                "description": description,
                "metrics": metrics,
                "timestamp": time.time(),
            },
        )

        return metrics

    def validate_latency_requirements(
        self,
        metrics: dict[str, float],
        max_p99_latency: float = MLConstants.MAX_INFERENCE_LATENCY_MS,
    ) -> None:
        """
        Validate that performance meets latency requirements.

        Parameters
        ----------
        metrics : dict[str, float]
            Performance metrics from profiling.
        max_p99_latency : float, default 5.0
            Maximum allowed P99 latency in milliseconds.

        Raises
        ------
        AssertionError
            If performance requirements are not met.

        """
        p99_latency = metrics.get("p99_latency_ms", float("inf"))

        if p99_latency > max_p99_latency:
            msg = (
                f"P99 latency requirement violated: "
                f"{p99_latency:.2f}ms > {max_p99_latency:.2f}ms"
            )
            raise AssertionError(msg)

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Get summary of all performance measurements.

        Returns
        -------
        dict[str, Any]
            Summary of performance measurements.

        """
        if not self.measurements:
            return {"total_measurements": 0}

        all_p99_latencies = [m["metrics"]["p99_latency_ms"] for m in self.measurements]
        all_mean_latencies = [m["metrics"]["mean_latency_ms"] for m in self.measurements]

        return {
            "total_measurements": len(self.measurements),
            "overall_max_p99_latency_ms": max(all_p99_latencies),
            "overall_mean_p99_latency_ms": np.mean(all_p99_latencies),
            "overall_max_mean_latency_ms": max(all_mean_latencies),
            "overall_mean_mean_latency_ms": np.mean(all_mean_latencies),
            "measurements": self.measurements,
        }
