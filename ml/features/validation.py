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
Feature parity validation for ML components.

This module provides utilities to validate that features computed in batch (training)
and real-time (inference) modes are identical. Feature parity is CRITICAL for ML
model performance in production - even small discrepancies can cause model failure.

"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt

# Import ML dependencies with centralized management
from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.config.constants import MLConstants
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId


POLARS_AVAILABLE = HAS_POLARS


class FeatureParityError(Exception):
    """
    Exception raised when feature parity validation fails.
    """

    def __init__(
        self,
        message: str,
        max_difference: float,
        tolerance: float,
        failing_features: list[str],
    ) -> None:
        """
        Initialize feature parity error.

        Parameters
        ----------
        message : str
            Error message.
        max_difference : float
            Maximum difference found between batch and online features.
        tolerance : float
            Tolerance threshold that was exceeded.
        failing_features : list[str]
            List of feature names that failed parity check.

        """
        super().__init__(message)
        self.max_difference = max_difference
        self.tolerance = tolerance
        self.failing_features = failing_features


class FeatureParityValidator:
    """
    Validates feature parity between batch and real-time computation.

    This class provides comprehensive validation to ensure that features computed during
    training (batch mode) are identical to those computed during inference (real-time
    mode). This is critical for ML model performance in production environments.

    """

    def __init__(self, config: FeatureConfig | None = None, tolerance: float | None = None) -> None:
        """
        Initialize feature parity validator.

        Parameters
        ----------
        config : FeatureConfig, optional
            Configuration for feature engineering. If None, uses default.
        tolerance : float, optional
            Tolerance for feature parity validation. If None, uses default
            from MLConstants.FEATURE_PARITY_TOLERANCE.

        """
        self.config = config or FeatureConfig()
        self.tolerance = tolerance or MLConstants.FEATURE_PARITY_TOLERANCE
        self.feature_engineer = FeatureEngineer(self.config)

    def _prepare_validation_data(
        self,
        df: Any,
        start_idx: int,
        end_idx: int,
    ) -> tuple[Any, IndicatorManager, BarType]:
        """
        Prepare data for validation.
        """
        # Calculate batch features
        batch_features_df, _ = self.feature_engineer.calculate_features_batch(df, fit_scaler=False)

        # Initialize indicator manager for online calculation
        indicator_mgr = IndicatorManager(self.config)

        # Create dummy bar type for online processing
        instrument_id = InstrumentId.from_str("VALIDATION.TEST")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        # CRITICAL FIX: Warm up indicators with data up to start_idx-1 to match batch processing
        # The batch processing processes all data, so indicators have state from the full dataset
        # We need to warm up online indicators to the same state as of start_idx-1
        for i in range(start_idx):
            bar = self._create_bar_from_row(df, i, bar_type)
            indicator_mgr.update_from_bar(bar)

        return batch_features_df, indicator_mgr, bar_type

    def _extract_current_bar_data(self, df: Any, i: int) -> dict[str, float]:
        """
        Extract current bar data from DataFrame.
        """
        if POLARS_AVAILABLE and hasattr(df, "to_numpy"):
            return {
                "open": float(df["open"][i]) if "open" in df.columns else float(df["close"][i]),
                "high": float(df["high"][i]) if "high" in df.columns else float(df["close"][i]),
                "low": float(df["low"][i]) if "low" in df.columns else float(df["close"][i]),
                "close": float(df["close"][i]),
                "volume": float(df["volume"][i]) if "volume" in df.columns else 0.0,
            }
        else:
            return {
                "open": (
                    float(df.iloc[i]["open"])
                    if "open" in df.columns
                    else float(df.iloc[i]["close"])
                ),
                "high": (
                    float(df.iloc[i]["high"])
                    if "high" in df.columns
                    else float(df.iloc[i]["close"])
                ),
                "low": (
                    float(df.iloc[i]["low"]) if "low" in df.columns else float(df.iloc[i]["close"])
                ),
                "close": float(df.iloc[i]["close"]),
                "volume": float(df.iloc[i]["volume"]) if "volume" in df.columns else 0.0,
            }

    def _calculate_online_features(
        self,
        df: Any,
        indicator_mgr: IndicatorManager,
        bar_type: BarType,
        start_idx: int,
        end_idx: int,
    ) -> list[npt.NDArray[np.float32]]:
        """
        Calculate online features for validation period.

        This method simulates TRUE real-time feature calculation by processing one bar
        at a time, which is essential for accurate parity validation.

        """
        online_features_list = []

        # Process each bar individually to match real-time processing
        for i in range(start_idx, end_idx):
            # Update indicator with current bar (one by one - true online simulation)
            bar = self._create_bar_from_row(df, i, bar_type)
            indicator_mgr.update_from_bar(bar)

            # Get current bar data for feature calculation
            current_bar = self._extract_current_bar_data(df, i)

            # Calculate online features using the current indicator state
            online_features = self.feature_engineer.calculate_features_online(
                current_bar,
                indicator_mgr,
                scaler=None,
            )
            # CRITICAL FIX: Copy the features since calculate_features_online returns a view
            # of the reused feature_buffer that gets overwritten on each call
            online_features_list.append(online_features.copy())

        return online_features_list

    def _create_validation_report(
        self,
        differences: npt.NDArray[np.float64],
        max_differences_per_feature: npt.NDArray[np.float64],
        max_difference_overall: float,
        feature_names: list[str],
        failing_features: list[str],
        validation_time: float,
        start_idx: int,
        end_idx: int,
        detailed_report: bool,
    ) -> dict[str, Any]:
        """
        Create validation report.
        """
        # Create detailed report
        feature_differences = {}
        if detailed_report:
            for i, feature_name in enumerate(feature_names):
                feature_differences[feature_name] = {
                    "max_difference": float(max_differences_per_feature[i]),
                    "mean_difference": float(np.mean(differences[:, i])),
                    "std_difference": float(np.std(differences[:, i])),
                    "passed": max_differences_per_feature[i] <= self.tolerance,
                }

        # Check if validation passed
        parity_passed = max_difference_overall <= self.tolerance

        return {
            "parity_passed": parity_passed,
            "max_difference": float(max_difference_overall),
            "tolerance": self.tolerance,
            "failing_features": failing_features,
            "n_failing_features": len(failing_features),
            "n_total_features": len(feature_names),
            "feature_differences": feature_differences,
            "validation_time": validation_time,
            "n_samples_validated": end_idx - start_idx,
            "validation_range": f"[{start_idx}:{end_idx}]",
        }

    def validate_parity(
        self,
        df: Any,  # pl.DataFrame or pd.DataFrame
        start_idx: int = 50,
        end_idx: int | None = None,
        detailed_report: bool = True,
    ) -> dict[str, Any]:
        """
        Validate feature parity between batch and online computation.

        Parameters
        ----------
        df : pl.DataFrame or pd.DataFrame
            DataFrame with OHLCV data for validation.
        start_idx : int, default 50
            Starting index for validation (allows indicators to initialize).
        end_idx : int, optional
            Ending index for validation. If None, validates to end of data.
        detailed_report : bool, default True
            Whether to include detailed per-feature analysis in the report.

        Returns
        -------
        dict[str, Any]
            Validation report containing:
            - 'parity_passed': bool indicating if validation passed
            - 'max_difference': maximum difference found
            - 'tolerance': tolerance used
            - 'failing_features': list of features that failed
            - 'feature_differences': dict of max difference per feature
            - 'validation_time': time taken for validation
            - 'n_samples_validated': number of samples validated

        Raises
        ------
        FeatureParityError
            If feature parity validation fails.

        """
        start_time = time.time()

        if end_idx is None:
            end_idx = len(df)

        # Ensure we don't go beyond data length
        end_idx = min(end_idx, len(df))

        if start_idx >= end_idx:
            msg = f"start_idx ({start_idx}) must be less than end_idx ({end_idx})"
            raise ValueError(msg)

        # Prepare validation data
        batch_features_df, indicator_mgr, bar_type = self._prepare_validation_data(
            df,
            start_idx,
            end_idx,
        )

        # Get feature names
        feature_names = self.config.get_feature_names()

        # Calculate online features
        online_features_list = self._calculate_online_features(
            df,
            indicator_mgr,
            bar_type,
            start_idx,
            end_idx,
        )

        # Convert online features to array
        online_features_array = np.array(online_features_list)

        # Get corresponding batch features
        if POLARS_AVAILABLE and hasattr(batch_features_df, "select"):
            batch_features_array = batch_features_df.select(feature_names).to_numpy()[
                start_idx:end_idx
            ]
        else:
            batch_features_array = batch_features_df[feature_names].to_numpy()[start_idx:end_idx]

        # Validate shapes match
        if online_features_array.shape != batch_features_array.shape:
            msg = (
                f"Shape mismatch: online {online_features_array.shape} "
                f"vs batch {batch_features_array.shape}"
            )
            raise ValueError(msg)

        # Calculate differences
        differences = np.abs(online_features_array - batch_features_array)
        max_differences_per_feature = np.max(differences, axis=0)
        max_difference_overall = np.max(max_differences_per_feature)

        # Find failing features
        failing_feature_indices = np.where(max_differences_per_feature > self.tolerance)[0]
        failing_features = [feature_names[i] for i in failing_feature_indices]

        validation_time = time.time() - start_time

        # Create report
        report = self._create_validation_report(
            differences,
            max_differences_per_feature,
            max_difference_overall,
            feature_names,
            failing_features,
            validation_time,
            start_idx,
            end_idx,
            detailed_report,
        )

        # Raise exception if validation failed
        if not report["parity_passed"]:
            error_msg = (
                f"Feature parity validation FAILED! "
                f"Max difference: {max_difference_overall:.2e} > tolerance: {self.tolerance:.2e}. "
                f"Failing features ({len(failing_features)}): {failing_features[:5]}..."
                if len(failing_features) > 5
                else f"Failing features: {failing_features}"
            )
            raise FeatureParityError(
                error_msg,
                max_difference_overall,
                self.tolerance,
                failing_features,
            )

        return report

    def _create_bar_from_row(self, df: Any, row_idx: int, bar_type: BarType) -> Any:
        """
        Create a Bar object from DataFrame row.

        This method is still needed for indicator warm-up during validation.

        """
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.objects import Price
        from nautilus_trader.model.objects import Quantity

        # Check if it's a Polars DataFrame by checking for specific Polars methods
        if POLARS_AVAILABLE and hasattr(df, "row") and callable(getattr(df, "row", None)):
            # Polars DataFrame
            row = df.row(row_idx)
            columns = df.columns
            row_dict = dict(zip(columns, row))
        else:
            # Pandas DataFrame
            row_dict = df.iloc[row_idx].to_dict()

        # Extract OHLCV values with defaults
        close_val = float(row_dict.get("close", 0.0))
        open_val = float(row_dict.get("open", close_val))
        high_val = float(row_dict.get("high", close_val))
        low_val = float(row_dict.get("low", close_val))
        volume_val = float(row_dict.get("volume", 0.0))

        # Create Price objects
        open_price = Price.from_str(f"{open_val:.8f}")
        high_price = Price.from_str(f"{high_val:.8f}")
        low_price = Price.from_str(f"{low_val:.8f}")
        close_price = Price.from_str(f"{close_val:.8f}")
        volume_qty = Quantity.from_str(f"{volume_val:.8f}")

        # Create timestamps (use dummy values for validation)
        ts_event = 1_000_000_000 + row_idx * 60_000_000_000  # 1 minute intervals
        ts_init = ts_event

        return Bar(
            bar_type=bar_type,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume_qty,
            ts_event=ts_event,
            ts_init=ts_init,
        )

    def validate_performance(
        self,
        df: Any,  # pl.DataFrame or pd.DataFrame
        n_iterations: int = 100,
        target_latency_ms: float | None = None,
    ) -> dict[str, Any]:
        """
        Validate feature computation performance.

        Parameters
        ----------
        df : pl.DataFrame or pd.DataFrame
            DataFrame with OHLCV data for performance testing.
        n_iterations : int, default 100
            Number of iterations to run for performance measurement.
        target_latency_ms : float, optional
            Target latency in milliseconds. If None, uses MLConstants.MAX_INFERENCE_LATENCY_MS.

        Returns
        -------
        dict[str, Any]
            Performance report containing timing statistics.

        """
        if target_latency_ms is None:
            target_latency_ms = MLConstants.MAX_INFERENCE_LATENCY_MS

        # Initialize indicator manager
        indicator_mgr = IndicatorManager(self.config)

        # Create dummy bar type
        instrument_id = InstrumentId.from_str("PERF.TEST")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        # Warm up indicators
        warmup_size = min(50, len(df) // 2)
        for i in range(warmup_size):
            bar = self._create_bar_from_row(df, i, bar_type)
            indicator_mgr.update_from_bar(bar)

        # Performance measurement
        latencies = []
        start_idx = warmup_size

        for iteration in range(n_iterations):
            # Use different rows for each iteration
            test_idx = start_idx + (iteration % (len(df) - start_idx))

            # Update indicator (simulate real-time update)
            bar = self._create_bar_from_row(df, test_idx, bar_type)
            indicator_mgr.update_from_bar(bar)

            # Prepare current bar data (handle both polars and pandas)
            if POLARS_AVAILABLE and hasattr(df, "to_numpy"):
                current_bar = {
                    "open": (
                        float(df["open"][test_idx])
                        if "open" in df.columns
                        else float(df["close"][test_idx])
                    ),
                    "high": (
                        float(df["high"][test_idx])
                        if "high" in df.columns
                        else float(df["close"][test_idx])
                    ),
                    "low": (
                        float(df["low"][test_idx])
                        if "low" in df.columns
                        else float(df["close"][test_idx])
                    ),
                    "close": float(df["close"][test_idx]),
                    "volume": float(df["volume"][test_idx]) if "volume" in df.columns else 0.0,
                }
            else:
                current_bar = {
                    "open": (
                        float(df.iloc[test_idx]["open"])
                        if "open" in df.columns
                        else float(df.iloc[test_idx]["close"])
                    ),
                    "high": (
                        float(df.iloc[test_idx]["high"])
                        if "high" in df.columns
                        else float(df.iloc[test_idx]["close"])
                    ),
                    "low": (
                        float(df.iloc[test_idx]["low"])
                        if "low" in df.columns
                        else float(df.iloc[test_idx]["close"])
                    ),
                    "close": float(df.iloc[test_idx]["close"]),
                    "volume": float(df.iloc[test_idx]["volume"]) if "volume" in df.columns else 0.0,
                }

            # Measure feature calculation time
            start_time = time.perf_counter()
            self.feature_engineer.calculate_features_online(current_bar, indicator_mgr, scaler=None)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        latencies_array = np.array(latencies)

        performance_report = {
            "n_iterations": n_iterations,
            "target_latency_ms": target_latency_ms,
            "mean_latency_ms": float(np.mean(latencies_array)),
            "std_latency_ms": float(np.std(latencies_array)),
            "min_latency_ms": float(np.min(latencies_array)),
            "max_latency_ms": float(np.max(latencies_array)),
            "p50_latency_ms": float(np.percentile(latencies_array, 50)),
            "p95_latency_ms": float(np.percentile(latencies_array, 95)),
            "p99_latency_ms": float(np.percentile(latencies_array, 99)),
            "performance_passed": float(np.percentile(latencies_array, 99)) <= target_latency_ms,
            "n_features": len(self.config.get_feature_names()),
        }

        return performance_report

    def generate_test_data(
        self,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> Any:
        """
        Generate synthetic test data for validation.

        Parameters
        ----------
        n_samples : int, default 1000
            Number of samples to generate.
        seed : int, default 42
            Random seed for reproducible data.

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            Generated test data with OHLCV columns.

        """
        rng = np.random.default_rng(seed)

        # Generate realistic price data using geometric Brownian motion
        initial_price = 100.0
        dt = 1.0 / 252  # Daily data
        drift = 0.05  # 5% annual drift
        volatility = 0.2  # 20% annual volatility

        prices = [initial_price]
        for _ in range(n_samples - 1):
            prev_price = prices[-1]
            random_shock = rng.normal(0, 1)
            price_change = prev_price * (drift * dt + volatility * np.sqrt(dt) * random_shock)
            new_price = prev_price + price_change
            prices.append(max(new_price, 0.01))  # Ensure positive prices

        # Generate OHLC from close prices
        closes = np.array(prices)

        # Add some intraday variation
        high_noise = rng.uniform(0.001, 0.02, n_samples)
        low_noise = rng.uniform(0.001, 0.02, n_samples)
        open_noise = rng.uniform(-0.01, 0.01, n_samples)

        highs = closes * (1 + high_noise)
        lows = closes * (1 - low_noise)
        opens = np.roll(closes, 1) * (1 + open_noise)
        opens[0] = closes[0]  # First open equals first close

        # Ensure OHLC consistency
        for i in range(n_samples):
            high_val = max(opens[i], closes[i]) * (1 + high_noise[i] / 2)
            low_val = min(opens[i], closes[i]) * (1 - low_noise[i] / 2)
            highs[i] = high_val
            lows[i] = low_val

        # Generate volume data
        base_volume = 1000000
        volume_noise = rng.uniform(0.5, 2.0, n_samples)
        volumes = base_volume * volume_noise

        # Create DataFrame (Polars or Pandas based on availability)
        if POLARS_AVAILABLE:
            data = {
                "timestamp": pl.datetime_range(
                    start=pl.datetime(2024, 1, 1),
                    end=pl.datetime(2024, 1, 1) + pl.duration(days=n_samples - 1),
                    interval="1d",
                    eager=True,
                ),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
            return pl.DataFrame(data)
        else:
            import pandas as pd

            data = {
                "timestamp": pd.date_range(
                    start="2024-01-01",
                    periods=n_samples,
                    freq="D",
                ),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
            return pd.DataFrame(data)


def validate_feature_parity(
    df: Any,  # pl.DataFrame or pd.DataFrame
    config: FeatureConfig | None = None,
    tolerance: float | None = None,
    start_idx: int = 50,
    end_idx: int | None = None,
) -> dict[str, Any]:
    """
    Validate feature parity.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        DataFrame with OHLCV data for validation.
    config : FeatureConfig, optional
        Configuration for feature engineering. If None, uses default.
    tolerance : float, optional
        Tolerance for feature parity validation. If None, uses default.
    start_idx : int, default 50
        Starting index for validation.
    end_idx : int, optional
        Ending index for validation.

    Returns
    -------
    dict[str, Any]
        Validation report.

    Raises
    ------
    FeatureParityError
        If feature parity validation fails.

    """
    validator = FeatureParityValidator(config=config, tolerance=tolerance)
    return validator.validate_parity(
        df=df,
        start_idx=start_idx,
        end_idx=end_idx,
        detailed_report=True,
    )
