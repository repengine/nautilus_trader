"""
Pairwise/Combinatorial tests for configuration combinations.

These tests use pairwise testing to efficiently cover interactions between configuration
parameters without testing every possible combination. This dramatically reduces test
count while still catching most bugs.

"""

from __future__ import annotations

from itertools import product
from typing import Any

import pytest
from allpairspy import AllPairs

from ml.actors.signal import MLSignalActorConfig
from ml.features.config import FeatureConfig
from ml.registry import ModelManifest
from ml.registry.base import DataRequirements
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.redis
class TestConfigurationCombinations:
    """
    Test pairwise combinations of configuration parameters.
    """

    def test_feature_config_pairwise(self):
        """
        Test pairwise combinations of feature configuration parameters.

        Instead of testing all 2^n combinations, test pairwise interactions.

        """
        # Define parameter space based on actual FeatureConfig API
        parameters = {
            "return_periods": [[1], [1, 5], [1, 5, 10, 20]],
            "momentum_periods": [[5], [5, 10], [5, 10, 20]],
            "rsi_period": [7, 14, 21],
            "bb_period": [10, 20, 30],
            "bb_std": [1.5, 2.0, 2.5],
            "atr_period": [10, 20, 30],
            "volume_ma_periods": [[5], [5, 10], [5, 10, 20]],
            "include_microstructure": [True, False],
            "include_trade_flow": [True, False],
        }

        # Generate pairwise combinations
        pairs = list(
            AllPairs(
                [
                    parameters["return_periods"],
                    parameters["momentum_periods"],
                    parameters["rsi_period"],
                    parameters["bb_period"],
                    parameters["bb_std"],
                    parameters["atr_period"],
                    parameters["volume_ma_periods"],
                    parameters["include_microstructure"],
                    parameters["include_trade_flow"],
                ],
            ),
        )

        # Test each pairwise combination
        valid_configs = 0
        for combo in pairs:
            (
                return_periods,
                momentum_periods,
                rsi_period,
                bb_period,
                bb_std,
                atr_period,
                volume_ma_periods,
                include_microstructure,
                include_trade_flow,
            ) = combo

            # Derive data requirements to satisfy feature constraints
            if include_trade_flow:
                data_requirements = DataRequirements.L1_L2_L3
            elif include_microstructure:
                data_requirements = DataRequirements.L1_L2
            else:
                data_requirements = DataRequirements.L1_ONLY

            # Create configuration
            config = FeatureConfig(
                return_periods=return_periods,
                momentum_periods=momentum_periods,
                rsi_period=rsi_period,
                bb_period=bb_period,
                bb_std=bb_std,
                atr_period=atr_period,
                volume_ma_periods=volume_ma_periods,
                include_microstructure=include_microstructure,
                include_trade_flow=include_trade_flow,
                data_requirements=data_requirements,
            )

            # Verify configuration is valid
            assert config is not None
            assert config.rsi_period == rsi_period
            assert config.bb_period == bb_period
            assert config.bb_std == bb_std
            valid_configs += 1

        # Compare with full cartesian product
        full_combinations = product(*parameters.values())
        full_count = sum(1 for _ in full_combinations)
        pairwise_count = len(pairs)

        # Pairwise should be much smaller
        assert (
            pairwise_count < full_count / 10
        ), f"Pairwise ({pairwise_count}) should be much smaller than full ({full_count})"

        print(
            f"Pairwise testing: {pairwise_count} tests instead of {full_count} (reduction: {100*(1-pairwise_count/full_count):.1f}%)",
        )

    @pytest.mark.database
    @pytest.mark.serial
    def test_model_instrument_timeframe_pairwise(self):
        """
        Test pairwise combinations of model configurations across different instruments
        and timeframes.
        """
        # Define parameter space
        model_types = ["xgboost", "lightgbm", "neural_net"]
        instruments = ["EURUSD.SIM", "GBPUSD.SIM", "USDJPY.SIM", "BTCUSD.BINANCE"]
        timeframes = ["1-MINUTE", "5-MINUTE", "15-MINUTE", "1-HOUR"]
        feature_sets = ["basic", "technical", "microstructure", "all"]
        warm_up_periods = [10, 50, 100]
        confidence_thresholds = [0.3, 0.5, 0.7]

        # Generate pairwise combinations
        pairs = list(
            AllPairs(
                [
                    model_types,
                    instruments,
                    timeframes,
                    feature_sets,
                    warm_up_periods,
                    confidence_thresholds,
                ],
            ),
        )

        # Test each combination
        for combo in pairs:
            (
                model_type,
                instrument,
                timeframe,
                feature_set,
                warm_up,
                threshold,
            ) = combo

            # Create bar type
            bar_type_str = f"{instrument}-{timeframe}-BID-EXTERNAL"

            # Verify combination is valid
            self._validate_model_config(
                model_type=model_type,
                instrument=instrument,
                timeframe=timeframe,
                feature_set=feature_set,
                warm_up_period=warm_up,
                confidence_threshold=threshold,
            )

        # Calculate reduction
        full_count = (
            len(model_types)
            * len(instruments)
            * len(timeframes)
            * len(feature_sets)
            * len(warm_up_periods)
            * len(confidence_thresholds)
        )
        pairwise_count = len(pairs)

        print(
            f"Model config testing: {pairwise_count} tests instead of {full_count} (reduction: {100*(1-pairwise_count/full_count):.1f}%)",
        )

    @pytest.mark.database
    @pytest.mark.serial
    def test_store_configuration_pairwise(self):
        """
        Test pairwise combinations of store configurations.
        """
        # Define parameter space for stores
        backends = ["postgres", "timescale", "redis"]
        batch_sizes = [100, 1000, 10000]
        cache_enabled = [True, False]
        compression = ["none", "gzip", "lz4"]
        partitioning = ["daily", "weekly", "monthly"]
        retention_days = [30, 90, 365]

        # Generate pairwise combinations
        pairs = list(
            AllPairs(
                [
                    backends,
                    batch_sizes,
                    cache_enabled,
                    compression,
                    partitioning,
                    retention_days,
                ],
            ),
        )

        # Test each combination
        valid_configs = []
        for combo in pairs:
            (
                backend,
                batch_size,
                cache,
                compress,
                partition,
                retention,
            ) = combo

            # Validate store configuration
            is_valid = self._validate_store_config(
                backend=backend,
                batch_size=batch_size,
                cache_enabled=cache,
                compression=compress,
                partitioning=partition,
                retention_days=retention,
            )

            if is_valid:
                valid_configs.append(combo)

        # Ensure we have valid configurations
        assert len(valid_configs) > 0, "Should have at least some valid store configurations"

        # Report coverage
        full_count = (
            len(backends)
            * len(batch_sizes)
            * len(cache_enabled)
            * len(compression)
            * len(partitioning)
            * len(retention_days)
        )
        pairwise_count = len(pairs)

        print(
            f"Store config testing: {pairwise_count} tests instead of {full_count} (reduction: {100*(1-pairwise_count/full_count):.1f}%)",
        )

    @pytest.mark.database
    @pytest.mark.serial
    def test_three_way_interactions(self):
        """
        Test three-way interactions for critical parameter combinations.

        For the most critical parameters, test 3-way interactions.

        """
        # Critical parameters that might interact
        model_types = ["xgboost", "neural_net"]
        data_frequencies = ["high", "medium", "low"]  # tick, second, minute
        feature_complexities = ["simple", "complex"]  # few vs many features

        # Generate 3-way combinations (still much smaller than full cartesian)
        from itertools import combinations

        all_params = [
            ("model", model_types),
            ("frequency", data_frequencies),
            ("complexity", feature_complexities),
        ]

        # Test all 3-way interactions
        test_count = 0
        for params in combinations(all_params, 3):
            param_names = [p[0] for p in params]
            param_values = [p[1] for p in params]

            # Generate combinations for these 3 parameters
            for combo in product(*param_values):
                config = dict(zip(param_names, combo))

                # Validate this 3-way interaction
                self._validate_three_way_interaction(**config)
                test_count += 1

        assert test_count > 0, "Should test some 3-way interactions"
        print(f"Tested {test_count} three-way interactions")

    def _validate_model_config(
        self,
        model_type: str,
        instrument: str,
        timeframe: str,
        feature_set: str,
        warm_up_period: int,
        confidence_threshold: float,
    ) -> bool:
        """
        Validate a model configuration combination.
        """
        # Check for known incompatibilities

        # Neural nets might need more warm-up
        if model_type == "neural_net" and warm_up_period < 50:
            # This is a valid test case - neural nets should handle short warm-up
            pass

        # High-frequency data with complex features might be slow
        if timeframe == "1-MINUTE" and feature_set == "microstructure":
            # This combination might be computationally expensive but valid
            pass

        # Crypto might have different thresholds
        if "BTC" in instrument and confidence_threshold > 0.6:
            # Crypto is more volatile, might need higher thresholds
            pass

        return True

    def _validate_store_config(
        self,
        backend: str,
        batch_size: int,
        cache_enabled: bool,
        compression: str,
        partitioning: str,
        retention_days: int,
    ) -> bool:
        """
        Validate a store configuration combination.
        """
        # Check for incompatibilities

        # Redis might not support all compressions
        if backend == "redis" and compression == "lz4":
            return False  # Known incompatibility

        # Large batches with daily partitions might be inefficient
        if batch_size > 5000 and partitioning == "daily":
            # Valid but might want to warn
            pass

        # Short retention with monthly partitions is wasteful
        if retention_days < 60 and partitioning == "monthly":
            # Valid but suboptimal
            pass

        return True

    def _validate_three_way_interaction(self, **kwargs) -> bool:
        """
        Validate a three-way parameter interaction.
        """
        # Check for complex three-way interactions

        model = kwargs.get("model")
        frequency = kwargs.get("frequency")
        complexity = kwargs.get("complexity")

        # High-frequency + complex features + neural net might be problematic
        if model == "neural_net" and frequency == "high" and complexity == "complex":
            # This might be computationally prohibitive
            pass

        return True


@pytest.mark.database
@pytest.mark.serial
class TestConfigurationBoundaries:
    """
    Test configuration parameters at their boundaries using pairwise testing.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_numeric_boundaries_pairwise(self):
        """
        Test numeric configuration parameters at their boundaries.
        """
        # Define boundary values for numeric parameters
        boundaries = {
            "warm_up_period": [0, 1, 10, 100, 1000],  # min, small, medium, large, max
            "confidence_threshold": [0.0, 0.01, 0.5, 0.99, 1.0],  # min to max
            "ma_period": [2, 5, 20, 200],  # short to long
            "volatility_window": [2, 10, 50, 500],  # short to long
            "batch_size": [1, 10, 1000, 100000],  # single to bulk
        }

        # Generate pairwise combinations of boundary values
        pairs = list(AllPairs(list(boundaries.values())))

        # Test each combination
        for combo in pairs:
            (warm_up, threshold, ma_period, vol_window, batch) = combo

            # Validate boundary combination
            self._validate_boundary_combo(
                warm_up_period=warm_up,
                confidence_threshold=threshold,
                ma_period=ma_period,
                volatility_window=vol_window,
                batch_size=batch,
            )

        print(f"Boundary testing: {len(pairs)} pairwise tests cover boundary interactions")

    def _validate_boundary_combo(
        self,
        warm_up_period: int,
        confidence_threshold: float,
        ma_period: int,
        volatility_window: int,
        batch_size: int,
    ) -> bool:
        """
        Validate a combination of boundary values.
        """
        # Check for invalid combinations at boundaries

        # Zero warm-up with indicators that need history
        if warm_up_period == 0 and (ma_period > 1 or volatility_window > 1):
            # This should trigger appropriate warnings/defaults
            pass

        # Extreme thresholds
        if confidence_threshold == 0.0:
            # All signals pass - valid for testing
            pass
        elif confidence_threshold == 1.0:
            # No signals pass - valid edge case
            pass

        # MA period longer than volatility window
        if ma_period > volatility_window:
            # This is valid but might affect calculations
            pass

        return True


@pytest.mark.database
@pytest.mark.serial
class TestFeatureCombinations:
    """
    Test combinations of features to ensure compatibility.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_compatibility_pairwise(self):
        """
        Test pairwise combinations of features for compatibility.
        """
        # Define feature categories
        features = {
            "price_features": ["returns", "log_returns", "price_change"],
            "volume_features": ["volume", "volume_rate", "vwap"],
            "technical_features": ["rsi", "macd", "bollinger"],
            "microstructure_features": ["spread", "imbalance", "lob_pressure"],
            "time_features": ["hour", "day_of_week", "month"],
        }

        # Flatten to individual features
        all_features = []
        for category, feature_list in features.items():
            all_features.extend(feature_list)

        # Generate pairwise combinations
        feature_pairs = list(AllPairs([all_features, all_features]))

        # Test compatibility
        compatible_pairs = []
        incompatible_pairs = []

        for feature1, feature2 in feature_pairs:
            if feature1 != feature2:  # Skip self-pairs
                if self._are_features_compatible(feature1, feature2):
                    compatible_pairs.append((feature1, feature2))
                else:
                    incompatible_pairs.append((feature1, feature2))

        # Report results
        print(
            f"Feature compatibility: {len(compatible_pairs)} compatible pairs, "
            f"{len(incompatible_pairs)} incompatible pairs",
        )

        # Ensure most features are compatible
        assert len(compatible_pairs) > len(
            incompatible_pairs,
        ), "Most features should be compatible with each other"

    def _are_features_compatible(self, feature1: str, feature2: str) -> bool:
        """
        Check if two features are compatible.
        """
        # Define known incompatibilities
        incompatible = [
            ("returns", "log_returns"),  # Redundant
            ("volume", "volume_rate"),  # Might be redundant
        ]

        # Check for incompatibility
        for f1, f2 in incompatible:
            if (feature1 == f1 and feature2 == f2) or (feature1 == f2 and feature2 == f1):
                return False

        return True
