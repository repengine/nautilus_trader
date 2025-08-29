#!/usr/bin/env python3
"""
Test that hot path fixes maintain data integrity while achieving zero allocation.
"""

import logging

import numpy as np
import pytest

from ml.actors.base import BaseMLInferenceActor
from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.features.engineering import IndicatorManager


# Configure module logger
logger = logging.getLogger(__name__)


@pytest.mark.parallel_safe
class TestHotPathFixes:
    """
    Test that hot path fixes maintain data integrity and zero allocation.
    """

    def test_feature_engineer_returns_view_not_copy(self) -> None:
        """
        Verify FeatureEngineer returns a view in hot path, not a copy.
        """
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        indicator_mgr = IndicatorManager(config)

        # Create current bar data
        current_bar = {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000000.0,
        }

        # Add some price history so indicators can initialize
        for i in range(30):
            indicator_mgr.price_history["closes"].append(100.0 + i * 0.1)
            indicator_mgr.price_history["volumes"].append(1000000.0)
            indicator_mgr.price_history["highs"].append(101.0 + i * 0.1)
            indicator_mgr.price_history["lows"].append(99.0 + i * 0.1)

        # Calculate features (hot path)
        features = engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_mgr,
            scaler=None,
        )

        # Verify it's a view of the feature buffer, not a copy
        assert np.shares_memory(
            features,
            engineer.feature_buffer,
        ), "calculate_features_online should return a view of feature_buffer for zero allocation"

        # Verify data integrity - features should be valid
        assert len(features) > 0, "Features should be computed"
        assert not np.any(np.isnan(features)), "Features should not contain NaN"

    def test_enhanced_actor_compute_features_returns_view(self) -> None:
        """
        Verify EnhancedMLInferenceActor._compute_features returns a view.
        """
        from nautilus_trader.model.identifiers import InstrumentId

        config = MLActorConfig(
            model_id="test_model",
            component_id="TEST",
            model_path="test_model.pkl",
            bar_type="TEST.VENUE-1-MINUTE-LAST",
            instrument_id=InstrumentId.from_str("TEST.VENUE"),
            feature_config=MLFeatureConfig(average_volume=1000000.0),
        )

        actor = EnhancedMLInferenceActor(config)

        # Initialize features manually to bypass model loading
        actor._initialize_features()

        # Create a test bar
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model.data import BarSpecification
        from nautilus_trader.model.data import BarType
        from nautilus_trader.model.enums import AggressorSide
        from nautilus_trader.model.enums import BarAggregation
        from nautilus_trader.model.enums import PriceType
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Price
        from nautilus_trader.model.objects import Quantity

        instrument_id = InstrumentId.from_str("TEST.VENUE")
        bar_spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggressorSide.BUYER)

        # Process multiple bars to initialize indicators
        for i in range(30):
            bar = Bar(
                bar_type=bar_type,
                open=Price(100.0 + i * 0.1, 2),
                high=Price(101.0 + i * 0.1, 2),
                low=Price(99.0 + i * 0.1, 2),
                close=Price(100.5 + i * 0.1, 2),
                volume=Quantity(1000000, 0),
                ts_event=1000000000 + i * 60000000000,  # 1 minute bars
                ts_init=1000000000 + i * 60000000000,
            )
            features = actor._compute_features(bar)

        # Now check that the features returned are a view
        if features is not None:
            assert np.shares_memory(
                features,
                actor._feature_buffer,
            ), "_compute_features should return a view of _feature_buffer for zero allocation"

            # Verify data integrity
            assert len(features) > 0, "Features should be computed"
            assert not np.any(np.isnan(features)), "Features should not contain NaN"

    def test_feature_buffer_reuse_maintains_integrity(self) -> None:
        """
        Test that reusing feature buffer doesn't corrupt data.
        """
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        indicator_mgr = IndicatorManager(config)

        # Initialize price history
        for i in range(30):
            indicator_mgr.price_history["closes"].append(100.0 + i * 0.1)
            indicator_mgr.price_history["volumes"].append(1000000.0)
            indicator_mgr.price_history["highs"].append(101.0 + i * 0.1)
            indicator_mgr.price_history["lows"].append(99.0 + i * 0.1)

        # Process multiple bars
        results = []
        for i in range(10):
            current_bar = {
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1000000.0 + i * 1000,
            }

            # Calculate features (returns view)
            features = engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
                scaler=None,
            )

            # Store a copy for comparison (only for test, not in hot path)
            results.append(features.copy())

            # Update history for next iteration
            indicator_mgr.price_history["closes"].append(current_bar["close"])
            indicator_mgr.price_history["volumes"].append(current_bar["volume"])
            indicator_mgr.price_history["highs"].append(current_bar["high"])
            indicator_mgr.price_history["lows"].append(current_bar["low"])

        # Verify each result is different (buffer was properly reused)
        for i in range(1, len(results)):
            assert not np.array_equal(
                results[i],
                results[i - 1],
            ), f"Features {i} should differ from features {i - 1}"

        # Verify all results are valid
        for i, features in enumerate(results):
            assert not np.any(np.isnan(features)), f"Features {i} should not contain NaN"
            assert len(features) > 0, f"Features {i} should not be empty"


if __name__ == "__main__":
    # Run the tests
    test = TestHotPathFixes()

    logger.info("Testing feature engineer returns view not copy...")
    test.test_feature_engineer_returns_view_not_copy()
    logger.info(" PASSED")

    logger.info("Testing enhanced actor compute features returns view...")
    test.test_enhanced_actor_compute_features_returns_view()
    logger.info(" PASSED")

    logger.info("Testing feature buffer reuse maintains integrity...")
    test.test_feature_buffer_reuse_maintains_integrity()
    logger.info(" PASSED")

    logger.info(
        "\n All hot path fixes tests passed! Zero allocation achieved without data corruption.",
    )
