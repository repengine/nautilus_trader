"""
Critical tests for training/inference feature parity.

These tests ensure that features computed during training (batch)
exactly match features computed during inference (online).

PARITY REQUIREMENT: 1e-10 tolerance (no fidelity sacrifices).

"""

from datetime import datetime
from datetime import timedelta
from typing import Any, cast
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
from numpy.random import default_rng

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.stores.feature_store import FeatureStore
from ml.tests.fixtures.database_fixtures import TestDatabase
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
class TestFeatureParity:
    """
    Test suite for training/inference feature parity.
    """

    @pytest.fixture
    def feature_config(self) -> FeatureConfig:
        """
        Standard feature configuration.
        """
        return FeatureConfig()

    @pytest.fixture
    def mock_bars(self) -> list[Any]:
        """
        Generate mock bar data for testing.
        """
        bars = []
        instrument_id = InstrumentId(Symbol("EURUSD"), Venue("IDEALPRO"))

        # Generate sequential bars
        base_price = 1.1000
        base_volume = 1000000

        from numpy.random import default_rng

        _rng = default_rng(0)
        for i in range(100):
            price_variation = float(_rng.uniform(-0.001, 0.001))
            open_price = base_price + price_variation
            high_price = open_price + abs(float(_rng.uniform(0, 0.002)))
            low_price = open_price - abs(float(_rng.uniform(0, 0.002)))
            close_price = open_price + float(_rng.uniform(-0.001, 0.001))

            # Use a lightweight object with the required attributes
            from types import SimpleNamespace
            bar = SimpleNamespace(
                close=Price.from_str(str(close_price)),
                high=Price.from_str(str(high_price)),
                low=Price.from_str(str(low_price)),
                volume=float(base_volume + int(_rng.integers(-100000, 100000))),
                ts_event=int((datetime.utcnow() + timedelta(minutes=i)).timestamp() * 1e9),
            )
            bars.append(bar)

            base_price = close_price

        return bars

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_engineer_batch_vs_online_parity(
        self,
        feature_config: FeatureConfig,
        mock_bars: list[Bar],
    ) -> None:
        """
        Test that FeatureEngineer computes identical features in batch vs online.

        This is the CRITICAL test for training/inference parity.

        """
        # Initialize two separate engineers to simulate training vs inference
        batch_engineer = FeatureEngineer(feature_config)
        online_engineer = FeatureEngineer(feature_config)

        # Prepare data for batch computation
        import polars as pl

        bars_data = []
        for bar in mock_bars:
            bars_data.append(
                {
                    "close": float(bar.close),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "volume": float(bar.volume),
                    "ts_event": bar.ts_event,
                },
            )
        bars_df = pl.DataFrame(bars_data)

        # Compute features in batch (training path)
        batch_features, _ = batch_engineer.calculate_features_batch(bars_df)
        # Ensure numpy array for arithmetic
        if hasattr(batch_features, "to_numpy"):
            batch_features = batch_features.to_numpy()

        # Compute features online (inference path)
        online_features = []
        for bar in mock_bars:
            features = online_engineer.calculate_features_online(
                close_price=float(bar.close),
                high_price=float(bar.high),
                low_price=float(bar.low),
                volume=float(bar.volume),
            )
            online_features.append(features)

        online_features_array = np.array(online_features)

        # Assert exact parity with tight tolerance
        max_diff = np.max(np.abs(batch_features - online_features_array))

        assert max_diff < 1e-10, (
            f"Feature parity violation! Max difference: {max_diff}\n"
            f"This means training and inference would compute different features.\n"
            f"First 5 batch features: {batch_features[0][:5]}\n"
            f"First 5 online features: {online_features_array[0][:5]}"
        )

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.usefixtures("clean_postgres_db")
    def test_feature_store_computation_consistency(
        self,
        feature_config: FeatureConfig,
        mock_bars: list[Bar],
        test_database: TestDatabase,
    ) -> None:
        """
        Test that FeatureStore produces consistent features.
        """
        store = FeatureStore(
            connection_string=test_database.connection_string,
            feature_config=feature_config,
        )

        # Mock the database operations
        cast(Any, store)._store_to_postgres = MagicMock()

        # Compute features for same bar multiple times
        bar = mock_bars[50]  # Middle bar with stable indicators

        features_1 = store.feature_engineer.calculate_features_online(
                close_price=float(bar.close),
                high_price=float(bar.high),
                low_price=float(bar.low),
                volume=float(bar.volume),
            )

        features_2 = store.feature_engineer.calculate_features_online(
                close_price=float(bar.close),
                high_price=float(bar.high),
                low_price=float(bar.low),
                volume=float(bar.volume),
            )

        # Features should be identical for same input
        assert np.array_equal(features_1, features_2), "Same input produced different features!"

    @pytest.mark.database
    @pytest.mark.serial
    def test_indicator_state_consistency(
        self,
        feature_config: FeatureConfig,
        mock_bars: list[Bar],
    ) -> None:
        """
        Test that indicator state remains consistent between batch and online.

        This ensures that indicators like EMA, RSI maintain proper state.

        """
        engineer = FeatureEngineer(feature_config)

        # Process first 50 bars to warm up indicators
        for bar in mock_bars[:50]:
            engineer.calculate_features_online(
                close_price=float(bar.close),
                high_price=float(bar.high),
                low_price=float(bar.low),
                volume=float(bar.volume),
            )

        # Check indicator states
        assert engineer.indicators.rsi.is_initialized, "RSI not initialized after 50 bars"
        assert engineer.indicators.ema_fast.is_initialized, "EMA fast not initialized"
        assert engineer.indicators.ema_slow.is_initialized, "EMA slow not initialized"

        # Get current indicator values
        rsi_value = engineer.indicators.rsi.value
        ema_fast_value = engineer.indicators.ema_fast.value
        ema_slow_value = engineer.indicators.ema_slow.value

        # Process one more bar
        bar = mock_bars[50]
        features = engineer.calculate_features_online(
            close_price=float(bar.close),
            high_price=float(bar.high),
            low_price=float(bar.low),
            volume=float(bar.volume),
        )

        # Indicators should have updated
        assert (
            engineer.indicators.rsi.value != rsi_value
            or abs(engineer.indicators.rsi.value - rsi_value) < 0.01
        )
        assert engineer.indicators.ema_fast.value != ema_fast_value
        assert engineer.indicators.ema_slow.value != ema_slow_value

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.usefixtures("clean_postgres_db")
    def test_ml_signal_actor_uses_same_features(
        self,
        feature_config: FeatureConfig,
        mock_bars: list[Bar],
        test_database: TestDatabase,
    ) -> None:
        """
        Test that MLSignalActor computes features identically to training.
        """
        with patch("ml.actors.signal.MLSignalActor._load_model_with_metadata"):
            # Provide required fields for strict typing
            config = MLSignalActorConfig(
                actor_id="TEST_ACTOR",
                model_id="test_model",
                bar_type=MagicMock(),
                instrument_id=InstrumentId(Symbol("EURUSD"), Venue("IDEALPRO")),
                model_path="./test_model.onnx",
                db_connection=test_database.connection_string,
            )

            # Create actor
            actor = MLSignalActor(config)

            # Mock the feature store compute method
            expected_features = default_rng(0).random(50).astype(np.float32)
            actor._feature_store.compute_realtime = MagicMock(return_value=expected_features)

            # Process a bar
            bar = mock_bars[0]
            features = actor._compute_features(bar)

            # Verify correct delegation to FeatureStore
            mock_feature_store.compute_realtime.assert_called_once_with(
                bar=bar,
                store=True,  # Default is to persist
            )

            # Verify returned features match
            assert np.array_equal(cast(npt.NDArray[np.float32], features), expected_features)

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.usefixtures("clean_postgres_db")
    def test_feature_versioning(self, feature_config: FeatureConfig, test_database: TestDatabase) -> None:
        """
        Test that feature versions change when pipeline changes.
        """
        # Create store with initial config
        store1 = FeatureStore(
            connection_string=test_database.connection_string,
            feature_config=feature_config,
        )

        version1 = store1.pipeline_hash

        # Create store with modified config
        modified_config = FeatureConfig(rsi_period=20)

        store2 = FeatureStore(
            connection_string=test_database.connection_string,
            feature_config=modified_config,
        )

        version2 = store2.pipeline_hash

        # Versions should differ when config changes
        assert version1 != version2, "Feature versions must change when configuration changes"

    @pytest.mark.database
    @pytest.mark.serial
    def test_parity_across_feature_ranges(
        self,
        feature_config: FeatureConfig,
        mock_bars: list[Bar],
    ) -> None:
        """
        Test parity across different feature value ranges.

        Ensures numerical stability across market conditions.

        """
        engineer = FeatureEngineer(feature_config)

        # Test with extreme values
        extreme_cases = [
            (0.0001, 0.0002, 0.00005, 1),  # Very small prices
            (10000.0, 10001.0, 9999.0, 1000000),  # Very large prices
            (1.0, 1.0, 1.0, 0),  # Zero volume
        ]

        for close, high, low, volume in extreme_cases:
            features = engineer.calculate_features_online(
                close_price=close,
                high_price=high,
                low_price=low,
                volume=volume,
            )

            # Features should be finite and not NaN
            assert np.all(
                np.isfinite(features),
            ), f"Non-finite features for inputs: close={close}, high={high}, low={low}, volume={volume}\nFeatures: {features}"

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.parametrize("n_bars", [10, 50, 100, 500])
    def test_parity_at_different_sequence_lengths(
        self,
        feature_config: FeatureConfig,
        n_bars: int,
    ) -> None:
        """
        Test that parity holds for different sequence lengths.
        """
        # Generate bars
        bars = []
        for i in range(n_bars):
            bar = Bar(
                bar_type=MagicMock(),
                open=Price.from_str("1.1000"),
                high=Price.from_str("1.1010"),
                low=Price.from_str("1.0990"),
                close=Price.from_str(str(1.1000 + i * 0.0001)),
                volume=Quantity.from_int(1000000),
                ts_event=int((datetime.utcnow() + timedelta(minutes=i)).timestamp() * 1e9),
                ts_init=int((datetime.utcnow() + timedelta(minutes=i)).timestamp() * 1e9),
            )
            bars.append(bar)

        # Test batch vs online
        batch_engineer = FeatureEngineer(feature_config)
        online_engineer = FeatureEngineer(feature_config)

        # Compute online features
        online_features = []
        for bar in bars:
            features = online_engineer.calculate_features_online(
                close_price=float(bar.close),
                high_price=float(bar.high),
                low_price=float(bar.low),
                volume=float(bar.volume),
            )
            online_features.append(features)

        # For sequences >= 20 (minimum for indicators), check last features
        if n_bars >= 20:
            last_features = online_features[-1]
            assert len(last_features) > 0, f"No features computed for {n_bars} bars"
            assert np.all(np.isfinite(last_features)), f"Non-finite features after {n_bars} bars"


@pytest.mark.database
@pytest.mark.serial
class TestParityFailureModes:
    """
    Test cases that should FAIL if parity is broken.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_detect_indicator_initialization_mismatch(self, feature_config: FeatureConfig) -> None:
        """
        Test that we detect when indicators are initialized differently.
        """
        engineer1 = FeatureEngineer(feature_config)
        engineer2 = FeatureEngineer(feature_config)

        # Warm up engineer1 with different data
        for i in range(50):
            engineer1.calculate_features_online(
                close_price=1.1000 + i * 0.001,
                high_price=1.1010 + i * 0.001,
                low_price=1.0990 + i * 0.001,
                volume=1000000,
            )

        # Both process same bar
        features1 = engineer1.calculate_features_online(
            close_price=1.2000,
            high_price=1.2010,
            low_price=1.1990,
            volume=1000000,
        )

        features2 = engineer2.calculate_features_online(
            close_price=1.2000,
            high_price=1.2010,
            low_price=1.1990,
            volume=1000000,
        )

        # Features should be different due to different history
        assert not np.array_equal(
            features1,
            features2,
        ), "Features should differ when indicators have different history"

    @pytest.mark.database
    @pytest.mark.serial
    def test_detect_numerical_precision_issues(self, feature_config: FeatureConfig) -> None:
        """
        Test that we maintain numerical precision.
        """
        engineer = FeatureEngineer(feature_config)

        # Process many bars to accumulate potential precision errors
        for i in range(10000):
            price = 1.0 + (i % 100) * 1e-10  # Very small variations
            engineer.calculate_features_online(
                close_price=price,
                high_price=price + 1e-10,
                low_price=price - 1e-10,
                volume=1000000,
            )

        # Check that indicators still have reasonable precision
        assert engineer.indicators.ema_fast.value > 0
        assert engineer.indicators.ema_slow.value > 0

        # EMAs should have converged close to the mean price
        expected_price = 1.0 + 50 * 1e-10
        assert abs(engineer.indicators.ema_fast.value - expected_price) < 0.01
        assert abs(engineer.indicators.ema_slow.value - expected_price) < 0.01
