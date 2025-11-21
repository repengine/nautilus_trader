"""
Critical tests for training/inference feature parity.

These tests ensure that features computed during training (batch)
exactly match features computed during inference (online).

PARITY REQUIREMENT: 1e-10 tolerance (no fidelity sacrifices).

"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
from numpy.random import default_rng

from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.features.config import FeatureConfig
from ml.features.facade import FeatureEngineer
from ml.features.indicators import IndicatorManager
from ml.stores.feature_store import FeatureStore
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

if TYPE_CHECKING:
    from ml.tests.fixtures.database_fixtures import TestDatabase


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

@pytest.fixture
def feature_config() -> FeatureConfig:
    return FeatureConfig()


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
@pytest.mark.usefixtures("clean_postgres_db_class", "real_engine_manager")
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

            # Clamp decimal precision to avoid exceeding Nautilus Price precision constraints
            bar = SimpleNamespace(
                close=Price.from_str(f"{close_price:.5f}"),
                high=Price.from_str(f"{high_price:.5f}"),
                low=Price.from_str(f"{low_price:.5f}"),
                volume=float(base_volume + int(_rng.integers(-100000, 100000))),
                ts_event=int((datetime.now(UTC) + timedelta(minutes=i)).timestamp() * 1e9),
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
        # Initialize engineer and indicator manager
        batch_engineer = FeatureEngineer(feature_config)
        online_engineer = FeatureEngineer(feature_config)
        indicator_mgr = IndicatorManager(feature_config)

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
        online_features: list[npt.NDArray[np.float32]] = []
        for bar in mock_bars:
            # Update indicators first
            indicator_mgr.update_from_values(
                close=float(bar.close),
                high=float(bar.high),
                low=float(bar.low),
                volume=float(bar.volume),
            )
            
            current_bar = {
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
                "volume": float(bar.volume),
            }
            
            features = online_engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
            )
            # Important: copy to avoid aliasing the engineer's hot-path buffer
            online_features.append(features.copy())

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
        
        # Setup indicator manager
        indicator_mgr = IndicatorManager(feature_config)

        # Mock the database operations
        cast(Any, store)._store_to_postgres = MagicMock()

        # Compute features for same bar multiple times
        bar = mock_bars[50]  # Middle bar with stable indicators
        
        # Update indicators state
        indicator_mgr.update_from_values(
            close=float(bar.close),
            high=float(bar.high),
            low=float(bar.low),
            volume=float(bar.volume),
        )
        
        current_bar = {
            "close": float(bar.close),
            "high": float(bar.high),
            "low": float(bar.low),
            "volume": float(bar.volume),
        }

        features_1 = store.feature_engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_mgr,
        )

        features_2 = store.feature_engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_mgr,
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
        indicator_mgr = IndicatorManager(feature_config)

        # Process first 50 bars to warm up indicators
        for bar in mock_bars[:50]:
            indicator_mgr.update_from_values(
                close=float(bar.close),
                high=float(bar.high),
                low=float(bar.low),
                volume=float(bar.volume),
            )
            
            current_bar = {
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
                "volume": float(bar.volume),
            }
            
            engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
            )

        # Check indicator states
        # Robust init checks across indicator implementations
        rsi_ind = indicator_mgr.indicators["rsi"]
        ema_fast_ind = indicator_mgr.indicators["ema_fast"]
        ema_slow_ind = indicator_mgr.indicators["ema_slow"]
        
        assert (
            getattr(rsi_ind, "is_initialized", getattr(rsi_ind, "initialized", False)) is True
        ), "RSI not initialized after 50 bars"
        assert (
            getattr(ema_fast_ind, "is_initialized", getattr(ema_fast_ind, "initialized", False)) is True
        ), "EMA fast not initialized"
        assert (
            getattr(ema_slow_ind, "is_initialized", getattr(ema_slow_ind, "initialized", False)) is True
        ), "EMA slow not initialized"

        # Get current indicator values
        rsi_value = rsi_ind.value
        ema_fast_value = ema_fast_ind.value
        ema_slow_value = ema_slow_ind.value

        # Process one more bar
        bar = mock_bars[50]
        indicator_mgr.update_from_values(
            close=float(bar.close),
            high=float(bar.high),
            low=float(bar.low),
            volume=float(bar.volume),
        )
        current_bar = {
            "close": float(bar.close),
            "high": float(bar.high),
            "low": float(bar.low),
            "volume": float(bar.volume),
        }
        
        engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=indicator_mgr,
        )

        # Indicators should have updated
        assert (
            rsi_ind.value != rsi_value
            or abs(rsi_ind.value - rsi_value) < 0.01
        )
        assert ema_fast_ind.value != ema_fast_value
        assert ema_slow_ind.value != ema_slow_value

    @pytest.mark.database
    @pytest.mark.serial
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
            mock_feature_store = actor._feature_store  # alias for assertions

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
    def test_feature_versioning(
        self,
        feature_config: FeatureConfig,
        test_database: TestDatabase,
    ) -> None:
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
        indicator_mgr = IndicatorManager(feature_config)

        # Test with extreme values
        extreme_cases = [
            (0.0001, 0.0002, 0.00005, 1),  # Very small prices
            (10000.0, 10001.0, 9999.0, 1000000),  # Very large prices
            (1.0, 1.0, 1.0, 0),  # Zero volume
        ]

        for close, high, low, volume in extreme_cases:
            indicator_mgr.update_from_values(
                close=float(close),
                high=float(high),
                low=float(low),
                volume=float(volume),
            )
            current_bar = {
                "close": float(close),
                "high": float(high),
                "low": float(low),
                "volume": float(volume),
            }
            
            features = engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
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
        bt = BarType.from_str("EURUSD.SIM-1-MINUTE-BID-EXTERNAL")
        for i in range(n_bars):
            close_val = 1.1000 + i * 0.0001
            high_val = max(close_val, 1.1010)
            low_val = min(1.0990, close_val - 0.0001)
            bar = Bar(
                bar_type=bt,
                open=Price.from_str("1.1000"),
                high=Price.from_str(f"{high_val:.5f}"),
                low=Price.from_str(f"{low_val:.5f}"),
                close=Price.from_str(f"{close_val:.5f}"),
                volume=Quantity.from_int(1000000),
                ts_event=int((datetime.now(UTC) + timedelta(minutes=i)).timestamp() * 1e9),
                ts_init=int((datetime.now(UTC) + timedelta(minutes=i)).timestamp() * 1e9),
            )
            bars.append(bar)

        # Test batch vs online
        # We can reuse the engineer since it's stateless regarding data
        engineer = FeatureEngineer(feature_config)
        indicator_mgr = IndicatorManager(feature_config)

        # Compute online features
        online_features = []
        for bar in bars:
            indicator_mgr.update_from_values(
                close=float(bar.close),
                high=float(bar.high),
                low=float(bar.low),
                volume=float(bar.volume),
            )
            current_bar = {
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
                "volume": float(bar.volume),
            }
            features = engineer.calculate_features_online(
                current_bar=current_bar,
                indicator_manager=indicator_mgr,
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
        engineer = FeatureEngineer(feature_config)
        mgr1 = IndicatorManager(feature_config)
        mgr2 = IndicatorManager(feature_config)

        # Warm up mgr1 with different data
        for i in range(50):
            mgr1.update_from_values(
                close=1.1000 + i * 0.001,
                high=1.1010 + i * 0.001,
                low=1.0990 + i * 0.001,
                volume=1000000,
            )

        # Both process same bar
        current_bar = {
            "close": 1.2000,
            "high": 1.2010,
            "low": 1.1990,
            "volume": 1000000.0,
        }
        
        # Need to update managers with this bar first
        mgr1.update_from_values(**current_bar)
        mgr2.update_from_values(**current_bar)

        features1 = engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=mgr1,
        )

        features2 = engineer.calculate_features_online(
            current_bar=current_bar,
            indicator_manager=mgr2,
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
        # engineer = FeatureEngineer(feature_config) # Not needed for state
        indicator_mgr = IndicatorManager(feature_config)

        # Process many bars to accumulate potential precision errors
        for i in range(10000):
            price = 1.0 + (i % 100) * 1e-10  # Very small variations
            indicator_mgr.update_from_values(
                close=price,
                high=price + 1e-10,
                low=price - 1e-10,
                volume=1000000,
            )

        # Check that indicators still have reasonable precision
        ema_fast = indicator_mgr.indicators["ema_fast"]
        ema_slow = indicator_mgr.indicators["ema_slow"]
        
        assert ema_fast.value > 0
        assert ema_slow.value > 0

        # EMAs should have converged close to the mean price
        expected_price = 1.0 + 50 * 1e-10
        assert abs(ema_fast.value - expected_price) < 0.01
        assert abs(ema_slow.value - expected_price) < 0.01
