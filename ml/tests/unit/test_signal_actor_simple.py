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
Simple unit tests for MLSignalActor to increase coverage.
"""

from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import SignalStrategy


class TestMLSignalActorSimple:
    """
    Simple tests for MLSignalActor coverage.
    """

    def test_adaptive_signal_properties(self):
        """
        Test AdaptiveSignal data class.
        """
        signal = AdaptiveSignal(
            instrument_id="EURUSD",
            prediction=0.8,
            confidence=0.9,
            adaptive_threshold=0.7,
            signal_strength=1.2,
            market_regime="trending",
            ts_event=123456789,
            ts_init=123456790,
        )

        assert signal.instrument_id == "EURUSD"
        assert signal.prediction == 0.8
        assert signal.confidence == 0.9
        assert signal.adaptive_threshold == 0.7
        assert signal.signal_strength == 1.2
        assert signal.market_regime == "trending"
        assert signal.ts_event == 123456789
        assert signal.ts_init == 123456790

    def test_signal_strategy_enum_values(self):
        """
        Test SignalStrategy enum.
        """
        assert SignalStrategy.THRESHOLD.value == "threshold"
        assert SignalStrategy.EXTREMES.value == "extremes"
        assert SignalStrategy.MOMENTUM.value == "momentum"
        assert SignalStrategy.ENSEMBLE.value == "ensemble"
        assert SignalStrategy.ADAPTIVE.value == "adaptive"

    def test_mocked_signal_actor_methods(self):
        """
        Test MLSignalActor methods with mocking.
        """
        from ml.actors.signal import MLSignalActor

        # Test __init__ logic without full initialization
        with patch("ml.actors.base.BaseMLInferenceActor.__init__"):
            with patch("ml.features.engineering.FeatureEngineer") as mock_fe:
                mock_fe.return_value.n_features = 10

                # Create a mock config
                config = Mock()
                config.feature_config = None
                config.min_signal_separation_bars = 2
                config.prediction_threshold = 0.7
                config.adaptive_window = 10
                config.ensemble_weights = None
                config.signal_strategy = Mock(value="threshold")

                # Mock the logger
                with patch.object(MLSignalActor, "log", create=True):
                    actor = MLSignalActor(config)

                    # Verify initialization
                    assert actor._signal_config == config
                    assert actor._adaptive_threshold == 0.7
                    assert actor._market_regime == "unknown"
                    assert actor._window_index == 0
                    assert actor._ensemble_weights["threshold"] == 0.4

    def test_direct_method_calls(self):
        """
        Test MLSignalActor methods directly.
        """
        from ml.actors.signal import MLSignalActor

        # Create a mock actor instance
        actor = Mock(spec=MLSignalActor)

        # Test _load_model with model
        actor._model = Mock()
        actor.log = Mock()
        MLSignalActor._load_model(actor)
        actor.log.info.assert_called()

        # Test _load_model without model
        actor._model = None
        MLSignalActor._load_model(actor)
        actor.log.warning.assert_called()

        # Test prediction methods with zero values
        features = np.array([0.1, 0.2, 0.3])

        # Test _predict_sklearn with zero prediction
        actor._model = Mock()
        actor._model.predict.return_value = np.array([0.0])
        pred, conf = MLSignalActor._predict_sklearn(actor, features)
        assert pred == 0.0
        assert conf == 0.5  # Special case for zero prediction

        # Test _predict with exception
        actor._model = Mock()
        actor._model.predict.side_effect = Exception("Test error")
        actor.log = Mock()
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0
        assert conf == 0.0
        actor.log.error.assert_called()

    def test_signal_generation_edge_cases(self):
        """
        Test signal generation edge cases.
        """
        from ml.actors.signal import MLSignalActor

        actor = Mock(spec=MLSignalActor)
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = False
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 10
        actor._signal_config.extremes_top_pct = 0.1
        actor._signal_config.momentum_lookback = 3
        actor._bars_processed = 10
        actor._last_signal_bar = 0
        actor.clock = Mock()
        actor.clock.timestamp_ns.return_value = 1000

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "TEST"
        bar.ts_event = 0
        features = np.array([0.1, 0.2])

        # Test threshold signal with log_predictions=True
        actor._config.log_predictions = True
        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.9, features)
        assert signal is not None
        assert signal.features is not None

        # Test extremes signal with exact percentile match
        actor._prediction_history = list(range(100))

        # Calculate exact thresholds
        predictions = np.array(actor._prediction_history[-10:])
        # top_threshold = np.percentile(predictions, 90)
        bottom_threshold = np.percentile(predictions, 10)

        # Test at exact threshold
        signal = MLSignalActor._generate_extremes_signal(
            actor,
            bar,
            bottom_threshold,
            0.8,
            features,
        )
        assert signal is not None

        # Test momentum signal with zero momentum
        actor._prediction_history = [0.5, 0.5, 0.5, 0.5]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.5, 0.8, features)
        assert signal is None  # No momentum

        # Test adaptive signal with exact threshold match
        actor._adaptive_threshold = 0.5
        actor._market_regime = "volatile"
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.6, 0.5, features)
        assert signal is not None
        assert signal.signal_strength == 1.0  # Exact match

        # Test ensemble with mixed signals
        actor._ensemble_weights = {
            "threshold": 0.5,
            "extremes": 0.3,
            "momentum": 0.2,
        }

        # Create mock signals with different confidences
        signal1 = Mock()
        signal1.confidence = 0.9
        signal2 = Mock()
        signal2.confidence = 0.7

        actor._generate_threshold_signal = Mock(return_value=signal1)
        actor._generate_extremes_signal = Mock(return_value=signal2)
        actor._generate_momentum_signal = Mock(return_value=None)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is not None
        # Weighted average: (0.5 * 0.9 + 0.3 * 0.7) / (0.5 + 0.3) = 0.825
        assert signal.confidence >= 0.8

    def test_indicator_state_operations(self):
        """
        Test indicator state backup and restore operations.
        """
        from ml.actors.signal import MLSignalActor

        actor = Mock(spec=MLSignalActor)
        actor._indicator_manager = Mock()
        actor._indicator_manager.price_history = {"closes": [1.10, 1.11]}

        # Create mock indicators with different states
        indicator1 = Mock()
        indicator1.value = 50.0
        indicator1.initialized = True

        indicator2 = Mock()
        indicator2.value = 100.0
        indicator2.initialized = False  # Not initialized

        actor._indicator_manager.indicators = {
            "sma": indicator1,
            "ema": indicator2,
        }

        actor._prediction_history = [0.1, 0.2, 0.3]
        actor._confidence_history = [0.6, 0.7, 0.8]
        actor._prediction_window = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        actor._confidence_window = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        actor._volatility_window = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        actor._window_index = 3
        actor._adaptive_threshold = 0.75
        actor._market_regime = "trending"
        actor._last_signal_bar = 15
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 5
        actor._signal_config.min_signal_separation_bars = 2
        actor._config = Mock()
        actor._config.prediction_threshold = 0.5
        actor._indicator_state_backup = {}
        actor.log = Mock()

        # Test backup
        MLSignalActor._backup_indicator_state(actor)

        # Verify backup contents
        backup = actor._indicator_state_backup
        assert "indicators" in backup
        assert "sma" in backup["indicators"]  # Initialized indicator
        assert "ema" not in backup["indicators"]  # Not initialized
        assert backup["indicators"]["sma"]["value"] == 50.0
        assert backup["window_index"] == 3
        assert backup["adaptive_threshold"] == 0.75

        # Test restore with different window sizes
        actor._prediction_window = np.zeros(10)  # Different size
        actor._confidence_window = np.zeros(10)
        actor._volatility_window = np.zeros(10)

        MLSignalActor._restore_indicator_state(actor)

        # Should restore with correct handling of window sizes
        assert actor._adaptive_threshold == 0.75
        assert actor._market_regime == "trending"

    def test_market_regime_edge_cases(self):
        """
        Test market regime detection edge cases.
        """
        from ml.actors.signal import MLSignalActor

        actor = Mock(spec=MLSignalActor)
        actor._indicator_manager = Mock()
        actor._market_regime = "unknown"
        actor._market_regime_metric = Mock()
        actor.log = Mock()
        actor.id = Mock()
        actor.id.value = "TEST"

        bar = Mock()

        # Test with exactly 20 data points (boundary condition)
        actor._indicator_manager.price_history = {
            "closes": [1.10 + i * 0.0001 for i in range(20)],
        }

        MLSignalActor._detect_market_regime(actor, bar)
        # Should detect regime with exactly 20 points
        assert actor._market_regime != "unknown"

        # Test with high correlation (trending)
        prices = [1.10 + i * 0.01 for i in range(25)]  # Strong trend
        actor._indicator_manager.price_history = {"closes": prices}
        actor._market_regime = "unknown"

        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "trending"

        # Test volatility calculation edge case
        # All same price (zero volatility)
        actor._indicator_manager.price_history = {
            "closes": [1.10] * 25,
        }
        actor._market_regime = "unknown"

        MLSignalActor._detect_market_regime(actor, bar)
        assert actor._market_regime == "ranging"  # Zero volatility = ranging

    def test_performance_metrics_initialization(self):
        """
        Test performance metrics initialization and tracking.
        """
        from ml.actors.signal import MLSignalActor

        actor = Mock(spec=MLSignalActor)
        actor._prediction_count = 0
        actor._total_inference_time = 0.0
        actor._config = Mock()
        actor._config.log_predictions = True
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.ADAPTIVE
        actor.id = Mock()
        actor.id.value = "TEST"
        actor.log = Mock()

        # First call should create metrics
        MLSignalActor._track_performance_metrics(actor, 0.75, 0.85, 3.5)

        # Check that metrics were created
        assert hasattr(actor, "_prediction_distribution_metric")
        assert hasattr(actor, "_confidence_distribution_metric")
        assert actor._prediction_count == 1
        assert actor._total_inference_time == 3.5

        # Verify logging with adaptive strategy
        actor.log.debug.assert_called()
        call_args = actor.log.debug.call_args[0][0]
        assert "Prediction: 0.7500" in call_args
        assert "Confidence: 0.8500" in call_args
        assert "Signal time: 3.500ms" in call_args
        assert "Strategy: adaptive" in call_args

    def test_prediction_protected_edge_cases(self):
        """
        Test _generate_prediction_protected edge cases.
        """
        from ml.actors.base import MLSignal
        from ml.actors.signal import MLSignalActor

        actor = Mock(spec=MLSignalActor)
        actor._config = Mock()
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.ENSEMBLE
        actor._signal_config.enable_regime_detection = False  # Disabled
        actor._circuit_breaker = Mock()
        actor._circuit_breaker.record_success = Mock()
        actor._circuit_breaker.record_failure = Mock()
        actor._health_monitor = Mock()
        actor._signal_generation_time_metric = Mock()
        actor._signals_generated_metric = Mock()
        actor.log = Mock()
        actor.id = Mock()
        actor.id.value = "TEST"

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        features = np.array([0.1, 0.2])

        # Test with regime detection disabled
        actor._predict = Mock(return_value=(0.8, 0.9))
        actor._update_prediction_history = Mock()
        actor._detect_market_regime = Mock()
        actor._generate_signal_by_strategy = Mock(return_value=None)
        actor._track_performance_metrics = Mock()

        MLSignalActor._generate_prediction_protected(actor, bar, features)

        # Should not call regime detection when disabled
        actor._detect_market_regime.assert_not_called()

        # Test with different signal types
        signal = MLSignal(
            instrument_id="EURUSD",
            prediction=-0.8,  # Negative prediction (sell signal)
            confidence=0.9,
            features=None,
            ts_event=0,
            ts_init=0,
        )
        actor._generate_signal_by_strategy.return_value = signal
        actor._publish_signal = Mock()

        MLSignalActor._generate_prediction_protected(actor, bar, features)

        # Verify sell signal type in metrics
        actor._signals_generated_metric.inc.assert_called()
        call_args = actor._signals_generated_metric.inc.call_args[0]
        labels = call_args[1]
        assert labels["signal_type"] == "sell"
