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
Final unit tests to reach 90% coverage for MLSignalActor.
"""

from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from ml.actors.base import MLSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import SignalStrategy


class TestSignalActorFinal90:
    """
    Tests to reach 90% coverage for MLSignalActor.
    """

    def test_predict_paths_coverage(self):
        """
        Test _predict method paths (lines 389-409).
        """
        actor = Mock()
        actor.log = Mock()
        features = np.array([0.1, 0.2, 0.3])

        # No model (lines 389-391)
        actor._model = None
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0

        # ONNX model (line 397)
        actor._model = Mock()
        actor._model.run = Mock(return_value=[np.array([[0.8]])])
        actor._model_metadata = {"input_names": ["input"]}
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.8 and conf == 0.8

        # Exception handling (lines 406-409)
        actor._model = Mock()
        actor._model.run = Mock(side_effect=Exception("Test error"))
        pred, conf = MLSignalActor._predict(actor, features)
        assert pred == 0.0 and conf == 0.0

    def test_signal_generation_coverage(self):
        """
        Test signal generation methods (lines 689-690, 702-712, 724-744, 756-775).
        """
        actor = Mock()
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = False
        actor._signal_config = Mock()
        actor._signal_config.adaptive_window = 10
        actor._signal_config.extremes_top_pct = 0.1
        actor._signal_config.momentum_lookback = 3
        actor._bars_processed = 20
        actor._last_signal_bar = 0
        actor.clock = Mock()
        actor.clock.timestamp_ns = Mock(return_value=2000)

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        bar.ts_event = 1000
        features = None

        # Threshold signal - below threshold (lines 689-690)
        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.5, 0.6, features)
        assert signal is None

        # Threshold signal - above threshold (lines 702-712)
        signal = MLSignalActor._generate_threshold_signal(actor, bar, 0.8, 0.9, features)
        assert isinstance(signal, MLSignal)
        assert signal.features is None  # log_predictions=False

        # Extremes signal - insufficient history (lines 724-726)
        actor._prediction_history = [0.5]
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 0.9, 0.8, features)
        assert signal is None

        # Extremes signal - with history (lines 734-744)
        actor._prediction_history = list(range(100))
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 95.0, 0.8, features)
        assert signal is not None

        # Bottom extreme
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 5.0, 0.8, features)
        assert signal is not None

        # Not extreme
        signal = MLSignalActor._generate_extremes_signal(actor, bar, 50.0, 0.8, features)
        assert signal is None

        # Momentum signal - insufficient history (lines 756-758)
        actor._prediction_history = []
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

        # Momentum signal - with trend (lines 765-775)
        actor._prediction_history = [0.2, 0.4, 0.6, 0.8]
        signal = MLSignalActor._generate_momentum_signal(actor, bar, 0.9, 0.8, features)
        assert signal is not None

    def test_ensemble_adaptive_signals(self):
        """
        Test ensemble and adaptive signals (lines 788-821, 834-851).
        """
        actor = Mock()
        actor._config = Mock()
        actor._config.prediction_threshold = 0.7
        actor._config.log_predictions = False
        actor._ensemble_weights = {"threshold": 0.4, "extremes": 0.3, "momentum": 0.3}
        actor._adaptive_threshold = 0.5
        actor._market_regime = "trending"
        actor._bars_processed = 20
        actor._last_signal_bar = 0
        actor.clock = Mock()
        actor.clock.timestamp_ns = Mock(return_value=2000)

        bar = Mock()
        bar.bar_type = Mock()
        bar.bar_type.instrument_id = "EURUSD"
        bar.ts_event = 1000
        features = None

        # Ensemble - all signals (lines 788-821)
        signal1 = Mock(confidence=0.9)
        signal2 = Mock(confidence=0.8)
        signal3 = Mock(confidence=0.7)

        actor._generate_threshold_signal = Mock(return_value=signal1)
        actor._generate_extremes_signal = Mock(return_value=signal2)
        actor._generate_momentum_signal = Mock(return_value=signal3)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is not None
        assert signal.confidence == 0.81  # Weighted average

        # Ensemble - no signals (lines 811-813)
        actor._generate_threshold_signal = Mock(return_value=None)
        actor._generate_extremes_signal = Mock(return_value=None)
        actor._generate_momentum_signal = Mock(return_value=None)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is None

        # Ensemble - below threshold (lines 815-816)
        low_signal = Mock(confidence=0.5)
        actor._generate_threshold_signal = Mock(return_value=low_signal)

        signal = MLSignalActor._generate_ensemble_signal(actor, bar, 0.8, 0.85, features)
        assert signal is None

        # Adaptive signal - below threshold (lines 834-835)
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.4, 0.9, features)
        assert signal is None

        # Adaptive signal - zero threshold (lines 837-839)
        actor._adaptive_threshold = 0.0
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert signal is None

        # Adaptive signal - valid (lines 840-851)
        actor._adaptive_threshold = 0.5
        signal = MLSignalActor._generate_adaptive_signal(actor, bar, 0.8, 0.9, features)
        assert signal is not None

    def test_performance_metrics_initial(self):
        """
        Test performance metrics initialization (lines 878, 891).
        """
        actor = Mock()
        actor._prediction_count = 0
        actor._total_inference_time = 0.0
        actor._config = Mock()
        actor._config.log_predictions = False
        actor._signal_config = Mock()
        actor._signal_config.signal_strategy = SignalStrategy.THRESHOLD
        actor.id = Mock()
        actor.id.value = "TEST"
        actor.log = Mock()

        # Test first call creates metrics (line 878)
        with patch("prometheus_client.Histogram") as mock_histogram:
            mock_histogram.return_value = Mock()

            MLSignalActor._track_performance_metrics(actor, 0.75, 0.85, 3.5)

            assert hasattr(actor, "_prediction_distribution_metric")
            assert hasattr(actor, "_confidence_distribution_metric")

        # Test with existing metrics and averaging (line 891)
        actor._prediction_count = 10
        actor._total_inference_time = 50.0
        actor._prediction_distribution_metric = Mock()
        actor._confidence_distribution_metric = Mock()

        MLSignalActor._track_performance_metrics(actor, 0.8, 0.9, 5.0)

        assert actor._prediction_count == 11
        assert actor._total_inference_time == 55.0
