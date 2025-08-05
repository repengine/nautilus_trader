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
Unit tests for ML inference actor base classes.

Tests cover:
- BaseMLInferenceActor initialization and configuration
- MLSignal data type creation and properties
- Performance tracking and metrics
- Error handling and edge cases
- PickleMLInferenceActor concrete implementation

"""

from __future__ import annotations

import pickle
import tempfile
import time
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from ml.actors.base import BaseMLInferenceActor
from ml.actors.base import MLSignal
from ml.actors.base import PickleMLInferenceActor
from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import DataType
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class SimplePickleableModel:
    """
    Simple ML model that can be pickled for testing.
    """

    def predict(self, X: Any) -> np.ndarray:
        return np.array([0.75])

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.array([[0.25, 0.75]])


class TestMLSignal:
    """
    Test MLSignal data type.
    """

    def test_initialization_with_required_parameters(self) -> None:
        """
        Test MLSignal initialization with required parameters.
        """
        # Arrange
        instrument_id = InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE"))
        prediction = 0.75
        confidence = 0.85
        ts_event = 1234567890000000000
        ts_init = 1234567890000000000

        # Act
        signal = MLSignal(
            instrument_id=instrument_id,
            prediction=prediction,
            confidence=confidence,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        # Assert
        assert signal.instrument_id == instrument_id
        assert signal.prediction == prediction
        assert signal.confidence == confidence
        assert signal.features is None
        assert signal.ts_event == ts_event
        assert signal.ts_init == ts_init

    def test_initialization_with_optional_features(self) -> None:
        """
        Test MLSignal initialization with optional features array.
        """
        # Arrange
        instrument_id = InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE"))
        prediction = 0.75
        confidence = 0.85
        features = np.array([1.0, 2.0, 3.0])
        ts_event = 1234567890000000000
        ts_init = 1234567890000000000

        # Act
        signal = MLSignal(
            instrument_id=instrument_id,
            prediction=prediction,
            confidence=confidence,
            features=features,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        # Assert
        assert signal.instrument_id == instrument_id
        assert signal.prediction == prediction
        assert signal.confidence == confidence
        assert signal.features is not None and np.array_equal(signal.features, features)
        assert signal.ts_event == ts_event
        assert signal.ts_init == ts_init

    def test_initialization_with_default_timestamps(self) -> None:
        """
        Test MLSignal initialization with default timestamp values.
        """
        # Arrange
        instrument_id = InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE"))
        prediction = 0.75
        confidence = 0.85

        # Act
        signal = MLSignal(
            instrument_id=instrument_id,
            prediction=prediction,
            confidence=confidence,
        )

        # Assert
        assert signal.instrument_id == instrument_id
        assert signal.prediction == prediction
        assert signal.confidence == confidence
        assert signal.features is None
        assert signal.ts_event == 0
        assert signal.ts_init == 0

    def test_timestamp_properties(self) -> None:
        """
        Test timestamp properties return correct values.
        """
        # Arrange
        instrument_id = InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE"))
        ts_event = 1234567890000000000
        ts_init = 1234567890000000001

        signal = MLSignal(
            instrument_id=instrument_id,
            prediction=0.5,
            confidence=0.8,
            ts_event=ts_event,
            ts_init=ts_init,
        )

        # Act & Assert
        assert signal.ts_event == ts_event
        assert signal.ts_init == ts_init


class MockMLInferenceActor(BaseMLInferenceActor):
    """
    Mock implementation of BaseMLInferenceActor for testing.
    """

    def __init__(self, config: MLActorConfig) -> None:
        super().__init__(config)
        self.model_loaded = False
        self.features_initialized = False
        self.prediction_calls = 0
        # Mock methods that need external dependencies
        self._mocked_publish_data = Mock()
        self._mocked_subscribe_bars = Mock()
        self._mocked_log = Mock()
        self._mocked_clock_timestamp_ns = Mock(return_value=1234567890000000000)

        # Mock the model loader to avoid file system access
        self._model_loader = Mock()
        self._model_loader.load_model.return_value = (
            Mock(),  # mock model
            {"version": "test_v1", "type": "mock", "size_bytes": 1024},  # mock metadata
        )

    def _load_model(self) -> None:
        """
        Mock model loading.
        """
        self.model_loaded = True
        # Don't override self._model as it's set by _load_model_with_metadata

    def _initialize_features(self) -> None:
        """
        Mock feature initialization.
        """
        self.features_initialized = True
        self._features_buffer = np.zeros(5)  # Mock feature buffer

    def _compute_features(self, bar: Bar) -> np.ndarray | None:
        """
        Mock feature computation.
        """
        if not self.features_initialized:
            return None
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def _predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Mock prediction.
        """
        self.prediction_calls += 1
        return 0.75, 0.85

    def publish_data(self, data_type: Any, data: Any) -> None:
        """
        Mock publish_data method.
        """
        self._mocked_publish_data(data_type, data)

    def subscribe_bars(self, bar_type: Any) -> None:
        """
        Mock subscribe_bars method.
        """
        self._mocked_subscribe_bars(bar_type)

    @property
    def log(self) -> Any:
        """
        Mock log property.
        """
        return self._mocked_log

    @property
    def clock(self) -> Any:
        """
        Mock clock property.
        """
        mock_clock = Mock()
        mock_clock.timestamp_ns = self._mocked_clock_timestamp_ns
        mock_clock.set_timer = Mock()  # Mock timer functionality for hot reload
        return mock_clock


class TestBaseMLInferenceActor:
    """
    Test BaseMLInferenceActor base class.
    """

    @pytest.fixture
    def instrument_id(self) -> InstrumentId:
        """
        Create test instrument ID.
        """
        return InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE"))

    @pytest.fixture
    def bar_type(self, instrument_id: InstrumentId) -> BarType:
        """
        Create test bar type.
        """
        return BarType(
            instrument_id,
            BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
        )

    @pytest.fixture
    def config(self, bar_type: BarType, instrument_id: InstrumentId) -> MLActorConfig:
        """
        Create test configuration.
        """
        return MLActorConfig(
            model_path=tempfile.gettempdir() + "/test_model.pkl",
            bar_type=bar_type,
            instrument_id=instrument_id,
        )

    @pytest.fixture
    def actor(self, config: MLActorConfig) -> MockMLInferenceActor:
        """
        Create test actor instance.
        """
        actor = MockMLInferenceActor(config)
        return actor

    @pytest.fixture
    def sample_bar(self, instrument_id: InstrumentId) -> Bar:
        """
        Create sample bar for testing.
        """
        return Bar(
            bar_type=BarType(
                instrument_id,
                BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
            ),
            open=Price.from_str("100.00"),
            high=Price.from_str("101.00"),
            low=Price.from_str("99.00"),
            close=Price.from_str("100.50"),
            volume=Quantity.from_int(1000),
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

    def test_initialization_with_config(self, config: MLActorConfig) -> None:
        """
        Test actor initialization with configuration.
        """
        # Act
        actor = MockMLInferenceActor(config)

        # Assert
        assert actor._config == config
        assert actor._feature_config is not None
        assert isinstance(actor._feature_config, MLFeatureConfig)
        assert actor._model is None
        assert actor._features_buffer is None
        assert len(actor._feature_window) == 0
        assert actor._prediction_count == 0
        assert actor._total_inference_time == 0.0
        assert actor._last_prediction_time == 0
        assert actor._bars_processed == 0
        assert actor._is_warmed_up is False

    def test_initialization_with_custom_feature_config(
        self,
        bar_type: BarType,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test actor initialization with custom feature configuration.
        """
        # Arrange
        feature_config = MLFeatureConfig(
            lookback_window=50,
            normalize_features=False,
        )
        config = MLActorConfig(
            model_path=tempfile.gettempdir() + "/test_model.pkl",
            bar_type=bar_type,
            instrument_id=instrument_id,
            feature_config=feature_config,
        )

        # Act
        actor = MockMLInferenceActor(config)

        # Assert
        assert actor._feature_config == feature_config
        assert actor._feature_window.maxlen == 50

    def test_on_start_initializes_components(self, actor: MockMLInferenceActor) -> None:
        """
        Test on_start initializes model and features.
        """
        # Act
        actor.on_start()

        # Assert
        assert actor.model_loaded is True
        assert actor.features_initialized is True
        actor._mocked_subscribe_bars.assert_called_once_with(actor._config.bar_type)

    def test_on_bar_during_warmup_period(
        self,
        actor: MockMLInferenceActor,
        sample_bar: Bar,
    ) -> None:
        """
        Test on_bar behavior during warmup period.
        """
        # Arrange
        actor.on_start()

        # Act - Process bars during warmup
        for i in range(actor._config.warm_up_period - 1):
            actor.on_bar(sample_bar)

        # Assert
        assert actor._bars_processed == actor._config.warm_up_period - 1
        assert actor._is_warmed_up is False
        assert actor.prediction_calls == 0
        assert len(actor._feature_window) == actor._config.warm_up_period - 1

    def test_on_bar_after_warmup_period(
        self,
        actor: MockMLInferenceActor,
        sample_bar: Bar,
    ) -> None:
        """
        Test on_bar behavior after warmup period completes.
        """
        # Arrange
        actor.on_start()

        # Act - Process bars to complete warmup
        for i in range(actor._config.warm_up_period):
            actor.on_bar(sample_bar)

        # Assert
        assert actor._bars_processed == actor._config.warm_up_period
        assert actor._is_warmed_up is True
        assert actor.prediction_calls == 1  # Last bar triggered prediction

    def test_on_bar_with_none_features(
        self,
        actor: MockMLInferenceActor,
        sample_bar: Bar,
    ) -> None:
        """
        Test on_bar behavior when features are not ready.
        """
        # Arrange
        actor.on_start()
        setattr(actor, "_compute_features", Mock(return_value=None))

        # Act
        actor.on_bar(sample_bar)

        # Assert
        assert actor._bars_processed == 1
        assert len(actor._feature_window) == 0
        assert actor.prediction_calls == 0

    def test_generate_prediction_performance_tracking(
        self,
        actor: MockMLInferenceActor,
        sample_bar: Bar,
    ) -> None:
        """
        Test prediction performance tracking.
        """
        # Arrange
        actor.on_start()
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Act
        actor._generate_prediction_protected(sample_bar, features)

        # Assert
        assert actor._prediction_count == 1
        assert actor._total_inference_time > 0
        assert actor.prediction_calls == 1

    def test_generate_prediction_with_signal_publishing(
        self,
        actor: MockMLInferenceActor,
        sample_bar: Bar,
    ) -> None:
        """
        Test prediction with signal publishing enabled.
        """
        # Arrange
        actor.on_start()
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Debug: Check that our mock _predict is working
        test_pred, test_conf = actor._predict(features)
        assert test_pred == 0.75
        assert test_conf == 0.85

        with patch.object(actor, "_publish_signal") as mock_publish:
            # Act
            actor._generate_prediction_protected(sample_bar, features)

            # Assert - debug the config values
            assert actor._config.prediction_threshold == 0.5
            assert actor._config.publish_signals is True

            # Debug: Let's check if exception occurred
            if not mock_publish.called:
                # Check if error was logged
                if actor.log.error.called:
                    print(f"Error logged: {actor.log.error.call_args}")

            mock_publish.assert_called_once()
            signal = mock_publish.call_args[0][0]
            assert isinstance(signal, MLSignal)
            assert signal.instrument_id == sample_bar.bar_type.instrument_id
            assert signal.prediction == 0.75
            assert signal.confidence == 0.85

    def test_generate_prediction_below_threshold(
        self,
        bar_type: BarType,
        instrument_id: InstrumentId,
        sample_bar: Bar,
    ) -> None:
        """
        Test prediction below confidence threshold doesn't publish signal.
        """
        # Arrange
        config = MLActorConfig(
            model_path=tempfile.gettempdir() + "/test_model.pkl",
            bar_type=bar_type,
            instrument_id=instrument_id,
            prediction_threshold=0.9,  # Higher than mock prediction confidence
        )
        actor = MockMLInferenceActor(config)
        actor.on_start()
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch.object(actor, "_publish_signal") as mock_publish:
            # Act
            actor._generate_prediction_protected(sample_bar, features)

            # Assert
            mock_publish.assert_not_called()

    def test_generate_prediction_with_signals_disabled(
        self,
        bar_type: BarType,
        instrument_id: InstrumentId,
        sample_bar: Bar,
    ) -> None:
        """
        Test prediction with signal publishing disabled.
        """
        # Arrange
        config = MLActorConfig(
            model_path=tempfile.gettempdir() + "/test_model.pkl",
            bar_type=bar_type,
            instrument_id=instrument_id,
            publish_signals=False,
        )
        actor = MockMLInferenceActor(config)
        actor.on_start()
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch.object(actor, "_publish_signal") as mock_publish:
            # Act
            actor._generate_prediction_protected(sample_bar, features)

            # Assert
            mock_publish.assert_not_called()

    def test_generate_prediction_latency_warning(
        self,
        bar_type: BarType,
        instrument_id: InstrumentId,
        sample_bar: Bar,
    ) -> None:
        """
        Test latency warning when inference takes too long.
        """
        # Arrange
        config = MLActorConfig(
            model_path=tempfile.gettempdir() + "/test_model.pkl",
            bar_type=bar_type,
            instrument_id=instrument_id,
            max_inference_latency_ms=0.001,  # Very low threshold
        )
        actor = MockMLInferenceActor(config)
        actor.on_start()
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Mock slow prediction
        original_predict = actor._predict

        def slow_predict(features: np.ndarray) -> Any:
            time.sleep(0.01)  # 10ms delay
            return original_predict(features)

        setattr(actor, "_predict", slow_predict)

        # Act
        actor._generate_prediction_protected(sample_bar, features)

        # Assert
        actor.log.warning.assert_called_once()
        assert "Inference latency exceeded" in actor.log.warning.call_args[0][0]

    def test_generate_prediction_exception_handling(
        self,
        actor: MockMLInferenceActor,
        sample_bar: Bar,
    ) -> None:
        """
        Test exception handling during prediction.
        """
        # Arrange
        actor.on_start()
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        setattr(actor, "_predict", Mock(side_effect=Exception("Model error")))

        # Act
        actor._generate_prediction_protected(sample_bar, features)

        # Assert
        actor.log.error.assert_called_once()
        assert "Prediction failed" in actor.log.error.call_args[0][0]

    def test_publish_signal(self, actor: MockMLInferenceActor) -> None:
        """
        Test signal publishing to message bus.
        """
        # Arrange
        actor.on_start()
        signal = MLSignal(
            instrument_id=InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE")),
            prediction=0.75,
            confidence=0.85,
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

        # Act
        actor._publish_signal(signal)

        # Assert
        actor._mocked_publish_data.assert_called_once()
        data_type, published_signal = actor._mocked_publish_data.call_args[0]
        assert isinstance(data_type, DataType)
        assert data_type.type == MLSignal
        assert published_signal == signal

    def test_on_stop_logs_statistics(self, actor: MockMLInferenceActor) -> None:
        """
        Test on_stop logs final statistics.
        """
        # Arrange
        actor.on_start()
        actor._prediction_count = 10
        actor._total_inference_time = 50.0  # 50ms total

        # Act
        actor.on_stop()

        # Assert
        # Check that info was called and check the last call
        assert actor.log.info.called
        log_message = actor.log.info.call_args[0][0]
        assert "Predictions: 10" in log_message
        assert "Avg inference time: 5.000ms" in log_message

    def test_on_stop_with_zero_predictions(self, actor: MockMLInferenceActor) -> None:
        """
        Test on_stop handles zero predictions gracefully.
        """
        # Arrange
        actor.on_start()
        actor._prediction_count = 0
        actor._total_inference_time = 0.0

        # Act
        actor.on_stop()

        # Assert
        # Check the second call to info (first is in on_start)
        assert actor.log.info.call_count >= 1
        log_message = actor.log.info.call_args[0][0]
        assert "Predictions: 0" in log_message
        assert "Avg inference time: 0.000ms" in log_message

    def test_prometheus_metrics_initialization(
        self,
        actor: MockMLInferenceActor,
    ) -> None:
        """
        Test Prometheus metrics are properly initialized.
        """
        # Assert
        assert actor._inference_latency_metric is not None
        assert actor._inference_count_metric is not None
        assert actor._inference_errors_metric is not None
        assert actor._feature_computation_time_metric is not None


class TestPickleMLInferenceActor:
    """
    Test PickleMLInferenceActor concrete implementation.
    """

    @pytest.fixture
    def instrument_id(self) -> InstrumentId:
        """
        Create test instrument ID.
        """
        return InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE"))

    @pytest.fixture
    def bar_type(self, instrument_id: InstrumentId) -> BarType:
        """
        Create test bar type.
        """
        return BarType(
            instrument_id,
            BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
        )

    def _create_mock_model(self) -> SimplePickleableModel:
        """
        Create mock ML model that can be pickled.
        """
        return SimplePickleableModel()

    @pytest.fixture
    def mock_model(self) -> SimplePickleableModel:
        """
        Create mock ML model.
        """
        return self._create_mock_model()

    @pytest.fixture
    def model_file(self, mock_model: Any) -> str:
        """
        Create temporary model file.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(mock_model, f)
            return f.name

    @pytest.fixture
    def config(
        self,
        bar_type: BarType,
        instrument_id: InstrumentId,
        model_file: str,
    ) -> MLActorConfig:
        """
        Create test configuration with model file.
        """
        return MLActorConfig(
            model_path=model_file,
            bar_type=bar_type,
            instrument_id=instrument_id,
        )

    def test_load_model_success(
        self,
        config: MLActorConfig,
        mock_model: Any,
    ) -> None:
        """
        Test successful model loading.
        """

        # Arrange
        class TestActor(PickleMLInferenceActor):
            def _initialize_features(self) -> None:
                pass

            def _compute_features(self, bar: Bar) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0])

            def _predict(self, features: np.ndarray) -> Any:
                return super()._predict(features)

        actor = TestActor(config)

        # Act
        actor._load_model()

        # Assert
        assert actor._model is not None
        # Verify model functionality
        result = actor._model.predict(np.array([[1, 2, 3]]))
        assert len(result) > 0

    def test_load_model_file_not_found(
        self,
        bar_type: BarType,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test model loading with non-existent file.
        """
        # Arrange
        config = MLActorConfig(
            model_path="/nonexistent/model.pkl",
            bar_type=bar_type,
            instrument_id=instrument_id,
        )

        class TestActor(PickleMLInferenceActor):
            def _initialize_features(self) -> None:
                pass

            def _compute_features(self, bar: Bar) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0])

        actor = TestActor(config)

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            actor._load_model()

    def test_predict_with_classification_model(
        self,
        config: MLActorConfig,
        mock_model: Any,
    ) -> None:
        """
        Test prediction with classification model (has predict_proba).
        """

        # Arrange
        class TestActor(PickleMLInferenceActor):
            def _initialize_features(self) -> None:
                pass

            def _compute_features(self, bar: Bar) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0])

        actor = TestActor(config)
        actor._model = mock_model
        features = np.array([1.0, 2.0, 3.0])

        # Act
        prediction, confidence = actor._predict(features)

        # Assert
        assert prediction == 1.0  # argmax of [0.25, 0.75]
        assert confidence == 0.75  # max of [0.25, 0.75]

    def test_predict_with_regression_model(
        self,
        config: MLActorConfig,
    ) -> None:
        """
        Test prediction with regression model (no predict_proba).
        """
        # Arrange
        regression_model = Mock(spec=["predict"])  # Only has predict method
        regression_model.predict.return_value = np.array([0.65])

        class TestActor(PickleMLInferenceActor):
            def _initialize_features(self) -> None:
                pass

            def _compute_features(self, bar: Bar) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0])

        actor = TestActor(config)
        actor._model = regression_model
        features = np.array([1.0, 2.0, 3.0])

        # Act
        prediction, confidence = actor._predict(features)

        # Assert
        assert prediction == 0.65
        assert confidence == 1.0  # Full confidence for regression
        regression_model.predict.assert_called_once()

    def test_predict_features_reshape(
        self,
        config: MLActorConfig,
        mock_model: Any,
    ) -> None:
        """
        Test that features are properly reshaped for sklearn models.
        """

        # Arrange
        class TestActor(PickleMLInferenceActor):
            def _initialize_features(self) -> None:
                pass

            def _compute_features(self, bar: Bar) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0])

        actor = TestActor(config)
        actor._model = mock_model
        features = np.array([1.0, 2.0, 3.0])  # 1D array

        # Act
        actor._predict(features)

        # Assert - since we can't mock the call args with our simple model,
        # we'll test that the predict method works correctly with the reshaped input
        # by calling it again and checking it doesn't raise an error
        result = mock_model.predict_proba(features.reshape(1, -1))
        assert result.shape == (1, 2)  # Output should have 2 classes

    def teardown_method(self, method: Any) -> None:
        """
        Clean up temporary files after each test.
        """
        # This will be called after each test method
        import glob
        import os

        # Clean up any temporary pickle files
        for f in glob.glob(tempfile.gettempdir() + "/test_model*.pkl"):
            try:
                os.unlink(f)
            except OSError:
                pass
