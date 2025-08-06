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
Comprehensive tests for the ML configuration solution.

Tests that verify the complete solution for bridging msgspec configs with Cython
components.

"""

from __future__ import annotations

from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from ml.examples.simple_ml_actor import SimpleMLActor
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class TestMLConfigurationSolution:
    """
    Comprehensive tests for ML configuration solution.
    """

    def test_ml_actor_with_full_configuration(self):
        """
        Test ML actor with complete configuration including all features.
        """
        # Arrange
        feature_config = MLFeatureConfig(
            lookback_window=50,
            normalize_features=True,
            fill_missing_with=0.0,
        )

        config = MLActorConfig(
            model_path="test_model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            component_id=ComponentId("MLActor-TEST"),
            prediction_threshold=0.65,
            max_inference_latency_ms=2.0,
            feature_config=feature_config,
            warm_up_period=20,
            publish_signals=True,
            log_predictions=True,
            enable_hot_reload=False,
            enable_health_monitoring=True,
            log_events=False,
            log_commands=True,
        )

        # Act
        actor = SimpleMLActor(config)

        # Assert
        assert actor._config == config
        assert actor._config.component_id == ComponentId("MLActor-TEST")
        assert actor._config.prediction_threshold == 0.65
        assert actor._config.feature_config.lookback_window == 50
        assert actor._config.log_events is False
        assert actor._config.log_commands is True

    def test_ml_actor_config_compatibility(self):
        """
        Test ML actor configuration is compatible with Nautilus base Actor.
        """
        # Arrange
        config = MLActorConfig(
            model_path="test_model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            component_id=ComponentId("MLActor-002"),
            warm_up_period=5,
        )

        # Act
        actor = SimpleMLActor(config)

        # Assert - Verify actor was created with correct base configuration
        assert hasattr(actor, "_config")
        assert actor._config == config
        assert isinstance(actor, SimpleMLActor)

        # The actor should have access to all ML config fields
        assert actor._config.model_path == "test_model.pkl"
        assert actor._config.warm_up_period == 5
        assert actor._config.component_id == ComponentId("MLActor-002")

    def test_ml_actor_processes_bars_with_features(self):
        """
        Test that ML actor properly processes bars and computes features.
        """
        # Arrange
        bar_type = BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL")
        config = MLActorConfig(
            model_path="test_model.pkl",
            bar_type=bar_type,
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            warm_up_period=5,
            log_predictions=True,
        )

        actor = SimpleMLActor(config)

        # Initialize the actor's features (normally done in on_start)
        actor._initialize_features()
        actor._load_model()
        actor._is_warmed_up = False
        actor._bars_processed = 0

        # Create test bars
        bars = []
        for i in range(25):
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"1.{1000 + i:04d}"),
                high=Price.from_str(f"1.{1010 + i:04d}"),
                low=Price.from_str(f"1.{990 + i:04d}"),
                close=Price.from_str(f"1.{1005 + i:04d}"),
                volume=Quantity.from_int(1000000 + i * 1000),
                ts_event=i * 60_000_000_000,
                ts_init=i * 60_000_000_000,
            )
            bars.append(bar)

        # Act - Process bars
        for bar in bars:
            actor.on_bar(bar)

        # Assert
        assert actor._bars_processed == 25
        assert actor._is_warmed_up is True
        assert actor._prediction_count > 0  # Made predictions after warm-up

        # Verify feature computation worked
        assert actor._sma_fast.initialized
        assert actor._sma_slow.initialized
        assert actor._rsi.initialized
        assert actor._ema.initialized

    def test_ml_actor_configuration_serialization(self):
        """
        Test that ML configuration can be serialized and deserialized.
        """
        # Arrange
        config = MLActorConfig(
            model_path="test_model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            component_id=ComponentId("MLActor-001"),
            prediction_threshold=0.75,
            warm_up_period=30,
        )

        # Act
        json_bytes = config.json()
        json_dict = config.json_primitives()

        # Deserialize
        config_restored = MLActorConfig.parse(json_bytes)

        # Assert
        assert config_restored.model_path == config.model_path
        assert config_restored.prediction_threshold == config.prediction_threshold
        assert config_restored.warm_up_period == config.warm_up_period
        assert config_restored.component_id == config.component_id
        assert "model_path" in json_dict
        assert json_dict["prediction_threshold"] == 0.75

    def test_ml_actor_with_invalid_model_path_uses_dummy(self):
        """
        Test that ML actor handles missing model file gracefully.
        """
        # Arrange
        config = MLActorConfig(
            model_path="/nonexistent/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
        )

        # Act
        actor = SimpleMLActor(config)

        # Initialize components (normally done in on_start)
        actor._initialize_features()
        actor._load_model()

        # Assert - should use DummyModel
        assert actor._model is not None  # DummyModel loaded
        assert hasattr(actor._model, "predict")  # Has predict method
