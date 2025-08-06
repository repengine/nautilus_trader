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
Tests for ML actor configuration handling.

Tests that ML actors properly work with msgspec configurations and Cython components.

"""

from __future__ import annotations

import pytest

from ml.config.adapters import ConfigurationHelper
from ml.config.adapters import create_actor_config
from ml.config.base import MLActorConfig
from ml.examples.simple_ml_actor import SimpleMLActor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.identifiers import InstrumentId


class TestMLActorConfiguration:
    """
    Tests for ML actor configuration handling.
    """

    def test_ml_actor_config_creation(self):
        """
        Test creating an ML actor configuration.
        """
        # Arrange & Act
        config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            component_id=ComponentId("MLActor-001"),
            prediction_threshold=0.7,
            warm_up_period=30,
        )

        # Assert
        assert config.model_path == "/path/to/model.pkl"
        assert config.prediction_threshold == 0.7
        assert config.warm_up_period == 30
        assert config.component_id == ComponentId("MLActor-001")

    def test_create_actor_config_from_ml_config(self):
        """
        Test creating a standard ActorConfig from ML config.
        """
        # Arrange
        ml_config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            component_id=ComponentId("TestActor"),
            log_events=False,
            log_commands=True,
        )

        # Act
        actor_config = create_actor_config(ml_config)

        # Assert
        assert isinstance(actor_config, ActorConfig)
        assert actor_config.component_id == ComponentId("TestActor")
        assert actor_config.log_events is False
        assert actor_config.log_commands is True

    def test_simple_ml_actor_initialization(self):
        """
        Test that SimpleMLActor can be initialized with ML config.
        """
        # Arrange
        config = MLActorConfig(
            model_path="dummy_model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            prediction_threshold=0.65,
            warm_up_period=25,
            log_predictions=True,
        )

        # Act
        actor = SimpleMLActor(config)

        # Assert
        assert actor is not None
        assert actor._config == config
        assert actor._config.prediction_threshold == 0.65
        assert actor._config.warm_up_period == 25
        assert actor._config.log_predictions is True

    def test_configuration_helper_get_bar_type(self):
        """
        Test ConfigurationHelper.get_bar_type method.
        """
        # Arrange
        bar_type = BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL")
        config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=bar_type,
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
        )

        # Act
        result = ConfigurationHelper.get_bar_type(config)

        # Assert
        assert result == bar_type

    def test_configuration_helper_get_instrument_id(self):
        """
        Test ConfigurationHelper.get_instrument_id method.
        """
        # Arrange
        instrument_id = InstrumentId.from_str("EURUSD.IDEALPRO")
        config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=instrument_id,
        )

        # Act
        result = ConfigurationHelper.get_instrument_id(config)

        # Assert
        assert result == instrument_id

    def test_configuration_helper_get_model_path(self):
        """
        Test ConfigurationHelper.get_model_path method.
        """
        # Arrange
        model_path = "/path/to/model.pkl"
        config = MLActorConfig(
            model_path=model_path,
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
        )

        # Act
        result = ConfigurationHelper.get_model_path(config)

        # Assert
        assert result == model_path

    def test_configuration_helper_missing_attribute_raises(self):
        """
        Test that ConfigurationHelper raises for missing attributes.
        """
        # Arrange
        config = ActorConfig()  # Doesn't have ML fields

        # Act & Assert
        with pytest.raises(AttributeError, match="No bar_type found"):
            ConfigurationHelper.get_bar_type(config)

        with pytest.raises(AttributeError, match="No instrument_id found"):
            ConfigurationHelper.get_instrument_id(config)

        with pytest.raises(AttributeError, match="No model_path found"):
            ConfigurationHelper.get_model_path(config)
