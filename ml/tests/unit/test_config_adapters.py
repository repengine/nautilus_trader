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
Tests for ML configuration adapters.

Tests the adapter pattern that bridges msgspec configs with Nautilus Cython components.

"""

from __future__ import annotations

import pytest

from ml.config.adapters import ConfigurationHelper
from ml.config.adapters import create_actor_config
from ml.config.base import MLActorConfig
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.identifiers import InstrumentId


class TestCreateActorConfig:
    """
    Tests for create_actor_config function.
    """

    def test_create_actor_config_from_ml_config(self) -> None:
        """
        Test creating an ActorConfig from an ML configuration.
        """
        # Arrange
        ml_config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            component_id=ComponentId("MLActor-001"),
            log_events=False,
            log_commands=True,
        )

        # Act
        actor_config = create_actor_config(ml_config)

        # Assert
        assert isinstance(actor_config, ActorConfig)
        assert actor_config.component_id == ComponentId("MLActor-001")
        assert actor_config.log_events is False
        assert actor_config.log_commands is True

    def test_create_actor_config_with_defaults(self) -> None:
        """
        Test creating an ActorConfig with default values.
        """
        # Arrange
        ml_config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
        )

        # Act
        actor_config = create_actor_config(ml_config)

        # Assert
        assert isinstance(actor_config, ActorConfig)
        assert actor_config.component_id is None
        assert actor_config.log_events is True
        assert actor_config.log_commands is True

    def test_create_actor_config_from_object_without_attributes(self) -> None:
        """
        Test creating an ActorConfig from object without expected attributes.
        """

        # Arrange
        class MinimalConfig:
            pass

        config = MinimalConfig()

        # Act
        actor_config = create_actor_config(config)

        # Assert
        assert isinstance(actor_config, ActorConfig)
        assert actor_config.component_id is None
        assert actor_config.log_events is True
        assert actor_config.log_commands is True


class TestConfigurationHelper:
    """
    Tests for ConfigurationHelper utility class.
    """

    def test_get_bar_type_from_ml_config(self) -> None:
        """
        Test extracting BarType from ML config.
        """
        # Arrange
        bar_type = BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL")
        ml_config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=bar_type,
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
        )

        # Act
        result = ConfigurationHelper.get_bar_type(ml_config)

        # Assert
        assert result == bar_type

    def test_get_bar_type_missing_raises_error(self) -> None:
        """
        Test that accessing missing bar_type raises AttributeError.
        """
        # Arrange
        config = ActorConfig()  # Doesn't have bar_type

        # Act & Assert
        with pytest.raises(AttributeError, match="No bar_type found"):
            ConfigurationHelper.get_bar_type(config)

    def test_get_instrument_id_from_config(self) -> None:
        """
        Test extracting InstrumentId from configuration.
        """
        # Arrange
        instrument_id = InstrumentId.from_str("EURUSD.IDEALPRO")
        ml_config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=instrument_id,
        )

        # Act
        result = ConfigurationHelper.get_instrument_id(ml_config)

        # Assert
        assert result == instrument_id

    def test_get_instrument_id_missing_raises_error(self) -> None:
        """
        Test that accessing missing instrument_id raises AttributeError.
        """
        # Arrange
        config = ActorConfig()  # Doesn't have instrument_id

        # Act & Assert
        with pytest.raises(AttributeError, match="No instrument_id found"):
            ConfigurationHelper.get_instrument_id(config)

    def test_get_model_path_from_config(self) -> None:
        """
        Test extracting model path from configuration.
        """
        # Arrange
        model_path = "/path/to/model.pkl"
        ml_config = MLActorConfig(
            model_path=model_path,
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
        )

        # Act
        result = ConfigurationHelper.get_model_path(ml_config)

        # Assert
        assert result == model_path

    def test_get_model_path_missing_raises_error(self) -> None:
        """
        Test that accessing missing model_path raises AttributeError.
        """
        # Arrange
        config = ActorConfig()  # Doesn't have model_path

        # Act & Assert
        with pytest.raises(AttributeError, match="No model_path found"):
            ConfigurationHelper.get_model_path(config)
