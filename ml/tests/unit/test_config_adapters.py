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

from ml.config.adapters import ActorConfigWrapper
from ml.config.adapters import ConfigurationHelper
from ml.config.adapters import MLActorConfigBridge
from ml.config.adapters import create_actor_config_wrapper
from ml.config.base import MLActorConfig
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.identifiers import InstrumentId


class TestActorConfigWrapper:
    """
    Tests for ActorConfigWrapper class.
    """

    def test_wrapper_creation_with_ml_config(self):
        """
        Test creating a wrapper from an ML configuration.
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
        wrapper = ActorConfigWrapper(ml_config)
        actor_config = wrapper.actor_config

        # Assert
        assert isinstance(actor_config, ActorConfig)
        assert actor_config.component_id == ComponentId("MLActor-001")
        assert actor_config.log_events is False
        assert actor_config.log_commands is True
        assert hasattr(actor_config, "_ml_config")
        assert actor_config._ml_config == ml_config

    def test_wrapper_preserves_ml_config(self):
        """
        Test that wrapper preserves ML configuration.
        """
        # Arrange
        ml_config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            prediction_threshold=0.75,
            max_inference_latency_ms=3.0,
        )
        wrapper = ActorConfigWrapper(ml_config)

        # Act & Assert
        assert wrapper.ml_config == ml_config
        assert wrapper.ml_config.model_path == "/path/to/model.pkl"
        assert wrapper.ml_config.prediction_threshold == 0.75
        assert wrapper.ml_config.max_inference_latency_ms == 3.0

    def test_create_actor_config_wrapper_factory(self):
        """
        Test the factory function for creating wrapped configs.
        """
        # Arrange
        ml_config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
            component_id=ComponentId("TestActor"),
        )

        # Act
        actor_config = create_actor_config_wrapper(ml_config)

        # Assert
        assert isinstance(actor_config, ActorConfig)
        assert actor_config.component_id == ComponentId("TestActor")
        assert hasattr(actor_config, "_ml_config")
        assert actor_config._ml_config == ml_config


class TestMLActorConfigBridge:
    """
    Tests for MLActorConfigBridge class.
    """

    def test_adapt_ml_config(self):
        """
        Test adapting an ML configuration.
        """
        # Arrange
        ml_config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
        )

        # Act
        adapted = MLActorConfigBridge.adapt(ml_config)

        # Assert
        assert isinstance(adapted, ActorConfig)
        assert hasattr(adapted, "_ml_config")
        assert adapted._ml_config == ml_config

    def test_adapt_actor_config_returns_unchanged(self):
        """
        Test that adapting an ActorConfig returns it unchanged.
        """
        # Arrange
        config = ActorConfig(component_id=ComponentId("TestActor"))

        # Act
        adapted = MLActorConfigBridge.adapt(config)

        # Assert
        assert adapted is config

    def test_adapt_invalid_type_raises_error(self):
        """
        Test that adapting invalid type raises TypeError.
        """
        # Arrange
        invalid_config = {"model_path": "/path/to/model.pkl"}

        # Act & Assert
        with pytest.raises(TypeError, match="Cannot adapt configuration"):
            MLActorConfigBridge.adapt(invalid_config)

    def test_extract_ml_config_from_wrapped(self):
        """
        Test extracting ML config from a wrapped config.
        """
        # Arrange
        ml_config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL"),
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
        )
        actor_config = create_actor_config_wrapper(ml_config)

        # Act
        extracted = MLActorConfigBridge.extract_ml_config(actor_config)

        # Assert
        assert extracted == ml_config

    def test_extract_ml_config_from_non_adapter_raises_error(self):
        """
        Test extracting ML config from non-adapter raises error.
        """
        # Arrange
        config = ActorConfig()

        # Act & Assert
        with pytest.raises(ValueError, match="does not contain an ML configuration"):
            MLActorConfigBridge.extract_ml_config(config)


class TestConfigurationHelper:
    """
    Tests for ConfigurationHelper utility class.
    """

    def test_get_bar_type_from_ml_config(self):
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

    def test_get_bar_type_from_wrapped_config(self):
        """
        Test extracting BarType from wrapped config.
        """
        # Arrange
        bar_type = BarType.from_str("EURUSD.IDEALPRO-1-MINUTE-MID-EXTERNAL")
        ml_config = MLActorConfig(
            model_path="/path/to/model.pkl",
            bar_type=bar_type,
            instrument_id=InstrumentId.from_str("EURUSD.IDEALPRO"),
        )
        actor_config = create_actor_config_wrapper(ml_config)

        # Act
        result = ConfigurationHelper.get_bar_type(actor_config)

        # Assert
        assert result == bar_type

    def test_get_instrument_id_from_config(self):
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

    def test_get_model_path_from_config(self):
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

    def test_get_missing_attribute_raises_error(self):
        """
        Test that accessing missing attribute raises AttributeError.
        """
        # Arrange
        config = ActorConfig()  # Doesn't have bar_type

        # Act & Assert
        with pytest.raises(AttributeError, match="No bar_type found"):
            ConfigurationHelper.get_bar_type(config)
