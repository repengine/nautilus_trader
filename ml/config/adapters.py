"""
Configuration utilities for ML actors.

This module provides helper functions for working with ML configurations in the context
of Nautilus Trader's actor system.

"""

from __future__ import annotations

from typing import Any, Protocol

from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId

from nautilus_trader.common.config import ActorConfig
from nautilus_trader.common.config import NautilusConfig


def create_actor_config(ml_config: NautilusConfig) -> ActorConfig:
    """
    Create a standard ActorConfig from an ML configuration.

    Parameters
    ----------
    ml_config : NautilusConfig
        The ML configuration containing actor-related fields.

    Returns
    -------
    ActorConfig
        A standard actor configuration suitable for Cython components.

    """
    return ActorConfig(
        component_id=getattr(ml_config, "component_id", None),
        log_events=getattr(ml_config, "log_events", True),
        log_commands=getattr(ml_config, "log_commands", True),
    )


class _ConfigWithBarType(Protocol):
    bar_type: BarType


class _ConfigWithInstrument(Protocol):
    instrument_id: InstrumentId


class _ConfigWithModelPath(Protocol):
    model_path: str | Any


class ConfigurationHelper:
    """
    Helper class for working with ML configurations in actors.

    Provides utility methods for extracting specific configuration values from ML
    configurations.

    """

    @staticmethod
    def get_bar_type(config: _ConfigWithBarType) -> BarType:
        """
        Extract BarType from configuration.

        Parameters
        ----------
        config : Any
            The configuration object.

        Returns
        -------
        BarType
            The bar type from the configuration.

        """
        return config.bar_type

    @staticmethod
    def get_instrument_id(config: _ConfigWithInstrument) -> InstrumentId:
        """
        Extract InstrumentId from configuration.

        Parameters
        ----------
        config : Any
            The configuration object.

        Returns
        -------
        InstrumentId
            The instrument ID from the configuration.

        """
        return config.instrument_id

    @staticmethod
    def get_model_path(config: _ConfigWithModelPath) -> str:
        """
        Extract model path from configuration.

        Parameters
        ----------
        config : Any
            The configuration object.

        Returns
        -------
        str
            The model path from the configuration.

        """
        return str(config.model_path)
