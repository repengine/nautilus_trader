"""
Configuration utilities for ML actors.

This module provides helper functions for working with ML configurations in the context
of Nautilus Trader's actor system.

"""

from __future__ import annotations

from typing import Any

from nautilus_trader.common.config import ActorConfig
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId


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


class ConfigurationHelper:
    """
    Helper class for working with ML configurations in actors.

    Provides utility methods for extracting specific configuration values from ML
    configurations.

    """

    @staticmethod
    def get_bar_type(config: Any) -> BarType:
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
        if hasattr(config, "bar_type"):
            return config.bar_type

        raise AttributeError(f"No bar_type found in configuration of type {type(config).__name__}")

    @staticmethod
    def get_instrument_id(config: Any) -> InstrumentId:
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
        if hasattr(config, "instrument_id"):
            return config.instrument_id

        raise AttributeError(
            f"No instrument_id found in configuration of type {type(config).__name__}",
        )

    @staticmethod
    def get_model_path(config: Any) -> str:
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
        if hasattr(config, "model_path"):
            return str(config.model_path)

        raise AttributeError(
            f"No model_path found in configuration of type {type(config).__name__}",
        )
