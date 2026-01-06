"""
Helpers to attach message bus publishers to the integration manager.

This module keeps the `ml.core.integration` import path stable while providing
explicit helpers that callers can opt into.

"""

from __future__ import annotations

from ml.common.message_bus import publisher_from_config
from ml.config.bus import MessageBusConfig
from ml.core.integration import MLIntegrationManager


def attach_publisher_from_env(manager: MLIntegrationManager) -> None:
    """
    Attach a message publisher to `manager` based on environment flags.

    Safe to call regardless of whether publishing is enabled; when disabled, attaches a
    NoopPublisher implementation.

    """
    cfg = MessageBusConfig.from_env()
    manager.set_message_publisher(publisher_from_config(cfg))


__all__ = ["attach_publisher_from_env"]
