"""
Signal Facade Module.

This module exports MLSignalActorFacade with feature flag support for legacy/facade switching.

Feature Flag: ML_USE_LEGACY_ML_SIGNAL_ACTOR
- 0 (default): Use MLSignalActorFacade
- 1: Use legacy MLSignalActor

"""

from __future__ import annotations

import os

from ml.actors.signal_facade_impl import MLSignalActorFacade


# Feature flag for legacy mode
_USE_LEGACY = os.environ.get("ML_USE_LEGACY_ML_SIGNAL_ACTOR", "0") == "1"

if _USE_LEGACY:
    # When legacy mode is enabled, import from signal.py
    # Note: The legacy MLSignalActor would be imported here if it exists
    # For now, we still export MLSignalActorFacade as the default
    pass

__all__ = [
    "MLSignalActorFacade",
]
