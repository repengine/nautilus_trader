"""
Feature flags for ML module refactoring.

This module provides feature flags to control which implementation is used
during god class decomposition refactoring. Flags enable safe rollback by
allowing runtime switching between legacy and refactored implementations.

All flags default to using the NEW (facade/refactored) implementation.
Set environment variable to "1" to use legacy implementation.

Example:
    >>> import os
    >>> os.environ["ML_USE_LEGACY_FEATURE_ENGINEER"] = "1"
    >>> from ml.features import FeatureEngineer  # Uses legacy
    >>> # vs
    >>> os.environ["ML_USE_LEGACY_FEATURE_ENGINEER"] = "0"
    >>> from ml.features import FeatureEngineer  # Uses facade

"""

from __future__ import annotations

import os


def use_legacy_feature_engineer() -> bool:
    """
    Return True if using legacy FeatureEngineer, False for new facade.

    Controlled by environment variable ML_USE_LEGACY_FEATURE_ENGINEER.
    Default: False (use facade/refactored implementation).

    Returns:
        True if legacy mode enabled (env var = "1"), False otherwise.

    Example:
        >>> import os
        >>> os.environ["ML_USE_LEGACY_FEATURE_ENGINEER"] = "1"
        >>> use_legacy_feature_engineer()
        True
        >>> os.environ["ML_USE_LEGACY_FEATURE_ENGINEER"] = "0"
        >>> use_legacy_feature_engineer()
        False
        >>> del os.environ["ML_USE_LEGACY_FEATURE_ENGINEER"]
        >>> use_legacy_feature_engineer()  # Default
        False

    """
    return os.getenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0") == "1"


def use_legacy_data_store() -> bool:
    """
    Return True if using legacy DataStore, False for new facade.

    Controlled by environment variable ML_USE_LEGACY_DATA_STORE.
    Default: False (use facade/refactored implementation).

    Returns:
        True if legacy mode enabled (env var = "1"), False otherwise.

    """
    return os.getenv("ML_USE_LEGACY_DATA_STORE", "0") == "1"


def use_legacy_ml_pipeline_orchestrator() -> bool:
    """
    Return True if using legacy MLPipelineOrchestrator, False for new facade.

    Controlled by environment variable ML_USE_LEGACY_ML_PIPELINE_ORCHESTRATOR.
    Default: False (use facade/refactored implementation).

    Returns:
        True if legacy mode enabled (env var = "1"), False otherwise.

    """
    return os.getenv("ML_USE_LEGACY_ML_PIPELINE_ORCHESTRATOR", "0") == "1"


def use_legacy_ml_signal_actor() -> bool:
    """
    Return True if using legacy MLSignalActor, False for new facade.

    Controlled by environment variable ML_USE_LEGACY_ML_SIGNAL_ACTOR.
    Default: False (use facade/refactored implementation).

    Returns:
        True if legacy mode enabled (env var = "1"), False otherwise.

    """
    return os.getenv("ML_USE_LEGACY_ML_SIGNAL_ACTOR", "0") == "1"


__all__ = [
    "use_legacy_data_store",
    "use_legacy_feature_engineer",
    "use_legacy_ml_pipeline_orchestrator",
    "use_legacy_ml_signal_actor",
]
