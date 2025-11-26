"""
Feature flags for MLPipelineOrchestrator gradual rollout.

This module provides feature flags to enable gradual migration from legacy
orchestrator to new component-based facade architecture.

Environment Variables:
    ML_USE_LEGACY_ORCHESTRATOR: Set to "1" to use legacy orchestrator,
        "0" (default) to use new facade.

Examples:
    >>> import os
    >>> # Enable legacy mode
    >>> os.environ["ML_USE_LEGACY_ORCHESTRATOR"] = "1"
    >>> from ml.orchestration.feature_flags import use_legacy_orchestrator
    >>> assert use_legacy_orchestrator() is True
    >>>
    >>> # Enable facade mode (default)
    >>> os.environ["ML_USE_LEGACY_ORCHESTRATOR"] = "0"
    >>> assert use_legacy_orchestrator() is False
    >>>
    >>> # Not set - defaults to facade
    >>> os.environ.pop("ML_USE_LEGACY_ORCHESTRATOR", None)
    >>> assert use_legacy_orchestrator() is False

"""

from __future__ import annotations

import os


def use_legacy_orchestrator() -> bool:
    """
    Check if legacy orchestrator should be used.

    Controls whether the MLPipelineOrchestrator uses the legacy implementation
    or the new facade implementation with 7 extracted components.

    Returns:
        True if ML_USE_LEGACY_ORCHESTRATOR is set to "true", "1", or "yes"
        (case-insensitive), False otherwise (default: new facade)

    Note:
        Default is False (facade mode), enabling safe gradual rollout.
        Set ML_USE_LEGACY_ORCHESTRATOR to "true", "1", or "yes" to
        rollback to legacy behavior.

    Example:
        >>> import os
        >>> os.environ["ML_USE_LEGACY_ORCHESTRATOR"] = "1"
        >>> assert use_legacy_orchestrator() is True
        >>> os.environ["ML_USE_LEGACY_ORCHESTRATOR"] = "true"
        >>> assert use_legacy_orchestrator() is True
        >>> os.environ["ML_USE_LEGACY_ORCHESTRATOR"] = "yes"
        >>> assert use_legacy_orchestrator() is True
        >>> os.environ["ML_USE_LEGACY_ORCHESTRATOR"] = "0"
        >>> assert use_legacy_orchestrator() is False
        >>> os.environ.pop("ML_USE_LEGACY_ORCHESTRATOR", None)
        >>> assert use_legacy_orchestrator() is False

    """
    value = os.getenv("ML_USE_LEGACY_ORCHESTRATOR", "false").lower()
    return value in ("true", "1", "yes")
