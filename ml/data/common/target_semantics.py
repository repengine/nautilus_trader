"""
Target semantics helpers for dataset builds.

Centralizes enforcement and column resolution so facade, legacy, and build
utilities share one implementation.
"""

from __future__ import annotations

from ml.config.targets import TargetSemanticsConfig
from ml.config.targets import build_binary_target_column


def resolve_target_semantics(
    target_semantics: TargetSemanticsConfig | None,
    *,
    error_message: str,
) -> TargetSemanticsConfig:
    """
    Resolve target semantics for dataset generation.

    Args:
        target_semantics: Explicit target semantics configuration.
        error_message: Message to raise when target semantics are missing.

    Returns:
        TargetSemanticsConfig instance.

    Example:
        >>> semantics = TargetSemanticsConfig()
        >>> resolve_target_semantics(semantics, error_message="target_semantics is required")
    """
    if target_semantics is None:
        raise ValueError(error_message)
    return target_semantics


def resolve_binary_target_column(target_semantics: TargetSemanticsConfig) -> str | None:
    """
    Resolve the binary target column name for positive-rate checks.

    Args:
        target_semantics: Target semantics configuration.

    Returns:
        Binary target column name if available.

    Example:
        >>> cfg = TargetSemanticsConfig()
        >>> _ = resolve_binary_target_column(cfg)
    """
    if not target_semantics.binary.enabled:
        return None
    primary = target_semantics.resolved_primary_target()
    if primary and primary.startswith("target_bin_"):
        return primary
    labels = target_semantics.horizon_labels
    if labels:
        return build_binary_target_column(labels[0])
    return None
