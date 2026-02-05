"""
Feature cache policy configuration for dataset builders.

Defines the canonical policy tokens used to control micro/L2 cache usage in
dataset builders and orchestration config. Policies are normalized to
lower-case and validated to avoid silent drift in builder behavior.
"""

from __future__ import annotations

from typing import Literal, TypeGuard


FeatureCachePolicy = Literal["cache_first", "cache_only", "live_only"]

_VALID_FEATURE_CACHE_POLICIES: tuple[FeatureCachePolicy, ...] = (
    "cache_first",
    "cache_only",
    "live_only",
)


def _is_feature_cache_policy(value: str) -> TypeGuard[FeatureCachePolicy]:
    return value in _VALID_FEATURE_CACHE_POLICIES


def normalize_feature_cache_policy(
    value: FeatureCachePolicy | str | None,
    *,
    label: str,
) -> FeatureCachePolicy:
    """
    Normalize and validate a feature cache policy token.

    Args:
        value: Policy token or None (defaults to ``cache_first``).
        label: Label used in validation errors.

    Returns:
        Normalized policy token.

    Raises:
        ValueError: If the policy token is invalid.
    """
    if value is None:
        return "cache_first"
    token = str(value).strip().lower()
    if not token:
        return "cache_first"
    if not _is_feature_cache_policy(token):
        msg = (
            f"{label} must be one of {_VALID_FEATURE_CACHE_POLICIES}, got {value!r}"
        )
        raise ValueError(msg)
    return token


__all__ = ["FeatureCachePolicy", "normalize_feature_cache_policy"]
