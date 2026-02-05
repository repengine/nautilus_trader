"""
Canonical feature name guardrails.

Centralizes legacy alias detection so datasets and manifests avoid
non-canonical feature names.
"""

from __future__ import annotations

from collections.abc import Iterable


LEGACY_FEATURE_ALIASES: dict[str, str] = {
    "volume_ratio": "volume_ratio_20",
    "price_position": "price_position_20",
    "sma_5": "price_sma_5",
    "sma_20": "price_sma_20",
    "tod_sin": "hour_sin",
    "tod_cos": "hour_cos",
    "is_market_open": "is_market_hours",
    "is_premarket": "is_pre_market",
    "is_aftermarket": "is_after_hours",
    "hour": "hour_sin/hour_cos",
    "minute": "minute_sin/minute_cos",
    "dow": "dow_sin/dow_cos",
}


def find_legacy_feature_aliases(feature_names: Iterable[str]) -> dict[str, str]:
    """
    Return legacy feature names mapped to their canonical replacements.

    Args:
        feature_names: Iterable of feature names to inspect.

    Returns:
        Mapping of legacy names to canonical replacements.
    """
    return {
        name: LEGACY_FEATURE_ALIASES[name]
        for name in feature_names
        if name in LEGACY_FEATURE_ALIASES
    }


def assert_no_legacy_feature_aliases(feature_names: Iterable[str]) -> None:
    """
    Raise when legacy feature aliases are detected.

    Args:
        feature_names: Iterable of feature names to validate.

    Raises:
        ValueError: If legacy aliases are present.
    """
    legacy = find_legacy_feature_aliases(feature_names)
    if not legacy:
        return
    formatted = ", ".join(f"{name} -> {replacement}" for name, replacement in sorted(legacy.items()))
    raise ValueError(f"Legacy feature names detected: {formatted}")


__all__ = [
    "LEGACY_FEATURE_ALIASES",
    "assert_no_legacy_feature_aliases",
    "find_legacy_feature_aliases",
]
