"""Derive Phase 1 feature signal annotations from dataset columns."""

from __future__ import annotations

from collections.abc import Iterable


_CALENDAR_PREFIXES: tuple[str, ...] = ("hours_to_", "days_to_", "days_since_")
_CALENDAR_KEYWORDS: tuple[str, ...] = ("_within_", "_notice_", "_lag_")
_CALENDAR_EXACT: frozenset[str] = frozenset({"events_notice_minutes"})
_CLUSTERING_SUBSTRINGS: tuple[str, ...] = ("clustering", "density")
_CLUSTERING_PREFIXES: tuple[str, ...] = ("total_events_",)
_CONTEXT_PREFIXES: tuple[str, ...] = ("is_", "has_")
_CONTEXT_EXTRAS: frozenset[str] = frozenset({"event_importance_score"})
_MACRO_DELTA_TOKENS: tuple[str, ...] = ("_delta", "_change")


def _normalized_columns(columns: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in columns:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def derive_phase_one_signals(columns: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """
    Return categorised Phase 1 feature families derived from column names.

    Args:
        columns: Iterable of feature column names.

    Returns:
        Mapping with tuples describing macro delta, calendar lag, clustering, and
        context feature columns.
    """
    names = _normalized_columns(columns)

    def _is_macro_delta(name: str) -> bool:
        lowered = name.lower()
        return any(token in lowered for token in _MACRO_DELTA_TOKENS)

    def _is_calendar_lag(name: str) -> bool:
        lowered = name.lower()
        if name in _CALENDAR_EXACT:
            return True
        if lowered.startswith(_CALENDAR_PREFIXES):
            return True
        return any(token in lowered for token in _CALENDAR_KEYWORDS)

    def _is_clustering(name: str) -> bool:
        lowered = name.lower()
        if lowered.startswith(_CLUSTERING_PREFIXES):
            return True
        return any(token in lowered for token in _CLUSTERING_SUBSTRINGS)

    macro_deltas = {name for name in names if _is_macro_delta(name)}
    calendar_lags = {name for name in names if _is_calendar_lag(name)}
    clustering_tags = {name for name in names if _is_clustering(name)}
    context_features = {
        name
        for name in names
        if (
            name.lower().startswith(_CONTEXT_PREFIXES)
            or name in _CONTEXT_EXTRAS
        )
    }
    # Avoid overlapping categories.
    context_features.difference_update(calendar_lags)
    context_features.difference_update(clustering_tags)

    def _sorted_tuple(values: set[str]) -> tuple[str, ...]:
        return tuple(sorted(values))

    return {
        "macro_delta_columns": _sorted_tuple(macro_deltas),
        "calendar_lag_columns": _sorted_tuple(calendar_lags),
        "clustering_tag_columns": _sorted_tuple(clustering_tags),
        "context_feature_columns": _sorted_tuple(context_features),
    }


__all__ = ["derive_phase_one_signals"]
