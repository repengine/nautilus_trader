"""
Canonical macro series universes used by Tier-1 dataset builds.

This module centralises the macro indicator universe so that CLI helpers,
feature audit scripts, and ingestion flows reference a single, typed source of
truth instead of copying lists in ad-hoc scripts.  Keeping the series catalog in
``ml.config`` satisfies the AGENTS.md guidance about configuration-driven
development and makes it obvious when new indicators need ALFRED coverage or
dataset validation updates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


__all__ = sorted(
    [
        "MacroSeriesUniverse",
        "MARKET_BASED_MACRO_SERIES",
        "TIER1_MACRO_SERIES_UNIVERSE",
    ],
)


@dataclass(frozen=True, slots=True)
class MacroSeriesUniverse:
    """
    Immutable catalog describing macro indicators grouped by category.

    Args:
        categories: Mapping of category label → tuple of series identifiers.

    Raises:
        ValueError: If no categories are provided or a category is empty.
    """

    categories: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not self.categories:
            msg = "Macro series universe requires at least one category"
            raise ValueError(msg)
        normalized: dict[str, tuple[str, ...]] = {}
        for label, series in self.categories.items():
            key = label.strip()
            if not key:
                msg = "Category labels must be non-empty strings"
                raise ValueError(msg)
            values: list[str] = []
            for raw in series:
                token = raw.strip()
                if not token:
                    msg = f"Category '{label}' contains an empty series identifier"
                    raise ValueError(msg)
                values.append(token)
            if not values:
                msg = f"Category '{label}' must contain at least one series identifier"
                raise ValueError(msg)
            normalized[key] = tuple(values)
        object.__setattr__(self, "categories", MappingProxyType(normalized))

    def all_series(self) -> tuple[str, ...]:
        """Flattened tuple of unique series identifiers preserving category order."""
        series_ids: list[str] = []
        seen: set[str] = set()
        for label in self.categories:
            for series in self.categories[label]:
                if series in seen:
                    continue
                seen.add(series)
                series_ids.append(series)
        return tuple(series_ids)

    def category_items(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return the categories as ``(label, series_ids)`` tuples."""
        return tuple((label, values) for label, values in self.categories.items())

    def contains(self, series: str) -> bool:
        """Return True when ``series`` exists anywhere in the universe."""
        token = series.strip()
        if not token:
            return False
        return any(token in values for values in self.categories.values())


TIER1_MACRO_SERIES_UNIVERSE = MacroSeriesUniverse(
    categories={
        "Rates/Duration": ("DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "DFII10", "FEDFUNDS"),
        "Credit/Risk": ("BAMLC0A0CM", "BAMLH0A0HYM2", "TEDRATE", "VIXCLS"),
        "Growth/Labor": ("PAYEMS", "UNRATE", "INDPRO", "CFNAI"),
        "Inflation/Costs": ("CPIAUCSL", "PCEPI", "PPIACO", "WTISPLC"),
        "Commodities/Metals": ("PALLFNFINDEXM", "PCOPPUSDM", "NASDAQQGLDI"),
        "Cross-Asset": ("DTWEXBGS", "DEXUSAL", "DEXUSEU", "DEXJPUS"),
        "Liquidity": ("WALCL", "TOTBKCR"),
    },
)

# Market-based indicators that ALFRED does not publish as vintage series.
# These require fallback hydration via the real-time FRED feed so REAL_TIME dataset
# builds can still emit *_value_vintage_ts columns.
MARKET_BASED_MACRO_SERIES: tuple[str, ...] = ("NASDAQQGLDI",)
