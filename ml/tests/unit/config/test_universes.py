from __future__ import annotations

from ml.config.universes import (
    SUPPLEMENTARY_ETFS,
    TIER1_CORE,
    TIER1_CORE_12,
    TIER1_DEFAULT,
    TIER1_FULL_95,
    TIER1_SYMBOL_SETS,
)


def test_universes_nonempty_and_types() -> None:
    assert isinstance(TIER1_CORE, list) and len(TIER1_CORE) > 0
    assert all(isinstance(x, str) and x for x in TIER1_CORE)
    assert isinstance(SUPPLEMENTARY_ETFS, dict)
    assert set(SUPPLEMENTARY_ETFS.keys()) >= {"sectors", "bonds", "commodities"}


def test_tier1_symbol_sets_alignment() -> None:
    assert TIER1_SYMBOL_SETS["full"] == TIER1_FULL_95
    assert TIER1_SYMBOL_SETS["core"] == TIER1_CORE_12
    assert TIER1_DEFAULT == TIER1_FULL_95
