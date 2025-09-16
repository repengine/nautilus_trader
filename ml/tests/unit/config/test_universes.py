from __future__ import annotations

from ml.config.universes import SUPPLEMENTARY_ETFS, TIER1_CORE


def test_universes_nonempty_and_types() -> None:
    assert isinstance(TIER1_CORE, list) and len(TIER1_CORE) > 0
    assert all(isinstance(x, str) and x for x in TIER1_CORE)
    assert isinstance(SUPPLEMENTARY_ETFS, dict)
    assert set(SUPPLEMENTARY_ETFS.keys()) >= {"sectors", "bonds", "commodities"}

