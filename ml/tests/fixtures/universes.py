#!/usr/bin/env python3
"""
Universe helpers for ML tests.

Provides deterministic tier-1 symbol stubs so ingestion/task suites never rely on
the real Databento tier files or environment toggles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable

import pytest


DEFAULT_TIER1_SYMBOLS: Final = ("SPY", "QQQ")


def _normalize_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(symbol).strip().upper() for symbol in symbols if symbol)
    if not normalized:
        raise ValueError("tier1_symbol_loader_stub requires at least one symbol")
    return normalized


@pytest.fixture
def tier1_symbol_loader_stub(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    """
    Install deterministic tier-1 symbol loaders for canonical ingestion modules.

    Returns the normalized symbol tuple used for both the L2 population helpers
    (`get_tier1_symbols`) and the alternative data loaders (`load_tier1_symbols`).
    """

    import importlib

    l2_loader_module = importlib.import_module("ml.data.ingest.l2_efficient")
    alt_loader_module = importlib.import_module("ml.data.loaders.alternative")

    symbols = _normalize_symbols(DEFAULT_TIER1_SYMBOLS)

    def _l2_stub(symbol_set: str | None = None) -> list[str]:
        del symbol_set
        return list(symbols)

    def _alt_stub(progress_path: Path | None = None) -> tuple[str, ...]:
        del progress_path
        return symbols

    for module in (l2_loader_module,):
        monkeypatch.setattr(module, "get_tier1_symbols", _l2_stub, raising=True)

    for module in (alt_loader_module,):
        monkeypatch.setattr(module, "load_tier1_symbols", _alt_stub, raising=True)

    return symbols


__all__ = ["DEFAULT_TIER1_SYMBOLS", "tier1_symbol_loader_stub"]
