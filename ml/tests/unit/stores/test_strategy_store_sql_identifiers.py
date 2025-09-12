from __future__ import annotations

import pytest

from ml.stores.strategy_store import StrategyStore


class _StubDialect:
    name = "sqlite"


class _StubEngine:
    dialect = _StubDialect()


def make_store_stub() -> StrategyStore:
    # Create an uninitialized instance and patch engine for _qualified_table
    store = StrategyStore.__new__(StrategyStore)  # type: ignore[call-arg]
    store.engine = _StubEngine()  # type: ignore[attr-defined]
    return store  # type: ignore[return-value]


def test_safe_identifier_allows_known_table() -> None:
    store = make_store_stub()
    # Should not raise and should return qualified table (sqlite -> base only)
    assert store._safe_table("ml_strategy_signals") == "ml_strategy_signals"


def test_safe_identifier_blocks_unknown() -> None:
    store = make_store_stub()
    with pytest.raises(ValueError):
        store._safe_table("ml_strategy_signals;DROP TABLE")

