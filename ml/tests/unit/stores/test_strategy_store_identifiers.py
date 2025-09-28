from __future__ import annotations

import pytest

from ml.stores.strategy_store import StrategyStore


@pytest.fixture
def strategy_store(monkeypatch: pytest.MonkeyPatch) -> StrategyStore:
    """
    Return a strategy store instance without touching a real database.
    """

    monkeypatch.setattr(
        "ml.stores.strategy_store.StrategyStore._init_engine_and_tables",
        lambda self: None,
    )
    return StrategyStore(connection_string=None)


def test_safe_identifier_allows_only_whitelist(strategy_store: StrategyStore) -> None:
    s = strategy_store
    assert (
        s._safe_identifier("ml_strategy_signals", {"ml_strategy_signals", "x"})
        == "ml_strategy_signals"
    )
    with pytest.raises(ValueError):
        s._safe_identifier("unsafe;drop", {"ml_strategy_signals"})


def test_safe_table_is_qualified(strategy_store: StrategyStore) -> None:
    s = strategy_store
    # Returns schema-qualified by default when no engine/dialect is set
    assert s._safe_table("ml_strategy_signals").endswith("ml_strategy_signals")
    assert s._safe_table("ml_strategy_performance").endswith("ml_strategy_performance")
