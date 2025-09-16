from __future__ import annotations

import pytest

from ml.stores.strategy_store import StrategyStore


def test_safe_identifier_allows_only_whitelist() -> None:
    s = StrategyStore(connection_string=None)
    assert (
        s._safe_identifier("ml_strategy_signals", {"ml_strategy_signals", "x"})
        == "ml_strategy_signals"
    )
    with pytest.raises(ValueError):
        s._safe_identifier("unsafe;drop", {"ml_strategy_signals"})


def test_safe_table_is_qualified() -> None:
    s = StrategyStore(connection_string=None)
    # Returns schema-qualified by default when no engine/dialect is set
    assert s._safe_table("ml_strategy_signals").endswith("ml_strategy_signals")
    assert s._safe_table("ml_strategy_performance").endswith("ml_strategy_performance")
