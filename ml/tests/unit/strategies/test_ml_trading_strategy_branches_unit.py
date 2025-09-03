"""
Additional branch coverage tests for MLTradingStrategy.

Covers reversal and hold branches using a dummy self instance and
stubs for methods/attributes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ml.actors.base import MLSignal
from ml.strategies.ml_strategy import MLTradingStrategy
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue


class _Log:
    def info(self, *a: Any, **k: Any) -> None:  # noqa: D401
        """No-op info"""
        return None

    def warning(self, *a: Any, **k: Any) -> None:  # noqa: D401
        """No-op warn"""
        return None

    def debug(self, *a: Any, **k: Any) -> None:  # noqa: D401
        """No-op debug"""
        return None


def _sig(pred: float, conf: float = 0.8) -> MLSignal:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return MLSignal(instrument_id=inst, model_id="m1", prediction=pred, confidence=conf, ts_event=1, ts_init=1)


def test_reversal_branch_dry_run() -> None:
    class _Pos:
        def __init__(self, side_name: str) -> None:
            self.side = SimpleNamespace(name=side_name)
            self.quantity = SimpleNamespace()

    strat = SimpleNamespace()
    strat.log = _Log()
    strat._config = SimpleNamespace(execute_trades=False)
    strat._active_positions = 1
    strat._dry_run_trades = 0
    strat.track_performance = False
    # Stubs
    strat._get_current_position = lambda: _Pos("LONG")
    strat._calculate_position_size = lambda: None
    strat._should_reverse_position = MLTradingStrategy._should_reverse_position.__get__(strat, MLTradingStrategy)  # type: ignore[misc]
    strat._enter_position = lambda side, sig: None
    strat._reverse_position = MLTradingStrategy._reverse_position.__get__(strat, MLTradingStrategy)  # type: ignore[misc]
    strat._place_market_order = lambda *a, **k: None
    strat._persist_strategy_decision = lambda **k: None

    # Signal indicates SELL vs current LONG -> reversal path
    MLTradingStrategy._process_ml_signal(strat, _sig(0.0))  # type: ignore[misc]
    assert strat._dry_run_trades == 1


def test_hold_branch_persists_decision() -> None:
    class _Pos:
        def __init__(self, side_name: str) -> None:
            self.side = SimpleNamespace(name=side_name)

    strat = SimpleNamespace()
    strat.log = _Log()
    strat._config = SimpleNamespace(execute_trades=False)
    strat._active_positions = 1
    strat._dry_run_trades = 0
    strat.track_performance = False
    decisions: list[dict[str, Any]] = []

    # Stubs
    strat._get_current_position = lambda: _Pos("LONG")
    strat._calculate_position_size = lambda: None
    strat._should_reverse_position = lambda cur, tgt: False
    strat._enter_position = lambda side, sig: None
    strat._reverse_position = lambda cur, side, sig: None
    strat._persist_strategy_decision = lambda **kw: decisions.append(kw)

    # Signal indicates BUY vs current LONG -> HOLD branch
    MLTradingStrategy._process_ml_signal(strat, _sig(0.9))  # type: ignore[misc]
    assert decisions and decisions[-1]["decision_type"] == "HOLD"

