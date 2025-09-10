"""
Smoke test for MLTradingStrategy decision flow in dry-run mode.

Bypasses Nautilus base initialization by constructing an instance via object.__new__ and
stubbing required methods/attributes. Verifies that _process_ml_signal records a
decision and increments dry-run counters.

"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ml.actors.base import MLSignal
from ml.strategies.ml_strategy import MLTradingStrategy
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


class _Log:
    def info(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op info.
        """
        return None

    def warning(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op warning.
        """
        return None

    def debug(self, *args: Any, **kwargs: Any) -> None:
        """
        No-op debug.
        """
        return None


def _build_signal(pred: float, conf: float) -> MLSignal:
    inst = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
    return MLSignal(
        instrument_id=inst,
        model_id="m1",
        prediction=pred,
        confidence=conf,
        ts_event=1,
        ts_init=1,
    )


def test_ml_trading_strategy_process_signal_dry_run() -> None:
    # Dummy self object implementing attributes used by _process_ml_signal
    class _Dummy:
        pass

    strat = _Dummy()
    strat.log = _Log()
    strat._config = SimpleNamespace(execute_trades=False)
    strat._active_positions = 0
    strat._dry_run_trades = 0
    strat.track_performance = False

    decisions: list[dict[str, Any]] = []

    def _persist_strategy_decision(**kwargs: Any) -> None:
        decisions.append(kwargs)

    # Stubs for position state and actions
    strat._get_current_position = lambda: None
    strat._calculate_position_size = lambda: None
    strat._should_reverse_position = lambda cur, tgt: False
    strat._enter_position = lambda side, sig: None
    strat._reverse_position = lambda cur, side, sig: None
    strat._persist_strategy_decision = _persist_strategy_decision

    # Process a bullish signal (>0.5) by calling the unbound method
    sig = _build_signal(0.9, 0.8)
    MLTradingStrategy._process_ml_signal(strat, sig)  # type: ignore[misc]

    # One dry run trade and one persisted decision
    assert strat._dry_run_trades == 1
    assert len(decisions) == 1
    assert decisions[0]["decision_type"] in {"BUY", "SELL", "HOLD"}
