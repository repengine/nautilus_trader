from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.actors.base import MLSignal
from ml.config.base import MLStrategyConfig
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.strategies.protocols import OrderExecutorProtocol
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId, StrategyId, TraderId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.orders import LimitOrder, MarketOrder, Order
from nautilus_trader.test_kit.stubs.identifiers import TestIdStubs


class _Px:
    def __init__(self, x: float) -> None:
        self._x = x

    def as_double(self) -> float:
        return self._x


class _Tick:
    def __init__(self, bid: float, ask: float) -> None:
        self.bid_price = _Px(bid)
        self.ask_price = _Px(ask)


class _EngineCache:
    def __init__(self, inst: InstrumentId) -> None:
        self._inst = inst

    def instrument(self, inst: InstrumentId) -> Any:
        class _I:
            def __init__(self, iid: InstrumentId) -> None:
                self.id = iid
                self.venue = iid.venue
                self.size_precision = 6
                self.price_precision = 5

            class _MinQ:
                def as_double(self) -> float:
                    return 0.0001

            min_quantity = _MinQ()

        if inst == self._inst:
            return _I(inst)
        return None

    def account_for_venue(self, _venue: Any) -> Any:
        class _A:
            class _B:
                def as_double(self) -> float:
                    return 10000.0

            def balance_total(self) -> Any:
                return self._B()

        return _A()

    def positions_open(self, venue: Any, instrument_id: InstrumentId) -> list[Any]:
        del venue, instrument_id
        return []

    def quote_tick(self, instrument_id: InstrumentId) -> Any:
        del instrument_id
        return _Tick(1.0, 1.0002)

    def trade_tick(self, instrument_id: InstrumentId) -> Any:
        del instrument_id
        return None

    def client_order_id(self) -> Any:
        return TestIdStubs.client_order_id()


@dataclass
class _EnginePortfolio:
    def account(self, _venue: Any) -> Any:
        class _A:
            class _B:
                def as_double(self) -> float:
                    return 10000.0

            def balance_total(self) -> Any:
                return self._B()

        return _A()


class _EngineStrategy(MLTradingStrategy):
    def __init__(self, cfg: MLStrategyConfig) -> None:
        super().__init__(cfg)
        self._submitted: list[Any] = []
        self._cache = _EngineCache(cfg.instrument_id)
        self._portfolio = _EnginePortfolio()

    @property
    def cache(self) -> _EngineCache:
        return self._cache

    @property
    def portfolio(self) -> _EnginePortfolio:
        return self._portfolio

    @property
    def trader_id(self) -> TraderId:
        return TraderId("TRADER-TEST")

    @property
    def id(self) -> StrategyId:
        return StrategyId("STRAT-TEST")

    def submit_order(self, order: Any) -> None:
        self._submitted.append(order)

    def _place_market_order(
        self,
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool = False,
    ) -> ClientOrderId:
        del reduce_only
        self._submitted.append(
            {
                "type": "market",
                "side": side.name,
                "qty": float(quantity.as_double()),
                "trader_id": str(self.trader_id),
                "strategy_id": str(self.id),
            },
        )
        return TestIdStubs.client_order_id()


class _EngineExecutor(OrderExecutorProtocol):
    def __init__(self, strat: _EngineStrategy) -> None:
        self._s = strat

    def create_order(
        self,
        side: OrderSide,
        quantity: Quantity,
        signal: MLSignal,
        market_state: dict[str, float],
        instrument: Any,
        *,
        trader_id: TraderId | None = None,
        strategy_id: StrategyId | None = None,
        client_order_id: ClientOrderId | None = None,
        init_id: UUID4 | None = None,
        ts_init: int | None = None,
    ) -> Order | None:
        del signal, trader_id, strategy_id, client_order_id, init_id, ts_init
        bid = market_state.get("bid", 0.0)
        ask = market_state.get("ask", 0.0)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
        spread_bps = ((ask - bid) / mid) * 10_000 if mid > 0 else 9999
        # Defer order construction to strategy fallback to ensure engine context completes IDs
        return None


def test_order_executor_integration_submits_order_with_ids() -> None:
    cfg = MLStrategyConfig(
        strategy_id="S-TEST",
        ml_signal_source="SRC",
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        position_size_pct=0.02,
        min_confidence=0.5,
        max_positions=1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        execute_trades=True,
        use_strategy_store=False,
    )
    strat = _EngineStrategy(cfg)
    strat.order_executor = _EngineExecutor(strat)

    sig = MLSignal(
        instrument_id=cfg.instrument_id,
        model_id="M",
        prediction=0.9,
        confidence=0.8,
        ts_event=1,
        ts_init=1,
    )
    qty = Quantity.from_str("100.0")
    order_id = strat._submit_smart_order(side=OrderSide.BUY, quantity=qty, signal=sig)
    assert order_id is not None
    assert len(strat._submitted) == 1
    submitted = strat._submitted[0]
    # Ensure IDs are set (handle dict stub or real order instance)
    if isinstance(submitted, dict):
        assert submitted.get("trader_id")
        assert submitted.get("strategy_id")
        assert submitted.get("type") == "market"
    else:
        assert getattr(submitted, "trader_id", None) is not None
        assert getattr(submitted, "strategy_id", None) is not None
        assert isinstance(submitted, (LimitOrder, MarketOrder))
