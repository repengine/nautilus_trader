"""
RecorderActor — forwards live market data to LiveDataRecorder.

Hot path considerations:
- O(1) handlers calling pre-bound recorder methods
- No allocations in on_* methods beyond call frames
- Persistence happens off-thread via LiveDataRecorder flush tasks

Public API is intentionally minimal; this actor is added alongside signal/strategy
actors to mirror the same subscriptions and persist raw bars (optionally quotes/trades
in the future) via DataStore.
"""

from __future__ import annotations

from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick

from ml.actors.base import NautilusActor
from ml.stores.writers import LiveDataRecorder


__all__ = ["RecorderActor"]


class RecorderActor(NautilusActor):
    """Lightweight actor that forwards data events to a LiveDataRecorder."""

    def __init__(
        self,
        *,
        recorder: LiveDataRecorder,
        record_bars: bool = True,
        record_quotes: bool = False,
        record_trades: bool = False,
    ) -> None:
        # Parent takes an ActorConfig; RecorderActor does not require special config
        super().__init__()
        self._recorder = recorder
        self._record_bars = record_bars
        self._record_quotes = record_quotes
        self._record_trades = record_trades

        # Pre-bind methods to avoid attribute lookups in hot path
        self._on_bar_cb = self._recorder.on_bar
        self._on_quote_cb = self._recorder.on_quote
        self._on_trade_cb = self._recorder.on_trade

    # Hot path handlers (no allocations, direct forward)
    def on_bar(self, bar: Bar) -> None:  # pragma: no cover - exercised in integration
        if self._record_bars:
            self._on_bar_cb(bar)

    def on_quote_tick(self, tick: QuoteTick) -> None:  # pragma: no cover - future use
        if self._record_quotes:
            self._on_quote_cb(tick)

    def on_trade_tick(self, tick: TradeTick) -> None:  # pragma: no cover - future use
        if self._record_trades:
            self._on_trade_cb(tick)

    def on_stop(self) -> None:
        # Best-effort flush on shutdown (schedule async flush if loop active)
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._recorder.flush_all())
        except Exception:
            # Silent best-effort; recorder will also flush when stopped via its API
            pass
