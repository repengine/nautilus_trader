"""
Shared signal batching for portfolio allocation.

This helper maintains a rolling buffer of recent ML signals grouped by portfolio
context so portfolio allocation can consider multiple instruments in a time window.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ml.strategies.portfolio import PortfolioBatchingConfig


if TYPE_CHECKING:
    from ml.actors.base import MLSignal


class PortfolioSignalBatcher:
    """
    Maintain recent signals for cross-instrument portfolio allocation.

    The batcher groups signals by a resolved portfolio key and returns the
    most recent signals within the configured window, capped by max size.
    """

    def __init__(self) -> None:
        """Initialize an empty batcher."""
        self._buffers: dict[str, dict[Any, MLSignal]] = {}

    def update_and_get_batch(
        self,
        signal: MLSignal,
        *,
        config: PortfolioBatchingConfig,
        portfolio: Any | None,
        strategy_id: str,
    ) -> list[MLSignal]:
        """
        Add a signal to the buffer and return the current batch.

        Parameters
        ----------
        signal : MLSignal
            The incoming signal.
        config : PortfolioBatchingConfig
            Batching configuration.
        portfolio : Any | None
            Portfolio instance used to derive grouping.
        strategy_id : str
            Strategy identifier used when portfolio is unavailable.

        Returns
        -------
        list[MLSignal]
            Signals eligible for allocation.

        """
        if not config.enabled:
            return [signal]

        group_key = self._resolve_group_key(portfolio, strategy_id)
        buffer = self._buffers.setdefault(group_key, {})
        buffer[signal.instrument_id] = signal
        return self._resolve_batch(buffer, signal, config)

    def clear(self, group_key: str | None = None) -> None:
        """
        Clear buffered signals.

        Parameters
        ----------
        group_key : str | None, optional
            Clear a specific group when provided, else clear all.

        """
        if group_key is None:
            self._buffers.clear()
            return
        self._buffers.pop(group_key, None)

    @staticmethod
    def _resolve_group_key(portfolio: Any | None, strategy_id: str) -> str:
        """
        Resolve the grouping key for a portfolio context.
        """
        if portfolio is None:
            return f"strategy:{strategy_id}"
        portfolio_id = getattr(portfolio, "id", None) or getattr(portfolio, "name", None)
        if portfolio_id is not None:
            return f"portfolio:{portfolio_id}"
        return f"portfolio_obj:{id(portfolio)}"

    @staticmethod
    def _resolve_signal_ts(signal: MLSignal) -> int:
        """
        Resolve a nanosecond timestamp for a signal.
        """
        ts_event = int(getattr(signal, "ts_event", 0) or 0)
        if ts_event > 0:
            return ts_event
        ts_init = int(getattr(signal, "ts_init", 0) or 0)
        if ts_init > 0:
            return ts_init
        return time.time_ns()

    def _resolve_batch(
        self,
        buffer: dict[Any, MLSignal],
        current_signal: MLSignal,
        config: PortfolioBatchingConfig,
    ) -> list[MLSignal]:
        """
        Resolve the current batch from a buffer.
        """
        if not buffer:
            return []

        signals_with_ts = [
            (signal, self._resolve_signal_ts(signal)) for signal in buffer.values()
        ]
        latest_ts = max(ts for _, ts in signals_with_ts)

        window_ns = int(config.window_ms) * 1_000_000
        if window_ns > 0:
            for instrument_id, signal in list(buffer.items()):
                ts = self._resolve_signal_ts(signal)
                if latest_ts - ts > window_ns:
                    del buffer[instrument_id]

        if not buffer:
            return []

        signals_with_ts = [
            (signal, self._resolve_signal_ts(signal)) for signal in buffer.values()
        ]

        max_batch_size = int(config.max_batch_size)
        if max_batch_size > 0 and len(signals_with_ts) > max_batch_size:
            signals_with_ts.sort(key=lambda item: item[1], reverse=True)
            kept = signals_with_ts[:max_batch_size]
            keep_instruments = {signal.instrument_id for signal, _ in kept}
            for instrument_id in list(buffer.keys()):
                if instrument_id not in keep_instruments:
                    del buffer[instrument_id]
            signals_with_ts = kept

        signals = [signal for signal, _ in signals_with_ts]
        min_batch_size = int(config.min_batch_size)
        if min_batch_size > 1 and len(signals) < min_batch_size:
            return [current_signal]
        return signals


_PORTFOLIO_SIGNAL_BATCHER = PortfolioSignalBatcher()


def get_portfolio_signal_batcher() -> PortfolioSignalBatcher:
    """
    Return the shared portfolio signal batcher instance.
    """
    return _PORTFOLIO_SIGNAL_BATCHER


__all__ = [
    "PortfolioSignalBatcher",
    "get_portfolio_signal_batcher",
]
