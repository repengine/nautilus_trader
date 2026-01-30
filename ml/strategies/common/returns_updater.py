"""
Returns update helper for sizing and portfolio volatility signals.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from nautilus_trader.model.data import BarSpecification

from ml.common.metrics_bootstrap import get_counter
from ml.config.base import ReturnsConfig
from ml.config.base import ReturnsPriceSource
from ml.config.base import ReturnsUpdateMode
from ml.strategies.common.decision_persistence import _SafeLogger


if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import InstrumentId

    from ml.actors.base import MLSignal


logger = logging.getLogger(__name__)

returns_update_total = get_counter(
    "ml_returns_update_total",
    "Total returns updates",
    labels=["source"],
)
returns_update_skipped_total = get_counter(
    "ml_returns_update_skipped_total",
    "Returns updates skipped",
    labels=["reason"],
)
returns_update_fallback_total = get_counter(
    "ml_returns_update_fallback_total",
    "Returns update price fallbacks",
    labels=["from_source", "reason"],
)


@runtime_checkable
class CacheProtocol(Protocol):
    """
    Protocol for cache access.
    """

    def quote_tick(self, instrument_id: Any) -> Any:
        """
        Get latest quote tick for instrument.
        """
        ...

    def trade_tick(self, instrument_id: Any) -> Any:
        """
        Get latest trade tick for instrument.
        """
        ...


@runtime_checkable
class PositionSizerProtocol(Protocol):
    """
    Protocol for position sizing market-data updates.
    """

    def update_market_data(self, return_pct: float) -> None:
        """
        Update volatility sizing with a return percentage.
        """
        ...

    def set_annualization_factor(self, factor: float) -> None:
        """
        Set the annualization factor for volatility calculations.
        """
        ...


@runtime_checkable
class PortfolioManagerProtocol(Protocol):
    """
    Protocol for portfolio return updates.
    """

    def update_returns(self, instrument: Any, return_pct: float) -> None:
        """
        Update returns buffer for an instrument.
        """
        ...

    def set_annualization_factor(self, factor: float) -> None:
        """
        Set the annualization factor for portfolio metrics.
        """
        ...


@dataclass(frozen=True)
class ReturnUpdateResult:
    """
    Result metadata for a returns update attempt.
    """

    instrument_id: InstrumentId
    source: ReturnsPriceSource
    return_pct: float | None
    updated: bool
    reason: str | None = None


@dataclass(frozen=True)
class _ResolvedPrice:
    price: float
    source: ReturnsPriceSource
    reason: str | None


class ReturnsUpdater:
    """
    Resolve prices and update sizing/portfolio returns consistently.
    """

    def __init__(
        self,
        *,
        config: ReturnsConfig | None = None,
        position_sizer: PositionSizerProtocol | None = None,
        portfolio_manager: PortfolioManagerProtocol | None = None,
        log: Any | None = None,
        strategy_id: str | None = None,
    ) -> None:
        self._config = config or ReturnsConfig()
        self._position_sizer = position_sizer
        self._portfolio_manager = portfolio_manager
        self._log = _SafeLogger(log if log is not None else logger)
        self._strategy_id = strategy_id
        self._last_price: dict[InstrumentId, float] = {}
        self._last_update_ns: dict[InstrumentId, int] = {}
        self._cadence_ns = self._resolve_cadence_ns(self._config)
        self._max_price_age_ns = self._resolve_max_age_ns(self._config)
        self._annualization_factor = self._resolve_annualization_factor(self._config)
        self._bar_spec: str | None = self._config.bar_spec
        if self._annualization_factor is not None:
            self._apply_annualization_factor(self._annualization_factor)

    def update_from_signal(
        self,
        signal: MLSignal,
        *,
        cache: CacheProtocol | None,
        reference_ts: int | None = None,
    ) -> ReturnUpdateResult:
        """
        Resolve a price from cache/signal metadata and update returns.
        """
        instrument_id = signal.instrument_id
        reference_ts = self._resolve_reference_ts(signal, reference_ts)
        self._maybe_apply_bar_spec_signal(signal)
        resolved = self._resolve_price(signal, cache, reference_ts)
        if resolved is None:
            reason = "price_unavailable"
            self._emit_skipped(reason)
            return ReturnUpdateResult(
                instrument_id=instrument_id,
                source=self._config.source_priority[0],
                return_pct=None,
                updated=False,
                reason=reason,
            )

        if self._cadence_ns is not None:
            last_ts = self._last_update_ns.get(instrument_id)
            if last_ts is not None and reference_ts - last_ts < self._cadence_ns:
                self._emit_skipped("cadence_skip")
                return ReturnUpdateResult(
                    instrument_id=instrument_id,
                    source=resolved.source,
                    return_pct=None,
                    updated=False,
                    reason="cadence_skip",
                )

        return_pct = self._update_returns(instrument_id, resolved.price, reference_ts)
        self._emit_update(resolved.source)
        return ReturnUpdateResult(
            instrument_id=instrument_id,
            source=resolved.source,
            return_pct=return_pct,
            updated=True,
            reason=resolved.reason,
        )

    def update_from_bar(
        self,
        bar: Any,
        *,
        cache: CacheProtocol | None,
        reference_ts: int | None = None,
    ) -> ReturnUpdateResult:
        """
        Resolve a price from bar/cached data and update returns.
        """
        if getattr(bar, "is_revision", False):
            self._emit_skipped("bar_revision")
            return ReturnUpdateResult(
                instrument_id=bar.bar_type.instrument_id,
                source=self._config.source_priority[0],
                return_pct=None,
                updated=False,
                reason="bar_revision",
            )
        instrument_id = bar.bar_type.instrument_id
        reference_ts = (
            reference_ts
            if reference_ts is not None and reference_ts > 0
            else int(getattr(bar, "ts_event", 0) or 0) or time.time_ns()
        )
        self._maybe_apply_bar_spec_value(str(bar.bar_type.spec))
        resolved = self._resolve_price_from_bar(bar, cache, reference_ts)
        if resolved is None:
            reason = "price_unavailable"
            self._emit_skipped(reason)
            return ReturnUpdateResult(
                instrument_id=instrument_id,
                source=self._config.source_priority[0],
                return_pct=None,
                updated=False,
                reason=reason,
            )

        if self._cadence_ns is not None:
            last_ts = self._last_update_ns.get(instrument_id)
            if last_ts is not None and reference_ts - last_ts < self._cadence_ns:
                self._emit_skipped("cadence_skip")
                return ReturnUpdateResult(
                    instrument_id=instrument_id,
                    source=resolved.source,
                    return_pct=None,
                    updated=False,
                    reason="cadence_skip",
                )

        return_pct = self._update_returns(instrument_id, resolved.price, reference_ts)
        self._emit_update(resolved.source)
        return ReturnUpdateResult(
            instrument_id=instrument_id,
            source=resolved.source,
            return_pct=return_pct,
            updated=True,
            reason=resolved.reason,
        )

    def should_update_from_signal(self) -> bool:
        """
        Return whether signal-driven updates are enabled.
        """
        return self._config.update_mode in (
            ReturnsUpdateMode.SIGNAL,
            ReturnsUpdateMode.BOTH,
        )

    def should_update_from_bar(self) -> bool:
        """
        Return whether bar-driven updates are enabled.
        """
        return self._config.update_mode in (
            ReturnsUpdateMode.BAR,
            ReturnsUpdateMode.BOTH,
        )

    def _resolve_reference_ts(self, signal: MLSignal, reference_ts: int | None) -> int:
        if reference_ts is not None and reference_ts > 0:
            return reference_ts
        ts_event = int(getattr(signal, "ts_event", 0) or 0)
        if ts_event > 0:
            return ts_event
        return time.time_ns()

    def _resolve_price(
        self,
        signal: MLSignal,
        cache: CacheProtocol | None,
        reference_ts: int,
    ) -> _ResolvedPrice | None:
        last_reason: str | None = None
        for source in self._config.source_priority:
            if source is ReturnsPriceSource.BAR_CLOSE:
                price = self._signal_bar_close(signal)
                if price is not None:
                    return _ResolvedPrice(price=price, source=source, reason=None)
                last_reason = "bar_close_missing"
            elif source is ReturnsPriceSource.QUOTE_MID:
                if cache is None:
                    last_reason = "quote_unavailable"
                else:
                    price, reason = self._quote_mid(cache, signal.instrument_id, reference_ts)
                    if price is not None:
                        return _ResolvedPrice(price=price, source=source, reason=reason)
                    last_reason = reason or "quote_unavailable"
            elif source is ReturnsPriceSource.LAST_TRADE:
                if cache is None:
                    last_reason = "trade_unavailable"
                else:
                    price, reason = self._last_trade(cache, signal.instrument_id, reference_ts)
                    if price is not None:
                        return _ResolvedPrice(price=price, source=source, reason=reason)
                    last_reason = reason or "trade_unavailable"
            else:
                last_reason = "price_unavailable"

            self._emit_fallback(source, last_reason)

        return None

    def _resolve_price_from_bar(
        self,
        bar: Any,
        cache: CacheProtocol | None,
        reference_ts: int,
    ) -> _ResolvedPrice | None:
        last_reason: str | None = None
        for source in self._config.source_priority:
            if source is ReturnsPriceSource.BAR_CLOSE:
                price = self._bar_close(bar)
                if price is not None:
                    return _ResolvedPrice(price=price, source=source, reason=None)
                last_reason = "bar_close_missing"
            elif source is ReturnsPriceSource.QUOTE_MID:
                if cache is None:
                    last_reason = "quote_unavailable"
                else:
                    price, reason = self._quote_mid(cache, bar.bar_type.instrument_id, reference_ts)
                    if price is not None:
                        return _ResolvedPrice(price=price, source=source, reason=reason)
                    last_reason = reason or "quote_unavailable"
            elif source is ReturnsPriceSource.LAST_TRADE:
                if cache is None:
                    last_reason = "trade_unavailable"
                else:
                    price, reason = self._last_trade(cache, bar.bar_type.instrument_id, reference_ts)
                    if price is not None:
                        return _ResolvedPrice(price=price, source=source, reason=reason)
                    last_reason = reason or "trade_unavailable"
            else:
                last_reason = "price_unavailable"

            self._emit_fallback(source, last_reason)

        return None

    def _signal_bar_close(self, signal: MLSignal) -> float | None:
        meta = getattr(signal, "metadata", None) or {}
        value = meta.get("bar_close")
        if value is None:
            return None
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if price > 0 else None

    def _bar_close(self, bar: Any) -> float | None:
        price = self._price_to_float(getattr(bar, "close", None))
        if price is None:
            return None
        return price if price > 0 else None

    def _quote_mid(
        self,
        cache: CacheProtocol,
        instrument_id: InstrumentId,
        reference_ts: int,
    ) -> tuple[float | None, str | None]:
        tick = cache.quote_tick(instrument_id)
        if tick is None:
            return None, "quote_unavailable"
        bid = self._price_to_float(getattr(tick, "bid_price", None))
        ask = self._price_to_float(getattr(tick, "ask_price", None))
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None, "quote_invalid"
        if ask < bid:
            return None, "quote_crossed"
        if self._max_price_age_ns is not None:
            age_ns = reference_ts - int(getattr(tick, "ts_event", 0) or 0)
            if age_ns < 0:
                age_ns = 0
            if age_ns > self._max_price_age_ns:
                return None, "quote_stale"
        return (bid + ask) / 2.0, None

    def _last_trade(
        self,
        cache: CacheProtocol,
        instrument_id: InstrumentId,
        reference_ts: int,
    ) -> tuple[float | None, str | None]:
        tick = cache.trade_tick(instrument_id)
        if tick is None:
            return None, "trade_unavailable"
        price = self._price_to_float(getattr(tick, "price", None))
        if price is None or price <= 0:
            return None, "trade_invalid"
        if self._max_price_age_ns is not None:
            age_ns = reference_ts - int(getattr(tick, "ts_event", 0) or 0)
            if age_ns < 0:
                age_ns = 0
            if age_ns > self._max_price_age_ns:
                return None, "trade_stale"
        return price, None

    def _update_returns(
        self,
        instrument_id: InstrumentId,
        price: float,
        reference_ts: int,
    ) -> float | None:
        last_price = self._last_price.get(instrument_id)
        self._last_price[instrument_id] = price
        self._last_update_ns[instrument_id] = reference_ts
        if last_price is None:
            return None
        if last_price <= 0:
            return None
        return_pct = (price / last_price) - 1.0
        if self._portfolio_manager is not None:
            try:
                self._portfolio_manager.update_returns(instrument_id, return_pct)
            except Exception as exc:
                self._log.debug(
                    "returns_updater.portfolio_update_failed",
                    strategy_id=self._strategy_id,
                    exc_info=True,
                    error=str(exc),
                )
        if self._position_sizer is not None:
            try:
                self._position_sizer.update_market_data(return_pct)
            except Exception as exc:
                self._log.debug(
                    "returns_updater.sizer_update_failed",
                    strategy_id=self._strategy_id,
                    exc_info=True,
                    error=str(exc),
                )
        return return_pct

    def _resolve_cadence_ns(self, config: ReturnsConfig) -> int | None:
        if config.update_cadence_ms is not None:
            return int(config.update_cadence_ms) * 1_000_000
        if config.bar_spec:
            try:
                spec = BarSpecification.from_str(config.bar_spec)
            except Exception as exc:
                self._log.debug(
                    "returns_updater.bar_spec_parse_failed",
                    strategy_id=self._strategy_id,
                    exc_info=True,
                    error=str(exc),
                    bar_spec=config.bar_spec,
                )
                return None
            interval_ns = int(spec.get_interval_ns())
            return interval_ns if interval_ns > 0 else None
        return None

    def _resolve_bar_spec(self, bar_spec: str) -> tuple[int | None, float | None]:
        try:
            spec = BarSpecification.from_str(bar_spec)
        except Exception:
            return None, None
        interval_ns = int(spec.get_interval_ns())
        if interval_ns <= 0:
            return None, None
        seconds_per_year = 365.25 * 24 * 60 * 60
        cadence_ns = interval_ns
        annualization_factor = seconds_per_year / (interval_ns / 1_000_000_000)
        return cadence_ns, annualization_factor

    def _maybe_apply_bar_spec_value(self, bar_spec: str | None) -> None:
        if self._bar_spec is not None or not bar_spec:
            return
        bar_spec_str = str(bar_spec)
        cadence_ns, annualization_factor = self._resolve_bar_spec(bar_spec_str)
        self._bar_spec = bar_spec_str
        if self._cadence_ns is None:
            self._cadence_ns = cadence_ns
        if self._annualization_factor is None and annualization_factor is not None:
            self._annualization_factor = annualization_factor
            self._apply_annualization_factor(annualization_factor)

    def _maybe_apply_bar_spec_signal(self, signal: MLSignal) -> None:
        meta = getattr(signal, "metadata", None) or {}
        bar_spec = meta.get("bar_spec")
        if not bar_spec:
            return
        self._maybe_apply_bar_spec_value(str(bar_spec))

    def _resolve_annualization_factor(self, config: ReturnsConfig) -> float | None:
        if config.annualization_factor is not None:
            return float(config.annualization_factor)
        if not config.bar_spec:
            return None
        try:
            spec = BarSpecification.from_str(config.bar_spec)
        except Exception as exc:
            self._log.debug(
                "returns_updater.annualization_parse_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(exc),
                bar_spec=config.bar_spec,
            )
            return None
        interval_ns = int(spec.get_interval_ns())
        if interval_ns <= 0:
            return None
        seconds_per_year = 365.25 * 24 * 60 * 60
        return seconds_per_year / (interval_ns / 1_000_000_000)

    def _resolve_max_age_ns(self, config: ReturnsConfig) -> int | None:
        if config.max_price_age_ms is None:
            return None
        try:
            max_age_ms = int(config.max_price_age_ms)
        except (TypeError, ValueError):
            return None
        if max_age_ms <= 0:
            return None
        return max_age_ms * 1_000_000

    def _apply_annualization_factor(self, factor: float) -> None:
        if self._portfolio_manager is not None:
            try:
                self._portfolio_manager.set_annualization_factor(factor)
            except Exception as exc:
                self._log.debug(
                    "returns_updater.portfolio_annualization_failed",
                    strategy_id=self._strategy_id,
                    exc_info=True,
                    error=str(exc),
                )
        if self._position_sizer is not None:
            try:
                self._position_sizer.set_annualization_factor(factor)
            except Exception as exc:
                self._log.debug(
                    "returns_updater.sizer_annualization_failed",
                    strategy_id=self._strategy_id,
                    exc_info=True,
                    error=str(exc),
                )

    def _price_to_float(self, price_obj: Any) -> float | None:
        if price_obj is None:
            return None
        try:
            return float(price_obj.as_double())
        except Exception:
            try:
                return float(price_obj)
            except Exception:
                return None

    def _emit_update(self, source: ReturnsPriceSource) -> None:
        try:
            returns_update_total.labels(source=source.value).inc()
        except Exception as exc:
            self._log.debug(
                "returns_updater.metric_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(exc),
            )

    def _emit_skipped(self, reason: str) -> None:
        try:
            returns_update_skipped_total.labels(reason=reason).inc()
        except Exception as exc:
            self._log.debug(
                "returns_updater.metric_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(exc),
            )

    def _emit_fallback(self, source: ReturnsPriceSource, reason: str | None) -> None:
        if reason is None:
            return
        try:
            returns_update_fallback_total.labels(
                from_source=source.value,
                reason=reason,
            ).inc()
        except Exception as exc:
            self._log.debug(
                "returns_updater.metric_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(exc),
            )
