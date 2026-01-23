"""
Positions provider implementation for Nautilus strategy components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ml.common.metrics_bootstrap import get_counter
from ml.config.base import PositionsConfig
from ml.config.base import PositionsSource
from ml.strategies.common.decision_persistence import _SafeLogger
from ml.strategies.common.positions import PositionsHealthStatus
from ml.strategies.common.positions import PositionsProviderProtocol
from ml.strategies.common.positions import PositionsSnapshot
from ml.strategies.common.positions import PositionViewProtocol


if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import InstrumentId


logger = logging.getLogger(__name__)


@runtime_checkable
class CachePositionsProtocol(Protocol):
    """
    Cache interface for positions access.
    """

    def positions_open(
        self,
        venue: Any = None,
        instrument_id: InstrumentId | None = None,
    ) -> list[PositionViewProtocol]:
        """
        Return open positions for the cache.
        """
        ...

    def positions(
        self,
        venue: Any = None,
        instrument_id: InstrumentId | None = None,
    ) -> list[PositionViewProtocol]:
        """
        Return all positions for the cache.
        """
        ...


@runtime_checkable
class PortfolioPositionsProtocol(Protocol):
    """
    Portfolio interface for positions access.
    """

    def positions(self) -> list[PositionViewProtocol]:
        """
        Return all portfolio positions.
        """
        ...

    def positions_open(self) -> list[PositionViewProtocol]:
        """
        Return open portfolio positions.
        """
        ...

    def net_position(self, instrument_id: InstrumentId) -> Decimal:
        """
        Return the net position for a single instrument.
        """
        ...


@dataclass
class _NetPositionView:
    """
    Lightweight position view backed by a net quantity.
    """

    instrument_id: InstrumentId
    net_qty: Decimal
    is_open: bool = field(init=False)

    def __post_init__(self) -> None:
        """
        Populate the derived open/closed flag.
        """
        self.is_open = self.net_qty != Decimal("0")

    def signed_decimal_qty(self) -> Decimal:
        """
        Return the signed net quantity.
        """
        return self.net_qty


class NautilusPositionsProvider(PositionsProviderProtocol):
    """
    Resolve positions via cache/portfolio APIs with fallback ordering.
    """

    def __init__(
        self,
        *,
        cache: CachePositionsProtocol | None,
        portfolio: PortfolioPositionsProtocol | None,
        config: PositionsConfig | None = None,
        log: Any = None,
        strategy_id: str | None = None,
    ) -> None:
        self._cache = cache
        self._portfolio = portfolio
        self._config = config or PositionsConfig()
        self._log = _SafeLogger(log if log is not None else logger)
        self._strategy_id = strategy_id

    def get_positions_snapshot(
        self,
        *,
        instrument_id: InstrumentId | None = None,
        require_full_list: bool = False,
    ) -> PositionsSnapshot:
        """
        Return a snapshot of open positions.

        Parameters
        ----------
        instrument_id : InstrumentId | None, optional
            Optional instrument filter for per-instrument lookups.
        require_full_list : bool, default False
            When True, ignore instrument filters for list-based sources.

        Returns
        -------
        PositionsSnapshot
            Snapshot of open positions with source metadata.

        """
        snapshot, _ = self._resolve_snapshot(
            instrument_id=instrument_id,
            require_full_list=require_full_list,
        )
        self._emit_snapshot_metrics(snapshot)
        return snapshot

    def check_positions_ready(
        self,
        *,
        instrument_id: InstrumentId | None = None,
        require_full_list: bool = False,
        require_positions: bool = False,
    ) -> PositionsHealthStatus:
        """
        Evaluate positions readiness for live trading.

        Parameters
        ----------
        instrument_id : InstrumentId | None, optional
            Optional instrument filter for per-instrument fallbacks.
        require_full_list : bool, default False
            Require full list access rather than per-instrument fallback.
        require_positions : bool, default False
            Whether a positions source must be available for readiness.

        Returns
        -------
        PositionsHealthStatus
            Health status describing positions readiness.

        """
        snapshot, source_available = self._resolve_snapshot(
            instrument_id=instrument_id,
            require_full_list=require_full_list,
        )
        self._emit_snapshot_metrics(snapshot)
        positions_count = len(snapshot.positions)
        reason: str | None = None
        degraded = False
        if not source_available:
            degraded = True
            reason = "positions_unavailable"
            ready = not require_positions
        elif snapshot.source is PositionsSource.PORTFOLIO_NET:
            degraded = True
            reason = "net_position_only"
            ready = not require_full_list
        else:
            ready = True

        if degraded and reason is not None:
            self._emit_degraded_metric(reason)

        return PositionsHealthStatus(
            ready=ready,
            degraded=degraded,
            source=snapshot.source,
            reason=reason,
            positions_count=positions_count,
        )

    def _resolve_snapshot(
        self,
        *,
        instrument_id: InstrumentId | None,
        require_full_list: bool,
    ) -> tuple[PositionsSnapshot, bool]:
        for index, source in enumerate(self._config.source_priority):
            snapshot = self._try_source(
                source=source,
                instrument_id=instrument_id,
                require_full_list=require_full_list,
            )
            if snapshot is None:
                continue
            if index > 0:
                self._emit_fallback_metric(source.value)
            return snapshot, True

        primary = self._config.source_priority[0]
        self._log.debug(
            "ml_positions_snapshot_unavailable",
            strategy_id=self._strategy_id,
            instrument=str(instrument_id) if instrument_id else None,
        )
        return PositionsSnapshot(positions=[], source=primary), False

    def _try_source(
        self,
        *,
        source: PositionsSource,
        instrument_id: InstrumentId | None,
        require_full_list: bool,
    ) -> PositionsSnapshot | None:
        if source is PositionsSource.CACHE_OPEN:
            return self._from_cache_open(instrument_id, require_full_list)
        if source is PositionsSource.CACHE_ALL:
            return self._from_cache_all(instrument_id, require_full_list)
        if source is PositionsSource.PORTFOLIO_POSITIONS:
            return self._from_portfolio_positions()
        if source is PositionsSource.PORTFOLIO_POSITIONS_OPEN:
            return self._from_portfolio_positions_open()
        if source is PositionsSource.PORTFOLIO_NET:
            return self._from_portfolio_net(instrument_id)
        return None

    def _from_cache_open(
        self,
        instrument_id: InstrumentId | None,
        require_full_list: bool,
    ) -> PositionsSnapshot | None:
        if self._cache is None or not hasattr(self._cache, "positions_open"):
            return None
        filter_id = None if require_full_list else instrument_id
        try:
            positions = self._cache.positions_open(venue=None, instrument_id=filter_id)
        except Exception as exc:
            self._log.debug(
                "ml_positions_cache_open_failed",
                strategy_id=self._strategy_id,
                instrument=str(instrument_id) if instrument_id else None,
                exc_info=True,
                error=str(exc),
            )
            return None
        return PositionsSnapshot(positions=list(positions), source=PositionsSource.CACHE_OPEN)

    def _from_cache_all(
        self,
        instrument_id: InstrumentId | None,
        require_full_list: bool,
    ) -> PositionsSnapshot | None:
        if self._cache is None or not hasattr(self._cache, "positions"):
            return None
        filter_id = None if require_full_list else instrument_id
        try:
            positions = self._cache.positions(venue=None, instrument_id=filter_id)
        except Exception as exc:
            self._log.debug(
                "ml_positions_cache_all_failed",
                strategy_id=self._strategy_id,
                instrument=str(instrument_id) if instrument_id else None,
                exc_info=True,
                error=str(exc),
            )
            return None
        return PositionsSnapshot(positions=list(positions), source=PositionsSource.CACHE_ALL)

    def _from_portfolio_positions(self) -> PositionsSnapshot | None:
        if self._portfolio is None or not hasattr(self._portfolio, "positions"):
            return None
        try:
            positions = self._portfolio.positions()
        except Exception as exc:
            self._log.debug(
                "ml_positions_portfolio_all_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(exc),
            )
            return None
        return PositionsSnapshot(
            positions=list(positions),
            source=PositionsSource.PORTFOLIO_POSITIONS,
        )

    def _from_portfolio_positions_open(self) -> PositionsSnapshot | None:
        if self._portfolio is None or not hasattr(self._portfolio, "positions_open"):
            return None
        try:
            positions = self._portfolio.positions_open()
        except Exception as exc:
            self._log.debug(
                "ml_positions_portfolio_open_failed",
                strategy_id=self._strategy_id,
                exc_info=True,
                error=str(exc),
            )
            return None
        return PositionsSnapshot(
            positions=list(positions),
            source=PositionsSource.PORTFOLIO_POSITIONS_OPEN,
        )

    def _from_portfolio_net(
        self,
        instrument_id: InstrumentId | None,
    ) -> PositionsSnapshot | None:
        if instrument_id is None:
            return None
        if self._portfolio is None or not hasattr(self._portfolio, "net_position"):
            return None
        try:
            net_position = self._portfolio.net_position(instrument_id)
        except Exception as exc:
            self._log.debug(
                "ml_positions_portfolio_net_failed",
                strategy_id=self._strategy_id,
                instrument=str(instrument_id),
                exc_info=True,
                error=str(exc),
            )
            return None
        net_qty = self._to_decimal(net_position)
        if net_qty == Decimal("0"):
            positions: list[PositionViewProtocol] = []
        else:
            positions = [_NetPositionView(instrument_id=instrument_id, net_qty=net_qty)]
        return PositionsSnapshot(positions=positions, source=PositionsSource.PORTFOLIO_NET)

    def _emit_fallback_metric(self, level: str) -> None:
        try:
            get_counter(
                "ml_fallback_activations_total",
                "Fallback activations",
                labelnames=("component", "level"),
            ).labels(component="positions_provider", level=level).inc()
        except Exception as exc:
            self._log.debug(
                "ml_positions_fallback_metric_failed",
                strategy_id=self._strategy_id,
                level=level,
                exc_info=True,
                error=str(exc),
            )

    def _emit_snapshot_metrics(self, snapshot: PositionsSnapshot) -> None:
        positions_count = len(snapshot.positions)
        source = snapshot.source.value
        try:
            get_counter(
                "ml_positions_snapshot_total",
                "Total positions snapshots",
                labelnames=("source",),
            ).labels(source=source).inc()
            if positions_count == 0:
                get_counter(
                    "ml_positions_snapshot_empty_total",
                    "Empty positions snapshots",
                    labelnames=("source",),
                ).labels(source=source).inc()
        except Exception as exc:
            self._log.debug(
                "ml_positions_snapshot_metric_failed",
                strategy_id=self._strategy_id,
                source=source,
                exc_info=True,
                error=str(exc),
            )

    def _emit_degraded_metric(self, reason: str) -> None:
        try:
            get_counter(
                "ml_positions_checks_degraded_total",
                "Positions readiness checks degraded",
                labelnames=("reason",),
            ).labels(reason=reason).inc()
        except Exception as exc:
            self._log.debug(
                "ml_positions_health_metric_failed",
                strategy_id=self._strategy_id,
                reason=reason,
                exc_info=True,
                error=str(exc),
            )

    @staticmethod
    def _to_decimal(value: object) -> Decimal:
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")


__all__ = [
    "CachePositionsProtocol",
    "NautilusPositionsProvider",
    "PortfolioPositionsProtocol",
]
