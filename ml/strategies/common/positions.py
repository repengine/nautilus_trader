"""
Positions provider contracts for ML strategies.

Defines protocol-first interfaces for reading open positions with minimal
assumptions about cache or portfolio APIs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ml.config.base import PositionsSource


if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import InstrumentId


@runtime_checkable
class PositionViewProtocol(Protocol):
    """
    Minimal position interface required by sizing and risk checks.

    Attributes
    ----------
    instrument_id : InstrumentId
        Identifier for the instrument this position belongs to.
    is_open : bool
        Whether the position is open.

    """

    instrument_id: InstrumentId
    is_open: bool

    def signed_decimal_qty(self) -> Decimal:
        """
        Return signed decimal quantity for the position.

        Returns
        -------
        Decimal
            Signed position quantity.

        """
        ...


@dataclass(frozen=True)
class PositionsSnapshot:
    """
    Normalized snapshot of open positions and source metadata.

    Parameters
    ----------
    positions : Sequence[PositionViewProtocol]
        Open positions available from the provider.
    source : PositionsSource
        Source used to populate the snapshot.

    Examples
    --------
    >>> snapshot = PositionsSnapshot(
    ...     positions=[position],
    ...     source=PositionsSource.CACHE_OPEN,
    ... )
    >>> len(snapshot.positions)
    1

    """

    positions: Sequence[PositionViewProtocol]
    source: PositionsSource


@dataclass(frozen=True)
class PositionsHealthStatus:
    """
    Health status summary for positions readiness checks.

    Parameters
    ----------
    ready : bool
        Whether positions access is ready for live trading requirements.
    degraded : bool
        Whether positions access is degraded (limited or unavailable).
    source : PositionsSource
        Positions source used during the check.
    reason : str | None
        Optional reason for degraded status.
    positions_count : int
        Count of positions returned in the snapshot.

    Examples
    --------
    >>> status = PositionsHealthStatus(
    ...     ready=True,
    ...     degraded=False,
    ...     source=PositionsSource.CACHE_OPEN,
    ...     reason=None,
    ...     positions_count=0,
    ... )
    >>> status.ready
    True

    """

    ready: bool
    degraded: bool
    source: PositionsSource
    reason: str | None
    positions_count: int


PositionsMetadata = dict[str, object]


def build_positions_metadata(
    health: PositionsHealthStatus | None,
) -> PositionsMetadata | None:
    """
    Build a metadata payload from positions health status.

    Parameters
    ----------
    health : PositionsHealthStatus | None
        Positions health status snapshot.

    Returns
    -------
    dict[str, object] | None
        Metadata payload or None when no health is available.

    Examples
    --------
    >>> metadata = build_positions_metadata(status)
    >>> metadata["source"]
    'cache_positions_open'

    """
    if health is None:
        return None
    return {
        "source": health.source.value,
        "ready": health.ready,
        "degraded": health.degraded,
        "reason": health.reason,
        "count": health.positions_count,
    }


@runtime_checkable
class PositionsProviderProtocol(Protocol):
    """
    Protocol for producing normalized position snapshots.
    """

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
            Optional instrument filter or requirement for per-instrument fallbacks.
        require_full_list : bool, default False
            When True, ignore instrument filters for list-based sources.

        Returns
        -------
        PositionsSnapshot
            Normalized positions snapshot with source metadata.

        """
        ...

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
            Optional instrument filter or requirement for per-instrument fallbacks.
        require_full_list : bool, default False
            Require a full positions list instead of a per-instrument fallback.
        require_positions : bool, default False
            Whether positions access must be available for readiness.

        Returns
        -------
        PositionsHealthStatus
            Status summary for positions readiness.

        Examples
        --------
        >>> status = provider.check_positions_ready(require_positions=True)
        >>> status.ready
        True

        """
        ...


__all__ = [
    "PositionViewProtocol",
    "PositionsHealthStatus",
    "PositionsMetadata",
    "PositionsProviderProtocol",
    "PositionsSnapshot",
    "build_positions_metadata",
]
