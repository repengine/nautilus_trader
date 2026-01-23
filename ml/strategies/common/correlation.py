"""
Correlation provider contracts for ML strategy risk checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import InstrumentId


@dataclass(frozen=True)
class CorrelationSnapshot:
    """
    Normalized correlation value and timestamp.

    Parameters
    ----------
    value : float
        Correlation coefficient.
    ts_event : int | None
        Event timestamp in nanoseconds for the correlation snapshot.
    source : str | None
        Optional source label for the correlation data.

    Examples
    --------
    >>> snapshot = CorrelationSnapshot(value=0.8, ts_event=1_700_000_000_000_000_000)
    >>> snapshot.value
    0.8

    """

    value: float
    ts_event: int | None
    source: str | None = None


@runtime_checkable
class CorrelationProviderProtocol(Protocol):
    """
    Protocol for supplying correlation snapshots between instruments.
    """

    def get_correlation_snapshot(
        self,
        inst1: InstrumentId,
        inst2: InstrumentId,
    ) -> CorrelationSnapshot | None:
        """
        Return a correlation snapshot between two instruments.

        Parameters
        ----------
        inst1 : InstrumentId
            First instrument.
        inst2 : InstrumentId
            Second instrument.

        Returns
        -------
        CorrelationSnapshot | None
            Correlation snapshot or None when unavailable.

        """
        ...


__all__ = ["CorrelationProviderProtocol", "CorrelationSnapshot"]
