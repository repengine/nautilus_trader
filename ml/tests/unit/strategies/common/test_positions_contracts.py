"""
Tests for positions provider contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ml.config.base import PositionsSource
from ml.strategies.common.positions import PositionViewProtocol
from ml.strategies.common.positions import PositionsHealthStatus
from ml.strategies.common.positions import PositionsProviderProtocol
from ml.strategies.common.positions import PositionsSnapshot
from ml.strategies.common.positions import build_positions_metadata
from nautilus_trader.model.identifiers import InstrumentId


@dataclass(frozen=True)
class DummyPosition:
    """
    Minimal position stub for protocol checks.
    """

    instrument_id: InstrumentId
    is_open: bool = True

    def signed_decimal_qty(self) -> Decimal:
        """
        Return signed quantity for the dummy position.
        """
        return Decimal("1.0")


class DummyPositionsProvider:
    """
    Simple provider stub for protocol checks.
    """

    def __init__(self, snapshot: PositionsSnapshot) -> None:
        """
        Initialize the provider with a snapshot.
        """
        self._snapshot = snapshot

    def get_positions_snapshot(
        self,
        *,
        instrument_id: InstrumentId | None = None,
        require_full_list: bool = False,
    ) -> PositionsSnapshot:
        """
        Return the stored positions snapshot.
        """
        del instrument_id, require_full_list
        return self._snapshot

    def check_positions_ready(
        self,
        *,
        instrument_id: InstrumentId | None = None,
        require_full_list: bool = False,
        require_positions: bool = False,
    ) -> PositionsHealthStatus:
        """
        Return a healthy status for protocol checks.
        """
        del instrument_id, require_full_list, require_positions
        return PositionsHealthStatus(
            ready=True,
            degraded=False,
            source=self._snapshot.source,
            reason=None,
            positions_count=len(self._snapshot.positions),
        )


def test_positions_snapshot_tracks_source_and_positions() -> None:
    """
    Test PositionsSnapshot retains source and positions.
    """
    instrument_id = InstrumentId.from_str("EURUSD.SIM")
    position = DummyPosition(instrument_id=instrument_id)
    snapshot = PositionsSnapshot(
        positions=[position],
        source=PositionsSource.CACHE_OPEN,
    )

    assert snapshot.source.value == "cache_positions_open"
    assert snapshot.positions == [position]


def test_positions_health_status_tracks_flags() -> None:
    """
    Test PositionsHealthStatus retains readiness flags.
    """
    status = PositionsHealthStatus(
        ready=True,
        degraded=False,
        source=PositionsSource.CACHE_OPEN,
        reason=None,
        positions_count=1,
    )

    assert status.ready is True
    assert status.degraded is False
    assert status.positions_count == 1


def test_build_positions_metadata_includes_source_and_flags() -> None:
    """
    Test positions metadata builder includes readiness fields.
    """
    status = PositionsHealthStatus(
        ready=False,
        degraded=True,
        source=PositionsSource.PORTFOLIO_NET,
        reason="net_position_only",
        positions_count=1,
    )

    metadata = build_positions_metadata(status)

    assert metadata is not None
    assert metadata["source"] == PositionsSource.PORTFOLIO_NET.value
    assert metadata["ready"] is False
    assert metadata["degraded"] is True
    assert metadata["reason"] == "net_position_only"
    assert metadata["count"] == 1


def test_positions_provider_protocol_matches_runtime() -> None:
    """
    Test runtime protocol checks for provider and position views.
    """
    instrument_id = InstrumentId.from_str("EURUSD.SIM")
    position = DummyPosition(instrument_id=instrument_id)
    snapshot = PositionsSnapshot(
        positions=[position],
        source=PositionsSource.CACHE_OPEN,
    )
    provider = DummyPositionsProvider(snapshot)

    assert isinstance(position, PositionViewProtocol)
    assert isinstance(provider, PositionsProviderProtocol)
    assert provider.get_positions_snapshot(require_full_list=True) == snapshot
