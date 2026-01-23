"""
Tests for NautilusPositionsProvider fallback behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from ml.config.base import PositionsConfig
from ml.config.base import PositionsSource
from ml.strategies.common.positions_provider import NautilusPositionsProvider
from nautilus_trader.model.identifiers import InstrumentId


@dataclass(frozen=True)
class DummyPosition:
    """
    Minimal position view for provider tests.
    """

    instrument_id: InstrumentId
    is_open: bool = True

    def signed_decimal_qty(self) -> Decimal:
        return Decimal("1.0")


class DummyCache:
    """
    Cache stub for provider tests.
    """

    def __init__(self, positions: list[DummyPosition], *, raise_open: bool = False) -> None:
        self._positions = positions
        self._raise_open = raise_open
        self.last_instrument_id: InstrumentId | None = None

    def positions_open(
        self,
        venue: object | None = None,
        instrument_id: InstrumentId | None = None,
    ) -> list[DummyPosition]:
        del venue
        self.last_instrument_id = instrument_id
        if self._raise_open:
            raise RuntimeError("positions_open failed")
        if instrument_id is None:
            return list(self._positions)
        return [pos for pos in self._positions if pos.instrument_id == instrument_id]

    def positions(
        self,
        venue: object | None = None,
        instrument_id: InstrumentId | None = None,
    ) -> list[DummyPosition]:
        del venue
        if instrument_id is None:
            return list(self._positions)
        return [pos for pos in self._positions if pos.instrument_id == instrument_id]


class DummyPortfolio:
    """
    Portfolio stub for net position fallback.
    """

    def __init__(self, net_value: Decimal) -> None:
        self._net_value = net_value

    def net_position(self, instrument_id: InstrumentId) -> Decimal:
        del instrument_id
        return self._net_value


class DummyPortfolioPositions:
    """
    Portfolio stub for positions list fallbacks.
    """

    def __init__(
        self,
        positions: list[DummyPosition],
        *,
        raise_positions: bool = False,
        raise_open: bool = False,
    ) -> None:
        self._positions = positions
        self._raise_positions = raise_positions
        self._raise_open = raise_open

    def positions(self) -> list[DummyPosition]:
        if self._raise_positions:
            raise RuntimeError("positions failed")
        return list(self._positions)

    def positions_open(self) -> list[DummyPosition]:
        if self._raise_open:
            raise RuntimeError("positions_open failed")
        return list(self._positions)


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


def test_provider_uses_cache_open_and_respects_full_list() -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    inst_b = InstrumentId.from_str("BBB.SIM")
    positions = [DummyPosition(inst_a), DummyPosition(inst_b)]
    cache = DummyCache(positions)
    config = PositionsConfig(source_priority=[PositionsSource.CACHE_OPEN])

    provider = NautilusPositionsProvider(cache=cache, portfolio=None, config=config)
    snapshot = provider.get_positions_snapshot(instrument_id=inst_a)

    assert snapshot.source is PositionsSource.CACHE_OPEN
    assert [pos.instrument_id for pos in snapshot.positions] == [inst_a]
    assert cache.last_instrument_id == inst_a

    full_snapshot = provider.get_positions_snapshot(
        instrument_id=inst_a,
        require_full_list=True,
    )

    assert {pos.instrument_id for pos in full_snapshot.positions} == {inst_a, inst_b}
    assert cache.last_instrument_id is None


def test_provider_falls_back_to_cache_all_when_open_fails() -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    positions = [DummyPosition(inst_a)]
    cache = DummyCache(positions, raise_open=True)
    config = PositionsConfig(
        source_priority=[
            PositionsSource.CACHE_OPEN,
            PositionsSource.CACHE_ALL,
        ],
    )

    provider = NautilusPositionsProvider(cache=cache, portfolio=None, config=config)
    snapshot = provider.get_positions_snapshot()

    assert snapshot.source is PositionsSource.CACHE_ALL
    assert snapshot.positions == positions


def test_provider_falls_back_to_portfolio_positions_when_cache_missing(
    isolated_prometheus_registry: Any,
) -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    positions = [DummyPosition(inst_a)]
    portfolio = DummyPortfolioPositions(positions)
    config = PositionsConfig(
        source_priority=[
            PositionsSource.CACHE_OPEN,
            PositionsSource.PORTFOLIO_POSITIONS,
        ],
    )

    provider = NautilusPositionsProvider(cache=None, portfolio=portfolio, config=config)
    snapshot = provider.get_positions_snapshot()

    assert snapshot.source is PositionsSource.PORTFOLIO_POSITIONS
    assert snapshot.positions == positions

    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "ml_fallback_activations_total",
        labels={"component": "positions_provider", "level": "portfolio_positions"},
    )
    assert metric_value == 1.0


def test_provider_falls_back_to_portfolio_positions_open_when_positions_fail(
    isolated_prometheus_registry: Any,
) -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    positions = [DummyPosition(inst_a)]
    portfolio = DummyPortfolioPositions(positions, raise_positions=True)
    config = PositionsConfig(
        source_priority=[
            PositionsSource.CACHE_OPEN,
            PositionsSource.PORTFOLIO_POSITIONS,
            PositionsSource.PORTFOLIO_POSITIONS_OPEN,
        ],
    )

    provider = NautilusPositionsProvider(cache=None, portfolio=portfolio, config=config)
    snapshot = provider.get_positions_snapshot()

    assert snapshot.source is PositionsSource.PORTFOLIO_POSITIONS_OPEN
    assert snapshot.positions == positions

    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "ml_fallback_activations_total",
        labels={"component": "positions_provider", "level": "portfolio_positions_open"},
    )
    assert metric_value == 1.0


def test_provider_falls_back_to_portfolio_net_when_cache_missing(
    isolated_prometheus_registry: Any,
) -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    portfolio = DummyPortfolio(Decimal("2.5"))
    config = PositionsConfig(
        source_priority=[
            PositionsSource.CACHE_OPEN,
            PositionsSource.PORTFOLIO_NET,
        ],
    )

    provider = NautilusPositionsProvider(cache=None, portfolio=portfolio, config=config)
    snapshot = provider.get_positions_snapshot(instrument_id=inst_a)

    assert snapshot.source is PositionsSource.PORTFOLIO_NET
    assert snapshot.positions

    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "ml_fallback_activations_total",
        labels={"component": "positions_provider", "level": "portfolio_net_position"},
    )
    assert metric_value == 1.0


def test_provider_uses_portfolio_net_position_when_configured() -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    portfolio = DummyPortfolio(Decimal("2.5"))
    config = PositionsConfig(source_priority=[PositionsSource.PORTFOLIO_NET])

    provider = NautilusPositionsProvider(cache=None, portfolio=portfolio, config=config)
    snapshot = provider.get_positions_snapshot(instrument_id=inst_a)

    assert snapshot.source is PositionsSource.PORTFOLIO_NET
    assert len(snapshot.positions) == 1
    position = snapshot.positions[0]
    assert position.instrument_id == inst_a
    assert position.signed_decimal_qty() == Decimal("2.5")
    assert position.is_open is True

    portfolio_zero = DummyPortfolio(Decimal("0"))
    provider_zero = NautilusPositionsProvider(
        cache=None,
        portfolio=portfolio_zero,
        config=config,
    )
    snapshot_zero = provider_zero.get_positions_snapshot(instrument_id=inst_a)

    assert snapshot_zero.positions == []


def test_provider_records_snapshot_metrics(isolated_prometheus_registry: Any) -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    cache = DummyCache([DummyPosition(inst_a)])
    config = PositionsConfig(source_priority=[PositionsSource.CACHE_OPEN])
    provider = NautilusPositionsProvider(cache=cache, portfolio=None, config=config)

    provider.get_positions_snapshot()
    provider_empty = NautilusPositionsProvider(cache=DummyCache([]), portfolio=None, config=config)
    provider_empty.get_positions_snapshot()

    registry = isolated_prometheus_registry.registry
    total_value = registry.get_sample_value(
        "ml_positions_snapshot_total",
        labels={"source": "cache_positions_open"},
    )
    empty_value = registry.get_sample_value(
        "ml_positions_snapshot_empty_total",
        labels={"source": "cache_positions_open"},
    )

    assert total_value == 2.0
    assert empty_value == 1.0


def test_provider_health_check_reports_unavailable_positions(
    isolated_prometheus_registry: Any,
) -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    config = PositionsConfig(source_priority=[PositionsSource.CACHE_OPEN])
    provider = NautilusPositionsProvider(cache=None, portfolio=None, config=config)

    health = provider.check_positions_ready(
        instrument_id=inst_a,
        require_full_list=True,
        require_positions=True,
    )

    assert health.ready is False
    assert health.degraded is True
    assert health.reason == "positions_unavailable"
    assert health.positions_count == 0

    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "ml_positions_checks_degraded_total",
        labels={"reason": "positions_unavailable"},
    )
    assert metric_value == 1.0


def test_provider_health_check_allows_empty_positions_when_source_available(
    isolated_prometheus_registry: Any,
) -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    cache = DummyCache([])
    config = PositionsConfig(source_priority=[PositionsSource.CACHE_OPEN])
    provider = NautilusPositionsProvider(cache=cache, portfolio=None, config=config)

    health = provider.check_positions_ready(
        instrument_id=inst_a,
        require_full_list=True,
        require_positions=True,
    )

    assert health.ready is True
    assert health.degraded is False
    assert health.reason is None
    assert health.positions_count == 0

    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "ml_positions_checks_degraded_total",
        labels={"reason": "positions_unavailable"},
    )
    assert metric_value is None


def test_provider_health_check_flags_net_position_only(
    isolated_prometheus_registry: Any,
) -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    portfolio = DummyPortfolio(Decimal("2.5"))
    config = PositionsConfig(source_priority=[PositionsSource.PORTFOLIO_NET])
    provider = NautilusPositionsProvider(cache=None, portfolio=portfolio, config=config)

    health = provider.check_positions_ready(
        instrument_id=inst_a,
        require_full_list=True,
        require_positions=True,
    )

    assert health.ready is False
    assert health.degraded is True
    assert health.reason == "net_position_only"
    assert health.source is PositionsSource.PORTFOLIO_NET

    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "ml_positions_checks_degraded_total",
        labels={"reason": "net_position_only"},
    )
    assert metric_value == 1.0
