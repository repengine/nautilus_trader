"""
Tests for correlation provider integration in RiskManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import time

import pytest

from ml.config.base import CorrelationDataConfig
from ml.config.base import PositionsSource
from ml.strategies.common.correlation import CorrelationSnapshot
from ml.strategies.common.positions import PositionsHealthStatus
from ml.strategies.common.positions import PositionsSnapshot
from ml.strategies.risk import RiskConfig
from ml.strategies.risk import RiskManager
from nautilus_trader.model.identifiers import InstrumentId


@dataclass(frozen=True)
class DummyPosition:
    """
    Minimal position view for correlation checks.
    """

    instrument_id: InstrumentId
    is_open: bool = True

    def signed_decimal_qty(self) -> Decimal:
        return Decimal("1.0")


class DummyPositionsProvider:
    """
    Positions provider returning a fixed snapshot.
    """

    def __init__(self, positions: list[DummyPosition]) -> None:
        self._snapshot = PositionsSnapshot(
            positions=positions,
            source=PositionsSource.CACHE_OPEN,
        )

    def get_positions_snapshot(
        self,
        *,
        instrument_id: InstrumentId | None = None,
        require_full_list: bool = False,
    ) -> PositionsSnapshot:
        del instrument_id, require_full_list
        return self._snapshot

    def check_positions_ready(
        self,
        *,
        instrument_id: InstrumentId | None = None,
        require_full_list: bool = False,
        require_positions: bool = False,
    ) -> PositionsHealthStatus:
        del instrument_id, require_full_list, require_positions
        return PositionsHealthStatus(
            ready=True,
            degraded=False,
            source=self._snapshot.source,
            reason=None,
            positions_count=len(self._snapshot.positions),
        )


class DummyCorrelationProvider:
    """
    Correlation provider returning a fixed snapshot.
    """

    def __init__(self, snapshot: CorrelationSnapshot | None) -> None:
        self._snapshot = snapshot
        self.last_pair: tuple[InstrumentId, InstrumentId] | None = None

    def get_correlation_snapshot(
        self,
        inst1: InstrumentId,
        inst2: InstrumentId,
    ) -> CorrelationSnapshot | None:
        self.last_pair = (inst1, inst2)
        return self._snapshot


class DummyPortfolio:
    """
    Portfolio stub for correlation checks.
    """

    def positions(self) -> list[DummyPosition]:
        return []

    def positions_open(self) -> list[DummyPosition]:
        return []


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


def test_risk_rejects_when_correlation_high_and_fresh() -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    inst_b = InstrumentId.from_str("BBB.SIM")
    snapshot = CorrelationSnapshot(
        value=0.9,
        ts_event=time.time_ns(),
        source="provider",
    )
    provider = DummyCorrelationProvider(snapshot)
    positions_provider = DummyPositionsProvider([DummyPosition(inst_b)])
    config = RiskConfig(
        correlation_threshold=0.7,
        max_correlated_positions=1,
        correlation_data_config=CorrelationDataConfig(
            max_age_seconds=60,
            fallback_value=0.0,
        ),
    )
    manager = RiskManager(
        config,
        positions_provider=positions_provider,
        correlation_provider=provider,
    )

    assert manager._check_correlation_limits(inst_a, DummyPortfolio()) is False


def test_risk_uses_fallback_when_correlation_stale(
    isolated_prometheus_registry: object,
) -> None:
    inst_a = InstrumentId.from_str("AAA.SIM")
    inst_b = InstrumentId.from_str("BBB.SIM")
    max_age = 1
    stale_ts = time.time_ns() - (max_age + 5) * 1_000_000_000
    snapshot = CorrelationSnapshot(
        value=0.95,
        ts_event=stale_ts,
        source="provider",
    )
    provider = DummyCorrelationProvider(snapshot)
    positions_provider = DummyPositionsProvider([DummyPosition(inst_b)])
    config = RiskConfig(
        correlation_threshold=0.7,
        max_correlated_positions=1,
        correlation_data_config=CorrelationDataConfig(
            max_age_seconds=max_age,
            fallback_value=0.0,
        ),
    )
    manager = RiskManager(
        config,
        positions_provider=positions_provider,
        correlation_provider=provider,
    )

    assert manager._check_correlation_limits(inst_a, DummyPortfolio()) is True

    metric_value = isolated_prometheus_registry.registry.get_sample_value(
        "ml_risk_correlation_degraded_total",
        labels={"reason": "stale"},
    )
    assert metric_value == 1.0
