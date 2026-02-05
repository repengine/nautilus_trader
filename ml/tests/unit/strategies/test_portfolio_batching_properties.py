from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import pytest

from ml.actors.base import MLSignal
from ml.strategies.common.portfolio_signal_batching import PortfolioSignalBatcher
from ml.strategies.portfolio import PortfolioBatchingConfig
from ml.strategies.portfolio import PortfolioConfig
from ml.strategies.portfolio import PortfolioManager
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


@dataclass(frozen=True)
class _DummyPortfolio:
    id: str = "portfolio-1"


def _instrument_pool() -> list[InstrumentId]:
    return [
        InstrumentId(Symbol("AAA"), Venue("SIM")),
        InstrumentId(Symbol("BBB"), Venue("SIM")),
        InstrumentId(Symbol("CCC"), Venue("SIM")),
        InstrumentId(Symbol("DDD"), Venue("SIM")),
        InstrumentId(Symbol("EEE"), Venue("SIM")),
        InstrumentId(Symbol("FFF"), Venue("SIM")),
        InstrumentId(Symbol("GGG"), Venue("SIM")),
        InstrumentId(Symbol("HHH"), Venue("SIM")),
    ]


@st.composite
def _batched_signals(
    draw: st.DrawFn,
) -> tuple[list[MLSignal], PortfolioBatchingConfig]:
    instruments = draw(
        st.lists(
            st.sampled_from(_instrument_pool()),
            min_size=2,
            max_size=8,
            unique=True,
        ),
    )
    confidences = draw(
        st.lists(
            st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=len(instruments),
            max_size=len(instruments),
        ),
    )
    base_ts = draw(st.integers(min_value=1, max_value=1_000_000))
    max_batch_size = draw(st.integers(min_value=2, max_value=len(instruments)))
    min_batch_size = draw(st.integers(min_value=2, max_value=max_batch_size))

    signals = [
        MLSignal(
            instrument_id=instrument,
            model_id="model",
            prediction=0.0,
            confidence=confidence,
            metadata={},
            ts_event=base_ts + idx,
        )
        for idx, (instrument, confidence) in enumerate(zip(instruments, confidences))
    ]
    config = PortfolioBatchingConfig(
        enabled=True,
        window_ms=10_000,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )
    return signals, config


@settings(max_examples=50)
@given(
    data=_batched_signals(),
    capital=st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
)
def test_batched_allocation_respects_limits_property(
    data: tuple[list[MLSignal], PortfolioBatchingConfig],
    capital: float,
) -> None:
    signals, batching_config = data
    batcher = PortfolioSignalBatcher()
    portfolio = _DummyPortfolio()

    batch: list[MLSignal] = []
    for signal in signals:
        batch = batcher.update_and_get_batch(
            signal,
            config=batching_config,
            portfolio=portfolio,
            strategy_id="strategy-1",
        )

    assert len(batch) >= batching_config.min_batch_size
    assert len(batch) <= batching_config.max_batch_size

    manager = PortfolioManager(
        PortfolioConfig(
            allocation_method="equal",
            use_correlation_adjustment=False,
            max_positions=50,
            min_position_weight=0.01,
            max_position_weight=1.0,
        ),
    )
    allocations = manager.allocate_signals(batch, capital)

    assert allocations
    assert set(allocations.keys()) == {signal.instrument_id for signal in batch}

    total_alloc = sum(allocations.values())
    assert total_alloc <= capital + 1e-6

    min_alloc = capital * manager.config.min_position_weight
    max_alloc = capital * manager.config.max_position_weight
    for value in allocations.values():
        assert value >= min_alloc - 1e-6
        assert value <= max_alloc + 1e-6
