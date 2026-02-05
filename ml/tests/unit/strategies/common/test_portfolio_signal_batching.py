from __future__ import annotations

from dataclasses import dataclass

from ml.actors.base import MLSignal
from ml.strategies.common.portfolio_signal_batching import PortfolioSignalBatcher
from ml.strategies.portfolio import PortfolioBatchingConfig
from nautilus_trader.model.identifiers import InstrumentId


@dataclass(frozen=True)
class _DummyPortfolio:
    id: str


def _signal(instrument_id: InstrumentId, ts_event: int) -> MLSignal:
    return MLSignal(
        instrument_id=instrument_id,
        model_id="model",
        prediction=0.0,
        confidence=0.8,
        metadata={},
        ts_event=ts_event,
    )


def test_batches_across_strategies_with_shared_portfolio() -> None:
    batcher = PortfolioSignalBatcher()
    config = PortfolioBatchingConfig(
        enabled=True,
        window_ms=1_000,
        min_batch_size=2,
        max_batch_size=10,
    )
    portfolio = _DummyPortfolio(id="shared-portfolio")
    signal_a = _signal(InstrumentId.from_str("AAA.SIM"), ts_event=1)
    signal_b = _signal(InstrumentId.from_str("BBB.SIM"), ts_event=2)

    batcher.update_and_get_batch(
        signal_a,
        config=config,
        portfolio=portfolio,
        strategy_id="strategy-a",
    )
    batch = batcher.update_and_get_batch(
        signal_b,
        config=config,
        portfolio=portfolio,
        strategy_id="strategy-b",
    )

    assert {signal.instrument_id for signal in batch} == {
        signal_a.instrument_id,
        signal_b.instrument_id,
    }


def test_batches_separate_by_strategy_when_portfolio_missing() -> None:
    batcher = PortfolioSignalBatcher()
    config = PortfolioBatchingConfig(
        enabled=True,
        window_ms=1_000,
        min_batch_size=2,
        max_batch_size=10,
    )
    signal_a = _signal(InstrumentId.from_str("AAA.SIM"), ts_event=1)
    signal_b = _signal(InstrumentId.from_str("BBB.SIM"), ts_event=2)

    batcher.update_and_get_batch(
        signal_a,
        config=config,
        portfolio=None,
        strategy_id="strategy-a",
    )
    batch = batcher.update_and_get_batch(
        signal_b,
        config=config,
        portfolio=None,
        strategy_id="strategy-b",
    )

    assert {signal.instrument_id for signal in batch} == {signal_b.instrument_id}
