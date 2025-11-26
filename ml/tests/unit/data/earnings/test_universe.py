from __future__ import annotations

from ml.config.earnings_ingestion import EarningsIngestionConfig
from ml.data.earnings.universe import resolve_ingestion_universe


def test_resolve_universe_strips_exchange_suffix_from_fallback() -> None:
    cfg = EarningsIngestionConfig(
        postgres_dsn="postgresql://unused",
        universe_mode="tier1_full",
        fallback_symbols=("AAPL.XNAS", "SPY.NYSE"),
    )

    universe = resolve_ingestion_universe(cfg)

    assert universe.source == "tier1_full"
    assert universe.tickers == ("AAPL", "SPY")


def test_resolve_universe_strips_exchange_suffix_from_overrides() -> None:
    cfg = EarningsIngestionConfig(
        postgres_dsn="postgresql://unused",
        universe_mode="postgres",
        override_symbols=("MSFT.XNAS", "msft.xnas"),
    )

    universe = resolve_ingestion_universe(cfg)

    assert universe.source == "override"
    assert universe.tickers == ("MSFT",)
