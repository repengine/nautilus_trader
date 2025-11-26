#!/usr/bin/env python3
"""
Helpers for resolving the ingestion universe for earnings pipelines.

The universe can be sourced from live PostgreSQL metadata (preferred) or fall
back to the static Tier-1 universe when discovery fails.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import text

from ml.common.db_utils import get_or_create_engine
from ml.config.earnings_ingestion import EarningsIngestionConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedUniverse:
    """Container describing the resolved ingestion universe."""

    tickers: tuple[str, ...]
    source: str


def resolve_ingestion_universe(config: EarningsIngestionConfig) -> ResolvedUniverse:
    """
    Resolve the ingestion universe according to the supplied configuration.
    """
    override = config.resolved_override()
    if override:
        normalized = _normalize_symbols(override)
        return ResolvedUniverse(tickers=normalized, source="override")

    mode = config.universe_mode.lower().strip()
    if mode == "postgres":
        discovered = _discover_from_postgres(config)
        if discovered is not None:
            return ResolvedUniverse(tickers=discovered, source="postgres")
        logger.warning("Falling back to static universe after PostgreSQL discovery failure")

    if mode in {"tier1_full", "tier1"}:
        normalized = _normalize_symbols(config.fallback_symbols)
        return ResolvedUniverse(tickers=normalized, source="tier1_full")

    # Default fallback when mode is unknown or discovery failed
    fallback = _normalize_symbols(config.fallback_symbols)
    return ResolvedUniverse(tickers=fallback, source="fallback")


def _discover_from_postgres(config: EarningsIngestionConfig) -> tuple[str, ...] | None:
    try:
        engine = get_or_create_engine(config.postgres_dsn)
    except Exception as exc:  # pragma: no cover - connection errors
        logger.warning("Unable to create engine for universe discovery: %s", exc, exc_info=True)
        return None

    query = text(
        """
        SELECT DISTINCT instrument_id
        FROM ml.instrument_metadata
        WHERE instrument_id IS NOT NULL
        """
    )

    instrument_ids: list[str] = []
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            instrument_ids = [
                str(row[0])
                for row in result
                if row[0] is not None
            ]
    except Exception as exc:  # pragma: no cover - discovery errors
        logger.warning("Instrument discovery query failed: %s", exc, exc_info=True)
        return None

    if not instrument_ids:
        return None

    return _normalize_instrument_ids(instrument_ids)


def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for symbol in symbols:
        if not symbol:
            continue
        upper = symbol.upper()
        base = upper.split(".", 1)[0]
        if base:
            normalized.add(base)
    return tuple(sorted(normalized))


def _normalize_instrument_ids(instrument_ids: Iterable[str]) -> tuple[str, ...]:
    tickers: set[str] = set()
    for instrument_id in instrument_ids:
        if not instrument_id:
            continue
        symbol = instrument_id.split(".", 1)[0].upper()
        if symbol:
            tickers.add(symbol)
    return tuple(sorted(tickers))


__all__ = ["ResolvedUniverse", "resolve_ingestion_universe"]
