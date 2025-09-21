"""
Tasks for backfilling recent OHLCV bars via Databento.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ml.data.ingest.api import ensure_service
from ml.data.ingest.policy import DatabentoCoveragePolicy
from ml.data.loaders.ohlcv_recent import OhlcvRecentBackfillConfig
from ml.data.loaders.ohlcv_recent import OhlcvRecentBackfillResult
from ml.data.loaders.ohlcv_recent import SymbolBackfillStatus
from ml.data.loaders.ohlcv_recent import backfill_recent_ohlcv as _backfill_recent_ohlcv


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class BackfillRecentOhlcvTaskConfig:
    """
    Arguments accepted by :func:`backfill_recent_ohlcv`.
    """

    data_dir: Path
    symbols: Sequence[str] | None = None
    tier: int | None = None
    start: datetime | None = None
    end: datetime | None = None
    lookback_days: int = 14


def backfill_recent_ohlcv(
    config: BackfillRecentOhlcvTaskConfig,
) -> OhlcvRecentBackfillResult:
    """
    Run a recent OHLCV backfill and return the aggregated result.
    """
    service = ensure_service()
    LOGGER.info("Resolved Databento ingestion service for OHLCV backfill")
    policy = DatabentoCoveragePolicy.from_env()
    domain_config = OhlcvRecentBackfillConfig(
        data_dir=config.data_dir,
        symbols=config.symbols,
        tier=config.tier,
        start=config.start,
        end=config.end,
        lookback_days=config.lookback_days,
    )
    return _backfill_recent_ohlcv(
        domain_config,
        service=service,
        policy=policy,
    )


__all__ = [
    "BackfillRecentOhlcvTaskConfig",
    "OhlcvRecentBackfillResult",
    "SymbolBackfillStatus",
    "backfill_recent_ohlcv",
]
