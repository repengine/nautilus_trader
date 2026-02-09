"""
Data ingestion utilities and services for Nautilus Trader ML.

This package exposes ingestion policy helpers eagerly and resolves loader-backed
task symbols lazily to avoid import cycles during store/data package bootstrap.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any

from ml.data.ingest.subscription import SubscriptionChecker
from ml.data.ingest.subscription import SubscriptionPolicy
from ml.data.ingest.subscription import get_effective_policy
from ml.data.ingest.subscription import get_max_lookback_days


__all__ = [
    "BackfillRecentOhlcvTaskConfig",
    "OhlcvRecentBackfillResult",
    "PopulateAlternativeDataTaskConfig",
    "PopulateL2TaskConfig",
    "PopulateSupplementaryTaskConfig",
    "PopulateYahooDataTaskConfig",
    "SubscriptionChecker",
    "SubscriptionPolicy",
    "SymbolBackfillStatus",
    "backfill_recent_ohlcv",
    "get_effective_policy",
    "get_max_lookback_days",
    "populate_alternative_data_task",
    "populate_l2_efficient",
    "populate_supplementary_data",
    "populate_yahoo_data",
]


def __getattr__(name: str) -> Any:
    """
    Lazily resolve loader-backed exports on first access.
    """
    if name in {"PopulateL2TaskConfig", "populate_l2_efficient"}:
        module = _import_module("ml.data.ingest.l2_efficient")
        return getattr(module, name)
    if name in {"PopulateAlternativeDataTaskConfig", "populate_alternative_data_task"}:
        module = _import_module("ml.data.loaders.alternative")
        return getattr(module, name)
    if name in {
        "BackfillRecentOhlcvTaskConfig",
        "OhlcvRecentBackfillResult",
        "SymbolBackfillStatus",
        "backfill_recent_ohlcv",
    }:
        module = _import_module("ml.data.loaders.ohlcv_recent")
        return getattr(module, name)
    if name in {
        "PopulateSupplementaryTaskConfig",
        "PopulateYahooDataTaskConfig",
        "populate_supplementary_data",
        "populate_yahoo_data",
    }:
        module = _import_module("ml.data.loaders.supplementary")
        return getattr(module, name)
    raise AttributeError(name)
