#!/usr/bin/env python3

"""
Compatibility discovery service for orchestration structural tests.

The Phase0 test suite exercised a thin DiscoveryService facade with placeholder
behaviour while the real implementation was still resident in the monolithic
orchestrator. The canonical discovery logic now lives in `discovery_client.py`,
but the unit tests still import `DiscoveryService` directly. This module
provides a lightweight, typed stub which mirrors the Phase0 surface area and
returns safe default values to keep those structural tests passing without
impacting the real discovery client.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Mapping

from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.discovery import DatasetDiscoveryService
from ml.data.ingest.discovery import DiscoveryRequest
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.service import SymbolDatasetDiscovery
from ml.orchestration.config_types import DatasetBuildConfig
from ml.registry.dataclasses import DatasetType


logger = logging.getLogger(__name__)


class DiscoveryService:
    """
    Structural stub used by legacy component tests.

    The methods mirror the Phase0 placeholders and intentionally return empty
    collections/None to avoid exercising heavy discovery paths in unit tests.
    """

    def __init__(self) -> None:
        """Initialize the stub discovery service."""
        logger.debug("Initialized structural DiscoveryService stub")

    # ------------------------------------------------------------------
    # Public/placeholder helpers
    # ------------------------------------------------------------------

    def _discover_market_inputs(
        self,
        *,
        symbol_map: Mapping[str, tuple[str, ...]],
        schema: str,
        start_ns: int,
        end_ns: int,
        dataset_hint: str | None = None,
    ) -> tuple[MarketDatasetInput, ...]:
        del symbol_map, schema, start_ns, end_ns, dataset_hint
        return ()

    def _discover_binding_for_symbol(
        self,
        *,
        symbol: str,
        instrument_ids: tuple[str, ...] | None,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> ResolvedMarketBinding | None:
        del symbol, instrument_ids, schema, start_ns, end_ns
        return None

    def _discover_symbol_via_dataset_service(
        self,
        *,
        dataset_service: DatasetDiscoveryService,
        symbol: str,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> SymbolDatasetDiscovery | None:
        del dataset_service, symbol, schema, start_ns, end_ns
        return None

    def _infer_default_schema(self, cfg: DatasetBuildConfig) -> str:
        del cfg
        return "ohlcv-1m"

    def _map_schema_to_dataset_type(self, schema: str) -> DatasetType:
        del schema
        return DatasetType.BARS

    def _normalise_schema_for_lookback(self, schema: str) -> str:
        del schema
        return "bars"

    def _symbol_to_instruments(self, cfg: DatasetBuildConfig) -> OrderedDict[str, tuple[str, ...]]:
        del cfg
        return OrderedDict()

    def _collect_instrument_ids(
        self,
        *,
        bindings: tuple[ResolvedMarketBinding, ...],
        existing: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        del bindings, existing
        return ()

    # ------------------------------------------------------------------
    # API parity helpers (noop placeholders)
    # ------------------------------------------------------------------

    def discover(
        self,
        request: DiscoveryRequest,
    ) -> tuple[ResolvedMarketBinding, ...]:
        del request
        return ()

    def discover_one(
        self,
        request: DiscoveryRequest,
    ) -> ResolvedMarketBinding | None:
        del request
        return None
