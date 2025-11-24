#!/usr/bin/env python3

"""
Discovery client for ML pipeline orchestration.

This module provides dataset discovery, service health checks, and availability
queries with error handling and policy enforcement.

This component is extracted from the MLPipelineOrchestrator god class to provide
focused, testable discovery functionality.

"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.config.market_data import MarketDatasetInput


if TYPE_CHECKING:
    from ml.data.ingest.discovery import DatasetDiscoveryService
    from ml.data.ingest.market_bindings import ResolvedMarketBinding
    from ml.data.ingest.service import SymbolDatasetDiscovery


logger = logging.getLogger(__name__)


# ========================================================================
# Protocol Definition
# ========================================================================


class DiscoveryClientProtocol(Protocol):
    """
    Protocol for dataset discovery operations.
    """

    def discover_market_inputs(
        self,
        symbol_map: Mapping[str, tuple[str, ...]],
        schema: str,
        start_ns: int,
        end_ns: int,
        dataset_hint: str | None = None,
    ) -> tuple[MarketDatasetInput, ...]:
        """
        Discover market inputs for given symbols and time range.

        Parameters
        ----------
        symbol_map : Mapping[str, tuple[str, ...]]
            Symbol to instrument IDs mapping
        schema : str
            Data schema (e.g. 'ohlcv-1m', 'tbbo')
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds
        dataset_hint : str | None
            Optional dataset ID hint

        Returns
        -------
        tuple[MarketDatasetInput, ...]
            Discovered market inputs

        """
        ...

    def discover_binding_for_symbol(
        self,
        symbol: str,
        instrument_ids: tuple[str, ...] | None,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> ResolvedMarketBinding | None:
        """
        Discover binding for a specific symbol.

        Parameters
        ----------
        symbol : str
            Symbol to discover
        instrument_ids : tuple[str, ...] | None
            Optional instrument IDs for the symbol
        schema : str
            Data schema
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        ResolvedMarketBinding | None
            Discovered binding or None if not found

        """
        ...


# ========================================================================
# DiscoveryClient Implementation
# ========================================================================


class DiscoveryClient:
    """
    Client for dataset discovery and service health checks.

    Provides high-level discovery operations with error handling,
    policy enforcement, and coverage validation.

    This component is extracted from the MLPipelineOrchestrator god class to
    provide focused, testable discovery functionality.

    """

    def __init__(
        self,
        dataset_discovery: DatasetDiscoveryService | None = None,
        ingestion_service: object | None = None,
    ) -> None:
        """
        Initialize discovery client.

        Parameters
        ----------
        dataset_discovery : DatasetDiscoveryService | None
            Dataset discovery service instance
        ingestion_service : DatabentoIngestionService | None
            Ingestion service for fallback discovery

        """
        self.dataset_discovery = dataset_discovery
        self.service = ingestion_service
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics using centralized bootstrap."""
        self.discovery_requests_counter = get_counter(
            "ml_discovery_requests_total",
            "Total discovery requests by status",
            labelnames=["status"],
        )
        self.discovery_errors_counter = get_counter(
            "ml_discovery_errors_total",
            "Total discovery errors",
            labelnames=["error"],
        )

    def discover_market_inputs(
        self,
        symbol_map: Mapping[str, tuple[str, ...]],
        schema: str,
        start_ns: int,
        end_ns: int,
        dataset_hint: str | None = None,
    ) -> tuple[MarketDatasetInput, ...]:
        """
        Discover market inputs for given symbols and time range.

        Uses the dataset discovery service to find available datasets for the
        requested symbols and schema. Applies coverage policy if available.

        Parameters
        ----------
        symbol_map : Mapping[str, tuple[str, ...]]
            Symbol to instrument IDs mapping
        schema : str
            Data schema (e.g. 'ohlcv-1m', 'tbbo')
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds
        dataset_hint : str | None
            Optional dataset ID hint

        Returns
        -------
        tuple[MarketDatasetInput, ...]
            Discovered market inputs

        """
        from ml.data.ingest.discovery import DatasetDiscoveryError
        from ml.data.ingest.discovery import DiscoveryRequest
        from ml.orchestration.config_loader import Stage

        service = self.dataset_discovery
        if service is None or start_ns >= end_ns:
            self.discovery_requests_counter.labels(status="skipped").inc()
            return ()

        start_dt = self.ns_to_datetime(start_ns)
        end_dt = self.ns_to_datetime(end_ns)

        requests = tuple(
            DiscoveryRequest(
                symbol=symbol,
                schema=schema,
                start=start_dt,
                end=end_dt,
            )
            for symbol in symbol_map
        )

        if not requests:
            self.discovery_requests_counter.labels(status="empty").inc()
            return ()

        try:
            inputs = service.discover(requests=requests, dataset_hint=dataset_hint)
            self.discovery_requests_counter.labels(status="success").inc(len(requests))

            # Apply coverage policy if available
            coverage_policy = None
            try:
                coverage_policy = service.policy.coverage
            except AttributeError:
                coverage_policy = None

            if coverage_policy is not None:
                for market_input in inputs:
                    coverage_policy.allow_dataset(market_input.dataset_id or "")

            return inputs

        except DatasetDiscoveryError as exc:
            self.discovery_errors_counter.labels(error="discovery_error").inc()
            logger.warning(
                "Dataset discovery unavailable",
                extra={
                    "stage": Stage.DATASET.value,
                    "reason": str(exc),
                    "symbol_count": len(requests),
                },
            )
            return ()

    def discover_binding_for_symbol(
        self,
        symbol: str,
        instrument_ids: tuple[str, ...] | None,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> ResolvedMarketBinding | None:
        """
        Discover binding for a specific symbol.

        Attempts to discover dataset information for a symbol using either the
        ingestion service's discover_symbol_dataset method or the dataset
        discovery service.

        Parameters
        ----------
        symbol : str
            Symbol to discover
        instrument_ids : tuple[str, ...] | None
            Optional instrument IDs for the symbol
        schema : str
            Data schema
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        ResolvedMarketBinding | None
            Discovered binding or None if not found

        """
        from ml.data.ingest.market_bindings import ResolvedMarketBinding

        service = self.service
        if service is None:
            return None

        schema_token = schema.strip()
        if not schema_token:
            return None

        # Try ingestion service discovery function first
        discovery_func = getattr(service, "discover_symbol_dataset", None)
        dataset_service = self.dataset_discovery

        # Fall back to dataset discovery service if needed
        if (discovery_func is None or not callable(discovery_func)) and dataset_service is not None:

            def _dataset_service_wrapper(
                *,
                symbol: str,
                schema: str,
                start_ns: int,
                end_ns: int,
            ) -> SymbolDatasetDiscovery | None:
                return self._discover_symbol_via_dataset_service(
                    dataset_service=dataset_service,
                    symbol=symbol,
                    schema=schema,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )

            discovery_func = _dataset_service_wrapper

        if discovery_func is None or not callable(discovery_func):
            return None

        try:
            discovery = discovery_func(
                symbol=symbol,
                schema=schema_token,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "Dataset discovery failed",
                exc_info=True,
                extra={
                    "symbol": symbol,
                    "schema": schema_token,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                },
            )
            return None

        if discovery is None:
            return None

        resolved_symbol = getattr(discovery, "symbol", symbol)
        instrument_tuple = instrument_ids or (resolved_symbol,)
        binding_id = f"discovered:{discovery.dataset_id}:{resolved_symbol}"

        return ResolvedMarketBinding(
            binding_id=binding_id,
            symbol=resolved_symbol,
            instrument_ids=tuple(instrument_tuple),
            dataset_id=discovery.dataset_id,
            descriptor_id=None,
            schema=discovery.schema,
            storage_kind=discovery.storage_kind,
            license_start=None,
            license_end=None,
            start=None,
            end=None,
            source="discovered",
        )

    def _discover_symbol_via_dataset_service(
        self,
        dataset_service: DatasetDiscoveryService,
        symbol: str,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> SymbolDatasetDiscovery | None:
        """
        Discover symbol using dataset discovery service.

        Internal helper method that wraps the dataset discovery service
        to provide symbol-specific discovery functionality.

        Parameters
        ----------
        dataset_service : DatasetDiscoveryService
            Dataset discovery service
        symbol : str
            Symbol to discover
        schema : str
            Data schema
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        SymbolDatasetDiscovery | None
            Discovery result or None if not found

        """
        from ml.data.ingest.discovery import DatasetDiscoveryError
        from ml.data.ingest.discovery import DiscoveryRequest
        from ml.data.ingest.service import SymbolDatasetDiscovery
        from ml.registry.dataclasses import StorageKind

        if start_ns >= end_ns:
            return None

        request = DiscoveryRequest(
            symbol=symbol,
            schema=schema,
            start=self.ns_to_datetime(start_ns),
            end=self.ns_to_datetime(end_ns),
        )

        try:
            discovered = dataset_service.discover_one(request=request)
        except DatasetDiscoveryError as exc:
            logger.debug(
                "Dataset discovery service rejected symbol",
                extra={
                    "symbol": symbol,
                    "schema": schema,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "reason": str(exc),
                },
            )
            return None
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "Dataset discovery service failed",
                exc_info=True,
                extra={
                    "symbol": symbol,
                    "schema": schema,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                },
            )
            return None

        storage_kind = discovered.storage_kind or StorageKind.POSTGRES
        resolved_symbol = discovered.symbol or symbol

        return SymbolDatasetDiscovery(
            dataset_id=discovered.dataset_id,
            schema=discovered.schema,
            storage_kind=storage_kind,
            symbol=resolved_symbol,
            requested_symbol=discovered.requested_symbol,
            available_start_ns=discovered.available_start_ns,
            available_end_ns=discovered.available_end_ns,
            cost_usd=discovered.cost_usd,
            instrument_id=discovered.instrument_id,
        )

    @staticmethod
    def ns_to_datetime(value: int) -> datetime:
        """
        Convert nanoseconds since epoch to an aware UTC datetime.

        Parameters
        ----------
        value : int
            Timestamp in nanoseconds since epoch

        Returns
        -------
        datetime
            Aware UTC datetime

        """
        seconds = value / 1_000_000_000
        return datetime.fromtimestamp(seconds, tz=UTC)
