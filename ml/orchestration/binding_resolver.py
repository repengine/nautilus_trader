#!/usr/bin/env python3

"""
Market binding resolution for ML pipeline orchestration.

This module provides market binding resolution, coverage validation, priority
selection, and binding filtering with policy enforcement.

This component is extracted from the MLPipelineOrchestrator god class to provide
focused, testable binding resolution functionality.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    from ml.config.market_data import MarketDatasetInput
    from ml.data.ingest.market_bindings import ResolvedMarketBinding
    from ml.orchestration.config_types import DatasetBuildConfig
    from ml.orchestration.discovery_client import DiscoveryClient
    from ml.stores.protocols import CoverageProviderProtocol


logger = logging.getLogger(__name__)


# ========================================================================
# Protocol Definition
# ========================================================================


class BindingResolverProtocol(Protocol):
    """
    Protocol for market binding resolution operations.
    """

    def resolve_market_inputs(
        self,
        cfg: DatasetBuildConfig,
        symbol_map: dict[str, tuple[str, ...]],
        start_ns: int,
        end_ns: int,
    ) -> tuple[
        tuple[MarketDatasetInput, ...] | None,
        tuple[ResolvedMarketBinding, ...],
    ]:
        """
        Resolve market inputs with coverage validation.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration
        symbol_map : dict[str, tuple[str, ...]]
            Symbol to instrument IDs mapping
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        tuple[tuple[MarketDatasetInput, ...] | None, tuple[ResolvedMarketBinding, ...]]
            Resolved inputs and bindings

        """
        ...

    def filter_candidate_bindings(
        self,
        candidates: tuple[ResolvedMarketBinding, ...],
        start_ns: int,
        end_ns: int,
        symbol: str,
        default_schema: str,
    ) -> tuple[ResolvedMarketBinding, ...]:
        """
        Filter candidate bindings based on availability and cost.

        Parameters
        ----------
        candidates : tuple[ResolvedMarketBinding, ...]
            Candidate bindings to filter
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds
        symbol : str
            Symbol being resolved
        default_schema : str
            Default schema to use

        Returns
        -------
        tuple[ResolvedMarketBinding, ...]
            Filtered bindings sorted by priority

        """
        ...

    def select_binding_with_coverage(
        self,
        candidates: tuple[ResolvedMarketBinding, ...],
        start_ns: int,
        end_ns: int,
    ) -> ResolvedMarketBinding | None:
        """
        Select first binding with available coverage.

        Parameters
        ----------
        candidates : tuple[ResolvedMarketBinding, ...]
            Candidate bindings
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        ResolvedMarketBinding | None
            First binding with coverage or None

        """
        ...


# ========================================================================
# BindingResolver Implementation
# ========================================================================


class BindingResolver:
    """
    Resolves market bindings with coverage validation.

    Handles binding discovery, filtering, priority selection, and
    validation against coverage and cost policies.

    This component is extracted from the MLPipelineOrchestrator god class to
    provide focused, testable binding resolution functionality.

    """

    def __init__(
        self,
        coverage_provider: CoverageProviderProtocol | None = None,
        ingestion_service: object | None = None,
        discovery_client: DiscoveryClient | None = None,
    ) -> None:
        """
        Initialize binding resolver.

        Parameters
        ----------
        coverage_provider : CoverageProviderProtocol | None
            Coverage provider for data availability checks
        ingestion_service : DatabentoIngestionService | None
            Ingestion service for availability and cost checks
        discovery_client : DiscoveryClient | None
            Discovery client for binding discovery

        """
        self.coverage = coverage_provider
        self.service = ingestion_service
        self.discovery_client = discovery_client
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics using centralized bootstrap."""
        self.bindings_resolved_counter = get_counter(
            "ml_bindings_resolved_total",
            "Total bindings resolved by status",
            labelnames=["status"],
        )
        self.binding_selection_time = get_histogram(
            "ml_binding_selection_seconds",
            "Time to select binding",
        )

    def resolve_market_inputs(
        self,
        cfg: DatasetBuildConfig,
        symbol_map: dict[str, tuple[str, ...]],
        start_ns: int,
        end_ns: int,
    ) -> tuple[
        tuple[MarketDatasetInput, ...] | None,
        tuple[ResolvedMarketBinding, ...],
    ]:
        """
        Resolve market inputs with coverage validation.

        Attempts to discover market inputs and resolve bindings for all
        symbols in the symbol map. Falls back to discovery if market inputs
        are not explicitly provided.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration
        symbol_map : dict[str, tuple[str, ...]]
            Symbol to instrument IDs mapping
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        tuple[tuple[MarketDatasetInput, ...] | None, tuple[ResolvedMarketBinding, ...]]
            Resolved inputs and bindings

        """
        from ml.config.market_data import MarketDatasetInput
        from ml.data.ingest.market_bindings import resolve_market_dataset_bindings

        # If market inputs provided, resolve bindings directly
        if cfg.market_inputs:
            try:
                bindings = resolve_market_dataset_bindings(cfg.market_inputs)
            except Exception:  # pragma: no cover - defensive guard
                logger.warning("Binding resolution failed", exc_info=True)
                bindings = ()

            self.bindings_resolved_counter.labels(status="from_config").inc(len(bindings))
            return cfg.market_inputs, bindings

        # Otherwise, discover market inputs
        if self.discovery_client is None:
            self.bindings_resolved_counter.labels(status="no_discovery").inc()
            return None, ()

        from ml.orchestration.config_resolver import ConfigResolver

        config_resolver = ConfigResolver()
        default_schema = config_resolver.infer_default_schema(cfg)

        resolved_inputs: list[MarketDatasetInput] = []
        resolved_bindings: list[ResolvedMarketBinding] = []

        # Discover inputs via discovery service
        discovered_inputs = self.discovery_client.discover_market_inputs(
            symbol_map=symbol_map,
            schema=default_schema,
            start_ns=start_ns,
            end_ns=end_ns,
            dataset_hint=cfg.market_dataset_id,
        )

        if discovered_inputs:
            resolved_inputs.extend(discovered_inputs)
            try:
                bindings = resolve_market_dataset_bindings(discovered_inputs)
                resolved_bindings.extend(bindings)
            except Exception:  # pragma: no cover - defensive guard
                logger.warning("Binding resolution from discovery failed", exc_info=True)

        # Fall back to symbol-by-symbol discovery if no inputs discovered
        if not resolved_inputs:
            for symbol, instrument_ids in symbol_map.items():
                binding = self.discovery_client.discover_binding_for_symbol(
                    symbol=symbol,
                    instrument_ids=instrument_ids if instrument_ids else None,
                    schema=default_schema,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )

                if binding is not None:
                    resolved_bindings.append(binding)

                    # Create market input from binding
                    market_input = MarketDatasetInput(
                        descriptor_id=binding.descriptor_id,
                        dataset_id=binding.dataset_id,
                        symbols=(binding.symbol,),
                        schema_override=binding.schema,
                        storage_kind_override=binding.storage_kind,
                    )
                    resolved_inputs.append(market_input)

                    logger.info(
                        "Resolved binding for symbol",
                        extra={
                            "symbol": symbol,
                            "dataset_id": binding.dataset_id,
                            "schema": binding.schema,
                            "source": binding.source,
                        },
                    )

        if not resolved_inputs:
            self.bindings_resolved_counter.labels(status="none_found").inc()
            return None, ()

        self.bindings_resolved_counter.labels(status="discovered").inc(len(resolved_inputs))
        return tuple(resolved_inputs), tuple(resolved_bindings)

    def filter_candidate_bindings(
        self,
        candidates: tuple[ResolvedMarketBinding, ...],
        start_ns: int,
        end_ns: int,
        symbol: str,
        default_schema: str,
    ) -> tuple[ResolvedMarketBinding, ...]:
        """
        Filter candidate bindings based on availability and cost.

        Applies availability checks, cost estimation, and priority ordering
        to candidate bindings.

        Parameters
        ----------
        candidates : tuple[ResolvedMarketBinding, ...]
            Candidate bindings to filter
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds
        symbol : str
            Symbol being resolved
        default_schema : str
            Default schema to use

        Returns
        -------
        tuple[ResolvedMarketBinding, ...]
            Filtered bindings sorted by priority

        """
        if not candidates:
            return ()

        filtered: list[ResolvedMarketBinding] = []
        for binding in candidates:
            if self._binding_allowed(
                binding=binding,
                start_ns=start_ns,
                end_ns=end_ns,
                symbol=symbol,
                default_schema=default_schema,
            ):
                filtered.append(binding)

        if filtered:
            filtered.sort(key=self._binding_priority_key)

        return tuple(filtered)

    def select_binding_with_coverage(
        self,
        candidates: tuple[ResolvedMarketBinding, ...],
        start_ns: int,
        end_ns: int,
    ) -> ResolvedMarketBinding | None:
        """
        Select first binding with available coverage.

        Queries the coverage provider for each candidate binding to find
        the first one with available data coverage.

        Parameters
        ----------
        candidates : tuple[ResolvedMarketBinding, ...]
            Candidate bindings (should be pre-sorted by priority)
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        ResolvedMarketBinding | None
            First binding with coverage or None

        """
        coverage = self.coverage
        if coverage is None:
            return None

        for binding in candidates:
            schema = binding.schema or ""
            if not schema:
                continue

            instruments = binding.instrument_ids or (binding.symbol,)
            for instrument in instruments:
                try:
                    buckets = coverage.read_bucket_coverage(
                        dataset_id=binding.dataset_id,
                        schema=schema,
                        instrument_id=instrument,
                        start_ns=start_ns,
                        end_ns=end_ns,
                    )
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug(
                        "Coverage lookup failed",
                        exc_info=True,
                        extra={
                            "dataset_id": binding.dataset_id,
                            "schema": schema,
                            "instrument_id": instrument,
                        },
                    )
                    buckets = set()

                if buckets:
                    return binding

        return None

    @staticmethod
    def _binding_priority_key(binding: ResolvedMarketBinding) -> tuple[int, str]:
        """
        Compute priority key for binding ordering.

        Lower priority values are selected first. Priorities are assigned
        based on dataset quality and reliability.

        Parameters
        ----------
        binding : ResolvedMarketBinding
            Binding to compute priority for

        Returns
        -------
        tuple[int, str]
            (priority, dataset_id) tuple for sorting

        """
        dataset_id = binding.dataset_id.upper()
        if dataset_id == "EQUS.MINI":
            return (0, dataset_id)
        if dataset_id == "XNAS.ITCH":
            return (1, dataset_id)
        return (2, dataset_id)

    def _binding_allowed(
        self,
        binding: ResolvedMarketBinding,
        start_ns: int,
        end_ns: int,
        symbol: str,
        default_schema: str,
    ) -> bool:
        """
        Check if binding is allowed based on availability and cost.

        Validates binding against ingestion service availability windows
        and cost policies.

        Parameters
        ----------
        binding : ResolvedMarketBinding
            Binding to validate
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds
        symbol : str
            Symbol being validated
        default_schema : str
            Default schema to use

        Returns
        -------
        bool
            True if binding is allowed

        """
        from ml.data.ingest.service import IngestionError
        from ml.orchestration.discovery_client import DiscoveryClient

        service = self.service
        schema = binding.schema or default_schema
        if not schema:
            return False

        # Check availability and cost with ingestion service
        if service is not None and binding.dataset_id:
            try:
                available_start_ns, available_end_ns = service.get_available_range_ns(
                    dataset=binding.dataset_id,
                    schema=schema,
                )
            except IngestionError as exc:
                logger.info(
                    "Binding rejected by ingestion service",
                    extra={
                        "dataset_id": binding.dataset_id,
                        "schema": schema,
                        "symbol": symbol,
                        "reason": str(exc),
                    },
                )
                return False
            except Exception:  # pragma: no cover - defensive guard
                logger.debug(
                    "Binding availability check failed",
                    exc_info=True,
                    extra={
                        "dataset_id": binding.dataset_id,
                        "schema": schema,
                        "symbol": symbol,
                    },
                )
            else:
                # Check if requested range is within available range
                if available_start_ns is not None and end_ns <= available_start_ns:
                    logger.info(
                        "Binding outside provider coverage",
                        extra={
                            "dataset_id": binding.dataset_id,
                            "schema": schema,
                            "symbol": symbol,
                            "available_start_ns": available_start_ns,
                            "requested_end_ns": end_ns,
                        },
                    )
                    return False

                if available_end_ns is not None and start_ns >= available_end_ns:
                    logger.info(
                        "Binding outside provider coverage",
                        extra={
                            "dataset_id": binding.dataset_id,
                            "schema": schema,
                            "symbol": symbol,
                            "available_end_ns": available_end_ns,
                            "requested_start_ns": start_ns,
                        },
                    )
                    return False

                # Check cost
                try:
                    cost_usd = service.estimate_cost_usd(
                        dataset=binding.dataset_id,
                        schema=schema,
                        symbols=(symbol,),
                        start=DiscoveryClient.ns_to_datetime(start_ns),
                        end=DiscoveryClient.ns_to_datetime(end_ns),
                    )
                except IngestionError as exc:
                    logger.info(
                        "Binding rejected by cost policy",
                        extra={
                            "dataset_id": binding.dataset_id,
                            "schema": schema,
                            "symbol": symbol,
                            "reason": str(exc),
                        },
                    )
                    return False
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug(
                        "Binding cost estimation failed",
                        exc_info=True,
                        extra={
                            "dataset_id": binding.dataset_id,
                            "schema": schema,
                            "symbol": symbol,
                        },
                    )
                else:
                    if cost_usd > 0.0:
                        logger.info(
                            "Binding rejected due to non-zero cost",
                            extra={
                                "dataset_id": binding.dataset_id,
                                "schema": schema,
                                "symbol": symbol,
                                "cost_usd": cost_usd,
                            },
                        )
                        return False

        return True
