"""
Helper methods for `MLPipelineOrchestratorFacade`.

This module keeps `ml/orchestration/pipeline_orchestrator_facade.py` thin by hosting
small parity helpers used by unit tests and legacy shims.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from ml.orchestration.binding_resolver import BindingResolver
from ml.orchestration.config_resolver import ConfigResolver
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.discovery_client import DiscoveryClient
from ml.stores.protocols import CoverageProviderProtocol


if TYPE_CHECKING:
    from ml.data.ingest.discovery import DatasetDiscoveryService
    from ml.data.ingest.market_bindings import ResolvedMarketBinding
    from ml.data.ingest.service import DatabentoIngestionService


logger = logging.getLogger(__name__)


class OrchestratorFacadeHelpers:
    """Shared helper methods used by `MLPipelineOrchestratorFacade`."""

    coverage: CoverageProviderProtocol
    service: DatabentoIngestionService | None
    dataset_discovery: object | None
    _config_resolver: ConfigResolver | None
    _discovery_client: DiscoveryClient | None

    def apply_default_market_inputs(self, cfg: DatasetBuildConfig) -> DatasetBuildConfig:
        """Apply default market inputs via ConfigResolver."""
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.apply_default_market_inputs(cfg)

    def collect_symbol_map(
        self,
        *,
        ds_cfg: DatasetBuildConfig | None = None,
        symbols: tuple[str, ...] | None = None,
        instruments: tuple[str, ...] | None = None,
        instrument_ids: tuple[str, ...] | None = None,
        market_inputs: tuple[Any, ...] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Collect the normalized symbol map via ConfigResolver."""
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.collect_symbol_map(
            ds_cfg=ds_cfg,
            symbols=symbols,
            instruments=instruments,
            instrument_ids=instrument_ids,
            market_inputs=market_inputs,
        )

    def compute_window_start_iso(self, end_iso: str, lookback_years: int = 1) -> str:
        """Compute an ISO start date from end date and lookback via ConfigResolver."""
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.compute_window_start_iso(end_iso=end_iso, lookback_years=lookback_years)

    def resolve_window_bounds_ns(self, cfg: DatasetBuildConfig) -> tuple[int, int]:
        """Resolve dataset time window bounds (ns) via ConfigResolver."""
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.resolve_window_bounds_ns(cfg)

    def prepare_dataset_config(
        self,
        *,
        cfg: DatasetBuildConfig,
        resolved_inputs: tuple[Any, ...] | None,
        bindings: tuple[Any, ...],
    ) -> DatasetBuildConfig:
        """Prepare a dataset config for build stage via ConfigResolver."""
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.prepare_dataset_config(
            cfg=cfg,
            resolved_inputs=resolved_inputs,
            bindings=bindings,
        )

    def discover_market_inputs(
        self,
        *,
        symbol_map: dict[str, tuple[str, ...]],
        schema: str,
        start_ns: int,
        end_ns: int,
        dataset_hint: str | None = None,
    ) -> tuple[Any, ...] | None:
        """Discover market inputs via DiscoveryClient when available."""
        discovery_service = cast("DatasetDiscoveryService | None", getattr(self, "dataset_discovery", None))
        discovery_client = self._discovery_client or (
            DiscoveryClient(dataset_discovery=discovery_service) if discovery_service else None
        )
        if discovery_client is None:
            return None
        try:
            return discovery_client.discover_market_inputs(
                symbol_map=symbol_map,
                schema=schema,
                start_ns=start_ns,
                end_ns=end_ns,
                dataset_hint=dataset_hint,
            )
        except Exception:
            logger.debug("discover_market_inputs failed (facade)", exc_info=True)
            return None

    def resolve_market_inputs(
        self,
        *,
        cfg: DatasetBuildConfig,
        symbol_map: dict[str, tuple[str, ...]],
        start_ns: int,
        end_ns: int,
    ) -> tuple[tuple[Any, ...] | None, tuple[Any, ...]]:
        """Resolve market bindings and inputs via BindingResolver."""
        resolver = BindingResolver(
            coverage_provider=self.coverage,
            ingestion_service=self.service,
            discovery_client=self._discovery_client,
        )
        return resolver.resolve_market_inputs(
            cfg=cfg,
            symbol_map=symbol_map,
            start_ns=start_ns,
            end_ns=end_ns,
        )

    def filter_candidate_bindings(
        self,
        *,
        candidates: tuple[ResolvedMarketBinding, ...],
        start_ns: int,
        end_ns: int,
        symbol: str,
        default_schema: str,
    ) -> tuple[ResolvedMarketBinding, ...]:
        """Filter candidate bindings via BindingResolver."""
        resolver = BindingResolver(
            coverage_provider=self.coverage,
            ingestion_service=self.service,
            discovery_client=self._discovery_client,
        )
        return resolver.filter_candidate_bindings(
            candidates=candidates,
            start_ns=start_ns,
            end_ns=end_ns,
            symbol=symbol,
            default_schema=default_schema,
        )
