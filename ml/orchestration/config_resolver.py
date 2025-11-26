#!/usr/bin/env python3

"""
Configuration resolution for ML pipeline orchestrator.

This module provides comprehensive configuration resolution including market input
defaults, symbol mapping, window bounds computation, and dataset config preparation
with proper defaults and validation.

This component is extracted from the MLPipelineOrchestrator god class to provide
focused, testable configuration resolution functionality.

"""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import UTC
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import load_market_feed_descriptors
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.orchestration.config_types import DEFAULT_LOOKBACK_YEARS
from ml.orchestration.config_types import DatasetBuildConfig


logger = logging.getLogger(__name__)


# ========================================================================
# Protocol Definition
# ========================================================================


class ConfigResolverProtocol(Protocol):
    """
    Protocol for configuration resolution operations.
    """

    def apply_default_market_inputs(
        self,
        cfg: DatasetBuildConfig,
    ) -> DatasetBuildConfig:
        """
        Seed dataset configs with descriptor-driven market inputs.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        DatasetBuildConfig
            Configuration with market inputs populated

        """
        ...

    def collect_symbol_map(
        self,
        ds_cfg: DatasetBuildConfig | None,
        symbols: tuple[str, ...] | None,
        instruments: tuple[str, ...] | None,
        instrument_ids: tuple[str, ...] | None,
        market_inputs: tuple[MarketDatasetInput, ...] | None,
    ) -> dict[str, tuple[str, ...]]:
        """
        Collect symbol to instrument ID mappings from configs.

        Parameters
        ----------
        ds_cfg : DatasetBuildConfig | None
            Dataset build configuration
        symbols : tuple[str, ...] | None
            Symbol list
        instruments : tuple[str, ...]  | None
            Instrument list
        instrument_ids : tuple[str, ...] | None
            Instrument ID list
        market_inputs : tuple[MarketDatasetInput, ...] | None
            Market inputs

        Returns
        -------
        dict[str, tuple[str, ...]]
            Symbol to instrument IDs mapping

        """
        ...

    def compute_window_start_iso(
        self,
        end_iso: str,
        lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    ) -> str:
        """
        Compute ISO8601 start date by subtracting lookback years.

        Parameters
        ----------
        end_iso : str
            End date in ISO8601 format
        lookback_years : int
            Number of years to look back

        Returns
        -------
        str
            Start date in ISO8601 format

        """
        ...

    def resolve_window_bounds_ns(
        self,
        cfg: DatasetBuildConfig,
    ) -> tuple[int, int]:
        """
        Resolve window bounds in nanoseconds from configuration.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        tuple[int, int]
            (start_ns, end_ns) tuple in nanoseconds since epoch

        """
        ...

    def prepare_dataset_config(
        self,
        cfg: DatasetBuildConfig,
        resolved_inputs: tuple[MarketDatasetInput, ...] | None,
        bindings: tuple[ResolvedMarketBinding, ...],
    ) -> DatasetBuildConfig:
        """
        Prepare dataset config with resolved market inputs and instrument IDs.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Base dataset configuration
        resolved_inputs : tuple[MarketDatasetInput, ...] | None
            Resolved market inputs
        bindings : tuple[ResolvedMarketBinding, ...]
            Resolved market bindings

        Returns
        -------
        DatasetBuildConfig
            Updated configuration with resolved values

        """
        ...

    def symbol_to_instruments(
        self,
        cfg: DatasetBuildConfig,
    ) -> OrderedDict[str, tuple[str, ...]]:
        """
        Extract symbol to instrument IDs mapping from config.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        OrderedDict[str, tuple[str, ...]]
            Ordered mapping from symbols to instrument IDs

        """
        ...

    def collect_instrument_ids(
        self,
        bindings: tuple[ResolvedMarketBinding, ...],
        existing: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """
        Collect instrument IDs from bindings and existing config.

        Parameters
        ----------
        bindings : tuple[ResolvedMarketBinding, ...]
            Resolved market bindings
        existing : tuple[str, ...] | None
            Existing instrument IDs from config

        Returns
        -------
        tuple[str, ...]
            Collected and deduplicated instrument IDs

        """
        ...

    def infer_default_schema(
        self,
        cfg: DatasetBuildConfig,
    ) -> str:
        """
        Infer a reasonable default schema for discovery lookups.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        str
            Inferred schema (e.g., 'ohlcv-1m')

        """
        ...

    def resolve_instrument_ids(
        self,
        dataset_cfg: DatasetBuildConfig,
        override: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """
        Resolve instrument IDs from config or override.

        Parameters
        ----------
        dataset_cfg : DatasetBuildConfig
            Dataset configuration
        override : tuple[str, ...] | None
            Optional override instrument IDs

        Returns
        -------
        tuple[str, ...]
            Resolved instrument IDs

        """
        ...


# ========================================================================
# ConfigResolver Implementation
# ========================================================================


class ConfigResolver:
    """
    Resolves and prepares configuration for ML pipeline operations.

    Handles market input resolution, symbol mapping, window bounds computation,
    and dataset config preparation with proper defaults and validation.

    This component is extracted from the MLPipelineOrchestrator god class to
    provide focused, testable configuration resolution functionality.

    """

    def __init__(self) -> None:
        """
        Initialize configuration resolver.
        """
        logger.debug("Initialized ConfigResolver")

    def apply_default_market_inputs(
        self,
        cfg: DatasetBuildConfig,
    ) -> DatasetBuildConfig:
        """
        Seed dataset configs with descriptor-driven market inputs.

        When ``market_dataset_id`` is explicitly provided but ``market_inputs``
        is empty, this method populates market inputs from the descriptor registry.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        DatasetBuildConfig
            Configuration with market inputs populated

        """
        if cfg.market_inputs or not cfg.market_dataset_id:
            return cfg

        descriptors = load_market_feed_descriptors().as_mapping()
        descriptor = descriptors.get(cfg.market_dataset_id)

        if descriptor is None:
            return cfg

        symbols: list[str] = []
        for raw_symbol in str(cfg.symbols).split(","):
            token = raw_symbol.strip().upper()
            if not token:
                continue
            base = token.split(".", maxsplit=1)[0]
            if base and base not in symbols:
                symbols.append(base)

        if not symbols:
            return cfg

        from dataclasses import replace

        inputs = tuple(
            MarketDatasetInput(
                descriptor_id=descriptor.descriptor_id,
                dataset_id=descriptor.dataset_id,
                symbols=(symbol,),
                schema_override=descriptor.schema,
                storage_kind_override=descriptor.storage_kind,
            )
            for symbol in symbols
        )

        updated_cfg: DatasetBuildConfig = replace(
            cfg,
            market_inputs=inputs,
            market_dataset_id=cfg.market_dataset_id,
        )
        return updated_cfg

    def collect_symbol_map(
        self,
        ds_cfg: DatasetBuildConfig | None,
        symbols: tuple[str, ...] | None = None,
        instruments: tuple[str, ...] | None = None,
        instrument_ids: tuple[str, ...] | None = None,
        market_inputs: tuple[MarketDatasetInput, ...] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """
        Collect symbol to instrument ID mappings from configs.

        Aggregates symbols and instruments from dataset config and ingestion config
        to produce a unified symbol-to-instruments mapping.

        Parameters
        ----------
        ds_cfg : DatasetBuildConfig | None
            Dataset build configuration
        symbols : tuple[str, ...] | None
            Symbol list from ingestion config
        instruments : tuple[str, ...] | None
            Instrument list from ingestion config
        instrument_ids : tuple[str, ...] | None
            Instrument ID list from ingestion config
        market_inputs : tuple[MarketDatasetInput, ...] | None
            Market inputs (fallback if ds_cfg has none)

        Returns
        -------
        dict[str, tuple[str, ...]]
            Symbol to instrument IDs mapping

        """
        symbol_to_instruments: dict[str, list[str]] = {}

        def _register(symbol: str, instrument_id: str | None = None) -> None:
            symbol_norm = symbol.strip().upper()
            if not symbol_norm:
                return
            bucket = symbol_to_instruments.setdefault(symbol_norm, [])
            if instrument_id is None:
                return
            inst_norm = instrument_id.strip().upper()
            if inst_norm and inst_norm not in bucket:
                bucket.append(inst_norm)

        def _extract_symbol(token: str) -> str:
            stripped = token.strip()
            if not stripped:
                return ""
            upper = stripped.upper()
            if "." in upper:
                return upper.split(".")[0]
            return upper

        # Process ingestion config sources
        for symbol in symbols or ():
            symbol_to_instruments.setdefault(symbol.strip().upper(), [])

        for instrument in instruments or ():
            base = _extract_symbol(instrument)
            if base:
                _register(base, instrument)

        for instrument in instrument_ids or ():
            base = _extract_symbol(instrument)
            if base:
                _register(base, instrument)

        # Process dataset config sources
        if ds_cfg is not None:
            for raw_symbol in str(ds_cfg.symbols).split(","):
                symbol_norm = raw_symbol.strip().upper()
                if symbol_norm:
                    symbol_to_instruments.setdefault(symbol_norm, [])

            for instrument in ds_cfg.instrument_ids or ():
                base = _extract_symbol(instrument)
                if base:
                    _register(base, instrument)

        # Process market inputs
        effective_inputs = market_inputs
        if effective_inputs is None and ds_cfg is not None:
            effective_inputs = ds_cfg.market_inputs

        if effective_inputs:
            for item in effective_inputs:
                for symbol in item.symbols or ():
                    symbol_to_instruments.setdefault(symbol.strip().upper(), [])

        # Fallback: if nothing collected, try instruments again
        if not symbol_to_instruments:
            for instrument in instruments or ():
                base = _extract_symbol(instrument)
                if base:
                    _register(base, instrument)

        return {symbol: tuple(values) for symbol, values in symbol_to_instruments.items()}

    def compute_window_start_iso(
        self,
        end_iso: str,
        lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    ) -> str:
        """
        Compute ISO8601 start date by subtracting lookback years.

        Parameters
        ----------
        end_iso : str
            End date in ISO8601 format
        lookback_years : int
            Number of years to look back

        Returns
        -------
        str
            Start date in ISO8601 format

        """
        from calendar import monthrange

        end_date = date.fromisoformat(end_iso)
        target_year = end_date.year - lookback_years
        days_in_month = monthrange(target_year, end_date.month)[1]
        day = min(end_date.day, days_in_month)
        start_date = date(target_year, end_date.month, day)
        return start_date.isoformat()

    def resolve_window_bounds_ns(
        self,
        cfg: DatasetBuildConfig,
    ) -> tuple[int, int]:
        """
        Resolve window bounds in nanoseconds from configuration.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        tuple[int, int]
            (start_ns, end_ns) tuple in nanoseconds since epoch

        """
        from ml.data.vintage import parse_dt
        from ml.stores.providers import DAY_NS

        end_dt = parse_dt(cfg.end_iso) if cfg.end_iso else None
        if end_dt is None:
            end_dt = datetime.now(tz=UTC)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=UTC)

        start_iso = cfg.start_iso
        if start_iso is None:
            start_iso = self.compute_window_start_iso(end_iso=end_dt.date().isoformat())

        start_dt = parse_dt(start_iso)
        if start_dt is None:
            start_dt = datetime.fromisoformat(start_iso).replace(tzinfo=UTC)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)

        start_ns = int(start_dt.timestamp() * 1_000_000_000)
        end_ns = int(end_dt.timestamp() * 1_000_000_000)

        if end_ns <= start_ns:
            end_ns = start_ns + DAY_NS

        return start_ns, end_ns

    def prepare_dataset_config(
        self,
        cfg: DatasetBuildConfig,
        resolved_inputs: tuple[MarketDatasetInput, ...] | None,
        bindings: tuple[ResolvedMarketBinding, ...],
    ) -> DatasetBuildConfig:
        """
        Prepare dataset config with resolved market inputs and instrument IDs.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Base dataset configuration
        resolved_inputs : tuple[MarketDatasetInput, ...] | None
            Resolved market inputs
        bindings : tuple[ResolvedMarketBinding, ...]
            Resolved market bindings

        Returns
        -------
        DatasetBuildConfig
            Updated configuration with resolved values

        """
        from dataclasses import replace

        base_cfg = self.apply_default_market_inputs(cfg)

        if resolved_inputs:
            instrument_ids = self.collect_instrument_ids(bindings, base_cfg.instrument_ids)
            updated_cfg: DatasetBuildConfig = replace(
                base_cfg,
                market_inputs=resolved_inputs,
                instrument_ids=instrument_ids,
            )
            base_cfg = updated_cfg

        logger.info(
            "Dataset config prepared",
            extra={
                "symbols": base_cfg.symbols,
                "instrument_ids": base_cfg.instrument_ids,
                "market_inputs": (
                    0 if base_cfg.market_inputs is None else len(base_cfg.market_inputs)
                ),
            },
        )

        return base_cfg

    def symbol_to_instruments(
        self,
        cfg: DatasetBuildConfig,
    ) -> OrderedDict[str, tuple[str, ...]]:
        """
        Extract symbol to instrument IDs mapping from config.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        OrderedDict[str, tuple[str, ...]]
            Ordered mapping from symbols to instrument IDs

        """
        symbols: OrderedDict[str, None] = OrderedDict()
        raw_symbols = str(cfg.symbols or "").split(",")
        for raw in raw_symbols:
            token = raw.strip()
            if not token:
                continue
            symbols.setdefault(token.split(".")[0].upper(), None)

        instrument_mapping: dict[str, list[str]] = {}
        for inst in cfg.instrument_ids or ():
            token = inst.strip()
            if not token:
                continue
            upper = token.upper()
            base = upper.split(".")[0]
            instrument_mapping.setdefault(base, []).append(upper)
            symbols.setdefault(base, None)

        ordered: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        for symbol in symbols.keys():
            ordered[symbol] = tuple(instrument_mapping.get(symbol, ()))
        return ordered

    def collect_instrument_ids(
        self,
        bindings: tuple[ResolvedMarketBinding, ...],
        existing: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """
        Collect instrument IDs from bindings and existing config.

        Parameters
        ----------
        bindings : tuple[ResolvedMarketBinding, ...]
            Resolved market bindings
        existing : tuple[str, ...] | None
            Existing instrument IDs from config

        Returns
        -------
        tuple[str, ...]
            Collected and deduplicated instrument IDs

        """
        collected: OrderedDict[str, None] = OrderedDict()

        if existing:
            for inst in existing:
                token = inst.strip()
                if token:
                    collected.setdefault(token.upper(), None)

        for binding in bindings:
            for inst in binding.instrument_ids or (binding.symbol,):
                token = inst.strip().upper()
                if token:
                    collected.setdefault(token, None)

        return tuple(collected.keys())

    def infer_default_schema(
        self,
        cfg: DatasetBuildConfig,
    ) -> str:
        """
        Infer a reasonable default schema for discovery lookups.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        str
            Inferred schema (default: 'ohlcv-1m')

        """
        return "ohlcv-1m"

    def resolve_instrument_ids(
        self,
        dataset_cfg: DatasetBuildConfig,
        override: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """
        Resolve instrument IDs from config or override.

        Parameters
        ----------
        dataset_cfg : DatasetBuildConfig
            Dataset configuration
        override : tuple[str, ...] | None
            Optional override instrument IDs

        Returns
        -------
        tuple[str, ...]
            Resolved instrument IDs

        """
        if override:
            return tuple(item.strip() for item in override if item.strip())

        if dataset_cfg.instrument_ids:
            return tuple(item.strip() for item in dataset_cfg.instrument_ids if item.strip())

        symbols_raw = dataset_cfg.symbols.split(",")
        return tuple(item.strip().upper() for item in symbols_raw if item.strip())

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

    # ------------------------------------------------------------------
    # Structural compatibility helpers (Phase0 placeholders)
    # ------------------------------------------------------------------

    def _resolve_window_bounds_ns(
        self,
        start: str | None,
        end: str | None,
    ) -> tuple[int | None, int | None]:
        del start, end
        return (None, None)

    def _resolve_instrument_ids(
        self,
        config_ids: tuple[str, ...],
        binding_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        del config_ids, binding_ids
        return ()

    def _resolve_market_inputs(
        self,
        market_inputs: dict[str, object] | None,
    ) -> OrderedDict[str, object]:
        del market_inputs
        return OrderedDict()

    def _symbol_to_instruments(
        self,
        symbols: list[str],
        venue: object | None,
    ) -> tuple[str, ...]:
        del symbols, venue
        return ()

    @staticmethod
    def _infer_default_schema(cfg: DatasetBuildConfig) -> str | tuple[str, ...] | None:
        del cfg
        return None

    @staticmethod
    def _infer_feature_names(feature_dir: Path) -> tuple[str, ...] | None:
        del feature_dir
        return ()

    def _auto_fill_universe(self, universe: list[str]) -> tuple[str, ...]:
        del universe
        return ()

    def _auto_fill_schema(
        self,
        schema: str | None,
        *,
        config: DatasetBuildConfig,
        feature_dir: Path | None,
    ) -> tuple[str, ...]:
        del schema, config, feature_dir
        return ()

    def _auto_fill_l2(self, l2_schemas: dict[str, object] | None) -> dict[str, object]:
        del l2_schemas
        return {}

    @staticmethod
    def _collect_instrument_ids(
        market_inputs: dict[str, object] | OrderedDict[str, object],
    ) -> tuple[str, ...]:
        del market_inputs
        return ()
