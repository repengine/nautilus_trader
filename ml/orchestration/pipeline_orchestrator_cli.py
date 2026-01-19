#!/usr/bin/env python3

"""
Pipeline orchestrator CLI and helpers (facade-only).

Composes existing ingestion, dataset build, HPO, and training CLIs into a typed,
testable interface suitable for a single long-running service or a nightly batch job.
All heavy work (DataFrames, file I/O, GPU training) remains strictly off the actor hot
paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid as _uuid
from calendar import monthrange
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
from datetime import date
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from ml.common.db_connections import ConnectionRole as _DbConnectionRole
from ml.common.db_connections import collect_postgres_candidates as _collect_db_candidates
from ml.common.logging_config import bind_log_context
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import coerce_storage_kind
from ml.config.market_data import load_market_feed_descriptors as _load_market_feed_descriptors
from ml.data import DatasetMetadata
from ml.data import DatasetValidationConfig
from ml.data.ingest.databento_adapter import DatabentoAPIClient
from ml.data.ingest.discovery import DatasetDiscoveryService
from ml.data.ingest.discovery import DiscoveryPolicy
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.service import IngestionError
from ml.data.ingest.subscription import SubscriptionPolicy as CoveragePolicy
from ml.data.ingest.subscription import get_max_lookback_days
from ml.data.ingest.symbology import DatabentoSymbologyResolver
from ml.data.vintage import VintagePolicy
from ml.orchestration.common.utils import map_schema_to_dataset_type
from ml.orchestration.config_loader import IngestionStageConfig
from ml.orchestration.config_loader import Stage
from ml.orchestration.config_loader import load_orchestrator_run_config
from ml.orchestration.config_loader import to_pipeline_args
from ml.orchestration.config_types import DEFAULT_LOOKBACK_YEARS
from ml.orchestration.config_types import DEFAULT_MACRO_SERIES
from ml.orchestration.config_types import AutoFillUniverseConfig
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import IntegrationConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import PromotionsConfig
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig
from ml.orchestration.pipeline_orchestrator_facade import MLPipelineOrchestratorFacade as MLPipelineOrchestrator
from ml.registry.protocols import RegistryProtocol
from ml.schema import schema_spec_for
from ml.stores.io_raw import ParquetCatalogRawWriter
from ml.stores.io_raw import RawIngestionWriterProtocol
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol
from ml.stores.providers import DAY_NS
from ml.stores.providers import CatalogCoverageProvider
from ml.stores.providers import SqlCoverageProvider
from ml.stores.providers import SqlMarketDataWriter
from ml.stores.writers import DataStoreMarketDataWriter
from ml.stores.writers import FanoutMarketDataWriter
from ml.stores.writers import ParquetCatalogMarketDataWriter
from ml.tasks.ingest import PopulateL2TaskConfig
from ml.tasks.ingest import populate_l2_efficient


load_market_feed_descriptors = _load_market_feed_descriptors


logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - type-only imports
    pass


@lru_cache(maxsize=1)
def _get_allowed_databento_datasets() -> frozenset[str] | None:
    try:
        from ml.config.databento_policy import load_databento_safety_config

        cfg = load_databento_safety_config(None)
        datasets = cfg.datasets if hasattr(cfg, "datasets") else None
        return frozenset(datasets) if datasets else None
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("databento_safety_config_unavailable", exc_info=True)
        return None


_WRITE_MODE_TOKEN_MAP: Final[dict[str, tuple[str, ...]]] = {
    "datastore": ("datastore",),
    "parquet": ("datastore", "parquet"),
    "datastore+parquet": ("datastore", "parquet"),
    "sql": ("sql",),
    "sql+datastore": ("sql", "datastore"),
    "sql+parquet": ("sql", "parquet"),
    "sql+datastore+parquet": ("sql", "datastore", "parquet"),
}

_WRITE_MODE_ALLOWED_TOKENS: Final[frozenset[str]] = frozenset({"sql", "datastore", "parquet"})

_SCHEMA_ALIASES: Final[dict[str, str]] = {
    "bars": "ohlcv-1m",
    "ohlcv": "ohlcv-1m",
    "tbbo": "tbbo",
    "quotes": "tbbo",
    "trades": "trades",
}


def _normalize_schema_token(schema: str) -> str:
    """
    Resolve schema aliases and validate against the schema registry.
    """
    resolved = _SCHEMA_ALIASES.get(schema.lower(), schema)
    schema_spec_for(resolved)
    return resolved


def _resolve_write_mode_tokens(raw_mode: str) -> tuple[str, ...]:
    """
    Normalize write-mode token strings to ordered mode tuples.
    """
    normalized = raw_mode.strip().lower()
    mapped = _WRITE_MODE_TOKEN_MAP.get(normalized)
    if mapped is not None:
        return mapped
    if normalized:
        tokens = tuple(token for token in normalized.split("+") if token)
        if tokens:
            invalid = [token for token in tokens if token not in _WRITE_MODE_ALLOWED_TOKENS]
            if invalid:
                raise SystemExit(
                    f"Unsupported write_mode tokens {invalid}; allowed tokens are "
                    f"{sorted(_WRITE_MODE_ALLOWED_TOKENS)}",
                )
            ordered = tuple(dict.fromkeys(tokens))
            return ordered
    raise SystemExit(f"Unsupported write_mode '{raw_mode}'")


def _parse_csv_tuple(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(token.strip() for token in str(raw).split(",") if token.strip())


def _apply_default_market_inputs(cfg: DatasetBuildConfig) -> DatasetBuildConfig:
    """
    Seed dataset configs with descriptor-driven market inputs when ``market_dataset_id``
    is explicitly provided.
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

    return replace(
        cfg,
        market_inputs=inputs,
        market_dataset_id=cfg.market_dataset_id,
    )


def _collect_symbol_map(
    *,
    ds_cfg: DatasetBuildConfig | None,
    ingestion_cfg: IngestionStageConfig,
) -> dict[str, tuple[str, ...]]:
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

    for symbol in ingestion_cfg.symbols or ():
        symbol_to_instruments.setdefault(symbol.strip().upper(), [])
    for instrument in ingestion_cfg.instruments:
        base = _extract_symbol(instrument)
        if base:
            _register(base, instrument)
    for instrument in ingestion_cfg.instrument_ids or ():
        base = _extract_symbol(instrument)
        if base:
            _register(base, instrument)

    if ds_cfg is not None:
        for raw_symbol in str(ds_cfg.symbols).split(","):
            symbol_norm = raw_symbol.strip().upper()
            if symbol_norm:
                symbol_to_instruments.setdefault(symbol_norm, [])
        for instrument in ds_cfg.instrument_ids or ():
            base = _extract_symbol(instrument)
            if base:
                _register(base, instrument)

    market_inputs = ingestion_cfg.market_inputs
    if market_inputs is None and ds_cfg is not None:
        market_inputs = ds_cfg.market_inputs
    if market_inputs:
        for item in market_inputs:
            for symbol in item.symbols or ():
                symbol_to_instruments.setdefault(symbol.strip().upper(), [])

    if not symbol_to_instruments:
        for instrument in ingestion_cfg.instruments:
            base = _extract_symbol(instrument)
            if base:
                _register(base, instrument)

    return {symbol: tuple(values) for symbol, values in symbol_to_instruments.items()}


def _compute_window_start_iso(*, end_iso: str, lookback_years: int = DEFAULT_LOOKBACK_YEARS) -> str:
    """
    Compute ISO8601 start date by subtracting ``lookback_years`` from ``end_iso``.
    """
    end_date = date.fromisoformat(end_iso)
    target_year = end_date.year - lookback_years
    days_in_month = monthrange(target_year, end_date.month)[1]
    day = min(end_date.day, days_in_month)
    start_date = date(target_year, end_date.month, day)
    return start_date.isoformat()


class _CliMain(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


class IntegrationManagerProtocol(Protocol):
    data_registry: object | None
    feature_registry: object | None
    model_registry: object | None
    strategy_registry: object | None
    data_store: object | None
    feature_store: object | None
    model_store: object | None
    strategy_store: object | None
    partition_manager: object | None


@dataclass(slots=True)
class BuildArtifacts:
    out_dir: Path
    feature_registry_dir: str | None
    feature_set_id: str | None
    feature_names: tuple[str, ...] = ()
    dataset_metadata: DatasetMetadata | None = None


class _EmptyDatasetError(RuntimeError):
    """
    Raised when the dataset build produces zero rows.
    """

    def __init__(self, message: str, *, row_count: int | None = None) -> None:
        super().__init__(message)
        self.row_count = row_count


@dataclass(slots=True, frozen=True)
class _AutoFillMetrics:
    operations_total: Any
    latency_seconds: Any

    @staticmethod
    def default() -> _AutoFillMetrics:
        return _AutoFillMetrics(
            operations_total=get_counter(
                "nautilus_ml_auto_fill_operations_total",
                "Auto-fill ingestion operations",
                ("schema", "status"),
            ),
            latency_seconds=get_histogram(
                "nautilus_ml_auto_fill_latency_seconds",
                "Auto-fill ingestion latency",
                ("schema",),
                buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            ),
        )


@dataclass(slots=True, frozen=True)
class _IngestionMetrics:
    """
    Instrumentation bundle for ingestion-stage bookkeeping.
    """

    runs_total: Any
    latency_seconds: Any
    fallback_total: Any

    @staticmethod
    def default() -> _IngestionMetrics:
        """
        Initialise lazily to ensure metrics bootstrap occurs once per process.
        """
        return _IngestionMetrics(
            runs_total=get_counter(
                "nautilus_ml_ingestion_stage_runs_total",
                "Pipeline ingestion stage executions",
                labelnames=("component", "status"),
            ),
            latency_seconds=get_histogram(
                "nautilus_ml_ingestion_stage_latency_seconds",
                "Pipeline ingestion stage latency",
                labelnames=("component", "status"),
                buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
            ),
            fallback_total=get_counter(
                "ml_fallback_activations_total",
                "Fallback activations",
                labelnames=("component", "level"),
            ),
        )


@dataclass(slots=True, frozen=True)
class _IngestionAttemptReport:
    """
    Structured outcome for an ingestion attempt.
    """

    success: bool
    context: dict[str, object]
    reason: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    import os

    parser = argparse.ArgumentParser(description="Run end-to-end ML pipeline (cold path)")
    parser.add_argument("--config", default=None, help="Path to orchestrator JSON/TOML config")
    parser.add_argument(
        "--stage",
        default=None,
        choices=[member.value for member in Stage],
        help="Pipeline stage to run (ingest, dataset, train, full)",
    )

    # Ingestion/backfill
    parser.add_argument("--ingest", action="store_true", help="Run ingestion backfill first")
    parser.add_argument("--dataset_id", default=None)
    parser.add_argument("--schema", default="bars", choices=["bars", "tbbo", "trades"])
    parser.add_argument("--instruments", default="SPY.NYSE")
    parser.add_argument("--lookback_days", type=int, default=7)
    parser.add_argument("--coverage_mode", default="catalog", choices=["catalog", "sql"])
    parser.add_argument("--catalog_path", default=os.getenv("CATALOG_PATH", ""))
    default_db_candidates = _collect_db_candidates(_DbConnectionRole.PRIMARY)
    default_db_url = (
        default_db_candidates.urls[0]
        if default_db_candidates.urls
        else "postgresql://postgres:postgres@localhost:5433/nautilus"
    )
    parser.add_argument(
        "--db",
        default=default_db_url,
    )

    # Writer mode for ingestion
    parser.add_argument(
        "--write_mode",
        default="parquet",
        choices=tuple(sorted(_WRITE_MODE_TOKEN_MAP.keys())),
        help=(
            "Ingestion writer fanout: parquet (DataStore+Parquet), datastore, sql, "
            "sql+datastore, sql+parquet, or sql+datastore+parquet"
        ),
    )

    # Dataset build
    parser.add_argument("--data_dir", default="data/tier1")
    parser.add_argument("--symbols", default="SPY.NYSE")
    parser.add_argument("--out_dir", default="ml_out")
    parser.add_argument("--include_macro", action="store_true")
    parser.add_argument("--macro_lag_days", type=int, default=1)
    parser.add_argument("--include_micro", action="store_true")
    parser.add_argument("--include_l2", action="store_true")
    parser.add_argument("--include_events", action="store_true")
    parser.add_argument("--include_calendar", action="store_true")
    parser.add_argument(
        "--instrument_ids",
        default=None,
        help="Comma-separated instrument identifiers (symbol.exchange)",
    )
    parser.add_argument(
        "--market_dataset_id",
        default=None,
        help="Identifier for the canonical market data dataset (defaults to auto-fill dataset when provided)",
    )
    parser.add_argument(
        "--market_inputs_json",
        default=None,
        help="JSON payload describing market feed inputs",
    )
    parser.add_argument(
        "--skip_macro_refresh",
        action="store_true",
        help="Skip automatic macro refresh even when macro features are included",
    )
    parser.add_argument(
        "--macro_freshness_hours",
        type=int,
        default=24,
        help="Maximum age (hours) before macro artifacts are refreshed",
    )
    parser.add_argument(
        "--macro_series_ids",
        default=None,
        help="Comma-separated list of macro series ids to refresh (defaults to loader configuration)",
    )
    parser.add_argument(
        "--macro_fred_path",
        default=None,
        help="Explicit target path for FRED ML parquet (defaults to data/fred/fred_indicators_ml_format.parquet)",
    )
    parser.add_argument(
        "--vintage_policy",
        default=VintagePolicy.REAL_TIME.value,
        choices=[policy.value for policy in VintagePolicy],
        help="Vintage policy for macro features (real_time or final)",
    )
    parser.add_argument(
        "--vintage_as_of",
        default=None,
        help="ISO8601 timestamp (UTC) limiting macro revisions (optional)",
    )
    parser.add_argument("--validation_min_rows", type=int, default=None)
    parser.add_argument("--validation_min_positive_rate", type=float, default=None)
    parser.add_argument("--validation_max_positive_rate", type=float, default=None)
    parser.add_argument("--validation_min_feature_coverage", type=float, default=None)
    parser.add_argument(
        "--skip_l2_ingest",
        action="store_true",
        help="Skip automatic L2 ingestion even when include_l2 is enabled",
    )
    parser.add_argument(
        "--l2_days",
        type=int,
        default=30,
        help="Number of calendar days to ingest depth data when include_l2 is enabled",
    )
    parser.add_argument(
        "--l2_progress_file",
        default=None,
        help="Optional path for tracking L2 ingestion progress (defaults to <data_dir>/.l2_progress.json)",
    )
    parser.add_argument(
        "--l2_symbols",
        default=None,
        help="Comma-separated list of symbols for L2 ingestion (defaults to Tier 1 universe)",
    )
    parser.add_argument(
        "--l2_tier",
        type=int,
        default=1,
        help="Tier to use for automatic L2 ingestion when symbols are not provided",
    )
    parser.add_argument(
        "--fred_vintage_dir",
        default=None,
        help="Optional ALFRED vintage directory",
    )
    parser.add_argument("--events_dir", default=None, help="Optional normalized events directory")
    parser.add_argument(
        "--student_mode",
        action="store_true",
        help="Build student-mode (L1-only) dataset",
    )
    parser.add_argument(
        "--emit_dataset_events",
        action="store_true",
        help="Emit dataset events via DataRegistry for the TFT build",
    )
    parser.add_argument("--horizon_minutes", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--lookback_periods", type=int, default=30)
    parser.add_argument("--start_iso", default=None, help="Optional start date ISO (YYYY-MM-DD)")
    parser.add_argument("--end_iso", default=None, help="Optional end date ISO (YYYY-MM-DD)")
    parser.add_argument(
        "--chunk_days",
        type=int,
        default=0,
        help="Chunk build by N days (0=disabled)",
    )
    parser.add_argument(
        "--dataset_write_csv",
        action="store_true",
        help="Always write dataset.csv (overrides size-based defaults)",
    )
    parser.add_argument(
        "--dataset_skip_csv",
        action="store_true",
        help="Skip writing dataset.csv (optional dataset_sample.csv still possible)",
    )
    parser.add_argument(
        "--dataset_csv_max_rows",
        type=int,
        default=None,
        help="Row threshold for auto CSV writing",
    )
    parser.add_argument(
        "--dataset_csv_sample_rows",
        type=int,
        default=0,
        help="Write dataset_sample.csv with N rows when full CSV is skipped",
    )
    parser.add_argument(
        "--auto_fill_universe",
        action="store_true",
        help="Automatically backfill market data coverage before dataset build",
    )
    parser.add_argument(
        "--auto_fill_dataset_id",
        default=None,
        help="Dataset identifier for auto-fill (defaults to --dataset_id)",
    )
    parser.add_argument(
        "--auto_fill_instrument_ids",
        default=None,
        help="Comma-separated instrument IDs overriding dataset config for auto-fill",
    )
    parser.add_argument(
        "--auto_fill_l2_days",
        type=int,
        default=None,
        help="Override L2 lookback window for auto-fill (days)",
    )
    parser.add_argument(
        "--auto_fill_skip_l2",
        action="store_true",
        help="Skip L2 ingestion during auto-fill",
    )
    parser.add_argument(
        "--auto_fill_l2_dataset_id",
        default=None,
        help="Dataset identifier for auto-fill L2 ingestion (default DBEQ.BASIC)",
    )
    parser.add_argument(
        "--auto_fill_l2_schema",
        default=None,
        help="Schema to use for auto-fill L2 ingestion (default mbp-10)",
    )
    parser.add_argument(
        "--auto_fill_l2_progress_file",
        default=None,
        help="Progress file path for auto-fill L2 ingestion",
    )
    parser.add_argument(
        "--auto_fill_include_l3",
        action="store_true",
        help="Attempt L3 auto-fill when helpers are available",
    )
    parser.add_argument(
        "--auto_fill_l3_dataset_id",
        default=None,
        help="Dataset identifier for auto-fill L3 ingestion",
    )
    parser.add_argument(
        "--auto_fill_l3_schema",
        default=None,
        help="Schema to use for auto-fill L3 ingestion",
    )
    parser.add_argument(
        "--auto_fill_l3_days",
        type=int,
        default=None,
        help="Override L3 lookback window for auto-fill (days)",
    )
    parser.add_argument(
        "--auto_fill_allow_dataset_l2_ingest",
        action="store_true",
        help="Allow dataset-stage L2 ingestion even when auto-fill runs",
    )
    parser.add_argument(
        "--attach-runtime",
        action="store_true",
        help="Attach MLIntegrationManager after pipeline completion",
    )
    parser.add_argument(
        "--runtime-db-connection",
        default=None,
        help="Override DB connection string for runtime attachment",
    )
    parser.add_argument(
        "--runtime-auto-start-db",
        action="store_true",
        help="Automatically start PostgreSQL when attaching runtime",
    )
    parser.add_argument(
        "--runtime-auto-migrate",
        action="store_true",
        help="Run database migrations when attaching runtime",
    )
    parser.add_argument(
        "--runtime-no-ensure-healthy",
        action="store_true",
        help="Skip health checks during runtime attachment",
    )
    parser.add_argument(
        "--runtime-strict-protocol-validation",
        action="store_true",
        help="Enable strict protocol validation when attaching runtime",
    )
    parser.add_argument(
        "--runtime-skip-validators",
        action="store_true",
        help="Skip metrics/events validators during runtime attachment",
    )

    # HPO
    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--hpo_epochs", type=int, default=2)
    parser.add_argument("--hpo_batch_size", type=int, default=32)
    parser.add_argument("--hpo_tail_rows", type=int, default=5000)
    parser.add_argument("--hpo_limit_groups", type=int, default=50)

    # Teacher training
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--teacher_model_id", default="teacher_model")
    parser.add_argument("--feature_registry_dir", default=None)
    parser.add_argument(
        "--dataset_register_features",
        action="store_true",
        help="Register features during dataset build using feature_registry_dir",
    )
    parser.add_argument("--feature_set_id", default=None)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Lightning accelerator for teacher training",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices for teacher training",
    )
    parser.add_argument(
        "--precision",
        default="32",
        help="Training precision (e.g. 32, 16, 16-mixed, bf16)",
    )
    parser.add_argument("--max_encoder_length", type=int, default=30)
    parser.add_argument("--max_prediction_length", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--attention_head_size", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument(
        "--loss",
        choices=["poisson", "bce"],
        default="poisson",
    )
    parser.add_argument("--pos_weight", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tail_rows", type=int, default=0)
    parser.add_argument("--limit_groups", type=int, default=0)
    parser.add_argument("--val_days", type=int, default=0)
    parser.add_argument("--embargo_hours", type=float, default=24.0)
    parser.add_argument("--purge_gap", type=int, default=0)
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--target_col", default="y")
    parser.add_argument("--time_index_col", default="time_index")
    parser.add_argument("--timestamp_col", default="timestamp")
    parser.add_argument("--group_id_col", default="instrument_id")
    parser.add_argument("--static_categoricals", default=None)
    parser.add_argument("--static_reals", default=None)
    parser.add_argument("--known_future_reals", default=None)
    parser.add_argument("--save_interpretability", action="store_true")
    parser.add_argument("--export_torchscript", action="store_true")
    parser.add_argument("--export_safetensors", action="store_true")
    parser.add_argument("--pretrained_state_path", default=None)
    parser.add_argument("--register_teacher", action="store_true")
    parser.add_argument("--decision_policy", default=None)
    parser.add_argument("--decision_config", default=None)
    parser.add_argument(
        "--prefer_parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer parquet inputs when training the TFT teacher",
    )
    parser.add_argument("--distill_student", action="store_true")
    parser.add_argument("--student_model_id", default="student_model")
    parser.add_argument("--student_parent_model_id", default=None)
    parser.add_argument("--student_model_registry_dir", default=None)
    parser.add_argument("--student_feature_registry_dir", default=None)
    parser.add_argument("--student_feature_set_id", default=None)
    parser.add_argument(
        "--student_objective",
        default="logit_mse",
        choices=["logit_mse", "soft_ce", "hybrid"],
    )
    parser.add_argument("--student_kd_lambda", type=float, default=0.5)
    parser.add_argument("--student_early_stopping", type=int, default=200)
    parser.add_argument("--student_opset", type=int, default=None)
    parser.add_argument("--student_use_val_for_distill", action="store_true")

    # Optional promotions and feature registration
    parser.add_argument("--auto_register_model", action="store_true")
    parser.add_argument("--gates_json", default=None)
    parser.add_argument("--auto_promote", action="store_true")
    parser.add_argument("--deploy_target", default=None)

    parser.add_argument("--auto_register_features", action="store_true")
    parser.add_argument("--feature_metrics_json", default=None)

    # Optional small feature refresh phase
    parser.add_argument("--refresh_features", action="store_true")

    # Promotion stage 2 (walk-forward + cost-aware backtest)
    parser.add_argument("--promote_stage2", action="store_true")
    parser.add_argument("--stage2_gates_json", default=None)
    parser.add_argument("--stage2_cost_bps", type=float, default=0.0)
    parser.add_argument(
        "--stage2_engine",
        choices=["returns", "backtest"],
        default="returns",
        help="Stage 2 engine: returns (default) or backtest (advisory)",
    )
    parser.add_argument("--stage2_commission_bps", type=float, default=0.0)
    parser.add_argument("--stage2_slippage_bps", type=float, default=0.0)
    parser.add_argument(
        "--final_model_id",
        default=None,
        help="Model ID to promote in stage 2 (optional)",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def _extract_config_args(
    raw_args: Sequence[str],
) -> tuple[str | None, str | None, list[str]]:
    """
    Split ``raw_args`` into config path, stage override, and remaining args.
    """
    config_path: str | None = None
    stage_override: str | None = None
    passthrough: list[str] = []
    idx = 0
    while idx < len(raw_args):
        token = raw_args[idx]
        if token == "--config":
            if idx + 1 >= len(raw_args):
                raise SystemExit("--config requires a file path")
            config_path = raw_args[idx + 1]
            idx += 2
            continue
        if token.startswith("--config="):
            config_path = token.split("=", 1)[1]
            idx += 1
            continue
        if token == "--stage":
            if idx + 1 >= len(raw_args):
                raise SystemExit("--stage requires a value")
            stage_override = raw_args[idx + 1]
            idx += 2
            continue
        if token.startswith("--stage="):
            stage_override = token.split("=", 1)[1]
            idx += 1
            continue
        passthrough.append(token)
        idx += 1
    return config_path, stage_override, passthrough


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    config_path, stage_override, passthrough = _extract_config_args(raw_args)
    stage_default: Stage | None = None

    if config_path is not None:
        run_cfg = load_orchestrator_run_config(config_path)
        stage_default = run_cfg.stage
        if stage_override is not None:
            try:
                stage_for_args = Stage(stage_override)
            except ValueError as exc:  # pragma: no cover - defensive
                raise SystemExit(f"Unsupported stage '{stage_override}'") from exc
        else:
            stage_for_args = run_cfg.stage
        ingestion_cfg = run_cfg.ingestion if stage_for_args in {Stage.FULL, Stage.INGEST} else None
        config_args: list[str]
        if run_cfg.dataset is None:
            if stage_for_args is not Stage.INGEST:
                raise SystemExit("Dataset configuration is required for non-ingestion stages")
            effective_ingestion = ingestion_cfg or IngestionStageConfig(enabled=True)
            config_args = _ingestion_config_to_args(effective_ingestion)
        else:
            orchestrator_cfg = run_cfg.compose_orchestrator_config()
            config_args = to_pipeline_args(orchestrator_cfg, ingestion=ingestion_cfg)
        combined_args = config_args + passthrough
        if stage_override is not None:
            combined_args += ["--stage", stage_override]
        args = parse_args(combined_args)
        if args.stage is None:
            args.stage = stage_for_args.value
    else:
        if stage_override is not None:
            passthrough += ["--stage", stage_override]
        args = parse_args(passthrough)

    return _execute_with_namespace(args, stage_default=stage_default)


def _execute_with_namespace(
    args: argparse.Namespace,
    *,
    stage_default: Stage | None = None,
) -> int:
    _run_id: str = f"orch_{_uuid.uuid4().hex[:12]}"
    bind_log_context(run_id=_run_id, component="ml.pipeline_orchestrator")

    from ml.core.integration import MLIntegrationManager

    mgr = MLIntegrationManager(
        db_connection=args.db,
        auto_start_postgres=False,
        auto_migrate=True,
        ensure_healthy=False,
    )
    data_store = getattr(mgr, "data_store", None)
    if data_store is None:
        logger.info(
            "DataStore unavailable; falling back to catalog-only runtime attachment",
        )
    if mgr.data_registry is None:
        raise SystemExit(
            "DataRegistry unavailable; configure ML_DB_CONNECTION for pipeline orchestration"
        )

    registry = mgr.data_registry
    manifest_resolver = None
    if registry is not None and hasattr(registry, "get_manifest"):
        manifest_resolver = cast(RegistryProtocol, registry).get_manifest

    parquet_catalog: Any | None = None
    raw_writer: RawIngestionWriterProtocol | None = None
    if args.catalog_path:
        try:
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

            parquet_catalog = ParquetDataCatalog(args.catalog_path)
        except Exception as exc:  # pragma: no cover - import env issue
            raise SystemExit(f"ParquetDataCatalog unavailable: {exc}")
        raw_writer = ParquetCatalogRawWriter(parquet_catalog)

    coverage: CoverageProviderProtocol
    if args.coverage_mode == "catalog":
        if parquet_catalog is None:
            raise SystemExit("catalog_path is required for catalog coverage mode")
        coverage = CatalogCoverageProvider(catalog_path=args.catalog_path)
    else:
        coverage = SqlCoverageProvider(connection_string=args.db)

    mode_tokens = _resolve_write_mode_tokens(args.write_mode)
    writer_chain: list[MarketDataWriterProtocol] = []

    if "sql" in mode_tokens:
        writer_chain.append(SqlMarketDataWriter(connection_string=args.db))

    if "datastore" in mode_tokens:
        if data_store is None:
            logger.warning(
                "write_mode requested DataStore persistence but DataStore is unavailable; "
                "skipping datastore writer",
            )
        else:
            from ml.stores.protocols import DataStoreFacadeProtocol

            writer_chain.append(
                DataStoreMarketDataWriter(
                    store=cast(DataStoreFacadeProtocol, data_store),
                ),
            )

    if "parquet" in mode_tokens:
        if parquet_catalog is None:
            raise SystemExit("catalog_path is required when write_mode includes parquet")
        writer_chain.append(
            ParquetCatalogMarketDataWriter(
                catalog=parquet_catalog,
                manifest_resolver=manifest_resolver,
            ),
        )

        if not writer_chain:
            if data_store is not None:
                from ml.stores.protocols import DataStoreFacadeProtocol

                writer_chain.append(
                    DataStoreMarketDataWriter(
                        store=cast(DataStoreFacadeProtocol, data_store),
                    ),
                )
        elif parquet_catalog is not None:
            writer_chain.append(
                ParquetCatalogMarketDataWriter(
                    catalog=parquet_catalog,
                    manifest_resolver=manifest_resolver,
                ),
            )
        else:
            raise SystemExit("No ingestion writers available; configure DataStore or catalog")

    primary_writer = writer_chain[0]
    mirror_writers = tuple(writer_chain[1:])
    writer = FanoutMarketDataWriter(primary=primary_writer, mirrors=mirror_writers)
    integration_factory: Callable[..., IntegrationManagerProtocol] | None = cast(
        Callable[..., IntegrationManagerProtocol],
        MLIntegrationManager,
    )

    ingestor: object | None = None
    ingestion_service: DatabentoIngestionService | None = None
    dataset_discovery: DatasetDiscoveryService | None = None
    need_databento = bool(args.ingest or getattr(args, "auto_fill_universe", False))
    if need_databento:
        api_key = os.getenv("DATABENTO_API_KEY", "").strip()
        if api_key:
            client = DatabentoAPIClient(api_key=api_key)
            ingestor = DatabentoIngestor(client=client)
            discovery_policy = DiscoveryPolicy.from_env(os.environ)
            resolver = DatabentoSymbologyResolver(
                client=client.symbology_client,
            )
            dataset_discovery = DatasetDiscoveryService(
                metadata=client.metadata_client,
                policy=discovery_policy,
                resolver=resolver,
            )
            try:
                ingestion_service = DatabentoIngestionService.from_env()
            except Exception as exc:  # pragma: no cover - runtime warning only
                logging.getLogger(__name__).warning(
                    "Failed to initialise ingestion service: %s",
                    exc,
                )
        elif args.ingest:
            logging.getLogger(__name__).warning(
                "--ingest requested but DATABENTO_API_KEY is missing; skipping",
            )

    from ml.scripts.build_tft_dataset import main as build_main
    from ml.training.teacher.tft_cli import main as teacher_main

    try:
        from ml.cli.hpo_tft import main as _hpo_main

        hpo_main_cli: _CliMain | None = _hpo_main
    except Exception:  # pragma: no cover - optional dependency
        hpo_main_cli = None

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=registry,
        ingestor=ingestor if ingestor is not None else None,
        build_main=build_main,
        hpo_main=hpo_main_cli,
        teacher_main=teacher_main,
        raw_writer=raw_writer,
        service=ingestion_service,
        model_registry=getattr(mgr, "model_registry", None),
        feature_registry=getattr(mgr, "feature_registry", None),
        strategy_registry=getattr(mgr, "strategy_registry", None),
        feature_store=getattr(mgr, "feature_store", None),
        model_store=getattr(mgr, "model_store", None),
        strategy_store=getattr(mgr, "strategy_store", None),
        data_store=getattr(mgr, "data_store", None),
        partition_manager=getattr(mgr, "partition_manager", None),
        integration_manager_factory=integration_factory,
        dataset_discovery=dataset_discovery,
    )

    # Store write_mode_tokens for determining storage_kind
    orch.write_mode_tokens = mode_tokens

    # Deferred ingestion block runs after dataset config is prepared
    data_dir_effective = Path(args.data_dir)
    if args.catalog_path and str(args.data_dir) == "data/tier1":
        data_dir_effective = Path(args.catalog_path)

    raw_macro_series_ids = tuple(
        item.strip()
        for item in (str(args.macro_series_ids).split(",") if args.macro_series_ids else [])
        if item.strip()
    )
    macro_series_ids: tuple[str, ...] | None = raw_macro_series_ids or None
    if bool(args.include_macro) and macro_series_ids is None:
        macro_series_ids = DEFAULT_MACRO_SERIES

    raw_instrument_ids = tuple(
        item.strip()
        for item in (str(args.instrument_ids).split(",") if args.instrument_ids else [])
        if item.strip()
    )
    instrument_ids: tuple[str, ...] | None = raw_instrument_ids or None
    static_categoricals = _parse_csv_tuple(args.static_categoricals)
    static_reals = _parse_csv_tuple(args.static_reals)
    known_future_reals = _parse_csv_tuple(args.known_future_reals)

    validation_cfg = _build_validation_config_from_args(
        args,
        macro_series_ids,
    )

    auto_fill_enabled = bool(getattr(args, "auto_fill_universe", False))
    auto_fill_blocks_l2 = auto_fill_enabled and not bool(
        getattr(args, "auto_fill_allow_dataset_l2_ingest", False),
    )

    if args.include_l2 and not args.skip_l2_ingest and not auto_fill_blocks_l2:
        l2_symbols = None
        if args.l2_symbols:
            l2_symbols = tuple(
                s.strip().upper() for s in str(args.l2_symbols).split(",") if s.strip()
            )
        l2_tier = None if l2_symbols else args.l2_tier
        progress_file = (
            Path(args.l2_progress_file)
            if args.l2_progress_file
            else data_dir_effective / ".l2_progress.json"
        )
        try:
            l2_config = PopulateL2TaskConfig(
                data_dir=data_dir_effective,
                progress_file=progress_file,
                symbols=l2_symbols,
                tier=l2_tier,
                days=int(args.l2_days),
            )
            symbols_desc = f"custom:{len(l2_symbols)}" if l2_symbols else f"tier:{l2_tier}"
            logger.info(
                "Starting L2 ingestion (symbols=%s, days=%s)",
                symbols_desc,
                args.l2_days,
            )
            populate_l2_efficient(l2_config)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("L2 ingestion failed: %s", exc, exc_info=True)
            raise

    try:
        effective_vintage_policy = VintagePolicy(str(args.vintage_policy))
    except ValueError as exc:
        raise SystemExit(f"Invalid vintage_policy: {args.vintage_policy}") from exc

    market_dataset_id = (
        args.market_dataset_id
        or getattr(args, "auto_fill_dataset_id", None)
        or getattr(args, "dataset_id", "")
    )

    market_inputs_tuple = _parse_market_inputs_json(getattr(args, "market_inputs_json", None))

    end_iso = args.end_iso
    start_iso = args.start_iso
    if start_iso is None and end_iso:
        start_iso = _compute_window_start_iso(end_iso=end_iso)

    if args.dataset_write_csv and args.dataset_skip_csv:
        raise SystemExit("--dataset_write_csv and --dataset_skip_csv are mutually exclusive")
    if args.dataset_write_csv:
        write_csv: bool | None = True
    elif args.dataset_skip_csv:
        write_csv = False
    else:
        write_csv = None
    default_csv_max_rows = None
    for field in fields(DatasetBuildConfig):
        if field.name == "csv_max_rows":
            if isinstance(field.default, int):
                default_csv_max_rows = field.default
            break
    if default_csv_max_rows is None:
        default_csv_max_rows = 1_000_000
    csv_max_rows = (
        int(args.dataset_csv_max_rows)
        if args.dataset_csv_max_rows is not None
        else int(default_csv_max_rows)
    )

    ds_cfg = DatasetBuildConfig(
        data_dir=str(data_dir_effective),
        symbols=str(args.symbols),
        out_dir=str(args.out_dir),
        dataset_id=str(getattr(args, "dataset_id", "tft_dataset")),
        market_dataset_id=str(market_dataset_id) if market_dataset_id else None,
        market_inputs=market_inputs_tuple,
        include_macro=bool(args.include_macro),
        macro_lag_days=int(args.macro_lag_days),
        include_micro=bool(args.include_micro),
        include_l2=bool(args.include_l2),
        include_events=bool(getattr(args, "include_events", False)),
        include_calendar=bool(getattr(args, "include_calendar", False)),
        instrument_ids=instrument_ids,
        auto_refresh_macro=not bool(args.skip_macro_refresh),
        macro_staleness_hours=int(args.macro_freshness_hours),
        macro_series_ids=macro_series_ids,
        macro_fred_path=str(args.macro_fred_path) if args.macro_fred_path else None,
        fred_vintage_dir=str(args.fred_vintage_dir) if args.fred_vintage_dir else None,
        events_dir=str(args.events_dir) if args.events_dir else None,
        student_mode=bool(args.student_mode),
        emit_dataset_events=bool(getattr(args, "emit_dataset_events", False)),
        horizon_minutes=int(args.horizon_minutes),
        threshold=float(args.threshold),
        lookback_periods=int(args.lookback_periods),
        start_iso=start_iso,
        end_iso=end_iso,
        chunk_days=int(args.chunk_days),
        write_csv=write_csv,
        csv_max_rows=csv_max_rows,
        csv_sample_rows=int(args.dataset_csv_sample_rows),
        register_features=bool(args.dataset_register_features),
        feature_registry_dir=args.feature_registry_dir,
        feature_role="teacher",
        validation=validation_cfg,
        vintage_policy=effective_vintage_policy,
        vintage_as_of=args.vintage_as_of,
    )

    prepare_cfg = getattr(orch, "_prepare_dataset_config", None)
    if callable(prepare_cfg):
        ds_cfg = prepare_cfg(ds_cfg)

    auto_fill_cfg = _build_auto_fill_config_from_args(args, ds_cfg)
    ingestion_cfg = _build_ingestion_config_from_args(args, ds_cfg)

    stage_token = args.stage or (stage_default.value if stage_default is not None else None)
    stage = Stage(stage_token) if stage_token is not None else Stage.FULL

    ingestion_requested = bool(ingestion_cfg.enabled or auto_fill_cfg.enabled)
    if stage in {Stage.FULL, Stage.INGEST} and ingestion_requested:
        rc = _run_ingestion_stage(
            orch=orch,
            ds_cfg=ds_cfg,
            auto_fill_cfg=auto_fill_cfg,
            ingestion_cfg=ingestion_cfg,
            ingestor=ingestor,
            ingestion_service=ingestion_service,
        )
        if rc != 0:
            return rc
        if stage is Stage.INGEST:
            return 0
    elif stage is Stage.INGEST:
        logger.info("Ingestion stage requested but ingestion inputs are disabled")
        return 0

    hpo_cfg = HPOConfig(
        enabled=bool(args.hpo),
        epochs=int(args.hpo_epochs),
        batch_size=int(args.hpo_batch_size),
        tail_rows=int(args.hpo_tail_rows),
        limit_groups=int(args.hpo_limit_groups),
    )

    teacher_cfg = TeacherTrainConfig(
        enabled=bool(args.train),
        model_id=str(args.teacher_model_id),
        feature_registry_dir=args.feature_registry_dir,
        feature_set_id=args.feature_set_id,
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        dataloader_workers=int(args.dataloader_workers),
        accelerator=str(args.accelerator),
        devices=int(args.devices),
        precision=str(args.precision),
        max_encoder_length=int(args.max_encoder_length),
        max_prediction_length=int(args.max_prediction_length),
        hidden_size=int(args.hidden_size),
        lstm_layers=int(args.lstm_layers),
        attention_head_size=int(args.attention_head_size),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        loss=str(args.loss),
        pos_weight=args.pos_weight,
        seed=None if args.seed is None else int(args.seed),
        tail_rows=int(args.tail_rows),
        limit_groups=int(args.limit_groups),
        val_days=int(args.val_days),
        embargo_hours=float(args.embargo_hours),
        purge_gap=int(args.purge_gap),
        cv_splits=int(args.cv_splits),
        test_fraction=float(args.test_fraction),
        target_col=str(args.target_col),
        time_index_col=str(args.time_index_col),
        timestamp_col=str(args.timestamp_col),
        group_id_col=str(args.group_id_col),
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        known_future_reals=known_future_reals,
        save_interpretability=bool(args.save_interpretability),
        export_torchscript=bool(args.export_torchscript),
        export_safetensors=bool(args.export_safetensors),
        pretrained_state_path=args.pretrained_state_path,
        register_teacher=bool(args.register_teacher),
        decision_policy=args.decision_policy,
        decision_config=args.decision_config,
        prefer_parquet=bool(args.prefer_parquet),
    )

    student_cfg = StudentDistillConfig(
        enabled=bool(args.distill_student),
        model_id=str(args.student_model_id),
        parent_model_id=args.student_parent_model_id,
        model_registry_dir=args.student_model_registry_dir,
        feature_registry_dir=args.student_feature_registry_dir,
        feature_set_id=args.student_feature_set_id,
        objective=str(args.student_objective),
        kd_lambda=float(args.student_kd_lambda),
        early_stopping=int(args.student_early_stopping),
        opset=None if args.student_opset is None else int(args.student_opset),
        use_val_for_distill=bool(args.student_use_val_for_distill),
    )

    promotions_cfg = PromotionsConfig(
        auto_register_model=bool(args.auto_register_model),
        gates_json=args.gates_json,
        auto_promote=bool(args.auto_promote),
        deploy_target=args.deploy_target,
        auto_register_features=bool(args.auto_register_features),
        feature_metrics_json=args.feature_metrics_json,
        refresh_features=bool(args.refresh_features),
    )

    integration_cfg = IntegrationConfig(
        enabled=bool(args.attach_runtime),
        db_connection=(args.runtime_db_connection or args.db),
        auto_start_postgres=bool(args.runtime_auto_start_db),
        auto_migrate=bool(args.runtime_auto_migrate),
        ensure_healthy=not bool(args.runtime_no_ensure_healthy),
        strict_protocol_validation=(True if args.runtime_strict_protocol_validation else None),
        run_validators=not bool(args.runtime_skip_validators),
    )

    orchestrator_cfg = OrchestratorConfig(
        dataset=ds_cfg,
        hpo=hpo_cfg,
        teacher=teacher_cfg,
        student=student_cfg,
        promotions=promotions_cfg,
        integration=integration_cfg if integration_cfg.enabled else None,
        auto_fill=auto_fill_cfg if auto_fill_cfg.enabled else None,
    )

    return _execute_stage(
        orch=orch,
        orchestrator_cfg=orchestrator_cfg,
        stage=stage,
        ds_cfg=ds_cfg,
        auto_fill_cfg=auto_fill_cfg,
        args=args,
        ingestor=ingestor,
        ingestion_service=ingestion_service,
    )


def _run_ingestion_stage(
    *,
    orch: MLPipelineOrchestrator,
    ds_cfg: DatasetBuildConfig | None,
    auto_fill_cfg: AutoFillUniverseConfig,
    ingestion_cfg: IngestionStageConfig,
    ingestor: object | None,
    ingestion_service: DatabentoIngestionService | None,
) -> int:
    """
    Run ingestion/backfill operations prior to dataset construction.
    """
    metrics = _IngestionMetrics.default()
    component_label = "pipeline_orchestrator_ingestion"
    stage_status = "skipped"
    work_performed = False
    stage_start = time.perf_counter()
    fallback_reports: list[dict[str, object]] = []
    coverage_metric_emitted = False
    file_metric_emitted = False

    def _finalize() -> None:
        elapsed = time.perf_counter() - stage_start
        metrics.runs_total.labels(component=component_label, status=stage_status).inc()
        metrics.latency_seconds.labels(component=component_label, status=stage_status).observe(
            elapsed
        )

    def _normalise_schema_for_lookback(raw_schema: str | None) -> str:
        token = (raw_schema or "bars").lower()
        if "ohlcv" in token or "bar" in token:
            return "bars"
        if "tbbo" in token or "bbo" in token or "quote" in token:
            return "quotes"
        if "trade" in token:
            return "trades"
        if "mbp" in token or token.startswith(("l2", "l3")):
            return "mbp"
        return token

    def _attempt_primary_ingestion(
        plan_items: tuple[_IngestionPlanItem, ...],
        *,
        policy: CoveragePolicy,
    ) -> _IngestionAttemptReport:
        bindings = tuple(item.binding for item in plan_items if item.binding is not None)
        context: dict[str, object] = {
            "stage": Stage.INGEST.value,
            "attempt": "primary",
            "binding_count": len(bindings),
            "datasets": sorted({item.dataset_id for item in plan_items}),
        }
        rows_written = 0
        attempted_windows = 0
        try:
            for item in plan_items:
                if item.binding is None:
                    continue
                binding = item.binding
                schema_token = _normalise_schema_for_lookback(binding.schema or item.schema)
                lookback_days = get_max_lookback_days(schema_token, policy)
                results = orch.backfill_binding(
                    binding=binding,
                    lookback_days=lookback_days,
                )
                for window_list in results.values():
                    rows_written += window_list.rows_written
                    attempted_windows += window_list.attempted_window_count
            context["rows_written"] = rows_written
            context["attempted_windows"] = attempted_windows
            return _IngestionAttemptReport(success=True, context=context)
        except Exception as exc:  # pragma: no cover - defensive guard
            context["error_type"] = exc.__class__.__name__
            return _IngestionAttemptReport(
                success=False,
                context=context,
                reason=str(exc),
            )

    def _attempt_coverage_ingestion(
        plan_items: tuple[_IngestionPlanItem, ...],
        *,
        policy: CoveragePolicy,
    ) -> _IngestionAttemptReport:
        context: dict[str, object] = {
            "stage": Stage.INGEST.value,
            "attempt": "coverage",
            "plan_items": len(plan_items),
            "datasets": sorted({item.dataset_id for item in plan_items}),
            "instrument_total": sum(len(item.instrument_ids) for item in plan_items),
        }
        window_count = 0
        try:
            for item in plan_items:
                if not item.instrument_ids:
                    continue
                for instrument_id in item.instrument_ids:
                    windows = orch.backfill_coverage(
                        dataset_id=item.dataset_id,
                        schema=item.schema,
                        instrument_id=instrument_id,
                        policy=policy,
                    )
                    window_count += len(windows)
            context["window_count"] = window_count
            return _IngestionAttemptReport(success=True, context=context)
        except Exception as exc:  # pragma: no cover - defensive guard
            context["error_type"] = exc.__class__.__name__
            return _IngestionAttemptReport(
                success=False,
                context=context,
                reason=str(exc),
            )

    def _attempt_manual_ingestion(
        plan_items: tuple[_IngestionPlanItem, ...],
        *,
        lookback_days: int,
        policy: CoveragePolicy,
    ) -> _IngestionAttemptReport:
        context: dict[str, object] = {
            "stage": Stage.INGEST.value,
            "attempt": "manual",
            "lookback_days": lookback_days,
            "plan_items": len(plan_items),
            "datasets": sorted({item.dataset_id for item in plan_items}),
            "instrument_total": sum(len(item.instrument_ids) for item in plan_items),
        }
        if context["instrument_total"] == 0:
            return _IngestionAttemptReport(
                success=False,
                context=context,
                reason="no_instruments",
            )
        rows_written = 0
        attempted_windows = 0
        try:
            for item in plan_items:
                if not item.instrument_ids:
                    continue
                schema_token = _normalise_schema_for_lookback(item.schema)
                effective_lookback = lookback_days or get_max_lookback_days(schema_token, policy)
                for instrument_id in item.instrument_ids:
                    windows = orch.backfill(
                        dataset_id=item.dataset_id,
                        schema=item.schema,
                        instrument_id=instrument_id,
                        lookback_days=effective_lookback,
                    )
                    rows_written += windows.rows_written
                    attempted_windows += windows.attempted_window_count
            context["rows_written"] = rows_written
            context["attempted_windows"] = attempted_windows
            return _IngestionAttemptReport(success=True, context=context)
        except Exception as exc:  # pragma: no cover - defensive guard
            context["error_type"] = exc.__class__.__name__
            return _IngestionAttemptReport(
                success=False,
                context=context,
                reason=str(exc),
            )

    def _find_existing_artifact() -> Path | None:
        if ds_cfg is None:
            return None
        candidates: list[Path] = []
        out_dir = Path(ds_cfg.out_dir)
        data_dir = Path(ds_cfg.data_dir)
        dataset_id_local = ds_cfg.dataset_id
        candidates.append(out_dir / "dataset_metadata.json")
        if dataset_id_local:
            candidates.append(out_dir / dataset_id_local / "dataset_metadata.json")
            candidates.append(data_dir / dataset_id_local / "dataset_metadata.json")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    ingestion_requested = ingestion_cfg.enabled or auto_fill_cfg.enabled
    if not ingestion_requested:
        logger.info(
            "Ingestion stage skipped (disabled)",
            extra={"stage": Stage.INGEST.value, "status": stage_status},
        )
        return 0

    symbol_map_for_ingestion = _collect_symbol_map(ds_cfg=ds_cfg, ingestion_cfg=ingestion_cfg)

    discovery_inputs: tuple[MarketDatasetInput, ...] | None = None
    discover_method = getattr(orch, "_discover_market_inputs", None)
    discovery_service = getattr(orch, "dataset_discovery", None)
    if (
        ingestion_cfg.market_inputs is None
        and callable(discover_method)
        and discovery_service is not None
        and symbol_map_for_ingestion
    ):
        schema_token = _normalize_schema_token(ingestion_cfg.schema)
        end_ns = time.time_ns()
        lookback_days = max(int(ingestion_cfg.lookback_days or 1), 1)
        start_ns = end_ns - lookback_days * DAY_NS
        discovery_inputs = discover_method(
            symbol_map=symbol_map_for_ingestion,
            schema=schema_token,
            start_ns=start_ns,
            end_ns=end_ns,
            dataset_hint=ingestion_cfg.market_dataset_id or ingestion_cfg.dataset_id,
        )
    if discovery_inputs:
        dataset_id_hint = ingestion_cfg.dataset_id or discovery_inputs[0].dataset_id
        ingestion_cfg = replace(
            ingestion_cfg,
            market_inputs=discovery_inputs,
            dataset_id=dataset_id_hint,
        )

    try:
        plan_items = _build_ingestion_plan(ds_cfg=ds_cfg, ingestion_cfg=ingestion_cfg)
        binding_count = sum(1 for item in plan_items if item.binding is not None)
        datasets_in_plan = sorted({item.dataset_id for item in plan_items})
        schema_set = sorted({item.schema for item in plan_items})

        ingestion_policy = getattr(ingestion_service, "_policy", None)
        if ingestion_policy is not None:
            for item in plan_items:
                try:
                    ingestion_policy.allow_dataset(item.dataset_id)
                except Exception:
                    logger.debug(
                        "Unable to extend ingestion coverage policy",
                        exc_info=True,
                        extra={
                            "dataset_id": item.dataset_id,
                            "stage": Stage.INGEST.value,
                        },
                    )

        ingestion_coordinator = getattr(orch, "_ingestion_coordinator", None)
        registration_fn = None
        if ingestion_coordinator is not None:
            registration_fn = getattr(ingestion_coordinator, "_ensure_dataset_registered", None)

        should_register = getattr(orch, "data_store", None) is not None and callable(
            registration_fn,
        )
        if should_register and plan_items and registration_fn is not None:
            location_root = Path(
                ds_cfg.data_dir if ds_cfg is not None else (ingestion_cfg.catalog_path or "ml_out"),
            )
            for item in plan_items:
                try:
                    registration = cast(Callable[..., None], registration_fn)
                    registration(
                        dataset_id=item.dataset_id,
                        dataset_type=map_schema_to_dataset_type(item.schema),
                        location=str(location_root),
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.debug(
                        "Dataset auto-registration skipped",
                        exc_info=True,
                        extra={
                            "stage": Stage.INGEST.value,
                            "dataset_id": item.dataset_id,
                            "schema": item.schema,
                            "reason": str(exc),
                        },
                    )

        logger.info(
            "Ingestion stage starting",
            extra={
                "stage": Stage.INGEST.value,
                "plan_items": len(plan_items),
                "binding_count": binding_count,
                "datasets": datasets_in_plan,
                "schemas": schema_set,
            },
        )

        if auto_fill_cfg.enabled:
            if ds_cfg is None:
                logger.warning(
                    "Auto-fill requested but dataset configuration missing; skipping",
                    extra={"stage": Stage.INGEST.value},
                )
            else:
                work_performed = True
                logger.info(
                    "Executing auto-fill ingestion",
                    extra={
                        "stage": Stage.INGEST.value,
                        "symbol_count": len([s for s in ds_cfg.symbols.split(",") if s.strip()]),
                        "instrument_count": (
                            0 if ds_cfg.instrument_ids is None else len(ds_cfg.instrument_ids)
                        ),
                    },
                )
                auto_fill = getattr(orch, "_auto_fill_universe", None)
                if callable(auto_fill):
                    auto_fill(ds_cfg, auto_fill_cfg)

        if not ingestion_cfg.enabled:
            stage_status = "success" if work_performed else "skipped"
            logger.info(
                "Ingestion stage skipped (disabled)",
                extra={"stage": Stage.INGEST.value, "status": stage_status},
            )
            return 0

        work_performed = True

        if ingestor is None or ingestion_service is None:
            stage_status = "degraded"
            missing_key = not bool(os.getenv("DATABENTO_API_KEY", "").strip())
            detail = (
                "missing_databento_api_key" if missing_key else "ingestion_components_unavailable"
            )
            logger.error(
                "Databento ingestion unavailable; running in degraded mode",
                extra={"stage": Stage.INGEST.value, "detail": detail},
            )
            metrics.fallback_total.labels(component=component_label, level="dummy").inc()
            return 0

        policy = CoveragePolicy.from_env()
        primary_bindings = tuple(item.binding for item in plan_items if item.binding is not None)
        if primary_bindings:
            primary_report = _attempt_primary_ingestion(plan_items, policy=policy)
            if primary_report.success:
                stage_status = "success"
                logger.info(
                    "Ingestion completed via primary bindings",
                    extra={**primary_report.context},
                )
                return 0
            fallback_reports.append(
                {
                    "level": "primary",
                    "reason": primary_report.reason or "unknown",
                    **primary_report.context,
                },
            )
        else:
            fallback_reports.append(
                {
                    "level": "primary",
                    "reason": "no_bindings",
                    "stage": Stage.INGEST.value,
                },
            )

        coverage_candidates = tuple(item for item in plan_items if item.instrument_ids)
        if coverage_candidates:
            metrics.fallback_total.labels(component=component_label, level="cached").inc()
            coverage_metric_emitted = True
            coverage_report = _attempt_coverage_ingestion(
                coverage_candidates,
                policy=policy,
            )
            if coverage_report.success:
                stage_status = "success"
                logger.info(
                    "Ingestion fallback succeeded via cached coverage",
                    extra={**coverage_report.context},
                )
                return 0
            fallback_reports.append(
                {
                    "level": "cached",
                    "reason": coverage_report.reason or "unknown",
                    **coverage_report.context,
                },
            )

        metrics.fallback_total.labels(component=component_label, level="file").inc()
        file_metric_emitted = True
        manual_report = _attempt_manual_ingestion(
            plan_items,
            lookback_days=int(ingestion_cfg.lookback_days),
            policy=policy,
        )
        if manual_report.success:
            stage_status = "success"
            logger.info(
                "Ingestion fallback succeeded via manual lookback",
                extra={**manual_report.context},
            )
            return 0
        fallback_reports.append(
            {
                "level": "file",
                "reason": manual_report.reason or "unknown",
                **manual_report.context,
            },
        )

        artifact_path = _find_existing_artifact()
        if artifact_path is not None:
            stage_status = "degraded"
            if not file_metric_emitted:
                metrics.fallback_total.labels(component=component_label, level="file").inc()
            logger.warning(
                "Using existing dataset artifacts as ingestion fallback",
                extra={
                    "stage": Stage.INGEST.value,
                    "artifact": str(artifact_path),
                },
            )
            return 0

        stage_status = "error"
        if not coverage_metric_emitted:
            metrics.fallback_total.labels(component=component_label, level="cached").inc()
        metrics.fallback_total.labels(component=component_label, level="dummy").inc()
        logger.error(
            "Ingestion fallback exhausted; no viable data sources",
            extra={
                "stage": Stage.INGEST.value,
                "datasets": datasets_in_plan,
                "schemas": schema_set,
                "reports": fallback_reports,
            },
        )
        return 1
    except IngestionError as exc:
        stage_status = "error"
        logger.error(
            "Ingestion stage failed",
            extra={"stage": Stage.INGEST.value, "error": str(exc)},
            exc_info=True,
        )
        if not coverage_metric_emitted:
            metrics.fallback_total.labels(component=component_label, level="cached").inc()
        metrics.fallback_total.labels(component=component_label, level="dummy").inc()
        return 1
    except Exception as exc:  # pragma: no cover - defensive guard
        stage_status = "error"
        logger.exception(
            "Unexpected ingestion stage failure",
            extra={"stage": Stage.INGEST.value, "error": str(exc)},
        )
        if not coverage_metric_emitted:
            metrics.fallback_total.labels(component=component_label, level="cached").inc()
        metrics.fallback_total.labels(component=component_label, level="dummy").inc()
        return 1
    finally:
        _finalize()


def _dataset_only_config(cfg: OrchestratorConfig) -> OrchestratorConfig:
    """
    Return a copy of ``cfg`` with training/promotions disabled.
    """
    hpo_disabled = replace(cfg.hpo, enabled=False)
    teacher_disabled = replace(cfg.teacher, enabled=False)
    student_disabled = replace(cfg.student, enabled=False)
    return replace(
        cfg,
        hpo=hpo_disabled,
        teacher=teacher_disabled,
        student=student_disabled,
        promotions=None,
        integration=None,
    )


def _execute_stage(
    *,
    orch: MLPipelineOrchestrator,
    orchestrator_cfg: OrchestratorConfig,
    stage: Stage,
    ds_cfg: DatasetBuildConfig,
    auto_fill_cfg: AutoFillUniverseConfig,
    args: argparse.Namespace,
    ingestor: object | None,
    ingestion_service: DatabentoIngestionService | None,
) -> int:
    """
    Execute the requested pipeline ``stage`` using the prepared orchestrator.
    """
    if stage is Stage.DATASET:
        dataset_only_cfg = _dataset_only_config(orchestrator_cfg)
        return orch.run(dataset_only_cfg)
    if stage is Stage.TRAIN:
        return orch.run_training_only(orchestrator_cfg)
    if stage is Stage.FULL:
        return orch.run(orchestrator_cfg)
    # Stage.INGEST handled earlier; reaching here implies nothing to do.
    return 0


def _parse_market_inputs_json(
    value: str | None,
) -> tuple[MarketDatasetInput, ...] | None:
    """
    Parse CLI-provided JSON payload into MarketDatasetInput entries.
    """
    if value is None:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"market_inputs_json must be valid JSON: {exc}") from exc

    items: list[object]
    if isinstance(payload, str | dict):
        items = [payload]
    elif isinstance(payload, list):
        items = list(payload)
    else:
        raise SystemExit("market_inputs_json must encode a list, object, or descriptor string")

    inputs: list[MarketDatasetInput] = []
    for entry in items:
        if isinstance(entry, str):
            inputs.append(MarketDatasetInput(descriptor_id=entry))
            continue
        if isinstance(entry, dict):
            descriptor_id = entry.get("descriptor_id")
            dataset_id = entry.get("dataset_id")
            if descriptor_id is None and dataset_id is None:
                raise SystemExit("market_inputs_json entries require descriptor_id or dataset_id")

            symbols_field = entry.get("symbols")
            symbols_tuple: tuple[str, ...] | None
            if symbols_field is None:
                symbols_tuple = None
            elif isinstance(symbols_field, str):
                symbols_tuple = (
                    tuple(
                        token.strip().upper() for token in symbols_field.split(",") if token.strip()
                    )
                    or None
                )
            elif isinstance(symbols_field, list | tuple):
                symbols_tuple = (
                    tuple(
                        str(token).strip().upper() for token in symbols_field if str(token).strip()
                    )
                    or None
                )
            else:
                raise SystemExit("market_inputs_json symbols must be string or iterable")

            schema_override = entry.get("schema") or entry.get("schema_override")
            storage_raw = entry.get("storage_kind") or entry.get("storage_kind_override")
            storage_kind = None
            if storage_raw is not None:
                try:
                    storage_kind = coerce_storage_kind(storage_raw)
                except ValueError as exc:  # pragma: no cover - defensive guard
                    raise SystemExit(
                        f"Invalid storage_kind '{storage_raw}' in market_inputs_json",
                    ) from exc

            inputs.append(
                MarketDatasetInput(
                    descriptor_id=str(descriptor_id) if descriptor_id is not None else None,
                    dataset_id=str(dataset_id) if dataset_id is not None else None,
                    symbols=symbols_tuple,
                    schema_override=str(schema_override) if schema_override is not None else None,
                    storage_kind_override=storage_kind,
                    start=str(entry.get("start")) if entry.get("start") is not None else None,
                    end=str(entry.get("end")) if entry.get("end") is not None else None,
                ),
            )
            continue
        raise SystemExit("market_inputs_json entries must be strings or objects")

    return tuple(inputs) if inputs else None


def _build_validation_config_from_args(
    args: argparse.Namespace,
    macro_series_ids: tuple[str, ...] | None,
) -> DatasetValidationConfig | None:
    config = DatasetValidationConfig()
    modified = False
    if args.validation_min_rows is not None:
        config = replace(config, min_rows=int(args.validation_min_rows))
        modified = True
    if args.validation_min_positive_rate is not None:
        config = replace(config, min_positive_rate=float(args.validation_min_positive_rate))
        modified = True
    if args.validation_max_positive_rate is not None:
        config = replace(config, max_positive_rate=float(args.validation_max_positive_rate))
        modified = True
    if args.validation_min_feature_coverage is not None:
        config = replace(
            config,
            min_feature_coverage=float(args.validation_min_feature_coverage),
        )
        modified = True
    if macro_series_ids and config.require_macro_series is None:
        config = replace(config, require_macro_series=macro_series_ids)
        modified = True
    return config if modified else None


def _build_ingestion_config_from_args(
    args: argparse.Namespace,
    ds_cfg: DatasetBuildConfig | None,
) -> IngestionStageConfig:
    """
    Construct an ingestion stage config from CLI arguments.
    """
    default_cfg = IngestionStageConfig()
    raw_dataset_id = getattr(args, "dataset_id", None)
    dataset_id = str(raw_dataset_id).strip() if raw_dataset_id else None
    if dataset_id is None and ds_cfg is not None and ds_cfg.market_dataset_id:
        dataset_id = ds_cfg.market_dataset_id
    schema = str(getattr(args, "schema", default_cfg.schema))

    raw_instruments = getattr(args, "instruments", None)
    instruments: tuple[str, ...]
    if raw_instruments:
        tokens = [token.strip() for token in str(raw_instruments).split(",") if token.strip()]
        instruments = tuple(tokens) if tokens else default_cfg.instruments
    elif ds_cfg is not None and ds_cfg.instrument_ids:
        instruments = ds_cfg.instrument_ids
    else:
        instruments = default_cfg.instruments

    raw_symbol_override = getattr(args, "symbols", None)
    if raw_symbol_override:
        symbol_tokens = tuple(
            token.strip().upper() for token in str(raw_symbol_override).split(",") if token.strip()
        )
    elif ds_cfg is not None:
        symbol_tokens = tuple(
            token.strip().upper() for token in str(ds_cfg.symbols).split(",") if token.strip()
        )
    else:
        symbol_tokens = tuple(inst.split(".")[0].upper() for inst in instruments if inst)
    symbols: tuple[str, ...] | None = symbol_tokens or None

    raw_instrument_ids = getattr(args, "instrument_ids", None)
    if raw_instrument_ids:
        instrument_ids_override = tuple(
            token.strip() for token in str(raw_instrument_ids).split(",") if token.strip()
        )
    elif ds_cfg is not None and ds_cfg.instrument_ids:
        instrument_ids_override = ds_cfg.instrument_ids
    else:
        instrument_ids_override = tuple(instruments)
    instrument_ids: tuple[str, ...] | None = instrument_ids_override or None

    lookback_days = int(getattr(args, "lookback_days", default_cfg.lookback_days))
    coverage_mode = str(getattr(args, "coverage_mode", default_cfg.coverage_mode))
    write_mode = str(getattr(args, "write_mode", default_cfg.write_mode))
    catalog_path_raw = getattr(args, "catalog_path", None)
    catalog_path = str(catalog_path_raw) if catalog_path_raw else None

    market_dataset_id_raw = getattr(args, "market_dataset_id", None)
    market_dataset_id = str(market_dataset_id_raw).strip() if market_dataset_id_raw else None
    if market_dataset_id is None and ds_cfg is not None:
        market_dataset_id = ds_cfg.market_dataset_id

    market_inputs = _parse_market_inputs_json(
        getattr(args, "market_inputs_json", None),
    )
    if not market_inputs:
        market_inputs = ds_cfg.market_inputs if ds_cfg is not None else None

    return IngestionStageConfig(
        enabled=bool(getattr(args, "ingest", False)),
        dataset_id=dataset_id,
        schema=schema,
        instruments=instruments,
        lookback_days=lookback_days,
        coverage_mode=coverage_mode,
        write_mode=write_mode,
        catalog_path=catalog_path,
        symbols=symbols,
        instrument_ids=instrument_ids,
        market_dataset_id=market_dataset_id,
        market_inputs=market_inputs,
    )


def _ingestion_config_to_args(cfg: IngestionStageConfig) -> list[str]:
    """
    Convert an ingestion stage config into CLI arguments.
    """
    args: list[str] = []
    if cfg.enabled:
        args.append("--ingest")
    if cfg.dataset_id:
        args += ["--dataset_id", cfg.dataset_id]
    args += ["--schema", cfg.schema]
    if cfg.instruments:
        args += ["--instruments", ",".join(cfg.instruments)]
    if cfg.symbols:
        args += ["--symbols", ",".join(cfg.symbols)]
    if cfg.instrument_ids:
        args += ["--instrument_ids", ",".join(cfg.instrument_ids)]
    args += ["--lookback_days", str(cfg.lookback_days)]
    args += ["--coverage_mode", cfg.coverage_mode]
    args += ["--write_mode", cfg.write_mode]
    if cfg.catalog_path:
        args += ["--catalog_path", cfg.catalog_path]
    if cfg.market_dataset_id:
        args += ["--market_dataset_id", cfg.market_dataset_id]
    if cfg.market_inputs:
        payload: list[dict[str, object]] = []
        for item in cfg.market_inputs:
            entry: dict[str, object] = {}
            if item.descriptor_id is not None:
                entry["descriptor_id"] = item.descriptor_id
            if item.dataset_id is not None:
                entry["dataset_id"] = item.dataset_id
            if item.symbols is not None:
                entry["symbols"] = list(item.symbols)
            if item.schema_override is not None:
                entry["schema"] = item.schema_override
            if item.storage_kind_override is not None:
                entry["storage_kind"] = item.storage_kind_override.value
            if item.start is not None:
                entry["start"] = item.start
            if item.end is not None:
                entry["end"] = item.end
        payload.append(entry)
        args += ["--market_inputs_json", json.dumps(payload)]
    return args


@dataclass(slots=True, frozen=True)
class _IngestionPlanItem:
    """
    Resolved ingestion work unit derived from configuration inputs.
    """

    binding: ResolvedMarketBinding | None
    dataset_id: str
    schema: str
    instrument_ids: tuple[str, ...]


def _build_ingestion_plan(
    *,
    ds_cfg: DatasetBuildConfig | None,
    ingestion_cfg: IngestionStageConfig,
) -> tuple[_IngestionPlanItem, ...]:
    """
    Construct per-binding ingestion plan items from configuration.
    """
    symbol_to_instruments = _collect_symbol_map(ds_cfg=ds_cfg, ingestion_cfg=ingestion_cfg)

    market_inputs = ingestion_cfg.market_inputs
    if market_inputs is None and ds_cfg is not None:
        market_inputs = ds_cfg.market_inputs

    symbols_tuple = tuple(symbol_to_instruments.keys())
    instrument_ids_all = tuple(
        dict.fromkeys(chain.from_iterable(symbol_to_instruments.values())),
    )

    market_dataset_id = (
        ingestion_cfg.market_dataset_id
        or (ds_cfg.market_dataset_id if ds_cfg is not None else None)
        or ingestion_cfg.dataset_id
    )

    fallback_candidates: list[str | None] = [ingestion_cfg.dataset_id, market_dataset_id]
    if market_inputs:
        fallback_candidates.extend(item.dataset_id for item in market_inputs if item.dataset_id)

    resolved_bindings: tuple[ResolvedMarketBinding, ...] = ()
    if symbols_tuple:
        try:
            resolved_bindings = IngestionOrchestrator.resolve_market_bindings(
                symbols=symbols_tuple,
                instrument_ids=instrument_ids_all or None,
                market_dataset_id=market_dataset_id,
                market_inputs=market_inputs,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Ingestion binding resolution failed", exc_info=True)

    def _select_fallback_dataset() -> str | None:
        allowed = _get_allowed_databento_datasets()
        candidates_ordered: list[str] = []
        candidates_ordered.extend(candidate for candidate in fallback_candidates if candidate)
        candidates_ordered.extend(
            binding.dataset_id for binding in resolved_bindings if binding.dataset_id
        )
        if ds_cfg is not None and ds_cfg.market_dataset_id:
            candidates_ordered.append(ds_cfg.market_dataset_id)

        if allowed:
            for candidate in candidates_ordered:
                if candidate in allowed:
                    return candidate
        for candidate in candidates_ordered:
            if candidate:
                return candidate
        if allowed:
            # Deterministic fallback so ingest runs without manual dataset wiring.
            ordered_allowed = sorted(allowed)
            if ordered_allowed:
                return ordered_allowed[0]
        return None

    fallback_dataset_id = _select_fallback_dataset()
    if fallback_dataset_id is None:
        raise ValueError("Ingestion configuration requires a dataset identifier")

    fallback_schema = _normalize_schema_token(ingestion_cfg.schema)
    if not fallback_schema:
        raise ValueError("Ingestion configuration requires a schema value")

    plan_items: list[_IngestionPlanItem] = []
    for binding in resolved_bindings:
        dataset_id = binding.dataset_id or fallback_dataset_id
        schema = binding.schema or fallback_schema
        schema = _normalize_schema_token(schema)
        binding_instruments = tuple(
            dict.fromkeys(
                binding.instrument_ids or symbol_to_instruments.get(binding.symbol.upper(), ()),
            ),
        )
        if not binding_instruments:
            fallback_symbol = binding.symbol.strip().upper()
            binding_instruments = (fallback_symbol,) if fallback_symbol else ()
        plan_items.append(
            _IngestionPlanItem(
                binding=binding,
                dataset_id=dataset_id,
                schema=schema,
                instrument_ids=binding_instruments,
            ),
        )

    if not plan_items:
        manual_instruments = instrument_ids_all
        if not manual_instruments and ingestion_cfg.instrument_ids:
            manual_instruments = tuple(
                dict.fromkeys(
                    instrument.strip().upper()
                    for instrument in ingestion_cfg.instrument_ids
                    if instrument.strip()
                ),
            )
        if not manual_instruments and ingestion_cfg.instruments:
            manual_instruments = tuple(
                dict.fromkeys(
                    instrument.strip().upper()
                    for instrument in ingestion_cfg.instruments
                    if instrument.strip()
                ),
            )
        if not manual_instruments:
            manual_instruments = tuple(symbol_to_instruments.keys())
        if not manual_instruments:
            manual_instruments = tuple(
                instrument.strip().upper().split(".")[0]
                for instrument in ingestion_cfg.instruments
                if instrument.strip()
            )
        manual_instruments = tuple(dict.fromkeys(filter(None, manual_instruments)))
        if not manual_instruments:
            raise ValueError(
                "Ingestion configuration requires at least one instrument for manual fallback",
            )
        plan_items.append(
            _IngestionPlanItem(
                binding=None,
                dataset_id=fallback_dataset_id,
                schema=fallback_schema,
                instrument_ids=manual_instruments,
            ),
        )

    return tuple(plan_items)


def _build_auto_fill_config_from_args(
    args: argparse.Namespace,
    _dataset_cfg: DatasetBuildConfig,
) -> AutoFillUniverseConfig:
    enabled = bool(getattr(args, "auto_fill_universe", False))
    instrument_override: tuple[str, ...] | None = None
    raw_override = getattr(args, "auto_fill_instrument_ids", None)
    if raw_override:
        instrument_override = tuple(
            item.strip() for item in str(raw_override).split(",") if item.strip()
        )
    dataset_id_arg = getattr(args, "auto_fill_dataset_id", None)
    dataset_id = str(dataset_id_arg or getattr(args, "dataset_id", "EQUS.MINI"))
    include_l2 = bool(getattr(args, "include_l2", False)) and not bool(
        getattr(args, "auto_fill_skip_l2", False),
    )
    l2_dataset_id = str(
        getattr(args, "auto_fill_l2_dataset_id", None) or "DBEQ.BASIC",
    )
    l2_schema = str(
        getattr(args, "auto_fill_l2_schema", None) or "mbp-10",
    )
    l2_days_raw = getattr(args, "auto_fill_l2_days", None)
    l2_days = int(l2_days_raw) if l2_days_raw is not None else None
    l2_progress_file_raw = getattr(args, "auto_fill_l2_progress_file", None)
    l2_progress_file = str(l2_progress_file_raw) if l2_progress_file_raw else None
    allow_dataset_l2 = bool(getattr(args, "auto_fill_allow_dataset_l2_ingest", False))
    include_l3 = bool(getattr(args, "auto_fill_include_l3", False))
    l3_dataset_id_raw = getattr(args, "auto_fill_l3_dataset_id", None)
    l3_schema_raw = getattr(args, "auto_fill_l3_schema", None)
    l3_days_raw = getattr(args, "auto_fill_l3_days", None)
    l3_days = int(l3_days_raw) if l3_days_raw is not None else None

    include_bars = True
    include_tbbo = True
    include_trades = True
    if dataset_id_arg and dataset_id != getattr(args, "dataset_id", dataset_id):
        include_bars = False
        include_tbbo = False
        include_trades = False

    return AutoFillUniverseConfig(
        enabled=enabled,
        dataset_id=dataset_id,
        include_bars=include_bars,
        include_tbbo=include_tbbo,
        include_trades=include_trades,
        include_l2=include_l2,
        include_l3=include_l3,
        l2_dataset_id=l2_dataset_id,
        l2_schema=l2_schema,
        l2_days=l2_days,
        l2_progress_file=l2_progress_file,
        disable_dataset_l2_ingest=not allow_dataset_l2,
        instrument_ids=instrument_override,
        l3_dataset_id=str(l3_dataset_id_raw) if l3_dataset_id_raw else None,
        l3_schema=str(l3_schema_raw) if l3_schema_raw else None,
        l3_days=l3_days,
    )
