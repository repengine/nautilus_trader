#!/usr/bin/env python
"""
ML Pipeline Docker entrypoint.

This script serves as the entry point for the ML pipeline container, handling
environment configuration and launching the appropriate pipeline mode.

"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
import uuid as _uuid
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict
from typing import Any as _Any
from typing import cast as _cast

from flask import Flask
from flask import Response
from flask import jsonify


# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml._imports import check_ml_dependencies
from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.common.metrics_export import CONTENT_TYPE_LATEST
from ml.common.metrics_export import generate_latest
from ml.config.dataset_coverage import CoverageDatasetEntry
from ml.config.dataset_coverage import load_dataset_coverage_entries
from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID
from ml.config.dataset_ids import EVENTS_CALENDAR_DATASET_ID
from ml.config.dataset_ids import L2_MINUTE_DATASET_ID
from ml.config.dataset_ids import MACRO_OBSERVATIONS_DATASET_ID
from ml.config.dataset_ids import MACRO_RELEASES_DATASET_ID
from ml.config.dataset_ids import MICRO_MINUTE_DATASET_ID
from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import MarketFeedDescriptor
from ml.config.market_data import coerce_storage_kind
from ml.config.market_data import load_market_feed_descriptors
from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.config.scheduler_config import UniverseConfig
from ml.core.db_engine import EngineManager
from ml.core.integration import MLIntegrationManager
from ml.data.coverage.feature_restorer import SUPPORTED_FEATURE_DATASET_IDS
from ml.data.coverage.feature_restorer import FeatureCoverageRestorer
from ml.data.coverage.manager import BucketClassification
from ml.data.coverage.manager import BucketSpec
from ml.data.coverage.manager import BucketStatus
from ml.data.coverage.manager import CoverageManager
from ml.data.coverage.manager import CoverageManagerConfig
from ml.data.coverage.manager import DatasetCoverageConfig
from ml.data.ingest.market_bindings import resolve_instrument_ids_for_symbols
from ml.data.rehydration import CatalogRehydrationConfig
from ml.data.rehydration import ParquetCatalogRehydrator
from ml.data.scheduler import DataScheduler
from ml.deployment.scheduling_utils import DailyTime
from ml.deployment.scheduling_utils import compute_next_utc_run
from ml.deployment.scheduling_utils import parse_bool_env
from ml.deployment.scheduling_utils import parse_daily_spec
from ml.deployment.scheduling_utils import parse_dataset_template_map_env
from ml.deployment.scheduling_utils import parse_template_map_env
from ml.observability.bootstrap import auto_start_if_configured
from ml.registry.dataclasses import DatasetType
from ml.schema import DATASET_TYPE_IDENTIFIER_DEFAULTS
from ml.schema import DEFAULT_BAR_IDENTIFIER_TEMPLATE
from ml.schema import schema_spec_for
from ml.schema import validate_dataset_type_templates
from ml.schema import validate_identifier_template
from ml.schema import validate_schema_identifier_templates
from ml.stores.feature_store import FeatureStore
from ml.stores.migrations_runner import MigrationRunnerError
from ml.stores.migrations_runner import SchemaHealthCheckError
from ml.stores.migrations_runner import apply_profiled_migrations
from ml.stores.migrations_runner import is_postgres_url
from ml.stores.migrations_runner import verify_instrumentation_tables
from ml.stores.migrations_runner import verify_market_data_schema
from ml.stores.model_store import ModelStore
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.schema_audit import SchemaAuditor


# Provide a patchable symbol for tests; resolved lazily at runtime.
# Runtime Parquet catalog type (resolved lazily) to avoid import-time overhead.
_ParquetDataCatalogRT: type[_Any] | None = None
# Public alias for tests to patch directly (e.g., via unittest.mock.patch)
# When None, the runtime will lazily import ParquetDataCatalog.
ParquetDataCatalog: type[_Any] | None = None

if TYPE_CHECKING:  # pragma: no cover - avoid heavy import at module import time
    from ml.features import FeatureEngineer
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.io_raw import RawIngestionWriterProtocol
    from ml.stores.protocols import DataStoreFacadeProtocol
    from ml.stores.providers import ParquetCoverageSpec
    from ml.stores.providers import SqlCoverageOverride
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog as _PDC_T


configure_logging()
_run_id: str = f"pipeline_{_uuid.uuid4().hex[:12]}"
bind_log_context(run_id=_run_id, component="ml.entrypoint_pipeline")
logger = logging.getLogger(__name__)
DEFAULT_DB_URL = "postgresql://postgres:postgres@postgres:5432/nautilus"

_coverage_buckets_total = get_counter(
    "nautilus_ml_coverage_buckets_total",
    "Coverage classification outcome counts.",
    ["status"],
)
_coverage_restore_failures_total = get_counter(
    "nautilus_ml_coverage_restore_failures_total",
    "Coverage restoration failures grouped by stage.",
    ["stage"],
)
_coverage_manifest_events_total = get_counter(
    "nautilus_ml_coverage_manifest_events_total",
    "Coverage manifest load outcomes (missing, invalid, loaded).",
    ["event"],
)
_coverage_latency_seconds = get_histogram(
    "nautilus_ml_coverage_latency_seconds",
    "End-to-end coverage restoration latency in seconds.",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
_feature_reingest_total = get_counter(
    "nautilus_ml_feature_reingest_total",
    "Feature reingest attempts grouped by dataset/status.",
    ["dataset_id", "status"],
)


class CoverageRestorationError(RuntimeError):
    """Raised when coverage restoration cannot complete and gating is enforced."""


# Health check Flask app
class PipelineStatus(TypedDict):
    healthy: bool
    last_run: str | None
    errors: list[str]
    last_rehydrate: str | None
    coverage: CoverageStatus


class CoverageStatus(TypedDict):
    last_run: str | None
    last_success: str | None
    buckets_total: int
    buckets_restore_catalog: int
    buckets_reingest_source: int
    buckets_healthy: int
    last_error: str | None


def _default_coverage_status() -> CoverageStatus:
    return CoverageStatus(
        last_run=None,
        last_success=None,
        buckets_total=0,
        buckets_restore_catalog=0,
        buckets_reingest_source=0,
        buckets_healthy=0,
        last_error=None,
    )


app = Flask(__name__)
# Simple runtime status structure used by health endpoint
pipeline_status: PipelineStatus = {
    "healthy": False,
    "last_run": None,
    "errors": [],
    "last_rehydrate": None,
    "coverage": _default_coverage_status(),
}


@lru_cache(maxsize=1)
def _market_descriptor_resources() -> (
    tuple[Mapping[str, MarketFeedDescriptor], Mapping[str, tuple[str, ...]]]
):
    """
    Load feed descriptors and derive the dataset/schema compatibility matrix.
    """
    descriptor_set = load_market_feed_descriptors()
    descriptor_map = dict(descriptor_set.as_mapping())
    dataset_schemas: dict[str, set[str]] = defaultdict(set)
    for descriptor in descriptor_set.descriptors:
        dataset_schemas[descriptor.dataset_id].add(descriptor.schema)
    normalized = {dataset: tuple(sorted(schemas)) for dataset, schemas in dataset_schemas.items()}
    return descriptor_map, normalized


def _ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, tolerating existing directories and symlinks.

    This avoids spurious FileExistsError when the path is a symlink (common in
    mounted volumes) by treating existing symlinks as acceptable.
    """
    if path.is_symlink():
        return path
    if path.exists():
        if path.is_dir():
            return path
        raise FileExistsError(f"{path} exists and is not a directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


@app.route("/health")
def health_check() -> tuple[Any, int]:
    """
    Health check endpoint for Docker.
    """
    return jsonify(pipeline_status), 200 if pipeline_status["healthy"] else 503


@app.route("/metrics")
def metrics() -> Response:  # pragma: no cover - simple pass-through
    """
    Prometheus metrics endpoint.
    """
    payload = generate_latest()
    return Response(payload, mimetype=CONTENT_TYPE_LATEST)


class PipelineRunner:
    """
    ML Pipeline runner for Docker deployment.
    """

    def __init__(self) -> None:
        """
        Initialize the pipeline runner.
        """
        self.scheduler: DataScheduler | None = None
        self.running: bool = False
        self._shutdown_event = threading.Event()
        self._rehydrator: ParquetCatalogRehydrator | None = None
        self._rehydrator_config: CatalogRehydrationConfig | None = None
        self._scheduler_config: SchedulerConfig | None = None
        self._catalog_path: Path | None = None
        self._catalog_obj: _PDC_T | None = None

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, _frame: object | None) -> None:
        """
        Handle shutdown signals gracefully.
        """
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        self._shutdown_event.set()
        if self.scheduler is not None:
            try:
                # Scheduler implements cooperative stop
                self.scheduler.stop()
            except Exception:
                logger.debug("Scheduler.stop() failed during shutdown", exc_info=True)

    def _create_config(self) -> SchedulerConfig:
        """
        Create scheduler configuration from environment variables.
        """
        # Parse universe symbols
        raw_symbols = os.environ.get("UNIVERSE_SYMBOLS", "SPY.XNAS")
        symbols = [s.strip() for s in raw_symbols.split(",") if s.strip()]

        expand_universe = parse_bool_env(os.environ.get("UNIVERSE_EXPAND", "1"))

        # Create Databento config (API key consumed by underlying loader via env)
        databento_config = DatabentoConfig(
            dataset=os.environ.get("DATABENTO_DATASET", "EQUS.MINI"),
            schema=os.environ.get("DATABENTO_SCHEMA", "ohlcv-1m"),
            stype_in=os.environ.get("DATABENTO_STYPE_IN", "raw_symbol"),
        )

        # Universe expansion (optional)
        full_universe = symbols
        if expand_universe:
            universe = UniverseConfig(
                expansion_mode=os.environ.get("UNIVERSE_MODE", "moderate"),  # type: ignore[arg-type]
            )
            # Merge user-provided symbols with expanded lists
            full_universe = list(dict.fromkeys(symbols + universe.get_full_universe()))

        # Normalize instrument suffixes for EQUS.MINI to use .EQUS by default
        if databento_config.dataset.upper() == "EQUS.MINI":
            normalized: list[str] = []
            for sym in full_universe:
                token = sym.strip()
                if not token:
                    continue
                if "." in token:
                    base, _sep, _suffix = token.partition(".")
                    normalized.append(f"{base}.EQUS")
                else:
                    normalized.append(f"{token}.EQUS")
            full_universe = list(dict.fromkeys(normalized))

        market_inputs = self._parse_market_dataset_inputs()
        market_lookback_days = self._get_int_env("MARKET_BACKFILL_LOOKBACK_DAYS", 1)
        dynamic_backfill = parse_bool_env(os.environ.get("MARKET_BACKFILL_DYNAMIC_LOOKBACK", "1"))
        dynamic_min_days = self._get_int_env("MARKET_BACKFILL_MIN_DAYS", 1)
        dynamic_max_raw = os.environ.get("MARKET_BACKFILL_MAX_DAYS")
        dynamic_max_days = None
        if dynamic_max_raw:
            try:
                dynamic_max_days = int(dynamic_max_raw)
            except ValueError:
                logger.warning(
                    "Invalid MARKET_BACKFILL_MAX_DAYS value (%s); ignoring override",
                    dynamic_max_raw,
                )

        # Create scheduler config
        config = SchedulerConfig(
            symbols=full_universe,
            databento=databento_config,
            feature_store_enabled=True,
            feature_store_connection=os.environ.get(
                "DB_CONNECTION",
                os.environ.get("FEATURE_STORE_CONNECTION", os.environ.get("DATABASE_URL")),
            ),
            market_inputs=market_inputs,
            market_backfill_lookback_days=market_lookback_days,
            market_backfill_dynamic=dynamic_backfill,
            market_backfill_min_days=dynamic_min_days,
            market_backfill_max_days=dynamic_max_days,
        )

        return config

    def _initialize_stores(self, config: SchedulerConfig) -> tuple[FeatureStore, ModelStore]:
        """
        Initialize the feature and model stores.
        """
        # Initialize feature store
        fs_conn = config.feature_store_connection or os.environ.get(
            "DB_CONNECTION",
            os.environ.get(
                "FEATURE_STORE_CONNECTION",
                os.environ.get("DATABASE_URL", DEFAULT_DB_URL),
            ),
        )
        if not fs_conn:
            raise RuntimeError("Feature store connection string is not configured")
        feature_store = FeatureStore(connection_string=fs_conn)

        # Initialize model store
        ms_conn = os.environ.get(
            "MODEL_STORE_CONNECTION",
            os.environ.get("DB_CONNECTION", os.environ.get("DATABASE_URL", fs_conn)),
        )
        model_store = ModelStore(connection_string=ms_conn)

        return feature_store, model_store

    def _build_feature_engineer(self) -> FeatureEngineer | None:
        """
        Build a FeatureEngineer instance for scheduled feature computation.
        """
        try:
            from ml.features import FeatureConfig
            from ml.features import FeatureEngineer
        except Exception:
            logger.warning("FeatureEngineer imports unavailable", exc_info=True)
            return None
        try:
            config = FeatureConfig()
            return FeatureEngineer(config)
        except Exception:
            logger.warning("FeatureEngineer initialization failed", exc_info=True)
            return None

    def _initialize_catalog(self, config: SchedulerConfig) -> _PDC_T:
        """
        Initialize the data catalog.
        """
        catalog_path = Path(os.environ.get("CATALOG_PATH", "/app/data/catalog"))
        catalog_path = _ensure_directory(catalog_path)
        self._catalog_path = catalog_path
        # Resolve catalog class: prefer patched module-level alias if present
        global _ParquetDataCatalogRT, ParquetDataCatalog
        ctor_any: type[_Any] | None = ParquetDataCatalog or _ParquetDataCatalogRT
        if ctor_any is None:
            # Lazy import to avoid heavy dependency at module import time
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog as _PDC

            _ParquetDataCatalogRT = _PDC
            ctor_any = _ParquetDataCatalogRT
        ctor = _cast(type["_PDC_T"], ctor_any)
        catalog_instance = ctor(str(catalog_path))
        self._catalog_obj = catalog_instance
        return catalog_instance

    def _get_int_env(self, name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                "Invalid integer for %s (value=%s); using default %s",
                name,
                raw,
                default,
            )
            return default

    def _parse_template_map_env(self, raw: str | None) -> dict[str, str]:
        """
        Parse a schema→template mapping from environment payloads.
        """
        return parse_template_map_env(raw)

    def _parse_dataset_template_map_env(self, raw: str | None) -> dict[DatasetType, str]:
        """
        Parse a dataset-type→template mapping from environment payloads.
        """
        return parse_dataset_template_map_env(raw)

    def _dual_write_dataset_types_from_env(self) -> dict[DatasetType, bool]:
        """
        Parse per-schema dual-write toggles; defaults stay enabled.
        """

        def _flag(env_name: str, default: bool = True) -> bool:
            raw = os.environ.get(env_name)
            return parse_bool_env(raw) if raw is not None else default

        return {
            DatasetType.BARS: _flag("DUAL_WRITE_BARS"),
            DatasetType.TRADES: _flag("DUAL_WRITE_TRADES"),
            DatasetType.TBBO: _flag("DUAL_WRITE_TBBO"),
            DatasetType.MBP1: _flag("DUAL_WRITE_MBP"),
        }

    def _resolve_rehydrator_config(self) -> CatalogRehydrationConfig:
        """
        Return (and cache) the catalog rehydration config.
        """
        if self._rehydrator_config is None:
            self._rehydrator_config = self._build_catalog_rehydration_config()
        return self._rehydrator_config

    def _parse_market_dataset_inputs(self) -> tuple[MarketDatasetInput, ...] | None:
        file_path = os.environ.get("MARKET_DATASET_INPUTS_FILE")
        raw: str | None = None
        if file_path:
            try:
                raw = Path(file_path).read_text(encoding="utf-8")
            except Exception:
                logger.warning(
                    "Failed to read MARKET_DATASET_INPUTS_FILE",
                    exc_info=True,
                    extra={"path": file_path},
                )
        if raw is None:
            raw = os.environ.get("MARKET_DATASET_INPUTS")
        if raw is None:
            return None
        raw = raw.strip()
        if not raw:
            return None

        def _instantiate(entry: dict[str, object] | str) -> MarketDatasetInput | None:
            if isinstance(entry, str):
                token = entry.strip()
                if not token:
                    return None
                try:
                    return MarketDatasetInput(descriptor_id=token)
                except Exception:
                    logger.warning(
                        "Failed to build MarketDatasetInput from descriptor_id '%s'",
                        token,
                        exc_info=True,
                    )
                    return None

            payload = dict(entry)

            symbols_value = payload.get("symbols")
            symbols_tuple: tuple[str, ...] | None = None
            if isinstance(symbols_value, str):
                token = symbols_value.strip()
                if token:
                    symbols_tuple = (token,)
            elif isinstance(symbols_value, list | tuple):
                filtered = [str(item).strip() for item in symbols_value if str(item).strip()]
                if filtered:
                    symbols_tuple = tuple(filtered)

            storage_override_token = payload.get("storage_kind_override")
            storage_kind_override = None
            if storage_override_token is not None:
                candidate = (
                    storage_override_token
                    if isinstance(storage_override_token, str)
                    else str(storage_override_token)
                )
                try:
                    storage_kind_override = coerce_storage_kind(candidate)
                except ValueError:
                    logger.warning(
                        "Invalid storage_kind_override in MARKET_DATASET_INPUTS entry",
                        exc_info=True,
                        extra={"value": storage_override_token},
                    )
                    storage_kind_override = None

            descriptor_obj = payload.get("descriptor_id")
            descriptor_id = None
            if descriptor_obj is not None:
                descriptor_str = str(descriptor_obj).strip()
                descriptor_id = descriptor_str or None

            dataset_obj = payload.get("dataset_id")
            dataset_id = None
            if dataset_obj is not None:
                dataset_str = str(dataset_obj).strip()
                dataset_id = dataset_str or None

            provider_dataset_obj = payload.get("provider_dataset_id")
            provider_dataset_id = None
            if provider_dataset_obj is not None:
                provider_dataset_str = str(provider_dataset_obj).strip()
                provider_dataset_id = provider_dataset_str or None

            provider_schema_obj = payload.get("provider_schema")
            provider_schema = None
            if provider_schema_obj is not None:
                provider_schema_str = str(provider_schema_obj).strip()
                provider_schema = provider_schema_str or None

            schema_obj = payload.get("schema_override")
            schema_override = None
            if schema_obj is not None:
                schema_str = str(schema_obj).strip()
                schema_override = schema_str or None

            start_obj = payload.get("start")
            start = None
            if start_obj is not None:
                start_str = str(start_obj).strip()
                start = start_str or None

            end_obj = payload.get("end")
            end = None
            if end_obj is not None:
                end_str = str(end_obj).strip()
                end = end_str or None

            try:
                return MarketDatasetInput(
                    descriptor_id=descriptor_id,
                    dataset_id=dataset_id,
                    provider_dataset_id=provider_dataset_id,
                    provider_schema=provider_schema,
                    symbols=symbols_tuple,
                    schema_override=schema_override,
                    storage_kind_override=storage_kind_override,
                    start=start,
                    end=end,
                )
            except Exception:
                logger.warning(
                    "Failed to parse MARKET_DATASET_INPUTS entry",
                    exc_info=True,
                    extra={"entry": entry},
                )
                return None

        entries: list[MarketDatasetInput] = []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            tokens = [token.strip() for token in raw.split(",") if token.strip()]
            for token in tokens:
                item = _instantiate(token)
                if item is not None:
                    entries.append(item)
        else:
            if isinstance(parsed, dict):
                candidate = _instantiate(_cast(dict[str, object], parsed))
                if candidate is not None:
                    entries.append(candidate)
            elif isinstance(parsed, list):
                for element in parsed:
                    if isinstance(element, str):
                        candidate = _instantiate(element)
                    elif isinstance(element, dict):
                        candidate = _instantiate(_cast(dict[str, object], element))
                    else:
                        logger.warning(
                            "Skipping unsupported MARKET_DATASET_INPUTS list entry",
                            extra={"type": type(element).__name__},
                        )
                        continue
                    if candidate is not None:
                        entries.append(candidate)
            elif isinstance(parsed, str):
                candidate = _instantiate(parsed)
                if candidate is not None:
                    entries.append(candidate)
            else:
                logger.warning(
                    "Unsupported MARKET_DATASET_INPUTS payload type",
                    extra={"type": type(parsed).__name__},
                )
        if not entries:
            return None
        self._validate_market_dataset_inputs(entries)
        return tuple(entries)

    def _validate_market_dataset_inputs(self, entries: Sequence[MarketDatasetInput]) -> None:
        """
        Ensure dataset/schema combinations are supported before building configs.
        """
        descriptor_map, dataset_schemas = _market_descriptor_resources()
        allowed_datasets = ", ".join(sorted(dataset_schemas)) or "<none>"
        errors: list[str] = []
        for index, entry in enumerate(entries):
            descriptor: MarketFeedDescriptor | None = None
            descriptor_id = entry.descriptor_id
            if descriptor_id:
                descriptor = descriptor_map.get(descriptor_id)
                if descriptor is None:
                    errors.append(
                        f"entry[{index}] references unknown descriptor '{descriptor_id}'",
                    )
                    continue
            dataset_id = entry.dataset_id
            if descriptor is not None:
                dataset_id = descriptor.dataset_id
                if entry.dataset_id and entry.dataset_id != descriptor.dataset_id:
                    errors.append(
                        f"entry[{index}] dataset_id '{entry.dataset_id}' "
                        f"conflicts with descriptor '{descriptor_id}' ({descriptor.dataset_id})",
                    )
                    continue
            if dataset_id is None:
                errors.append(f"entry[{index}] did not resolve to a dataset id")
                continue
            allowed_schemas = dataset_schemas.get(dataset_id)
            if allowed_schemas is None:
                errors.append(
                    f"entry[{index}] dataset '{dataset_id}' is not supported "
                    f"(allowed datasets: {allowed_datasets})",
                )
                continue
            schema_candidate = entry.schema_override or (descriptor.schema if descriptor else None)
            if schema_candidate is not None and schema_candidate not in allowed_schemas:
                errors.append(
                    f"entry[{index}] schema '{schema_candidate}' is not allowed for dataset '{dataset_id}'. "
                    f"Allowed schemas: {', '.join(allowed_schemas)}",
                )
        if errors:
            raise ValueError("Invalid MARKET_DATASET_INPUTS: " + "; ".join(errors))

    def _build_catalog_rehydration_config(self) -> CatalogRehydrationConfig:
        enabled = parse_bool_env(os.environ.get("CATALOG_REHYDRATE_ENABLED"))
        lookback = self._get_int_env("CATALOG_REHYDRATE_LOOKBACK_DAYS", 5)
        batch_size = self._get_int_env("CATALOG_REHYDRATE_BATCH_SIZE", 1_000)
        identifier_template = os.environ.get(
            "CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE",
            DEFAULT_BAR_IDENTIFIER_TEMPLATE,
        )
        identifier_template = validate_identifier_template(
            identifier_template,
            label="CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE",
        )
        schema_template_map = validate_schema_identifier_templates(
            self._parse_template_map_env(
                os.environ.get("CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE_MAP"),
            ),
        )
        dataset_template_map = validate_dataset_type_templates(
            self._parse_dataset_template_map_env(
                os.environ.get("CATALOG_REHYDRATE_DATASET_TYPE_TEMPLATES"),
            ),
        )
        table_name = os.environ.get("CATALOG_REHYDRATE_TABLE", "market_data")
        rescan = parse_bool_env(os.environ.get("CATALOG_REHYDRATE_RESCAN"))
        exhaustive = parse_bool_env(os.environ.get("CATALOG_REHYDRATE_EXHAUSTIVE"))
        return CatalogRehydrationConfig(
            enabled=enabled,
            lookback_days=lookback,
            batch_size=batch_size,
            identifier_template=identifier_template,
            schema_identifier_templates=schema_template_map,
            dataset_type_identifier_templates=(
                dataset_template_map or DATASET_TYPE_IDENTIFIER_DEFAULTS.copy()
            ),
            table_name=table_name,
            rescan_on_schedule=rescan,
            exhaustive=exhaustive,
        )

    def _coverage_restore_enabled(self) -> bool:
        return parse_bool_env(os.environ.get("COVERAGE_RESTORE_ENABLED"))

    def _coverage_reingest_enabled(self) -> bool:
        return parse_bool_env(os.environ.get("COVERAGE_RESTORE_REINGEST_ENABLED", "1"))

    def _feature_reingest_enabled(self) -> bool:
        return parse_bool_env(os.environ.get("FEATURE_REINGEST_ENABLED", "1"))

    def _schema_audit_allowlist(self) -> tuple[str, ...] | None:
        """
        Return an optional allowlist of tables to include in the schema audit.

        Returns:
            Lowercased table names or fully qualified table identifiers.
        """
        raw = os.environ.get("COVERAGE_SCHEMA_AUDIT_TABLES")
        if not raw:
            return None
        tables = tuple(
            token.strip().lower() for token in raw.split(",") if token.strip()
        )
        return tables or None

    def _build_schema_auditor(self, db_connection: str) -> SchemaAuditor:
        """
        Build a SchemaAuditor, optionally restricted to an allowlist of tables.

        Args:
            db_connection: PostgreSQL connection string.

        Returns:
            SchemaAuditor configured for the current environment.
        """
        allowlist = self._schema_audit_allowlist()
        if not allowlist:
            return SchemaAuditor(db_url=db_connection)
        from ml.stores.schema_audit import default_table_expectations

        normalized = set(allowlist)
        expectations = tuple(
            expectation
            for expectation in default_table_expectations()
            if expectation.table.lower() in normalized
            or f"{expectation.schema}.{expectation.table}".lower() in normalized
        )
        if not expectations:
            logger.warning(
                "coverage.schema_audit_allowlist_empty",
                extra={"tables": sorted(normalized)},
            )
            return SchemaAuditor(db_url=db_connection)
        return SchemaAuditor(db_url=db_connection, expectations=expectations)

    def _build_dataset_coverage_configs(
        self,
        config: SchedulerConfig,
    ) -> tuple[CoverageDatasetEntry, ...]:
        datasets: list[CoverageDatasetEntry] = []
        descriptor_map, _ = _market_descriptor_resources()
        market_inputs = config.market_inputs or ()
        if market_inputs:
            for entry in market_inputs:
                descriptor = (
                    descriptor_map.get(entry.descriptor_id) if entry.descriptor_id else None
                )
                dataset_id = (
                    descriptor.dataset_id if descriptor is not None else entry.dataset_id
                ) or config.databento.dataset
                schema = (
                    entry.schema_override
                    or (descriptor.schema if descriptor is not None else None)
                    or config.databento.schema
                )
                if schema:
                    schema = schema.strip().lower()
                    schema_spec_for(schema)
                symbols = entry.symbols or tuple(config.symbols)
                if not dataset_id or not schema or not symbols:
                    continue
                datasets.append(
                    CoverageDatasetEntry(
                        dataset=DatasetCoverageConfig(
                            dataset_id=dataset_id,
                            schema=schema,
                            instruments=symbols,
                        ),
                    ),
                )
        else:
            symbols = tuple(config.symbols)
            if symbols:
                schema = config.databento.schema.strip().lower()
                schema_spec_for(schema)
                datasets.append(
                    CoverageDatasetEntry(
                        dataset=DatasetCoverageConfig(
                            dataset_id=config.databento.dataset,
                            schema=schema,
                            instruments=symbols,
                        ),
                    ),
                )
        return tuple(datasets)

    def _load_feature_coverage_entries(self) -> tuple[CoverageDatasetEntry, ...]:
        config_path = os.environ.get("COVERAGE_DATASETS_FILE")
        resolved_path: Path | None = None
        default_path = Path("ml/config/coverage_datasets_tier1.toml")
        if config_path:
            resolved_path = Path(config_path)
            if not resolved_path.exists():
                logger.warning("coverage.feature_config_missing", extra={"path": config_path})
                _coverage_manifest_events_total.labels(event="missing").inc()
                self._record_coverage_error(reason="feature_manifest_missing")
                pipeline_status["errors"].append(f"feature_manifest_missing:{config_path}")
                return tuple()
        else:
            if default_path.exists():
                resolved_path = default_path
            else:
                logger.warning("coverage.feature_config_missing", extra={"path": str(default_path)})
                _coverage_manifest_events_total.labels(event="missing").inc()
                if self._coverage_restore_enabled():
                    self._record_coverage_error(reason="feature_manifest_missing")
                    pipeline_status["errors"].append(f"feature_manifest_missing:{default_path}")
                return tuple()
        try:
            entries = load_dataset_coverage_entries(str(resolved_path))
            validated = self._validate_feature_manifest(entries)
        except FileNotFoundError:
            logger.warning("coverage.feature_config_missing", extra={"path": str(resolved_path)})
            _coverage_manifest_events_total.labels(event="missing").inc()
            self._record_coverage_error(reason="feature_manifest_missing")
            if resolved_path is not None:
                pipeline_status["errors"].append(f"feature_manifest_missing:{resolved_path}")
            return tuple()
        except Exception:
            logger.error(
                "coverage.feature_config_invalid",
                exc_info=True,
                extra={"path": str(resolved_path)},
            )
            _coverage_manifest_events_total.labels(event="invalid").inc()
            self._record_coverage_error(reason="feature_manifest_invalid")
            if resolved_path is not None:
                pipeline_status["errors"].append(f"feature_manifest_invalid:{resolved_path}")
            return tuple()
        _coverage_manifest_events_total.labels(event="loaded").inc()
        return validated

    def _validate_feature_manifest(
        self,
        entries: tuple[CoverageDatasetEntry, ...],
    ) -> tuple[CoverageDatasetEntry, ...]:
        """
        Ensure the feature coverage manifest only references supported dataset IDs.
        """
        if not entries:
            return entries
        dataset_ids = {entry.dataset.dataset_id for entry in entries}
        unsupported = sorted(dataset_ids - SUPPORTED_FEATURE_DATASET_IDS)
        if unsupported:
            msg = f"Unsupported feature coverage datasets in manifest: {unsupported}"
            raise ValueError(msg)
        return entries

    def _compose_coverage_entries(
        self,
        config: SchedulerConfig,
    ) -> tuple[CoverageDatasetEntry, ...]:
        entries = list(self._build_dataset_coverage_configs(config))
        entries.extend(self._load_feature_coverage_entries())
        return tuple(entries)

    def _build_catalog_provider(
        self,
        parquet_specs: Mapping[str, ParquetCoverageSpec],
    ) -> CoverageProviderProtocol:
        from ml.stores.providers import CatalogCoverageProvider
        from ml.stores.providers import NullCoverageProvider
        from ml.stores.providers import PartitionedParquetCoverageProvider
        from ml.stores.providers import UnionCoverageProvider

        providers: list[CoverageProviderProtocol] = []
        if self._catalog_path:
            identifier_template = (
                self._rehydrator_config.identifier_template
                if self._rehydrator_config
                else DEFAULT_BAR_IDENTIFIER_TEMPLATE
            )
            schema_templates = (
                self._rehydrator_config.schema_identifier_templates
                if self._rehydrator_config
                else {}
            )
            dataset_templates = (
                self._rehydrator_config.dataset_type_identifier_templates
                if self._rehydrator_config
                else DATASET_TYPE_IDENTIFIER_DEFAULTS
            )
            try:
                providers.append(
                    CatalogCoverageProvider(
                        catalog_path=str(self._catalog_path),
                        identifier_template=identifier_template,
                        schema_identifier_templates=schema_templates,
                        dataset_type_identifier_templates=dataset_templates,
                        use_uri_safe_identifiers=True,
                    ),
                )
            except TypeError:
                # Backward compatibility with test doubles or legacy providers.
                providers.append(
                    CatalogCoverageProvider(
                        catalog_path=str(self._catalog_path),
                        identifier_template=identifier_template,
                    ),
                )
        if parquet_specs:
            providers.append(PartitionedParquetCoverageProvider(specs=parquet_specs))
        if not providers:
            return NullCoverageProvider()
        if len(providers) == 1:
            return providers[0]
        return UnionCoverageProvider(list(providers))

    def _resolve_db_connection(self, config: SchedulerConfig) -> str | None:
        candidates = (
            config.feature_store_connection,
            os.environ.get("DB_CONNECTION"),
            os.environ.get("FEATURE_STORE_CONNECTION"),
            os.environ.get("DATABASE_URL"),
            DEFAULT_DB_URL,
        )
        for candidate in candidates:
            if candidate:
                return candidate
        return None

    def _bootstrap_database(self, config: SchedulerConfig) -> None:
        """
        Apply SQL migrations and validate the market_data schema.
        """
        connection = self._resolve_db_connection(config)
        if not connection:
            raise RuntimeError("Database connection string is required for migrations")

        if not is_postgres_url(connection):
            logger.debug(
                "Skipping schema migrations for non-PostgreSQL connection",
                extra={"db_url": connection},
            )
            return

        summary = apply_profiled_migrations(db_url=connection)
        logger.info(
            "schema.migrations",
            extra={
                "applied": summary.applied_count,
                "already_applied": summary.already_applied_count,
                "profile": summary.profile.value,
            },
        )
        engine = EngineManager.get_engine(connection)
        report = verify_market_data_schema(engine)
        logger.info(
            "schema.market_data_verified",
            extra={
                "profile": report.profile.value,
                "tables": [table.table_name for table in report.tables],
            },
        )
        verify_instrumentation_tables(engine)
        logger.info("schema.instrumentation_tables_verified")

    def _build_catalog_rehydrator(
        self,
        *,
        catalog: _PDC_T,
        scheduler_config: SchedulerConfig,
    ) -> None:
        if self._rehydrator_config is None:
            self._rehydrator_config = self._build_catalog_rehydration_config()
        if not self._rehydrator_config.enabled:
            logger.debug("Catalog rehydration disabled via configuration")
            return

        db_connection = self._resolve_db_connection(scheduler_config)
        if not db_connection:
            logger.warning(
                "Catalog rehydration enabled but DB connection missing; skipping rehydration",
            )
            self._rehydrator_config = None
            return

        self._rehydrator = ParquetCatalogRehydrator(
            catalog=catalog,
            db_connection=db_connection,
            config=self._rehydrator_config,
            registry=self._resolve_rehydration_registry(),
        )

    def _resolve_rehydration_registry(self) -> RegistryProtocol | None:
        if self.scheduler is None:
            return None
        return getattr(self.scheduler, "data_registry", None)

    def _run_catalog_rehydration(
        self,
        scheduler_config: SchedulerConfig,
        *,
        note: str,
    ) -> None:
        if (
            self._rehydrator is None
            or self._rehydrator_config is None
            or not self._rehydrator_config.enabled
        ):
            return

        instrument_ids = self._select_rehydration_instruments(scheduler_config)
        if not instrument_ids:
            logger.debug("Catalog rehydration skipped because no symbols are configured or stale")
            return

        try:
            result = self._rehydrator.rehydrate_missing_data(
                dataset_id=scheduler_config.databento.dataset,
                schema=scheduler_config.databento.schema,
                instrument_ids=list(instrument_ids),
            )
        except Exception as exc:
            message = f"Catalog rehydration failed during {note}: {exc.__class__.__name__}"
            logger.warning(
                "catalog_rehydrate.failed",
                exc_info=True,
                extra={"note": note},
            )
            pipeline_status["errors"].append(message)
            return

        pipeline_status["last_rehydrate"] = datetime.now(UTC).isoformat()

        if result.failures:
            for instrument, reason in result.failures.items():
                pipeline_status["errors"].append(
                    f"rehydrate:{instrument}:{reason}",
                )
            logger.warning(
                "catalog_rehydrate.partial",
                extra={
                    "note": note,
                    "buckets_restored": result.buckets_restored,
                    "rows_written": result.rows_written,
                    "failures": result.failures,
                },
            )
            return

        logger.info(
            "catalog_rehydrate.completed",
            extra={
                "note": note,
                "instruments_processed": result.instruments_processed,
                "buckets_restored": result.buckets_restored,
                "rows_written": result.rows_written,
            },
        )

    def _select_rehydration_instruments(self, scheduler_config: SchedulerConfig) -> list[str]:
        symbols = list(dict.fromkeys(scheduler_config.symbols))
        if not symbols:
            return []
        descriptor_map, _ = _market_descriptor_resources()
        descriptor: MarketFeedDescriptor | None = None
        for candidate in descriptor_map.values():
            if (
                candidate.dataset_id == scheduler_config.databento.dataset
                and candidate.schema == scheduler_config.databento.schema
            ):
                descriptor = candidate
                break
        instrument_ids = resolve_instrument_ids_for_symbols(
            symbols=symbols,
            descriptor=descriptor,
        )
        if not instrument_ids:
            return []
        if not parse_bool_env(os.environ.get("CATALOG_REHYDRATE_STALE_ONLY", "1")):
            return list(instrument_ids)
        db_connection = self._resolve_db_connection(scheduler_config)
        if not db_connection or self._rehydrator_config is None:
            return list(instrument_ids)

        from ml.stores.providers import SqlCoverageProvider

        try:
            provider = SqlCoverageProvider(
                connection_string=db_connection,
                table_name=self._rehydrator_config.table_name,
            )
        except Exception:
            logger.debug(
                "catalog_rehydrate.staleness_probe_failed",
                exc_info=True,
                extra={"db_connection": db_connection},
            )
            return list(instrument_ids)
        threshold_hours = max(1, self._get_int_env("CATALOG_REHYDRATE_STALENESS_HOURS", 6))
        now_ns = int(datetime.now(tz=UTC).timestamp() * 1_000_000_000)
        stale: list[str] = []
        for instrument_id in instrument_ids:
            try:
                latest = provider.latest_timestamp_ns(
                    dataset_id=scheduler_config.databento.dataset,
                    instrument_id=instrument_id,
                )
            except Exception:
                logger.debug(
                    "catalog_rehydrate.instrument_probe_failed",
                    exc_info=True,
                    extra={"instrument_id": instrument_id},
                )
                stale.append(instrument_id)
                continue
            if latest is None:
                stale.append(instrument_id)
                continue
            delta_ns = max(now_ns - latest, 0)
            delta_hours = delta_ns / 3_600_000_000_000
            if delta_hours >= threshold_hours:
                stale.append(instrument_id)
        if stale:
            logger.info(
                "catalog_rehydrate.stale_instruments_detected",
                extra={"count": len(stale)},
            )
        return stale

    def _coverage_status(self) -> CoverageStatus:
        return pipeline_status["coverage"]

    def _record_coverage_summary(self, classifications: Sequence[BucketClassification]) -> None:
        coverage_status = self._coverage_status()
        now_iso = datetime.now(UTC).isoformat()
        total = len(classifications)
        restore_catalog = sum(
            1
            for classification in classifications
            if classification.status is BucketStatus.RESTORE_FROM_CATALOG
        )
        reingest = sum(
            1
            for classification in classifications
            if classification.status is BucketStatus.REINGEST_FROM_SOURCE
        )
        healthy = total - restore_catalog - reingest
        coverage_status["last_run"] = now_iso
        coverage_status["last_success"] = now_iso
        coverage_status["buckets_total"] = total
        coverage_status["buckets_restore_catalog"] = restore_catalog
        coverage_status["buckets_reingest_source"] = reingest
        coverage_status["buckets_healthy"] = healthy
        coverage_status["last_error"] = None

    def _record_coverage_error(self, *, reason: str) -> None:
        coverage_status = self._coverage_status()
        coverage_status["last_run"] = datetime.now(UTC).isoformat()
        coverage_status["last_error"] = reason

    def _get_coverage_bucket_cap(self) -> int:
        cap = self._get_int_env("COVERAGE_MAX_BUCKETS_PER_RUN", 500)
        if cap < 1:
            logger.warning(
                "Invalid COVERAGE_MAX_BUCKETS_PER_RUN; using default",
                extra={"configured": cap},
            )
            cap = 500
        return cap

    @staticmethod
    def _apply_bucket_cap(
        catalog_specs: list[BucketSpec],
        source_specs: list[BucketSpec],
        cap: int,
    ) -> tuple[list[BucketSpec], list[BucketSpec], int, int]:
        remaining = cap
        capped_catalog = catalog_specs[:remaining]
        remaining = max(0, remaining - len(capped_catalog))
        capped_source = source_specs[:remaining] if remaining > 0 else []
        skipped_catalog = max(0, len(catalog_specs) - len(capped_catalog))
        skipped_source = max(0, len(source_specs) - len(capped_source))
        return capped_catalog, capped_source, skipped_catalog, skipped_source

    def _run_coverage_restoration(
        self,
        scheduler_config: SchedulerConfig,
        *,
        dry_run: bool = False,
    ) -> None:
        if not self._coverage_restore_enabled():
            return
        if self.scheduler is None:
            logger.warning("Coverage restoration enabled but scheduler is missing")
            self._record_coverage_error(reason="scheduler_unavailable")
            self._maybe_raise_coverage_failure(reason="scheduler_unavailable")
            return
        db_connection = self._resolve_db_connection(scheduler_config)
        if not db_connection:
            logger.warning("Coverage restoration enabled but DB connection missing")
            self._record_coverage_error(reason="db_connection_missing")
            self._maybe_raise_coverage_failure(reason="db_connection_missing")
            return
        entries = self._compose_coverage_entries(scheduler_config)
        if not entries:
            logger.warning("Coverage restoration enabled but no dataset configs resolved")
            self._record_coverage_error(reason="dataset_configs_empty")
            self._maybe_raise_coverage_failure(reason="dataset_configs_empty")
            return

        from ml.stores.providers import SqlCoverageProvider

        parquet_specs = {
            entry.dataset.dataset_id: entry.parquet_spec
            for entry in entries
            if entry.parquet_spec is not None
        }
        sql_overrides: dict[str, SqlCoverageOverride] = {
            entry.dataset.dataset_id: entry.sql_override
            for entry in entries
            if entry.sql_override is not None
        }
        dataset_overrides: dict[str, object] | None = (
            {dataset_id: override for dataset_id, override in sql_overrides.items()}
            if sql_overrides
            else None
        )
        needs_catalog = any(entry.parquet_spec is None for entry in entries)
        if needs_catalog and self._catalog_path is None:
            logger.warning("Catalog path required for market dataset coverage restoration")
            self._record_coverage_error(reason="catalog_path_missing")
            self._maybe_raise_coverage_failure(reason="catalog_path_missing")
            return
        dataset_configs = tuple(entry.dataset for entry in entries)
        lookback = self._get_int_env("COVERAGE_RESTORE_LOOKBACK_DAYS", 5)
        manager = CoverageManager(
            config=CoverageManagerConfig(datasets=dataset_configs, lookback_days=lookback),
            sql_provider=SqlCoverageProvider(
                connection_string=db_connection,
                dataset_overrides=dataset_overrides,
            ),
            catalog_provider=self._build_catalog_provider(parquet_specs),
            schema_auditor=self._build_schema_auditor(db_connection),
        )
        try:
            start_time = time.perf_counter()
            classifications = manager.restore_all()
            elapsed = time.perf_counter() - start_time
            _coverage_latency_seconds.observe(elapsed)
        except Exception as exc:
            logger.error("coverage_manager.restore_failed", exc_info=True)
            self._record_coverage_error(reason=f"coverage_manager_failed:{exc.__class__.__name__}")
            _coverage_restore_failures_total.labels(stage="classification").inc()
            self._maybe_raise_coverage_failure(
                reason=f"coverage_manager_failed:{exc.__class__.__name__}",
                raise_immediately=True,
            )
            return
        self._record_coverage_summary(classifications)
        if not classifications:
            return
        full_catalog_specs = [
            item.spec
            for item in classifications
            if item.status is BucketStatus.RESTORE_FROM_CATALOG
        ]
        full_source_specs = [
            item.spec
            for item in classifications
            if item.status is BucketStatus.REINGEST_FROM_SOURCE
        ]
        feature_dataset_ids = set(parquet_specs.keys())
        market_catalog_specs = [
            spec for spec in full_catalog_specs if spec.dataset_id not in feature_dataset_ids
        ]
        feature_catalog_specs = [
            spec for spec in full_catalog_specs if spec.dataset_id in feature_dataset_ids
        ]
        market_source_specs = [
            spec for spec in full_source_specs if spec.dataset_id not in feature_dataset_ids
        ]
        max_buckets = self._get_coverage_bucket_cap()
        catalog_specs, source_specs, skipped_catalog, skipped_source = self._apply_bucket_cap(
            market_catalog_specs,
            market_source_specs,
            max_buckets,
        )
        if feature_catalog_specs:
            logger.info(
                "coverage.feature_restore.pending",
                extra={
                    "datasets": sorted({spec.dataset_id for spec in feature_catalog_specs}),
                    "buckets": len(feature_catalog_specs),
                },
            )
            if not dry_run:
                self._restore_feature_buckets(
                    specs=feature_catalog_specs,
                    scheduler_config=scheduler_config,
                    parquet_specs=parquet_specs,
                )
        feature_source_specs = [
            spec for spec in full_source_specs if spec.dataset_id in feature_dataset_ids
        ]
        if feature_source_specs:
            logger.info(
                "coverage.feature_reingest.pending",
                extra={
                    "datasets": sorted({spec.dataset_id for spec in feature_source_specs}),
                    "buckets": len(feature_source_specs),
                },
            )
            if not dry_run:
                self._reingest_feature_buckets(
                    specs=feature_source_specs,
                    scheduler_config=scheduler_config,
                )
        if dry_run:
            self._log_coverage_dry_run_summary(
                entries=entries,
                classifications=classifications,
                parquet_specs=parquet_specs,
            )
            logger.info(
                "coverage_manager.dry_run",
                extra={
                    "feature_restore_buckets": len(feature_catalog_specs),
                    "feature_reingest_buckets": len(feature_source_specs),
                    "catalog_restore_buckets": len(catalog_specs),
                    "source_reingest_buckets": len(source_specs),
                },
            )
            for classification in classifications:
                _coverage_buckets_total.labels(status=classification.status.name.lower()).inc()
            return
        if skipped_catalog or skipped_source:
            logger.warning(
                "coverage_manager.bucket_cap_applied",
                extra={
                    "cap": max_buckets,
                    "skipped_catalog": skipped_catalog,
                    "skipped_source": skipped_source,
                    "total_requested": len(full_catalog_specs) + len(full_source_specs),
                },
            )
            pipeline_status["errors"].append(
                f"coverage_cap:cap={max_buckets}:skipped_catalog={skipped_catalog}:skipped_source={skipped_source}",
            )
        self._restore_catalog_buckets(catalog_specs, scheduler_config)
        if source_specs:
            if self._coverage_reingest_enabled():
                try:
                    self.scheduler.run_targeted_update(source_specs)
                except Exception:
                    logger.warning("scheduler.targeted_update_failed", exc_info=True)
                    self._record_coverage_error(reason="targeted_update_failed")
                    _coverage_restore_failures_total.labels(stage="targeted_update").inc()
                    self._maybe_raise_coverage_failure(reason="targeted_update_failed")
            else:
                logger.info(
                    "coverage.source_reingest.disabled",
                    extra={"buckets": len(source_specs)},
                )
        for classification in classifications:
            _coverage_buckets_total.labels(status=classification.status.name.lower()).inc()
        last_error = self._coverage_status().get("last_error")
        if last_error:
            self._maybe_raise_coverage_failure(reason=last_error)

    def _log_coverage_dry_run_summary(
        self,
        *,
        entries: Sequence[CoverageDatasetEntry],
        classifications: Sequence[BucketClassification],
        parquet_specs: Mapping[str, ParquetCoverageSpec],
    ) -> None:
        """
        Emit a dataset-level dry-run summary with inclusion/exclusion reasons.
        """
        if not entries:
            return
        stats: dict[str, dict[str, int]] = {
            entry.dataset.dataset_id: {"total": 0, "healthy": 0, "restore": 0, "reingest": 0}
            for entry in entries
        }
        for classification in classifications:
            dataset_id = classification.spec.dataset_id
            counters = stats.setdefault(
                dataset_id,
                {"total": 0, "healthy": 0, "restore": 0, "reingest": 0},
            )
            counters["total"] += 1
            if classification.status is BucketStatus.HEALTHY:
                counters["healthy"] += 1
            elif classification.status is BucketStatus.RESTORE_FROM_CATALOG:
                counters["restore"] += 1
            elif classification.status is BucketStatus.REINGEST_FROM_SOURCE:
                counters["reingest"] += 1
        parquet_dataset_ids = set(parquet_specs.keys())
        summaries: list[dict[str, object]] = []
        for entry in entries:
            dataset_id = entry.dataset.dataset_id
            instruments = entry.dataset.normalized_instruments()
            counts = stats.get(dataset_id, {"total": 0, "healthy": 0, "restore": 0, "reingest": 0})
            restore = counts["restore"]
            reingest = counts["reingest"]
            reason = "no_action_needed"
            status = "healthy"
            if dataset_id in SUPPORTED_FEATURE_DATASET_IDS and dataset_id not in parquet_dataset_ids:
                reason = "parquet_spec_missing"
                status = "excluded"
            elif not instruments:
                reason = "no_instruments_configured"
                status = "excluded"
            elif counts["total"] == 0:
                reason = "no_buckets_in_window"
                status = "excluded"
            elif restore or reingest:
                status = "included"
                if restore and reingest:
                    reason = "restore_and_reingest_required"
                elif restore:
                    reason = "restore_required"
                else:
                    reason = "reingest_required"
            summaries.append(
                {
                    "dataset_id": dataset_id,
                    "schema": entry.dataset.schema,
                    "instruments": len(instruments),
                    "buckets_total": counts["total"],
                    "buckets_healthy": counts["healthy"],
                    "buckets_restore": restore,
                    "buckets_reingest": reingest,
                    "status": status,
                    "reason": reason,
                },
            )
        logger.info(
            "coverage_manager.dry_run_dataset_summary",
            extra={"datasets": summaries},
        )

    def _maybe_raise_coverage_failure(self, *, reason: str, raise_immediately: bool = False) -> None:
        if not self._coverage_restore_enabled():
            return
        allow_failure = parse_bool_env(os.environ.get("COVERAGE_RESTORE_ALLOW_FAILURE"))
        if allow_failure and not raise_immediately:
            logger.warning(
                "coverage_restore.allow_failure",
                extra={"reason": reason},
            )
            return
        raise CoverageRestorationError(reason)

    def _restore_catalog_buckets(
        self,
        specs: list[BucketSpec],
        scheduler_config: SchedulerConfig,
    ) -> None:
        if not specs:
            return
        rehydrator = self._rehydrator
        if rehydrator is None:
            if self._catalog_obj is None:
                logger.warning(
                    "catalog coverage restoration skipped because catalog instance unavailable",
                )
                return
            base_config = self._rehydrator_config or CatalogRehydrationConfig(enabled=True)
            if not base_config.enabled:
                base_config = replace(base_config, enabled=True)
            db_connection = self._resolve_db_connection(scheduler_config)
            if not db_connection:
                logger.warning("catalog coverage restoration skipped because DB connection missing")
                return
            rehydrator = ParquetCatalogRehydrator(
                catalog=self._catalog_obj,
                db_connection=db_connection,
                config=base_config,
                registry=self._resolve_rehydration_registry(),
            )
        grouped: dict[tuple[str, str], list[BucketSpec]] = defaultdict(list)
        for spec in specs:
            grouped[(spec.dataset_id, spec.schema)].append(spec)
        for (dataset_id, schema), bucket_specs in grouped.items():
            instrument_ids = tuple({spec.instrument_id for spec in bucket_specs})
            try:
                rehydrator.rehydrate_missing_data(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_ids=list(instrument_ids),
                    buckets=list(bucket_specs),
                )
            except Exception:
                logger.warning(
                    "coverage_manager.catalog_restore_failed",
                    exc_info=True,
                    extra={"dataset_id": dataset_id, "schema": schema},
                )

    def _restore_feature_buckets(
        self,
        *,
        specs: Sequence[BucketSpec],
        scheduler_config: SchedulerConfig,
        parquet_specs: Mapping[str, ParquetCoverageSpec],
    ) -> None:
        if not specs:
            return
        if not parquet_specs:
            logger.debug(
                "feature coverage restoration skipped because no parquet specs were resolved",
            )
            return
        db_connection = self._resolve_db_connection(scheduler_config)
        if not db_connection:
            logger.warning("feature coverage restoration skipped because DB connection missing")
            self._record_coverage_error(reason="feature_restore_db_missing")
            return
        try:
            restorer = FeatureCoverageRestorer(
                db_connection=db_connection,
                parquet_specs=parquet_specs,
            )
            result = restorer.restore(specs)
        except Exception:
            logger.warning("coverage.feature_restore.failed", exc_info=True)
            _coverage_restore_failures_total.labels(stage="feature_restore").inc()
            pipeline_status["errors"].append("feature_restore_failed")
            self._record_coverage_error(reason="feature_restore_failed")
            return
        logger.info(
            "coverage.feature_restore.completed",
            extra={
                "datasets": result.datasets_processed,
                "instruments": result.instruments_processed,
                "rows_written": result.rows_written,
                "requested_buckets": result.buckets_requested,
                "restored_buckets": result.buckets_restored,
            },
        )
        if result.failures:
            logger.warning(
                "coverage.feature_restore.partial",
                extra={"failures": result.failures},
            )
            pipeline_status["errors"].append(f"feature_restore_partial:{len(result.failures)}")
            self._record_coverage_error(reason="feature_restore_partial")

    @staticmethod
    def _bucket_datetime_window(specs: Sequence[BucketSpec]) -> tuple[datetime, datetime]:
        if not specs:
            raise ValueError("specs cannot be empty")
        start = min(spec.bucket_start for spec in specs)
        end = max(spec.bucket_start for spec in specs) + timedelta(days=1)
        if end <= start:
            end = start + timedelta(days=1)
        return start, end

    @staticmethod
    def _bucket_date_window(specs: Sequence[BucketSpec]) -> tuple[date, date]:
        if not specs:
            raise ValueError("specs cannot be empty")
        days = [spec.bucket_start.date() for spec in specs]
        return min(days), max(days)

    @staticmethod
    def _unique_instruments(specs: Sequence[BucketSpec]) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for spec in specs:
            token = spec.instrument_id.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return tuple(ordered)

    @staticmethod
    def _load_macro_series_ids(
        path: Path = Path("ml/config/macro_fred_series.txt"),
    ) -> tuple[str, ...]:
        """
        Load macro series identifiers from the configured series list file.
        """
        if not path.exists():
            return ()
        series: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            token = raw.strip()
            if not token or token.startswith("#"):
                continue
            series.append(token)
        return tuple(series)

    @staticmethod
    def _resolve_raw_tier1_dir() -> Path:
        from ml.config.base import DataCollectorConfig

        return Path(DataCollectorConfig().data_dir)

    @staticmethod
    def _resolve_feature_paths() -> tuple[Path, Path, Path]:
        from ml.stores.feature_raw_writer import FeatureDatasetParquetRawWriter

        writer = FeatureDatasetParquetRawWriter()
        return writer.events_path.parent, writer.micro_base_dir, writer.l2_base_dir

    def _build_feature_raw_writer(
        self,
        *,
        earnings_config: object | None,
    ) -> RawIngestionWriterProtocol | None:
        try:
            from ml.stores.feature_raw_writer import CompositeRawIngestionWriter
            from ml.stores.feature_raw_writer import FeatureDatasetParquetRawWriter
        except Exception:
            logger.debug("feature_reingest.raw_writer_unavailable", exc_info=True)
            return None

        writers: list[RawIngestionWriterProtocol] = [FeatureDatasetParquetRawWriter()]
        if earnings_config is not None:
            try:
                from ml.features.earnings.raw_writer import EarningsParquetRawWriter
            except Exception:
                logger.debug("feature_reingest.earnings_writer_unavailable", exc_info=True)
            else:
                base_path = getattr(earnings_config, "parquet_root", None)
                partition_keys = getattr(earnings_config, "parquet_partition_keys", None)
                if base_path is not None:
                    writers.append(
                        EarningsParquetRawWriter(
                            base_path=base_path,
                            partition_keys=partition_keys or ("ticker",),
                        ),
                    )
        if not writers:
            return None
        if len(writers) == 1:
            return writers[0]
        return CompositeRawIngestionWriter(writers)

    def _build_feature_data_store(
        self,
        *,
        db_connection: str,
        earnings_config: object | None,
    ) -> DataStoreFacadeProtocol | None:
        try:
            from ml.core.common.registry_initialization import build_data_store

            raw_writer = self._build_feature_raw_writer(earnings_config=earnings_config)
            return build_data_store(
                db_connection=db_connection,
                raw_writer=raw_writer,
            )
        except Exception:
            logger.warning("coverage.feature_reingest.store_init_failed", exc_info=True)
            return None

    def _reingest_feature_cache(
        self,
        *,
        data_store: DataStoreFacadeProtocol,
        specs: Sequence[BucketSpec],
        cache_dir: Path,
        label: str,
    ) -> None:
        symbols = self._unique_instruments(specs)
        if not symbols:
            return
        start_date, end_date = self._bucket_date_window(specs)
        raw_base_dir = self._resolve_raw_tier1_dir()
        max_workers = max(1, self._get_int_env("MAX_WORKERS", 4))

        if label == "micro":
            from ml.tasks.caches import MicroCacheHydrationConfig
            from ml.tasks.caches import hydrate_micro_caches
            from ml.tasks.caches import ingest_micro_cache_partitions

            cfg_micro = MicroCacheHydrationConfig(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                raw_base_dir=raw_base_dir,
                cache_dir=cache_dir,
                max_workers=max_workers,
                force_rebuild=False,
            )
            result = hydrate_micro_caches(cfg_micro)
            ingest_micro_cache_partitions(
                data_store=data_store,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                run_id=f"coverage_micro_{int(time.time())}",
            )
        else:
            from ml.tasks.caches import L2CacheHydrationConfig
            from ml.tasks.caches import hydrate_l2_caches
            from ml.tasks.caches import ingest_l2_cache_partitions

            cfg_l2 = L2CacheHydrationConfig(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                raw_base_dir=raw_base_dir,
                cache_dir=cache_dir,
                max_workers=max_workers,
                force_rebuild=False,
            )
            result = hydrate_l2_caches(cfg_l2)
            ingest_l2_cache_partitions(
                data_store=data_store,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                run_id=f"coverage_l2_{int(time.time())}",
            )

        if result.failed:
            failed_symbols = ", ".join(item.symbol for item in result.failed)
            raise RuntimeError(f"{label}_cache_failed:{failed_symbols}")

    def _reingest_feature_buckets(
        self,
        *,
        specs: Sequence[BucketSpec],
        scheduler_config: SchedulerConfig,
    ) -> None:
        if not specs:
            return
        if not self._feature_reingest_enabled():
            logger.info(
                "coverage.feature_reingest.disabled",
                extra={"buckets": len(specs)},
            )
            return
        db_connection = self._resolve_db_connection(scheduler_config)
        if not db_connection:
            logger.warning("coverage.feature_reingest.db_missing")
            self._record_coverage_error(reason="feature_reingest_db_missing")
            self._maybe_raise_coverage_failure(reason="feature_reingest_db_missing")
            return

        grouped: dict[str, list[BucketSpec]] = defaultdict(list)
        for spec in specs:
            grouped[spec.dataset_id].append(spec)

        supported = {
            MACRO_RELEASES_DATASET_ID,
            MACRO_OBSERVATIONS_DATASET_ID,
            EVENTS_CALENDAR_DATASET_ID,
            EARNINGS_ACTUALS_DATASET_ID,
            EARNINGS_ESTIMATES_DATASET_ID,
            MICRO_MINUTE_DATASET_ID,
            L2_MINUTE_DATASET_ID,
        }
        unsupported = sorted(set(grouped) - supported)
        if unsupported:
            logger.warning(
                "coverage.feature_reingest.unsupported",
                extra={"datasets": unsupported},
            )
            pipeline_status["errors"].append(
                f"feature_reingest_unsupported:{','.join(unsupported)}",
            )

        earnings_specs = grouped.get(EARNINGS_ACTUALS_DATASET_ID, []) + grouped.get(
            EARNINGS_ESTIMATES_DATASET_ID,
            [],
        )
        earnings_config: object | None = None
        earnings_tickers = self._unique_instruments(earnings_specs)
        if earnings_tickers:
            from ml.config.earnings_ingestion import EarningsIngestionConfig

            earnings_config = EarningsIngestionConfig(
                postgres_dsn=db_connection,
                override_symbols=earnings_tickers,
            )

        data_store = self._build_feature_data_store(
            db_connection=db_connection,
            earnings_config=earnings_config,
        )
        if data_store is None:
            self._record_coverage_error(reason="feature_reingest_store_missing")
            self._maybe_raise_coverage_failure(reason="feature_reingest_store_missing")
            return

        events_dir, micro_dir, l2_dir = self._resolve_feature_paths()
        failures: dict[str, str] = {}

        macro_specs = grouped.get(MACRO_RELEASES_DATASET_ID, []) + grouped.get(
            MACRO_OBSERVATIONS_DATASET_ID,
            [],
        )
        if macro_specs:
            dataset_ids = sorted({spec.dataset_id for spec in macro_specs})
            try:
                from ml.data.ingest.macro_refresh import ensure_macro_ready
                from ml.orchestration.config_types import MacroIngestionConfig

                cfg = MacroIngestionConfig()
                series_ids = self._unique_instruments(macro_specs) or cfg.series_ids
                ensure_macro_ready(
                    fred_path=Path(cfg.fred_path),
                    vintage_dir=Path(cfg.vintage_dir) if cfg.vintage_dir else None,
                    max_age=timedelta(hours=cfg.max_staleness_hours),
                    data_store=data_store,
                    series_ids=series_ids,
                    watermark_config=cfg.watermark_config,
                )
                for dataset_id in dataset_ids:
                    _feature_reingest_total.labels(dataset_id=dataset_id, status="success").inc()
            except Exception as exc:
                for dataset_id in dataset_ids:
                    _feature_reingest_total.labels(dataset_id=dataset_id, status="error").inc()
                failures["macro"] = str(exc)
                logger.warning("coverage.feature_reingest.macro_failed", exc_info=True)

        event_specs = grouped.get(EVENTS_CALENDAR_DATASET_ID, [])
        if event_specs:
            try:
                from ml.orchestration.config_types import MacroIngestionConfig
                from ml.preprocessing.event_ingestion import EventIngestionConfig
                from ml.preprocessing.event_ingestion import EventIngestionUtility

                start_dt, end_dt = self._bucket_datetime_window(event_specs)
                macro_cfg = MacroIngestionConfig()
                series_ids = self._unique_instruments(event_specs)
                if not series_ids:
                    series_ids = macro_cfg.series_ids or self._load_macro_series_ids()
                alfred_dir = Path(macro_cfg.vintage_dir) if macro_cfg.vintage_dir else None
                config = EventIngestionConfig(
                    start=start_dt,
                    end=end_dt,
                    out_dir=events_dir,
                    alfred_vintage_dir=alfred_dir,
                    economic_series=series_ids or ("CPI",),
                )
                utility = EventIngestionUtility(
                    config,
                    data_store=data_store,
                    ingest_run_id="coverage_events",
                )
                utility.ingest()
                _feature_reingest_total.labels(
                    dataset_id=EVENTS_CALENDAR_DATASET_ID,
                    status="success",
                ).inc()
            except Exception as exc:
                _feature_reingest_total.labels(
                    dataset_id=EVENTS_CALENDAR_DATASET_ID,
                    status="error",
                ).inc()
                failures["events"] = str(exc)
                logger.warning("coverage.feature_reingest.events_failed", exc_info=True)

        if earnings_config is not None:
            try:
                from ml.features.earnings.ingestion.service import EarningsIngestionService

                result = EarningsIngestionService(
                    config=_cast(Any, earnings_config),
                    writer=data_store,
                ).run()
                logger.info(
                    "coverage.feature_reingest.earnings_completed",
                    extra={
                        "tickers": result.tickers_attempted,
                        "actuals_written": result.actuals_written,
                        "estimates_written": result.estimates_written,
                        "failures": result.failures,
                    },
                )
                for dataset_id in (EARNINGS_ACTUALS_DATASET_ID, EARNINGS_ESTIMATES_DATASET_ID):
                    _feature_reingest_total.labels(dataset_id=dataset_id, status="success").inc()
            except Exception as exc:
                for dataset_id in (EARNINGS_ACTUALS_DATASET_ID, EARNINGS_ESTIMATES_DATASET_ID):
                    _feature_reingest_total.labels(dataset_id=dataset_id, status="error").inc()
                failures["earnings"] = str(exc)
                logger.warning("coverage.feature_reingest.earnings_failed", exc_info=True)

        if MICRO_MINUTE_DATASET_ID in grouped:
            try:
                self._reingest_feature_cache(
                    data_store=data_store,
                    specs=grouped[MICRO_MINUTE_DATASET_ID],
                    cache_dir=micro_dir,
                    label="micro",
                )
                _feature_reingest_total.labels(
                    dataset_id=MICRO_MINUTE_DATASET_ID,
                    status="success",
                ).inc()
            except Exception as exc:
                _feature_reingest_total.labels(
                    dataset_id=MICRO_MINUTE_DATASET_ID,
                    status="error",
                ).inc()
                failures["micro"] = str(exc)
                logger.warning("coverage.feature_reingest.micro_failed", exc_info=True)

        if L2_MINUTE_DATASET_ID in grouped:
            try:
                self._reingest_feature_cache(
                    data_store=data_store,
                    specs=grouped[L2_MINUTE_DATASET_ID],
                    cache_dir=l2_dir,
                    label="l2",
                )
                _feature_reingest_total.labels(
                    dataset_id=L2_MINUTE_DATASET_ID,
                    status="success",
                ).inc()
            except Exception as exc:
                _feature_reingest_total.labels(
                    dataset_id=L2_MINUTE_DATASET_ID,
                    status="error",
                ).inc()
                failures["l2"] = str(exc)
                logger.warning("coverage.feature_reingest.l2_failed", exc_info=True)

        if failures:
            _coverage_restore_failures_total.labels(stage="feature_reingest").inc()
            pipeline_status["errors"].append(f"feature_reingest_failed:{len(failures)}")
            self._record_coverage_error(reason="feature_reingest_failed")
            self._maybe_raise_coverage_failure(reason="feature_reingest_failed")
            return

        logger.info(
            "coverage.feature_reingest.completed",
            extra={"datasets": sorted(set(grouped) & supported)},
        )

    def _should_rescan_catalog(self) -> bool:
        return bool(self._rehydrator_config and self._rehydrator_config.rescan_on_schedule)

    def run_coverage_restoration_once(self, *, dry_run: bool = False) -> CoverageStatus:
        """
        Execute a standalone coverage restoration cycle.
        """
        logger.info("Starting standalone coverage restoration cycle")
        config = self._create_config()
        self._scheduler_config = config
        catalog = None
        try:
            self._bootstrap_database(config)
            catalog = self._initialize_catalog(config)
            use_orchestrator = parse_bool_env(os.environ.get("USE_ORCHESTRATOR"))
            dual_write = parse_bool_env(os.environ.get("DUAL_WRITE"))
            rehydrator_config = self._resolve_rehydrator_config()
            feature_engineer = self._build_feature_engineer()
            self.scheduler = DataScheduler(
                catalog=catalog,
                config=config,
                feature_engineer=feature_engineer,
                use_orchestrator=use_orchestrator,
                dual_write=dual_write,
                dual_write_dataset_types=self._dual_write_dataset_types_from_env(),
                dataset_type_identifier_templates=rehydrator_config.dataset_type_identifier_templates,
            )
            self._build_catalog_rehydrator(catalog=catalog, scheduler_config=config)
            self._run_coverage_restoration(config, dry_run=dry_run)
        finally:
            if self.scheduler is not None:
                try:
                    self.scheduler.stop()
                except Exception:
                    logger.debug(
                        "Scheduler.stop() failed after coverage restoration run",
                        exc_info=True,
                    )
                self.scheduler = None
        return self._coverage_status()

    def run(self) -> None:
        """
        Run the ML pipeline based on environment configuration.
        """
        try:
            # Check dependencies
            check_ml_dependencies(["databento", "polars", "pandas", "numpy"])

            # Get pipeline mode
            mode = os.environ.get("PIPELINE_MODE", "daily").lower()

            # Create configuration
            config = self._create_config()
            self._scheduler_config = config
            logger.info(f"Starting ML pipeline in {mode} mode")
            logger.info(f"Universe symbols: {len(config.symbols)}")

            try:
                self._bootstrap_database(config)
            except (MigrationRunnerError, SchemaHealthCheckError) as exc:
                logger.error("Database bootstrap failed: %s", exc, exc_info=True)
                raise

            # Initialize components
            _feature_store, _model_store = self._initialize_stores(config)
            catalog = self._initialize_catalog(config)

            # Create scheduler with unified ingestion flags (env truthy: 1,true,yes,on)
            use_orchestrator = parse_bool_env(os.environ.get("USE_ORCHESTRATOR"))
            dual_write = parse_bool_env(os.environ.get("DUAL_WRITE"))
            rehydrator_config = self._resolve_rehydrator_config()
            feature_engineer = self._build_feature_engineer()
            self.scheduler = DataScheduler(
                catalog=catalog,
                config=config,
                feature_engineer=feature_engineer,
                use_orchestrator=use_orchestrator,
                dual_write=dual_write,
                dual_write_dataset_types=self._dual_write_dataset_types_from_env(),
                dataset_type_identifier_templates=rehydrator_config.dataset_type_identifier_templates,
            )

            self._build_catalog_rehydrator(catalog=catalog, scheduler_config=config)
            self._run_coverage_restoration(config)
            if not self._coverage_restore_enabled():
                self._run_catalog_rehydration(config, note="startup")

            # Update health status
            pipeline_status["healthy"] = True
            pipeline_status["last_run"] = datetime.now().isoformat()

            # Run based on mode
            if mode == "backfill":
                self._run_backfill()
            elif mode == "daily":
                self._run_daily()
            elif mode == "realtime":
                self._run_realtime()
            else:
                raise ValueError(f"Unknown pipeline mode: {mode}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            pipeline_status["healthy"] = False
            pipeline_status["errors"].append(str(e))
            sys.exit(1)

    def _require_scheduler(self) -> DataScheduler:
        """
        Return the initialized scheduler or raise if unavailable.
        """
        if self.scheduler is None:
            raise RuntimeError("Pipeline scheduler has not been initialized")
        return self.scheduler

    def _run_backfill(self) -> None:
        """
        Run pipeline in backfill mode.
        """
        logger.info("Running backfill mode")

        # Backfill is currently mapped to a single daily update for simplicity.
        # A full backfill loop should iterate dates and call collection per day.
        scheduler = self._require_scheduler()
        if (
            self._scheduler_config is not None
            and self._should_rescan_catalog()
            and not self._coverage_restore_enabled()
        ):
            self._run_catalog_rehydration(self._scheduler_config, note="backfill")
        try:
            scheduler.run_daily_update()
        except Exception as exc:
            logger.error("Backfill update failed: %s", exc, exc_info=True)
            pipeline_status["errors"].append(str(exc))
            logger.warning("Backfill completed with errors")
            return

        logger.info("Backfill completed successfully")

    def _run_daily(self) -> None:
        """
        Run pipeline in daily scheduled mode.
        """
        logger.info("Running daily scheduled mode")
        self.running = True
        scheduler = self._require_scheduler()

        if (
            self._scheduler_config is not None
            and self._should_rescan_catalog()
            and not self._coverage_restore_enabled()
        ):
            self._run_catalog_rehydration(self._scheduler_config, note="daily-initial")

        # Always perform an immediate daily update on entering daily mode
        try:
            scheduler.run_daily_update()
            pipeline_status["last_run"] = datetime.now(UTC).isoformat()
        except Exception as exc:
            logger.error("Initial daily update failed: %s", exc, exc_info=True)
            pipeline_status["errors"].append(str(exc))

        # Resolve schedule spec; allow HH:MM or crontab-like "M H * * *"
        schedule_raw = os.environ.get("PIPELINE_SCHEDULE")
        interval_seconds = int(os.environ.get("REALTIME_INTERVAL", "300"))

        while self.running and not self._shutdown_event.is_set():
            sleep_seconds: float
            if schedule_raw:
                try:
                    daily: DailyTime = parse_daily_spec(schedule_raw)
                    now = datetime.now(UTC)
                    next_run = compute_next_utc_run(now, daily)
                    sleep_seconds = max(0.0, (next_run - now).total_seconds())
                except ValueError:
                    # Bad spec; fall back to interval mode
                    sleep_seconds = float(interval_seconds)
            else:
                sleep_seconds = float(interval_seconds)

            # Sleep in short chunks to honor shutdown quickly
            end_time = time.monotonic() + sleep_seconds
            while time.monotonic() < end_time:
                if self._shutdown_event.wait(timeout=min(1.0, end_time - time.monotonic())):
                    break
            if self._shutdown_event.is_set() or not self.running:
                break

            try:
                if (
                    self._scheduler_config is not None
                    and self._should_rescan_catalog()
                    and not self._coverage_restore_enabled()
                ):
                    self._run_catalog_rehydration(self._scheduler_config, note="daily-loop")
                scheduler.run_daily_update()
                pipeline_status["last_run"] = datetime.now(UTC).isoformat()
            except Exception as exc:
                logger.error("Scheduled daily update failed: %s", exc, exc_info=True)
                pipeline_status["errors"].append(str(exc))

        logger.info("Daily scheduler stopped")

    def _run_realtime(self) -> None:
        """
        Run pipeline in realtime mode.
        """
        logger.info("Running realtime mode")
        self.running = True

        # Run continuous updates
        while self.running and not self._shutdown_event.is_set():
            try:
                scheduler = self._require_scheduler()

                if (
                    self._scheduler_config is not None
                    and self._should_rescan_catalog()
                    and not self._coverage_restore_enabled()
                ):
                    self._run_catalog_rehydration(self._scheduler_config, note="realtime")

                # Use standardized retry/backoff for transient realtime update failures
                from ml.common.retry_utils import retry_with_backoff as _retry

                def _on_exc(attempt: int, exc: BaseException) -> None:
                    wait_time = min(60, 2 ** (attempt + 1))
                    logger.warning(
                        f"Realtime update attempt {attempt + 1} failed: {exc}. "
                        f"Retrying in {wait_time}s...",
                    )

                def _do_update() -> None:
                    scheduler.run_daily_update()

                _retry(
                    _do_update,
                    max_attempts=int(os.environ.get("REALTIME_MAX_RETRIES", "3")),
                    initial_delay=1.0,
                    multiplier=2.0,
                    max_delay=60.0,
                    on_exception=_on_exc,
                    sleep_fn=time.sleep,
                )

                pipeline_status["last_run"] = datetime.now().isoformat()

                # Wait before next update (configurable)
                interval = int(os.environ.get("REALTIME_INTERVAL", "300"))  # 5 minutes default
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Realtime update failed after retries: {e}")
                pipeline_status["errors"].append(str(e))
                # Cooldown before continuing loop
                time.sleep(60)

        logger.info("Realtime mode stopped")


def main() -> None:
    """
    Run main entry point.
    """
    # Start health check server in background
    health_host = os.environ.get("HEALTH_CHECK_HOST", "127.0.0.1")
    health_thread = threading.Thread(
        target=lambda: app.run(
            host=health_host,
            port=int(os.environ.get("HEALTH_CHECK_PORT", "8080")),
            debug=False,
        ),
    )
    health_thread.daemon = True
    health_thread.start()

    # Auto-start observability flushing if configured via env
    try:
        mgr: MLIntegrationManager = MLIntegrationManager.__new__(MLIntegrationManager)
        auto_start_if_configured(mgr)
    except Exception:
        logger.debug(
            "Observability auto-start skipped due to configuration or environment",
            exc_info=True,
        )

    # Run pipeline
    runner = PipelineRunner()
    runner.run()


if __name__ == "__main__":
    main()
