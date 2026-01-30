"""
Coverage management primitives for database restoration workflows.

These helpers classify bucket-level coverage across SQL and parquet stores so
the pipeline can decide whether to restore data from backups or re-ingest via
external feeds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from enum import Enum
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING

from ml.data.coverage.types import DAY_NS
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.providers import CatalogCoverageProvider
from ml.stores.providers import NullCoverageProvider
from ml.stores.providers import ParquetCoverageSpec
from ml.stores.providers import PartitionedParquetCoverageProvider
from ml.stores.providers import SqlCoverageOverride
from ml.stores.providers import SqlCoverageProvider
from ml.stores.providers import UnionCoverageProvider
from ml.stores.schema_audit import SchemaAuditor


if TYPE_CHECKING:
    from ml.config.dataset_coverage import CoverageDatasetEntry


logger = logging.getLogger(__name__)


class CoverageBucketMode(str, Enum):
    """
    Bucket generation strategy for coverage.

    DAILY: fixed daily buckets across lookback window.
    CATALOG: use catalog/SQL bucket union to avoid sparse-dataset false gaps.
    """

    DAILY = "daily"
    CATALOG = "catalog"


@dataclass(frozen=True, slots=True)
class DatasetCoverageConfig:
    """
    Coverage configuration for a specific dataset/schema/instrument set.
    """

    dataset_id: str
    schema: str
    instruments: tuple[str, ...]
    entity_field: str = "instrument_id"
    bucket_mode: CoverageBucketMode = CoverageBucketMode.DAILY

    def __post_init__(self) -> None:
        if not self.dataset_id:
            msg = "dataset_id must be provided"
            raise ValueError(msg)
        if not self.schema:
            msg = "schema must be provided"
            raise ValueError(msg)
        if not self.entity_field:
            msg = "entity_field must be provided"
            raise ValueError(msg)
        if not isinstance(self.bucket_mode, CoverageBucketMode):
            msg = "bucket_mode must be a CoverageBucketMode"
            raise ValueError(msg)

    def normalized_instruments(self) -> tuple[str, ...]:
        """
        Return the configured identifiers with whitespace trimmed.
        """
        return tuple(item.strip() for item in self.instruments if item and item.strip())


@dataclass(frozen=True, slots=True)
class CoverageManagerConfig:
    """
    Global coverage manager configuration.
    """

    datasets: tuple[DatasetCoverageConfig, ...]
    lookback_days: int = 5

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            msg = "lookback_days must be >= 1"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BucketSpec:
    """
    Normalized bucket specification (day-level granularity).
    """

    dataset_id: str
    schema: str
    instrument_id: str
    bucket_start_ns: int
    entity_field: str = "instrument_id"

    @property
    def bucket_index(self) -> int:
        return self.bucket_start_ns // DAY_NS

    @property
    def bucket_start(self) -> datetime:
        return datetime.fromtimestamp(self.bucket_start_ns / 1_000_000_000, tz=UTC)


class BucketStatus(Enum):
    """
    Classification of bucket coverage.
    """

    HEALTHY = auto()
    RESTORE_FROM_CATALOG = auto()
    REINGEST_FROM_SOURCE = auto()


@dataclass(frozen=True, slots=True)
class BucketClassification:
    """
    Classification result for a bucket.
    """

    spec: BucketSpec
    has_sql: bool
    has_catalog: bool

    @property
    def status(self) -> BucketStatus:
        if self.has_sql:
            return BucketStatus.HEALTHY
        if self.has_catalog:
            return BucketStatus.RESTORE_FROM_CATALOG
        return BucketStatus.REINGEST_FROM_SOURCE


@dataclass(slots=True)
class CoverageManager:
    """
    Coordinates coverage inspection across SQL and catalog stores.
    """

    config: CoverageManagerConfig
    sql_provider: CoverageProviderProtocol
    catalog_provider: CoverageProviderProtocol
    schema_auditor: SchemaAuditor | None = None
    _last_classification: tuple[BucketClassification, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def generate_bucket_specs(self, *, reference_time: datetime | None = None) -> tuple[BucketSpec, ...]:
        """
        Produce bucket specs for all datasets/instruments within the lookback window.
        """
        day_start, window_start, window_end = self._coverage_window(reference_time)
        specs: list[BucketSpec] = []
        for dataset in self.config.datasets:
            for instrument in dataset.normalized_instruments():
                if not instrument:
                    continue
                bucket_indices, _, _ = self._resolve_bucket_indices(
                    dataset=dataset,
                    instrument_id=instrument,
                    day_start=day_start,
                    window_start=window_start,
                    window_end=window_end,
                )
                for bucket_idx in bucket_indices:
                    specs.append(
                        BucketSpec(
                            dataset_id=dataset.dataset_id,
                            schema=dataset.schema,
                            instrument_id=instrument,
                            bucket_start_ns=bucket_idx * DAY_NS,
                            entity_field=dataset.entity_field,
                        ),
                    )
        return tuple(specs)

    def classify_buckets(self, *, reference_time: datetime | None = None) -> tuple[BucketClassification, ...]:
        """
        Compute coverage classification for all configured buckets.
        """
        day_start, window_start, window_end = self._coverage_window(reference_time)
        results: list[BucketClassification] = []
        for dataset in self.config.datasets:
            for instrument in dataset.normalized_instruments():
                if not instrument:
                    continue
                bucket_indices, sql_buckets, catalog_buckets = self._resolve_bucket_indices(
                    dataset=dataset,
                    instrument_id=instrument,
                    day_start=day_start,
                    window_start=window_start,
                    window_end=window_end,
                )
                for bucket_idx in bucket_indices:
                    results.append(
                        BucketClassification(
                            spec=BucketSpec(
                                dataset_id=dataset.dataset_id,
                                schema=dataset.schema,
                                instrument_id=instrument,
                                bucket_start_ns=bucket_idx * DAY_NS,
                                entity_field=dataset.entity_field,
                            ),
                            has_sql=bucket_idx in sql_buckets,
                            has_catalog=bucket_idx in catalog_buckets,
                        ),
                    )
        classifications = tuple(results)
        self._last_classification = classifications
        return classifications

    def _coverage_window(
        self,
        reference_time: datetime | None,
    ) -> tuple[datetime, int, int]:
        if reference_time is None:
            reference_time = datetime.now(tz=UTC)
        day_start = datetime(
            reference_time.year,
            reference_time.month,
            reference_time.day,
            tzinfo=UTC,
        )
        window_start = day_start - timedelta(days=self.config.lookback_days - 1)
        start_ns = int(window_start.timestamp() * 1_000_000_000)
        end_ns = int((day_start + timedelta(days=1)).timestamp() * 1_000_000_000)
        return day_start, start_ns, end_ns

    def _resolve_bucket_indices(
        self,
        *,
        dataset: DatasetCoverageConfig,
        instrument_id: str,
        day_start: datetime,
        window_start: int,
        window_end: int,
    ) -> tuple[list[int], set[int], set[int]]:
        sql_buckets = self.sql_provider.read_bucket_coverage(
            dataset_id=dataset.dataset_id,
            schema=dataset.schema,
            instrument_id=instrument_id,
            start_ns=window_start,
            end_ns=window_end,
            entity_field=dataset.entity_field,
        )
        catalog_buckets = self.catalog_provider.read_bucket_coverage(
            dataset_id=dataset.dataset_id,
            schema=dataset.schema,
            instrument_id=instrument_id,
            start_ns=window_start,
            end_ns=window_end,
            entity_field=dataset.entity_field,
        )
        if dataset.bucket_mode is CoverageBucketMode.CATALOG:
            combined = sorted(sql_buckets | catalog_buckets)
            return combined, sql_buckets, catalog_buckets
        bucket_indices: list[int] = []
        for offset in range(self.config.lookback_days):
            bucket_start = day_start - timedelta(days=offset)
            bucket_ns = int(bucket_start.timestamp() * 1_000_000_000)
            bucket_indices.append(bucket_ns // DAY_NS)
        return bucket_indices, sql_buckets, catalog_buckets

    def restore_all(self) -> tuple[BucketClassification, ...]:
        """
        Run schema audit (if configured) and classify coverage buckets.
        """
        if self.schema_auditor is not None:
            report = self.schema_auditor.inspect()
            if not report.healthy:
                msg = "Schema audit failed; aborting coverage restoration"
                logger.error(
                    "coverage_manager.schema_audit_failed",
                    extra={"report": report.to_dict()},
                )
                raise RuntimeError(msg)
        classifications = self.classify_buckets()
        _log_classification_summary(classifications)
        _log_parity_gaps(classifications)
        return classifications


def _group_by_instrument(
    specs: Iterable[BucketSpec],
) -> Mapping[tuple[str, str, str, str], list[BucketSpec]]:
    buckets: dict[tuple[str, str, str, str], list[BucketSpec]] = {}
    for spec in specs:
        key = (spec.dataset_id, spec.schema, spec.instrument_id, spec.entity_field)
        buckets.setdefault(key, []).append(spec)
    return buckets


def _log_classification_summary(classifications: Sequence[BucketClassification]) -> None:
    total = len(classifications)
    catalog = sum(1 for c in classifications if c.status is BucketStatus.RESTORE_FROM_CATALOG)
    source = sum(1 for c in classifications if c.status is BucketStatus.REINGEST_FROM_SOURCE)
    healthy = total - catalog - source
    logger.info(
        "coverage_manager.summary",
        extra={
            "buckets_total": total,
            "buckets_healthy": healthy,
            "buckets_restore_catalog": catalog,
            "buckets_reingest_source": source,
        },
    )


@dataclass(slots=True)
class _ParityStats:
    catalog: int = 0
    sql: int = 0
    catalog_only: int = 0
    instruments: set[str] = field(default_factory=set)


def _log_parity_gaps(classifications: Sequence[BucketClassification]) -> None:
    """
    Emit a warning when catalog coverage exists but SQL coverage is empty.
    """
    if not classifications:
        return
    stats: dict[tuple[str, str], _ParityStats] = {}
    for classification in classifications:
        key = (classification.spec.dataset_id, classification.spec.schema)
        entry = stats.setdefault(key, _ParityStats())
        if classification.has_catalog:
            entry.catalog += 1
        if classification.has_sql:
            entry.sql += 1
        if classification.has_catalog and not classification.has_sql:
            entry.catalog_only += 1
            entry.instruments.add(classification.spec.instrument_id)
    for (dataset_id, schema), entry in stats.items():
        if entry.catalog > 0 and entry.sql == 0:
            instruments = sorted(entry.instruments)
            logger.warning(
                "coverage_manager.parity_gap",
                extra={
                    "dataset_id": dataset_id,
                    "schema": schema,
                    "catalog_buckets": entry.catalog,
                    "sql_buckets": entry.sql,
                    "catalog_only_buckets": entry.catalog_only,
                    "instruments": instruments,
                },
            )

def _parse_dataset_arg(value: str) -> DatasetCoverageConfig:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("Dataset spec must be dataset_id:schema:symbol1,symbol2")
    dataset_id, schema, symbols_raw = parts
    instruments = tuple(
        sym.strip() for sym in symbols_raw.split(",") if sym.strip()
    )
    return DatasetCoverageConfig(
        dataset_id=dataset_id.strip(),
        schema=schema.strip(),
        instruments=instruments,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coverage manager CLI")
    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        help="PostgreSQL connection string (falls back to DB_CONNECTION/DATABASE_URL)",
    )
    parser.add_argument(
        "--catalog-path",
        dest="catalog_path",
        default=None,
        help="Path to the Parquet catalog",
    )
    parser.add_argument(
        "--coverage-config",
        dest="coverage_config",
        default=None,
        help="Optional TOML manifest describing dataset coverage entries (defaults to COVERAGE_DATASETS_FILE)",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Dataset spec in the form dataset_id:schema:symbol1,symbol2",
    )
    parser.add_argument(
        "--lookback-days",
        dest="lookback_days",
        type=int,
        default=5,
        help="Number of days to inspect (default: 5)",
    )
    parser.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit classification result as JSON",
    )
    return parser


def _default_db_url() -> str | None:
    for key in ("DB_CONNECTION", "DATABASE_URL", "NAUTILUS_DB"):
        value = os.getenv(key)
        if value:
            return value
    return None


def _coverage_config_candidates(cli_value: str | None) -> tuple[str, ...]:
    if cli_value:
        return (cli_value,)
    env_path = os.getenv("COVERAGE_DATASETS_FILE")
    if env_path:
        return (env_path,)
    default_path = Path("ml/config/coverage_datasets_tier1.toml")
    if default_path.exists():
        return (str(default_path),)
    return tuple()


def _load_manifest_entries(cli_value: str | None) -> tuple[CoverageDatasetEntry, ...]:
    """
    Resolve coverage dataset entries from CLI/env manifests.
    """
    from ml.config.dataset_coverage import load_dataset_coverage_entries

    for candidate in _coverage_config_candidates(cli_value):
        try:
            return load_dataset_coverage_entries(candidate)
        except FileNotFoundError:
            logger.warning("coverage.manager.config_missing", extra={"path": candidate})
        except Exception:
            logger.error("coverage.manager.config_invalid", exc_info=True, extra={"path": candidate})
            break
    return tuple()


def _entries_from_datasets_arg(specs: Sequence[str] | None) -> tuple[CoverageDatasetEntry, ...]:
    if not specs:
        return tuple()
    from ml.config.dataset_coverage import CoverageDatasetEntry

    entries: list[CoverageDatasetEntry] = []
    for raw in specs:
        dataset_cfg = _parse_dataset_arg(raw)
        entries.append(CoverageDatasetEntry(dataset=dataset_cfg))
    return tuple(entries)


def _resolve_dataset_entries(
    *,
    cli_specs: Sequence[str] | None,
    coverage_config: str | None,
) -> tuple[CoverageDatasetEntry, ...]:
    entries = list(_load_manifest_entries(coverage_config))
    entries.extend(_entries_from_datasets_arg(cli_specs))
    return tuple(entries)


def _build_catalog_provider(
    *,
    catalog_path: str | None,
    parquet_specs: Mapping[str, ParquetCoverageSpec],
) -> CoverageProviderProtocol:
    providers: list[CoverageProviderProtocol] = []
    if catalog_path:
        providers.append(CatalogCoverageProvider(catalog_path=catalog_path))
    if parquet_specs:
        providers.append(PartitionedParquetCoverageProvider(specs=parquet_specs))
    if not providers:
        return NullCoverageProvider()
    if len(providers) == 1:
        return providers[0]
    return UnionCoverageProvider(list(providers))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    db_url = args.db_url or _default_db_url()
    if not db_url:
        parser.error("Set --db-url or export DB_CONNECTION/DATABASE_URL")
    entries = _resolve_dataset_entries(cli_specs=args.datasets, coverage_config=args.coverage_config)
    if not entries:
        parser.error("No datasets resolved (set COVERAGE_DATASETS_FILE or pass --dataset)")
    parquet_specs: dict[str, ParquetCoverageSpec] = {
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
    catalog_path = args.catalog_path or os.getenv("CATALOG_PATH")
    needs_catalog = any(entry.parquet_spec is None for entry in entries)
    if needs_catalog and not catalog_path:
        parser.error("Set --catalog-path for datasets without parquet coverage (e.g., market data)")

    dataset_configs = tuple(entry.dataset for entry in entries)
    manager_config = CoverageManagerConfig(
        datasets=dataset_configs,
        lookback_days=max(1, args.lookback_days),
    )
    manager = CoverageManager(
        config=manager_config,
        sql_provider=SqlCoverageProvider(
            connection_string=db_url,
            dataset_overrides=dataset_overrides,
        ),
        catalog_provider=_build_catalog_provider(
            catalog_path=catalog_path,
            parquet_specs=parquet_specs,
        ),
        schema_auditor=SchemaAuditor(db_url=db_url),
    )
    classifications = manager.restore_all()
    if args.emit_json:
        payload = [
            {
                "dataset_id": item.spec.dataset_id,
                "schema": item.spec.schema,
                "instrument_id": item.spec.instrument_id,
                "bucket_start": item.spec.bucket_start.isoformat(),
                "status": item.status.name,
            }
            for item in classifications
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _log_classification_summary(classifications)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
