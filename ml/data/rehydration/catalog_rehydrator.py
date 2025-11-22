"""
Catalog-backed database rehydration service.

This module inspects a Nautilus ``ParquetDataCatalog`` and replays historical data
into the canonical SQL ``market_data`` table before running external ingestion.
It adheres to the AGENTS.md guardrails by using protocol-first patterns, structured
logging, centralized metrics, and progressive fallbacks.

"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Final, cast

import pandas as pd

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.data.coverage.manager import BucketSpec
from ml.data.ingest.orchestrator import _schema_to_dataset_type as _map_schema_to_dataset_type
from ml.registry.dataclasses import DatasetType
from ml.stores.providers import SqlCoverageProvider
from ml.stores.providers import SqlMarketDataWriter
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


logger = logging.getLogger(__name__)

DAY_NS: Final[int] = 86_400_000_000_000

_rehydrate_rows_total = get_counter(
    "nautilus_ml_catalog_rehydrate_rows_total",
    "Rows restored from Parquet catalog into SQL market_data.",
    ["instrument"],
)
_rehydrate_failures_total = get_counter(
    "nautilus_ml_catalog_rehydrate_failures_total",
    "Failures encountered during catalog rehydration.",
    ["instrument", "reason"],
)
_rehydrate_latency_seconds = get_histogram(
    "nautilus_ml_catalog_rehydrate_latency_seconds",
    "Latency for catalog rehydration operations.",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


@dataclass(frozen=True, slots=True)
class CatalogRehydrationConfig:
    """
    Configuration for catalog-driven database rehydration.

    Attributes
    ----------
    enabled :
        Whether rehydration is active.
    lookback_days :
        Rolling window (in days) considered for gap detection.
    batch_size :
        Maximum rows written per SQL batch (passed through to writer).
    identifier_template :
        Template used to resolve catalog identifiers (defaults to bar template).
    table_name :
        SQL table to populate (defaults to ``market_data``).
    rescan_on_schedule :
        If True, perform rehydration on every scheduler loop; otherwise only at startup.
    exhaustive :
        When True, expand the inspection window to include the full catalog coverage for
        each instrument instead of limiting to ``lookback_days``.

    """

    enabled: bool = False
    lookback_days: int = 5
    batch_size: int = 1_000
    identifier_template: str = "{instrument_id}-1-MINUTE-LAST-EXTERNAL"
    table_name: str = "market_data"
    rescan_on_schedule: bool = False
    exhaustive: bool = False

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            msg = "lookback_days must be >= 1"
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if not self.identifier_template:
            msg = "identifier_template must be provided"
            raise ValueError(msg)
        if not self.table_name:
            msg = "table_name must be provided"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CatalogRehydrationResult:
    """
    Result summary for a rehydration run.
    """

    instruments_processed: int
    buckets_considered: int
    buckets_restored: int
    rows_written: int
    failures: dict[str, str]


class ParquetCatalogRehydrator:
    """
    Service that rehydrates SQL market data from an on-disk Parquet catalog.
    """

    def __init__(
        self,
        *,
        catalog: ParquetDataCatalog,
        db_connection: str,
        config: CatalogRehydrationConfig,
        writer: SqlMarketDataWriter | None = None,
        coverage_provider: SqlCoverageProvider | None = None,
    ) -> None:
        """
        Initialize the rehydrator.

        Parameters
        ----------
        catalog :
            Parquet catalog containing historical data.
        db_connection :
            SQL connection string for the canonical market data table.
        config :
            Rehydration configuration.
        writer :
            Optional writer override (primarily for testing).
        coverage_provider :
            Optional SQL coverage provider override (primarily for testing).

        """
        self._catalog = catalog
        self._config = config
        self._writer = writer or SqlMarketDataWriter(
            connection_string=db_connection,
            table_name=config.table_name,
        )
        self._coverage = coverage_provider or SqlCoverageProvider(
            connection_string=db_connection,
            table_name=config.table_name,
        )

    def rehydrate_missing_data(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_ids: list[str],
        reference_time: datetime | None = None,
        buckets: list[BucketSpec] | None = None,
    ) -> CatalogRehydrationResult:
        """
        Replay catalog data for instruments with missing SQL coverage.
        """
        if not self._config.enabled:
            logger.debug("Catalog rehydration disabled; skipping")
            return CatalogRehydrationResult(
                instruments_processed=0,
                buckets_considered=0,
                buckets_restored=0,
                rows_written=0,
                failures={},
            )

        if not instrument_ids:
            logger.debug("No instruments provided for catalog rehydration")
            return CatalogRehydrationResult(
                instruments_processed=0,
                buckets_considered=0,
                buckets_restored=0,
                rows_written=0,
                failures={},
            )

        ref_time = reference_time or datetime.now(tz=UTC)
        start_time = ref_time - timedelta(days=self._config.lookback_days)
        start_ns = _datetime_to_ns(start_time)
        end_ns = _datetime_to_ns(ref_time)

        dataset_type = _map_schema_to_dataset_type(schema)
        data_class = _data_class_for_dataset(dataset_type)

        total_buckets_considered = 0
        total_buckets_restored = 0
        total_rows_written = 0
        failures: dict[str, str] = {}

        bucket_map: dict[str, set[int]] | None = None
        if buckets is not None:
            bucket_map = {}
            for bucket in buckets:
                key = bucket.instrument_id.strip()
                if not key:
                    continue
                bucket_map.setdefault(key, set()).add(bucket.bucket_index)

        for instrument in instrument_ids:
            instrument_normalized = instrument.strip()
            if not instrument_normalized:
                continue
            start_perf = time.perf_counter()
            identifier = self._resolve_identifier(instrument_normalized, data_class=data_class)
            effective_start_ns, effective_end_ns = self._resolve_effective_window(
                identifier=identifier,
                data_class=data_class,
                start_ns=start_ns,
                end_ns=end_ns,
            )
            try:
                restored_buckets, restored_rows, considered = self._rehydrate_instrument(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_normalized,
                    identifier=identifier,
                    dataset_type=dataset_type,
                    data_class=data_class,
                    start_ns=effective_start_ns,
                    end_ns=effective_end_ns,
                    target_buckets=bucket_map.get(instrument_normalized) if bucket_map else None,
                )
            except Exception as exc:
                failures[instrument_normalized] = str(exc)
                _rehydrate_failures_total.labels(
                    instrument=instrument_normalized,
                    reason=exc.__class__.__name__,
                ).inc()
                logger.warning(
                    "catalog_rehydrate.instrument_failed",
                    exc_info=True,
                    extra={
                        "instrument_id": instrument_normalized,
                        "dataset_id": dataset_id,
                        "schema": schema,
                    },
                )
                continue
            finally:
                duration = time.perf_counter() - start_perf
                _rehydrate_latency_seconds.observe(duration)

            if self._config.exhaustive and effective_start_ns < start_ns:
                logger.debug(
                    "catalog_rehydrate.exhaustive_window_expanded",
                    extra={
                        "instrument_id": instrument_normalized,
                        "effective_start_ns": effective_start_ns,
                        "effective_end_ns": effective_end_ns,
                        "baseline_start_ns": start_ns,
                        "baseline_end_ns": end_ns,
                    },
                )

            total_buckets_considered += considered
            total_buckets_restored += restored_buckets
            total_rows_written += restored_rows

        return CatalogRehydrationResult(
            instruments_processed=len(tuple(filter(str.strip, instrument_ids))),
            buckets_considered=total_buckets_considered,
            buckets_restored=total_buckets_restored,
            rows_written=total_rows_written,
            failures=failures,
        )

    def _rehydrate_instrument(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        identifier: str,
        dataset_type: DatasetType,
        data_class: type[Bar | QuoteTick | TradeTick],
        start_ns: int,
        end_ns: int,
        target_buckets: set[int] | None = None,
    ) -> tuple[int, int, int]:
        catalog_buckets = self._catalog_bucket_set(
            data_class=data_class,
            identifier=identifier,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        sql_buckets = self._coverage.read_bucket_coverage(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            start_ns=start_ns,
            end_ns=end_ns,
        )

        missing_buckets_set = catalog_buckets - sql_buckets
        if target_buckets is not None:
            missing_buckets_set &= target_buckets

        if not missing_buckets_set:
            logger.debug(
                "catalog_rehydrate.instrument_up_to_date",
                extra={"instrument_id": instrument_id, "dataset_id": dataset_id},
            )
            return 0, 0, 0

        missing_buckets = sorted(missing_buckets_set)
        buckets_restored = 0
        rows_written = 0

        logger.info(
            "catalog_rehydrate.instrument_start",
            extra={
                "instrument_id": instrument_id,
                "dataset_id": dataset_id,
                "schema": schema,
                "missing_buckets": len(missing_buckets),
            },
        )

        instrument_obj = InstrumentId.from_str(instrument_id)
        for bucket in missing_buckets:
            bucket_start_ns = bucket * DAY_NS
            bucket_end_ns = (bucket + 1) * DAY_NS
            frame = self._load_bucket_frame(
                dataset_type=dataset_type,
                instrument=instrument_obj,
                identifier=identifier,
                dataset_id=dataset_id,
                bucket_start_ns=bucket_start_ns,
                bucket_end_ns=bucket_end_ns,
            )
            if frame.empty:
                logger.debug(
                    "catalog_rehydrate.bucket_empty",
                    extra={"instrument_id": instrument_id, "bucket": bucket},
                )
                continue
            write_count = self._writer.write(
                dataset_id=dataset_id,
                schema=schema,
                instrument_id=instrument_id,
                df=frame,
            )
            if write_count > 0:
                buckets_restored += 1
                rows_written += write_count
                _rehydrate_rows_total.labels(instrument=instrument_id).inc(write_count)
                logger.info(
                    "catalog_rehydrate.bucket_restored",
                    extra={
                        "instrument_id": instrument_id,
                        "dataset_id": dataset_id,
                        "schema": schema,
                        "bucket": bucket,
                        "rows": write_count,
                    },
                )

        return buckets_restored, rows_written, len(catalog_buckets)

    def _load_bucket_frame(
        self,
        *,
        dataset_type: DatasetType,
        instrument: InstrumentId,
        dataset_id: str,
        identifier: str,
        bucket_start_ns: int,
        bucket_end_ns: int,
    ) -> pd.DataFrame:
        import os

        data_class = _data_class_for_dataset(dataset_type)
        # Access internal catalog path construction
        # Path structure: {catalog_path}/{Class}/{identifier}/*.parquet
        # We can use the catalog's fs to glob

        # Construct path manually to avoid protected member access if possible,
        # but we need to know exactly how catalog constructs it.
        # Assuming standard Nautilus structure:
        cls_name = data_class.__name__
        base_dir = os.path.join(self._catalog.path, cls_name)
        directory = os.path.join(base_dir, identifier)

        candidate_dirs = []
        if self._catalog.fs.exists(directory):
            candidate_dirs.append(directory)
        else:
            # If exact match doesn't exist, try globbing for directories starting with identifier
            # This handles Bar types where directory is {instrument}-{spec}
            # Only do this if identifier looks like an instrument ID (not already a specific spec)
            # and we are looking for Bars (or potentially other types if they use suffixes)
            if dataset_type == DatasetType.BARS:
                # Glob for directories starting with identifier
                # We use the filesystem glob
                pattern = os.path.join(base_dir, f"{identifier}*")
                matches = self._catalog.fs.glob(pattern)
                # Filter for directories
                candidate_dirs.extend([d for d in matches if self._catalog.fs.isdir(d)])

        if not candidate_dirs:
            return pd.DataFrame()

        # Glob all parquet files from all candidate directories
        files = []
        for d in candidate_dirs:
            files.extend(self._catalog.fs.glob(os.path.join(d, "*.parquet")))

        if not files:
            return pd.DataFrame()

        # Filter files by timestamp range
        # Filename format: {start}-{end}.parquet (int) or {start_iso}_{end_iso}.parquet (ISO)
        relevant_files = []
        for f in files:
            basename = os.path.basename(f)
            name_part = os.path.splitext(basename)[0]

            # Try integer timestamp format {start}-{end}
            try:
                parts = name_part.split("-")
                if len(parts) == 2:
                    f_start = int(parts[0])
                    f_end = int(parts[1])
                    if f_end > bucket_start_ns and f_start < bucket_end_ns:
                        relevant_files.append(f)
                    continue
            except ValueError:
                pass

            # Try ISO timestamp format {start_iso}_{end_iso}
            try:
                parts = name_part.split("_")
                if len(parts) == 2:
                    # Basic validation that it looks like ISO8601 (e.g. starts with year)
                    if parts[0][0].isdigit() and parts[1][0].isdigit():
                        # Parsing ISO strings to int ns is expensive and strict.
                        # Since we already filtered by directory (identifier) and Catalog intervals,
                        # and these files are typically partitioned by day or similar,
                        # we can optimistically include them if they are in the right directory.
                        # But to be safe, let's try to parse at least the year/month if possible,
                        # or rely on pd.read_parquet filtering later.
                        # For now, let's include it and let the dataframe time filter handle exact range.
                        relevant_files.append(f)
                        continue
            except Exception:
                pass

        if not relevant_files:
            return pd.DataFrame()

        # Read parquet files
        dfs = []
        for f in relevant_files:
            try:
                # We only need columns that map to SQL
                # But we don't know exactly which columns are in the file vs what we need.
                # We'll read all and rename/filter.
                df_partial = pd.read_parquet(f, filesystem=self._catalog.fs)
                dfs.append(df_partial)
            except Exception:
                logger.warning("Failed to read parquet file", extra={"path": f}, exc_info=True)
                continue

        if not dfs:
            return pd.DataFrame()

        frame = pd.concat(dfs, ignore_index=True)

        # Filter by exact time range
        if "ts_event" in frame.columns:
            frame = frame[
                (frame["ts_event"] >= bucket_start_ns) & (frame["ts_event"] < bucket_end_ns)
            ]
        elif "ts_init" in frame.columns:
            frame = frame[
                (frame["ts_init"] >= bucket_start_ns) & (frame["ts_init"] < bucket_end_ns)
            ]

        if frame.empty:
            return pd.DataFrame()

        # Normalize columns for SQL writer
        # SQL writer expects: ts_event, ts_init, open, high, low, close, volume, etc.
        # Parquet columns should match these mostly.

        # Handle specific renames if necessary
        # For Quotes: bid_price -> bid, ask_price -> ask
        if dataset_type is DatasetType.TBBO:
            rename_map = {
                "bid_price": "bid",
                "ask_price": "ask",
            }
            frame = frame.rename(columns=rename_map)

        # Ensure instrument_id is set (it might not be in the file if partitioned)
        # The writer adds it, but we can add it here to be safe or if writer expects it in df (it does not, it takes it as arg)

        frame["source_dataset"] = dataset_id
        return frame

    def _catalog_bucket_set(
        self,
        *,
        data_class: type[Bar | QuoteTick | TradeTick],
        identifier: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        intervals = self._catalog.get_intervals(data_cls=data_class, identifier=identifier)
        buckets: set[int] = set()
        for interval_start, interval_end in intervals:
            if interval_end <= start_ns or interval_start >= end_ns:
                continue
            start_clamped = max(interval_start, start_ns)
            end_clamped = min(interval_end, end_ns)
            if start_clamped >= end_clamped:
                continue
            bucket_start = start_clamped // DAY_NS
            bucket_end = (end_clamped - 1) // DAY_NS
            buckets.update(range(int(bucket_start), int(bucket_end) + 1))
        return buckets

    def _resolve_identifier(self, instrument_id: str, data_class: type | None = None) -> str:
        identifier = self._config.identifier_template.format(instrument_id=instrument_id)

        if data_class:
            import os

            cls_name = data_class.__name__
            base_dir = os.path.join(self._catalog.path, cls_name)
            exact_dir = os.path.join(base_dir, identifier)

            if not self._catalog.fs.exists(exact_dir):
                # Try globbing for directories starting with identifier
                pattern = os.path.join(base_dir, f"{identifier}*")
                matches = self._catalog.fs.glob(pattern)
                dirs = [d for d in matches if self._catalog.fs.isdir(d)]
                if dirs:
                    # Return the basename of the first match
                    return os.path.basename(dirs[0])

        return identifier

    def _resolve_effective_window(
        self,
        *,
        identifier: str,
        data_class: type[Bar | QuoteTick | TradeTick],
        start_ns: int,
        end_ns: int,
    ) -> tuple[int, int]:
        if not self._config.exhaustive:
            return start_ns, end_ns

        try:
            intervals = self._catalog.get_intervals(data_cls=data_class, identifier=identifier)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.debug(
                "catalog_rehydrate.interval_lookup_failed",
                exc_info=True,
                extra={"identifier": identifier, "reason": exc.__class__.__name__},
            )
            return start_ns, end_ns

        if not intervals:
            return start_ns, end_ns

        earliest = min(int(interval_start) for interval_start, _ in intervals)
        latest = max(int(interval_end) for _, interval_end in intervals)

        effective_start = min(start_ns, earliest)
        effective_end = max(end_ns, latest)

        return effective_start, effective_end


def _datetime_to_ns(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1_000_000_000)


def _data_class_for_dataset(dataset_type: DatasetType) -> type[Bar | QuoteTick | TradeTick]:
    if dataset_type is DatasetType.BARS:
        return cast(type[Bar | QuoteTick | TradeTick], Bar)
    if dataset_type is DatasetType.TBBO:
        return cast(type[Bar | QuoteTick | TradeTick], QuoteTick)
    if dataset_type is DatasetType.TRADES:
        return cast(type[Bar | QuoteTick | TradeTick], TradeTick)
    return cast(type[Bar | QuoteTick | TradeTick], Bar)
