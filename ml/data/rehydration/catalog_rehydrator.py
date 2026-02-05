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
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Final

import pandas as pd
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import OrderBookDelta
from nautilus_trader.model.data import OrderBookDepth10
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.data import capsule_to_list
from nautilus_trader.model.identifiers import InstrumentId

from ml.common.event_emitter import emit_dataset_event_and_watermark
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.config.market_data import MarketDataTableConfig
from ml.data.coverage.manager import BucketSpec
from ml.registry.dataclasses import DatasetType
from ml.schema import DATASET_TYPE_IDENTIFIER_DEFAULTS
from ml.schema import DEFAULT_BAR_IDENTIFIER_TEMPLATE
from ml.schema import dataset_type_to_dataclass
from ml.schema import map_schema_to_dataset_type
from ml.schema import validate_dataset_type_templates
from ml.schema import validate_identifier_template
from ml.schema import validate_schema_identifier_templates
from ml.stores.providers import SqlCoverageProvider
from ml.stores.providers import SqlMarketDataWriter
from ml.stores.providers import resolve_catalog_identifier
from nautilus_trader.core.nautilus_pyo3 import DataBackendSession
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol


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
        Template used to resolve catalog identifiers (defaults to the bar registry template).
    schema_identifier_templates :
        Optional per-schema identifier templates (case-insensitive keys).
    dataset_type_identifier_templates :
        Per-dataset-type identifier templates. Defaults align with Nautilus catalog
        conventions (bars use bar type; TBBO/Trades/MBP use instrument_id).
    uri_safe_identifiers :
        Whether to normalize resolved identifiers using ``urisafe_identifier``.
    table_name :
        SQL table to populate (defaults to ``market_data``).
    table_config :
        Optional table routing config to resolve per-class table names.
    rescan_on_schedule :
        If True, perform rehydration on every scheduler loop; otherwise only at startup.
    exhaustive :
        When True, expand the inspection window to include the full catalog coverage for
        each instrument instead of limiting to ``lookback_days``.
    stream_chunk_size :
        Optional chunk size for streaming tick datasets using the Rust backend. When set,
        tick rehydration uses DataBackendSession(chunk_size=...) to limit memory usage.

    """

    enabled: bool = False
    lookback_days: int = 5
    batch_size: int = 1_000
    identifier_template: str = DEFAULT_BAR_IDENTIFIER_TEMPLATE
    schema_identifier_templates: Mapping[str, str] = field(default_factory=dict)
    dataset_type_identifier_templates: Mapping[DatasetType, str] = field(
        default_factory=lambda: DATASET_TYPE_IDENTIFIER_DEFAULTS.copy(),
    )
    uri_safe_identifiers: bool = True
    table_name: str = "market_data"
    table_config: MarketDataTableConfig | None = None
    rescan_on_schedule: bool = False
    exhaustive: bool = False
    stream_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            msg = "lookback_days must be >= 1"
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if self.stream_chunk_size is not None and self.stream_chunk_size <= 0:
            msg = "stream_chunk_size must be positive when set"
            raise ValueError(msg)
        validate_identifier_template(self.identifier_template, label="identifier_template")
        validate_schema_identifier_templates(self.schema_identifier_templates)
        validate_dataset_type_templates(self.dataset_type_identifier_templates)
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
        registry: RegistryProtocol | None = None,
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
        registry :
            Optional DataRegistry for emitting events and updating watermarks.

        """
        self._catalog = catalog
        self._config = config
        base_table_config = config.table_config or MarketDataTableConfig.from_env(
            legacy_table=config.table_name,
        )
        if base_table_config.write_batch_size != config.batch_size:
            base_table_config = replace(
                base_table_config,
                write_batch_size=config.batch_size,
            )

        self._writer = writer or SqlMarketDataWriter(
            connection_string=db_connection,
            table_name=config.table_name,
            table_config=(
                base_table_config
            ),
        )
        self._coverage = coverage_provider or SqlCoverageProvider(
            connection_string=db_connection,
            table_name=config.table_name,
            table_config=(
                base_table_config
            ),
        )
        self._schema_templates = validate_schema_identifier_templates(
            config.schema_identifier_templates,
        )
        self._dataset_templates = validate_dataset_type_templates(
            config.dataset_type_identifier_templates,
        )
        self._registry = registry

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

        dataset_type = map_schema_to_dataset_type(schema)
        data_class = dataset_type_to_dataclass(dataset_type)

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
            identifier = self._resolve_identifier(
                schema=schema,
                instrument_id=instrument_normalized,
                dataset_type=dataset_type,
            )
            effective_start_ns, effective_end_ns = self._resolve_effective_window(
                identifier=identifier,
                data_class=data_class,
                start_ns=start_ns,
                end_ns=end_ns,
                target_buckets=bucket_map.get(instrument_normalized) if bucket_map else None,
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
            if dataset_type in (DatasetType.QUOTES, DatasetType.TRADES):
                stream_rows, ts_min_ns, ts_max_ns = self._rehydrate_bucket_streaming(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_id,
                    identifier=identifier,
                    dataset_type=dataset_type,
                    instrument=instrument_obj,
                    bucket_start_ns=bucket_start_ns,
                    bucket_end_ns=bucket_end_ns,
                )
                if stream_rows <= 0:
                    logger.debug(
                        "catalog_rehydrate.bucket_empty",
                        extra={"instrument_id": instrument_id, "bucket": bucket},
                    )
                    continue
                buckets_restored += 1
                rows_written += stream_rows
                _rehydrate_rows_total.labels(instrument=instrument_id).inc(stream_rows)
                logger.info(
                    "catalog_rehydrate.bucket_restored",
                    extra={
                        "instrument_id": instrument_id,
                        "dataset_id": dataset_id,
                        "schema": schema,
                        "bucket": bucket,
                        "rows": stream_rows,
                    },
                )
                if ts_min_ns is not None and ts_max_ns is not None:
                    self._emit_registry_event_stats(
                        dataset_id=dataset_id,
                        schema=schema,
                        instrument_id=instrument_id,
                        identifier=identifier,
                        dataset_type=dataset_type,
                        bucket=bucket,
                        ts_min_ns=ts_min_ns,
                        ts_max_ns=ts_max_ns,
                        count=stream_rows,
                    )
                continue

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
                self._emit_registry_event(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_id,
                    identifier=identifier,
                    dataset_type=dataset_type,
                    bucket=bucket,
                    frame=frame,
                )

        return buckets_restored, rows_written, len(catalog_buckets)

    def _emit_registry_event(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        identifier: str,
        dataset_type: DatasetType,
        bucket: int,
        frame: pd.DataFrame,
    ) -> None:
        if self._registry is None or frame.empty or "ts_event" not in frame.columns:
            return
        ts_series = frame["ts_event"]
        if pd.api.types.is_datetime64_any_dtype(ts_series):
            ts_min = ts_series.min()
            ts_max = ts_series.max()
            if ts_min is None or ts_max is None:
                return
            ts_min_ns = int(ts_min.value)
            ts_max_ns = int(ts_max.value)
        else:
            numeric = pd.to_numeric(ts_series, errors="coerce").dropna()
            if numeric.empty:
                return
            ts_min_ns = int(numeric.min())
            ts_max_ns = int(numeric.max())
        self._emit_registry_event_stats(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            identifier=identifier,
            dataset_type=dataset_type,
            bucket=bucket,
            ts_min_ns=ts_min_ns,
            ts_max_ns=ts_max_ns,
            count=len(frame.index),
        )

    def _emit_registry_event_stats(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        identifier: str,
        dataset_type: DatasetType,
        bucket: int,
        ts_min_ns: int,
        ts_max_ns: int,
        count: int,
    ) -> None:
        if self._registry is None:
            return
        try:
            emit_dataset_event_and_watermark(
                self._registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=Stage.DATA_INGESTED,
                source=Source.BACKFILL,
                run_id="catalog_rehydrate",
                ts_min=ts_min_ns,
                ts_max=ts_max_ns,
                count=count,
                status=EventStatus.SUCCESS,
                dataset_type=dataset_type.value,
                component=self.__class__.__name__,
                metadata={
                    "schema": schema,
                    "bucket": int(bucket),
                    "identifier": identifier,
                },
            )
        except Exception:
            logger.warning(
                "catalog_rehydrate.event_emit_failed",
                exc_info=True,
                extra={
                    "dataset_id": dataset_id,
                    "schema": schema,
                    "instrument_id": instrument_id,
                    "bucket": bucket,
                },
            )

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
        data_class = _data_class_for_dataset(dataset_type)
        objects = self._catalog.query(
            data_cls=data_class,
            identifiers=[identifier],
            start=bucket_start_ns,
            end=bucket_end_ns,
        )
        if not objects:
            if dataset_type in (
                DatasetType.QUOTES,
                DatasetType.TRADES,
                DatasetType.TBBO,
                DatasetType.MBP1,
                DatasetType.MBP10,
                DatasetType.MBO,
            ):
                return pd.DataFrame()
            try:
                objects = self._catalog.query(
                    data_cls=data_class,
                    identifiers=None,
                    start=bucket_start_ns,
                    end=bucket_end_ns,
                )
            except Exception:
                logger.debug(
                    "catalog_rehydrate.bucket_query_failed",
                    exc_info=True,
                    extra={
                        "identifier": identifier,
                        "instrument_id": instrument.value,
                        "bucket_start_ns": bucket_start_ns,
                    },
                )
                objects = []
        if not objects:
            logger.debug(
                "catalog_rehydrate.bucket_missing",
                extra={
                    "identifier": identifier,
                    "instrument_id": instrument.value,
                    "bucket_start_ns": bucket_start_ns,
                },
            )
            return pd.DataFrame()

        frame = pd.DataFrame.from_records(
            [data_class.to_dict(obj) for obj in objects],
        )
        return self._normalize_bucket_frame(
            dataset_type=dataset_type,
            instrument=instrument,
            dataset_id=dataset_id,
            frame=frame,
        )

    def _iter_bucket_frames(
        self,
        *,
        dataset_type: DatasetType,
        instrument: InstrumentId,
        identifier: str,
        dataset_id: str,
        bucket_start_ns: int,
        bucket_end_ns: int,
    ) -> Iterator[pd.DataFrame]:
        data_class = _data_class_for_dataset(dataset_type)
        session = (
            DataBackendSession(chunk_size=self._config.stream_chunk_size)
            if self._config.stream_chunk_size is not None
            else None
        )
        session = self._catalog.backend_session(
            data_cls=data_class,
            identifiers=[identifier],
            start=bucket_start_ns,
            end=bucket_end_ns,
            session=session,
        )
        result = session.to_query_result()
        for chunk in result:
            objects = capsule_to_list(chunk)
            if not objects:
                continue
            frame = pd.DataFrame.from_records(
                [data_class.to_dict(obj) for obj in objects],
            )
            if frame.empty:
                continue
            yield self._normalize_bucket_frame(
                dataset_type=dataset_type,
                instrument=instrument,
                dataset_id=dataset_id,
                frame=frame,
            )

    def _rehydrate_bucket_streaming(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        identifier: str,
        dataset_type: DatasetType,
        instrument: InstrumentId,
        bucket_start_ns: int,
        bucket_end_ns: int,
    ) -> tuple[int, int | None, int | None]:
        total_rows = 0
        ts_min_ns: int | None = None
        ts_max_ns: int | None = None
        frames = self._iter_bucket_frames(
            dataset_type=dataset_type,
            instrument=instrument,
            identifier=identifier,
            dataset_id=dataset_id,
            bucket_start_ns=bucket_start_ns,
            bucket_end_ns=bucket_end_ns,
        )
        for frame in frames:
            if frame.empty:
                continue
            write_count = self._writer.write(
                dataset_id=dataset_id,
                schema=schema,
                instrument_id=instrument_id,
                df=frame,
            )
            if write_count <= 0:
                continue
            total_rows += write_count
            _rehydrate_rows_total.labels(instrument=instrument_id).inc(write_count)
            frame_min = frame["ts_event"].min()
            frame_max = frame["ts_event"].max()
            if frame_min is not None:
                ts_min_ns = frame_min if ts_min_ns is None else min(ts_min_ns, int(frame_min))
            if frame_max is not None:
                ts_max_ns = frame_max if ts_max_ns is None else max(ts_max_ns, int(frame_max))
        return total_rows, ts_min_ns, ts_max_ns

    def _normalize_bucket_frame(
        self,
        *,
        dataset_type: DatasetType,
        instrument: InstrumentId,
        dataset_id: str,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        if "instrument_id" not in frame.columns:
            frame["instrument_id"] = instrument.value
        frame["source_dataset"] = dataset_id

        if dataset_type == DatasetType.BARS:
            for column in ("open", "high", "low", "close", "volume"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
        elif dataset_type in (DatasetType.TBBO, DatasetType.MBP1, DatasetType.QUOTES):
            frame = frame.rename(columns={"bid_price": "bid", "ask_price": "ask"})
            for column in ("bid", "ask", "bid_size", "ask_size"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
        elif dataset_type is DatasetType.MBP10:
            if "type" in frame.columns:
                frame = frame.drop(columns=["type"])
        elif dataset_type is DatasetType.MBO:
            if "order" in frame.columns and "order_payload" not in frame.columns:
                frame = frame.rename(columns={"order": "order_payload"})
            if "type" in frame.columns:
                frame = frame.drop(columns=["type"])
        elif dataset_type == DatasetType.TRADES:
            frame = frame.rename(columns={"price": "last", "size": "volume"})
            if "volume" in frame.columns:
                frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
            frame["trade_count"] = 1

        for column in ("ts_event", "ts_init"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                    "int64",
                    errors="ignore",
                )
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

    def _resolve_identifier(
        self,
        *,
        schema: str,
        instrument_id: str,
        dataset_type: DatasetType | None = None,
    ) -> str:
        return resolve_catalog_identifier(
            schema=schema,
            instrument_id=instrument_id,
            identifier_template=self._config.identifier_template,
            schema_templates=self._schema_templates,
            dataset_type=dataset_type or map_schema_to_dataset_type(schema),
            dataset_templates=self._dataset_templates,
            uri_safe=self._config.uri_safe_identifiers,
        )

    def _resolve_effective_window(
        self,
        *,
        identifier: str,
        data_class: type[Bar | QuoteTick | TradeTick],
        start_ns: int,
        end_ns: int,
        target_buckets: set[int] | None,
    ) -> tuple[int, int]:
        if target_buckets:
            bucket_start = min(target_buckets) * DAY_NS
            bucket_end = (max(target_buckets) + 1) * DAY_NS
            return bucket_start, bucket_end
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


def _data_class_for_dataset(
    dataset_type: DatasetType,
) -> type[Bar | QuoteTick | TradeTick | OrderBookDepth10 | OrderBookDelta]:
    return dataset_type_to_dataclass(dataset_type)
