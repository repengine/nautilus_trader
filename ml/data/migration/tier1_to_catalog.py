"""
Utilities to migrate tier1 parquet shards into a ParquetDataCatalog.

This module is intentionally isolated from CLI concerns; see
``ml/scripts/migrate_tier1_to_catalog.py`` for the thin entrypoint.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick

from ml.registry.dataclasses import DatasetType
from ml.stores.io_raw import ParquetCatalogRawWriter
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


logger = logging.getLogger(__name__)

DatasetKind = Literal["bars", "quotes", "trades"]


@dataclass(frozen=True)
class MigrationStats:
    """Aggregate migration counters."""

    bars_files: int = 0
    bars_rows: int = 0
    quotes_files: int = 0
    quotes_rows: int = 0
    trades_files: int = 0
    trades_rows: int = 0
    skipped: tuple[str, ...] = ()
    analyzed: tuple[str, ...] = ()

    def with_update(
        self,
        *,
        bars_files: int | None = None,
        bars_rows: int | None = None,
        quotes_files: int | None = None,
        quotes_rows: int | None = None,
        trades_files: int | None = None,
        trades_rows: int | None = None,
        skipped: list[str] | None = None,
        analyzed: list[str] | None = None,
    ) -> MigrationStats:
        """Return a new instance with updated counters."""
        return MigrationStats(
            bars_files=self.bars_files if bars_files is None else bars_files,
            bars_rows=self.bars_rows if bars_rows is None else bars_rows,
            quotes_files=self.quotes_files if quotes_files is None else quotes_files,
            quotes_rows=self.quotes_rows if quotes_rows is None else quotes_rows,
            trades_files=self.trades_files if trades_files is None else trades_files,
            trades_rows=self.trades_rows if trades_rows is None else trades_rows,
            skipped=tuple(self.skipped) if skipped is None else tuple(skipped),
            analyzed=tuple(self.analyzed) if analyzed is None else tuple(analyzed),
        )


def _catalog_identifier_for_bar(instrument_id: str) -> str:
    """Compose the catalog identifier for bars."""
    return f"{instrument_id}-1-MINUTE-LAST-EXTERNAL"


def _has_overlap(
    *,
    catalog: ParquetDataCatalog,
    data_cls: type[Bar | QuoteTick | TradeTick],
    identifier: str,
    start_ns: int,
    end_ns: int,
) -> bool:
    """Return True if the interval overlaps existing catalog intervals."""
    intervals = catalog.get_intervals(data_cls=data_cls, identifier=identifier) or []
    for existing_start, existing_end in intervals:
        if start_ns < int(existing_end) and end_ns > int(existing_start):
            return True
    return False


def _drop_overlaps(
    *,
    catalog: ParquetDataCatalog,
    data_cls: type[Bar | QuoteTick | TradeTick],
    identifier: str,
    ts_series: pd.Series,
) -> pd.Series:
    """
    Remove rows whose timestamps overlap existing catalog intervals.
    """
    intervals = catalog.get_intervals(data_cls=data_cls, identifier=identifier) or []
    if not intervals:
        return ts_series
    mask = pd.Series(True, index=ts_series.index)
    for start, end in intervals:
        mask &= ~ts_series.between(int(start), int(end), inclusive="both")
    return ts_series[mask]


def _to_ns(values: Sequence[object] | pd.Series | pd.Index) -> pd.Series:
    """Convert an iterable of timestamps to int nanoseconds."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    return pd.to_datetime(series).astype("int64")


def _stream_parquet_batches(path: Path, batch_size: int) -> Iterable[pd.DataFrame]:
    """Yield pandas DataFrames from a parquet file in batches."""
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def _normalize_instrument(
    symbol: str,
    existing: dict[str, str],
    default_venue: str,
) -> str | None:
    """Resolve a symbol to an instrument_id using the catalog mapping when available."""
    normalized = symbol.strip()
    if not normalized:
        return None
    if "." in normalized:
        return normalized
    mapped = existing.get(normalized)
    if mapped:
        return mapped
    return f"{normalized}.{default_venue}"


def _discover_existing_instruments(catalog_path: Path) -> dict[str, str]:
    """
    Build a best-effort mapping from base symbol -> instrument_id from the catalog layout.
    """
    mapping: dict[str, str] = {}
    bar_root = catalog_path / "data" / "bar"
    if not bar_root.exists():
        return mapping
    for entry in bar_root.iterdir():
        if not entry.is_dir():
            continue
        token = entry.name.split("-")[0]
        if "." not in token:
            continue
        base = token.split(".", maxsplit=1)[0]
        mapping.setdefault(base, token)
    return mapping


def _is_minute_bar_file(path: Path) -> bool:
    """Filter tier1 files to those that look like 1-minute bars."""
    name = path.name.lower()
    if "1d" in name or "hour" in name:
        return False
    return "ohlcv-1m" in name or name.endswith("_ohlcv.parquet")


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Merge overlapping/contiguous intervals to enable fast coverage checks."""
    if not intervals:
        return ()
    ordered = sorted((int(s), int(e)) for s, e in intervals)
    merged: list[tuple[int, int]] = []
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return tuple(merged)


def _is_interval_fully_covered(
    *,
    start_ns: int,
    end_ns: int,
    intervals: Sequence[tuple[int, int]],
) -> bool:
    """Return True if [start_ns, end_ns] is entirely contained within existing intervals."""
    merged = _merge_intervals(intervals)
    for existing_start, existing_end in merged:
        if start_ns >= existing_start and end_ns <= existing_end:
            return True
    return False


def _are_row_groups_covered(
    *,
    row_groups: Sequence[tuple[int, int]],
    intervals: Sequence[tuple[int, int]],
) -> bool:
    """Return True if every row-group interval is fully covered by catalog intervals."""
    if not row_groups:
        return False
    merged = _merge_intervals(intervals)
    for start_ns, end_ns in row_groups:
        covered = False
        for existing_start, existing_end in merged:
            if start_ns >= existing_start and end_ns <= existing_end:
                covered = True
                break
        if not covered:
            return False
    return True


def _coerce_parquet_stat_to_ns(value: object, field_type: pa.DataType | None) -> int | None:
    """Coerce parquet statistics min/max to nanoseconds since epoch."""
    if value is None:
        return None
    try:
        if field_type is not None and pa.types.is_timestamp(field_type):
            unit = getattr(field_type, "unit", "ns")
            multiplier = {
                "s": 1_000_000_000,
                "ms": 1_000_000,
                "us": 1_000,
                "ns": 1,
            }.get(str(unit), 1)
            if isinstance(value, (int, float)):
                return int(value * multiplier)
        return int(pd.Timestamp(cast(Any, value)).value)
    except Exception:  # pragma: no cover - defensive
        return None


def _parquet_bounds(path: Path) -> tuple[int, int] | None:
    """
    Read parquet footer statistics to infer [min_ts, max_ts] without loading data.
    """
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception:  # pragma: no cover - defensive
        return None

    schema = parquet_file.schema_arrow
    field_map = {field.name: field.type for field in schema}
    min_ns: int | None = None
    max_ns: int | None = None
    for rg_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(rg_index)
        for col_index in range(row_group.num_columns):
            column_chunk = row_group.column(col_index)
            name = column_chunk.path_in_schema
            if name not in {"ts_event", "ts", "timestamp", "datetime", "time"}:
                continue
            stats = column_chunk.statistics
            if stats is None:
                continue
            field_type = field_map.get(name)
            rg_min = _coerce_parquet_stat_to_ns(getattr(stats, "min", None), field_type)
            rg_max = _coerce_parquet_stat_to_ns(getattr(stats, "max", None), field_type)
            if rg_min is None or rg_max is None:
                continue
            min_ns = rg_min if min_ns is None else min(min_ns, rg_min)
            max_ns = rg_max if max_ns is None else max(max_ns, rg_max)

    if min_ns is None or max_ns is None:
        return None
    return (min_ns, max_ns)


@dataclass(frozen=True)
class FileDisposition:
    """Decision for a tier1 file during migration planning."""

    path: Path
    dataset: DatasetKind
    instrument_id: str
    status: Literal["skip_full_overlap", "partial_overlap", "uncovered", "no_bounds"]
    bounds: tuple[int, int] | None


def _row_group_bounds(path: Path) -> list[tuple[int, int]]:
    """
    Return per-row-group [min, max] ts_event bounds using parquet statistics.
    """
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception:  # pragma: no cover - defensive
        return []
    schema = parquet_file.schema_arrow
    field_map = {field.name: field.type for field in schema}
    groups: list[tuple[int, int]] = []
    for rg_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(rg_index)
        rg_min: int | None = None
        rg_max: int | None = None
        for col_index in range(row_group.num_columns):
            column_chunk = row_group.column(col_index)
            name = column_chunk.path_in_schema
            if name not in {"ts_event", "ts", "timestamp", "datetime", "time"}:
                continue
            stats = column_chunk.statistics
            if stats is None:
                continue
            field_type = field_map.get(name)
            cmin = _coerce_parquet_stat_to_ns(getattr(stats, "min", None), field_type)
            cmax = _coerce_parquet_stat_to_ns(getattr(stats, "max", None), field_type)
            if cmin is None or cmax is None:
                continue
            rg_min = cmin if rg_min is None else min(rg_min, cmin)
            rg_max = cmax if rg_max is None else max(rg_max, cmax)
        if rg_min is not None and rg_max is not None:
            groups.append((rg_min, rg_max))
    return groups


def _save_plan(plan: Sequence[FileDisposition], path: Path) -> None:
    """
    Persist the planned dispositions as newline-delimited JSON.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for entry in plan:
            fp.write(
                json.dumps(
                    {
                        "path": str(entry.path),
                        "dataset": entry.dataset,
                        "instrument_id": entry.instrument_id,
                        "status": entry.status,
                        "bounds": entry.bounds,
                    },
                ),
            )
            fp.write("\n")


def _load_plan(path: Path) -> list[FileDisposition]:
    """
    Load a previously saved plan file produced by ``_save_plan``.
    """
    import json

    plan: list[FileDisposition] = []
    if not path.exists():
        return plan
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            plan.append(
                FileDisposition(
                    path=Path(payload.get("path", "")),
                    dataset=payload.get("dataset", "bars"),
                    instrument_id=payload.get("instrument_id", ""),
                    status=payload.get("status", "no_bounds"),
                    bounds=tuple(payload.get("bounds")) if payload.get("bounds") else None,
                ),
            )
    return plan


def migrate_tier1_to_catalog(
    *,
    tier1_root: Path,
    catalog_path: Path,
    datasets: set[DatasetKind],
    symbols: set[str] | None,
    batch_size: int = 20000,
    default_venue: str = "XNAS",
    analyze_only: bool = False,
    plan_output: Path | None = None,
    plan_input: Path | None = None,
    process_statuses: set[str] | None = None,
) -> MigrationStats:
    """
    Migrate tier1 parquet shards into a ParquetDataCatalog without overwriting overlaps.

    Args:
        tier1_root: Root directory containing per-symbol tier1 shards.
        catalog_path: Target ParquetDataCatalog path.
        datasets: Subset of datasets to migrate (bars, quotes, trades).
        symbols: Optional allowlist of symbols (case-insensitive, base symbols).
        batch_size: Rows per streamed batch when reading parquet.
        default_venue: Venue suffix to append when symbol lacks a venue.
        analyze_only: When True, do not write anything; only report dispositions.
        plan_output: Optional path to write planned dispositions as NDJSON.
        plan_input: Optional path to a saved plan; only process files present in the plan.
        process_statuses: Allowed statuses to process when using a plan (defaults to all except skip_full_overlap).

    Returns:
        MigrationStats summarising processed files/rows and skips.
    """
    catalog = ParquetDataCatalog(str(catalog_path))
    writer = ParquetCatalogRawWriter(catalog)
    existing_map = _discover_existing_instruments(catalog_path)
    allowed = {s.upper() for s in symbols} if symbols else None

    stats = MigrationStats()
    skipped: list[str] = []
    analyzed: list[str] = []
    planned: list[FileDisposition] = []

    plan_filter: dict[Path, FileDisposition] | None = None
    allowed_statuses = process_statuses or {"partial_overlap", "uncovered", "no_bounds"}
    if plan_input is not None:
        plan_entries = _load_plan(plan_input)
        plan_filter = {entry.path: entry for entry in plan_entries}

    for sym_dir in sorted(tier1_root.iterdir()):
        if not sym_dir.is_dir():
            continue
        symbol = sym_dir.name.upper()
        if allowed and symbol not in allowed:
            continue
        instrument_id = _normalize_instrument(symbol, existing_map, default_venue)
        if instrument_id is None:
            skipped.append(f"{symbol}:no_instrument")
            continue

        # Bars
        if "bars" in datasets:
            identifier = _catalog_identifier_for_bar(instrument_id)
            intervals = catalog.get_intervals(data_cls=Bar, identifier=identifier) or []
            merged_intervals = _merge_intervals(intervals)
            for path in sym_dir.rglob("*.parquet"):
                if not _is_minute_bar_file(path):
                    continue
                if plan_filter is not None and path not in plan_filter:
                    continue
                bounds = _parquet_bounds(path)
                rg_bounds = _row_group_bounds(path)
                status_bars: Literal["skip_full_overlap", "partial_overlap", "uncovered", "no_bounds"]
                if bounds is None:
                    status_bars = "no_bounds"
                elif not merged_intervals:
                    status_bars = "uncovered"
                elif _is_interval_fully_covered(start_ns=bounds[0], end_ns=bounds[1], intervals=merged_intervals) or _are_row_groups_covered(row_groups=rg_bounds, intervals=merged_intervals):
                    status_bars = "skip_full_overlap"
                else:
                    status_bars = "partial_overlap"
                disposition = FileDisposition(
                    path=path,
                    dataset="bars",
                    instrument_id=instrument_id,
                    status=status_bars,
                    bounds=bounds,
                )
                planned.append(disposition)
                if status_bars == "skip_full_overlap":
                    analyzed.append(f"bars_meta_overlap:{path}")
                    continue
                if plan_filter is not None and disposition.status not in allowed_statuses:
                    continue
                if analyze_only:
                    continue
                logger.info(
                    "processing_file",
                    extra={
                        "dataset": "bars",
                        "path": str(path),
                        "instrument_id": instrument_id,
                        "status": status_bars,
                    },
                )
                try:
                    for batch in _stream_parquet_batches(path, batch_size):
                        if batch.empty:
                            continue
                        ts_source = (
                            batch.index if isinstance(batch.index, pd.DatetimeIndex) else batch.get("ts_event")
                        )
                        if ts_source is None:
                            skipped.append(f"bars_no_ts:{path}")
                            continue
                        ts_ns = _to_ns(ts_source)
                        required = {"open", "high", "low", "close", "volume"}
                        if not required.issubset(set(batch.columns)):
                            skipped.append(f"bars_cols:{path}")
                            continue
                        start_ns = int(ts_ns.min())
                        end_ns = int(ts_ns.max())
                        if _has_overlap(
                            catalog=catalog,
                            data_cls=Bar,
                            identifier=identifier,
                            start_ns=start_ns,
                            end_ns=end_ns,
                        ):
                            ts_ns = _drop_overlaps(
                                catalog=catalog,
                                data_cls=Bar,
                                identifier=identifier,
                                ts_series=ts_ns,
                            )
                            if ts_ns.empty:
                                skipped.append(f"bars_overlap:{path}")
                                continue
                        out = pd.DataFrame(
                            {
                                "instrument_id": instrument_id,
                                "ts_event": ts_ns,
                                "ts_init": ts_ns,
                                "open": batch["open"],
                                "high": batch["high"],
                                "low": batch["low"],
                                "close": batch["close"],
                                "volume": batch["volume"],
                            },
                        )
                        out = out.sort_values("ts_event").reset_index(drop=True)
                        written = writer.write(dataset_type=DatasetType.BARS, data=out)
                        stats = stats.with_update(
                            bars_rows=stats.bars_rows + written,
                        )
                    stats = stats.with_update(bars_files=stats.bars_files + 1)
                except Exception:  # pragma: no cover - migration safety
                    logger.warning("bars_migration_failed path=%s", path, exc_info=True)
                    skipped.append(f"bars_exc:{path}")

        # Quotes (TBBO)
        if "quotes" in datasets:
            intervals_q = catalog.get_intervals(data_cls=QuoteTick, identifier=instrument_id) or []
            merged_intervals_q = _merge_intervals(intervals_q)
            for path in sym_dir.rglob("*bbo*.parquet"):
                if plan_filter is not None and path not in plan_filter:
                    continue
                bounds = _parquet_bounds(path)
                rg_bounds = _row_group_bounds(path)
                status_quotes: Literal["skip_full_overlap", "partial_overlap", "uncovered", "no_bounds"]
                if bounds is None:
                    status_quotes = "no_bounds"
                elif not merged_intervals_q:
                    status_quotes = "uncovered"
                elif _is_interval_fully_covered(
                    start_ns=bounds[0],
                    end_ns=bounds[1],
                    intervals=merged_intervals_q,
                ):
                    status_quotes = "skip_full_overlap"
                elif _are_row_groups_covered(row_groups=rg_bounds, intervals=merged_intervals_q):
                    status_quotes = "skip_full_overlap"
                else:
                    status_quotes = "partial_overlap"
                disposition = FileDisposition(
                    path=path,
                    dataset="quotes",
                    instrument_id=instrument_id,
                    status=status_quotes,
                    bounds=bounds,
                )
                planned.append(disposition)
                if status_quotes == "skip_full_overlap":
                    analyzed.append(f"quotes_meta_overlap:{path}")
                    continue
                if plan_filter is not None and disposition.status not in allowed_statuses:
                    continue
                if analyze_only:
                    continue
                logger.info(
                    "processing_file",
                    extra={
                        "dataset": "quotes",
                        "path": str(path),
                        "instrument_id": instrument_id,
                        "status": status_quotes,
                    },
                )
                try:
                    for batch in _stream_parquet_batches(path, batch_size):
                        if batch.empty:
                            continue
                        ts_col = batch.get("ts_event")
                        if ts_col is None:
                            skipped.append(f"quotes_no_ts:{path}")
                            continue
                        ts_ns = _to_ns(ts_col)
                        if not {"bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}.issubset(set(batch.columns)):
                            skipped.append(f"quotes_cols:{path}")
                            continue
                        start_ns = int(ts_ns.min())
                        end_ns = int(ts_ns.max())
                        if _has_overlap(
                            catalog=catalog,
                            data_cls=QuoteTick,
                            identifier=instrument_id,
                            start_ns=start_ns,
                            end_ns=end_ns,
                        ):
                            ts_ns = _drop_overlaps(
                                catalog=catalog,
                                data_cls=QuoteTick,
                                identifier=instrument_id,
                                ts_series=ts_ns,
                            )
                            if ts_ns.empty:
                                skipped.append(f"quotes_overlap:{path}")
                                continue
                        out = pd.DataFrame(
                            {
                                "instrument_id": instrument_id,
                                "ts_event": ts_ns,
                                "ts_init": ts_ns,
                                "bid": batch["bid_px_00"],
                                "ask": batch["ask_px_00"],
                                "bid_size": batch["bid_sz_00"],
                                "ask_size": batch["ask_sz_00"],
                            },
                        )
                        out = out.sort_values("ts_event").reset_index(drop=True)
                        written = writer.write(dataset_type=DatasetType.QUOTES, data=out)
                        stats = stats.with_update(
                            quotes_rows=stats.quotes_rows + written,
                        )
                    stats = stats.with_update(quotes_files=stats.quotes_files + 1)
                except Exception:  # pragma: no cover - migration safety
                    logger.warning("quotes_migration_failed path=%s", path, exc_info=True)
                    skipped.append(f"quotes_exc:{path}")

        # Trades
        if "trades" in datasets:
            intervals_t = catalog.get_intervals(data_cls=TradeTick, identifier=instrument_id) or []
            merged_intervals_t = _merge_intervals(intervals_t)
            for path in sym_dir.rglob("*trades*.parquet"):
                if plan_filter is not None and path not in plan_filter:
                    continue
                bounds = _parquet_bounds(path)
                rg_bounds = _row_group_bounds(path)
                status_trades: Literal["skip_full_overlap", "partial_overlap", "uncovered", "no_bounds"]
                if bounds is None:
                    status_trades = "no_bounds"
                elif not merged_intervals_t:
                    status_trades = "uncovered"
                elif _is_interval_fully_covered(
                    start_ns=bounds[0],
                    end_ns=bounds[1],
                    intervals=merged_intervals_t,
                ):
                    status_trades = "skip_full_overlap"
                elif _are_row_groups_covered(row_groups=rg_bounds, intervals=merged_intervals_t):
                    status_trades = "skip_full_overlap"
                else:
                    status_trades = "partial_overlap"
                disposition = FileDisposition(
                    path=path,
                    dataset="trades",
                    instrument_id=instrument_id,
                    status=status_trades,
                    bounds=bounds,
                )
                planned.append(disposition)
                if status_trades == "skip_full_overlap":
                    analyzed.append(f"trades_meta_overlap:{path}")
                    continue
                if plan_filter is not None and disposition.status not in allowed_statuses:
                    continue
                if analyze_only:
                    continue
                logger.info(
                    "processing_file",
                    extra={
                        "dataset": "trades",
                        "path": str(path),
                        "instrument_id": instrument_id,
                        "status": status_trades,
                    },
                )
                try:
                    for batch in _stream_parquet_batches(path, batch_size):
                        if batch.empty:
                            continue
                        ts_col = batch.get("ts_event")
                        if ts_col is None:
                            skipped.append(f"trades_no_ts:{path}")
                            continue
                        ts_ns = _to_ns(ts_col)
                        if not {"price", "size"}.issubset(set(batch.columns)):
                            skipped.append(f"trades_cols:{path}")
                            continue
                        start_ns = int(ts_ns.min())
                        end_ns = int(ts_ns.max())
                        if _has_overlap(
                            catalog=catalog,
                            data_cls=TradeTick,
                            identifier=instrument_id,
                            start_ns=start_ns,
                            end_ns=end_ns,
                        ):
                            ts_ns = _drop_overlaps(
                                catalog=catalog,
                                data_cls=TradeTick,
                                identifier=instrument_id,
                                ts_series=ts_ns,
                            )
                            if ts_ns.empty:
                                skipped.append(f"trades_overlap:{path}")
                                continue
                        side_map = {"B": "BUYER", "A": "SELLER"}
                        side_series = batch.get("side")
                        if side_series is not None:
                            aggressor = side_series.map(
                                lambda s: side_map.get(str(s).upper(), "BUYER"),
                            )
                        else:
                            aggressor = pd.Series(["BUYER"] * len(batch))
                        out = pd.DataFrame(
                            {
                                "instrument_id": instrument_id,
                                "ts_event": ts_ns,
                                "ts_init": ts_ns,
                                "price": batch["price"],
                                "size": batch["size"],
                                "aggressor_side": aggressor,
                            },
                        )
                        out = out.sort_values("ts_event").drop_duplicates(subset="ts_event").reset_index(drop=True)
                        written = writer.write(dataset_type=DatasetType.TRADES, data=out)
                        stats = stats.with_update(
                            trades_rows=stats.trades_rows + written,
                        )
                    stats = stats.with_update(trades_files=stats.trades_files + 1)
                except Exception:  # pragma: no cover - migration safety
                    logger.warning("trades_migration_failed path=%s", path, exc_info=True)
                    skipped.append(f"trades_exc:{path}")

    if analyze_only:
        for disposition in planned:
            logger.info(
                "planned_file",
                extra={
                    "path": str(disposition.path),
                    "dataset": disposition.dataset,
                    "instrument_id": disposition.instrument_id,
                    "status": disposition.status,
                    "bounds": disposition.bounds,
                },
            )
        if plan_output is not None:
            _save_plan(planned, plan_output)
    return stats.with_update(skipped=skipped, analyzed=analyzed)


__all__ = ["MigrationStats", "migrate_tier1_to_catalog"]
