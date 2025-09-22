"""
Efficient L2 data population helpers reused by CLI and task entry points.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from random import shuffle as _shuffle
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import polars as pl
import psutil
import pyarrow.parquet as pq

from ml.config.universes import TIER1_CORE
from ml.data.ingest.api import fetch_symbol_data
from ml.data.ingest.common import load_progress_json
from ml.data.ingest.common import save_progress_json


if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from ml.data.ingest.service import DatabentoIngestionService


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class L2PopulateConfig:
    """
    Configuration describing the L2 population workflow.
    """

    symbols: Sequence[str]
    data_dir: Path
    progress_file: Path
    resume: bool
    start_date: datetime
    end_date: datetime
    check_gaps: bool
    force: bool
    max_symbols: int | None
    symbol_offset: int
    shuffle: bool
    rate_limit: int
    dataset: str
    schema: str
    sleep_between_symbols: float


@dataclass(slots=True, frozen=True)
class L2PopulateResult:
    """
    Aggregated outcome for an L2 population run.
    """

    total_records: int
    total_size_mb: float
    symbols_processed: int


def _load_progress(path: Path) -> dict[str, list[str]]:
    data_any = load_progress_json(path)
    if isinstance(data_any, dict):
        out: dict[str, list[str]] = {}
        for key, value in data_any.items():
            if isinstance(value, list):
                out[str(key)] = [str(item) for item in value if isinstance(item, str)]
            else:
                out[str(key)] = []
        return out
    return {}


def _save_progress(path: Path, progress: dict[str, list[str]]) -> None:
    save_progress_json(path, progress)


def validate_data_integrity(file_path: Path, symbol: str, expected_date: datetime) -> bool:
    """
    Validate that a data file contains reasonable data for the given date.
    """
    if not file_path.exists():
        return False

    try:
        df = pl.read_parquet(file_path)

        if df.is_empty():
            LOGGER.warning("  %s: File exists but is empty", expected_date.date())
            return False

        min_ts = df["ts_event"].min()
        max_ts = df["ts_event"].max()
        if min_ts is None or max_ts is None:
            return False
        min_ts_i: int = cast(int, min_ts)
        max_ts_i: int = cast(int, max_ts)
        min_date = pd.to_datetime(min_ts_i, unit="ns").date()
        max_date = pd.to_datetime(max_ts_i, unit="ns").date()

        if min_date != expected_date.date() or max_date != expected_date.date():
            LOGGER.warning(
                "  %s: Data spans %s to %s (date mismatch)",
                expected_date.date(),
                min_date,
                max_date,
            )
            return False

        record_count = len(df)
        if record_count < 1_000:
            LOGGER.warning(
                "  %s: Only %s records (likely incomplete)",
                expected_date.date(),
                record_count,
            )
            return False

        market_hours_data = df.filter(
            (pl.from_epoch("ts_event", time_unit="ns").dt.hour() >= 9)
            & (pl.from_epoch("ts_event", time_unit="ns").dt.hour() <= 16),
        )

        if len(market_hours_data) < record_count * 0.8:
            LOGGER.warning(
                "  %s: Low market hours coverage (%s/%s records)",
                expected_date.date(),
                len(market_hours_data),
                record_count,
            )

        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning(
            "  %s: Error validating file: %s",
            expected_date.date(),
            exc,
            exc_info=True,
        )
        return False


def get_business_dates(start_date: datetime, end_date: datetime) -> list[datetime]:
    """
    Return Monday-Friday datetimes inclusive between ``start_date`` and ``end_date``.
    """
    dates: list[datetime] = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def detect_data_gaps(
    symbol: str,
    output_dir: Path,
    start_date: datetime,
    end_date: datetime,
    *,
    schema: str,
) -> list[datetime]:
    """
    Detect missing dates in existing L2 data for a symbol.
    """
    final_tag = "mbp-10" if schema == "mbp-10" else schema
    final_file = output_dir / f"{symbol}_{final_tag}.parquet"

    if not final_file.exists():
        return get_business_dates(start_date, end_date)

    if not validate_data_integrity(final_file, symbol, start_date):
        LOGGER.warning(
            "Existing data file failed integrity check - will re-download affected dates",
        )

    try:
        df = pl.read_parquet(final_file)
        if df.is_empty():
            return get_business_dates(start_date, end_date)
        existing_dates = (
            df.select(pl.from_epoch("ts_event", time_unit="ns").dt.date().alias("date"))
            .unique()
            .sort("date")
        )
        dates_in_data = set(existing_dates.to_series().to_list())
        expected_dates = get_business_dates(start_date, end_date)
        missing_dates: list[datetime] = []
        for date in expected_dates:
            if date.date() not in dates_in_data:
                missing_dates.append(date)
            else:
                tag = "mbp10" if schema == "mbp-10" else schema.replace("-", "")
                daily_file = output_dir / f"{symbol}_{tag}_{date.strftime('%Y%m%d')}.parquet"
                if daily_file.exists() and not validate_data_integrity(daily_file, symbol, date):
                    LOGGER.info("Re-downloading %s due to integrity issues", date.date())
                    missing_dates.append(date)
        return missing_dates
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Error reading existing data for %s: %s", symbol, exc, exc_info=True)
        return get_business_dates(start_date, end_date)


def merge_new_with_existing(symbol: str, output_dir: Path, *, schema: str) -> None:
    """
    Merge newly downloaded daily files with existing aggregate using streaming writes.
    """
    tag = "mbp10" if schema == "mbp-10" else schema.replace("-", "")
    daily_files = sorted(output_dir.glob(f"{symbol}_{tag}_*.parquet"))
    if not daily_files:
        return

    final_tag = "mbp-10" if schema == "mbp-10" else schema
    final_file = output_dir / f"{symbol}_{final_tag}.parquet"
    tmp_file = output_dir / f"{symbol}_{final_tag}.tmp.parquet"

    def _append_files(
        writer: pq.ParquetWriter | None,
        files: Sequence[Path],
    ) -> pq.ParquetWriter | None:
        for file_path in files:
            try:
                pf = pq.ParquetFile(file_path)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_file, pf.schema_arrow)
                for rg in range(pf.num_row_groups or 1):
                    table = pf.read_row_group(rg) if pf.num_row_groups else pf.read()
                    if writer.schema and table.schema != writer.schema:
                        table = table.select(writer.schema.names)
                    writer.write_table(table)
            except Exception as exc:
                LOGGER.error("  Error appending %s: %s", file_path.name, exc, exc_info=True)
        return writer

    writer: pq.ParquetWriter | None = None
    try:
        if final_file.exists():
            writer = _append_files(writer, [final_file])
        writer = _append_files(writer, daily_files)
        if writer is not None:
            writer.close()
        tmp_file.replace(final_file)
        for file_path in daily_files:
            try:
                file_path.unlink()
            except OSError as exc:
                LOGGER.debug("Failed to unlink %s: %s", file_path, exc, exc_info=True)
        size_mb = final_file.stat().st_size / (1024 * 1024)
        LOGGER.info("Merged data (streaming): %.1f MB -> %s", size_mb, final_file.name)
    except Exception as exc:
        try:
            if writer is not None:
                writer.close()
        except Exception as close_exc:  # pragma: no cover - defensive logging
            LOGGER.debug("Closing Parquet writer failed: %s", close_exc, exc_info=True)
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError as rm_exc:  # pragma: no cover - defensive logging
                LOGGER.debug("Failed to remove tmp file %s: %s", tmp_file, rm_exc, exc_info=True)
        raise exc


def get_tier1_symbols() -> list[str]:
    """
    Return Tier 1 symbols sourced from progress file or universe constant.
    """
    progress_file = Path("tier1_l1_progress.json")
    if progress_file.exists():
        try:
            with progress_file.open("r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
        except Exception:  # pragma: no cover - defensive logging
            data = {}
        symbols = sorted({str(item) for item in data.get("completed_bbo", [])})
        if symbols:
            return symbols
    return list(TIER1_CORE)


def download_l2_daily(
    service: DatabentoIngestionService,
    symbol: str,
    date: datetime,
    output_dir: Path,
    *,
    dataset: str,
    schema: str,
    retry_impl: Callable[..., Any] | None = None,
) -> int:
    """
    Download L2 data for a single day with retries.
    """
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    if start.weekday() >= 5:
        return 0

    class _NonTransientSkip(Exception):
        pass

    def _attempt() -> int:
        df = fetch_symbol_data(
            service=service,
            dataset=dataset,
            schema=schema,
            symbol=symbol,
            start=start,
            end=end,
            reason=f"l2_daily_{symbol}_{date:%Y%m%d}",
        )

        if df is None or df.empty:
            LOGGER.info("  %s: No data available", date.date())
            return 0

        if len(df) < 100:
            LOGGER.warning("  %s: Only %s records (may be incomplete)", date.date(), len(df))

        date_str = date.strftime("%Y%m%d")
        tag = "mbp10" if schema == "mbp-10" else schema.replace("-", "")
        output_file = output_dir / f"{symbol}_{tag}_{date_str}.parquet"
        df.to_parquet(output_file)
        size_mb = output_file.stat().st_size / (1024 * 1024)
        LOGGER.info("  %s: %s records, %.1f MB", date.date(), len(df), size_mb)
        return len(df)

    def _on_exc(attempt: int, exc: BaseException) -> None:
        message = str(exc)
        if "403" in message or "license" in message:
            raise _NonTransientSkip(message)
        transient = any(code in message for code in ("504", "502", "500", "timeout", "429"))
        if transient:
            LOGGER.warning(
                "  %s: Transient error (attempt %s) - %s",
                date.date(),
                attempt + 1,
                message,
            )
        else:
            raise _NonTransientSkip(message)

    retry_callable: Callable[..., Any]
    if retry_impl is None:
        from ml.common.retry_utils import retry_with_backoff

        retry_callable = retry_with_backoff
    else:
        retry_callable = retry_impl

    try:
        result = retry_callable(
            _attempt,
            max_attempts=3,
            initial_delay=2.0,
            multiplier=2.0,
            max_delay=30.0,
            jitter=0.0,
            on_exception=_on_exc,
            retry_on=(Exception,),
        )
        return int(result)
    except _NonTransientSkip as exc:
        LOGGER.error("  %s: Error - %s", date.date(), exc, exc_info=True)
        return 0


def _clean_stale_temp_files(symbol: str, output_dir: Path) -> None:
    for pattern in (f"{symbol}_temp_chunk_*.parquet", f"{symbol}_temp_merged_*.parquet"):
        for file_path in output_dir.glob(pattern):
            try:
                file_path.unlink()
            except OSError as exc:
                LOGGER.debug(
                    "Failed to unlink stale temp file %s: %s",
                    file_path,
                    exc,
                    exc_info=True,
                )


def _validate_daily_file(file_path: Path) -> bool:
    try:
        pf = pq.ParquetFile(file_path)
        return pf.metadata is not None and pf.metadata.num_rows > 0
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("  Corrupted file detected: %s - %s", file_path.name, exc, exc_info=True)
        return False


def _stream_merge_daily_files(daily_files: Sequence[Path], tmp_output: Path) -> None:
    writer: pq.ParquetWriter | None = None
    valid_files: list[Path] = []
    for file_path in daily_files:
        if _validate_daily_file(file_path):
            valid_files.append(file_path)
        else:
            LOGGER.warning(
                "  Skipping corrupted file: %s (will be re-downloaded on next run)",
                file_path.name,
            )
    if not valid_files:
        raise ValueError("No valid daily files found to combine")

    try:
        for idx, file_path in enumerate(valid_files, 1):
            try:
                pf = pq.ParquetFile(file_path)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_output, pf.schema_arrow)
                for rg in range(pf.num_row_groups or 1):
                    table = pf.read_row_group(rg) if pf.num_row_groups else pf.read()
                    if writer.schema and table.schema != writer.schema:
                        table = table.select(writer.schema.names)
                    writer.write_table(table)
                if idx % 3 == 0 or idx == len(valid_files):
                    mem_percent = psutil.virtual_memory().percent
                    LOGGER.info(
                        "  Appended %s/%s daily files (Memory: %.1f%%)",
                        idx,
                        len(valid_files),
                        mem_percent,
                    )
            except Exception as exc:
                LOGGER.error("  Error processing %s: %s", file_path.name, exc, exc_info=True)
    finally:
        if writer is not None:
            writer.close()


def combine_daily_files(symbol: str, output_dir: Path, *, schema: str) -> None:
    """
    Combine per-day L2 files into a single parquet file via streaming.
    """
    _clean_stale_temp_files(symbol, output_dir)
    tag = "mbp10" if schema == "mbp-10" else schema.replace("-", "")
    daily_files = sorted(output_dir.glob(f"{symbol}_{tag}_*.parquet"))
    if not daily_files:
        return

    LOGGER.info("Combining %s daily files for %s", len(daily_files), symbol)
    total_size_mb = sum(file_path.stat().st_size for file_path in daily_files) / (1024 * 1024)
    LOGGER.info("  Total size: %.1f MB; streaming merge", total_size_mb)

    final_tag = "mbp-10" if schema == "mbp-10" else schema
    output_file = output_dir / f"{symbol}_{final_tag}.parquet"
    tmp_output = output_dir / f"{symbol}_{final_tag}.tmp.parquet"

    try:
        _stream_merge_daily_files(daily_files, tmp_output)
        if tmp_output.exists():
            tmp_output.replace(output_file)
        for file_path in daily_files:
            try:
                file_path.unlink()
            except OSError as exc:
                LOGGER.debug("Failed to unlink daily file %s: %s", file_path, exc, exc_info=True)
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            pf_final = pq.ParquetFile(output_file)
            total_records = int(pf_final.metadata.num_rows) if pf_final.metadata else 0
            LOGGER.info(
                "Created %s: %s records, %.1f MB",
                output_file.name,
                total_records,
                size_mb,
            )
    except Exception as exc:
        if tmp_output.exists():
            try:
                tmp_output.unlink()
            except OSError as rm_exc:  # pragma: no cover - defensive logging
                LOGGER.debug(
                    "Failed to unlink tmp output %s: %s",
                    tmp_output,
                    rm_exc,
                    exc_info=True,
                )
        raise exc


def populate_l2_data(
    config: L2PopulateConfig,
    *,
    service: DatabentoIngestionService,
) -> L2PopulateResult:
    """
    Populate L2 data according to ``config`` returning aggregated metrics.
    """
    symbols_work = list(config.symbols)
    if config.shuffle:
        _shuffle(symbols_work)
    if config.symbol_offset:
        symbols_work = symbols_work[config.symbol_offset :]
    if config.max_symbols is not None:
        symbols_work = symbols_work[: max(0, int(config.max_symbols))]

    LOGGER.info("Downloading L2 data for %s symbols", len(symbols_work))
    LOGGER.info(
        "Date range: %s to %s",
        config.start_date.date(),
        config.end_date.date(),
    )
    LOGGER.info("Dataset: %s | Schema: %s", config.dataset, config.schema)

    total_records = 0
    total_size_mb = 0.0

    progress_path = config.progress_file
    progress = _load_progress(progress_path)
    terminate: dict[str, bool] = {"stop": False}

    def _handle_sig(_signum: int, _frame: Any) -> None:
        terminate["stop"] = True
        _save_progress(progress_path, progress)
        LOGGER.info("Received termination signal; progress saved.")

    try:
        signal.signal(signal.SIGTERM, _handle_sig)
        signal.signal(signal.SIGINT, _handle_sig)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.debug("Signal handler setup failed: %s", exc, exc_info=True)

    min_interval = 60.0 / max(1, int(config.rate_limit))
    last_call_ts = 0.0

    for idx, symbol in enumerate(symbols_work, 1):
        LOGGER.info("\n[%s/%s] Processing %s...", idx, len(symbols_work), symbol)
        output_dir = config.data_dir / symbol / "l2"
        output_dir.mkdir(parents=True, exist_ok=True)

        final_tag = "mbp-10" if config.schema == "mbp-10" else config.schema
        final_file = output_dir / f"{symbol}_{final_tag}.parquet"

        if config.force:
            dates_to_download = get_business_dates(config.start_date, config.end_date)
            if final_file.exists():
                LOGGER.info("  Force mode: removing existing data")
                try:
                    final_file.unlink()
                except OSError as exc:
                    LOGGER.debug(
                        "Failed to unlink existing file %s: %s",
                        final_file,
                        exc,
                        exc_info=True,
                    )
        elif config.check_gaps:
            done_dates = set(progress.get(symbol, []))
            if done_dates:
                expected = get_business_dates(config.start_date, config.end_date)
                dates_to_download = [
                    date for date in expected if date.strftime("%Y-%m-%d") not in done_dates
                ]
            else:
                dates_to_download = detect_data_gaps(
                    symbol,
                    output_dir,
                    config.start_date,
                    config.end_date,
                    schema=config.schema,
                )
            if not dates_to_download:
                if final_file.exists():
                    size_mb = final_file.stat().st_size / (1024 * 1024)
                    LOGGER.info("  No gaps found: %.1f MB - complete", size_mb)
                    total_size_mb += size_mb
                else:
                    LOGGER.info("  No existing data and no gaps to fill")
                continue
            LOGGER.info("  Found %s missing dates to download", len(dates_to_download))
        else:
            if final_file.exists() and config.resume:
                size_mb = final_file.stat().st_size / (1024 * 1024)
                LOGGER.info("  Already exists: %.1f MB - skipping", size_mb)
                total_size_mb += size_mb
                continue
            dates_to_download = get_business_dates(config.start_date, config.end_date)

        symbol_records = 0
        done_dates = set(progress.get(symbol, []))
        for date in dates_to_download:
            if terminate["stop"]:
                break
            date_iso = date.strftime("%Y-%m-%d")
            if date_iso in done_dates:
                LOGGER.info("  %s: already completed (progress file)", date.date())
                continue
            now = time.time()
            elapsed = now - last_call_ts
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            records = download_l2_daily(
                service,
                symbol,
                date,
                output_dir,
                dataset=config.dataset,
                schema=config.schema,
            )
            last_call_ts = time.time()
            symbol_records += records
            if records > 0:
                progress.setdefault(symbol, []).append(date_iso)
                _save_progress(progress_path, progress)

        if symbol_records > 0:
            if config.check_gaps and final_file.exists():
                merge_new_with_existing(symbol, output_dir, schema=config.schema)
            else:
                combine_daily_files(symbol, output_dir, schema=config.schema)
            if final_file.exists():
                size_mb = final_file.stat().st_size / (1024 * 1024)
                total_size_mb += size_mb

        total_records += symbol_records
        LOGGER.info("  Downloaded: %s new records", symbol_records)
        if config.sleep_between_symbols > 0:
            time.sleep(float(config.sleep_between_symbols))

    LOGGER.info("\n%s", "=" * 50)
    LOGGER.info("COMPLETE: %s total records, %.1f MB", total_records, total_size_mb)
    return L2PopulateResult(
        total_records=total_records,
        total_size_mb=total_size_mb,
        symbols_processed=len(symbols_work),
    )


__all__ = [
    "L2PopulateConfig",
    "L2PopulateResult",
    "combine_daily_files",
    "detect_data_gaps",
    "download_l2_daily",
    "get_business_dates",
    "get_tier1_symbols",
    "merge_new_with_existing",
    "populate_l2_data",
    "validate_data_integrity",
]
