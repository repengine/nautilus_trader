"""Calibration capture workflow for EQUS fallback normalization."""

from __future__ import annotations

import math
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import structlog

from ml.data.ingest.calibration import CalibrationBundle
from ml.data.ingest.calibration import SymbolCalibration
from ml.data.ingest.calibration import dump_calibration_bundle
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionRequest


logger = structlog.get_logger(__name__)


class IngestionServiceProtocol(Protocol):
    """Minimal interface of :class:`DatabentoIngestionService` used here."""

    def ingest(
        self,
        request: IngestionRequest,
        *,
        on_chunk: Callable[[IngestionChunk], None] | None = None,
    ) -> Sequence[object]: ...


@dataclass(slots=True, frozen=True)
class CalibrationCaptureConfig:
    """Configuration for generating calibration artefacts."""

    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    output_path: Path | None = None
    eq_dataset: str = "EQUS.MINI"
    eq_schema: str = "ohlcv-1m"
    fallback_dataset: str = "XNAS.ITCH"
    trades_schema: str = "trades"
    depth_schema: str = "mbp-1"
    allow_cost: bool = True
    chunk_days: int = 7
    min_ratio_minutes: int = 10
    price_scale_clip: tuple[float, float] = (0.05, 20.0)
    volume_scale_clip: tuple[float, float] = (0.01, 100.0)


@dataclass(slots=True, frozen=True)
class CalibrationCaptureResult:
    """Result of a calibration capture run."""

    bundle: CalibrationBundle
    output_path: Path | None


class CalibrationCaptureService:
    """Service that derives calibration artefacts from overlapping EQUS/ITCH data."""

    def __init__(self, ingestion_service: IngestionServiceProtocol) -> None:
        self._ingestion_service = ingestion_service

    def capture(self, config: CalibrationCaptureConfig) -> CalibrationCaptureResult:
        if config.end <= config.start:
            raise ValueError("Calibration window end must be after start")
        if not config.symbols:
            raise ValueError("At least one symbol must be provided")

        generated_at = datetime.now(tz=UTC)
        symbol_map: dict[str, SymbolCalibration] = {}
        for raw_symbol in config.symbols:
            symbol = raw_symbol.strip()
            if not symbol:
                continue
            logger.info(
                "calibration.capture.symbol.start",
                symbol=symbol,
                start=str(config.start),
                end=str(config.end),
            )
            eq_frame = self._fetch_dataframe(
                dataset=config.eq_dataset,
                schema=config.eq_schema,
                symbol=symbol,
                start=config.start,
                end=config.end,
                request_reason="calibration_eq_window",
                allow_cost=config.allow_cost,
                chunk_days=config.chunk_days,
            )
            trades_frame = self._fetch_dataframe(
                dataset=config.fallback_dataset,
                schema=config.trades_schema,
                symbol=symbol,
                start=config.start,
                end=config.end,
                request_reason="calibration_trades_window",
                allow_cost=config.allow_cost,
                chunk_days=config.chunk_days,
            )
            depth_frame = self._fetch_dataframe(
                dataset=config.fallback_dataset,
                schema=config.depth_schema,
                symbol=symbol,
                start=config.start,
                end=config.end,
                request_reason="calibration_depth_window",
                allow_cost=config.allow_cost,
                chunk_days=config.chunk_days,
            )
            calibration = self._build_symbol_calibration(
                symbol=symbol,
                eq_frame=eq_frame,
                trades_frame=trades_frame,
                depth_frame=depth_frame,
                min_ratio_minutes=config.min_ratio_minutes,
                price_clip=config.price_scale_clip,
                volume_clip=config.volume_scale_clip,
            )
            symbol_map[symbol.upper()] = calibration
            logger.info(
                "calibration.capture.symbol.completed",
                symbol=symbol,
                allowlist=sorted(calibration.sale_condition_allowlist),
                volume_minutes=len(calibration.volume_scale_by_minute),
                price_minutes=len(calibration.price_scaling_by_minute),
                split_events=len(calibration.split_events),
                auction_minutes=len(calibration.exclude_auction_minutes),
                allowlist_empty=not calibration.sale_condition_allowlist,
                volume_scalers_empty=not calibration.volume_scale_by_minute,
                price_scalers_empty=not calibration.price_scaling_by_minute,
            )

        bundle = CalibrationBundle(generated_at=generated_at, symbols=symbol_map)
        if config.output_path is not None:
            dump_calibration_bundle(bundle, config.output_path)
            logger.info(
                "calibration.capture.bundle.persisted",
                path=str(config.output_path),
                symbols=len(symbol_map),
            )
        return CalibrationCaptureResult(bundle=bundle, output_path=config.output_path)

    def _fetch_dataframe(
        self,
        *,
        dataset: str,
        schema: str,
        symbol: str,
        start: datetime,
        end: datetime,
        request_reason: str,
        allow_cost: bool,
        chunk_days: int,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        def _collect(chunk: IngestionChunk) -> None:
            if chunk.frame.empty:
                return
            frames.append(chunk.frame.copy(deep=True))

        request = IngestionRequest(
            dataset=dataset,
            schema=schema,
            symbols=(symbol,),
            start=start,
            end=end,
            allow_cost=allow_cost,
            chunk_days=chunk_days,
            reason=request_reason,
        )
        self._ingestion_service.ingest(request, on_chunk=_collect)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if "ts_event" in combined.columns:
            combined = combined.sort_values("ts_event").reset_index(drop=True)
        return combined

    def _build_symbol_calibration(
        self,
        *,
        symbol: str,
        eq_frame: pd.DataFrame,
        trades_frame: pd.DataFrame,
        depth_frame: pd.DataFrame,
        min_ratio_minutes: int,
        price_clip: tuple[float, float],
        volume_clip: tuple[float, float],
    ) -> SymbolCalibration:
        eq_processed = self._prepare_equity_bars(eq_frame)
        trades_processed = self._prepare_trades(trades_frame)
        allowlist, volume_series, notional_series = self._derive_sale_condition_allowlist(
            eq_series=eq_processed.volume_by_minute,
            trades_by_condition=trades_processed.by_condition,
        )
        fallback_volume_series = volume_series
        fallback_notional_series = notional_series
        volume_scale_by_minute = self._compute_volume_scaling(
            eq_series=eq_processed.volume_by_minute,
            fallback_series=fallback_volume_series,
            min_samples=min_ratio_minutes,
            clip=volume_clip,
        )
        price_scale_by_minute = self._compute_price_scaling(
            eq_close_series=eq_processed.close_by_minute,
            fallback_volume_series=fallback_volume_series,
            fallback_notional_series=fallback_notional_series,
            min_samples=min_ratio_minutes,
            clip=price_clip,
        )
        split_events = self._derive_split_events(depth_frame)
        auction_minutes = self._derive_auction_minutes(depth_frame)
        return SymbolCalibration(
            sale_condition_allowlist=allowlist,
            volume_scale_by_minute=volume_scale_by_minute,
            price_scaling_by_minute=price_scale_by_minute,
            split_events=split_events,
            exclude_auction_minutes=auction_minutes,
        )

    @dataclass(slots=True, frozen=True)
    class _EquityBars:
        volume_by_minute: pd.Series
        close_by_minute: pd.Series

    @dataclass(slots=True, frozen=True)
    class _TradesByCondition:
        by_condition: Mapping[str, tuple[pd.Series, pd.Series]]

    def _prepare_equity_bars(self, frame: pd.DataFrame) -> _EquityBars:
        if frame.empty or "ts_event" not in frame.columns:
            return self._EquityBars(pd.Series(dtype=float), pd.Series(dtype=float))
        working = frame.copy(deep=True)
        ts_numeric = pd.to_numeric(working["ts_event"], errors="coerce")
        volume_source = working["volume"] if "volume" in working.columns else pd.Series(np.nan, index=working.index)
        close_source = working["close"] if "close" in working.columns else pd.Series(np.nan, index=working.index)
        working = working.assign(
            _ts=ts_numeric.astype("Int64"),
            _volume=pd.to_numeric(volume_source, errors="coerce"),
            _close=pd.to_numeric(close_source, errors="coerce"),
        ).dropna(subset=["_ts"])
        if working.empty:
            return self._EquityBars(pd.Series(dtype=float), pd.Series(dtype=float))
        minute_ns = (working["_ts"].astype(np.int64) // 60_000_000_000) * 60_000_000_000
        grouped = working.groupby(minute_ns)
        volume_series = grouped["_volume"].sum(min_count=1).astype(float)
        close_series = grouped["_close"].last()
        volume_series.index = volume_series.index.astype(np.int64)
        close_series.index = close_series.index.astype(np.int64)
        return self._EquityBars(volume_series, close_series)

    def _prepare_trades(self, frame: pd.DataFrame) -> _TradesByCondition:
        if frame.empty or "ts_event" not in frame.columns:
            return self._TradesByCondition(by_condition={})
        working = frame.copy(deep=True)
        ts_numeric = pd.to_numeric(working["ts_event"], errors="coerce")
        working = working.assign(
            _ts=ts_numeric.astype("Int64"),
        ).dropna(subset=["_ts"])
        size_column = self._resolve_column(working, ("size", "quantity", "volume", "trade_size"))
        price_column = self._resolve_column(working, ("price", "trade_px", "last", "px_last"))
        if size_column is None or price_column is None:
            return self._TradesByCondition(by_condition={})
        assert size_column is not None
        assert price_column is not None
        sale_series = working.get("sale_condition")
        sale_normalised = (
            sale_series.astype(str).str.strip().str.upper()
            if sale_series is not None
            else pd.Series(["" for _ in range(len(working.index))], index=working.index)
        )
        working = working.assign(
            _size=pd.to_numeric(working[size_column], errors="coerce"),
            _price=pd.to_numeric(working[price_column], errors="coerce"),
            _sale=sale_normalised,
        ).dropna(subset=["_size", "_price"])
        if working.empty:
            return self._TradesByCondition(by_condition={})
        minute_ns = (working["_ts"].astype(np.int64) // 60_000_000_000) * 60_000_000_000
        working = working.assign(_minute=minute_ns.astype(np.int64))
        grouped: dict[str, tuple[pd.Series, pd.Series]] = {}
        for sale_condition, group in working.groupby("_sale"):
            volume_series = group.groupby("_minute")["_size"].sum(min_count=1).astype(float)
            notional_series = (
                group.assign(_notional=group["_price"] * group["_size"])
                .groupby("_minute")["_notional"]
                .sum(min_count=1)
                .astype(float)
            )
            volume_series.index = volume_series.index.astype(np.int64)
            notional_series.index = notional_series.index.astype(np.int64)
            grouped[str(sale_condition)] = (volume_series, notional_series)
        return self._TradesByCondition(by_condition=grouped)

    @staticmethod
    def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        for column in candidates:
            if column in frame.columns:
                return column
        return None

    def _derive_sale_condition_allowlist(
        self,
        *,
        eq_series: pd.Series,
        trades_by_condition: Mapping[str, tuple[pd.Series, pd.Series]],
    ) -> tuple[frozenset[str], pd.Series, pd.Series]:
        if eq_series.empty or not trades_by_condition:
            return frozenset(), pd.Series(dtype=float), pd.Series(dtype=float)

        eq_series = eq_series.astype(float)
        combined_volume = pd.Series(dtype=float)
        combined_notional = pd.Series(dtype=float)
        residual = eq_series.abs().sum()
        allowlist: list[str] = []
        orders = sorted(
            trades_by_condition.items(),
            key=lambda item: float(item[1][0].sum()),
            reverse=True,
        )
        for sale_condition, (volume_series, notional_series) in orders:
            if not sale_condition:
                continue
            candidate_volume = combined_volume.add(volume_series, fill_value=0.0)
            candidate_notional = combined_notional.add(notional_series, fill_value=0.0)
            candidate_residual = self._volume_residual(eq_series, candidate_volume)
            if candidate_residual <= residual:
                allowlist.append(sale_condition)
                combined_volume = candidate_volume
                combined_notional = candidate_notional
                residual = candidate_residual

        if not allowlist and orders:
            first_condition, (volume_series, notional_series) = next(
                ((cond, series_pair) for cond, series_pair in orders if cond),
                orders[0],
            )
            allowlist.append(first_condition)
            combined_volume = volume_series.copy()
            combined_notional = notional_series.copy()

        combined_volume.index = combined_volume.index.astype(np.int64)
        combined_notional.index = combined_notional.index.astype(np.int64)
        allowlist_set = frozenset(item for item in allowlist if item)
        return allowlist_set, combined_volume, combined_notional

    @staticmethod
    def _volume_residual(eq_series: pd.Series, fallback_series: pd.Series) -> float:
        aligned = pd.DataFrame(index=eq_series.index.union(fallback_series.index))
        aligned["eq"] = eq_series.reindex(aligned.index, fill_value=0.0)
        aligned["fallback"] = fallback_series.reindex(aligned.index, fill_value=0.0)
        return float((aligned["eq"] - aligned["fallback"]).abs().sum())

    def _compute_volume_scaling(
        self,
        *,
        eq_series: pd.Series,
        fallback_series: pd.Series,
        min_samples: int,
        clip: tuple[float, float],
    ) -> dict[int, float]:
        if eq_series.empty or fallback_series.empty:
            return {}
        aligned = pd.DataFrame(index=eq_series.index.union(fallback_series.index))
        aligned["eq"] = eq_series.reindex(aligned.index, fill_value=0.0).astype(float)
        aligned["fallback"] = fallback_series.reindex(aligned.index, fill_value=0.0).astype(float)
        mask = (aligned["eq"] > 0) & (aligned["fallback"] > 0)
        if not mask.any():
            return {}
        aligned = aligned.loc[mask]
        ratios = aligned["eq"] / aligned["fallback"]
        minute_index = aligned.index.to_numpy(dtype=np.int64, copy=False)
        minute_of_day = ((minute_index // 60_000_000_000) % 1440).astype(int)
        aligned = aligned.assign(_ratio=ratios.to_numpy(), _minute=minute_of_day)
        grouped = aligned.groupby("_minute")
        result: dict[int, float] = {}
        for minute, group in grouped:
            if group.shape[0] < min_samples:
                continue
            median_ratio = float(group["_ratio"].median())
            clipped = float(np.clip(median_ratio, clip[0], clip[1]))
            minute_int = int(cast(int, minute))
            result[minute_int] = clipped
        return result

    def _compute_price_scaling(
        self,
        *,
        eq_close_series: pd.Series,
        fallback_volume_series: pd.Series,
        fallback_notional_series: pd.Series,
        min_samples: int,
        clip: tuple[float, float],
    ) -> dict[int, float]:
        if eq_close_series.empty or fallback_volume_series.empty:
            return {}
        aligned_index = (
            eq_close_series.index
            .union(fallback_volume_series.index)
            .union(fallback_notional_series.index)
        )
        aligned = pd.DataFrame(index=aligned_index)
        aligned["eq_close"] = eq_close_series.reindex(aligned.index, fill_value=np.nan).astype(float)
        aligned["fallback_volume"] = fallback_volume_series.reindex(aligned.index, fill_value=0.0).astype(float)
        aligned["fallback_notional"] = fallback_notional_series.reindex(aligned.index, fill_value=0.0).astype(float)
        mask = (aligned["eq_close"] > 0) & (aligned["fallback_volume"] > 0)
        aligned = aligned.loc[mask]
        if aligned.empty:
            return {}
        aligned = aligned.assign(
            _vwap=aligned["fallback_notional"] / aligned["fallback_volume"],
        )
        price_mask = aligned["_vwap"] > 0
        aligned = aligned.loc[price_mask]
        if aligned.empty:
            return {}
        ratios = aligned["eq_close"] / aligned["_vwap"]
        minute_index = aligned.index.to_numpy(dtype=np.int64, copy=False)
        minute_of_day = ((minute_index // 60_000_000_000) % 1440).astype(int)
        aligned = aligned.assign(_ratio=ratios.to_numpy(), _minute=minute_of_day)
        grouped = aligned.groupby("_minute")
        result: dict[int, float] = {}
        for minute, group in grouped:
            if group.shape[0] < min_samples:
                continue
            median_ratio = float(group["_ratio"].median())
            clipped = float(np.clip(median_ratio, clip[0], clip[1]))
            minute_int = int(cast(int, minute))
            result[minute_int] = clipped
        return result

    def _derive_auction_minutes(self, depth_frame: pd.DataFrame) -> frozenset[int]:
        if depth_frame.empty:
            return frozenset()
        working = depth_frame.copy(deep=True)
        ts_column = self._resolve_column(working, ("ts_event", "ts_recv", "ts"))
        if ts_column is None:
            return frozenset()
        ts_numeric = pd.to_numeric(working[ts_column], errors="coerce")
        dt_index = pd.to_datetime(ts_numeric, utc=True, unit="ns", errors="coerce")
        working = working.assign(_timestamp=dt_index)
        working = working.dropna(subset=["_timestamp"])
        if working.empty:
            return frozenset()
        mask = pd.Series(False, index=working.index)
        for column in ("is_auction", "auction", "auction_type", "event_type", "auction_flag"):
            if column not in working.columns:
                continue
            series = working[column]
            if series.dtype == bool:
                mask = mask | series
            else:
                mask = mask | series.astype(str).str.contains("auction", case=False, na=False)
        if not mask.any():
            return frozenset()
        minutes = (
            working.loc[mask, "_timestamp"]
            .dt.floor("min")
            .dropna()
            .astype("int64")
        )
        minute_of_day = ((minutes // 60_000_000_000) % 1440).astype(int)
        return frozenset(int(value) for value in minute_of_day)

    def _derive_split_events(self, depth_frame: pd.DataFrame) -> dict[str, float]:
        if depth_frame.empty:
            return {}
        working = depth_frame.copy(deep=True)
        ts_column = self._resolve_column(working, ("ts_event", "ts_recv", "ts"))
        if ts_column is None:
            return {}
        ts_numeric = pd.to_numeric(working[ts_column], errors="coerce")
        dt_index = pd.to_datetime(ts_numeric, utc=True, unit="ns", errors="coerce")
        working = working.assign(_timestamp=dt_index)
        split_columns = [
            column
            for column in ("split_factor", "split_ratio", "stock_split_factor")
            if column in working.columns
        ]
        if not split_columns:
            return {}
        result: dict[str, float] = {}
        for _, row in working.iterrows():
            timestamp = row.get("_timestamp")
            if pd.isna(timestamp):
                continue
            for column in split_columns:
                factor = self._parse_split_factor(row[column])
                if factor is None or math.isclose(factor, 1.0, rel_tol=1e-6):
                    continue
                date_key = cast(datetime, timestamp).date().isoformat()
                result[date_key] = float(factor)
        return result

    @staticmethod
    def _parse_split_factor(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            factor = float(value)
            return factor if factor > 0 else None
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            if ":" in cleaned:
                left, right = cleaned.split(":", 1)
                try:
                    numerator = float(right)
                    denominator = float(left)
                    if denominator > 0:
                        return numerator / denominator
                except ValueError:
                    return None
            try:
                factor = float(cleaned)
            except ValueError:
                return None
            return factor if factor > 0 else None
        return None


__all__ = [
    "CalibrationCaptureConfig",
    "CalibrationCaptureResult",
    "CalibrationCaptureService",
    "IngestionServiceProtocol",
]
