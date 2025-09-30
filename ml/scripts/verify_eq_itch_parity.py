"""CLI utilities to verify EQUS↔ITCH normalization and optional backfill."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import structlog

from ml.data.ingest.calibration import load_calibration_bundle
from ml.data.ingest.canonicalization import canonicalize_equities_minute_bars
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionRequest
from ml.features.engineering import FeatureConfig
from ml.features.engineering import FeatureEngineer
from ml.stores.providers import SqlMarketDataWriter


DEFAULT_SCHEMA = "ohlcv-1m"


logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class CanonicalizedSlice:
    dataframe: pd.DataFrame
    instrument_id: str | None


@dataclass(slots=True, frozen=True)
class ParityScenario:
    """Parity comparison definition for a single EQUS vs fallback slice."""

    label: str
    eq_symbol: str
    fallback_symbol: str
    fallback_dataset: str
    start: datetime
    end: datetime
    schema: str = DEFAULT_SCHEMA


@dataclass(slots=True, frozen=True)
class ParityMetrics:
    """Feature parity metrics for a single scenario."""

    timestamp_count: float
    max_abs_diff: float
    mean_abs_diff: float
    p99_abs_diff: float
    worst_timestamp_ns: int | None
    worst_timestamp_iso: str | None
    price_close_max_abs_diff: float | None
    price_close_p99_abs_diff: float | None
    price_close_mean_abs_diff: float | None
    price_close_correlation: float | None
    volume_correlation: float | None
    volume_ratio_stats: Mapping[str, float] | None
    volume_residual_abs: float | None
    volume_residual_rel: float | None


@dataclass(slots=True, frozen=True)
class ParityScenarioResult:
    """Report row combining scenario metadata and computed metrics."""

    scenario: ParityScenario
    metrics: ParityMetrics


@dataclass(slots=True, frozen=True)
class ParitySuiteReport:
    """Aggregated parity report across multiple scenarios."""

    generated_at: datetime
    results: tuple[ParityScenarioResult, ...]


def ensure_calibration_fresh(
    *,
    calibration_path: Path | None,
    max_age_days: int,
    allow_missing: bool,
) -> None:
    """Validate calibration freshness before running parity diagnostics."""
    resolved_path = calibration_path
    if resolved_path is None:
        env_path = os.getenv("ML_EQUS_CALIBRATION_PATH")
        resolved_path = Path(env_path) if env_path else None

    if resolved_path is None:
        if allow_missing:
            logger.warning("calibration.bundle.missing")
            return
        raise RuntimeError(
            "Calibration bundle required but ML_EQUS_CALIBRATION_PATH is not set.",
        )

    if not resolved_path.exists():
        if allow_missing:
            logger.warning("calibration.bundle.not_found", path=str(resolved_path))
            return
        raise RuntimeError(f"Calibration bundle not found at {resolved_path}")

    bundle = load_calibration_bundle(resolved_path)
    age = datetime.now(tz=UTC) - bundle.generated_at.astimezone(UTC)
    if age > timedelta(days=max_age_days):
        raise RuntimeError(
            f"Calibration bundle at {resolved_path} is stale (age={age.days} days)",
        )
    logger.info(
        "calibration.bundle.valid",
        path=str(resolved_path),
        generated_at=bundle.generated_at.isoformat(),
        age_days=age.days,
    )


DEFAULT_SUITE_OUTPUT: Final[Path] = Path("ml/tests/validation_reports/equs_itch_parity_summary.json")

_DEFAULT_SUITE_ROWS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("AAPL_2023_06_15", "AAPL", "2023-06-15T08:00Z", "2023-06-16T00:00Z"),
    ("MSFT_2024_03_12", "MSFT", "2024-03-12T08:00Z", "2024-03-13T00:00Z"),
    ("NVDA_2024_11_05", "NVDA", "2024-11-05T08:00Z", "2024-11-06T00:00Z"),
    ("AMZN_2025_01_15", "AMZN", "2025-01-15T08:00Z", "2025-01-16T00:00Z"),
    ("INTC_2023_09_18", "INTC", "2023-09-18T08:00Z", "2023-09-19T00:00Z"),
)


def build_synthetic_minute_bars(
    *,
    symbol: str,
    start: datetime,
    minutes: int,
    price_base: float,
    price_step: float,
    volume_base: float,
    volume_step: float,
) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=minutes, freq="1min", tz="UTC")
    ts_values = index.astype("int64").to_numpy()
    increments = np.arange(minutes, dtype=np.float64)
    price_series = price_base + price_step * increments + 0.05 * np.sin(increments / 5.0)
    dataframe = pd.DataFrame(
        {
            "ts_event": ts_values,
            "ts_init": ts_values,
            "open": price_series,
            "high": price_series + 0.1,
            "low": price_series - 0.1,
            "close": price_series + 0.02,
            "volume": volume_base + volume_step * increments,
            "symbol": symbol,
        },
    )
    return dataframe


def _parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def ingest_and_canonicalize(
    *,
    service: DatabentoIngestionService,
    dataset: str,
    symbol: str,
    start: datetime,
    end: datetime,
    schema: str = DEFAULT_SCHEMA,
) -> CanonicalizedSlice:
    frames: list[pd.DataFrame] = []
    instrument_holder: dict[str, str] = {}

    def _collect(chunk: IngestionChunk) -> None:
        frame = chunk.frame
        if frame.empty:
            return
        if "instrument_id" in frame.columns:
            instrument = str(frame["instrument_id"].iloc[0])
            instrument_holder.setdefault("instrument_id", instrument)
        canonical = canonicalize_equities_minute_bars(
            frame,
            source_dataset=dataset,
            symbol=symbol,
            instrument_id=instrument_holder.get("instrument_id"),
        ).frame
        frames.append(canonical)

    request = IngestionRequest(
        dataset=dataset,
        schema=schema,
        symbols=(symbol,),
        start=start,
        end=end,
        chunk_days=7,
        allow_cost=True,
        reason="verify_eq_itch_parity",
    )
    service.ingest(request, on_chunk=_collect)
    dataframe = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    dataframe = dataframe.sort_values("ts_event").reset_index(drop=True)
    return CanonicalizedSlice(dataframe=dataframe, instrument_id=instrument_holder.get("instrument_id"))


def compute_feature_matrix(
    *,
    dataframe: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    engineer = FeatureEngineer(config or FeatureConfig())
    timestamps: list[int] = []
    feature_rows: list[np.ndarray] = []
    numeric = dataframe.copy()
    numeric["close"] = pd.to_numeric(numeric["close"], errors="coerce")
    numeric["high"] = pd.to_numeric(numeric["high"], errors="coerce")
    numeric["low"] = pd.to_numeric(numeric["low"], errors="coerce")
    numeric["volume"] = pd.to_numeric(numeric["volume"], errors="coerce")
    numeric["ts_event"] = pd.to_numeric(numeric["ts_event"], errors="coerce").astype("int64")
    for row in numeric.itertuples(index=False):
        features = engineer.calculate_features_online(
            close_price=float(cast(float, row.close)),
            high_price=float(cast(float, row.high)),
            low_price=float(cast(float, row.low)),
            volume=float(cast(float, row.volume)),
        )
        feature_rows.append(features.copy())
        timestamps.append(int(cast(int, row.ts_event)))
    if not feature_rows:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int64)
    matrix = np.vstack(feature_rows)
    ts_array = np.array(timestamps, dtype=np.int64)
    return matrix, ts_array


def _align_features(
    features_a: np.ndarray,
    ts_a: np.ndarray,
    features_b: np.ndarray,
    ts_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_a = {ts: features_a[i] for i, ts in enumerate(ts_a)}
    index_b = {ts: features_b[i] for i, ts in enumerate(ts_b)}
    common = sorted(set(index_a.keys()) & set(index_b.keys()))
    if not common:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.array([], dtype=np.int64),
        )
    aligned_a = np.vstack([index_a[ts] for ts in common])
    aligned_b = np.vstack([index_b[ts] for ts in common])
    return aligned_a, aligned_b, np.array(common, dtype=np.int64)


def compare_feature_sets(
    *,
    eq_slice: CanonicalizedSlice,
    fallback_slice: CanonicalizedSlice,
) -> dict[str, object]:
    eq_features, eq_ts = compute_feature_matrix(dataframe=eq_slice.dataframe)
    fb_features, fb_ts = compute_feature_matrix(dataframe=fallback_slice.dataframe)
    aligned_eq, aligned_fb, aligned_ts = _align_features(
        eq_features,
        eq_ts,
        fb_features,
        fb_ts,
    )

    eq_numeric = eq_slice.dataframe.copy()
    fb_numeric = fallback_slice.dataframe.copy()
    for frame in (eq_numeric, fb_numeric):
        for column in ("close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    aligned_prices = eq_numeric.merge(
        fb_numeric,
        on="ts_event",
        suffixes=("_eq", "_fb"),
    ).dropna(subset=["close_eq", "close_fb", "volume_eq", "volume_fb"])

    if aligned_eq.size == 0 or aligned_prices.empty:
        return {
            "timestamp_count": 0.0,
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
            "p99_abs_diff": float("nan"),
            "worst_timestamp_ns": None,
            "worst_timestamp_iso": None,
            "price_close_max_abs_diff": float("nan"),
            "price_close_p99_abs_diff": float("nan"),
            "price_close_mean_abs_diff": float("nan"),
            "price_close_correlation": None,
            "volume_correlation": None,
            "volume_ratio_stats": None,
            "volume_residual_abs": None,
            "volume_residual_rel": None,
        }
    diff = np.abs(aligned_eq - aligned_fb)
    return {
        **_compute_feature_metrics(diff=diff, timestamps=aligned_ts),
        **_compute_price_volume_metrics(aligned_prices=aligned_prices),
    }


def _compute_feature_metrics(
    *,
    diff: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, object]:
    if diff.size == 0:
        return {
            "timestamp_count": 0.0,
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
            "p99_abs_diff": float("nan"),
            "worst_timestamp_ns": None,
            "worst_timestamp_iso": None,
        }

    flattened = diff.reshape(-1)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())
    p99_abs_diff = float(np.percentile(flattened, 99))

    per_row_max = diff.max(axis=1)
    worst_index = int(np.argmax(per_row_max))
    worst_timestamp_ns = int(timestamps[worst_index]) if timestamps.size > 0 else None
    worst_timestamp_iso = (
        datetime.fromtimestamp(worst_timestamp_ns / 1_000_000_000, tz=UTC).isoformat()
        if worst_timestamp_ns is not None
        else None
    )

    return {
        "timestamp_count": float(diff.shape[0]),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "p99_abs_diff": p99_abs_diff,
        "worst_timestamp_ns": worst_timestamp_ns,
        "worst_timestamp_iso": worst_timestamp_iso,
    }


def _compute_price_volume_metrics(
    *,
    aligned_prices: pd.DataFrame,
) -> dict[str, object]:
    price_diff = (aligned_prices["close_eq"] - aligned_prices["close_fb"]).abs()
    price_close_max_abs = float(price_diff.max())
    price_close_p99 = float(price_diff.quantile(0.99))
    price_close_mean = float(price_diff.mean())

    price_corr = aligned_prices["close_eq"].corr(aligned_prices["close_fb"])
    volume_corr = aligned_prices["volume_eq"].corr(aligned_prices["volume_fb"])

    fallback_volume = aligned_prices["volume_fb"].replace(0, np.nan)
    volume_ratio = aligned_prices["volume_eq"] / fallback_volume
    volume_ratio_clean = volume_ratio.dropna()

    if volume_ratio_clean.empty:
        ratio_stats: Mapping[str, float] | None = None
    else:
        ratio_stats = {
            "min": float(volume_ratio_clean.min()),
            "max": float(volume_ratio_clean.max()),
            "median": float(volume_ratio_clean.median()),
            "p05": float(volume_ratio_clean.quantile(0.05)),
            "p95": float(volume_ratio_clean.quantile(0.95)),
        }

    volume_residual_abs = float(
        abs(aligned_prices["volume_eq"].sum() - aligned_prices["volume_fb"].sum()),
    )
    volume_residual_rel = (
        volume_residual_abs / float(aligned_prices["volume_eq"].sum())
        if aligned_prices["volume_eq"].sum() > 0
        else None
    )

    return {
        "price_close_max_abs_diff": price_close_max_abs,
        "price_close_p99_abs_diff": price_close_p99,
        "price_close_mean_abs_diff": price_close_mean,
        "price_close_correlation": float(price_corr) if not np.isnan(price_corr) else None,
        "volume_correlation": float(volume_corr) if not np.isnan(volume_corr) else None,
        "volume_ratio_stats": ratio_stats,
        "volume_residual_abs": volume_residual_abs,
        "volume_residual_rel": volume_residual_rel,
    }


def default_parity_suite() -> tuple[ParityScenario, ...]:
    """Return the built-in Tier-1 parity suite covering 2023–2025 windows."""
    scenarios: list[ParityScenario] = []
    for label, symbol, start_str, end_str in _DEFAULT_SUITE_ROWS:
        start_dt = _parse_datetime(start_str)
        end_dt = _parse_datetime(end_str)
        scenario = ParityScenario(
            label=label,
            eq_symbol=symbol,
            fallback_symbol=f"{symbol}.XNAS",
            fallback_dataset="XNAS.ITCH",
            start=start_dt,
            end=end_dt,
        )
        scenarios.append(scenario)
    return tuple(scenarios)


def load_parity_suite(path: Path) -> tuple[ParityScenario, ...]:
    """Load a parity suite definition from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios_raw = payload.get("scenarios")
    if not isinstance(scenarios_raw, Sequence):
        raise ValueError("suite config must define a 'scenarios' array")

    scenarios: list[ParityScenario] = []
    for entry in scenarios_raw:
        if not isinstance(entry, dict):
            raise ValueError("suite scenario entries must be objects")
        try:
            eq_symbol_raw = entry["eq_symbol"]
            start_raw = entry["start"]
            end_raw = entry["end"]
        except KeyError as missing:
            raise ValueError(f"scenario missing required field: {missing}") from missing
        if not isinstance(eq_symbol_raw, str):
            raise ValueError("eq_symbol must be a string")
        if not isinstance(start_raw, str) or not isinstance(end_raw, str):
            raise ValueError("start/end must be ISO-8601 strings")
        label = entry.get("label")
        label_value = label if isinstance(label, str) and label else f"{eq_symbol_raw}_{start_raw}"
        fallback_symbol_raw = entry.get("fallback_symbol")
        fallback_symbol = (
            fallback_symbol_raw
            if isinstance(fallback_symbol_raw, str) and fallback_symbol_raw
            else f"{eq_symbol_raw}.XNAS"
        )
        fallback_dataset_raw = entry.get("fallback_dataset")
        fallback_dataset = (
            fallback_dataset_raw
            if isinstance(fallback_dataset_raw, str) and fallback_dataset_raw
            else "XNAS.ITCH"
        )
        schema_raw = entry.get("schema")
        schema = schema_raw if isinstance(schema_raw, str) and schema_raw else DEFAULT_SCHEMA
        scenario = ParityScenario(
            label=label_value,
            eq_symbol=eq_symbol_raw,
            fallback_symbol=fallback_symbol,
            fallback_dataset=fallback_dataset,
            start=_parse_datetime(start_raw),
            end=_parse_datetime(end_raw),
            schema=schema,
        )
        scenarios.append(scenario)
    return tuple(scenarios)


def run_parity_suite(
    *,
    service: DatabentoIngestionService,
    scenarios: Sequence[ParityScenario],
    ingest: Callable[..., CanonicalizedSlice] | None = None,
) -> ParitySuiteReport:
    """Execute the parity suite and capture metrics for every scenario."""
    handler = ingest or ingest_and_canonicalize
    results: list[ParityScenarioResult] = []
    for scenario in scenarios:
        eq_slice = handler(
            service=service,
            dataset="EQUS.MINI",
            symbol=scenario.eq_symbol,
            start=scenario.start,
            end=scenario.end,
            schema=scenario.schema,
        )
        fallback_slice = handler(
            service=service,
            dataset=scenario.fallback_dataset,
            symbol=scenario.fallback_symbol,
            start=scenario.start,
            end=scenario.end,
            schema=scenario.schema,
        )
        fallback_df = fallback_slice.dataframe.copy(deep=True)
        fallback_df["symbol"] = scenario.eq_symbol
        normalized_fallback = CanonicalizedSlice(
            dataframe=fallback_df,
            instrument_id=fallback_slice.instrument_id,
        )
        metrics_raw = compare_feature_sets(
            eq_slice=eq_slice,
            fallback_slice=normalized_fallback,
        )
        metrics = ParityMetrics(
            timestamp_count=float(cast(float, metrics_raw["timestamp_count"])),
            max_abs_diff=float(cast(float, metrics_raw["max_abs_diff"])),
            mean_abs_diff=float(cast(float, metrics_raw["mean_abs_diff"])),
            p99_abs_diff=float(cast(float, metrics_raw["p99_abs_diff"])),
            worst_timestamp_ns=cast(int | None, metrics_raw["worst_timestamp_ns"]),
            worst_timestamp_iso=cast(str | None, metrics_raw["worst_timestamp_iso"]),
            price_close_max_abs_diff=cast(float | None, metrics_raw["price_close_max_abs_diff"]),
            price_close_p99_abs_diff=cast(float | None, metrics_raw["price_close_p99_abs_diff"]),
            price_close_mean_abs_diff=cast(float | None, metrics_raw["price_close_mean_abs_diff"]),
            price_close_correlation=cast(float | None, metrics_raw["price_close_correlation"]),
            volume_correlation=cast(float | None, metrics_raw["volume_correlation"]),
            volume_ratio_stats=cast(Mapping[str, float] | None, metrics_raw["volume_ratio_stats"]),
            volume_residual_abs=cast(float | None, metrics_raw["volume_residual_abs"]),
            volume_residual_rel=cast(float | None, metrics_raw["volume_residual_rel"]),
        )
        results.append(ParityScenarioResult(scenario=scenario, metrics=metrics))
    return ParitySuiteReport(
        generated_at=datetime.now(tz=UTC),
        results=tuple(results),
    )


def serialize_parity_report(report: ParitySuiteReport) -> dict[str, Any]:
    """Convert a parity report into a JSON-serialisable mapping."""
    return {
        "generated_at": report.generated_at.isoformat(),
        "scenarios": [
            {
                "label": result.scenario.label,
                "eq_symbol": result.scenario.eq_symbol,
                "fallback_symbol": result.scenario.fallback_symbol,
                "fallback_dataset": result.scenario.fallback_dataset,
                "start": result.scenario.start.isoformat(),
                "end": result.scenario.end.isoformat(),
                "schema": result.scenario.schema,
                "metrics": {
                    "timestamp_count": result.metrics.timestamp_count,
                    "max_abs_diff": result.metrics.max_abs_diff,
                    "mean_abs_diff": result.metrics.mean_abs_diff,
                    "p99_abs_diff": result.metrics.p99_abs_diff,
                    "worst_timestamp_ns": result.metrics.worst_timestamp_ns,
                    "worst_timestamp_iso": result.metrics.worst_timestamp_iso,
                    "price_close_max_abs_diff": result.metrics.price_close_max_abs_diff,
                    "price_close_p99_abs_diff": result.metrics.price_close_p99_abs_diff,
                    "price_close_mean_abs_diff": result.metrics.price_close_mean_abs_diff,
                    "price_close_correlation": result.metrics.price_close_correlation,
                    "volume_correlation": result.metrics.volume_correlation,
                    "volume_ratio_stats": result.metrics.volume_ratio_stats,
                    "volume_residual_abs": result.metrics.volume_residual_abs,
                    "volume_residual_rel": result.metrics.volume_residual_rel,
                },
            }
            for result in report.results
        ],
    }


def backfill_equs_ohlcv(
    *,
    service: DatabentoIngestionService,
    symbol: str,
    start: datetime,
    end: datetime,
    writer: SqlMarketDataWriter | None,
    output_path: Path | None,
    schema: str = DEFAULT_SCHEMA,
) -> None:
    frames_written = 0
    output_collector: list[pd.DataFrame] = []

    def _persist(chunk: IngestionChunk) -> None:
        nonlocal frames_written
        frame = chunk.frame
        if frame.empty:
            return
        instrument = str(frame.get("instrument_id", "")) if "instrument_id" in frame.columns else None
        canonical = canonicalize_equities_minute_bars(
            frame,
            source_dataset="EQUS.MINI",
            symbol=symbol,
            instrument_id=instrument,
        ).frame
        canonical = canonical.sort_values("ts_event").reset_index(drop=True)
        if writer and instrument:
            writer.write(
                dataset_id="EQUS.MINI",
                schema=schema,
                instrument_id=instrument,
                df=canonical,
            )
        else:
            output_collector.append(canonical)
        frames_written += 1

    request = IngestionRequest(
        dataset="EQUS.MINI",
        schema=schema,
        symbols=(symbol,),
        start=start,
        end=end,
        chunk_days=7,
        allow_cost=True,
        reason="backfill_eq_us_pre2023",
    )
    service.ingest(request, on_chunk=_persist)
    if output_collector and output_path is not None:
        combined = pd.concat(output_collector, ignore_index=True).sort_values("ts_event")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path)
    if frames_written == 0:
        raise RuntimeError("No data ingested for the requested window.")


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Verify EQUS normalization parity and optional backfill.")
    parser.add_argument("--eq-symbol", default="INTC", help="EQUS symbol (default: INTC)")
    parser.add_argument(
        "--fallback-symbol",
        default="INTC.XNAS",
        help="Fallback dataset symbol (with venue when required).",
    )
    parser.add_argument("--fallback-dataset", default="XNAS.ITCH", help="Fallback dataset id")
    parser.add_argument("--start", help="ISO8601 start time (inclusive)")
    parser.add_argument("--end", help="ISO8601 end time (exclusive)")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("ml/tests/validation_reports/eq_itch_feature_parity.json"),
        help="Path to write parity summary JSON.",
    )
    parser.add_argument(
        "--backfill-start",
        help="ISO8601 start for pre-2023 backfill (optional)",
    )
    parser.add_argument(
        "--backfill-end",
        help="ISO8601 end for pre-2023 backfill (optional)",
    )
    parser.add_argument(
        "--sql-connection",
        help="Optional SQLAlchemy connection string for SqlMarketDataWriter.",
    )
    parser.add_argument(
        "--parquet-output",
        type=Path,
        help="Optional Parquet file to store backfilled data when no SQL writer provided.",
    )
    parser.add_argument(
        "--market-inputs",
        help="Optional JSON array of MarketDatasetInput overrides for discovery scenarios.",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run the built-in Tier-1 parity suite (2023-2025 symbols).",
    )
    parser.add_argument(
        "--suite-config",
        type=Path,
        help="Path to a JSON file defining custom parity scenarios.",
    )
    parser.add_argument(
        "--suite-output",
        type=Path,
        default=DEFAULT_SUITE_OUTPUT,
        help=(
            "Output path for suite summary JSON (default: "
            "ml/tests/validation_reports/equs_itch_parity_summary.json)."
        ),
    )
    parser.add_argument(
        "--calibration-path",
        type=Path,
        help="Optional override for the calibration bundle path.",
    )
    parser.add_argument(
        "--max-calibration-age-days",
        type=int,
        default=45,
        help="Maximum allowed calibration age in days (default: 45).",
    )
    parser.add_argument(
        "--allow-missing-calibration",
        action="store_true",
        help="Continue even if the calibration bundle is missing or stale.",
    )
    args = parser.parse_args()

    if args.suite and args.suite_config is not None:
        parser.error("--suite and --suite-config are mutually exclusive")

    service = DatabentoIngestionService.from_env()
    ensure_calibration_fresh(
        calibration_path=args.calibration_path,
        max_age_days=args.max_calibration_age_days,
        allow_missing=args.allow_missing_calibration,
    )

    if args.suite or args.suite_config is not None:
        scenarios = (
            load_parity_suite(args.suite_config)
            if args.suite_config is not None
            else default_parity_suite()
        )
        report = run_parity_suite(service=service, scenarios=scenarios)
        output_path = args.suite_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = serialize_parity_report(report)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    if args.start is None or args.end is None:
        parser.error("--start and --end are required when not running a parity suite")

    start_dt = _parse_datetime(args.start)
    end_dt = _parse_datetime(args.end)

    parity_eq = ingest_and_canonicalize(
        service=service,
        dataset="EQUS.MINI",
        symbol=args.eq_symbol,
        start=start_dt,
        end=end_dt,
    )
    parity_fb = ingest_and_canonicalize(
        service=service,
        dataset=args.fallback_dataset,
        symbol=args.fallback_symbol,
        start=start_dt,
        end=end_dt,
    )

    parity_fb.dataframe["symbol"] = args.eq_symbol
    parity_summary = compare_feature_sets(eq_slice=parity_eq, fallback_slice=parity_fb)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(parity_summary, indent=2), encoding="utf-8")
    print(json.dumps(parity_summary, indent=2))

    if args.backfill_start and args.backfill_end:
        backfill_start = _parse_datetime(args.backfill_start)
        backfill_end = _parse_datetime(args.backfill_end)
        writer: SqlMarketDataWriter | None = None
        if args.sql_connection:
            writer = SqlMarketDataWriter(connection_string=args.sql_connection)
        elif args.parquet_output is None:
            raise ValueError("Provide --sql-connection or --parquet-output when running backfill.")
        backfill_equs_ohlcv(
            service=service,
            symbol=args.eq_symbol,
            start=backfill_start,
            end=backfill_end,
            writer=writer,
            output_path=args.parquet_output,
        )


if __name__ == "__main__":
    run_cli()
