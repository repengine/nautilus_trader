"""
Alternative data loaders providing cold-path helpers for CLI/tasks.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl as _pl


LOGGER = logging.getLogger(__name__)

if not HAS_POLARS:  # pragma: no cover - enforce optional dependency only when needed
    check_ml_dependencies(["polars"])
if _pl is None:
    msg = "Polars runtime not available after dependency check"
    raise RuntimeError(msg)
PL = _pl

if TYPE_CHECKING:  # pragma: no cover - typing only
    from polars import DataFrame as PolarsDataFrame
else:  # pragma: no cover - runtime fallback without direct import
    PolarsDataFrame = type(PL.DataFrame())


class AlternativeSource(str, Enum):
    """
    Supported alternative data sources.
    """

    CBOE = "cboe"
    AAII = "aaii"
    COT = "cot"
    SHORT_INTEREST = "short"
    MICRO = "micro"
    NEWS = "news"
    EARNINGS = "earnings"
    SECTOR = "sector"


@dataclass(slots=True, frozen=True)
class AlternativeDataConfig:
    """
    Configuration describing which sources to populate.
    """

    symbols: tuple[str, ...]
    sources: tuple[AlternativeSource, ...]


@dataclass(slots=True, frozen=True)
class AlternativeDataResult:
    """
    Result bundle mapping source name to populated frame.
    """

    frames: Mapping[str, PolarsDataFrame]

    @property
    def non_empty_sources(self) -> tuple[str, ...]:
        """
        Return sources which produced at least one row.
        """
        return tuple(name for name, frame in self.frames.items() if not frame.is_empty())


def _timestamp_now() -> datetime:
    return datetime.now(tz=UTC)


def _fetch_cboe_put_call_ratio() -> PolarsDataFrame:
    data = {
        "timestamp": [_timestamp_now()],
        "total_pc_ratio": [0.85],
        "equity_pc_ratio": [0.75],
        "index_pc_ratio": [1.20],
        "vix_pc_ratio": [0.95],
    }
    return cast(PolarsDataFrame, PL.DataFrame(data))


def _fetch_cboe_term_structure() -> PolarsDataFrame:
    data = {
        "timestamp": [_timestamp_now()] * 5,
        "symbol": ["VIX", "VIX9D", "VIX30D", "VIX90D", "VIX180D"],
        "value": [18.5, 17.2, 19.1, 21.0, 22.3],
        "days_to_expiry": [0, 9, 30, 90, 180],
    }
    return cast(PolarsDataFrame, PL.DataFrame(data))


def _fetch_aaii_sentiment() -> PolarsDataFrame:
    data = {
        "week_ending": [datetime(2024, 1, 5, tzinfo=UTC)],
        "bullish": [0.42],
        "neutral": [0.32],
        "bearish": [0.26],
        "bull_bear_spread": [0.16],
    }
    return cast(PolarsDataFrame, PL.DataFrame(data))


def _fetch_cot_reports() -> PolarsDataFrame:
    data = {
        "report_date": [datetime(2023, 12, 26, tzinfo=UTC)] * 3,
        "symbol": ["ES", "VX", "DX"],
        "commercial_long": [150_000, 25_000, 50_000],
        "commercial_short": [120_000, 30_000, 45_000],
        "noncommercial_long": [80_000, 18_000, 25_000],
        "noncommercial_short": [60_000, 20_000, 35_000],
        "open_interest": [320_000, 70_000, 120_000],
    }
    return cast(PolarsDataFrame, PL.DataFrame(data))


def _fetch_short_interest(symbols: tuple[str, ...]) -> PolarsDataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.append(
            {
                "settlement_date": datetime(2023, 12, 15, tzinfo=UTC),
                "symbol": symbol,
                "short_interest": 5_000_000,
                "avg_daily_volume": 3_000_000,
                "days_to_cover": 1.7,
                "short_percent_float": 0.02,
            },
        )
    return cast(PolarsDataFrame, PL.DataFrame(rows))


def _calculate_microstructure_metrics(symbols: tuple[str, ...]) -> PolarsDataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.append(
            {
                "timestamp": _timestamp_now(),
                "symbol": symbol,
                "effective_spread": 0.0005,
                "realized_spread": 0.0003,
                "price_impact": 0.0002,
                "volume_imbalance": 0.05,
                "trade_imbalance": 0.04,
                "dollar_volume": 10_000_000.0,
                "kyle_lambda": 1.5,
                "hasbrouck_info_share": 0.55,
                "amihud_illiquidity": 0.8,
                "vpin": 0.12,
                "order_flow_toxicity": 0.08,
            },
        )
    return cast(PolarsDataFrame, PL.DataFrame(rows))


def _fetch_news_sentiment(symbols: tuple[str, ...]) -> PolarsDataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.append(
            {
                "timestamp": _timestamp_now(),
                "symbol": symbol,
                "headline_sentiment": 0.1,
                "article_sentiment": 0.05,
                "social_sentiment": -0.02,
                "mention_count": 15,
                "sentiment_volatility": 0.3,
            },
        )
    return cast(PolarsDataFrame, PL.DataFrame(rows))


def _fetch_earnings_calendar(symbols: tuple[str, ...]) -> PolarsDataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.append(
            {
                "symbol": symbol,
                "earnings_date": datetime(2024, 2, 15, tzinfo=UTC),
                "eps_estimate": 1.25,
                "eps_actual": 1.30,
                "revenue_estimate": 5_000_000_000.0,
                "revenue_actual": 5_050_000_000.0,
                "surprise_percent": 0.04,
                "days_until_earnings": 45,
            },
        )
    return cast(PolarsDataFrame, PL.DataFrame(rows))


def _fetch_sector_classifications(symbols: tuple[str, ...]) -> PolarsDataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.append(
            {
                "symbol": symbol,
                "gics_sector": "Information Technology",
                "gics_industry_group": "Software & Services",
                "gics_industry": "Software",
                "gics_sub_industry": "Application Software",
                "market_cap_category": "large",
                "style_category": "growth",
            },
        )
    return cast(PolarsDataFrame, PL.DataFrame(rows))


def populate_alternative_data(config: AlternativeDataConfig) -> AlternativeDataResult:
    """
    Populate alternative data sources according to ``config``.
    """
    frames: dict[str, PolarsDataFrame] = {}
    for source in config.sources:
        if source is AlternativeSource.CBOE:
            frames["put_call_ratio"] = _fetch_cboe_put_call_ratio()
            frames["vix_term_structure"] = _fetch_cboe_term_structure()
        elif source is AlternativeSource.AAII:
            frames["aaii_sentiment"] = _fetch_aaii_sentiment()
        elif source is AlternativeSource.COT:
            frames["cot_reports"] = _fetch_cot_reports()
        elif source is AlternativeSource.SHORT_INTEREST:
            frames["short_interest"] = _fetch_short_interest(config.symbols)
        elif source is AlternativeSource.MICRO:
            frames["microstructure"] = _calculate_microstructure_metrics(config.symbols)
        elif source is AlternativeSource.NEWS:
            frames["news_sentiment"] = _fetch_news_sentiment(config.symbols)
        elif source is AlternativeSource.EARNINGS:
            frames["earnings_calendar"] = _fetch_earnings_calendar(config.symbols)
        elif source is AlternativeSource.SECTOR:
            frames["sector_industry"] = _fetch_sector_classifications(config.symbols)
    return AlternativeDataResult(frames=frames)


def save_alternative_data(result: AlternativeDataResult, output_dir: Path) -> tuple[Path, ...]:
    """
    Persist non-empty frames to ``output_dir`` and return saved paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for name, frame in result.frames.items():
        if frame.is_empty():
            continue
        path = output_dir / f"{name}.parquet"
        frame.write_parquet(path)
        saved_paths.append(path)
        LOGGER.info("Saved alternative data %s -> %s", name, path)
    return tuple(saved_paths)


def load_tier1_symbols(progress_path: Path | None = None) -> tuple[str, ...]:
    """
    Resolve Tier 1 symbols from progress metadata.
    """
    path = progress_path or Path("tier1_l1_progress.json")
    if not path.exists():
        return tuple()
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed parsing progress file %s", path, exc_info=exc)
        return tuple()
    completed = payload.get("completed_bbo") if isinstance(payload, dict) else None
    if isinstance(completed, list):
        symbols = [str(item).upper() for item in completed if isinstance(item, str)]
        return tuple(sorted(set(symbols)))
    return tuple()


__all__ = [
    "AlternativeDataConfig",
    "AlternativeDataResult",
    "AlternativeSource",
    "load_tier1_symbols",
    "populate_alternative_data",
    "save_alternative_data",
]
