"""
Earnings feature join helpers for dataset building.

This module centralizes earnings feature construction so both legacy and
facade builders share one implementation.
"""

from __future__ import annotations

import logging
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING, Any, SupportsFloat, cast

import numpy as np

from ml._imports import pl as pl_runtime
from ml.data.common.time_series_windowing import TimeSeriesWindowingComponent
from ml.stores.protocols import DataStoreFacadeProtocol


if TYPE_CHECKING:
    import polars as _pl
else:  # pragma: no cover - typing fallback
    _pl = Any


pl: Any = cast(Any, pl_runtime)

logger = logging.getLogger(__name__)


def fetch_earnings_features(
    *,
    data_store: DataStoreFacadeProtocol,
    ticker: str,
    timestamps: _pl.Series,
    earnings_lag_days: int,
    as_of_date: datetime | None = None,
) -> _pl.DataFrame | None:
    """
    Fetch earnings-derived features for the provided timestamps.

    Args:
        data_store: DataStore facade providing earnings accessors.
        ticker: Instrument ticker (venue suffix optional).
        timestamps: Polars Series of timestamps for the dataset.
        earnings_lag_days: Publication lag to apply for earnings features.
        as_of_date: Optional point-in-time cutoff for estimates.

    Returns:
        Polars DataFrame with earnings features aligned to timestamps, or ``None``
        when data is unavailable.

    Example:
        >>> pl = __import__("polars")
        >>> series = pl.Series("timestamp", [datetime(2024, 1, 1, tzinfo=UTC)])
        >>> _ = fetch_earnings_features(
        ...     data_store=cast(DataStoreFacadeProtocol, object()),
        ...     ticker="AAPL",
        ...     timestamps=series,
        ...     earnings_lag_days=1,
        ... )
    """
    if pl is None:
        logger.debug("Polars unavailable; skipping earnings feature join")
        return None
    if len(timestamps) == 0:
        return None

    base_ticker = ticker.split(".")[0]

    as_of_ts = TimeSeriesWindowingComponent.datetime_to_ns(
        datetime.now(tz=UTC),
        fallback=0,
    )
    ts_max_ns = TimeSeriesWindowingComponent.coerce_to_ns(timestamps.max())
    if ts_max_ns is not None:
        as_of_ts = ts_max_ns
    if as_of_date is not None:
        as_of_ts = min(
            as_of_ts,
            TimeSeriesWindowingComponent.datetime_to_ns(as_of_date, fallback=as_of_ts),
        )

    actuals_desc = data_store.get_earnings_actuals_at_or_before(
        ticker=base_ticker,
        ts_event=as_of_ts,
        limit=6,
    )
    if not actuals_desc:
        logger.debug("No earnings actuals for %s", base_ticker)
        return None

    actuals = list(reversed(actuals_desc))

    def _to_float(value: object) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        try:
            return float(cast(SupportsFloat, value))
        except (TypeError, ValueError):
            return 0.0

    def _to_iso(value: object) -> str:
        if value is None:
            return ""
        iso_fun = getattr(value, "isoformat", None)
        if callable(iso_fun):
            iso_callable = cast(Any, iso_fun)
            try:
                iso_val = iso_callable()
                if isinstance(iso_val, str):
                    return iso_val
            except Exception:
                pass
        return str(value)

    eps_series = np.array(
        [_to_float(record.get("eps_diluted")) for record in actuals],
        dtype=np.float64,
    )
    if eps_series.size == 0:
        return None

    consensus_series: list[float] = []
    for record in actuals:
        period_end = _to_iso(record.get("period_end"))
        ts_event = int(record.get("ts_event") or as_of_ts)
        estimate = data_store.get_earnings_estimate_at_or_before(
            ticker=base_ticker,
            period_end=period_end,
            ts_event=ts_event,
        )
        consensus_series.append(
            _to_float(
                estimate.get("eps_consensus") if estimate is not None else record.get("eps_diluted"),
            ),
        )

    estimates = np.array(consensus_series, dtype=np.float64)

    from ml.features.earnings.earnings_features import compute_earnings_growth_batch
    from ml.features.earnings.earnings_features import compute_earnings_momentum_batch
    from ml.features.earnings.earnings_features import compute_earnings_surprise_batch

    surprise = compute_earnings_surprise_batch(eps_series, estimates)
    momentum = compute_earnings_momentum_batch(surprise["eps_surprise_q0"], eps_series)
    growth = compute_earnings_growth_batch(eps_series)

    column_prefix = base_ticker
    quarterly_df = pl.DataFrame(
        {
            "period_end": [_to_iso(record.get("period_end")) for record in actuals],
            "filing_date": [_to_iso(record.get("filing_date")) for record in actuals],
            f"eps_surprise_q0_{column_prefix}": surprise["eps_surprise_q0"],
            f"eps_surprise_pct_q0_{column_prefix}": surprise["eps_surprise_pct_q0"],
            f"eps_growth_yoy_{column_prefix}": growth["eps_growth_yoy"],
            f"eps_growth_qoq_{column_prefix}": growth["eps_growth_qoq"],
            f"earnings_beat_streak_{column_prefix}": momentum["earnings_beat_streak"],
            f"eps_volatility_4q_{column_prefix}": momentum["eps_volatility_4q"],
        },
    )

    quarterly_df = quarterly_df.with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
    ).with_columns(
        pl.col("filing_date").cast(pl.Datetime("ns", "UTC")),
    ).with_columns(
        (pl.col("filing_date") + pl.duration(days=earnings_lag_days)).alias("effective_date"),
    ).with_columns(
        pl.col("effective_date").cast(pl.Datetime("ns", "UTC")),
    )

    bar_df = pl.DataFrame({"timestamp": timestamps}).sort("timestamp")
    if bar_df["timestamp"].dtype != pl.Datetime:
        bar_df = bar_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
        )
    else:
        bar_df = bar_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
        )

    earnings_joined_any = bar_df.join_asof(
        quarterly_df.sort("effective_date"),
        left_on="timestamp",
        right_on="effective_date",
        strategy="backward",
    )
    earnings_joined = cast("_pl.DataFrame", earnings_joined_any)
    earnings_joined = earnings_joined.drop(
        ["period_end", "filing_date", "effective_date"],
        strict=False,
    )

    feature_cols = [col for col in earnings_joined.columns if col != "timestamp"]
    if not feature_cols:
        return None

    fills: list[Any] = []
    availability: list[Any] = []
    for col in feature_cols:
        try:
            if earnings_joined.schema[col].is_numeric():
                fills.append(pl.col(col).fill_null(0))
        except Exception:
            pass
        availability.append(pl.col(col).is_not_null())

    if fills:
        earnings_joined = earnings_joined.with_columns(fills)

    has_any = None
    for expr in availability:
        has_any = expr if has_any is None else (has_any | expr)
    if has_any is not None:
        earnings_joined = earnings_joined.with_columns(
            has_any.cast(pl.Int32).alias("is_earnings_available"),
        )
    else:
        earnings_joined = earnings_joined.with_columns(
            pl.lit(0).alias("is_earnings_available"),
        )

    logger.debug(
        "joined_earnings_features",
        extra={"ticker": base_ticker, "rows": earnings_joined.height},
    )

    return earnings_joined
