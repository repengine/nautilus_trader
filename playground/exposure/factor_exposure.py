"""Pipeline utilities for computing factor exposures via EWMA beta."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime

import polars as pl

from ml.features.cross_asset.beta import compute_ewma_beta_incremental
from ml.features.cross_asset.state import DEFAULT_ALPHA
from ml.features.cross_asset.state import EWMABetaState


@dataclass(slots=True)
class FactorExposureConfig:
    """Configuration for factor exposure computation."""

    feature_set_id: str
    alpha: float = DEFAULT_ALPHA
    source: str = "historical"
    ts_init: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


def prepare_factor_returns(
    factor_features: pl.DataFrame,
    *,
    columns: Iterable[str],
    method: str = "difference",
    winsorize_percentile: float | None = 0.99,
) -> pl.DataFrame:
    """
    Compute factor returns with robust handling of edge cases.

    Parameters
    ----------
    factor_features
        DataFrame with timestamp and factor level columns.
    columns
        Factor column names to convert to returns.
    method
        Return computation method: "difference" (additive) or "pct_change"
        (multiplicative). Use "difference" for spreads and indices that can
        cross zero to avoid division-by-zero infinities. Default: "difference".
    winsorize_percentile
        Cap extreme returns at this percentile to prevent outliers from
        dominating EWMA calculations (None to disable). Default: 0.99.

    Returns
    -------
    pl.DataFrame
        Factor returns with inf/nan values handled and outliers winsorized.

    Notes
    -----
    For yield spreads (e.g., 10Y-2Y) and financial stress indices that can
    cross zero, additive returns (difference) are more appropriate than
    multiplicative returns (pct_change), which produce infinities when the
    previous value is zero.
    """
    column_list = list(columns)
    selected = factor_features.sort("timestamp").select(["timestamp", *column_list])

    if method == "difference":
        # Additive returns: appropriate for spreads and indices
        returns = selected.with_columns(
            [pl.col(col).diff().alias(col) for col in column_list],
        )
    elif method == "pct_change":
        # Multiplicative returns: traditional for prices
        returns = selected.with_columns(
            [pl.col(col).pct_change().alias(col) for col in column_list],
        )
    else:
        msg = f"method must be 'difference' or 'pct_change', got {method}"
        raise ValueError(msg)

    # Replace inf/-inf with large finite values to prevent NaN propagation
    for col in column_list:
        returns = returns.with_columns(
            pl.when(pl.col(col).is_infinite())
            .then(pl.when(pl.col(col) > 0).then(pl.lit(10.0)).otherwise(pl.lit(-10.0)))
            .otherwise(pl.col(col))
            .alias(col),
        )

    # Winsorize extreme values to prevent outliers from dominating EWMA
    if winsorize_percentile is not None:
        for col in column_list:
            non_null = returns.filter(pl.col(col).is_not_null() & pl.col(col).is_finite())
            if non_null.height > 10:
                lower = non_null[col].quantile(1 - winsorize_percentile)
                upper = non_null[col].quantile(winsorize_percentile)
                returns = returns.with_columns(
                    pl.col(col).clip(lower, upper).alias(col),
                )

    return returns.drop_nulls(subset=column_list)


def compute_factor_exposures(
    asset_returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
    config: FactorExposureConfig,
) -> pl.DataFrame:
    """Compute EWMA betas for each asset against each factor column."""
    joined = (
        asset_returns
        .select(["timestamp", "symbol", "return"])  # ensure expected columns
        .join(factor_returns, on="timestamp", how="inner")
        .drop_nulls()
        .sort(["symbol", "timestamp"])
    )

    factor_columns = [col for col in joined.columns if col not in {"timestamp", "symbol", "return"}]
    if not factor_columns:
        raise ValueError("No factor columns available for exposure computation")

    ts_init_ns = _to_nanoseconds(config.ts_init)
    records: list[dict[str, object]] = []

    for group in joined.partition_by("symbol", maintain_order=True):
        symbol = group["symbol"][0]
        timestamps = group["timestamp"].to_list()
        asset_series = group["return"].to_list()

        for factor_name in factor_columns:
            factor_series = group[factor_name].to_list()
            state = EWMABetaState(alpha=config.alpha)

            for ts, asset_ret, factor_ret in zip(timestamps, asset_series, factor_series):
                beta = compute_ewma_beta_incremental(
                    state,
                    float(asset_ret),
                    float(factor_ret),
                )
                records.append(
                    {
                        "feature_set_id": config.feature_set_id,
                        "asset_id": str(symbol),
                        "benchmark_id": str(factor_name),
                        "ts_event": _to_nanoseconds(ts),
                        "ts_init": ts_init_ns,
                        "ewma_beta": beta,
                        "ewma_cov": state.ewma_cov,
                        "ewma_var_market": state.ewma_var_market,
                        "n_observations": state.n,
                        "alpha": state.alpha,
                        "source": config.source,
                    },
                )

    return pl.DataFrame(records)


def compute_stable_sector_positions(
    exposures: pl.DataFrame,
    *,
    factor_columns: Iterable[str],
    aggregation: str = "median",
    min_observations: int = 100,
) -> dict[str, dict[str, float]]:
    """
    Compute long-term stable positions for each sector across all time.

    Parameters
    ----------
    exposures
        Long-form EWMA beta DataFrame with columns: asset_id, benchmark_id,
        ts_event, ewma_beta (from compute_factor_exposures).
    factor_columns
        Factor names (e.g., "factor_duration", "factor_credit", "factor_liquidity").
    aggregation
        Method to compute stable center: "median" (default), "mean", "trimmed_mean".
    min_observations
        Minimum number of observations required per sector to compute stable position.

    Returns
    -------
    dict[str, dict[str, float]]
        Stable positions keyed by sector ID, e.g.:
        {
            "XLK": {"factor_duration": -0.12, "factor_credit": -0.20, ...},
            "XLU": {"factor_duration": 0.25, "factor_credit": 0.05, ...},
        }

    Raises
    ------
    ValueError
        If any sector has fewer than min_observations, if aggregation method is
        invalid, if exposures is empty, or if required columns are missing.

    Notes
    -----
    The stable position represents the long-term "home" of each sector in
    factor space. Sectors should remain in small "clouds" around these centers
    with year-to-year variations, rather than the coordinate system shifting.
    """
    # Validate aggregation method
    valid_methods = {"median", "mean", "trimmed_mean"}
    if aggregation not in valid_methods:
        msg = f"aggregation must be one of {valid_methods}, got {aggregation!r}"
        raise ValueError(msg)

    # Validate min_observations
    if min_observations < 1:
        msg = f"min_observations must be at least 1, got {min_observations}"
        raise ValueError(msg)

    # Validate input DataFrame
    if exposures.is_empty():
        msg = "exposures DataFrame cannot be empty"
        raise ValueError(msg)

    required_columns = {"asset_id", "benchmark_id", "ewma_beta"}
    missing_columns = required_columns - set(exposures.columns)
    if missing_columns:
        msg = f"exposures missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    # Convert factor_columns to list for consistency
    factor_list = list(factor_columns)
    if not factor_list:
        msg = "factor_columns cannot be empty"
        raise ValueError(msg)

    # Filter to only include valid factors and remove inf/nan values
    filtered_exposures = (
        exposures
        .filter(pl.col("benchmark_id").is_in(factor_list))
        .filter(pl.col("ewma_beta").is_finite())
    )

    if filtered_exposures.is_empty():
        msg = "No valid exposures after filtering for factors and finite values"
        raise ValueError(msg)

    # Pivot to wide format: asset_id x factor columns
    # Each row represents one observation (timestamp) for a sector
    pivot_exposures = (
        filtered_exposures
        .pivot(
            index=["asset_id", "ts_event"],
            on="benchmark_id",
            values="ewma_beta",
        )
        .drop_nulls()
    )

    if pivot_exposures.is_empty():
        msg = "No complete observations after pivoting (all sectors have missing factors)"
        raise ValueError(msg)

    # Compute stable positions for each sector
    stable_positions: dict[str, dict[str, float]] = {}

    for sector_frame in pivot_exposures.partition_by("asset_id", maintain_order=False):
        if sector_frame.is_empty():
            continue

        sector_id = str(sector_frame["asset_id"][0])
        n_observations = sector_frame.height

        # Validate minimum observations requirement
        if n_observations < min_observations:
            msg = (
                f"Sector {sector_id!r} has only {n_observations} observations, "
                f"but min_observations={min_observations}"
            )
            raise ValueError(msg)

        # Extract factor columns that exist in this frame
        available_factors = [col for col in factor_list if col in sector_frame.columns]

        if not available_factors:
            msg = f"Sector {sector_id!r} has no factor columns in the data"
            raise ValueError(msg)

        # Compute stable position based on aggregation method
        sector_position: dict[str, float] = {}

        for factor in available_factors:
            if aggregation == "median":
                stable_value = float(
                    sector_frame.select(pl.col(factor).median()).item()
                )
            elif aggregation == "mean":
                stable_value = float(
                    sector_frame.select(pl.col(factor).mean()).item()
                )
            elif aggregation == "trimmed_mean":
                # 10% trim on each tail (90% of data)
                lower_quantile = float(
                    sector_frame.select(pl.col(factor).quantile(0.1)).item()
                )
                upper_quantile = float(
                    sector_frame.select(pl.col(factor).quantile(0.9)).item()
                )
                # Use DataFrame filtering for proper type checking
                trimmed_frame = sector_frame.filter(
                    (pl.col(factor) >= lower_quantile) & (pl.col(factor) <= upper_quantile)
                )
                if trimmed_frame.is_empty():
                    # Fall back to mean if trimming removes all data
                    stable_value = float(
                        sector_frame.select(pl.col(factor).mean()).item()
                    )
                else:
                    stable_value = float(
                        trimmed_frame.select(pl.col(factor).mean()).item()
                    )
            else:
                # Should never reach here due to earlier validation
                msg = f"Unknown aggregation method: {aggregation!r}"
                raise ValueError(msg)

            sector_position[factor] = stable_value

        stable_positions[sector_id] = sector_position

    return stable_positions


def _to_nanoseconds(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return int(ts.timestamp() * 1_000_000_000)
