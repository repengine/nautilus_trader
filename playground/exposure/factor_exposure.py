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
) -> pl.DataFrame:
    """Compute factor returns (percentage change) for the selected columns."""
    column_list = list(columns)
    selected = factor_features.sort("timestamp").select(["timestamp", *column_list])
    return selected.with_columns(
        [pl.col(col).pct_change().alias(col) for col in column_list],
    ).drop_nulls(subset=column_list)


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


def _to_nanoseconds(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return int(ts.timestamp() * 1_000_000_000)
