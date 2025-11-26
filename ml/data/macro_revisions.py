"""
Macro revision feature computation for ALFRED vintage data.

Implements revision-aware features that capture what traders see:
- Current release (headline)
- Prior releases (context)
- Revision deltas (surprise factors)
- Net signals (headline adjusted for revisions)
"""

from __future__ import annotations

from typing import Literal, cast

import ml._imports as _ml_imports
from ml.ml_types import PolarsDF


pl = _ml_imports.pl
check_ml_dependencies = _ml_imports.check_ml_dependencies

if pl is None:
    check_ml_dependencies(["polars"])  # pragma: no cover


def compute_revision_features_pl(
    release_df: PolarsDF,
    *,
    series_filter: set[str] | None = None,
    mode: Literal["minimal", "core", "full"] = "core",
    monthly_windows: list[int] | None = None,
) -> PolarsDF:
    """
    Compute revision-aware macro features from ALFRED vintage data.

    This transforms raw vintage releases into features that capture what traders see:
    - Current value (headline)
    - Prior values (context)
    - Revision deltas (surprise factors)
    - Net signals (headline adjusted for revisions)

    Parameters
    ----------
    release_df : PolarsDF
        ALFRED vintage data with schema:
        - series_id: str
        - observation_ts: datetime
        - value: float
        - release_ts: datetime
        - release_end_ts: datetime (optional)
    series_filter : set[str] | None
        If provided, only compute features for these series
    mode : {"minimal", "core", "full"}
        Feature mode:
        - minimal: current, prior_1m, revision_1m (3 features/series)
        - core: + mom_1m, pct_1m, net_signal_1m (6 features/series)
        - full: + prior_3m/12m, revision_3m, mom_3m/12m, pct_12m (12 features/series)
    monthly_windows : list[int] | None
        Months to use for prior/revision features. Defaults to [1, 3, 12].

    Returns
    -------
    PolarsDF
        DataFrame with revision features in long format:
        - timestamp: datetime (observation_ts)
        - series_id: str
        - value: float
        - release_ts: datetime (for as-of join)
        - feature_name: str (e.g., "PAYEMS_revision_1m")
        - feature_value: float

    Notes
    -----
    - Uses only data available at release_ts (no lookahead bias)
    - For each release, computes features based on prior releases
    - Returns long format suitable for concatenation with base vintages
    """
    if pl is None:
        check_ml_dependencies(["polars"])  # pragma: no cover
    _pl = pl
    assert _pl is not None

    if release_df.is_empty():
        return cast(
            PolarsDF,
            _pl.DataFrame(
                schema={
                    "timestamp": _pl.Datetime("ns"),
                    "series_id": _pl.Utf8,
                    "value": _pl.Float64,
                    "release_ts": _pl.Datetime("ns"),
                },
            ),
        )

    if series_filter is not None:
        release_df = release_df.filter(_pl.col("series_id").is_in(list(series_filter)))

    if release_df.is_empty():
        return cast(
            PolarsDF,
            _pl.DataFrame(
                schema={
                    "timestamp": _pl.Datetime("ns"),
                    "series_id": _pl.Utf8,
                    "value": _pl.Float64,
                    "release_ts": _pl.Datetime("ns"),
                },
            ),
        )

    windows = monthly_windows or [1, 3, 12]

    # Required columns
    required = {"series_id", "observation_ts", "value", "release_ts"}
    if not required.issubset(set(release_df.columns)):
        msg = f"release_df missing required columns. Need: {required}, Have: {release_df.columns}"
        raise ValueError(msg)

    # Sort by observation and release to establish vintage order
    df = (
        release_df.sort(["series_id", "observation_ts", "release_ts"])
        .with_row_count("row_idx")
    )

    # For each observation_ts, track which release this is (1st, 2nd, 3rd...)
    df = df.with_columns(
        [
            _pl.col("release_ts")
            .cum_count()
            .over(["series_id", "observation_ts"])
            .alias("release_number"),
        ],
    )

    # Track initial (first) release value for each observation
    df = df.with_columns(
        [
            _pl.when(_pl.col("release_number") == 1)
            .then(_pl.col("value"))
            .otherwise(None)
            .over(["series_id", "observation_ts"])
            .forward_fill()
            .alias("initial_value"),
        ],
    )

    # Compute revision from initial release
    df = df.with_columns(
        [
            (_pl.col("value") - _pl.col("initial_value")).alias("revision_from_initial"),
        ],
    )

    # Create revision features for each window
    revision_features: list[PolarsDF] = []

    for window_months in windows:
        # Prior observation timestamp (e.g., 1 month ago)
        df_with_prior = df.with_columns(
            [
                _pl.col("observation_ts")
                .dt.offset_by(f"-{window_months}mo")
                .alias(f"prior_{window_months}m_obs_ts"),
            ],
        )

        # Build lookup table for prior observations (all releases)
        prior_lookup = df.select(
            [
                _pl.col("series_id"),
                _pl.col("observation_ts").alias("prior_obs_ts"),
                _pl.col("release_ts").alias("prior_release_ts"),
                _pl.col("value").alias(f"prior_{window_months}m_value"),
                _pl.col("initial_value").alias(f"prior_{window_months}m_initial"),
                _pl.col("row_idx").alias("prior_row_idx"),
            ],
        )

        # Join on series_id + prior observation, then keep latest release <= current release
        df_with_prior = df_with_prior.join(
            prior_lookup,
            left_on=["series_id", f"prior_{window_months}m_obs_ts"],
            right_on=["series_id", "prior_obs_ts"],
            how="left",
        )

        df_with_prior = df_with_prior.filter(
            _pl.col("prior_release_ts").is_not_null()
            & (_pl.col("prior_release_ts") <= _pl.col("release_ts")),
        )

        if df_with_prior.is_empty():
            continue

        df_with_prior = (
            df_with_prior.sort(["row_idx", "prior_release_ts"])
            .group_by("row_idx")
            .tail(1)
        )

        # Compute revision for prior month
        # revision_1m = (prior month's current value) - (prior month's initial value)
        df_with_prior = df_with_prior.with_columns(
            [
                (
                    _pl.col(f"prior_{window_months}m_value")
                    - _pl.col(f"prior_{window_months}m_initial")
                ).alias(f"revision_{window_months}m"),
            ],
        )

        # Minimal mode: current, prior_1m, revision_1m
        if window_months == 1 or mode in ["core", "full"]:
            # Add prior value feature
            prior_feat = df_with_prior.select(
                [
                    _pl.col("observation_ts").alias("timestamp"),
                    _pl.col("series_id"),
                    _pl.col(f"prior_{window_months}m_value").alias("value"),
                    _pl.col("release_ts"),
                    _pl.concat_str(
                        [
                            _pl.col("series_id"),
                            _pl.lit(f"_prior_{window_months}m"),
                        ],
                    ).alias("feature_type"),
                ],
            ).with_columns(_pl.col("value").cast(_pl.Float64))
            revision_features.append(prior_feat)

            # Add revision delta feature
            revision_feat = df_with_prior.select(
                [
                    _pl.col("observation_ts").alias("timestamp"),
                    _pl.col("series_id"),
                    _pl.col(f"revision_{window_months}m").alias("value"),
                    _pl.col("release_ts"),
                    _pl.concat_str(
                        [
                            _pl.col("series_id"),
                            _pl.lit(f"_revision_{window_months}m"),
                        ],
                    ).alias("feature_type"),
                ],
            ).with_columns(_pl.col("value").cast(_pl.Float64))
            revision_features.append(revision_feat)

        # Core mode: add momentum and net signal for 1m window
        if window_months == 1 and mode in ["core", "full"]:
            # Month-over-month change
            mom_feat = df_with_prior.with_columns(
                [
                    (_pl.col("value") - _pl.col(f"prior_{window_months}m_value")).alias("mom_value"),
                ],
            ).select(
                [
                    _pl.col("observation_ts").alias("timestamp"),
                    _pl.col("series_id"),
                    _pl.col("mom_value").alias("value"),
                    _pl.col("release_ts"),
                    _pl.concat_str([_pl.col("series_id"), _pl.lit("_mom_1m")]).alias("feature_type"),
                ],
            ).with_columns(_pl.col("value").cast(_pl.Float64))
            revision_features.append(mom_feat)

            # Percentage change
            pct_feat = df_with_prior.with_columns(
                [
                    (
                        (_pl.col("value") / _pl.col(f"prior_{window_months}m_value")) - 1.0
                    ).alias("pct_value"),
                ],
            ).select(
                [
                    _pl.col("observation_ts").alias("timestamp"),
                    _pl.col("series_id"),
                    _pl.col("pct_value").alias("value"),
                    _pl.col("release_ts"),
                    _pl.concat_str([_pl.col("series_id"), _pl.lit("_pct_1m")]).alias("feature_type"),
                ],
            ).with_columns(_pl.col("value").cast(_pl.Float64))
            revision_features.append(pct_feat)

            # Net signal (headline adjusted for revision)
            net_signal_feat = df_with_prior.with_columns(
                [
                    (_pl.col("value") - _pl.col(f"revision_{window_months}m")).alias(
                        "net_signal_value",
                    ),
                ],
            ).select(
                [
                    _pl.col("observation_ts").alias("timestamp"),
                    _pl.col("series_id"),
                    _pl.col("net_signal_value").alias("value"),
                    _pl.col("release_ts"),
                    _pl.concat_str([_pl.col("series_id"), _pl.lit("_net_signal_1m")]).alias(
                        "feature_type",
                    ),
                ],
            ).with_columns(_pl.col("value").cast(_pl.Float64))
            revision_features.append(net_signal_feat)

        # Full mode: add 12m features
        if window_months == 12 and mode == "full":
            # Year-over-year change
            yoy_feat = df_with_prior.with_columns(
                [
                    (_pl.col("value") - _pl.col(f"prior_{window_months}m_value")).alias("yoy_value"),
                ],
            ).select(
                [
                    _pl.col("observation_ts").alias("timestamp"),
                    _pl.col("series_id"),
                    _pl.col("yoy_value").alias("value"),
                    _pl.col("release_ts"),
                    _pl.concat_str([_pl.col("series_id"), _pl.lit("_mom_12m")]).alias(
                        "feature_type",
                    ),
                ],
            ).with_columns(_pl.col("value").cast(_pl.Float64))
            revision_features.append(yoy_feat)

            # YoY percentage
            yoy_pct_feat = df_with_prior.with_columns(
                [
                    (
                        (_pl.col("value") / _pl.col(f"prior_{window_months}m_value")) - 1.0
                    ).alias("yoy_pct_value"),
                ],
            ).select(
                [
                    _pl.col("observation_ts").alias("timestamp"),
                    _pl.col("series_id"),
                    _pl.col("yoy_pct_value").alias("value"),
                    _pl.col("release_ts"),
                    _pl.concat_str([_pl.col("series_id"), _pl.lit("_pct_12m")]).alias(
                        "feature_type",
                    ),
                ],
            ).with_columns(_pl.col("value").cast(_pl.Float64))
            revision_features.append(yoy_pct_feat)

    if not revision_features:
        return cast(
            PolarsDF,
            _pl.DataFrame(
                schema={
                    "timestamp": _pl.Datetime("ns"),
                    "series_id": _pl.Utf8,
                    "value": _pl.Float64,
                    "release_ts": _pl.Datetime("ns"),
                },
            ),
        )

    # Combine all revision features
    combined = _pl.concat(revision_features, how="vertical")

    # Ensure consistent timestamp dtype
    if "timestamp" in combined.columns:
        combined = combined.with_columns(_pl.col("timestamp").cast(_pl.Datetime("ns")))
    if "release_ts" in combined.columns:
        combined = combined.with_columns(_pl.col("release_ts").cast(_pl.Datetime("ns")))

    return cast(PolarsDF, combined)
