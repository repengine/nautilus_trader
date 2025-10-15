"""
Rolling window beta estimation and stability analysis for 3D risk model.

This module implements rolling window beta estimation to compare against stable
(full-sample) betas, validating which approach better predicts future returns
and remains stable over time.

Key capabilities:
- Rolling window OLS regression for beta estimation
- Beta stability metrics (coefficient of variation)
- Forecast accuracy comparison (rolling vs stable betas)
- Out-of-sample predictive performance testing
- Visual inspection of beta time series

Performance: Cold path only (training/validation, not real-time inference)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog


if TYPE_CHECKING:
    import statsmodels.api as sm
else:
    sm = None

# Lazy import of statsmodels to reduce startup time
_SM_MODULE: object | None = None


def _get_sm() -> object:
    """Lazy import of statsmodels to reduce startup time."""
    global _SM_MODULE
    if _SM_MODULE is None:
        import statsmodels.api

        _SM_MODULE = statsmodels.api
    return _SM_MODULE


LOGGER = structlog.get_logger(__name__)

EXPECTED_FACTOR_COLUMNS: tuple[str, str, str] = (
    "factor_duration",
    "factor_credit",
    "factor_liquidity",
)


@dataclass(slots=True)
class RollingBetaResult:
    """
    Results from rolling window beta estimation.

    Attributes
    ----------
    sector_id : str
        Sector ETF ticker (e.g., "XLK", "XLU").
    timestamps : list[datetime]
        Window end dates for each rolling regression.
    beta_duration : list[float]
        Rolling beta estimates for duration factor.
    beta_credit : list[float]
        Rolling beta estimates for credit factor.
    beta_liquidity : list[float]
        Rolling beta estimates for liquidity factor.
    alpha : list[float]
        Rolling alpha (intercept) estimates.
    r_squared : list[float]
        Rolling R² values.
    window_size : int
        Number of observations in each window.
    n_windows : int
        Total number of windows computed.
    """

    sector_id: str
    timestamps: list[datetime]
    beta_duration: list[float]
    beta_credit: list[float]
    beta_liquidity: list[float]
    alpha: list[float]
    r_squared: list[float]
    window_size: int
    n_windows: int


@dataclass(slots=True)
class BetaStabilityAnalysis:
    """
    Stability metrics comparing rolling vs stable betas.

    Attributes
    ----------
    sector_id : str
        Sector ETF ticker.
    stable_beta_duration : float
        Full-sample beta for duration factor.
    stable_beta_credit : float
        Full-sample beta for credit factor.
    stable_beta_liquidity : float
        Full-sample beta for liquidity factor.
    stable_r_squared : float
        Full-sample R².
    rolling_beta_mean_duration : float
        Mean of rolling duration betas.
    rolling_beta_std_duration : float
        Standard deviation of rolling duration betas.
    rolling_beta_mean_credit : float
        Mean of rolling credit betas.
    rolling_beta_std_credit : float
        Standard deviation of rolling credit betas.
    rolling_beta_mean_liquidity : float
        Mean of rolling liquidity betas.
    rolling_beta_std_liquidity : float
        Standard deviation of rolling liquidity betas.
    beta_duration_cv : float
        Coefficient of variation for duration beta (std/mean).
    beta_credit_cv : float
        Coefficient of variation for credit beta.
    beta_liquidity_cv : float
        Coefficient of variation for liquidity beta.
    stable_forecast_r2 : float
        Out-of-sample R² using stable betas.
    rolling_forecast_r2 : float
        Out-of-sample R² using rolling betas.
    recommended_approach : str
        "stable" or "rolling" based on analysis.
    rationale : str
        Explanation for recommendation.
    """

    sector_id: str
    stable_beta_duration: float
    stable_beta_credit: float
    stable_beta_liquidity: float
    stable_r_squared: float
    rolling_beta_mean_duration: float
    rolling_beta_std_duration: float
    rolling_beta_mean_credit: float
    rolling_beta_std_credit: float
    rolling_beta_mean_liquidity: float
    rolling_beta_std_liquidity: float
    beta_duration_cv: float
    beta_credit_cv: float
    beta_liquidity_cv: float
    stable_forecast_r2: float
    rolling_forecast_r2: float
    recommended_approach: str
    rationale: str


def compute_rolling_betas(
    sector_returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
    *,
    factor_columns: Sequence[str],
    window_days: int = 252,
    min_observations: int = 126,
) -> dict[str, RollingBetaResult]:
    """
    Compute rolling window betas for each sector.

    For each sector:
    1. Create overlapping windows of size window_days
    2. Run OLS regression in each window
    3. Extract betas, alpha, R² for each window
    4. Return time series of beta estimates

    Parameters
    ----------
    sector_returns : pl.DataFrame
        Sector returns with columns: timestamp, symbol, return.
    factor_returns : pl.DataFrame
        Factor returns with columns: timestamp + factor_columns.
    factor_columns : Sequence[str]
        Factor column names (e.g., ["factor_duration", "factor_credit", "factor_liquidity"]).
    window_days : int
        Rolling window size in observations (default: 252 = 1 year).
    min_observations : int
        Minimum observations required per window (default: 126 = 6 months).

    Returns
    -------
    dict[str, RollingBetaResult]
        Rolling beta results keyed by sector ID.

    Raises
    ------
    ValueError
        If inputs are invalid or no valid sectors found.

    Notes
    -----
    - Windows are overlapping (slide by 1 observation)
    - Requires at least min_observations in each window
    - Skips windows where regression fails
    """
    LOGGER.info(
        "Computing rolling betas",
        window_days=window_days,
        min_observations=min_observations,
        n_factors=len(factor_columns),
    )

    # Validate inputs
    _validate_rolling_beta_inputs(sector_returns, factor_returns, factor_columns)

    if window_days < min_observations:
        msg = f"window_days ({window_days}) must be >= min_observations ({min_observations})"
        raise ValueError(msg)

    # Join sector and factor data
    joined = sector_returns.join(factor_returns, on="timestamp", how="inner").sort("timestamp")

    if joined.is_empty():
        msg = "No data after joining sector and factor returns"
        raise ValueError(msg)

    results: dict[str, RollingBetaResult] = {}
    sm_module = _get_sm()

    for sector in joined["symbol"].unique().to_list():
        sector_data = joined.filter(pl.col("symbol") == sector)

        if sector_data.height < min_observations:
            LOGGER.warning(
                "Insufficient observations for sector",
                sector=sector,
                n_obs=sector_data.height,
                min_required=min_observations,
            )
            continue

        try:
            sector_result = _compute_sector_rolling_betas(
                sector_data,
                sector,
                factor_columns,
                window_days,
                min_observations,
                sm_module,
            )
            if sector_result.n_windows > 0:
                results[sector] = sector_result
                LOGGER.info(
                    "Computed rolling betas for sector",
                    sector=sector,
                    n_windows=sector_result.n_windows,
                )
        except Exception:
            LOGGER.exception("Failed to compute rolling betas for sector", sector=sector)
            continue

    if not results:
        msg = "No valid rolling beta results computed for any sector"
        raise ValueError(msg)

    LOGGER.info("Completed rolling beta computation", n_sectors=len(results))

    return results


def _compute_sector_rolling_betas(
    sector_data: pl.DataFrame,
    sector_id: str,
    factor_columns: Sequence[str],
    window_days: int,
    min_observations: int,
    sm_module: object,
) -> RollingBetaResult:
    """Compute rolling betas for a single sector."""
    timestamps = sector_data["timestamp"].to_list()
    n = len(timestamps)

    beta_duration_list: list[float] = []
    beta_credit_list: list[float] = []
    beta_liquidity_list: list[float] = []
    alpha_list: list[float] = []
    r_squared_list: list[float] = []
    window_end_dates: list[datetime] = []

    # Rolling regression
    for i in range(window_days, n + 1):
        window_data = sector_data[i - window_days : i]

        if window_data.height < min_observations:
            continue

        # Extract y and X
        y = window_data["return"].to_numpy()
        X = window_data.select(list(factor_columns)).to_numpy()
        X_with_const = sm_module.add_constant(X)  # type: ignore[attr-defined]

        # Run OLS
        try:
            model = sm_module.OLS(y, X_with_const).fit()  # type: ignore[attr-defined]

            params = model.params
            alpha_list.append(float(params[0]))
            beta_duration_list.append(float(params[1]))
            beta_credit_list.append(float(params[2]))
            beta_liquidity_list.append(float(params[3]))
            r_squared_list.append(float(model.rsquared))
            window_end_dates.append(timestamps[i - 1])
        except Exception:
            # Skip window if regression fails
            continue

    return RollingBetaResult(
        sector_id=sector_id,
        timestamps=window_end_dates,
        beta_duration=beta_duration_list,
        beta_credit=beta_credit_list,
        beta_liquidity=beta_liquidity_list,
        alpha=alpha_list,
        r_squared=r_squared_list,
        window_size=window_days,
        n_windows=len(window_end_dates),
    )


def compute_beta_stability_analysis(
    sector_returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
    rolling_results: dict[str, RollingBetaResult],
    *,
    factor_columns: Sequence[str],
    test_period_start: datetime,
) -> dict[str, BetaStabilityAnalysis]:
    """
    Compare rolling vs stable beta approaches.

    For each sector:
    1. Compute stable (full-sample) betas on training data
    2. Compute rolling beta statistics (mean, std, CV)
    3. Compare forecast accuracy on test period:
       - Stable: Use full-sample betas to predict test period
       - Rolling: Use most recent window betas to predict test period
    4. Recommend stable or rolling based on:
       - Beta stability (lower CV is better)
       - Forecast accuracy (higher R² is better)

    Parameters
    ----------
    sector_returns : pl.DataFrame
        Sector returns with columns: timestamp, symbol, return.
    factor_returns : pl.DataFrame
        Factor returns with columns: timestamp + factor_columns.
    rolling_results : dict[str, RollingBetaResult]
        Rolling beta results from compute_rolling_betas.
    factor_columns : Sequence[str]
        Factor column names.
    test_period_start : datetime
        Start date of test period for forecast evaluation.

    Returns
    -------
    dict[str, BetaStabilityAnalysis]
        Stability analysis results keyed by sector ID.

    Raises
    ------
    ValueError
        If inputs are invalid or no valid results computed.

    Notes
    -----
    - Train period: all data before test_period_start
    - Test period: all data >= test_period_start
    - Recommendation logic:
      * Prefer stable if CV < 0.3 and stable R² >= rolling R²
      * Prefer rolling if rolling R² > stable R² * 1.1 (10% improvement)
      * Otherwise prefer stable for simplicity
    """
    LOGGER.info(
        "Computing beta stability analysis",
        n_sectors=len(rolling_results),
        test_period_start=test_period_start.isoformat(),
    )

    # Validate inputs
    _validate_rolling_beta_inputs(sector_returns, factor_returns, factor_columns)

    if not rolling_results:
        msg = "rolling_results cannot be empty"
        raise ValueError(msg)

    # Split data into train and test
    train_sector = sector_returns.filter(pl.col("timestamp") < test_period_start)
    test_sector = sector_returns.filter(pl.col("timestamp") >= test_period_start)

    train_factor = factor_returns.filter(pl.col("timestamp") < test_period_start)
    test_factor = factor_returns.filter(pl.col("timestamp") >= test_period_start)

    if train_sector.is_empty():
        msg = f"No training data before {test_period_start.isoformat()}"
        raise ValueError(msg)

    if test_sector.is_empty():
        msg = f"No test data >= {test_period_start.isoformat()}"
        raise ValueError(msg)

    results: dict[str, BetaStabilityAnalysis] = {}
    sm_module = _get_sm()

    for sector_id, rolling_result in rolling_results.items():
        try:
            analysis = _compute_sector_stability_analysis(
                train_sector,
                test_sector,
                train_factor,
                test_factor,
                sector_id,
                rolling_result,
                factor_columns,
                sm_module,
            )
            results[sector_id] = analysis

            LOGGER.info(
                "Computed stability analysis for sector",
                sector=sector_id,
                recommended=analysis.recommended_approach,
                stable_r2=f"{analysis.stable_forecast_r2:.4f}",
                rolling_r2=f"{analysis.rolling_forecast_r2:.4f}",
            )
        except Exception:
            LOGGER.exception("Failed to compute stability analysis for sector", sector=sector_id)
            continue

    if not results:
        msg = "No valid stability analysis results computed"
        raise ValueError(msg)

    LOGGER.info("Completed stability analysis", n_sectors=len(results))

    return results


def _compute_sector_stability_analysis(
    train_sector: pl.DataFrame,
    test_sector: pl.DataFrame,
    train_factor: pl.DataFrame,
    test_factor: pl.DataFrame,
    sector_id: str,
    rolling_result: RollingBetaResult,
    factor_columns: Sequence[str],
    sm_module: object,
) -> BetaStabilityAnalysis:
    """Compute stability analysis for a single sector."""
    # 1. Compute stable betas (full training sample)
    train_joined = (
        train_sector.filter(pl.col("symbol") == sector_id)
        .join(train_factor, on="timestamp", how="inner")
        .sort("timestamp")
    )

    if train_joined.is_empty():
        msg = f"No training data for sector {sector_id}"
        raise ValueError(msg)

    y_train = train_joined["return"].to_numpy()
    X_train = train_joined.select(list(factor_columns)).to_numpy()
    X_train_const = sm_module.add_constant(X_train)  # type: ignore[attr-defined]

    stable_model = sm_module.OLS(y_train, X_train_const).fit()  # type: ignore[attr-defined]
    stable_params = stable_model.params
    stable_beta_duration = float(stable_params[1])
    stable_beta_credit = float(stable_params[2])
    stable_beta_liquidity = float(stable_params[3])
    stable_r2 = float(stable_model.rsquared)

    # 2. Rolling beta statistics
    rolling_beta_duration_array = np.asarray(rolling_result.beta_duration, dtype=float)
    rolling_beta_credit_array = np.asarray(rolling_result.beta_credit, dtype=float)
    rolling_beta_liquidity_array = np.asarray(rolling_result.beta_liquidity, dtype=float)

    rolling_beta_mean_dur = float(np.mean(rolling_beta_duration_array))
    rolling_beta_mean_cred = float(np.mean(rolling_beta_credit_array))
    rolling_beta_mean_liq = float(np.mean(rolling_beta_liquidity_array))

    rolling_beta_std_dur = float(
        np.std(rolling_beta_duration_array, ddof=1)
        if rolling_beta_duration_array.size > 1
        else 0.0
    )
    rolling_beta_std_cred = float(
        np.std(rolling_beta_credit_array, ddof=1) if rolling_beta_credit_array.size > 1 else 0.0
    )
    rolling_beta_std_liq = float(
        np.std(rolling_beta_liquidity_array, ddof=1)
        if rolling_beta_liquidity_array.size > 1
        else 0.0
    )

    # 3. Coefficient of variation (stability metric)
    cv_dur = (
        rolling_beta_std_dur / abs(rolling_beta_mean_dur)
        if abs(rolling_beta_mean_dur) > 1e-10
        else float("inf")
    )
    cv_cred = (
        rolling_beta_std_cred / abs(rolling_beta_mean_cred)
        if abs(rolling_beta_mean_cred) > 1e-10
        else float("inf")
    )
    cv_liq = (
        rolling_beta_std_liq / abs(rolling_beta_mean_liq)
        if abs(rolling_beta_mean_liq) > 1e-10
        else float("inf")
    )

    # 4. Forecast accuracy (test period)
    test_joined = (
        test_sector.filter(pl.col("symbol") == sector_id)
        .join(test_factor, on="timestamp", how="inner")
        .sort("timestamp")
    )

    if test_joined.is_empty():
        msg = f"No test data for sector {sector_id}"
        raise ValueError(msg)

    y_test = test_joined["return"].to_numpy()
    X_test = test_joined.select(list(factor_columns)).to_numpy()

    # Stable forecast
    stable_pred = stable_params[0] + X_test @ stable_params[1:]
    stable_ss_res = np.sum((y_test - stable_pred) ** 2)
    stable_ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    stable_forecast_r2 = (
        float(1 - stable_ss_res / stable_ss_tot) if stable_ss_tot > 1e-10 else 0.0
    )

    # Rolling forecast (use most recent window betas)
    rolling_beta_recent = np.array(
        [
            rolling_result.beta_duration[-1],
            rolling_result.beta_credit[-1],
            rolling_result.beta_liquidity[-1],
        ]
    )
    rolling_alpha_recent = rolling_result.alpha[-1]
    rolling_pred = rolling_alpha_recent + X_test @ rolling_beta_recent
    rolling_ss_res = np.sum((y_test - rolling_pred) ** 2)
    rolling_forecast_r2 = (
        float(1 - rolling_ss_res / stable_ss_tot) if stable_ss_tot > 1e-10 else 0.0
    )

    # 5. Recommendation
    mean_cv = (cv_dur + cv_cred + cv_liq) / 3

    if mean_cv < 0.3 and stable_forecast_r2 >= rolling_forecast_r2:
        recommended = "stable"
        rationale = (
            f"Betas are stable (mean CV={mean_cv:.2f}) and stable forecast is better "
            f"(R²={stable_forecast_r2:.3f} vs {rolling_forecast_r2:.3f})"
        )
    elif rolling_forecast_r2 > stable_forecast_r2 * 1.1:  # 10% improvement
        recommended = "rolling"
        rationale = (
            f"Rolling forecast significantly better "
            f"(R²={rolling_forecast_r2:.3f} vs {stable_forecast_r2:.3f})"
        )
    else:
        recommended = "stable"
        rationale = (
            f"Comparable performance, prefer stable for simplicity "
            f"(R² difference: {abs(stable_forecast_r2 - rolling_forecast_r2):.3f})"
        )

    return BetaStabilityAnalysis(
        sector_id=sector_id,
        stable_beta_duration=stable_beta_duration,
        stable_beta_credit=stable_beta_credit,
        stable_beta_liquidity=stable_beta_liquidity,
        stable_r_squared=stable_r2,
        rolling_beta_mean_duration=rolling_beta_mean_dur,
        rolling_beta_std_duration=rolling_beta_std_dur,
        rolling_beta_mean_credit=rolling_beta_mean_cred,
        rolling_beta_std_credit=rolling_beta_std_cred,
        rolling_beta_mean_liquidity=rolling_beta_mean_liq,
        rolling_beta_std_liquidity=rolling_beta_std_liq,
        beta_duration_cv=cv_dur,
        beta_credit_cv=cv_cred,
        beta_liquidity_cv=cv_liq,
        stable_forecast_r2=stable_forecast_r2,
        rolling_forecast_r2=rolling_forecast_r2,
        recommended_approach=recommended,
        rationale=rationale,
    )


def plot_rolling_betas(
    rolling_result: RollingBetaResult,
    *,
    output_path: Path | None = None,
) -> None:
    """
    Plot rolling beta time series for visual inspection.

    Creates a multi-panel plot:
    - Panel 1: Beta duration over time
    - Panel 2: Beta credit over time
    - Panel 3: Beta liquidity over time
    - Panel 4: Rolling R² over time

    Parameters
    ----------
    rolling_result : RollingBetaResult
        Rolling beta results for a single sector.
    output_path : Path | None
        Optional path to save the figure (PNG format).

    Notes
    -----
    - Requires matplotlib
    - Figure size: 14x10 inches
    - Each panel shows time series with markers
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib required for visualization"
        raise ImportError(msg) from e

    _fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Beta duration
    axes[0].plot(
        rolling_result.timestamps,
        rolling_result.beta_duration,
        marker="o",
        markersize=3,
        linestyle="-",
        linewidth=1,
    )
    axes[0].set_ylabel("Beta Duration")
    axes[0].set_title(f"Rolling Betas for {rolling_result.sector_id}")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Beta credit
    axes[1].plot(
        rolling_result.timestamps,
        rolling_result.beta_credit,
        marker="o",
        markersize=3,
        linestyle="-",
        linewidth=1,
        color="orange",
    )
    axes[1].set_ylabel("Beta Credit")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Beta liquidity
    axes[2].plot(
        rolling_result.timestamps,
        rolling_result.beta_liquidity,
        marker="o",
        markersize=3,
        linestyle="-",
        linewidth=1,
        color="green",
    )
    axes[2].set_ylabel("Beta Liquidity")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Rolling R²
    axes[3].plot(
        rolling_result.timestamps,
        rolling_result.r_squared,
        marker="o",
        markersize=3,
        linestyle="-",
        linewidth=1,
        color="red",
    )
    axes[3].set_ylabel("R²")
    axes[3].set_xlabel("Date")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        LOGGER.info("Rolling beta plot saved", path=str(output_path))

    plt.close()


def _validate_rolling_beta_inputs(
    sector_returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
    factor_columns: Sequence[str],
) -> None:
    """Validate inputs for rolling beta computation."""
    # Check sector returns columns
    required_sector_cols = {"timestamp", "symbol", "return"}
    missing = required_sector_cols - set(sector_returns.columns)
    if missing:
        msg = f"sector_returns missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    # Check factor returns columns
    required_factor_cols = {"timestamp"} | set(factor_columns)
    missing_factors = required_factor_cols - set(factor_returns.columns)
    if missing_factors:
        msg = f"factor_returns missing required columns: {sorted(missing_factors)}"
        raise ValueError(msg)

    # Check for empty DataFrames
    if sector_returns.is_empty():
        msg = "sector_returns cannot be empty"
        raise ValueError(msg)

    if factor_returns.is_empty():
        msg = "factor_returns cannot be empty"
        raise ValueError(msg)

    # Validate factor columns
    if not factor_columns:
        msg = "factor_columns cannot be empty"
        raise ValueError(msg)
    _ensure_expected_factor_columns(factor_columns)


__all__ = [
    "BetaStabilityAnalysis",
    "RollingBetaResult",
    "compute_beta_stability_analysis",
    "compute_rolling_betas",
    "plot_rolling_betas",
]


def _ensure_expected_factor_columns(factor_columns: Sequence[str]) -> None:
    """Ensure factor_columns matches the required order for the 3D model."""
    if tuple(factor_columns) != EXPECTED_FACTOR_COLUMNS:
        expected = ", ".join(EXPECTED_FACTOR_COLUMNS)
        msg = (
            "factor_columns must contain exactly the 3D risk model factors "
            f"in the order [{expected}], received {list(factor_columns)}"
        )
        raise ValueError(msg)
