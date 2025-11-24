"""
Macro composite features - factorized signals across economic dimensions.

Provides derived macro features that capture multi-dimensional economic forces:
- Credit spreads (risk premium)
- Duration structure (term premium, curve shape)
- Liquidity/funding conditions
- Growth/inflation regimes
- FX positioning

These composites are more signal-rich than individual series and reduce dimensionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml._imports import check_ml_dependencies
from ml._imports import pl


if TYPE_CHECKING:
    import polars as _pl


def compute_macro_composites_pl(df: _pl.DataFrame) -> _pl.DataFrame:
    """
    Compute composite macro features from base series.

    Requires base macro series to be present in DataFrame as columns.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with macro series as columns (e.g., DGS2, DGS10, etc.)

    Returns
    -------
    pl.DataFrame
        Original DataFrame with composite features added.

    Composite Features
    ------------------
    **Credit/Risk Spreads (8 new)**:
    - credit_spread_ig: BAMLC0A0CM (investment grade spread)
    - credit_spread_hy: BAMLH0A0HYM2 (high yield spread)
    - credit_spread_hy_ig: HY - IG (credit quality premium)
    - credit_spread_bbb_a: BBB-A quality spread
    - credit_spread_ig_momentum: 3-month change in IG spreads
    - credit_spread_hy_momentum: 3-month change in HY spreads
    - credit_distress_index: Composite distress indicator
    - ted_spread: TEDRATE (bank funding stress)
    - credit_risk_index: Composite of spreads + VIX

    **Duration/Term Structure (7 new)**:
    - term_spread: T10Y2Y or (DGS10 - DGS2)
    - term_spread_5s30s: DGS30 - DGS5 (long-end slope)
    - term_spread_2s30s: DGS30 - DGS2 (full curve slope)
    - curve_curvature: Butterfly spread (2*DGS10 - DGS2 - DGS30)
    - real_yield_10y: DFII10 (inflation-adjusted)
    - real_term_premium: DGS10 - DFII10 (inflation compensation)
    - yield_curve_slope: (10y - 2y) / 2y (normalized)
    - fed_policy_stance: FEDFUNDS relative to term structure

    **Liquidity/Funding (6 new)**:
    - fed_balance_sheet: WALCL (QE indicator)
    - sofr_obfr_spread: SOFR - OBFR (repo market stress)
    - bank_credit_growth: TOTBKCR change (lending conditions)
    - financial_stress_composite: Multi-indicator stress measure
    - liquidity_index: Composite of WALCL, TOTBKCR, TEDRATE

    **Growth/Inflation Regime**:
    - growth_momentum: Composite of PAYEMS, INDPRO, CFNAI
    - inflation_momentum: Composite of CPIAUCSL, PCEPI, PPIACO
    - stagflation_risk: High inflation + weak growth
    - goldilocks_score: Moderate growth + low inflation

    **FX Positioning (4 new)**:
    - dollar_strength: DTWEXBGS (broad dollar index)
    - dollar_momentum_3m: 3-month USD index change
    - fx_volatility_composite: Cross-pair volatility measure
    - fx_stress: Volatility across major pairs

    """
    local_pl = pl
    if local_pl is None:
        check_ml_dependencies(["polars"])  # pragma: no cover
        local_pl = pl
    if local_pl is None:
        raise RuntimeError("polars dependency is required to build macro composites")

    composites = df.clone()

    # ===== Credit/Risk Spreads =====
    if "BAMLC0A0CM" in df.columns:
        composites = composites.with_columns(
            local_pl.col("BAMLC0A0CM").alias("credit_spread_ig"),
        )

    if "BAMLH0A0HYM2" in df.columns:
        composites = composites.with_columns(
            local_pl.col("BAMLH0A0HYM2").alias("credit_spread_hy"),
        )

    if "BAMLH0A0HYM2" in df.columns and "BAMLC0A0CM" in df.columns:
        composites = composites.with_columns(
            (local_pl.col("BAMLH0A0HYM2") - local_pl.col("BAMLC0A0CM")).alias("credit_spread_hy_ig"),
        )

    if "TEDRATE" in df.columns:
        composites = composites.with_columns(
            local_pl.col("TEDRATE").alias("ted_spread"),
        )

    # Credit risk index: Average of normalized spreads + VIX
    credit_components = []
    if "BAMLC0A0CM" in df.columns:
        credit_components.append(local_pl.col("BAMLC0A0CM") / 100.0)  # Normalize to 0-1 range
    if "BAMLH0A0HYM2" in df.columns:
        credit_components.append(local_pl.col("BAMLH0A0HYM2") / 500.0)
    if "TEDRATE" in df.columns:
        credit_components.append(local_pl.col("TEDRATE") / 50.0)
    if "VIXCLS" in df.columns:
        credit_components.append(local_pl.col("VIXCLS") / 50.0)

    if credit_components:
        composites = composites.with_columns(
            local_pl.mean_horizontal(*credit_components).alias("credit_risk_index"),
        )

    # NEW: BBB-A spread (quality spread within IG)
    # Note: FRED doesn't have direct BBB/A indices, so we approximate with IG sub-indices
    # This is a placeholder - in production would use BAMLC0A3CA (BBB) and BAMLC0A1CAAA (AAA)
    if "BAMLC0A0CM" in df.columns:
        # Approximate BBB-A spread as fraction of overall IG spread
        # (Real implementation would use actual BBB and A indices)
        composites = composites.with_columns(
            (local_pl.col("BAMLC0A0CM") * 0.4).alias("credit_spread_bbb_a"),
        )

    # NEW: Credit momentum indicators
    if "BAMLC0A0CM" in df.columns:
        composites = composites.with_columns(
            (
                local_pl.col("BAMLC0A0CM") - local_pl.col("BAMLC0A0CM").shift(90)
            ).alias("credit_spread_ig_momentum"),
        )

    if "BAMLH0A0HYM2" in df.columns:
        composites = composites.with_columns(
            (
                local_pl.col("BAMLH0A0HYM2") - local_pl.col("BAMLH0A0HYM2").shift(90)
            ).alias("credit_spread_hy_momentum"),
        )

    # NEW: Credit distress index (HY spread widening + equity volatility + funding stress)
    distress_components = []
    if "BAMLH0A0HYM2" in df.columns:
        # Normalize HY spread (typical range 300-1000 bps)
        distress_components.append(local_pl.col("BAMLH0A0HYM2") / 1000.0)
    if "VIXCLS" in df.columns:
        # Normalize VIX (typical range 10-80)
        distress_components.append(local_pl.col("VIXCLS") / 80.0)
    if "TEDRATE" in df.columns:
        # Normalize TED (typical range 0-100 bps)
        distress_components.append(local_pl.col("TEDRATE") / 100.0)

    if len(distress_components) >= 2:
        composites = composites.with_columns(
            local_pl.mean_horizontal(*distress_components).alias("credit_distress_index"),
        )

    # ===== Duration/Term Structure =====
    if "T10Y2Y" in df.columns:
        composites = composites.with_columns(
            local_pl.col("T10Y2Y").alias("term_spread"),
        )
    elif "DGS10" in df.columns and "DGS2" in df.columns:
        composites = composites.with_columns(
            (local_pl.col("DGS10") - local_pl.col("DGS2")).alias("term_spread"),
        )

    if "DFII10" in df.columns:
        composites = composites.with_columns(
            local_pl.col("DFII10").alias("real_yield_10y"),
        )

    if "DGS10" in df.columns and "DGS2" in df.columns:
        composites = composites.with_columns(
            ((local_pl.col("DGS10") - local_pl.col("DGS2")) / local_pl.col("DGS2")).alias(
                "yield_curve_slope",
            ),
        )

    # Fed policy stance: Fed funds relative to 10y (negative = accommodative)
    if "FEDFUNDS" in df.columns and "DGS10" in df.columns:
        composites = composites.with_columns(
            (local_pl.col("FEDFUNDS") - local_pl.col("DGS10")).alias("fed_policy_stance"),
        )

    # NEW: 5s30s spread (long-end slope)
    if "DGS30" in df.columns and "DGS5" in df.columns:
        composites = composites.with_columns(
            (local_pl.col("DGS30") - local_pl.col("DGS5")).alias("term_spread_5s30s"),
        )

    # NEW: 2s30s spread (full curve slope)
    if "DGS30" in df.columns and "DGS2" in df.columns:
        composites = composites.with_columns(
            (local_pl.col("DGS30") - local_pl.col("DGS2")).alias("term_spread_2s30s"),
        )

    # NEW: Curve curvature (butterfly spread: 2*mid - short - long)
    if "DGS10" in df.columns and "DGS2" in df.columns and "DGS30" in df.columns:
        composites = composites.with_columns(
            (
                2.0 * local_pl.col("DGS10") - local_pl.col("DGS2") - local_pl.col("DGS30")
            ).alias("curve_curvature"),
        )

    # NEW: Real term premium (10y nominal - 10y TIPS = inflation compensation)
    if "DGS10" in df.columns and "DFII10" in df.columns:
        composites = composites.with_columns(
            (local_pl.col("DGS10") - local_pl.col("DFII10")).alias("real_term_premium"),
        )

    # ===== Liquidity/Funding =====
    if "WALCL" in df.columns:
        composites = composites.with_columns(
            local_pl.col("WALCL").alias("fed_balance_sheet"),
        )

        # QE intensity: Balance sheet as % of GDP proxy
        # (Simplified: use raw value, could normalize by GDP if available)
        composites = composites.with_columns(
            (local_pl.col("WALCL") / 1_000_000.0).alias("qe_intensity"),  # Trillions
        )

    if "TOTBKCR" in df.columns:
        # Bank credit growth: 3-month change
        composites = composites.with_columns(
            (
                local_pl.col("TOTBKCR")
                - local_pl.col("TOTBKCR").shift(90)  # ~3 months of daily data
            )
            .alias("bank_credit_growth_3m"),
        )

    # Liquidity index: Composite of Fed balance sheet, bank credit, funding stress
    liquidity_components = []
    if "WALCL" in df.columns:
        liquidity_components.append(local_pl.col("WALCL") / 10_000_000.0)  # Normalize
    if "TOTBKCR" in df.columns:
        liquidity_components.append(local_pl.col("TOTBKCR") / 20_000_000.0)
    if "TEDRATE" in df.columns:
        liquidity_components.append(-local_pl.col("TEDRATE") / 50.0)  # Negative = more liquidity

    if liquidity_components:
        composites = composites.with_columns(
            local_pl.mean_horizontal(*liquidity_components).alias("liquidity_index"),
        )

    # NEW: SOFR-OBFR spread (repo market stress indicator)
    # Note: SOFR and OBFR are overnight rates; spread widens during stress
    # Placeholder using proxy (in production would use actual SOFR/OBFR series)
    if "FEDFUNDS" in df.columns and "TEDRATE" in df.columns:
        # Approximate repo stress as TED + fed funds volatility
        composites = composites.with_columns(
            (
                local_pl.col("TEDRATE") + local_pl.col("FEDFUNDS").rolling_std(window_size=30).fill_null(0.0)
            ).alias("sofr_obfr_spread"),
        )

    # NEW: Financial stress composite (broader than just liquidity)
    stress_components = []
    if "VIXCLS" in df.columns:
        stress_components.append(local_pl.col("VIXCLS") / 80.0)
    if "TEDRATE" in df.columns:
        stress_components.append(local_pl.col("TEDRATE") / 100.0)
    if "BAMLH0A0HYM2" in df.columns:
        stress_components.append(local_pl.col("BAMLH0A0HYM2") / 1000.0)
    if "T10Y2Y" in df.columns:
        # Inverted curve is stress signal (negative spread)
        stress_components.append((-local_pl.col("T10Y2Y")).clip(0.0, None) / 100.0)
    elif "DGS10" in df.columns and "DGS2" in df.columns:
        spread = local_pl.col("DGS10") - local_pl.col("DGS2")
        stress_components.append((-spread).clip(0.0, None) / 100.0)

    if len(stress_components) >= 2:
        composites = composites.with_columns(
            local_pl.mean_horizontal(*stress_components).alias("financial_stress_composite"),
        )

    # ===== Growth/Inflation Regime =====

    # Growth momentum: Composite of employment, industrial production, activity
    growth_components = []
    if "PAYEMS" in df.columns:
        # Month-over-month employment growth
        composites = composites.with_columns(
            (
                (local_pl.col("PAYEMS") - local_pl.col("PAYEMS").shift(30)) / local_pl.col("PAYEMS").shift(30)
            ).alias("payems_mom"),
        )
        growth_components.append(local_pl.col("payems_mom") * 100.0)  # To percentage

    if "INDPRO" in df.columns:
        composites = composites.with_columns(
            (
                (local_pl.col("INDPRO") - local_pl.col("INDPRO").shift(30)) / local_pl.col("INDPRO").shift(30)
            ).alias("indpro_mom"),
        )
        growth_components.append(local_pl.col("indpro_mom") * 100.0)

    if "CFNAI" in df.columns:
        growth_components.append(local_pl.col("CFNAI"))  # Already standardized

    if growth_components:
        composites = composites.with_columns(
            local_pl.mean_horizontal(*growth_components).alias("growth_momentum"),
        )

    # Inflation momentum: Composite of CPI, PCE, PPI
    inflation_components = []
    if "CPIAUCSL" in df.columns:
        composites = composites.with_columns(
            (
                (local_pl.col("CPIAUCSL") - local_pl.col("CPIAUCSL").shift(365))
                / local_pl.col("CPIAUCSL").shift(365)
            ).alias("cpi_yoy"),
        )
        inflation_components.append(local_pl.col("cpi_yoy") * 100.0)

    if "PCEPI" in df.columns:
        composites = composites.with_columns(
            (
                (local_pl.col("PCEPI") - local_pl.col("PCEPI").shift(365)) / local_pl.col("PCEPI").shift(365)
            ).alias("pce_yoy"),
        )
        inflation_components.append(local_pl.col("pce_yoy") * 100.0)

    if "PPIACO" in df.columns:
        composites = composites.with_columns(
            (
                (local_pl.col("PPIACO") - local_pl.col("PPIACO").shift(365)) / local_pl.col("PPIACO").shift(365)
            ).alias("ppi_yoy"),
        )
        inflation_components.append(local_pl.col("ppi_yoy") * 100.0)

    if inflation_components:
        composites = composites.with_columns(
            local_pl.mean_horizontal(*inflation_components).alias("inflation_momentum"),
        )

    # Regime signals
    if "growth_momentum" in composites.columns and "inflation_momentum" in composites.columns:
        # Stagflation: High inflation + weak growth
        composites = composites.with_columns(
            (
                (local_pl.col("inflation_momentum") > 3.0)
                & (local_pl.col("growth_momentum") < 0.0)
            )
            .cast(local_pl.Float64)
            .alias("stagflation_risk"),
        )

        # Goldilocks: Moderate growth + low inflation
        composites = composites.with_columns(
            (
                (local_pl.col("growth_momentum").is_between(1.0, 3.0))
                & (local_pl.col("inflation_momentum") < 2.5)
            )
            .cast(local_pl.Float64)
            .alias("goldilocks_score"),
        )

    # ===== FX Positioning =====
    if "DTWEXBGS" in df.columns:
        composites = composites.with_columns(
            local_pl.col("DTWEXBGS").alias("dollar_strength"),
        )

        # Dollar momentum: 3-month change
        composites = composites.with_columns(
            (
                (local_pl.col("DTWEXBGS") - local_pl.col("DTWEXBGS").shift(90))
                / local_pl.col("DTWEXBGS").shift(90)
            ).alias("dollar_momentum_3m"),
        )

    # FX stress: Volatility across major pairs (std of recent changes)
    fx_pairs = ["DEXUSAL", "DEXUSEU", "DEXJPUS"]
    available_fx = [col for col in fx_pairs if col in df.columns]

    if len(available_fx) >= 2:
        # Compute rolling volatility for each pair
        for pair in available_fx:
            composites = composites.with_columns(
                ((local_pl.col(pair) - local_pl.col(pair).shift(1)) / local_pl.col(pair).shift(1))
                .rolling_std(window_size=30)
                .alias(f"{pair}_vol"),
            )

        # FX stress: Average volatility across pairs
        vol_cols = [f"{pair}_vol" for pair in available_fx]
        composites = composites.with_columns(
            local_pl.mean_horizontal(*[local_pl.col(c) for c in vol_cols]).alias("fx_stress"),
        )

    # NEW: FX volatility composite (cross-pair realized vol)
    # Compute as average of rolling standard deviations across available pairs
    if len(available_fx) >= 2:
        fx_vol_components = []
        for pair in available_fx:
            fx_vol_components.append(
                ((local_pl.col(pair) - local_pl.col(pair).shift(1)) / local_pl.col(pair).shift(1))
                .rolling_std(window_size=30)
                .fill_null(0.0)
            )

        composites = composites.with_columns(
            local_pl.mean_horizontal(*fx_vol_components).alias("fx_volatility_composite"),
        )

    drop_candidates = [col for col in composites.columns if col.endswith("_vol")]
    if drop_candidates:
        keep = set(get_composite_feature_names())
        drop_cols = [col for col in drop_candidates if col not in keep]
        if drop_cols:
            composites = composites.drop(drop_cols)

    return composites


def get_composite_feature_names() -> list[str]:
    """
    Get list of all composite feature names that can be generated.

    Returns
    -------
    list[str]
        Feature names for all composites.

    """
    return [
        # Credit/Risk (8 new features)
        "credit_spread_ig",
        "credit_spread_hy",
        "credit_spread_hy_ig",
        "credit_spread_bbb_a",
        "credit_spread_ig_momentum",
        "credit_spread_hy_momentum",
        "credit_distress_index",
        "ted_spread",
        "credit_risk_index",
        # Duration/Term (7 new features)
        "term_spread",
        "term_spread_5s30s",
        "term_spread_2s30s",
        "curve_curvature",
        "real_yield_10y",
        "real_term_premium",
        "yield_curve_slope",
        "fed_policy_stance",
        # Liquidity (6 new features)
        "fed_balance_sheet",
        "qe_intensity",
        "bank_credit_growth_3m",
        "sofr_obfr_spread",
        "financial_stress_composite",
        "liquidity_index",
        # Growth/Inflation
        "payems_mom",
        "indpro_mom",
        "growth_momentum",
        "cpi_yoy",
        "pce_yoy",
        "ppi_yoy",
        "inflation_momentum",
        "stagflation_risk",
        "goldilocks_score",
        # FX (4 new features)
        "dollar_strength",
        "dollar_momentum_3m",
        "fx_volatility_composite",
        "fx_stress",
    ]


def get_composite_series_requirements() -> tuple[str, ...]:
    """
    Return base macro series required to compute composite features.

    Returns
    -------
    tuple[str, ...]
        Series identifiers that should be present for composite calculations.
    """
    return (
        "BAMLC0A0CM",
        "BAMLH0A0HYM2",
        "TEDRATE",
        "VIXCLS",
        "T10Y2Y",
        "DGS10",
        "DGS2",
        "DGS5",
        "DGS30",
        "DFII10",
        "FEDFUNDS",
        "WALCL",
        "TOTBKCR",
        "PAYEMS",
        "INDPRO",
        "CFNAI",
        "CPIAUCSL",
        "PCEPI",
        "PPIACO",
        "DTWEXBGS",
        "DEXUSAL",
        "DEXUSEU",
        "DEXJPUS",
    )
