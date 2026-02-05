"""Contract tests for canonical feature family column sets."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ml.features.l2_aggregate import L2_MINUTE_COLUMNS
from ml.features.micro_aggregate import MICRO_COLUMNS
from ml.features.pipeline import TransformSpec
from ml.features.pipeline import transform_feature_names


pytestmark = pytest.mark.contracts


EXPECTED_RETURNS: list[str] = ["return_1", "return_5", "return_10", "return_20"]
EXPECTED_MOMENTUM: list[str] = ["momentum_5", "momentum_10", "momentum_20"]
EXPECTED_VOLATILITY: list[str] = ["volatility_5", "volatility_20"]
EXPECTED_VOLUME_RATIO: list[str] = ["volume_ratio_5", "volume_ratio_10", "volume_ratio_20"]
EXPECTED_CORE_INDICATORS: list[str] = [
    "rsi",
    "rsi_overbought",
    "rsi_oversold",
    "bb_width",
    "bb_position",
    "atr_normalized",
    "ema_fast_dist",
    "ema_slow_dist",
    "ema_cross",
    "macd_line",
    "macd_signal",
    "macd_diff",
    "price_position_20",
    "hl_spread",
]
EXPECTED_MICROSTRUCTURE: list[str] = [
    "spread_mean",
    "spread_std",
    "spread_relative",
    "size_imbalance_mean",
    "size_imbalance_std",
    "mid_return_std",
    "mid_return_autocorr",
]
EXPECTED_TRADE_FLOW: list[str] = [
    "trade_flow_imbalance",
    "vwap",
    "trade_intensity",
    "avg_price_impact",
]
EXPECTED_CALENDAR: list[str] = [
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "days_to_month_end",
    "days_from_month_start",
    "is_trading_day",
    "is_market_hours",
    "is_pre_market",
    "is_after_hours",
    "minutes_to_close",
]
EXPECTED_EVENT_TYPES: list[str] = [
    "earnings",
    "fed_meeting",
    "economic_release",
    "options_expiry",
]
EXPECTED_EVENT_HORIZONS: list[int] = [1, 4, 24, 72]
EXPECTED_EVENT_SCHEDULE: list[str] = [
    *[
        name
        for event in EXPECTED_EVENT_TYPES
        for name in (
            [
                f"hours_to_{event}",
                f"has_{event}_in_24h",
                f"has_{event}_in_week",
            ]
            + [f"{event}_within_{hours}h" for hours in EXPECTED_EVENT_HORIZONS]
        )
    ],
    "total_events_24h",
    "total_events_week",
    "event_density_24h",
    "event_density_week",
    "is_triple_witching",
    "is_fomc_week",
    "is_earnings_season",
    "is_holiday_week",
    "days_to_next_holiday",
]
EXPECTED_MACRO_INDICATORS: list[str] = [
    "vix",
    "dxy",
    "treasury_10y",
    "treasury_2y",
    "term_spread",
    "fed_funds_rate",
]
EXPECTED_MACRO_INDICATORS_FEATURES: list[str] = [
    *[
        name
        for indicator in EXPECTED_MACRO_INDICATORS
        for name in (
            [
                indicator,
                f"{indicator}_change_1d",
                f"{indicator}_change_5d",
                f"{indicator}_zscore_20d",
                f"{indicator}_zscore_60d",
            ]
        )
    ],
    "vix_regime",
    "yield_curve_regime",
    "rate_cycle_phase",
]
EXPECTED_STATIC_COVARIATES: list[str] = [
    "tick_size",
    "lot_size",
    "contract_size",
    "min_price_increment",
    "margin_initial",
    "margin_maintenance",
    "exchange",
    "asset_class",
    "currency",
    "fee_class",
    "market_segment",
]
EXPECTED_INSTRUMENT_METADATA: list[str] = [
    "duration_bucket",
    "issuer_type",
    "liquidity_tier",
]
EXPECTED_INSTRUMENT_METADATA_EXTENDED: list[str] = [
    "duration_bucket",
    "issuer_type",
    "liquidity_tier",
    "region_encoded",
    "sector_encoded",
    "rating_encoded",
]
EXPECTED_KELTNER: list[str] = ["keltner_width", "keltner_position"]
EXPECTED_OBV: list[str] = ["obv_norm"]
EXPECTED_EWMA_BETA: list[str] = ["ewma_beta_market"]
EXPECTED_ZSCORE_SPREAD: list[str] = ["zscore_spread_pair"]
EXPECTED_MACRO_COMPOSITES: list[str] = [
    "credit_spread_ig",
    "credit_spread_hy",
    "credit_spread_hy_ig",
    "credit_spread_bbb_a",
    "credit_spread_ig_momentum",
    "credit_spread_hy_momentum",
    "credit_distress_index",
    "ted_spread",
    "credit_risk_index",
    "term_spread",
    "term_spread_5s30s",
    "term_spread_2s30s",
    "curve_curvature",
    "real_yield_10y",
    "real_term_premium",
    "yield_curve_slope",
    "fed_policy_stance",
    "fed_balance_sheet",
    "qe_intensity",
    "bank_credit_growth_3m",
    "sofr_obfr_spread",
    "financial_stress_composite",
    "liquidity_index",
    "payems_mom",
    "indpro_mom",
    "growth_momentum",
    "cpi_yoy",
    "pce_yoy",
    "ppi_yoy",
    "inflation_momentum",
    "stagflation_risk",
    "goldilocks_score",
    "dollar_strength",
    "dollar_momentum_3m",
    "fx_volatility_composite",
    "fx_stress",
]
EXPECTED_L2_MINUTE_COLUMNS: list[str] = [
    "timestamp",
    "midprice",
    "spread_bps",
    "microprice_bps",
    "depth_imbalance_top1",
    "depth_imbalance_top3",
    "depth_imbalance_top5",
    "depth_imbalance_top10",
    "dwp_bps_top1",
    "dwp_bps_top3",
    "dwp_bps_top5",
    "dwp_bps_top10",
    "bid_slope_top1",
    "bid_slope_top3",
    "bid_slope_top5",
    "bid_slope_top10",
    "ask_slope_top1",
    "ask_slope_top3",
    "ask_slope_top5",
    "ask_slope_top10",
]


def _assert_transform_columns(
    name: str,
    params: dict[str, object],
    expected: Sequence[str],
) -> None:
    spec = TransformSpec(name=name, params=params)
    names = transform_feature_names(spec)
    assert names == list(expected)
    assert len(names) == len(set(names)), f"Duplicate feature names for {name}"


def test_returns_feature_family_columns_when_defaults_match_expected() -> None:
    """Returns transform emits canonical columns with default periods."""
    _assert_transform_columns("returns", {}, EXPECTED_RETURNS)


def test_momentum_feature_family_columns_when_defaults_match_expected() -> None:
    """Momentum transform emits canonical columns with default periods."""
    _assert_transform_columns("momentum", {}, EXPECTED_MOMENTUM)


def test_volatility_feature_family_columns_when_defaults_match_expected() -> None:
    """Volatility transform emits canonical columns."""
    _assert_transform_columns("volatility", {}, EXPECTED_VOLATILITY)


def test_volume_ratio_feature_family_columns_when_defaults_match_expected() -> None:
    """Volume ratio transform emits canonical columns."""
    _assert_transform_columns("volume_ratio", {}, EXPECTED_VOLUME_RATIO)


def test_core_indicator_feature_family_columns_when_defaults_match_expected() -> None:
    """Core indicators transform emits canonical columns."""
    _assert_transform_columns("core_indicators", {}, EXPECTED_CORE_INDICATORS)


def test_microstructure_feature_family_columns_when_defaults_match_expected() -> None:
    """Microstructure transform emits canonical columns."""
    _assert_transform_columns("microstructure", {}, EXPECTED_MICROSTRUCTURE)


def test_trade_flow_feature_family_columns_when_defaults_match_expected() -> None:
    """Trade flow transform emits canonical columns."""
    _assert_transform_columns("trade_flow", {}, EXPECTED_TRADE_FLOW)


def test_calendar_feature_family_columns_when_defaults_match_expected() -> None:
    """Calendar transform emits canonical columns."""
    _assert_transform_columns("calendar", {}, EXPECTED_CALENDAR)


def test_event_schedule_feature_family_columns_when_defaults_match_expected() -> None:
    """Event schedule transform emits canonical columns."""
    _assert_transform_columns("event_schedule", {}, EXPECTED_EVENT_SCHEDULE)


def test_macro_feature_family_columns_when_revisions_disabled_returns_base_values() -> None:
    """Macro transform emits base columns when revisions are disabled."""
    series_ids = ["PAYEMS", "UNRATE"]
    expected = [f"{series}__value_real_time" for series in series_ids]
    _assert_transform_columns(
        "macro",
        {"series_ids": series_ids, "include_revisions": False},
        expected,
    )


def test_macro_feature_family_columns_when_core_revisions_enabled() -> None:
    """Macro transform emits core revision columns when enabled."""
    series_ids = ["PAYEMS", "UNRATE"]
    expected: list[str] = []
    for series in series_ids:
        expected.extend(
            [
                f"{series}__value_real_time",
                f"{series}_prior_1m",
                f"{series}_revision_1m",
                f"{series}_mom_1m",
                f"{series}_pct_1m",
                f"{series}_net_signal_1m",
            ],
        )
    _assert_transform_columns(
        "macro",
        {"series_ids": series_ids, "include_revisions": True, "revision_mode": "core"},
        expected,
    )


def test_macro_deltas_feature_family_columns_when_series_configured() -> None:
    """Macro delta transform emits canonical delta columns."""
    series_ids = ["PAYEMS", "UNRATE"]
    expected = [f"{series}_delta_1d" for series in series_ids]
    _assert_transform_columns("macro_deltas", {"series_ids": series_ids}, expected)


def test_macro_composites_feature_family_columns_when_defaults_match_expected() -> None:
    """Macro composites transform emits canonical composite columns."""
    _assert_transform_columns("macro_composites", {}, EXPECTED_MACRO_COMPOSITES)


def test_macro_indicators_feature_family_columns_when_defaults_match_expected() -> None:
    """Macro indicators transform emits canonical columns."""
    _assert_transform_columns("macro_indicators", {}, EXPECTED_MACRO_INDICATORS_FEATURES)


def test_static_covariates_feature_family_columns_when_defaults_match_expected() -> None:
    """Static covariates transform emits canonical columns."""
    _assert_transform_columns("static_covariates", {}, EXPECTED_STATIC_COVARIATES)


def test_instrument_metadata_feature_family_columns_when_defaults_match_expected() -> None:
    """Instrument metadata transform emits canonical base columns."""
    _assert_transform_columns("instrument_metadata", {}, EXPECTED_INSTRUMENT_METADATA)


def test_instrument_metadata_feature_family_columns_when_extended_enabled() -> None:
    """Instrument metadata transform appends optional columns when enabled."""
    _assert_transform_columns(
        "instrument_metadata",
        {"include_region": True, "include_sector": True, "include_rating": True},
        EXPECTED_INSTRUMENT_METADATA_EXTENDED,
    )


def test_keltner_feature_family_columns_when_defaults_match_expected() -> None:
    """Keltner transform emits canonical columns."""
    _assert_transform_columns("keltner", {}, EXPECTED_KELTNER)


def test_obv_feature_family_columns_when_defaults_match_expected() -> None:
    """OBV transform emits canonical columns."""
    _assert_transform_columns("obv", {}, EXPECTED_OBV)


def test_ewma_beta_feature_family_columns_when_defaults_match_expected() -> None:
    """EWMA beta transform emits canonical columns."""
    _assert_transform_columns("ewma_beta", {}, EXPECTED_EWMA_BETA)


def test_zscore_spread_feature_family_columns_when_defaults_match_expected() -> None:
    """Z-score spread transform emits canonical columns."""
    _assert_transform_columns("zscore_spread", {}, EXPECTED_ZSCORE_SPREAD)


def test_micro_aggregate_feature_columns_match_canonical_list() -> None:
    """Microstructure aggregate output columns remain canonical."""
    assert list(MICRO_COLUMNS) == [
        "midprice",
        "spread_bps",
        "quote_imbalance",
        "trade_imbalance",
        "realized_vol",
    ]


def test_l2_aggregate_feature_columns_match_canonical_list() -> None:
    """L2 aggregate output columns remain canonical."""
    assert list(L2_MINUTE_COLUMNS) == EXPECTED_L2_MINUTE_COLUMNS
