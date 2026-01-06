from __future__ import annotations

import pytest

from ml.training.teacher.tft_cli import _resolve_tft_feature_columns

pd = pytest.importorskip("pandas")


def test_resolve_tft_feature_columns_auto_static_categoricals() -> None:
    df = pd.DataFrame(
        {
            "instrument_id": ["A", "A", "B", "B"],
            "time_index": [1, 2, 1, 2],
            "y": [0, 1, 0, 1],
            "feature_num": [0.1, 0.2, 0.3, 0.4],
            "asset_class": ["ETF", "ETF", "Stock", "Stock"],
            "exchange": ["NYSE", "NYSE", "NASDAQ", "NASDAQ"],
        },
    )
    feature_names = ["feature_num", "asset_class", "exchange"]

    numeric, static, encoded = _resolve_tft_feature_columns(
        df,
        feature_names=feature_names,
        group_id_col="instrument_id",
        static_categoricals=None,
    )

    assert numeric == ["feature_num"]
    assert static == ["asset_class", "exchange"]
    assert encoded == []
    assert df["asset_class"].dtype.name == "category"
    assert df["exchange"].dtype.name == "category"


def test_resolve_tft_feature_columns_encodes_dynamic_categoricals() -> None:
    df = pd.DataFrame(
        {
            "instrument_id": ["A", "A", "A", "A"],
            "time_index": [1, 2, 3, 4],
            "y": [0, 1, 0, 1],
            "regime": ["risk_on", "risk_off", "risk_on", "risk_off"],
            "feature_num": [0.1, 0.2, 0.3, 0.4],
        },
    )
    feature_names = ["regime", "feature_num"]

    numeric, static, encoded = _resolve_tft_feature_columns(
        df,
        feature_names=feature_names,
        group_id_col="instrument_id",
        static_categoricals=None,
    )

    assert static == []
    assert "regime" in numeric
    assert encoded == ["regime"]
    assert pd.api.types.is_numeric_dtype(df["regime"])


def test_resolve_tft_feature_columns_coerces_numeric_strings() -> None:
    df = pd.DataFrame(
        {
            "instrument_id": ["A", "A", "A"],
            "time_index": [1, 2, 3],
            "y": [0, 1, 0],
            "feature_str": ["1.0", "2.0", "3.0"],
        },
    )
    feature_names = ["feature_str"]

    numeric, static, encoded = _resolve_tft_feature_columns(
        df,
        feature_names=feature_names,
        group_id_col="instrument_id",
        static_categoricals=None,
    )

    assert numeric == ["feature_str"]
    assert static == []
    assert encoded == []
    assert pd.api.types.is_numeric_dtype(df["feature_str"])


def test_resolve_tft_feature_columns_coerces_datetimes() -> None:
    df = pd.DataFrame(
        {
            "instrument_id": ["A", "A"],
            "time_index": [1, 2],
            "y": [0, 1],
            "vintage_ts": [pd.Timestamp("2024-01-01"), pd.NaT],
        },
    )
    feature_names = ["vintage_ts"]

    numeric, static, encoded = _resolve_tft_feature_columns(
        df,
        feature_names=feature_names,
        group_id_col="instrument_id",
        static_categoricals=None,
    )

    assert numeric == ["vintage_ts"]
    assert static == []
    assert encoded == []
    assert pd.api.types.is_numeric_dtype(df["vintage_ts"])
    assert df["vintage_ts"].isna().sum() == 1


def test_resolve_tft_feature_columns_fills_missing_static_categoricals() -> None:
    df = pd.DataFrame(
        {
            "instrument_id": ["A", "A", "B"],
            "time_index": [1, 2, 3],
            "y": [0, 1, 0],
            "asset_class": [None, None, "ETF"],
        },
    )
    feature_names = ["asset_class"]

    numeric, static, encoded = _resolve_tft_feature_columns(
        df,
        feature_names=feature_names,
        group_id_col="instrument_id",
        static_categoricals=["asset_class"],
    )

    assert numeric == []
    assert static == ["asset_class"]
    assert encoded == []
    assert df["asset_class"].isna().sum() == 0
