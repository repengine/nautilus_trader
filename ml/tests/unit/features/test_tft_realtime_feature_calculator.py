#!/usr/bin/env python3

"""
Unit tests for TFTRealtimeFeatureCalculator.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock

import numpy as np

from ml.features.common.tft_realtime_feature_calculator import (
    TFTRealtimeFeatureCalculator,
)


class MockBar:
    """
    Mock Nautilus Bar for TFT realtime calculator tests.
    """

    def __init__(
        self,
        *,
        close: float,
        high: float,
        low: float,
        volume: float,
        ts_event: int,
        instrument_id: str = "SPY.EQUS",
    ) -> None:
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.ts_event = ts_event
        self.ts_init = ts_event
        self.bar_type = MagicMock()
        self.bar_type.instrument_id = instrument_id


class MockIndicatorManager:
    """
    Minimal indicator manager stub with price history only.
    """

    def __init__(self) -> None:
        self.price_history: dict[str, list[float]] = {
            "closes": [],
            "highs": [],
            "lows": [],
            "volumes": [],
        }

    def update_from_bar(self, bar: MockBar) -> None:
        self.price_history["closes"].append(float(bar.close))
        self.price_history["highs"].append(float(bar.high))
        self.price_history["lows"].append(float(bar.low))
        self.price_history["volumes"].append(float(bar.volume))


def _build_feature_names() -> list[str]:
    return [
        "close",
        "return_1",
        "return_5",
        "return_20",
        "volume_ratio_20",
        "volatility_20",
        "price_sma_5",
        "price_sma_20",
        "price_position_20",
        "tick_size",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "dow_sin",
        "dow_cos",
        "is_market_hours",
        "is_pre_market",
        "is_after_hours",
    ]


def _rolling_return_std(closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        return 0.0
    returns: list[float] = []
    start = len(closes) - period
    for idx in range(start, len(closes)):
        prev_close = closes[idx - 1]
        if prev_close == 0:
            returns.append(0.0)
        else:
            returns.append((closes[idx] - prev_close) / prev_close)
    if len(returns) <= 1:
        return 0.0
    return float(np.std(np.array(returns, dtype=np.float64), ddof=1))


def test_tft_realtime_feature_calculator_values() -> None:
    feature_names = _build_feature_names()
    calculator = TFTRealtimeFeatureCalculator(feature_names)
    indicator_manager = MockIndicatorManager()

    base_dt = datetime(2024, 7, 2, 15, 30, tzinfo=UTC)
    base_ts = int(base_dt.timestamp() * 1_000_000_000)

    closes: list[float] = []
    last_features: np.ndarray | None = None

    for idx in range(25):
        close = 100.0 + idx
        bar = MockBar(
            close=close,
            high=close + 1.0,
            low=close - 1.0,
            volume=1000.0,
            ts_event=base_ts + idx * 60 * 1_000_000_000,
        )
        indicator_manager.update_from_bar(bar)
        closes.append(close)
        last_features = calculator.compute(bar, indicator_manager)

    assert last_features is not None
    features = last_features
    assert features.dtype == np.float32
    assert features.shape[0] == len(feature_names)

    idx_map: dict[str, int] = {name: i for i, name in enumerate(feature_names)}

    expected_close = closes[-1]
    expected_return_1 = (closes[-1] - closes[-2]) / closes[-2]
    expected_return_5 = (closes[-1] - closes[-6]) / closes[-6]
    expected_return_20 = (closes[-1] - closes[-21]) / closes[-21]
    expected_sma_5 = float(sum(closes[-5:]) / 5.0)
    expected_sma_20 = float(sum(closes[-20:]) / 20.0)
    expected_vol = _rolling_return_std(closes, 20)
    expected_volume_ratio = 1.0

    min_low = closes[-20] - 1.0
    max_high = closes[-1] + 1.0
    expected_price_position = (expected_close - min_low) / (max_high - min_low)
    expected_tick_size = 0.01

    last_dt = base_dt + timedelta(minutes=24)
    hour_angle = 2.0 * np.pi * ((last_dt.hour + last_dt.minute / 60.0) / 24.0)
    expected_hour_sin = float(np.sin(hour_angle))
    expected_hour_cos = float(np.cos(hour_angle))
    minute_angle = 2.0 * np.pi * (last_dt.minute / 60.0)
    expected_minute_sin = float(np.sin(minute_angle))
    expected_minute_cos = float(np.cos(minute_angle))
    dow_angle = 2.0 * np.pi * last_dt.weekday() / 7.0
    expected_dow_sin = float(np.sin(dow_angle))
    expected_dow_cos = float(np.cos(dow_angle))

    expected_is_market_open = 1.0 if 9 <= last_dt.hour < 16 else 0.0
    expected_is_premarket = 1.0 if 4 <= last_dt.hour < 9 else 0.0
    expected_is_aftermarket = 1.0 if 16 <= last_dt.hour < 20 else 0.0

    assert np.isclose(features[idx_map["close"]], expected_close)
    assert np.isclose(features[idx_map["return_1"]], expected_return_1)
    assert np.isclose(features[idx_map["return_5"]], expected_return_5)
    assert np.isclose(features[idx_map["return_20"]], expected_return_20)
    assert np.isclose(features[idx_map["price_sma_5"]], expected_sma_5)
    assert np.isclose(features[idx_map["price_sma_20"]], expected_sma_20)
    assert np.isclose(features[idx_map["volatility_20"]], expected_vol)
    assert np.isclose(features[idx_map["volume_ratio_20"]], expected_volume_ratio)
    assert np.isclose(features[idx_map["price_position_20"]], expected_price_position)
    assert np.isclose(features[idx_map["tick_size"]], expected_tick_size)

    assert np.isclose(features[idx_map["hour_sin"]], expected_hour_sin)
    assert np.isclose(features[idx_map["hour_cos"]], expected_hour_cos)
    assert np.isclose(features[idx_map["minute_sin"]], expected_minute_sin)
    assert np.isclose(features[idx_map["minute_cos"]], expected_minute_cos)
    assert np.isclose(features[idx_map["dow_sin"]], expected_dow_sin)
    assert np.isclose(features[idx_map["dow_cos"]], expected_dow_cos)
    assert np.isclose(features[idx_map["is_market_hours"]], expected_is_market_open)
    assert np.isclose(features[idx_map["is_pre_market"]], expected_is_premarket)
    assert np.isclose(features[idx_map["is_after_hours"]], expected_is_aftermarket)
