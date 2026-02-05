"""
Realtime TFT feature calculator.

Computes a constrained subset of TFT-aligned features for online inference, mirroring
the FeatureAlignmentComponent + KnownFutureFeatureComponent outputs.

"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ml.common.safe_math import safe_divide
from ml.data.common.static_features import resolve_tick_size


if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

    from ml.features.indicators import IndicatorManager


logger = logging.getLogger(__name__)

_MINUTES_PER_DAY = 24 * 60
_SECONDS_PER_DAY = 24 * 60 * 60

_MINUTE_INDEX = np.arange(_MINUTES_PER_DAY, dtype=np.int32)
_HOUR_ANGLE = (2.0 * math.pi * _MINUTE_INDEX / float(_MINUTES_PER_DAY)).astype(np.float64)
_HOUR_SIN = np.sin(_HOUR_ANGLE).astype(np.float32)
_HOUR_COS = np.cos(_HOUR_ANGLE).astype(np.float32)
_MINUTE_ANGLE = (2.0 * math.pi * (_MINUTE_INDEX % 60) / 60.0).astype(np.float64)
_MINUTE_SIN = np.sin(_MINUTE_ANGLE).astype(np.float32)
_MINUTE_COS = np.cos(_MINUTE_ANGLE).astype(np.float32)

_MARKET_HOURS = ((_MINUTE_INDEX >= 9 * 60) & (_MINUTE_INDEX < 16 * 60)).astype(np.float32)
_PRE_MARKET = ((_MINUTE_INDEX >= 4 * 60) & (_MINUTE_INDEX < 9 * 60)).astype(np.float32)
_AFTER_HOURS = ((_MINUTE_INDEX >= 16 * 60) & (_MINUTE_INDEX < 20 * 60)).astype(np.float32)

_DOW_INDEX = np.arange(7, dtype=np.int32)
_DOW_ANGLE = (2.0 * math.pi * _DOW_INDEX / 7.0).astype(np.float64)
_DOW_SIN = np.sin(_DOW_ANGLE).astype(np.float32)
_DOW_COS = np.cos(_DOW_ANGLE).astype(np.float32)


@dataclass(slots=True)
class _ParsedFeatures:
    return_indices: dict[int, int]
    sma_indices: dict[int, int]
    volatility_indices: dict[int, int]
    volume_ratio_indices: dict[int, int]
    price_position_indices: dict[int, int]
    direct_indices: dict[str, int]
    required_history_bars: int


class TFTRealtimeFeatureCalculator:
    """
    Compute TFT-aligned features for online inference (hot path).

    Parameters
    ----------
    feature_names : list[str]
        Ordered feature names to produce. The calculator validates
        that all names are supported by the realtime implementation.

    """

    _DIRECT_FEATURES: set[str] = {
        "close",
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
    }

    def __init__(self, feature_names: list[str]) -> None:
        if not feature_names:
            raise ValueError("feature_names must be non-empty")
        self._feature_names = list(feature_names)
        self._feature_index = {name: idx for idx, name in enumerate(self._feature_names)}
        self._parsed = self._parse_features(self._feature_names)
        self._feature_buffer = np.zeros(len(self._feature_names), dtype=np.float32)
        self._tick_size_cache: dict[str, float] = {}

    @property
    def feature_names(self) -> list[str]:
        """
        Return the ordered feature names for this calculator.

        Returns
        -------
        list[str]
            Feature names in output order.

        """
        return list(self._feature_names)

    @property
    def required_history_bars(self) -> int:
        """
        Return the minimum bars required for non-zero features.

        Returns
        -------
        int
            Minimum history length before computing non-zero values.

        """
        return self._parsed.required_history_bars

    def compute(
        self,
        bar: Bar,
        indicator_manager: IndicatorManager,
    ) -> npt.NDArray[np.float32]:
        """
        Compute TFT-aligned feature vector for the latest bar.

        Parameters
        ----------
        bar : Bar
            Current bar containing OHLCV and timestamps.
        indicator_manager : IndicatorManager
            Indicator manager with updated price history.

        Returns
        -------
        npt.NDArray[np.float32]
            Feature vector aligned to ``feature_names``.

        """
        buffer = self._feature_buffer
        buffer.fill(0.0)

        close = float(bar.close)
        volume = float(bar.volume)

        closes = indicator_manager.price_history.get("closes") or []
        highs = indicator_manager.price_history.get("highs") or []
        lows = indicator_manager.price_history.get("lows") or []
        volumes = indicator_manager.price_history.get("volumes") or []

        direct = self._parsed.direct_indices
        if "close" in direct:
            buffer[direct["close"]] = close

        if "tick_size" in direct:
            buffer[direct["tick_size"]] = self._resolve_tick_size(bar)

        self._fill_time_features(buffer, direct, bar)
        self._fill_return_features(buffer, close, closes)
        self._fill_sma_features(buffer, closes)
        self._fill_volatility_features(buffer, closes)
        self._fill_volume_ratio_features(buffer, volume, volumes)
        self._fill_price_position_features(buffer, close, highs, lows)

        return buffer

    def _resolve_tick_size(self, bar: Bar) -> float:
        instrument_id = ""
        if hasattr(bar, "bar_type") and hasattr(bar.bar_type, "instrument_id"):
            instrument_id = str(bar.bar_type.instrument_id)
        elif hasattr(bar, "instrument_id"):
            instrument_id = str(bar.instrument_id)

        cached = self._tick_size_cache.get(instrument_id)
        if cached is not None:
            return cached

        tick_size = resolve_tick_size(instrument_id)
        self._tick_size_cache[instrument_id] = tick_size
        return tick_size

    @staticmethod
    def _parse_features(feature_names: list[str]) -> _ParsedFeatures:
        return_indices: dict[int, int] = {}
        sma_indices: dict[int, int] = {}
        volatility_indices: dict[int, int] = {}
        volume_ratio_indices: dict[int, int] = {}
        price_position_indices: dict[int, int] = {}
        direct_indices: dict[str, int] = {}

        unsupported: list[str] = []
        for idx, name in enumerate(feature_names):
            if name in TFTRealtimeFeatureCalculator._DIRECT_FEATURES:
                direct_indices[name] = idx
                continue

            if name.startswith("return_"):
                period = _parse_suffix_period(name, "return_")
                if period is not None:
                    return_indices[period] = idx
                    continue

            if name.startswith("price_sma_"):
                period = _parse_suffix_period(name, "price_sma_")
                if period is not None:
                    sma_indices[period] = idx
                    continue

            if name.startswith("volatility_"):
                period = _parse_suffix_period(name, "volatility_")
                if period is not None:
                    volatility_indices[period] = idx
                    continue

            if name.startswith("volume_ratio_"):
                period = _parse_suffix_period(name, "volume_ratio_")
                if period is not None:
                    volume_ratio_indices[period] = idx
                    continue

            if name.startswith("price_position_"):
                period = _parse_suffix_period(name, "price_position_")
                if period is not None:
                    price_position_indices[period] = idx
                    continue

            unsupported.append(name)

        if unsupported:
            raise ValueError(
                "Unsupported TFT realtime features: " f"{sorted(unsupported)}",
            )

        required = _compute_required_history(
            return_indices,
            sma_indices,
            volatility_indices,
            volume_ratio_indices,
            price_position_indices,
        )

        return _ParsedFeatures(
            return_indices=return_indices,
            sma_indices=sma_indices,
            volatility_indices=volatility_indices,
            volume_ratio_indices=volume_ratio_indices,
            price_position_indices=price_position_indices,
            direct_indices=direct_indices,
            required_history_bars=required,
        )

    def _fill_time_features(
        self,
        buffer: npt.NDArray[np.float32],
        direct_indices: dict[str, int],
        bar: Bar,
    ) -> None:
        ts_event = int(getattr(bar, "ts_event", 0) or 0)
        if ts_event <= 0:
            return
        seconds = ts_event // 1_000_000_000
        minute_of_day = int((seconds // 60) % _MINUTES_PER_DAY)
        buffer_idx = direct_indices

        if "hour_sin" in buffer_idx:
            buffer[buffer_idx["hour_sin"]] = _HOUR_SIN[minute_of_day]
        if "hour_cos" in buffer_idx:
            buffer[buffer_idx["hour_cos"]] = _HOUR_COS[minute_of_day]
        if "minute_sin" in buffer_idx:
            buffer[buffer_idx["minute_sin"]] = _MINUTE_SIN[minute_of_day]
        if "minute_cos" in buffer_idx:
            buffer[buffer_idx["minute_cos"]] = _MINUTE_COS[minute_of_day]
        if "is_market_hours" in buffer_idx:
            buffer[buffer_idx["is_market_hours"]] = _MARKET_HOURS[minute_of_day]
        if "is_pre_market" in buffer_idx:
            buffer[buffer_idx["is_pre_market"]] = _PRE_MARKET[minute_of_day]
        if "is_after_hours" in buffer_idx:
            buffer[buffer_idx["is_after_hours"]] = _AFTER_HOURS[minute_of_day]

        if "dow_sin" in buffer_idx or "dow_cos" in buffer_idx:
            days_since_epoch = int(seconds // _SECONDS_PER_DAY)
            dow = int((days_since_epoch + 3) % 7)
            if "dow_sin" in buffer_idx:
                buffer[buffer_idx["dow_sin"]] = _DOW_SIN[dow]
            if "dow_cos" in buffer_idx:
                buffer[buffer_idx["dow_cos"]] = _DOW_COS[dow]

    def _fill_return_features(
        self,
        buffer: npt.NDArray[np.float32],
        close: float,
        closes: list[float],
    ) -> None:
        for period, idx in self._parsed.return_indices.items():
            if len(closes) >= period + 1:
                prev_close = closes[-(period + 1)]
                value = safe_divide(close - prev_close, prev_close)
                buffer[idx] = float(value)
            else:
                buffer[idx] = 0.0

    def _fill_sma_features(
        self,
        buffer: npt.NDArray[np.float32],
        closes: list[float],
    ) -> None:
        for period, idx in self._parsed.sma_indices.items():
            if len(closes) >= period:
                start = len(closes) - period
                total = 0.0
                for val in closes[start:]:
                    total += float(val)
                buffer[idx] = float(total / period)
            else:
                buffer[idx] = 0.0

    def _fill_volatility_features(
        self,
        buffer: npt.NDArray[np.float32],
        closes: list[float],
    ) -> None:
        for period, idx in self._parsed.volatility_indices.items():
            buffer[idx] = float(_rolling_return_std(closes, period))

    def _fill_volume_ratio_features(
        self,
        buffer: npt.NDArray[np.float32],
        volume: float,
        volumes: list[float],
    ) -> None:
        for period, idx in self._parsed.volume_ratio_indices.items():
            if len(volumes) >= period:
                start = len(volumes) - period
                total = 0.0
                for val in volumes[start:]:
                    total += float(val)
                mean = total / period if period > 0 else 0.0
                buffer[idx] = float(safe_divide(volume, mean)) if mean > 0 else 0.0
            else:
                buffer[idx] = 0.0

    def _fill_price_position_features(
        self,
        buffer: npt.NDArray[np.float32],
        close: float,
        highs: list[float],
        lows: list[float],
    ) -> None:
        for period, idx in self._parsed.price_position_indices.items():
            if len(highs) >= period and len(lows) >= period:
                start = min(len(highs), len(lows)) - period
                min_low = float(lows[start])
                max_high = float(highs[start])
                for i in range(start + 1, start + period):
                    low_val = float(lows[i])
                    high_val = float(highs[i])
                    if low_val < min_low:
                        min_low = low_val
                    if high_val > max_high:
                        max_high = high_val
                if max_high > min_low:
                    buffer[idx] = float((close - min_low) / (max_high - min_low))
                else:
                    buffer[idx] = 0.0
            else:
                buffer[idx] = 0.0


def _parse_suffix_period(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _compute_required_history(
    return_indices: dict[int, int],
    sma_indices: dict[int, int],
    volatility_indices: dict[int, int],
    volume_ratio_indices: dict[int, int],
    price_position_indices: dict[int, int],
) -> int:
    required = 1
    if return_indices:
        required = max(required, max(return_indices.keys()) + 1)
    if sma_indices:
        required = max(required, max(sma_indices.keys()))
    if volatility_indices:
        required = max(required, max(volatility_indices.keys()) + 1)
    if volume_ratio_indices:
        required = max(required, max(volume_ratio_indices.keys()))
    if price_position_indices:
        required = max(required, max(price_position_indices.keys()))
    return required


def _rolling_return_std(closes: list[float], period: int) -> float:
    if period <= 1:
        return 0.0
    if len(closes) < period + 1:
        return 0.0

    start_idx = len(closes) - period
    sum_ret = 0.0
    sumsq_ret = 0.0
    count = 0

    for i in range(start_idx, len(closes)):
        prev_close = closes[i - 1]
        ret = safe_divide(float(closes[i]) - float(prev_close), float(prev_close))
        sum_ret += ret
        sumsq_ret += ret * ret
        count += 1

    if count <= 1:
        return 0.0
    mean = sum_ret / count
    var = (sumsq_ret - count * mean * mean) / (count - 1)
    if var < 0.0:
        var = 0.0
    return float(math.sqrt(var))


__all__ = ["TFTRealtimeFeatureCalculator"]
