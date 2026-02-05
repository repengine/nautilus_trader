#!/usr/bin/env python3

"""
Declarative feature pipeline scaffolding.

Provides a minimal transform catalog, pipeline spec, and runner utilities for schema
computation, signature hashing, and capability gating. The execution of batch/online
math remains sourced from FeatureEngineer for now.

"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Protocol, cast

from ml.config.constants import IndicatorNames
from ml.registry.base import DataRequirements


class FeatureTransform(Protocol):
    """
    Protocol for feature transform plugins.
    """

    name: str

    def feature_names(self, params: Mapping[str, Any]) -> list[str]: ...

    def requires(self) -> DataRequirements:
        """
        Return required data level for this transform (used for gating).
        """
        ...


class _ReturnsTransform:
    name = "returns"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        periods: Iterable[int] = params.get("periods", [1, 5, 10, 20])
        return [f"return_{int(p)}" for p in periods]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


class _MomentumTransform:
    name = "momentum"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        periods: Iterable[int] = params.get("periods", [5, 10, 20])
        return [f"momentum_{int(p)}" for p in periods]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


class _VolatilityTransform:
    name = "volatility"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        # Fixed names aligned with engineering.py
        return [
            IndicatorNames.VOLATILITY_5,
            IndicatorNames.VOLATILITY_20,
        ]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


class _VolumeRatioTransform:
    name = "volume_ratio"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        periods: Iterable[int] = params.get("periods", [5, 10, 20])
        return [f"volume_ratio_{int(p)}" for p in periods]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


class _CoreIndicatorsTransform:
    name = "core_indicators"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        # Mirrors engineering.py core indicator outputs using shared constants
        return [
            IndicatorNames.RSI,
            IndicatorNames.RSI_OVERBOUGHT,
            IndicatorNames.RSI_OVERSOLD,
            IndicatorNames.BB_WIDTH,
            IndicatorNames.BB_POSITION,
            "atr_normalized",
            "ema_fast_dist",
            "ema_slow_dist",
            "ema_cross",
            IndicatorNames.MACD_LINE,
            IndicatorNames.MACD_SIGNAL,
            IndicatorNames.MACD_DIFF,
            "price_position_20",
            "hl_spread",
        ]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


_CATALOG: dict[str, FeatureTransform] = {
    _ReturnsTransform().name: _ReturnsTransform(),
    _MomentumTransform().name: _MomentumTransform(),
    _VolatilityTransform().name: _VolatilityTransform(),
    _VolumeRatioTransform().name: _VolumeRatioTransform(),
    _CoreIndicatorsTransform().name: _CoreIndicatorsTransform(),
}


def register_transform(transform: FeatureTransform) -> None:
    _CATALOG[transform.name] = transform


# Optional/richer transforms for teacher/offline pipelines
class _KeltnerTransform:
    name = "keltner"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        return ["keltner_width", "keltner_position"]

    def requires(self) -> DataRequirements:
        # Requires at least high/low/close + ATR/EMA; treat as L1_L2 to gate in student
        return DataRequirements.L1_L2


class _OBVTransform:
    name = "obv"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        return ["obv_norm"]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_L2


class _MicrostructureTransform:
    name = "microstructure"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        return [
            "spread_mean",
            "spread_std",
            "spread_relative",
            "size_imbalance_mean",
            "size_imbalance_std",
            "mid_return_std",
            "mid_return_autocorr",
        ]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_L2


class _TradeFlowTransform:
    name = "trade_flow"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        return [
            "trade_flow_imbalance",
            "vwap",
            "trade_intensity",
            "avg_price_impact",
        ]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_L2_L3


class _MacroTransform:
    """
    Macro features from ALFRED/FRED with revision awareness.

    Provides economic indicators with optional revision features that capture
    what traders see (headline + revisions + net signals).

    """

    name = "macro"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        """
        Generate macro feature names based on params.

        Parameters
        ----------
        params : dict
            series_ids: List of FRED series (e.g., ["PAYEMS", "UNRATE"])
            include_revisions: Whether to include revision features (default: False)
            revision_mode: "minimal", "core", or "full" (default: "core")

        Returns
        -------
        list[str]
            Feature names in order.

        """
        series_ids: list[str] = params.get("series_ids", [])
        include_revisions: bool = params.get("include_revisions", False)
        revision_mode: str = params.get("revision_mode", "core")

        if not series_ids:
            return []

        feature_names: list[str] = []

        for series_id in series_ids:
            # Base value (always included)
            feature_names.append(f"{series_id}__value_real_time")

            # Minimal mode features
            if include_revisions and revision_mode in ["minimal", "core", "full"]:
                feature_names.append(f"{series_id}_prior_1m")
                feature_names.append(f"{series_id}_revision_1m")

            # Core mode features
            if include_revisions and revision_mode in ["core", "full"]:
                feature_names.extend([
                    f"{series_id}_mom_1m",
                    f"{series_id}_pct_1m",
                    f"{series_id}_net_signal_1m",
                ])

            # Full mode features
            if include_revisions and revision_mode == "full":
                feature_names.extend([
                    f"{series_id}_prior_3m",
                    f"{series_id}_prior_12m",
                    f"{series_id}_mom_3m",
                    f"{series_id}_mom_12m",
                    f"{series_id}_pct_12m",
                    f"{series_id}_revision_3m",
                ])

        return feature_names

    def requires(self) -> DataRequirements:
        """Macro features are L1_ONLY (no order book needed)."""
        return DataRequirements.L1_ONLY


class _MacroDeltasTransform:
    """
    Macro delta features for canonical batch/stream parity.

    Emits per-series 1-day deltas for configured macro series.
    """

    name = "macro_deltas"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        series_ids: list[str] = params.get("series_ids", [])
        if not series_ids:
            return []
        return [f"{series_id}_delta_1d" for series_id in series_ids]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


class _CalendarTransform:
    """
    Calendar-based known-future features for TFT models.

    These features are deterministically known in advance based on calendar time, making
    them suitable for TFT's known-future input category.

    """

    name = "calendar"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        """
        Generate calendar feature names based on params.

        Parameters
        ----------
        params : dict
            encoding: Type of encoding ('cyclic', 'fourier', 'onehot')
            granularity: Time granularity ('minute', 'hour', 'day')

        """
        encoding = params.get("encoding", "cyclic")
        granularity = params.get("granularity", "hour")

        base_features = []

        # Time-of-day features
        if granularity in ["minute", "hour"]:
            if encoding == "cyclic":
                base_features.extend(["hour_sin", "hour_cos", "minute_sin", "minute_cos"])
            elif encoding == "fourier":
                n_harmonics = params.get("n_harmonics", 3)
                for h in range(1, n_harmonics + 1):
                    base_features.extend(
                        [
                            f"hour_sin_{h}",
                            f"hour_cos_{h}",
                        ],
                    )
            else:  # onehot
                base_features.extend([f"hour_{h}" for h in range(24)])
                if granularity == "minute":
                    base_features.extend([f"minute_{m}" for m in range(0, 60, 15)])

        # Day-of-week features
        if encoding == "cyclic":
            base_features.extend(["dow_sin", "dow_cos"])
        else:  # onehot
            base_features.extend([f"dow_{d}" for d in range(7)])

        # Month features
        if encoding == "cyclic":
            base_features.extend(["month_sin", "month_cos"])
        else:  # onehot
            base_features.extend([f"month_{m}" for m in range(1, 13)])

        # Additional calendar indicators and session flags
        base_features.extend(
            [
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
            ],
        )

        return base_features

    def requires(self) -> DataRequirements:
        # Calendar features only need L1 data (timestamps)
        return DataRequirements.L1_ONLY


class _EventScheduleTransform:
    """
    Scheduled event features for TFT known-future inputs.

    Captures information about upcoming scheduled events like earnings releases,
    economic data releases, Fed meetings, options expiry, etc.

    """

    name = "event_schedule"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        """
        Generate event schedule feature names.

        Parameters
        ----------
        params : dict
            event_types: List of event types to track
            horizon_hours: Hours ahead to look for events

        """
        event_types = params.get(
            "event_types",
            [
                "earnings",
                "fed_meeting",
                "economic_release",
                "options_expiry",
            ],
        )

        horizon_hours = params.get("horizon_hours", [1, 4, 24, 72])

        features = []

        # Time to next event features
        for event in event_types:
            features.append(f"hours_to_{event}")
            features.append(f"has_{event}_in_24h")
            features.append(f"has_{event}_in_week")

            # Event proximity features at different horizons
            for h in horizon_hours:
                features.append(f"{event}_within_{h}h")

        # Aggregate event density
        features.extend(
            [
                "total_events_24h",
                "total_events_week",
                "event_density_24h",
                "event_density_week",
            ],
        )

        # Special trading conditions
        features.extend(
            [
                "is_triple_witching",
                "is_fomc_week",
                "is_earnings_season",
                "is_holiday_week",
                "days_to_next_holiday",
            ],
        )

        return features

    def requires(self) -> DataRequirements:
        # Event schedules only need L1 data plus external calendar
        return DataRequirements.L1_ONLY


class _MacroIndicatorsTransform:
    """
    Macroeconomic indicators as known-future features.

    These are typically released on a schedule and their values are known until the next
    release, making them suitable for TFT known-future inputs.

    """

    name = "macro_indicators"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        """
        Generate macro indicator feature names.

        Parameters
        ----------
        params : dict
            indicators: List of macro indicators to include
            transformations: List of transformations to apply

        """
        indicators = params.get(
            "indicators",
            [
                "vix",
                "dxy",
                "treasury_10y",
                "treasury_2y",
                "term_spread",
                "fed_funds_rate",
            ],
        )

        transformations = params.get("transformations", ["level", "change", "z_score"])

        features = []

        for indicator in indicators:
            for transform in transformations:
                if transform == "level":
                    features.append(indicator)
                elif transform == "change":
                    features.append(f"{indicator}_change_1d")
                    features.append(f"{indicator}_change_5d")
                elif transform == "z_score":
                    features.append(f"{indicator}_zscore_20d")
                    features.append(f"{indicator}_zscore_60d")

        # Regime indicators
        features.extend(
            [
                "vix_regime",  # Low/Medium/High volatility
                "yield_curve_regime",  # Normal/Flat/Inverted
                "rate_cycle_phase",  # Hiking/Neutral/Cutting
            ],
        )

        return features

    def requires(self) -> DataRequirements:
        # Macro data typically comes from external sources
        return DataRequirements.L1_ONLY


class _StaticCovariatesTransform:
    """
    Static instrument metadata features for TFT models.

    These are per-instrument attributes that don't change over time,
    making them ideal for TFT's static covariate input category.

    Features include:
    - Instrument specifications (tick_size, lot_size, contract_size)
    - Market metadata (exchange, asset_class, currency)
    - Trading parameters (fee_class, margin_requirements)

    """

    name = "static_covariates"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        """
        Generate static covariate feature names.

        Parameters
        ----------
        params : dict
            numeric_features: List of numeric static features
            categorical_features: List of categorical static features

        """
        numeric_features = cast(
            list[str],
            params.get(
                "numeric_features",
                [
                    "tick_size",
                    "lot_size",
                    "contract_size",
                    "min_price_increment",
                    "margin_initial",
                    "margin_maintenance",
                ],
            ),
        )

        categorical_features = cast(
            list[str],
            params.get(
                "categorical_features",
                [
                    "exchange",
                    "asset_class",
                    "currency",
                    "fee_class",
                    "market_segment",
                ],
            ),
        )

        # For categoricals, we'll need encoding (one-hot or embedding indices)
        # This returns the base feature names; actual encoding happens in transform
        all_features: list[str] = list(numeric_features) + list(categorical_features)
        return all_features

    def requires(self) -> DataRequirements:
        # Static metadata doesn't require market data
        return DataRequirements.L1_ONLY


class _InstrumentMetadataTransform:
    """
    Instrument metadata features from the instrument_metadata store.

    Provides temporal instrument metadata for factor-based portfolio construction:
    - Duration buckets: 0=Short (0-2Y), 1=Medium (2-7Y), 2=Long (7Y+)
    - Issuer types: 0=SOVEREIGN, 1=QUASI_SOVEREIGN, 2=CORPORATE_IG, 3=CORPORATE_HY
    - Liquidity tiers: 1=High, 2=Medium, 3=Low

    These features enable dynamic factor assignment and cross-sectional analysis.
    """

    name = "instrument_metadata"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        """
        Generate instrument metadata feature names.

        Parameters
        ----------
        params : dict
            include_region: Include region metadata (default: False)
            include_sector: Include sector metadata (default: False)
            include_rating: Include credit rating metadata (default: False)

        Returns
        -------
        list[str]
            Feature names in order

        """
        features = [
            "duration_bucket",
            "issuer_type",
            "liquidity_tier",
        ]

        if params.get("include_region", False):
            features.append("region_encoded")

        if params.get("include_sector", False):
            features.append("sector_encoded")

        if params.get("include_rating", False):
            features.append("rating_encoded")

        return features

    def requires(self) -> DataRequirements:
        """Instrument metadata is L1_ONLY (no market data needed)."""
        return DataRequirements.L1_ONLY


# Register optional transforms
register_transform(_KeltnerTransform())
register_transform(_OBVTransform())
register_transform(_MicrostructureTransform())
register_transform(_TradeFlowTransform())
register_transform(_InstrumentMetadataTransform())

# Register known-future transforms for TFT
register_transform(_CalendarTransform())
register_transform(_EventScheduleTransform())
register_transform(_MacroIndicatorsTransform())
register_transform(_StaticCovariatesTransform())

# Register ALFRED/FRED macro features with revision awareness
register_transform(_MacroTransform())
register_transform(_MacroDeltasTransform())


class _MacroCompositesTransform:
    """
    Factorized macro composites - derived signals across economic dimensions.

    Computes composite features from base macro series:
    - Credit spreads (IG/HY spread, quality premium)
    - Duration structure (term spread, curve slope, real yields)
    - Liquidity conditions (Fed balance sheet, bank credit, funding stress)
    - Growth/inflation regimes (momentum, stagflation risk, goldilocks)
    - FX positioning (dollar strength, carry, stress)

    These reduce dimensionality and increase signal density vs. raw series.
    """

    name = "macro_composites"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        """
        Get composite feature names.

        Returns all possible composites; actual availability depends on
        which base series are present in the data.
        """
        from ml.features.macro_composites import get_composite_feature_names

        return get_composite_feature_names()

    def requires(self) -> DataRequirements:
        """Composites only need L1 (timestamps) + base macro series."""
        return DataRequirements.L1_ONLY


# Register macro composites
register_transform(_MacroCompositesTransform())


class _EWMABetaTransform:
    """
    EWMA Beta features for cross-asset relationships.

    Computes exponentially weighted moving average beta between an asset and a
    benchmark/market. Measures systematic risk exposure with time-varying weights.

    Features:
    - ewma_beta: Rolling beta coefficient (cov / var_market)
    - ewma_cov: Exponentially weighted covariance
    - ewma_var_market: Exponentially weighted market variance
    """

    name = "ewma_beta"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        """
        Generate EWMA beta feature names.

        Parameters
        ----------
        params : dict
            benchmark_id: Benchmark instrument ID (e.g., "SPY")
            alpha: EWMA decay factor (default: 0.94)
            include_components: Include cov/var components (default: False)

        Returns
        -------
        list[str]
            Feature names in order
        """
        benchmark_id = params.get("benchmark_id", "market")
        include_components = params.get("include_components", False)

        features = [f"ewma_beta_{benchmark_id}"]

        if include_components:
            features.extend([
                f"ewma_cov_{benchmark_id}",
                f"ewma_var_{benchmark_id}",
            ])

        return features

    def requires(self) -> DataRequirements:
        """Beta computation requires L1 data from multiple instruments."""
        return DataRequirements.L1_ONLY


class _ZScoreSpreadTransform:
    """
    Z-Scored Spread features for pairs trading.

    Computes standardized price spreads between asset pairs using rolling statistics.
    Enables mean-reversion strategies and statistical arbitrage.

    Features:
    - zscore_spread: Standardized spread (spread - mean) / std
    - spread_raw: Raw price difference
    - spread_mean: Rolling mean of spread
    - spread_std: Rolling standard deviation
    """

    name = "zscore_spread"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        """
        Generate z-scored spread feature names.

        Parameters
        ----------
        params : dict
            pair_id: Identifier for the asset pair (e.g., "GLD_GDX")
            include_components: Include mean/std components (default: False)

        Returns
        -------
        list[str]
            Feature names in order
        """
        pair_id = params.get("pair_id", "pair")
        include_components = params.get("include_components", False)

        features = [f"zscore_spread_{pair_id}"]

        if include_components:
            features.extend([
                f"spread_raw_{pair_id}",
                f"spread_mean_{pair_id}",
                f"spread_std_{pair_id}",
            ])

        return features

    def requires(self) -> DataRequirements:
        """Spread computation requires L1 data from multiple instruments."""
        return DataRequirements.L1_ONLY


# Register cross-asset transforms
register_transform(_EWMABetaTransform())
register_transform(_ZScoreSpreadTransform())


@dataclass(frozen=True)
class EWMABetaTransformSpec:
    """Transform spec for EWMA beta computation between instrument and factor."""

    name: str = "ewma_beta"
    instrument_id: str = ""
    factor_name: str = ""
    alpha: float = 0.94

    def compute_feature_names(self) -> list[str]:
        """Return feature names produced by this transform."""
        return [f"ewma_beta_{self.instrument_id}_{self.factor_name}"]


@dataclass(frozen=True)
class ZScoreSpreadTransformSpec:
    """Transform spec for z-scored spread computation between two instruments."""

    name: str = "zscore_spread"
    instrument_id_1: str = ""
    instrument_id_2: str = ""
    window_size: int = 60

    def compute_feature_names(self) -> list[str]:
        """Return feature names produced by this transform."""
        return [f"zscore_spread_{self.instrument_id_1}_{self.instrument_id_2}"]


@dataclass(frozen=True)
class RollingCorrelationTransformSpec:
    """Transform spec for rolling correlation between two instruments."""

    name: str = "rolling_correlation"
    instrument_id_1: str = ""
    instrument_id_2: str = ""
    window_size: int = 60

    def compute_feature_names(self) -> list[str]:
        """Return feature names produced by this transform."""
        return [f"correlation_{self.instrument_id_1}_{self.instrument_id_2}"]


@dataclass(slots=True)
class TransformSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineSpec:
    transforms: list[TransformSpec]


def _hash_pipeline(transforms: list[TransformSpec]) -> str:
    h = hashlib.sha256()
    for spec in transforms:
        h.update(spec.name.encode("utf-8"))
        h.update(b"\0")
        # Stable param hashing (sorted by key)
        for k in sorted(spec.params.keys()):
            v = spec.params[k]
            h.update(str(k).encode("utf-8"))
            h.update(b"=")
            h.update(str(v).encode("utf-8"))
            h.update(b";")
        h.update(b"\n")
    return h.hexdigest()


def transform_feature_names(spec: TransformSpec) -> list[str]:
    """
    Resolve feature names for a single transform spec.

    Args:
        spec: Transform specification.

    Returns:
        Ordered feature names for the transform.

    Raises:
        ValueError: If the transform name is unknown.
    """
    if spec.name not in _CATALOG:
        msg = f"Unknown transform: {spec.name}"
        raise ValueError(msg)
    return _CATALOG[spec.name].feature_names(spec.params)


class PipelineRunner:
    """
    Compile-time utilities for feature pipelines.
    """

    _REQ_ORDER: dict[DataRequirements, int] = {
        DataRequirements.L1_ONLY: 0,
        DataRequirements.L1_L2: 1,
        DataRequirements.L1_L2_L3: 2,
        DataRequirements.HISTORICAL: 0,
        DataRequirements.STREAMING: 0,
    }

    def __init__(self, spec: PipelineSpec, allowable: DataRequirements) -> None:
        self._spec = spec
        self._allowable = allowable
        self._transforms: list[FeatureTransform] = []
        self._compile()

    def _compile(self) -> None:
        self._transforms.clear()
        for ts in self._spec.transforms:
            if ts.name not in _CATALOG:
                msg = f"Unknown transform: {ts.name}"
                raise ValueError(msg)
            t = _CATALOG[ts.name]
            # Gate by data requirements (simple lattice: L1_ONLY < L1_L2 < L1_L2_L3)
            req = t.requires()
            if self._requirement_level(req) > self._requirement_level(self._allowable):
                msg = f"Transform {ts.name} requires {req.value}, not allowed for {self._allowable.value}"
                raise ValueError(msg)
            self._transforms.append(t)

    def compute_feature_names(self) -> list[str]:
        names: list[str] = []
        for ts, t in zip(self._spec.transforms, self._transforms):
            names.extend(t.feature_names(ts.params))
        return names

    def compute_signature(self) -> str:
        return _hash_pipeline(self._spec.transforms)

    @classmethod
    def _requirement_level(cls, requirement: DataRequirements) -> int:
        return cls._REQ_ORDER.get(requirement, 0)
