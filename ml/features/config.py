"""
Feature configuration module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import msgspec

from ml.config.base import MLFeatureConfig
from ml.config.constants import IndicatorNames
from ml.config.constants import TechnicalIndicatorPeriods
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec
from ml.registry.base import DataRequirements


class FeatureConfig(MLFeatureConfig, kw_only=True, frozen=True):
    """
    Configuration for feature engineering with enhanced ML integration.

    This configuration extends the base MLFeatureConfig with specific
    parameters for technical indicators and feature computation.

    Parameters
    ----------
    return_periods : list[int], default [1, 5, 10, 20]
        Periods for return calculation features.
    momentum_periods : list[int], default [5, 10, 20]
        Periods for momentum calculation features.
    rsi_period : int, default 14
        Period for RSI calculation (must be between 2 and 100).
    bb_period : int, default 20
        Period for Bollinger Bands calculation (must be between 2 and 100).
    bb_std : float, default 2.0
        Standard deviation multiplier for Bollinger Bands (must be between 0.5 and 5.0).
    atr_period : int, default 20
        Period for ATR calculation (must be between 2 and 100).
    ema_fast : int, default 12
        Fast EMA period (must be between 2 and 50).
    ema_slow : int, default 26
        Slow EMA period (must be between 10 and 200, greater than ema_fast).
    macd_signal : int, default 9
        MACD signal line period (must be between 2 and 50).
    volume_ma_periods : list[int], default [5, 10, 20]
        Periods for volume moving average features.
    include_microstructure : bool, default False
        Whether to include microstructure features.
    include_trade_flow : bool, default False
        Whether to include trade flow features.
    include_macro : bool, default False
        Whether to include ALFRED/FRED macro features.
    macro_series_ids : list[str], optional
        FRED series identifiers to include when ``include_macro`` is enabled.
    include_macro_revisions : bool, default False
        Whether to add revision-aware macro features.
    macro_revision_mode : {"minimal", "core", "full"}, default "core"
        Revision feature mode controlling which derived columns are emitted.
    include_macro_composites : bool, default False
        Whether to append factorized macro composite signals (requires ``include_macro``).
    include_macro_deltas : bool, default False
        Whether to add first differences for macro series (requires ``include_macro``).
    include_calendar : bool, default False
        Whether to include calendar-based known-future covariates.
    include_event_schedule : bool, default False
        Whether to include scheduled event covariates (Fed meetings, earnings, etc.).

    """

    # Price-based features
    return_periods: list[int] = msgspec.field(default_factory=lambda: [1, 5, 10, 20])
    momentum_periods: list[int] = msgspec.field(default_factory=lambda: [5, 10, 20])

    # Technical indicators
    rsi_period: int = TechnicalIndicatorPeriods.RSI_DEFAULT_PERIOD
    bb_period: int = TechnicalIndicatorPeriods.BB_DEFAULT_PERIOD
    bb_std: float = TechnicalIndicatorPeriods.BB_DEFAULT_STD
    atr_period: int = 20

    # Moving averages
    ema_fast: int = TechnicalIndicatorPeriods.EMA_FAST_DEFAULT
    ema_slow: int = TechnicalIndicatorPeriods.EMA_SLOW_DEFAULT
    macd_signal: int = TechnicalIndicatorPeriods.MACD_SIGNAL_PERIOD

    # Volume features
    volume_ma_periods: list[int] = msgspec.field(default_factory=lambda: [5, 10, 20])

    # Optional advanced features (default False for backward compatibility)
    include_microstructure: bool = False
    include_trade_flow: bool = False
    validate_quality: bool = False

    # Macro features (ALFRED/FRED)
    include_macro: bool = False
    macro_series_ids: list[str] = msgspec.field(default_factory=list)
    include_macro_revisions: bool = False
    macro_revision_mode: str = "core"  # "minimal", "core", "full"
    include_macro_composites: bool = False
    include_macro_deltas: bool = False
    macro_min_coverage: float | None = None

    # Calendar features (known-future for TFT)
    include_calendar: bool = False
    include_event_schedule: bool = False
    calendar_encoding: str = "cyclic"  # "cyclic", "onehot", "fourier"

    # --- Compatibility toggles for legacy tests (no-ops by default) ---
    # These mirror older boolean switches used in tests such as
    # enable_returns/enable_momentum/enable_volatility/enable_technical and
    # ma_periods. They are optional and default to None to avoid affecting
    # normal configurations. The enable_* aliases allow older fixtures to pass
    # enable_rsi / enable_bollinger / enable_vwap without raising errors.
    enable_returns: bool | None = None
    enable_momentum: bool | None = None
    enable_volatility: bool | None = None
    enable_technical: bool | None = None
    ma_periods: list[int] | None = None
    enable_rsi: bool | None = None
    enable_bollinger: bool | None = None
    enable_vwap: bool | None = None

    data_requirements: DataRequirements = DataRequirements.L1_ONLY

    def __post_init__(self) -> None:
        """
        Post-initialization validation and setup.
        """
        # Validate EMA parameters
        if self.ema_slow <= self.ema_fast:
            msg = f"ema_slow ({self.ema_slow}) must be greater than ema_fast ({self.ema_fast})"
            raise ValueError(msg)

        # Validate range constraints
        if not (2 <= self.rsi_period <= 100):
            msg = f"rsi_period must be between 2 and 100, got {self.rsi_period}"
            raise ValueError(msg)

        if not (2 <= self.bb_period <= 100):
            msg = f"bb_period must be between 2 and 100, got {self.bb_period}"
            raise ValueError(msg)

        if not (0.5 <= self.bb_std <= 5.0):
            msg = f"bb_std must be between 0.5 and 5.0, got {self.bb_std}"
            raise ValueError(msg)

        if not (2 <= self.atr_period <= 100):
            msg = f"atr_period must be between 2 and 100, got {self.atr_period}"
            raise ValueError(msg)

        if not (2 <= self.ema_fast <= 50):
            msg = f"ema_fast must be between 2 and 50, got {self.ema_fast}"
            raise ValueError(msg)

        if not (10 <= self.ema_slow <= 200):
            msg = f"ema_slow must be between 10 and 200, got {self.ema_slow}"
            raise ValueError(msg)

        if not (2 <= self.macd_signal <= 50):
            msg = f"macd_signal must be between 2 and 50, got {self.macd_signal}"
            raise ValueError(msg)

        if self.include_macro_composites and not self.include_macro:
            raise ValueError("include_macro_composites requires include_macro to be True")

        if self.include_macro_composites and not self.macro_series_ids:
            raise ValueError(
                "include_macro_composites requires macro_series_ids to be configured",
            )

        if self.include_macro_deltas and not self.include_macro:
            raise ValueError("include_macro_deltas requires include_macro to be True")

        if self.macro_min_coverage is not None and not 0.0 < float(self.macro_min_coverage) <= 1.0:
            msg = "macro_min_coverage must be within (0, 1], received " f"{self.macro_min_coverage}"
            raise ValueError(msg)

        # Note: Do not mutate fields in frozen msgspec.Struct. Compatibility
        # handling for `ma_periods` occurs in pipeline spec construction.
        if self.include_microstructure or self.include_trade_flow:
            requirements = self.resolved_data_requirements()
            if self.include_trade_flow and requirements != DataRequirements.L1_L2_L3:
                raise ValueError(
                    "Trade flow features require data_requirements >= L1_L2_L3; "
                    f"received {requirements.value}.",
                )
            if self.include_microstructure and requirements not in {
                DataRequirements.L1_L2,
                DataRequirements.L1_L2_L3,
            }:
                raise ValueError(
                    "Microstructure features require data_requirements >= L1_L2; "
                    f"received {requirements.value}.",
                )

    def resolved_data_requirements(self) -> DataRequirements:
        """
        Return effective data requirements after applying feature constraints.
        """
        requirements = getattr(self, "data_requirements", DataRequirements.L1_ONLY)
        if requirements == DataRequirements.L1_ONLY:
            if self.include_trade_flow:
                return DataRequirements.L1_L2_L3
            if self.include_microstructure:
                return DataRequirements.L1_L2
        return requirements

    def get_feature_names(self) -> list[str]:
        """
        Generate complete list of feature names in order.

        Canonicalized to delegate to the declarative pipeline to avoid drift.

        Returns
        -------
        list[str]
            Ordered feature names generated by the configured pipeline.

        """
        # Build a PipelineSpec mirroring the config and compute names via PipelineRunner
        spec = build_pipeline_spec_from_feature_config(self)
        allowable = self.resolved_data_requirements()
        runner = PipelineRunner(spec, allowable=allowable)
        return runner.compute_feature_names()

    def get_indicator_specs(self) -> dict[str, dict[str, Any]]:
        """
        Generate specifications for creating Nautilus indicators.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping indicator names to their configuration parameters.

        """
        specs = {
            # Price SMAs for returns calculation
            IndicatorNames.PRICE_SMA_5: {
                "type": "SMA",
                "period": TechnicalIndicatorPeriods.MA_FAST_PERIOD,
                "input": "close",
            },
            IndicatorNames.PRICE_SMA_20: {
                "type": "SMA",
                "period": TechnicalIndicatorPeriods.MA_SLOW_PERIOD,
                "input": "close",
            },
        }

        # Add volume SMAs based on configured periods
        for period in self.volume_ma_periods:
            specs[f"volume_sma_{period}"] = {
                "type": "SMA",
                "period": period,
                "input": "volume",
            }

        # Technical indicators
        specs.update(
            {
                "rsi": {"type": "RSI", "period": self.rsi_period},
                "bb": {"type": "BB", "period": self.bb_period, "std": self.bb_std},
                "atr": {"type": "ATR", "period": self.atr_period},
                "ema_fast": {"type": "EMA", "period": self.ema_fast},
                "ema_slow": {"type": "EMA", "period": self.ema_slow},
                "macd": {
                    "type": "MACD",
                    "fast": self.ema_fast,
                    "slow": self.ema_slow,
                    "signal": self.macd_signal,
                },
            },
        )

        return specs


FeatureConfigLike: TypeAlias = FeatureConfig


def build_pipeline_spec_from_feature_config(cfg: FeatureConfigLike) -> PipelineSpec:
    """
    Build a PipelineSpec from a FeatureConfig, including optional transforms.

    This is the single source of truth for feature name enumeration.

    """
    transforms: list[TransformSpec] = []

    # Legacy compatibility: boolean toggles default to enabled if None.
    if getattr(cfg, "enable_returns", None) is not False:
        transforms.append(
            TransformSpec(name="returns", params={"periods": list(cfg.return_periods)}),
        )

    if getattr(cfg, "enable_momentum", None) is not False:
        transforms.append(
            TransformSpec(name="momentum", params={"periods": list(cfg.momentum_periods)}),
        )

    if getattr(cfg, "enable_volatility", None) is not False:
        transforms.append(TransformSpec(name="volatility", params={}))

    # Volume ratio belongs to core indicators group conceptually, but keep separate
    # to allow parameterization by periods.
    vr_periods = list(cfg.ma_periods) if cfg.ma_periods is not None else list(cfg.volume_ma_periods)
    transforms.append(TransformSpec(name="volume_ratio", params={"periods": vr_periods}))

    if getattr(cfg, "enable_technical", None) is not False:
        transforms.append(TransformSpec(name="core_indicators", params={}))

    if getattr(cfg, "include_microstructure", False):
        transforms.append(TransformSpec(name="microstructure", params={}))
    if getattr(cfg, "include_trade_flow", False):
        transforms.append(TransformSpec(name="trade_flow", params={}))

    # Macro features (ALFRED/FRED with optional revisions)
    if getattr(cfg, "include_macro", False):
        macro_params = {
            "series_ids": getattr(cfg, "macro_series_ids", []),
            "include_revisions": getattr(cfg, "include_macro_revisions", False),
            "revision_mode": getattr(cfg, "macro_revision_mode", "core"),
            "min_coverage": getattr(cfg, "macro_min_coverage", None),
        }
        transforms.append(TransformSpec(name="macro", params=macro_params))

        if getattr(cfg, "include_macro_composites", False):
            transforms.append(TransformSpec(name="macro_composites", params={}))
        if getattr(cfg, "include_macro_deltas", False):
            transforms.append(
                TransformSpec(
                    name="macro_deltas",
                    params={"series_ids": list(getattr(cfg, "macro_series_ids", []))},
                ),
            )
    elif getattr(cfg, "include_macro_composites", False):
        msg = "include_macro_composites requires include_macro to be True"
        raise ValueError(msg)

    # Calendar features (known-future for TFT)
    if getattr(cfg, "include_calendar", False):
        calendar_params = {
            "encoding": getattr(cfg, "calendar_encoding", "cyclic"),
        }
        transforms.append(TransformSpec(name="calendar", params=calendar_params))

    if getattr(cfg, "include_event_schedule", False):
        transforms.append(TransformSpec(name="event_schedule", params={}))

    return PipelineSpec(transforms=transforms)


def derive_ohlcv_feature_config(
    base: FeatureConfig,
    transforms: Sequence[TransformSpec],
    *,
    allowable: DataRequirements,
) -> FeatureConfig:
    """
    Derive a FeatureConfig for OHLCV-oriented transforms.

    This helper mirrors the canonical pipeline spec and enables only the
    transforms that are supported by the online FeatureCalculator.

    Args:
        base: Baseline FeatureConfig to copy defaults from.
        transforms: Pipeline transforms to project into FeatureConfig flags.
        allowable: DataRequirements gate to satisfy microstructure/trade flow needs.

    Returns:
        FeatureConfig with OHLCV transform flags and periods aligned to the spec.
    """
    names = {ts.name for ts in transforms}
    return_periods: list[int] | None = None
    momentum_periods: list[int] | None = None
    volume_periods: list[int] | None = None

    for ts in transforms:
        if ts.name == "returns":
            return_periods = list(ts.params.get("periods", [1, 5, 10, 20]))
        elif ts.name == "momentum":
            momentum_periods = list(ts.params.get("periods", [5, 10, 20]))
        elif ts.name == "volume_ratio":
            volume_periods = list(ts.params.get("periods", [5, 10, 20]))

    include_microstructure = "microstructure" in names
    include_trade_flow = "trade_flow" in names
    data_requirements = allowable
    if include_trade_flow and data_requirements != DataRequirements.L1_L2_L3:
        data_requirements = DataRequirements.L1_L2_L3
    elif include_microstructure and data_requirements == DataRequirements.L1_ONLY:
        data_requirements = DataRequirements.L1_L2

    return msgspec.structs.replace(
        base,
        enable_returns=True if "returns" in names else False,
        enable_momentum=True if "momentum" in names else False,
        enable_volatility=True if "volatility" in names else False,
        enable_technical=True if "core_indicators" in names else False,
        include_microstructure=include_microstructure,
        include_trade_flow=include_trade_flow,
        include_macro=False,
        include_macro_composites=False,
        include_macro_deltas=False,
        include_calendar=False,
        include_event_schedule=False,
        return_periods=return_periods if return_periods is not None else base.return_periods,
        momentum_periods=(
            momentum_periods if momentum_periods is not None else base.momentum_periods
        ),
        ma_periods=volume_periods if volume_periods is not None else [],
        data_requirements=data_requirements,
    )
