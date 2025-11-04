"""Validation policy for streaming training wave guardrails."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class StreamingEconomicThresholds:
    """Economic metric thresholds enforced during wave validation."""

    min_slippage_adjusted_sharpe: float = 0.25
    min_turnover: float = 0.05
    max_turnover: float = 0.5
    max_drawdown: float = 1.5
    max_ks_statistic: float = 0.2

    def __post_init__(self) -> None:
        if self.min_slippage_adjusted_sharpe < 0.0:
            raise ValueError("min_slippage_adjusted_sharpe must be non-negative")
        if self.min_turnover < 0.0:
            raise ValueError("min_turnover must be non-negative")
        if self.max_turnover <= 0.0:
            raise ValueError("max_turnover must be positive")
        if self.min_turnover > self.max_turnover:
            raise ValueError("min_turnover must be <= max_turnover")
        if self.max_drawdown <= 0.0:
            raise ValueError("max_drawdown must be positive")
        if self.max_ks_statistic < 0.0:
            raise ValueError("max_ks_statistic must be non-negative")


@dataclass(frozen=True, slots=True)
class StreamingInstrumentCoveragePolicy:
    """Instrument coverage tolerances enforced during wave validation."""

    min_selected_ratio: float = 0.7
    max_selected_ratio_range: float = 0.25
    max_skipped_ratio: float = 0.35

    def __post_init__(self) -> None:
        _validate_ratio("min_selected_ratio", self.min_selected_ratio)
        _validate_ratio("max_selected_ratio_range", self.max_selected_ratio_range)
        _validate_ratio("max_skipped_ratio", self.max_skipped_ratio)


@dataclass(frozen=True, slots=True)
class StreamingWaveValidationPolicy:
    """Aggregate policy controlling validate-wave guardrails."""

    economic: StreamingEconomicThresholds = field(default_factory=StreamingEconomicThresholds)
    coverage: StreamingInstrumentCoveragePolicy = field(default_factory=StreamingInstrumentCoveragePolicy)
    allow_validation_failures: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> StreamingWaveValidationPolicy:
        """Build a validation policy from environment overrides."""
        source = env or os.environ

        def _maybe_float(key: str, default: float) -> float:
            raw = source.get(key)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{key} must be a float") from exc

        def _maybe_bool(key: str, default: bool) -> bool:
            raw = source.get(key)
            if raw is None:
                return default
            lowered = raw.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{key} must be a boolean (0/1, true/false, yes/no)")

        default_economic = StreamingEconomicThresholds()
        default_coverage = StreamingInstrumentCoveragePolicy()
        economic = StreamingEconomicThresholds(
            min_slippage_adjusted_sharpe=_maybe_float(
                "ML_VALIDATE_WAVE_MIN_SHARPE",
                default_economic.min_slippage_adjusted_sharpe,
            ),
            min_turnover=_maybe_float(
                "ML_VALIDATE_WAVE_MIN_TURNOVER",
                default_economic.min_turnover,
            ),
            max_turnover=_maybe_float(
                "ML_VALIDATE_WAVE_MAX_TURNOVER",
                default_economic.max_turnover,
            ),
            max_drawdown=_maybe_float(
                "ML_VALIDATE_WAVE_MAX_DRAWDOWN",
                default_economic.max_drawdown,
            ),
            max_ks_statistic=_maybe_float(
                "ML_VALIDATE_WAVE_MAX_KS",
                default_economic.max_ks_statistic,
            ),
        )
        coverage = StreamingInstrumentCoveragePolicy(
            min_selected_ratio=_maybe_float(
                "ML_VALIDATE_WAVE_MIN_SELECTED_RATIO",
                default_coverage.min_selected_ratio,
            ),
            max_selected_ratio_range=_maybe_float(
                "ML_VALIDATE_WAVE_MAX_RATIO_RANGE",
                default_coverage.max_selected_ratio_range,
            ),
            max_skipped_ratio=_maybe_float(
                "ML_VALIDATE_WAVE_MAX_SKIPPED_RATIO",
                default_coverage.max_skipped_ratio,
            ),
        )

        return cls(
            economic=economic,
            coverage=coverage,
            allow_validation_failures=_maybe_bool(
                "ML_VALIDATE_WAVE_ALLOW_VALIDATION_FAILURES",
                False,
            ),
        )


__all__ = [
    "StreamingEconomicThresholds",
    "StreamingInstrumentCoveragePolicy",
    "StreamingWaveValidationPolicy",
]
