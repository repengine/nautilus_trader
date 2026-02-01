"""
Target semantics configuration and helpers.

This module centralizes target/label semantics for dataset generation and training.
It defines horizon specifications, cost model configuration, and label variants
along with naming helpers and metadata serialization helpers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Literal


def bps_to_decimal(bps: float) -> float:
    """
    Convert basis points (bps) to decimal return units.

    Args:
        bps: Basis points value (1 bps = 0.0001).

    Returns:
        Decimal representation of the bps value.
    """
    return float(bps) / 10_000.0


def decimal_to_bps(value: float) -> float:
    """
    Convert decimal return units to basis points (bps).

    Args:
        value: Decimal return value (e.g., 0.001).

    Returns:
        Basis points representation (e.g., 10.0).
    """
    return float(value) * 10_000.0


def build_forward_return_column(horizon_label: str) -> str:
    """
    Build the forward return column name for a horizon label.

    Args:
        horizon_label: Horizon label such as "15m".

    Returns:
        Column name (e.g., "forward_return_15m").
    """
    return f"forward_return_{horizon_label}"


def build_cost_return_column(horizon_label: str) -> str:
    """
    Build the cost-aware return column name for a horizon label.

    Args:
        horizon_label: Horizon label such as "15m".

    Returns:
        Column name (e.g., "cost_return_15m").
    """
    return f"cost_return_{horizon_label}"


def build_binary_target_column(horizon_label: str) -> str:
    """
    Build the binary target column name for a horizon label.

    Args:
        horizon_label: Horizon label such as "15m".

    Returns:
        Column name (e.g., "target_bin_15m").
    """
    return f"target_bin_{horizon_label}"


def build_multiclass_target_column(horizon_label: str) -> str:
    """
    Build the multiclass target column name for a horizon label.

    Args:
        horizon_label: Horizon label such as "15m".

    Returns:
        Column name (e.g., "target_class_15m").
    """
    return f"target_class_{horizon_label}"


def build_regression_target_column(horizon_label: str) -> str:
    """
    Build the regression target column name for a horizon label.

    Args:
        horizon_label: Horizon label such as "15m".

    Returns:
        Column name (e.g., "target_reg_15m").
    """
    return f"target_reg_{horizon_label}"


@dataclass(frozen=True)
class TargetHorizonSpec:
    """
    Horizon specification for target generation.

    Args:
        minutes: Horizon length in minutes.
        label: Optional label suffix (must end with "m" when provided).
    """

    minutes: int
    label: str | None = None

    def __post_init__(self) -> None:
        """
        Validate and normalize the horizon specification.
        """
        if self.minutes <= 0:
            raise ValueError(f"horizon minutes must be positive, got {self.minutes}")

        resolved = self.label or f"{self.minutes}m"
        if not resolved:
            raise ValueError("horizon label must be a non-empty string")
        if not resolved.endswith("m"):
            raise ValueError(f"horizon label must end with 'm', got {resolved!r}")
        object.__setattr__(self, "label", resolved)


@dataclass(frozen=True)
class TargetCostModelConfig:
    """
    Cost model configuration for cost-aware returns.

    Args:
        cost_bps: Base cost in basis points (per side).
        commission_bps: Commission in basis points (per side).
        slippage_bps: Slippage in basis points (per side).
    """

    cost_bps: float = 0.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        """
        Validate cost model configuration.
        """
        if self.cost_bps < 0.0:
            raise ValueError("cost_bps must be >= 0.0")
        if self.commission_bps < 0.0:
            raise ValueError("commission_bps must be >= 0.0")
        if self.slippage_bps < 0.0:
            raise ValueError("slippage_bps must be >= 0.0")

    @property
    def per_side_bps(self) -> float:
        """
        Sum of per-side cost components in basis points.
        """
        return float(self.cost_bps + self.commission_bps + self.slippage_bps)

    @property
    def round_trip_bps(self) -> float:
        """
        Round-trip cost in basis points (entry + exit).
        """
        return float(2.0 * self.per_side_bps)

    @property
    def round_trip_decimal(self) -> float:
        """
        Round-trip cost in decimal return units.
        """
        return bps_to_decimal(self.round_trip_bps)

    def as_metadata(self) -> dict[str, float]:
        """
        Serialize the cost model to metadata-friendly values.

        Returns:
            Dictionary containing cost components and round-trip totals.
        """
        return {
            "cost_bps": float(self.cost_bps),
            "commission_bps": float(self.commission_bps),
            "slippage_bps": float(self.slippage_bps),
            "round_trip_bps": self.round_trip_bps,
        }


@dataclass(frozen=True)
class BinaryTargetConfig:
    """
    Binary (long-only) target configuration.

    Args:
        enabled: Whether to generate binary targets.
        threshold_bps: Threshold in basis points.
        return_basis: Return basis ("raw" or "cost") for the label.
    """

    enabled: bool = True
    threshold_bps: float = 10.0
    return_basis: Literal["raw", "cost"] = "raw"

    def __post_init__(self) -> None:
        """
        Validate binary target configuration.
        """
        if self.threshold_bps < 0.0:
            raise ValueError("threshold_bps must be >= 0.0")
        if self.return_basis not in ("raw", "cost"):
            raise ValueError("return_basis must be 'raw' or 'cost'")

    @property
    def threshold(self) -> float:
        """
        Threshold in decimal return units.
        """
        return bps_to_decimal(self.threshold_bps)


@dataclass(frozen=True)
class MulticlassTargetConfig:
    """
    Multiclass (short/neutral/long) target configuration.

    Args:
        enabled: Whether to generate multiclass targets.
        short_threshold_bps: Absolute short threshold in basis points.
        long_threshold_bps: Absolute long threshold in basis points.
        return_basis: Return basis ("raw" or "cost") for the label.
    """

    enabled: bool = False
    short_threshold_bps: float = 10.0
    long_threshold_bps: float = 10.0
    return_basis: Literal["raw", "cost"] = "raw"

    def __post_init__(self) -> None:
        """
        Validate multiclass target configuration.
        """
        if self.short_threshold_bps < 0.0:
            raise ValueError("short_threshold_bps must be >= 0.0")
        if self.long_threshold_bps < 0.0:
            raise ValueError("long_threshold_bps must be >= 0.0")
        if self.return_basis not in ("raw", "cost"):
            raise ValueError("return_basis must be 'raw' or 'cost'")

    @property
    def short_threshold(self) -> float:
        """
        Negative short threshold in decimal return units.
        """
        return -bps_to_decimal(self.short_threshold_bps)

    @property
    def long_threshold(self) -> float:
        """
        Long threshold in decimal return units.
        """
        return bps_to_decimal(self.long_threshold_bps)


@dataclass(frozen=True)
class RegressionTargetConfig:
    """
    Regression target configuration.

    Args:
        enabled: Whether to generate regression targets.
        return_basis: Return basis ("raw" or "cost") for the label.
    """

    enabled: bool = False
    return_basis: Literal["raw", "cost"] = "raw"

    def __post_init__(self) -> None:
        """
        Validate regression target configuration.
        """
        if self.return_basis not in ("raw", "cost"):
            raise ValueError("return_basis must be 'raw' or 'cost'")


@dataclass(frozen=True)
class TargetSemanticsConfig:
    """
    Full target semantics configuration.

    Args:
        version: Target semantics version identifier.
        horizons: Tuple of horizon specifications.
        cost_model: Optional cost model for cost-aware returns.
        emit_cost_return: Emit cost-aware returns even when cost_model is None.
        binary: Binary target configuration.
        multiclass: Multiclass target configuration.
        regression: Regression target configuration.
        primary_target: Optional primary target column name.
        legacy_aliases: Whether to emit legacy aliases (y/forward_return).
    """

    version: str = "v1"
    horizons: tuple[TargetHorizonSpec, ...] = field(
        default_factory=lambda: (TargetHorizonSpec(minutes=15),),
    )
    cost_model: TargetCostModelConfig | None = None
    emit_cost_return: bool = False
    binary: BinaryTargetConfig = field(default_factory=BinaryTargetConfig)
    multiclass: MulticlassTargetConfig = field(default_factory=MulticlassTargetConfig)
    regression: RegressionTargetConfig = field(default_factory=RegressionTargetConfig)
    primary_target: str | None = None
    legacy_aliases: bool = False

    def __post_init__(self) -> None:
        """
        Validate target semantics configuration.
        """
        if not self.version:
            raise ValueError("target semantics version must be non-empty")
        if not self.horizons:
            raise ValueError("At least one horizon must be specified")

        labels = [spec.label for spec in self.horizons if spec.label is not None]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Horizon labels must be unique, got {labels}")

        if self.primary_target is not None and not self.primary_target:
            raise ValueError("primary_target must be a non-empty string when provided")

        uses_cost = any(
            cfg.return_basis == "cost"
            for cfg in (self.binary, self.multiclass, self.regression)
            if cfg.enabled
        )
        if uses_cost and not (self.cost_model or self.emit_cost_return):
            raise ValueError("cost-aware targets require cost_model or emit_cost_return=True")

        if self.primary_target is not None and self.primary_target not in self.label_columns():
            raise ValueError(
                f"primary_target {self.primary_target!r} not present in label columns",
            )

    @property
    def horizon_labels(self) -> tuple[str, ...]:
        """
        Return the normalized horizon labels.
        """
        return tuple(spec.label for spec in self.horizons if spec.label is not None)

    @property
    def max_horizon_minutes(self) -> int:
        """
        Return the maximum horizon length in minutes.
        """
        return max(spec.minutes for spec in self.horizons)

    def should_emit_cost_return(self) -> bool:
        """
        Determine whether cost-aware return columns should be emitted.
        """
        return self.cost_model is not None or self.emit_cost_return

    def return_column_for_basis(self, horizon_label: str, basis: Literal["raw", "cost"]) -> str:
        """
        Resolve the return column name for a basis.

        Args:
            horizon_label: Horizon label such as "15m".
            basis: "raw" or "cost".

        Returns:
            Column name for the requested basis.
        """
        if basis == "raw":
            return build_forward_return_column(horizon_label)
        return build_cost_return_column(horizon_label)

    def return_columns(self) -> tuple[str, ...]:
        """
        Enumerate return columns to be generated.
        """
        columns = [build_forward_return_column(label) for label in self.horizon_labels]
        if self.should_emit_cost_return():
            columns.extend(build_cost_return_column(label) for label in self.horizon_labels)
        return tuple(columns)

    def label_columns(self) -> tuple[str, ...]:
        """
        Enumerate target label columns to be generated.
        """
        columns: list[str] = []
        for label in self.horizon_labels:
            if self.binary.enabled:
                columns.append(build_binary_target_column(label))
            if self.multiclass.enabled:
                columns.append(build_multiclass_target_column(label))
            if self.regression.enabled:
                columns.append(build_regression_target_column(label))
        return tuple(columns)

    def resolved_primary_target(self) -> str | None:
        """
        Resolve the primary target column.

        Returns:
            Primary target column if configured or if only one label exists.
        """
        if self.primary_target:
            return self.primary_target
        labels = self.label_columns()
        if len(labels) == 1:
            return labels[0]
        return None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetSemanticsConfig:
        """
        Build target semantics config from a dictionary payload.

        Args:
            payload: Mapping containing target semantics fields.

        Returns:
            Parsed TargetSemanticsConfig instance.
        """
        version = str(payload.get("version", "v1"))

        horizons_payload = payload.get("horizons")
        horizons: tuple[TargetHorizonSpec, ...]
        if horizons_payload is None:
            horizons = (TargetHorizonSpec(minutes=15),)
        else:
            parsed: list[TargetHorizonSpec] = []
            for item in horizons_payload:
                if isinstance(item, Mapping):
                    minutes = int(item.get("minutes", 0))
                    label = item.get("label")
                    parsed.append(TargetHorizonSpec(minutes=minutes, label=label))
                else:
                    parsed.append(TargetHorizonSpec(minutes=int(item)))
            horizons = tuple(parsed)

        cost_model_payload = payload.get("cost_model")
        cost_model = None
        if isinstance(cost_model_payload, Mapping):
            cost_model = TargetCostModelConfig(
                cost_bps=float(cost_model_payload.get("cost_bps", 0.0)),
                commission_bps=float(cost_model_payload.get("commission_bps", 0.0)),
                slippage_bps=float(cost_model_payload.get("slippage_bps", 0.0)),
            )

        def _parse_return_basis(value: object) -> Literal["raw", "cost"]:
            token = str(value).strip().lower()
            if token == "cost":
                return "cost"
            return "raw"

        def _parse_binary(value: object) -> BinaryTargetConfig:
            if isinstance(value, Mapping):
                return BinaryTargetConfig(
                    enabled=bool(value.get("enabled", True)),
                    threshold_bps=float(value.get("threshold_bps", 10.0)),
                    return_basis=_parse_return_basis(value.get("return_basis", "raw")),
                )
            if isinstance(value, bool):
                return BinaryTargetConfig(enabled=value)
            return BinaryTargetConfig()

        def _parse_multiclass(value: object) -> MulticlassTargetConfig:
            if isinstance(value, Mapping):
                return MulticlassTargetConfig(
                    enabled=bool(value.get("enabled", False)),
                    short_threshold_bps=float(value.get("short_threshold_bps", 10.0)),
                    long_threshold_bps=float(value.get("long_threshold_bps", 10.0)),
                    return_basis=_parse_return_basis(value.get("return_basis", "raw")),
                )
            if isinstance(value, bool):
                return MulticlassTargetConfig(enabled=value)
            return MulticlassTargetConfig()

        def _parse_regression(value: object) -> RegressionTargetConfig:
            if isinstance(value, Mapping):
                return RegressionTargetConfig(
                    enabled=bool(value.get("enabled", False)),
                    return_basis=_parse_return_basis(value.get("return_basis", "raw")),
                )
            if isinstance(value, bool):
                return RegressionTargetConfig(enabled=value)
            return RegressionTargetConfig()

        binary_cfg = _parse_binary(payload.get("binary"))
        multiclass_cfg = _parse_multiclass(payload.get("multiclass"))
        regression_cfg = _parse_regression(payload.get("regression"))

        emit_cost_return = bool(payload.get("emit_cost_return", False))
        primary_target = payload.get("primary_target")
        if primary_target is not None:
            primary_target = str(primary_target)
        legacy_aliases = bool(payload.get("legacy_aliases", False))

        return cls(
            version=version,
            horizons=horizons,
            cost_model=cost_model,
            emit_cost_return=emit_cost_return,
            binary=binary_cfg,
            multiclass=multiclass_cfg,
            regression=regression_cfg,
            primary_target=primary_target,
            legacy_aliases=legacy_aliases,
        )

    @classmethod
    def from_json(cls, payload: str) -> TargetSemanticsConfig:
        """
        Build target semantics config from a JSON string.

        Args:
            payload: JSON-encoded target semantics.

        Returns:
            Parsed TargetSemanticsConfig instance.
        """
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_legacy(
        cls,
        *,
        horizon_minutes: int,
        threshold: float,
        legacy_aliases: bool = True,
        primary_target: str | None = None,
    ) -> TargetSemanticsConfig:
        """
        Build a legacy-compatible target semantics config.

        Args:
            horizon_minutes: Horizon in minutes.
            threshold: Threshold in decimal return units.
            legacy_aliases: Emit legacy aliases ("y", "forward_return").
            primary_target: Optional primary target column name.

        Returns:
            TargetSemanticsConfig that matches legacy target generation.
        """
        binary_cfg = BinaryTargetConfig(
            enabled=True,
            threshold_bps=decimal_to_bps(threshold),
            return_basis="raw",
        )
        return cls(
            horizons=(TargetHorizonSpec(minutes=horizon_minutes),),
            binary=binary_cfg,
            multiclass=MulticlassTargetConfig(enabled=False),
            regression=RegressionTargetConfig(enabled=False),
            primary_target=primary_target,
            legacy_aliases=legacy_aliases,
        )


__all__ = [
    "BinaryTargetConfig",
    "MulticlassTargetConfig",
    "RegressionTargetConfig",
    "TargetCostModelConfig",
    "TargetHorizonSpec",
    "TargetSemanticsConfig",
    "bps_to_decimal",
    "build_binary_target_column",
    "build_cost_return_column",
    "build_forward_return_column",
    "build_multiclass_target_column",
    "build_regression_target_column",
    "decimal_to_bps",
]
