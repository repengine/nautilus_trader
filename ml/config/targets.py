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
from typing import Any, Literal, cast


TARGET_SEMANTICS_EPOCH_VERSION = "epoch-1"
TARGET_SEMANTICS_CONTRACT_ID = "target_semantics_epoch"
TARGET_SEMANTICS_CONTRACT_MAJOR = 1
TARGET_SEMANTICS_REQUIRED_CAPABILITIES: tuple[str, ...] = (
    "horizons_declared",
    "horizon_resolution_declared",
    "execution_contract_declared",
    "execution_latency_declared",
    "unresolved_execution_handling_declared",
    "returns_declared",
    "labels_declared",
    "primary_target_resolved",
)
HORIZON_RESOLUTION_BAR_INDEX = "bar_index"
HORIZON_RESOLUTION_WALL_CLOCK = "wall_clock"
EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN: Literal["zero_return"] = "zero_return"
EXECUTION_UNRESOLVED_CONTEXT_FAIL: Literal["fail"] = "fail"
HorizonResolutionMode = Literal[
    "bar_index",
    "wall_clock",
]
ExecutionUnresolvedContextMode = Literal[
    "zero_return",
    "fail",
]
DEFAULT_WALL_CLOCK_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_EXECUTION_PRICE_COLUMN = "close"
DEFAULT_EXECUTION_LATENCY_BARS = 0


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
        version: Target semantics epoch identifier (single supported epoch).
        contract_id: Canonical contract identifier (single supported value).
        contract_major: Contract major version (single supported value).
        capabilities: Declared contract capabilities required by orchestrators.
        horizons: Tuple of horizon specifications.
        horizon_resolution_mode:
            Horizon resolution mode:
            "bar_index" uses fixed row-offset lookahead;
            "wall_clock" uses timestamp-aligned lookahead.
        wall_clock_timestamp_column:
            Timestamp column for wall-clock alignment.
        execution_entry_price_column:
            Price column used for entry execution context.
        execution_exit_price_column:
            Price column used for exit execution context.
        execution_latency_bars:
            Non-negative bar offset applied before computing horizon returns.
        unresolved_execution_context_mode:
            How unresolved execution context is handled:
            "zero_return" emits deterministic zero returns;
            "fail" raises ValueError.
        cost_model: Optional cost model for cost-aware returns.
        emit_cost_return: Emit cost-aware returns even when cost_model is None.
        binary: Binary target configuration.
        multiclass: Multiclass target configuration.
        regression: Regression target configuration.
        primary_target: Optional primary target column name.
        legacy_aliases: Whether to emit legacy aliases (y/forward_return).
    """

    version: str = TARGET_SEMANTICS_EPOCH_VERSION
    contract_id: str = TARGET_SEMANTICS_CONTRACT_ID
    contract_major: int = TARGET_SEMANTICS_CONTRACT_MAJOR
    capabilities: tuple[str, ...] = field(
        default_factory=lambda: TARGET_SEMANTICS_REQUIRED_CAPABILITIES,
    )
    horizons: tuple[TargetHorizonSpec, ...] = field(
        default_factory=lambda: (TargetHorizonSpec(minutes=15),),
    )
    horizon_resolution_mode: HorizonResolutionMode = "bar_index"
    wall_clock_timestamp_column: str = DEFAULT_WALL_CLOCK_TIMESTAMP_COLUMN
    execution_entry_price_column: str = DEFAULT_EXECUTION_PRICE_COLUMN
    execution_exit_price_column: str = DEFAULT_EXECUTION_PRICE_COLUMN
    execution_latency_bars: int = DEFAULT_EXECUTION_LATENCY_BARS
    unresolved_execution_context_mode: ExecutionUnresolvedContextMode = "zero_return"
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
        if self.version != TARGET_SEMANTICS_EPOCH_VERSION:
            raise ValueError(
                "target semantics epoch is fixed to "
                f"{TARGET_SEMANTICS_EPOCH_VERSION!r}, got {self.version!r}",
            )
        if self.contract_id != TARGET_SEMANTICS_CONTRACT_ID:
            raise ValueError(
                "target semantics contract_id is fixed to "
                f"{TARGET_SEMANTICS_CONTRACT_ID!r}, got {self.contract_id!r}",
            )
        if self.contract_major != TARGET_SEMANTICS_CONTRACT_MAJOR:
            raise ValueError(
                "target semantics contract_major is fixed to "
                f"{TARGET_SEMANTICS_CONTRACT_MAJOR}, got {self.contract_major}",
            )
        if not self.capabilities:
            raise ValueError("target semantics capabilities must be non-empty")
        normalized_capabilities = tuple(str(cap).strip() for cap in self.capabilities)
        if any(not cap for cap in normalized_capabilities):
            raise ValueError("target semantics capabilities must contain non-empty strings")
        if len(set(normalized_capabilities)) != len(normalized_capabilities):
            raise ValueError("target semantics capabilities must be unique")
        missing_required = [
            cap for cap in TARGET_SEMANTICS_REQUIRED_CAPABILITIES if cap not in normalized_capabilities
        ]
        if missing_required:
            raise ValueError(
                "target semantics capabilities missing required values: "
                f"{missing_required}",
            )
        object.__setattr__(self, "capabilities", normalized_capabilities)

        if not self.horizons:
            raise ValueError("At least one horizon must be specified")

        horizon_resolution_mode = str(self.horizon_resolution_mode).strip().lower()
        if horizon_resolution_mode not in (
            HORIZON_RESOLUTION_BAR_INDEX,
            HORIZON_RESOLUTION_WALL_CLOCK,
        ):
            raise ValueError(
                "horizon_resolution_mode must be one of "
                f"{(HORIZON_RESOLUTION_BAR_INDEX, HORIZON_RESOLUTION_WALL_CLOCK)}, "
                f"got {self.horizon_resolution_mode!r}",
            )
        object.__setattr__(
            self,
            "horizon_resolution_mode",
            cast(HorizonResolutionMode, horizon_resolution_mode),
        )

        timestamp_column = str(self.wall_clock_timestamp_column).strip()
        if (
            self.horizon_resolution_mode == HORIZON_RESOLUTION_WALL_CLOCK
            and not timestamp_column
        ):
            raise ValueError(
                "wall_clock_timestamp_column must be a non-empty string "
                "when horizon_resolution_mode='wall_clock'",
            )
        if not timestamp_column:
            timestamp_column = DEFAULT_WALL_CLOCK_TIMESTAMP_COLUMN
        object.__setattr__(self, "wall_clock_timestamp_column", timestamp_column)

        entry_price_column = str(self.execution_entry_price_column).strip()
        if not entry_price_column:
            raise ValueError("execution_entry_price_column must be a non-empty string")
        object.__setattr__(self, "execution_entry_price_column", entry_price_column)

        exit_price_column = str(self.execution_exit_price_column).strip()
        if not exit_price_column:
            raise ValueError("execution_exit_price_column must be a non-empty string")
        object.__setattr__(self, "execution_exit_price_column", exit_price_column)

        latency_bars = int(self.execution_latency_bars)
        if latency_bars < 0:
            raise ValueError("execution_latency_bars must be >= 0")
        object.__setattr__(self, "execution_latency_bars", latency_bars)

        unresolved_execution_mode = str(self.unresolved_execution_context_mode).strip().lower()
        if unresolved_execution_mode not in (
            EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN,
            EXECUTION_UNRESOLVED_CONTEXT_FAIL,
        ):
            raise ValueError(
                "unresolved_execution_context_mode must be one of "
                f"{(EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN, EXECUTION_UNRESOLVED_CONTEXT_FAIL)}, "
                f"got {self.unresolved_execution_context_mode!r}",
            )
        object.__setattr__(
            self,
            "unresolved_execution_context_mode",
            cast(ExecutionUnresolvedContextMode, unresolved_execution_mode),
        )

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

    @property
    def uses_wall_clock_horizons(self) -> bool:
        """
        Return whether horizons resolve via event timestamps.
        """
        return self.horizon_resolution_mode == HORIZON_RESOLUTION_WALL_CLOCK

    def horizon_alignment_metadata(self) -> dict[str, Any]:
        """
        Serialize horizon alignment semantics for metadata contracts.
        """
        if self.uses_wall_clock_horizons:
            return {
                "mode": self.horizon_resolution_mode,
                "timestamp_column": self.wall_clock_timestamp_column,
                "future_anchor": "first_timestamp_at_or_after_horizon",
                "insufficient_future_handling": "zero_return",
            }
        return {
            "mode": self.horizon_resolution_mode,
            "future_anchor": "fixed_row_offset",
            "insufficient_future_handling": "zero_return",
        }

    @property
    def fail_on_unresolved_execution_context(self) -> bool:
        """
        Return whether unresolved execution context should raise immediately.
        """
        return self.unresolved_execution_context_mode == EXECUTION_UNRESOLVED_CONTEXT_FAIL

    def execution_metadata(self) -> dict[str, Any]:
        """
        Serialize execution-aware label semantics for metadata contracts.
        """
        execution: dict[str, Any] = {
            "entry_price_column": self.execution_entry_price_column,
            "exit_price_column": self.execution_exit_price_column,
            "latency_bars": int(self.execution_latency_bars),
            "latency_unit": "bars",
            "unresolved_context_mode": self.unresolved_execution_context_mode,
        }
        if self.unresolved_execution_context_mode == EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN:
            execution["unresolved_context_return"] = 0.0
        return execution

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

    def resolved_primary_horizon_minutes(self, target_column: str | None = None) -> int | None:
        """
        Resolve horizon minutes for the primary target column.

        Args:
            target_column: Optional override for the target column name.

        Returns:
            Horizon minutes if the target column matches a known horizon label.
        """
        column = target_column or self.resolved_primary_target()
        if not column:
            return None
        for spec in self.horizons:
            label = spec.label or f"{spec.minutes}m"
            if column.endswith(label):
                return spec.minutes
        return None

    def contract_metadata(self) -> dict[str, Any]:
        """
        Serialize canonical contract metadata for dataset artifacts.

        Returns:
            Contract metadata payload.
        """
        return {
            "id": self.contract_id,
            "major": int(self.contract_major),
            "capabilities": list(self.capabilities),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetSemanticsConfig:
        """
        Build target semantics config from a dictionary payload.

        Args:
            payload: Mapping containing target semantics fields.

        Returns:
            Parsed TargetSemanticsConfig instance.
        """
        version_raw = payload.get("version", TARGET_SEMANTICS_EPOCH_VERSION)
        version = str(version_raw).strip() or TARGET_SEMANTICS_EPOCH_VERSION

        contract_payload = payload.get("contract")
        if contract_payload is not None and not isinstance(contract_payload, Mapping):
            raise ValueError("target semantics contract payload must be a mapping when provided")
        contract_map = contract_payload if isinstance(contract_payload, Mapping) else {}
        contract_id_raw = payload.get("contract_id", contract_map.get("id", TARGET_SEMANTICS_CONTRACT_ID))
        contract_id = str(contract_id_raw).strip() or TARGET_SEMANTICS_CONTRACT_ID
        contract_major_raw = payload.get(
            "contract_major",
            contract_map.get("major", TARGET_SEMANTICS_CONTRACT_MAJOR),
        )
        contract_major = int(contract_major_raw)
        capabilities_raw = payload.get("capabilities", contract_map.get("capabilities"))
        if capabilities_raw is None:
            capabilities = TARGET_SEMANTICS_REQUIRED_CAPABILITIES
        elif isinstance(capabilities_raw, (list, tuple)):
            capabilities = tuple(str(item).strip() for item in capabilities_raw)
        else:
            raise ValueError(
                "target semantics contract capabilities must be a list/tuple when provided",
            )

        horizon_alignment_payload = payload.get("horizon_alignment")
        horizon_alignment_map = (
            horizon_alignment_payload
            if isinstance(horizon_alignment_payload, Mapping)
            else {}
        )
        horizon_mode_raw = payload.get(
            "horizon_resolution_mode",
            horizon_alignment_map.get("mode", HORIZON_RESOLUTION_BAR_INDEX),
        )
        horizon_resolution_mode = str(horizon_mode_raw).strip().lower() or HORIZON_RESOLUTION_BAR_INDEX
        if "wall_clock_timestamp_column" in payload:
            wall_clock_timestamp_column_raw = payload.get("wall_clock_timestamp_column")
        elif "timestamp_column" in horizon_alignment_map:
            wall_clock_timestamp_column_raw = horizon_alignment_map.get("timestamp_column")
        else:
            wall_clock_timestamp_column_raw = DEFAULT_WALL_CLOCK_TIMESTAMP_COLUMN
        wall_clock_timestamp_column = str(wall_clock_timestamp_column_raw).strip()

        execution_payload = payload.get("execution")
        if execution_payload is not None and not isinstance(execution_payload, Mapping):
            raise ValueError("target semantics execution payload must be a mapping when provided")
        execution_map = execution_payload if isinstance(execution_payload, Mapping) else {}
        entry_price_column_raw = payload.get(
            "execution_entry_price_column",
            execution_map.get("entry_price_column", DEFAULT_EXECUTION_PRICE_COLUMN),
        )
        execution_entry_price_column = str(entry_price_column_raw).strip()
        exit_price_column_raw = payload.get(
            "execution_exit_price_column",
            execution_map.get("exit_price_column", DEFAULT_EXECUTION_PRICE_COLUMN),
        )
        execution_exit_price_column = str(exit_price_column_raw).strip()
        execution_latency_raw = payload.get(
            "execution_latency_bars",
            execution_map.get("latency_bars", DEFAULT_EXECUTION_LATENCY_BARS),
        )
        execution_latency_bars = int(execution_latency_raw)
        unresolved_execution_mode_raw = payload.get(
            "unresolved_execution_context_mode",
            execution_map.get(
                "unresolved_context_mode",
                EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN,
            ),
        )
        unresolved_execution_context_mode = str(unresolved_execution_mode_raw).strip().lower()

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
            contract_id=contract_id,
            contract_major=contract_major,
            capabilities=capabilities,
            horizons=horizons,
            horizon_resolution_mode=cast(HorizonResolutionMode, horizon_resolution_mode),
            wall_clock_timestamp_column=wall_clock_timestamp_column,
            execution_entry_price_column=execution_entry_price_column,
            execution_exit_price_column=execution_exit_price_column,
            execution_latency_bars=execution_latency_bars,
            unresolved_execution_context_mode=cast(
                ExecutionUnresolvedContextMode,
                unresolved_execution_context_mode,
            ),
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
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise ValueError("target semantics JSON payload must decode to an object")
        return cls.from_dict(decoded)

__all__ = [
    "DEFAULT_EXECUTION_LATENCY_BARS",
    "DEFAULT_EXECUTION_PRICE_COLUMN",
    "DEFAULT_WALL_CLOCK_TIMESTAMP_COLUMN",
    "EXECUTION_UNRESOLVED_CONTEXT_FAIL",
    "EXECUTION_UNRESOLVED_CONTEXT_ZERO_RETURN",
    "HORIZON_RESOLUTION_BAR_INDEX",
    "HORIZON_RESOLUTION_WALL_CLOCK",
    "TARGET_SEMANTICS_CONTRACT_ID",
    "TARGET_SEMANTICS_CONTRACT_MAJOR",
    "TARGET_SEMANTICS_EPOCH_VERSION",
    "TARGET_SEMANTICS_REQUIRED_CAPABILITIES",
    "BinaryTargetConfig",
    "ExecutionUnresolvedContextMode",
    "HorizonResolutionMode",
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
