"""Configuration classes for the streaming training pipeline."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from msgspec import ValidationError
from msgspec import field as msgspec_field

from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


def _env_seed(
    source: Mapping[str, str],
    name: str,
    default: int | None = None,
) -> int | None:
    raw = source.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_optional_positive_float(source: Mapping[str, str], name: str) -> float | None:
    raw = source.get(name)
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    return value if value > 0.0 else None


class DatasetServiceConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for the dataset planning microservice."""

    parquet_root: str
    shard_row_budget: PositiveInt = 200_000
    max_total_rows: PositiveInt | None = None
    max_total_sequences: PositiveInt | None = None
    max_shards: PositiveInt | None = None
    plan_cache_ttl_seconds: PositiveInt = 600
    retry_backoff_seconds: PositiveFloat = 5.0
    max_retry_attempts: PositiveInt = 3
    include_macro: bool = False
    include_calendar: bool = False
    include_events: bool = False
    include_earnings: bool = False
    include_micro: bool = False
    include_l2: bool = False
    include_macro_revisions: bool = False
    include_macro_deltas: bool = False
    include_calendar_lags: bool = False
    include_clustering_tags: bool = False
    include_context_features: bool = False
    min_positive_rate: float | None = None
    max_positive_rate: float | None = None
    positive_rate_baseline: float | None = None
    positive_rate_drift_tolerance: float = 0.05
    schema_reference_columns: tuple[str, ...] = ()
    schema_alert_on_unexpected: bool = True
    known_future_pairs: tuple[str, ...] = ()
    known_future_sample_rows: PositiveInt | None = None

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if self.max_total_rows is not None and self.max_total_rows < self.shard_row_budget:
            raise ValidationError("max_total_rows must be >= shard_row_budget when set")
        for name, value in (
            ("min_positive_rate", self.min_positive_rate),
            ("max_positive_rate", self.max_positive_rate),
            ("positive_rate_baseline", self.positive_rate_baseline),
        ):
            if value is not None and not (0.0 <= float(value) <= 1.0):
                raise ValidationError(f"{name} must be within [0, 1]")
        if (
            self.min_positive_rate is not None
            and self.max_positive_rate is not None
            and float(self.min_positive_rate) > float(self.max_positive_rate)
        ):
            raise ValidationError("min_positive_rate must be <= max_positive_rate when both set")
        if float(self.positive_rate_drift_tolerance) < 0.0:
            raise ValidationError("positive_rate_drift_tolerance must be >= 0.0")
        if self.known_future_sample_rows is not None and int(self.known_future_sample_rows) <= 0:
            raise ValidationError("known_future_sample_rows must be >= 1 when provided")

    @classmethod
    def from_env(
        cls,
        parquet_root: str,
        *,
        env: Mapping[str, str] | None = None,
    ) -> DatasetServiceConfig:
        """
        Build a planning configuration from environment variables.

        Environment overrides:
            ML_STREAMING_SHARD_ROW_BUDGET: integer shard budget
            ML_STREAMING_MAX_TOTAL_ROWS: integer or <=0 for unlimited
            ML_STREAMING_MAX_TOTAL_SEQUENCES: integer or <=0 for unlimited
            ML_STREAMING_MAX_SHARDS: integer or <=0 for unlimited
            ML_STREAMING_INCLUDE_MACRO: truthy/falsey toggle
            ML_STREAMING_INCLUDE_CALENDAR: truthy/falsey toggle
            ML_STREAMING_INCLUDE_EVENTS: truthy/falsey toggle
            ML_STREAMING_INCLUDE_EARNINGS: truthy/falsey toggle
            ML_STREAMING_INCLUDE_MICRO: truthy/falsey toggle
            ML_STREAMING_INCLUDE_L2: truthy/falsey toggle
            ML_STREAMING_INCLUDE_MACRO_REVISIONS: truthy/falsey toggle
            ML_STREAMING_INCLUDE_MACRO_DELTAS: truthy/falsey toggle
            ML_STREAMING_INCLUDE_CALENDAR_LAGS: truthy/falsey toggle
            ML_STREAMING_INCLUDE_CLUSTERING_TAGS: truthy/falsey toggle
            ML_STREAMING_INCLUDE_CONTEXT_FEATURES: truthy/falsey toggle
            ML_STREAMING_MIN_POSITIVE_RATE: float within [0, 1]
            ML_STREAMING_MAX_POSITIVE_RATE: float within [0, 1]
            ML_STREAMING_POSITIVE_RATE_BASELINE: float within [0, 1]
            ML_STREAMING_POSITIVE_RATE_DRIFT_TOLERANCE: non-negative float
            ML_STREAMING_SCHEMA_REFERENCE_COLUMNS: comma-separated feature names
            ML_STREAMING_SCHEMA_ALERT_ON_UNEXPECTED: truthy/falsey toggle
            ML_STREAMING_KNOWN_FUTURE_PAIRS: semicolon-separated column pairs (evaluation:effective)
            ML_STREAMING_KNOWN_FUTURE_SAMPLE_ROWS: integer row cap for known-future scans

        Args:
            parquet_root: Root directory containing parquet datasets.
            env: Optional environment mapping. Defaults to ``os.environ``.

        Returns:
            DatasetServiceConfig: Parsed configuration.
        """
        source = env or os.environ

        def _int(name: str, default: int | None) -> int | None:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                value = int(str(raw).strip())
            except ValueError:
                return default
            return None if value <= 0 else value

        def _truthy(name: str, default: bool) -> bool:
            raw = source.get(name)
            if raw is None:
                return default
            normalized = str(raw).strip().lower()
            return normalized in {"1", "true", "yes", "y", "on"}

        def _unit_interval(name: str, default: float | None) -> float | None:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                value = float(str(raw).strip())
            except ValueError:
                return default
            if value < 0.0 or value > 1.0:
                return default
            return value

        def _tuple(name: str) -> tuple[str, ...]:
            raw = source.get(name)
            if raw is None:
                return ()
            parts = [part.strip() for part in str(raw).split(",")]
            return tuple(part for part in parts if part)

        def _pair_list(name: str) -> tuple[str, ...]:
            raw = source.get(name)
            if raw is None:
                return ()
            sections = [section.strip() for section in str(raw).split(";")]
            return tuple(section for section in sections if section)

        def _float_default(name: str, default: float) -> float:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                return float(str(raw).strip())
            except ValueError:
                return default

        return cls(
            parquet_root=parquet_root,
            shard_row_budget=_int("ML_STREAMING_SHARD_ROW_BUDGET", 200_000) or 200_000,
            max_total_rows=_int("ML_STREAMING_MAX_TOTAL_ROWS", None),
            max_total_sequences=_int("ML_STREAMING_MAX_TOTAL_SEQUENCES", None),
            max_shards=_int("ML_STREAMING_MAX_SHARDS", None),
            include_macro=_truthy("ML_STREAMING_INCLUDE_MACRO", False),
            include_calendar=_truthy("ML_STREAMING_INCLUDE_CALENDAR", False),
            include_events=_truthy("ML_STREAMING_INCLUDE_EVENTS", False),
            include_earnings=_truthy("ML_STREAMING_INCLUDE_EARNINGS", False),
            include_micro=_truthy("ML_STREAMING_INCLUDE_MICRO", False),
            include_l2=_truthy("ML_STREAMING_INCLUDE_L2", False),
            include_macro_revisions=_truthy("ML_STREAMING_INCLUDE_MACRO_REVISIONS", False),
            include_macro_deltas=_truthy("ML_STREAMING_INCLUDE_MACRO_DELTAS", False),
            include_calendar_lags=_truthy("ML_STREAMING_INCLUDE_CALENDAR_LAGS", False),
            include_clustering_tags=_truthy("ML_STREAMING_INCLUDE_CLUSTERING_TAGS", False),
            include_context_features=_truthy("ML_STREAMING_INCLUDE_CONTEXT_FEATURES", False),
            min_positive_rate=_unit_interval("ML_STREAMING_MIN_POSITIVE_RATE", None),
            max_positive_rate=_unit_interval("ML_STREAMING_MAX_POSITIVE_RATE", None),
            positive_rate_baseline=_unit_interval("ML_STREAMING_POSITIVE_RATE_BASELINE", None),
            positive_rate_drift_tolerance=_float_default(
                "ML_STREAMING_POSITIVE_RATE_DRIFT_TOLERANCE",
                0.05,
            ),
            schema_reference_columns=_tuple("ML_STREAMING_SCHEMA_REFERENCE_COLUMNS"),
            schema_alert_on_unexpected=_truthy(
                "ML_STREAMING_SCHEMA_ALERT_ON_UNEXPECTED",
                True,
            ),
            known_future_pairs=_pair_list("ML_STREAMING_KNOWN_FUTURE_PAIRS"),
            known_future_sample_rows=_int("ML_STREAMING_KNOWN_FUTURE_SAMPLE_ROWS", None),
        )


@dataclass(frozen=True, slots=True)
class CurriculumGuardContext:
    """Runtime signals evaluated by curriculum guard rules."""

    total_rows: int
    recent_roc_auc: float | None = None
    current_backlog: int | None = None
    recent_gpu_mb: float | None = None


class CurriculumStageConfig(NautilusConfig, kw_only=True, frozen=True):
    """Single curriculum stage mapping a row ceiling to a train fraction."""

    max_total_rows: PositiveInt | None = None
    train_fraction: float = 0.8
    label: str | None = None

    def __post_init__(self) -> None:
        if self.max_total_rows is not None and int(self.max_total_rows) <= 0:
            raise ValidationError("max_total_rows must be >= 1 when provided")
        if not (0.0 < float(self.train_fraction) < 1.0):
            raise ValidationError("train_fraction must be within (0, 1)")
        if self.label is not None and not self.label.strip():
            raise ValidationError("label must be non-empty when provided")


class CurriculumGuardRule(NautilusConfig, kw_only=True, frozen=True):
    """Guard applied to a curriculum stage before the fraction takes effect."""

    stage_label: str
    min_total_rows: PositiveInt | None = None
    max_total_rows: PositiveInt | None = None
    min_roc_auc: float | None = None
    max_backlog: PositiveInt | None = None
    max_gpu_mb: PositiveFloat | None = None
    fallback_train_fraction: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.stage_label.strip():
            raise ValidationError("stage_label must be non-empty")
        if self.min_roc_auc is not None and not (0.0 <= float(self.min_roc_auc) <= 1.0):
            raise ValidationError("min_roc_auc must be within [0, 1]")
        if self.fallback_train_fraction is not None and not (
            0.0 < float(self.fallback_train_fraction) < 1.0
        ):
            raise ValidationError("fallback_train_fraction must be within (0, 1)")
        if self.reason is not None and not self.reason.strip():
            raise ValidationError("reason must be non-empty when provided")

    def applies_to(self, label: str | None) -> bool:
        """Return True when the guard should evaluate for the provided stage label."""
        return bool(label and label == self.stage_label)

    def is_satisfied(self, context: CurriculumGuardContext) -> bool:
        """Return True when the guard passes for the provided runtime context."""
        if self.min_total_rows is not None and context.total_rows < int(self.min_total_rows):
            return False
        if self.max_total_rows is not None and context.total_rows > int(self.max_total_rows):
            return False
        if self.min_roc_auc is not None:
            roc = context.recent_roc_auc
            if roc is None or float(roc) < float(self.min_roc_auc):
                return False
        if self.max_backlog is not None:
            backlog = context.current_backlog
            if backlog is not None and backlog > int(self.max_backlog):
                return False
        if self.max_gpu_mb is not None:
            gpu = context.recent_gpu_mb
            if gpu is not None and float(gpu) >= float(self.max_gpu_mb):
                return False
        return True

    def guard_reason(self, *, context: CurriculumGuardContext) -> str:
        """Return a human-readable reason for failing the guard."""
        if self.reason:
            return self.reason
        parts: list[str] = []
        if self.min_total_rows is not None and context.total_rows < int(self.min_total_rows):
            parts.append(
                f"rows<{int(self.min_total_rows)} (actual={context.total_rows})",
            )
        if self.max_total_rows is not None and context.total_rows > int(self.max_total_rows):
            parts.append(
                f"rows>{int(self.max_total_rows)} (actual={context.total_rows})",
            )
        if self.min_roc_auc is not None:
            roc = context.recent_roc_auc
            if roc is None:
                parts.append("roc_auc unavailable")
            elif float(roc) < float(self.min_roc_auc):
                parts.append(f"roc_auc<{float(self.min_roc_auc):.3f} (actual={roc:.3f})")
        if self.max_backlog is not None:
            backlog = context.current_backlog
            if backlog is not None and backlog > int(self.max_backlog):
                parts.append(f"backlog>{int(self.max_backlog)} (actual={backlog})")
        if self.max_gpu_mb is not None:
            gpu = context.recent_gpu_mb
            if gpu is not None and float(gpu) >= float(self.max_gpu_mb):
                parts.append(f"gpu>={float(self.max_gpu_mb):.1f} MiB (actual={gpu:.1f})")
        return "; ".join(parts) if parts else "curriculum guard blocked stage"


@dataclass(frozen=True, slots=True)
class CurriculumResolution:
    """Resolved train fraction along with guard metadata."""

    train_fraction: float
    stage_label: str | None
    guard_reason: str | None


class CurriculumScheduleConfig(NautilusConfig, kw_only=True, frozen=True):
    """Curriculum schedule controlling train fraction as datasets grow."""

    enabled: bool = False
    stages: tuple[CurriculumStageConfig, ...] = ()
    default_train_fraction: float = 0.8
    guards: tuple[CurriculumGuardRule, ...] = ()

    def __post_init__(self) -> None:
        if not (0.0 < float(self.default_train_fraction) < 1.0):
            raise ValidationError("default_train_fraction must be within (0, 1)")

    def resolve_with_context(
        self,
        *,
        total_rows: int,
        fallback: float | None = None,
        context: CurriculumGuardContext | None = None,
    ) -> CurriculumResolution:
        """Return the effective train fraction plus guard metadata."""
        baseline = fallback if fallback is not None else self.default_train_fraction
        if not self.enabled or not self.stages:
            return CurriculumResolution(train_fraction=baseline, stage_label=None, guard_reason=None)

        stage = self._select_stage(total_rows)
        if stage is None:
            return CurriculumResolution(train_fraction=baseline, stage_label=None, guard_reason=None)

        fraction = stage.train_fraction
        guard_reason: str | None = None
        guard_context = context or CurriculumGuardContext(total_rows=total_rows)
        for guard in self.guards:
            if not guard.applies_to(stage.label):
                continue
            if guard.is_satisfied(guard_context):
                continue
            guard_reason = guard.guard_reason(context=guard_context)
            fraction = (
                guard.fallback_train_fraction if guard.fallback_train_fraction is not None else baseline
            )
            break

        return CurriculumResolution(
            train_fraction=fraction,
            stage_label=stage.label,
            guard_reason=guard_reason,
        )

    def resolve_fraction(self, *, total_rows: int, fallback: float | None = None) -> float:
        """Return the effective train fraction for the provided row count."""
        return self.resolve_with_context(total_rows=total_rows, fallback=fallback).train_fraction

    def _select_stage(self, total_rows: int) -> CurriculumStageConfig | None:
        for stage in self.stages:
            limit = int(stage.max_total_rows) if stage.max_total_rows is not None else None
            if limit is None or total_rows <= limit:
                return stage
        return self.stages[-1] if self.stages else None


class EnsembleMemberConfig(NautilusConfig, kw_only=True, frozen=True):
    """External logits artefact blended with the current worker output."""

    artifact_path: str
    weight: PositiveFloat = 1.0
    required: bool = False

    def __post_init__(self) -> None:
        if not self.artifact_path.strip():
            raise ValidationError("artifact_path must be non-empty")
        if float(self.weight) <= 0.0:
            raise ValidationError("weight must be > 0")


class StreamingEnsembleConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for post-training ensemble blending."""

    enabled: bool = False
    blend_mode: str = "weighted"
    normalize_weights: bool = True
    members: tuple[EnsembleMemberConfig, ...] = ()

    def __post_init__(self) -> None:
        mode = self.blend_mode.strip().lower()
        if mode not in {"weighted", "mean"}:
            raise ValidationError("blend_mode must be either 'weighted' or 'mean'")
        if self.enabled and not self.members:
            raise ValidationError("ensemble enabled but no members configured")

    @property
    def normalized_blend_mode(self) -> str:
        """Return the normalized blend mode for downstream consumers."""
        return self.blend_mode.strip().lower()


def parse_curriculum_stage_spec(spec: str) -> CurriculumStageConfig:
    """Parse ``<max_rows>:<train_fraction>[:label]`` strings into ``CurriculumStageConfig``."""
    text = spec.strip()
    if not text:
        raise ValueError("curriculum stage specification cannot be empty")
    try:
        limit_part, remainder = text.split(":", 1)
    except ValueError as exc:  # pragma: no cover - trivial parsing guard
        raise ValueError("curriculum stage must be formatted as '<rows>:<fraction>'") from exc

    fraction_part: str
    label_value: str | None = None
    if ":" in remainder:
        fraction_part, label_part = remainder.split(":", 1)
        label_part = label_part.strip()
        label_value = label_part or None
    else:
        fraction_part = remainder

    limit_value: PositiveInt | None
    normalized_limit = limit_part.strip().lower()
    if not normalized_limit or normalized_limit in {"*", "inf", "max"}:
        limit_value = None
    else:
        try:
            parsed_limit = int(normalized_limit.replace("_", ""))
        except ValueError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"invalid curriculum max_rows '{limit_part}'") from exc
        if parsed_limit <= 0:
            limit_value = None
        else:
            limit_value = parsed_limit

    try:
        fraction_value = float(fraction_part)
    except ValueError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"invalid curriculum train_fraction '{fraction_part}'") from exc
    if not (0.0 < fraction_value < 1.0):
        raise ValueError("curriculum train_fraction must be within (0, 1)")

    return CurriculumStageConfig(
        max_total_rows=limit_value,
        train_fraction=fraction_value,
        label=label_value,
    )


def parse_curriculum_guard_spec(spec: str) -> CurriculumGuardRule:
    """Parse ``label:key=value,...`` strings into ``CurriculumGuardRule``."""
    text = spec.strip()
    if not text:
        raise ValueError("curriculum guard specification cannot be empty")
    label, sep, remainder = text.partition(":")
    if not sep:
        raise ValueError("curriculum guard must be formatted as 'label:key=value,...'")
    label = label.strip()
    if not label:
        raise ValueError("curriculum guard label cannot be empty")
    if not remainder.strip():
        raise ValueError("curriculum guard requires at least one key=value pair")
    min_rows_value: int | None = None
    max_rows_value: int | None = None
    min_roc_value: float | None = None
    max_backlog_value: int | None = None
    max_gpu_value: float | None = None
    fallback_fraction_value: float | None = None
    reason_value: str | None = None
    for chunk in remainder.split(","):
        token = chunk.strip()
        if not token:
            continue
        key, value_sep, raw_value = token.partition("=")
        if not value_sep:
            raise ValueError(f"curriculum guard token '{token}' must be formatted as key=value")
        key = key.strip().lower()
        raw_value = raw_value.strip()
        try:
            if key == "min_rows":
                min_rows_value = int(raw_value.replace("_", ""))
            elif key == "max_rows":
                max_rows_value = int(raw_value.replace("_", ""))
            elif key == "min_roc_auc":
                min_roc_value = float(raw_value)
            elif key == "max_backlog":
                max_backlog_value = int(raw_value.replace("_", ""))
            elif key == "max_gpu_mb":
                max_gpu_value = float(raw_value)
            elif key in {"fallback_fraction", "train_fraction"}:
                fallback_fraction_value = float(raw_value)
            elif key == "reason":
                reason_value = raw_value
            else:
                raise ValueError(f"unknown curriculum guard key '{key}'")
        except ValueError as exc:  # pragma: no cover - configuration guard
            raise ValueError(f"invalid value '{raw_value}' for curriculum guard key '{key}'") from exc
    return CurriculumGuardRule(
        stage_label=label,
        min_total_rows=min_rows_value,
        max_total_rows=max_rows_value,
        min_roc_auc=min_roc_value,
        max_backlog=max_backlog_value,
        max_gpu_mb=max_gpu_value,
        fallback_train_fraction=fallback_fraction_value,
        reason=reason_value,
    )


def parse_ensemble_member_spec(spec: str) -> EnsembleMemberConfig:
    """Parse ``path[:weight[:required]]`` strings into ``EnsembleMemberConfig``."""
    text = spec.strip()
    if not text:
        raise ValueError("ensemble member specification cannot be empty")
    parts = [segment.strip() for segment in text.split(":")]
    path = parts[0]
    if not path:
        raise ValueError("ensemble member path cannot be empty")

    weight_value = 1.0
    if len(parts) >= 2 and parts[1]:
        try:
            weight_value = float(parts[1])
        except ValueError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"invalid ensemble weight '{parts[1]}'") from exc
        if weight_value <= 0.0:
            raise ValueError("ensemble member weight must be > 0")

    required_flag = False
    if len(parts) >= 3 and parts[2]:
        normalized = parts[2].lower()
        if normalized in {"required", "true", "1", "yes", "y", "on"}:
            required_flag = True
        elif normalized in {"optional", "false", "0", "no", "n", "off"}:
            required_flag = False
        else:
            raise ValueError(
                "ensemble member requirement must be one of {required, optional, true, false}",
            )

    return EnsembleMemberConfig(
        artifact_path=path,
        weight=weight_value,
        required=required_flag,
    )


class StreamingWorkerConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for bounded streaming training workers."""

    max_total_rows: PositiveInt | None = 500_000
    max_total_sequences: PositiveInt | None = 300_000
    max_shards: PositiveInt | None = 4
    max_epochs: PositiveInt = 1
    bootstrap_sample_rows: PositiveInt = 10_000
    max_concurrent_jobs: PositiveInt = 1
    max_runtime_seconds: PositiveInt = 1_800
    heartbeat_interval_seconds: PositiveInt = 30
    max_retry_attempts: PositiveInt = 3
    retry_backoff_seconds: NonNegativeFloat = 5.0
    accelerator: str = "auto"
    devices: PositiveInt = 1
    model_id: str = "tft-streaming-teacher"
    train_fraction: PositiveFloat = 0.8
    logits_artifact_key: str = "logits"
    validation_metric: str = "roc_auc"
    gpu_memory_monitor_interval_seconds: NonNegativeFloat | None = 30.0
    hidden_size: PositiveInt = 16
    lstm_layers: PositiveInt = 1
    attention_head_size: PositiveInt = 2
    dropout: float = 0.1
    learning_rate: PositiveFloat = 3e-4
    optimizer: str = "adam"
    lr_scheduler: str = "reduce_on_plateau"
    loss_name: str = "bce"
    loss_pos_weight: PositiveFloat | None = None
    enable_temperature_calibration: bool = True
    temperature_calibration_min: PositiveFloat = 0.25
    temperature_calibration_max: PositiveFloat = 5.0
    temperature_calibration_steps: PositiveInt = 25
    enable_platt_calibration: bool = False
    enable_isotonic_calibration: bool = False
    precision: str = "32"
    enable_amp: bool = False
    amp_precision: str = "16-mixed"
    amp_guard_threshold_mb: PositiveFloat | None = None
    checkpoint_dir: str | None = None
    checkpoint_interval_seconds: PositiveFloat | None = None
    checkpoint_interval_steps: PositiveInt | None = None
    checkpoint_retention: PositiveInt = 2
    curriculum: CurriculumScheduleConfig = msgspec_field(default_factory=CurriculumScheduleConfig)
    ensemble: StreamingEnsembleConfig = msgspec_field(default_factory=StreamingEnsembleConfig)
    validation_return_column: str | None = "forward_return"
    dataset_seed: NonNegativeInt | None = None
    worker_seed: NonNegativeInt | None = None

    def __post_init__(self) -> None:
        """Ensure time and resource constraints are consistent."""
        if self.max_concurrent_jobs > 1 and self.max_shards is not None:
            if self.max_shards < self.max_concurrent_jobs:
                raise ValidationError("max_shards must be >= max_concurrent_jobs when both set")
        if not (0.0 < float(self.train_fraction) < 1.0):
            raise ValidationError("train_fraction must be in the open interval (0, 1)")
        if not self.model_id.strip():
            raise ValidationError("model_id must be non-empty")
        if not self.logits_artifact_key.strip():
            raise ValidationError("logits_artifact_key must be non-empty")
        metric = self.validation_metric.strip().lower()
        if metric not in {"roc_auc"}:
            raise ValidationError("validation_metric must be one of {'roc_auc'}")
        if self.max_retry_attempts < 1:
            raise ValidationError("max_retry_attempts must be >= 1")
        if float(self.retry_backoff_seconds) < 0.0:
            raise ValidationError("retry_backoff_seconds must be >= 0.0")
        if self.gpu_memory_monitor_interval_seconds is not None and float(
            self.gpu_memory_monitor_interval_seconds,
        ) <= 0.0:
            raise ValidationError("gpu_memory_monitor_interval_seconds must be > 0 when set")
        if int(self.max_epochs) < 1:
            raise ValidationError("max_epochs must be >= 1")
        if int(self.bootstrap_sample_rows) < 1:
            raise ValidationError("bootstrap_sample_rows must be >= 1")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValidationError("dropout must be in the range [0.0, 1.0)")
        optimizer_normalized = self.optimizer.strip().lower()
        if optimizer_normalized not in {"adam", "adamw", "sgd", "rmsprop"}:
            raise ValidationError("optimizer must be one of {'adam', 'adamw', 'sgd', 'rmsprop'}")
        scheduler_normalized = self.lr_scheduler.strip().lower()
        if scheduler_normalized not in {"reduce_on_plateau", "onecycle", "cosine", "none"}:
            raise ValidationError(
                "lr_scheduler must be one of {'reduce_on_plateau', 'onecycle', 'cosine', 'none'}",
            )
        loss_normalized = self.loss_name.strip().lower()
        if loss_normalized not in {"bce", "poisson"}:
            raise ValidationError("loss_name must be one of {'bce', 'poisson'}")
        if self.loss_pos_weight is not None and float(self.loss_pos_weight) <= 0.0:
            raise ValidationError("loss_pos_weight must be > 0 when set")
        if loss_normalized != "bce" and self.loss_pos_weight is not None:
            raise ValidationError("loss_pos_weight is only allowed when loss_name='bce'")
        if float(self.temperature_calibration_min) <= 0.0:
            raise ValidationError("temperature_calibration_min must be > 0")
        if float(self.temperature_calibration_max) <= float(self.temperature_calibration_min):
            raise ValidationError("temperature_calibration_max must be greater than min")
        if int(self.temperature_calibration_steps) < 1:
            raise ValidationError("temperature_calibration_steps must be >= 1")
        if self.validation_return_column is not None and not self.validation_return_column.strip():
            raise ValidationError("validation_return_column must be None or a non-empty string")
        if not self.precision.strip():
            raise ValidationError("precision must be non-empty")
        if not self.amp_precision.strip():
            raise ValidationError("amp_precision must be non-empty")
        if self.enable_amp and not self.amp_precision.strip():
            raise ValidationError("amp_precision must be non-empty when AMP is enabled")
        if self.amp_guard_threshold_mb is not None and float(self.amp_guard_threshold_mb) <= 0.0:
            raise ValidationError("amp_guard_threshold_mb must be > 0 when set")
        if self.checkpoint_dir is not None and not self.checkpoint_dir.strip():
            raise ValidationError("checkpoint_dir must be non-empty when provided")
        if self.checkpoint_interval_seconds is not None and float(self.checkpoint_interval_seconds) <= 0.0:
            raise ValidationError("checkpoint_interval_seconds must be > 0 when set")
        if self.checkpoint_interval_steps is not None and int(self.checkpoint_interval_steps) <= 0:
            raise ValidationError("checkpoint_interval_steps must be > 0 when set")
        if int(self.checkpoint_retention) < 1:
            raise ValidationError("checkpoint_retention must be >= 1")

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
    ) -> StreamingWorkerConfig:
        """
        Build a worker configuration from environment overrides.

        Environment overrides:
            ML_STREAMING_MAX_TOTAL_ROWS: integer or <=0 for unlimited
            ML_STREAMING_MAX_TOTAL_SEQUENCES: integer or <=0 for unlimited
            ML_STREAMING_MAX_SHARDS: integer or <=0 for unlimited
            ML_STREAMING_MAX_EPOCHS: integer epochs
            ML_STREAMING_BOOTSTRAP_SAMPLE_ROWS: integer rows for bootstrap sample
            ML_STREAMING_MAX_RUNTIME_SECONDS: integer runtime budget
            ML_STREAMING_HEARTBEAT_INTERVAL_SECONDS: integer heartbeat cadence
            ML_STREAMING_MAX_RETRY_ATTEMPTS: integer retry limit
            ML_STREAMING_RETRY_BACKOFF_SECONDS: float retry backoff
            ML_STREAMING_ACCELERATOR: accelerator identifier (e.g., "auto", "cuda")
            ML_STREAMING_DEVICES: integer device count
            ML_STREAMING_TRAIN_FRACTION: float in (0, 1)
            ML_STREAMING_LOGITS_KEY: artifact key for logits
            ML_STREAMING_VALIDATION_METRIC: validation metric name
            ML_STREAMING_GPU_MONITOR_INTERVAL_SECONDS: GPU monitor interval (<=0 disables)
            ML_STREAMING_TFT_HIDDEN_SIZE: integer TFT hidden dimension
            ML_STREAMING_TFT_LSTM_LAYERS: integer LSTM layer count
            ML_STREAMING_TFT_ATTENTION_HEAD_SIZE: integer attention head size
            ML_STREAMING_TFT_DROPOUT: float dropout rate in [0, 1)
            ML_STREAMING_TFT_LEARNING_RATE: float learning rate
            ML_STREAMING_TFT_OPTIMIZER: optimizer name (adam, adamw, sgd, rmsprop)
            ML_STREAMING_TFT_LR_SCHEDULER: scheduler name (reduce_on_plateau, onecycle, cosine, none)
            ML_STREAMING_TFT_LOSS: teacher loss function (bce, poisson)
            ML_STREAMING_TFT_LOSS_POS_WEIGHT: positive class weight for BCE (>0)
            ML_STREAMING_VALIDATION_RETURN_COLUMN: forward-return column name (blank disables)
            ML_STREAMING_CHECKPOINT_DIR: directory used to persist checkpoints
            ML_STREAMING_CHECKPOINT_INTERVAL_SECONDS: positive float cadence for checkpoint saves
            ML_STREAMING_CHECKPOINT_INTERVAL_STEPS: positive integer step cadence for checkpoint saves
            ML_STREAMING_CHECKPOINT_RETENTION: positive integer count of checkpoints to retain
        """
        source = env or os.environ

        def _int(name: str, default: int | None) -> int | None:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                value = int(str(raw).strip())
            except ValueError:
                return default
            return None if value <= 0 else value

        def _float(name: str, default: float) -> float:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                value = float(str(raw).strip())
            except ValueError:
                return default
            return value if value > 0.0 else default

        def _optional_float(name: str, default: float | None) -> float | None:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                value = float(str(raw).strip())
            except ValueError:
                return default
            return None if value <= 0.0 else value

        def _optional_str(name: str, default: str | None) -> str | None:
            raw = source.get(name)
            if raw is None:
                return default
            value = str(raw).strip()
            return value or None

        def _raw_float(name: str, default: float) -> float:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                return float(str(raw).strip())
            except ValueError:
                return default

        def _metric(name: str, default: str) -> str:
            raw = source.get(name)
            if raw is None:
                return default
            normalized = str(raw).strip().lower()
            return normalized or default

        def _truthy(name: str, default: bool) -> bool:
            raw = source.get(name)
            if raw is None:
                return default
            normalized = str(raw).strip().lower()
            return normalized in {"1", "true", "yes", "y", "on"}

        train_fraction_value = _float("ML_STREAMING_TRAIN_FRACTION", 0.8)
        if not (0.0 < train_fraction_value < 1.0):
            train_fraction_value = 0.8

        curriculum_stages: tuple[CurriculumStageConfig, ...] = ()
        curriculum_raw = source.get("ML_STREAMING_CURRICULUM_STAGES")
        if curriculum_raw:
            parsed_stages: list[CurriculumStageConfig] = []
            for chunk in str(curriculum_raw).split(";"):
                text = chunk.strip()
                if not text:
                    continue
                try:
                    parsed_stages.append(parse_curriculum_stage_spec(text))
                except ValueError as exc:  # pragma: no cover - configuration error
                    raise ValidationError(str(exc)) from exc
            curriculum_stages = tuple(parsed_stages)

        curriculum_default = _raw_float(
            "ML_STREAMING_CURRICULUM_DEFAULT_TRAIN_FRACTION",
            train_fraction_value,
        )
        if not (0.0 < curriculum_default < 1.0):
            curriculum_default = train_fraction_value

        curriculum_guards: tuple[CurriculumGuardRule, ...] = ()
        guards_raw = source.get("ML_STREAMING_CURRICULUM_GUARDS")
        if guards_raw:
            parsed_guards: list[CurriculumGuardRule] = []
            for chunk in str(guards_raw).split(";"):
                text = chunk.strip()
                if not text:
                    continue
                try:
                    parsed_guards.append(parse_curriculum_guard_spec(text))
                except ValueError as exc:  # pragma: no cover - configuration error
                    raise ValidationError(str(exc)) from exc
            curriculum_guards = tuple(parsed_guards)

        loss_value = _metric("ML_STREAMING_TFT_LOSS", "bce")
        if loss_value not in {"bce", "poisson"}:
            loss_value = "bce"
        loss_pos_weight_raw = source.get("ML_STREAMING_TFT_LOSS_POS_WEIGHT")
        loss_pos_weight_value: float | None
        if loss_pos_weight_raw is None:
            loss_pos_weight_value = None
        else:
            try:
                parsed_loss_pos_weight = float(str(loss_pos_weight_raw).strip())
            except ValueError:
                parsed_loss_pos_weight = float("nan")
            loss_pos_weight_value = (
                parsed_loss_pos_weight if parsed_loss_pos_weight > 0.0 else None
            )
        if loss_value != "bce":
            loss_pos_weight_value = None

        ensemble_members: tuple[EnsembleMemberConfig, ...] = ()
        ensemble_raw = source.get("ML_STREAMING_ENSEMBLE_MEMBERS")
        if ensemble_raw:
            parsed_members: list[EnsembleMemberConfig] = []
            for chunk in str(ensemble_raw).split(";"):
                text = chunk.strip()
                if not text:
                    continue
                try:
                    parsed_members.append(parse_ensemble_member_spec(text))
                except ValueError as exc:  # pragma: no cover - configuration error
                    raise ValidationError(str(exc)) from exc
            ensemble_members = tuple(parsed_members)

        checkpoint_dir = _optional_str("ML_STREAMING_CHECKPOINT_DIR", None)
        checkpoint_interval_seconds = _env_optional_positive_float(
            source,
            "ML_STREAMING_CHECKPOINT_INTERVAL_SECONDS",
        )
        checkpoint_interval_steps = _int("ML_STREAMING_CHECKPOINT_INTERVAL_STEPS", None)
        checkpoint_retention = 2
        checkpoint_retention_raw = source.get("ML_STREAMING_CHECKPOINT_RETENTION")
        if checkpoint_retention_raw is not None:
            try:
                checkpoint_retention_candidate = int(str(checkpoint_retention_raw).strip())
            except ValueError:
                checkpoint_retention_candidate = checkpoint_retention
            checkpoint_retention = max(1, checkpoint_retention_candidate)

        return cls(
            max_total_rows=_int("ML_STREAMING_MAX_TOTAL_ROWS", None),
            max_total_sequences=_int("ML_STREAMING_MAX_TOTAL_SEQUENCES", None),
            max_shards=_int("ML_STREAMING_MAX_SHARDS", None),
            max_epochs=int(_int("ML_STREAMING_MAX_EPOCHS", 1) or 1),
            bootstrap_sample_rows=int(
                _int("ML_STREAMING_BOOTSTRAP_SAMPLE_ROWS", 10_000) or 10_000,
            ),
            max_runtime_seconds=int(_int("ML_STREAMING_MAX_RUNTIME_SECONDS", 1_800) or 1_800),
            heartbeat_interval_seconds=int(_int("ML_STREAMING_HEARTBEAT_INTERVAL_SECONDS", 30) or 30),
            max_retry_attempts=int(_int("ML_STREAMING_MAX_RETRY_ATTEMPTS", 3) or 3),
            retry_backoff_seconds=_float("ML_STREAMING_RETRY_BACKOFF_SECONDS", 5.0),
            accelerator=str(source.get("ML_STREAMING_ACCELERATOR", "auto")),
            devices=int(_int("ML_STREAMING_DEVICES", 1) or 1),
            train_fraction=float(train_fraction_value),
            logits_artifact_key=str(source.get("ML_STREAMING_LOGITS_KEY", "logits")),
            validation_metric=_metric("ML_STREAMING_VALIDATION_METRIC", "roc_auc"),
            gpu_memory_monitor_interval_seconds=_optional_float("ML_STREAMING_GPU_MONITOR_INTERVAL_SECONDS", 30.0),
            hidden_size=int(_int("ML_STREAMING_TFT_HIDDEN_SIZE", 16) or 16),
            lstm_layers=int(_int("ML_STREAMING_TFT_LSTM_LAYERS", 1) or 1),
            attention_head_size=int(_int("ML_STREAMING_TFT_ATTENTION_HEAD_SIZE", 2) or 2),
            dropout=_raw_float("ML_STREAMING_TFT_DROPOUT", 0.1),
            learning_rate=_raw_float("ML_STREAMING_TFT_LEARNING_RATE", 3e-4),
            optimizer=str(
                source.get("ML_STREAMING_TFT_OPTIMIZER", "adam"),
            ),
            lr_scheduler=str(
                source.get("ML_STREAMING_TFT_LR_SCHEDULER", "reduce_on_plateau"),
            ),
            loss_name=loss_value,
            loss_pos_weight=loss_pos_weight_value,
            enable_temperature_calibration=_truthy(
                "ML_STREAMING_TFT_ENABLE_TEMPERATURE_CALIBRATION",
                True,
            ),
            temperature_calibration_min=_raw_float(
                "ML_STREAMING_TFT_TEMPERATURE_MIN",
                0.25,
            ),
            temperature_calibration_max=_raw_float(
                "ML_STREAMING_TFT_TEMPERATURE_MAX",
                5.0,
            ),
            temperature_calibration_steps=int(
                _int("ML_STREAMING_TFT_TEMPERATURE_STEPS", 25) or 25,
            ),
            enable_platt_calibration=_truthy(
                "ML_STREAMING_TFT_ENABLE_PLATT_CALIBRATION",
                False,
            ),
            enable_isotonic_calibration=_truthy(
                "ML_STREAMING_TFT_ENABLE_ISOTONIC_CALIBRATION",
                False,
            ),
            precision=str(source.get("ML_STREAMING_PRECISION", "32")),
            enable_amp=_truthy("ML_STREAMING_ENABLE_AMP", False),
            amp_precision=str(source.get("ML_STREAMING_AMP_PRECISION", "16-mixed")),
            amp_guard_threshold_mb=_optional_float("ML_STREAMING_AMP_GUARD_THRESHOLD_MB", None),
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            checkpoint_interval_steps=checkpoint_interval_steps,
            checkpoint_retention=checkpoint_retention,
            curriculum=CurriculumScheduleConfig(
                enabled=_truthy("ML_STREAMING_CURRICULUM_ENABLED", False),
                stages=curriculum_stages,
                default_train_fraction=curriculum_default,
                guards=curriculum_guards,
            ),
            ensemble=StreamingEnsembleConfig(
                enabled=_truthy("ML_STREAMING_ENSEMBLE_ENABLED", False),
                blend_mode=str(source.get("ML_STREAMING_ENSEMBLE_BLEND_MODE", "weighted")),
                normalize_weights=_truthy("ML_STREAMING_ENSEMBLE_NORMALIZE_WEIGHTS", True),
                members=ensemble_members,
            ),
            validation_return_column=_optional_str(
                "ML_STREAMING_VALIDATION_RETURN_COLUMN",
                "forward_return",
            ),
            dataset_seed=_env_seed(source, "ML_STREAMING_DATASET_SEED", None),
            worker_seed=_env_seed(source, "ML_STREAMING_WORKER_SEED", None),
        )


class TrainingOrchestratorConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for event-driven streaming orchestrator."""

    command_topic: str
    result_topic: str
    heartbeat_topic: str
    max_in_flight_plans: PositiveInt = 8
    dataset_retry_limit: PositiveInt = 2
    worker_timeout_seconds: PositiveInt = 600
    retry_window_seconds: PositiveInt = 300
    max_plan_age_seconds: PositiveInt = 7_200
    saturation_heartbeat_limit: PositiveInt = 5
    backlog_warning_threshold: NonNegativeInt = 10
    enable_state_persistence: bool = True
    publish_retry_attempts: PositiveInt = 3
    publish_retry_delay_seconds: NonNegativeFloat = 0.5
    adaptive_backlog_threshold: NonNegativeInt | None = None
    adaptive_gpu_threshold_mb: NonNegativeFloat | None = None
    adaptive_cooldown_seconds: PositiveFloat = 120.0
    adaptive_interval_multiplier: PositiveFloat = 2.0

    def __post_init__(self) -> None:
        """Validate orchestrator settings."""
        if self.command_topic and self.command_topic == self.result_topic:
            raise ValidationError("command_topic and result_topic must differ when provided")
        if self.heartbeat_topic and self.heartbeat_topic == self.command_topic:
            raise ValidationError("heartbeat_topic must differ from command_topic when provided")
        if self.heartbeat_topic and self.heartbeat_topic == self.result_topic:
            raise ValidationError("heartbeat_topic must differ from result_topic when provided")
        if self.worker_timeout_seconds <= self.heartbeat_interval_hint:
            raise ValidationError("worker_timeout_seconds must exceed heartbeat interval hint")
        if self.max_plan_age_seconds <= self.worker_timeout_seconds:
            raise ValidationError("max_plan_age_seconds must exceed worker_timeout_seconds")
        if self.retry_window_seconds >= self.max_plan_age_seconds:
            raise ValidationError("retry_window_seconds must be less than max_plan_age_seconds")
        if int(self.publish_retry_attempts) < 1:
            raise ValidationError("publish_retry_attempts must be >= 1")
        if float(self.publish_retry_delay_seconds) < 0.0:
            raise ValidationError("publish_retry_delay_seconds must be >= 0")
        if self.adaptive_backlog_threshold is not None and int(self.adaptive_backlog_threshold) < 1:
            raise ValidationError("adaptive_backlog_threshold must be >= 1 when set")
        if self.adaptive_gpu_threshold_mb is not None and float(self.adaptive_gpu_threshold_mb) <= 0.0:
            raise ValidationError("adaptive_gpu_threshold_mb must be > 0 when set")
        if float(self.adaptive_cooldown_seconds) <= 0.0:
            raise ValidationError("adaptive_cooldown_seconds must be > 0")
        if float(self.adaptive_interval_multiplier) < 1.0:
            raise ValidationError("adaptive_interval_multiplier must be >= 1.0")

    @property
    def heartbeat_interval_hint(self) -> NonNegativeFloat:
        """Preferred heartbeat cadence derived from backlog threshold."""
        # Ensure non-zero divisor; fallback to 1.0s when backlog threshold is zero
        divisor = float(max(1, int(self.backlog_warning_threshold or 1)))
        return float(self.worker_timeout_seconds) / divisor


class StreamingPromotionConfig(NautilusConfig, kw_only=True, frozen=True):
    """Threshold configuration applied before promoting streaming cohorts."""

    min_roc_auc: float | None = 0.55
    min_pr_auc_multiple: float | None = 1.1
    max_log_loss: float | None = 0.75
    min_slippage_adjusted_sharpe: float | None = None
    min_hit_rate: float | None = None
    max_turnover: float | None = None
    max_drawdown: float | None = None
    max_ks_statistic: float | None = None
    max_calibration_drift: float | None = None

    def __post_init__(self) -> None:
        """Validate threshold ranges."""
        if self.min_roc_auc is not None and not (0.0 <= float(self.min_roc_auc) <= 1.0):
            raise ValidationError("min_roc_auc must be within [0.0, 1.0]")
        if self.min_pr_auc_multiple is not None and float(self.min_pr_auc_multiple) < 0.0:
            raise ValidationError("min_pr_auc_multiple must be >= 0.0 when set")
        if self.max_log_loss is not None and float(self.max_log_loss) < 0.0:
            raise ValidationError("max_log_loss must be >= 0.0 when set")
        if self.min_hit_rate is not None and not (0.0 <= float(self.min_hit_rate) <= 1.0):
            raise ValidationError("min_hit_rate must be within [0.0, 1.0]")
        if self.max_turnover is not None and not (0.0 <= float(self.max_turnover) <= 1.0):
            raise ValidationError("max_turnover must be within [0.0, 1.0]")
        if self.max_drawdown is not None and float(self.max_drawdown) < 0.0:
            raise ValidationError("max_drawdown must be >= 0.0 when set")
        if self.max_ks_statistic is not None and not (0.0 <= float(self.max_ks_statistic) <= 1.0):
            raise ValidationError("max_ks_statistic must be within [0.0, 1.0]")
        if self.max_calibration_drift is not None and float(self.max_calibration_drift) < 0.0:
            raise ValidationError("max_calibration_drift must be >= 0.0 when set")

    def metric_rules(self) -> tuple[tuple[str, str, float, bool], ...]:
        """Return secondary promotion rules: (metric, comparator, threshold, absolute)."""
        rules: list[tuple[str, str, float, bool]] = []
        if self.min_pr_auc_multiple is not None:
            rules.append(("pr_auc_multiple", "ge", float(self.min_pr_auc_multiple), False))
        if self.max_log_loss is not None:
            rules.append(("log_loss", "le", float(self.max_log_loss), False))
        if self.min_slippage_adjusted_sharpe is not None:
            rules.append(
                (
                    "economic_slippage_adjusted_sharpe",
                    "ge",
                    float(self.min_slippage_adjusted_sharpe),
                    False,
                ),
            )
        if self.min_hit_rate is not None:
            rules.append(("economic_hit_rate", "ge", float(self.min_hit_rate), False))
        if self.max_turnover is not None:
            rules.append(("economic_turnover", "le", float(self.max_turnover), False))
        if self.max_drawdown is not None:
            rules.append(("economic_max_drawdown", "le", float(self.max_drawdown), False))
        if self.max_ks_statistic is not None:
            rules.append(("stability_ks_statistic", "le", float(self.max_ks_statistic), False))
        if self.max_calibration_drift is not None:
            rules.append(("stability_calibration_drift", "le", float(self.max_calibration_drift), True))
        return tuple(rules)

    @classmethod
    def from_env(cls, *, env: Mapping[str, str] | None = None) -> StreamingPromotionConfig:
        """Build promotion thresholds from environment variables."""
        source = env or os.environ
        base = cls()

        def _value(name: str, default: float | None) -> float | None:
            raw = source.get(name)
            if raw is None:
                return default
            text = str(raw).strip()
            if text == "":
                return None
            try:
                return float(text)
            except ValueError:
                return default

        return cls(
            min_roc_auc=_value("ML_STREAMING_PROMOTE_MIN_ROC_AUC", base.min_roc_auc),
            min_pr_auc_multiple=_value(
                "ML_STREAMING_PROMOTE_MIN_PR_AUC_MULTIPLE",
                base.min_pr_auc_multiple,
            ),
            max_log_loss=_value("ML_STREAMING_PROMOTE_MAX_LOG_LOSS", base.max_log_loss),
            min_slippage_adjusted_sharpe=_value(
                "ML_STREAMING_PROMOTE_MIN_SLIPPAGE_SHARPE",
                base.min_slippage_adjusted_sharpe,
            ),
            min_hit_rate=_value("ML_STREAMING_PROMOTE_MIN_HIT_RATE", base.min_hit_rate),
            max_turnover=_value("ML_STREAMING_PROMOTE_MAX_TURNOVER", base.max_turnover),
            max_drawdown=_value("ML_STREAMING_PROMOTE_MAX_DRAWDOWN", base.max_drawdown),
            max_ks_statistic=_value("ML_STREAMING_PROMOTE_MAX_KS_STATISTIC", base.max_ks_statistic),
            max_calibration_drift=_value(
                "ML_STREAMING_PROMOTE_MAX_CALIBRATION_DRIFT",
                base.max_calibration_drift,
            ),
        )


class AzureScheduledEventsConfig(NautilusConfig, kw_only=True, frozen=True):
    """Configuration for polling Azure scheduled events on spot instances."""

    enabled: bool = False
    poll_interval_seconds: PositiveFloat = 5.0
    request_timeout_seconds: PositiveFloat = 2.0
    metadata_endpoint: str = "http://169.254.169.254/metadata/scheduledevents"
    api_version: str = "2020-07-01"
    resource_filter: tuple[str, ...] = msgspec_field(default_factory=tuple)
    event_types: tuple[str, ...] = ("Preempt",)
    status_filter: tuple[str, ...] = ("Scheduled", "InProgress")

    def __post_init__(self) -> None:
        """Validate Azure scheduled event polling parameters."""
        if float(self.poll_interval_seconds) <= 0.0:
            raise ValidationError("poll_interval_seconds must be > 0")
        if float(self.request_timeout_seconds) <= 0.0:
            raise ValidationError("request_timeout_seconds must be > 0")
        if not self.metadata_endpoint.strip():
            raise ValidationError("metadata_endpoint must be non-empty")
        if not self.api_version.strip():
            raise ValidationError("api_version must be non-empty")
        if not self.event_types:
            raise ValidationError("event_types must contain at least one entry")
        if not self.status_filter:
            raise ValidationError("status_filter must contain at least one entry")
        for label, values in (
            ("event_types", self.event_types),
            ("status_filter", self.status_filter),
            ("resource_filter", self.resource_filter),
        ):
            for value in values:
                if not str(value).strip():
                    raise ValidationError(f"{label} entries must be non-empty strings")

    def build_request_url(self) -> str:
        """Return the metadata endpoint including the API version query string."""
        endpoint = self.metadata_endpoint.strip()
        if "api-version=" in endpoint.lower():
            return endpoint
        separator = "&" if "?" in endpoint else "?"
        return f"{endpoint}{separator}api-version={self.api_version}"


class StreamingPersistenceConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for the streaming training persistence worker.

    Args:
        enabled: Whether the worker should actively poll the message bus.
        state_path: Filesystem path used to persist streaming state snapshots.
        batch_size: Maximum number of Redis stream entries to process per poll.
        block_ms: Milliseconds to block on each Redis `XREAD` invocation.
        poll_interval_seconds: Delay applied after idle polls to reduce load.

    Example:
        >>> cfg = StreamingPersistenceConfig()
        >>> cfg.batch_size
        128
    """

    enabled: bool = True
    state_path: str = "./ml_out/streaming_training_state.json"
    batch_size: PositiveInt = 128
    block_ms: NonNegativeInt = 1_000
    poll_interval_seconds: NonNegativeFloat = 0.5

    def __post_init__(self) -> None:
        """Validate configuration fields."""
        if not self.state_path.strip():
            raise ValidationError("state_path must be non-empty")
        if float(self.poll_interval_seconds) < 0.0:
            raise ValidationError("poll_interval_seconds must be >= 0.0")

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> StreamingPersistenceConfig:
        """
        Build persistence configuration from environment variables.

        Environment overrides:
            ML_STREAM_PERSIST_ENABLE: truthy/falsey toggle (default: current value)
            ML_STREAM_PERSIST_STATE_PATH: filesystem path for persistence snapshot
            ML_STREAM_PERSIST_BATCH_SIZE: integer batch size per poll
            ML_STREAM_PERSIST_BLOCK_MS: integer Redis block duration in ms
            ML_STREAM_PERSIST_POLL_INTERVAL_SECONDS: float idle interval between polls

        Args:
            env: Optional mapping of environment variables to read. Defaults to ``os.environ``.

        Returns:
            StreamingPersistenceConfig: Parsed configuration with overrides applied.
        """
        source = env or os.environ

        def _truthy(name: str, default: bool) -> bool:
            raw = source.get(name)
            if raw is None:
                return default
            normalized = str(raw).strip().lower()
            return normalized in {"1", "true", "yes", "y", "on"}

        def _int(name: str, default: int) -> int:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                return int(str(raw).strip())
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                return float(str(raw).strip())
            except ValueError:
                return default

        base = cls()
        state_path_raw = source.get("ML_STREAM_PERSIST_STATE_PATH", base.state_path)
        return cls(
            enabled=_truthy("ML_STREAM_PERSIST_ENABLE", bool(base.enabled)),
            state_path=str(state_path_raw),
            batch_size=_int("ML_STREAM_PERSIST_BATCH_SIZE", int(base.batch_size)),
            block_ms=_int("ML_STREAM_PERSIST_BLOCK_MS", int(base.block_ms)),
            poll_interval_seconds=_float(
                "ML_STREAM_PERSIST_POLL_INTERVAL_SECONDS",
                float(base.poll_interval_seconds),
            ),
        )


__all__ = [
    "AzureScheduledEventsConfig",
    "CurriculumGuardContext",
    "CurriculumGuardRule",
    "CurriculumResolution",
    "CurriculumScheduleConfig",
    "CurriculumStageConfig",
    "DatasetServiceConfig",
    "EnsembleMemberConfig",
    "StreamingEnsembleConfig",
    "StreamingPersistenceConfig",
    "StreamingPromotionConfig",
    "StreamingWorkerConfig",
    "TrainingOrchestratorConfig",
    "parse_curriculum_guard_spec",
    "parse_curriculum_stage_spec",
    "parse_ensemble_member_spec",
]
