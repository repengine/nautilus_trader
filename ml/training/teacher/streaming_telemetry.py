"""Telemetry DTOs for TFT streaming training."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from ml.training.teacher.streaming_loader import StreamingLimitSummary
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import TFTStreamingMetadata
from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_loader import count_sequences


@dataclass(slots=True, frozen=True)
class StreamingLoaderTelemetry:
    """Aggregate selection statistics for a streaming DataLoader."""

    loader: str
    total_shards: int
    selected_shards: int
    skipped_shards: int
    total_rows: int
    selected_rows: int
    skipped_rows: int
    total_sequences: int
    selected_sequences: int
    skipped_sequences: int
    instrument_rows_total: dict[str, int] = field(default_factory=dict)
    instrument_rows_selected: dict[str, int] = field(default_factory=dict)
    instrument_rows_skipped: dict[str, int] = field(default_factory=dict)
    instrument_sequences_total: dict[str, int] = field(default_factory=dict)
    instrument_sequences_selected: dict[str, int] = field(default_factory=dict)
    instrument_sequences_skipped: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_metadata(
        cls,
        loader: str,
        metadata: TFTStreamingMetadata,
        summary: StreamingLimitSummary,
        config: TFTStreamingConfig,
    ) -> StreamingLoaderTelemetry:
        """Return telemetry derived from limited metadata and skip summary."""
        selected_shards = len(metadata.shard_indices)
        selected_rows = sum(metadata.instrument_row_counts.values())
        selected_sequences = max(0, count_sequences(metadata, config))
        total_shards = selected_shards + summary.skipped_shards
        total_rows = selected_rows + summary.skipped_rows
        total_sequences = selected_sequences + summary.skipped_sequences
        selected_rows_map = dict(metadata.instrument_row_counts)
        if not selected_rows_map and summary.selected_instrument_rows:
            selected_rows_map = dict(summary.selected_instrument_rows)
        total_rows_map = dict(summary.total_instrument_rows) or dict(selected_rows_map)
        skipped_rows_map = {
            instrument: max(0, total_rows_map.get(instrument, 0) - selected_rows_map.get(instrument, 0))
            for instrument in set(total_rows_map) | set(selected_rows_map)
            if total_rows_map.get(instrument, 0) - selected_rows_map.get(instrument, 0) > 0
        }
        total_sequences_map = dict(summary.total_instrument_sequences)
        if not total_sequences_map:
            total_sequences_map = dict.fromkeys(selected_rows_map, 0)
        selected_sequences_map = dict(summary.selected_instrument_sequences)
        if not selected_sequences_map:
            selected_sequences_map = dict.fromkeys(selected_rows_map, 0)
        skipped_sequences_map = {
            instrument: max(
                0,
                total_sequences_map.get(instrument, 0) - selected_sequences_map.get(instrument, 0),
            )
            for instrument in set(total_sequences_map) | set(selected_sequences_map)
            if total_sequences_map.get(instrument, 0) - selected_sequences_map.get(instrument, 0) > 0
        }
        return cls(
            loader=loader,
            total_shards=total_shards,
            selected_shards=selected_shards,
            skipped_shards=summary.skipped_shards,
            total_rows=total_rows,
            selected_rows=selected_rows,
            skipped_rows=summary.skipped_rows,
            total_sequences=total_sequences,
            selected_sequences=selected_sequences,
            skipped_sequences=summary.skipped_sequences,
            instrument_rows_total=dict(sorted(total_rows_map.items())),
            instrument_rows_selected=dict(sorted(selected_rows_map.items())),
            instrument_rows_skipped=dict(sorted(skipped_rows_map.items())),
            instrument_sequences_total=dict(sorted(total_sequences_map.items())),
            instrument_sequences_selected=dict(sorted(selected_sequences_map.items())),
            instrument_sequences_skipped=dict(sorted(skipped_sequences_map.items())),
        )

    def as_dict(self) -> dict[str, object]:
        """Return telemetry as a serializable dictionary."""
        return {
            "loader": self.loader,
            "total_shards": self.total_shards,
            "selected_shards": self.selected_shards,
            "skipped_shards": self.skipped_shards,
            "total_rows": self.total_rows,
            "selected_rows": self.selected_rows,
            "skipped_rows": self.skipped_rows,
            "total_sequences": self.total_sequences,
            "selected_sequences": self.selected_sequences,
            "skipped_sequences": self.skipped_sequences,
            "instrument_rows_total": self.instrument_rows_total,
            "instrument_rows_selected": self.instrument_rows_selected,
            "instrument_rows_skipped": self.instrument_rows_skipped,
            "instrument_sequences_total": self.instrument_sequences_total,
            "instrument_sequences_selected": self.instrument_sequences_selected,
            "instrument_sequences_skipped": self.instrument_sequences_skipped,
        }

    def as_logging_extra(self) -> dict[str, object]:
        """Return telemetry formatted for structured logging."""
        return self.as_dict()


@dataclass(slots=True, frozen=True)
class StreamingEnsembleMemberTelemetry:
    """Inventory entry describing an ensemble peer."""

    artifact_path: str
    weight: float
    required: bool
    used: bool
    skipped_reason: str | None
    train_row_count: int | None
    validation_row_count: int | None

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable mapping for the ensemble member."""
        payload: dict[str, object] = {
            "artifact_path": self.artifact_path,
            "weight": float(self.weight),
            "required": self.required,
            "used": self.used,
        }
        if self.skipped_reason is not None:
            payload["skipped_reason"] = self.skipped_reason
        if self.train_row_count is not None:
            payload["train_row_count"] = int(self.train_row_count)
        if self.validation_row_count is not None:
            payload["validation_row_count"] = int(self.validation_row_count)
        return payload


@dataclass(slots=True, frozen=True)
class StreamingEnsembleTelemetry:
    """Telemetry describing ensemble blending decisions."""

    blend_mode: str
    normalize_weights: bool
    members: tuple[StreamingEnsembleMemberTelemetry, ...]
    members_used: int
    optional_members_skipped: int
    misaligned_members: int

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable mapping for ensemble telemetry."""
        return {
            "blend_mode": self.blend_mode,
            "normalize_weights": self.normalize_weights,
            "members": [member.as_dict() for member in self.members],
            "members_used": int(self.members_used),
            "optional_members_skipped": int(self.optional_members_skipped),
            "misaligned_members": int(self.misaligned_members),
        }


@dataclass(slots=True, frozen=True)
class StreamingEconomicTelemetry:
    """Economic diagnostics tracked for streaming runs."""

    slippage_adjusted_sharpe: float | None = None
    hit_rate: float | None = None
    turnover: float | None = None
    max_drawdown: float | None = None

    def as_dict(self) -> dict[str, float]:
        """Return a serialisable mapping for economic metrics."""
        payload: dict[str, float] = {}
        if self.slippage_adjusted_sharpe is not None:
            payload["slippage_adjusted_sharpe"] = float(self.slippage_adjusted_sharpe)
        if self.hit_rate is not None:
            payload["hit_rate"] = float(self.hit_rate)
        if self.turnover is not None:
            payload["turnover"] = float(self.turnover)
        if self.max_drawdown is not None:
            payload["max_drawdown"] = float(self.max_drawdown)
        return payload


@dataclass(slots=True, frozen=True)
class StreamingStabilityTelemetry:
    """Stability diagnostics tracked for streaming runs."""

    ks_statistic: float | None = None
    calibration_drift: float | None = None

    def as_dict(self) -> dict[str, float]:
        """Return a serialisable mapping for stability metrics."""
        payload: dict[str, float] = {}
        if self.ks_statistic is not None:
            payload["ks_statistic"] = float(self.ks_statistic)
        if self.calibration_drift is not None:
            payload["calibration_drift"] = float(self.calibration_drift)
        return payload


@dataclass(slots=True, frozen=True)
class ValidationReturnsTelemetry:
    """Diagnostics describing validation-return extraction."""

    fallback_join: bool
    mismatch_count: int
    missing_count: int

    def as_dict(self) -> dict[str, int | bool]:
        """Return validation-return diagnostics as a serializable mapping."""
        return {
            "fallback_join": bool(self.fallback_join),
            "mismatch_count": int(self.mismatch_count),
            "missing_count": int(self.missing_count),
        }


@dataclass(slots=True, frozen=True)
class StreamingRunTelemetry:
    """Telemetry describing a full streaming training run."""

    metadata_summary: TFTStreamingSummary
    caps: dict[str, object]
    train: StreamingLoaderTelemetry
    validation: StreamingLoaderTelemetry
    max_gpu_memory_mb: float | None = None
    ensemble: StreamingEnsembleTelemetry | None = None
    economic: StreamingEconomicTelemetry | None = None
    stability: StreamingStabilityTelemetry | None = None
    validation_returns: ValidationReturnsTelemetry | None = None

    def as_dict(self) -> dict[str, object]:
        """Return run telemetry as a serializable mapping."""
        payload: dict[str, object] = {
            "caps": self.caps,
            "metadata": {
                "total_shards": self.metadata_summary.total_shards,
                "total_rows": self.metadata_summary.total_rows,
                "max_shard_rows": self.metadata_summary.max_shard_rows,
            },
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
            "resources": (
                {"max_gpu_memory_mb": self.max_gpu_memory_mb}
                if self.max_gpu_memory_mb is not None
                else {}
            ),
        }
        if self.ensemble is not None:
            payload["ensemble"] = self.ensemble.as_dict()
        if self.economic is not None:
            economic_payload = self.economic.as_dict()
            if economic_payload:
                payload["economic"] = economic_payload
        if self.stability is not None:
            stability_payload = self.stability.as_dict()
            if stability_payload:
                payload["stability"] = stability_payload
        if self.validation_returns is not None:
            payload["validation_returns"] = self.validation_returns.as_dict()
        return payload


__all__ = [
    "StreamingEconomicTelemetry",
    "StreamingEnsembleMemberTelemetry",
    "StreamingEnsembleTelemetry",
    "StreamingLoaderTelemetry",
    "StreamingRunTelemetry",
    "StreamingStabilityTelemetry",
    "ValidationReturnsTelemetry",
]
