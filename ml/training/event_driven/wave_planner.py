"""
Wave-based scaling utilities for the streaming training pipeline.

This module centralizes the heuristics used to decide the next capacity
increment ("wave") for the TFT streaming teacher. It inspects recent cohort
metrics and resource telemetry to recommend updated caps while flagging
regressions or resource saturation risks.
"""
from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from statistics import mean


@dataclass(frozen=True, slots=True)
class WaveBounds:
    """Operational limits for a streaming training wave."""

    shard_row_budget: int
    max_total_rows: int
    max_total_sequences: int
    max_shards: int

    def __post_init__(self) -> None:
        """Validate bound invariants."""
        if self.shard_row_budget <= 0:
            raise ValueError("shard_row_budget must be positive")
        if self.max_total_rows <= 0:
            raise ValueError("max_total_rows must be positive")
        if self.max_total_sequences <= 0:
            raise ValueError("max_total_sequences must be positive")
        if self.max_shards <= 0:
            raise ValueError("max_shards must be positive")
        if self.max_total_rows < self.shard_row_budget:
            raise ValueError("max_total_rows must be >= shard_row_budget")
        if self.max_total_sequences > self.max_total_rows:
            raise ValueError("max_total_sequences must be <= max_total_rows")


@dataclass(frozen=True, slots=True)
class WaveSample:
    """Observed telemetry for a completed cohort run."""

    completed_at: datetime
    roc_auc: float | None
    pr_auc: float | None
    max_gpu_memory_mb: float | None


@dataclass(frozen=True, slots=True)
class WaveRecommendation:
    """Recommended next wave along with diagnostic notes."""

    current: WaveBounds
    proposed: WaveBounds
    notes: tuple[str, ...]
    warnings: tuple[str, ...]


def summarize_samples(samples: Sequence[WaveSample]) -> tuple[float | None, float | None]:
    """
    Compute summary metrics for the provided samples.

    Args:
        samples: Historical cohort metrics sorted in any order.

    Returns:
        Tuple of (best_roc_auc, recent_roc_auc_mean).
    """
    best_candidates = [sample.roc_auc for sample in samples if sample.roc_auc is not None]
    best_roc = max(best_candidates) if best_candidates else None
    recent = _take_recent(samples, window=3)
    recent_values = [sample.roc_auc for sample in recent if sample.roc_auc is not None]
    recent_mean = mean(recent_values) if recent_values else None
    return best_roc, recent_mean


def recommend_next_wave(
    samples: Sequence[WaveSample],
    current_bounds: WaveBounds,
    *,
    row_increment: int = 30_000,
    shard_increment: int = 8,
    sequence_ratio_floor: float = 0.72,
    regression_delta: float = 0.01,
    device_memory_mb: float = 6_144.0,
    gpu_threshold_ratio: float = 0.85,
) -> WaveRecommendation:
    """
    Produce a recommendation for the next streaming wave.

    Args:
        samples: Historical cohort telemetry (newest first or unsorted).
        current_bounds: Limits used for the most recent wave.
        row_increment: Row cap increase to evaluate for the next wave.
        shard_increment: Additional shard budget for the next wave.
        sequence_ratio_floor: Minimum ratio of sequences/rows to maintain.
        regression_delta: ROC-AUC drop that triggers a regression warning.
        device_memory_mb: Physical GPU memory for saturation checks.
        gpu_threshold_ratio: Allowed GPU utilisation fraction before warning.

    Returns:
        WaveRecommendation describing the proposed bounds and diagnostics.

    Raises:
        ValueError: If increments or ratios are invalid.
    """
    if row_increment <= 0:
        raise ValueError("row_increment must be positive")
    if shard_increment <= 0:
        raise ValueError("shard_increment must be positive")
    if not (0.0 < sequence_ratio_floor <= 1.0):
        raise ValueError("sequence_ratio_floor must be in (0, 1]")
    if regression_delta < 0.0:
        raise ValueError("regression_delta must be non-negative")
    if device_memory_mb <= 0.0:
        raise ValueError("device_memory_mb must be positive")
    if not (0.0 < gpu_threshold_ratio <= 1.0):
        raise ValueError("gpu_threshold_ratio must be in (0, 1]")

    sorted_samples = sorted(samples, key=lambda sample: sample.completed_at, reverse=True)
    best_roc, recent_mean = summarize_samples(sorted_samples)

    sequence_ratio = current_bounds.max_total_sequences / current_bounds.max_total_rows
    target_ratio = max(sequence_ratio, sequence_ratio_floor)
    proposed_rows = current_bounds.max_total_rows + row_increment
    proposed_sequences = max(
        ceil(proposed_rows * target_ratio),
        current_bounds.max_total_sequences + ceil(row_increment * sequence_ratio_floor),
    )
    proposed_sequences = min(proposed_sequences, proposed_rows)

    proposed_shards = current_bounds.max_shards + shard_increment
    proposed_shard_budget = min(
        proposed_rows,
        current_bounds.shard_row_budget + row_increment,
    )

    proposed_bounds = WaveBounds(
        shard_row_budget=proposed_shard_budget,
        max_total_rows=proposed_rows,
        max_total_sequences=proposed_sequences,
        max_shards=proposed_shards,
    )

    notes: list[str] = []
    warnings: list[str] = []

    if best_roc is not None:
        notes.append(f"Best ROC-AUC observed: {best_roc:.3f}")
    if recent_mean is not None:
        notes.append(f"Mean ROC-AUC over last wave: {recent_mean:.3f}")

    if best_roc is not None and recent_mean is not None and best_roc - recent_mean >= regression_delta:
        warnings.append(
            f"Recent ROC-AUC mean {recent_mean:.3f} lags best {best_roc:.3f} by {best_roc - recent_mean:.3f}",
        )

    peak_gpu = _max_gpu(sorted_samples)
    if peak_gpu is not None:
        notes.append(f"Peak GPU usage: {peak_gpu:.1f} MiB")
        if peak_gpu / device_memory_mb >= gpu_threshold_ratio:
            warnings.append(
                "GPU consumption exceeds threshold "
                f"({peak_gpu:.1f} MiB ≥ {gpu_threshold_ratio * 100:.1f}% of {device_memory_mb:.0f} MiB)",
            )

    if not sorted_samples:
        warnings.append("No manifests available; defaulting to incremental increase.")

    return WaveRecommendation(
        current=current_bounds,
        proposed=proposed_bounds,
        notes=tuple(notes),
        warnings=tuple(warnings),
    )


def _take_recent(samples: Sequence[WaveSample], *, window: int) -> list[WaveSample]:
    if window <= 0:
        raise ValueError("window must be positive")
    sorted_samples = sorted(samples, key=lambda sample: sample.completed_at, reverse=True)
    return list(sorted_samples[:window])


def _max_gpu(samples: Iterable[WaveSample]) -> float | None:
    peak_values = [sample.max_gpu_memory_mb for sample in samples if sample.max_gpu_memory_mb is not None]
    if not peak_values:
        return None
    return max(peak_values)


__all__ = [
    "WaveBounds",
    "WaveRecommendation",
    "WaveSample",
    "recommend_next_wave",
    "summarize_samples",
]
