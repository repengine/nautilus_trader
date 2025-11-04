#!/usr/bin/env python3
"""
Compare top and bottom streaming cohorts by validation quality.

This helper examines saved streaming manifests (and their accompanying logits
artefacts) to surface differences between high- and low-performing cohorts.

Example:
    >>> from pathlib import Path
    >>> result = compare_cohorts(Path("ml_out/tft_streaming_artifacts/full_tft_95"))
    >>> len(result["top"])
    3
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_METRIC = "roc_auc"
DEFAULT_COUNT = 3

_METRIC_ALIASES: dict[str, str] = {
    "sharpe": "economic_slippage_adjusted_sharpe",
    "economic_sharpe": "economic_slippage_adjusted_sharpe",
    "hit_rate": "economic_hit_rate",
    "economic_hit_rate": "economic_hit_rate",
    "turnover": "economic_turnover",
    "economic_turnover": "economic_turnover",
    "drawdown": "economic_max_drawdown",
    "max_drawdown": "economic_max_drawdown",
    "z_std": "z_val_std",
    "z_mean": "z_val_mean",
    "val_returns_std": "val_returns_std",
    "val_returns_mean": "val_returns_mean",
}

_SPECIAL_METRICS: frozenset[str] = frozenset(
    (
        "z_val_std",
        "z_val_mean",
        "val_returns_std",
        "val_returns_mean",
    ),
)


@dataclass(slots=True, frozen=True)
class CohortStats:
    """Summary metrics for a streaming cohort."""

    plan_id: str
    metric_value: float | None
    metric_name: str
    manifest_path: Path
    logits_path: Path | None
    z_val_mean: float | None
    z_val_std: float | None
    val_returns_mean: float | None
    val_returns_std: float | None
    fallback_join: bool | None
    mismatch_count: int | None
    missing_count: int | None
    dataset_seed: int | None = None
    worker_seed: int | None = None
    worker_curriculum_enabled: bool | None = None
    worker_train_fraction: float | None = None
    worker_amp_enabled: bool | None = None
    worker_ensemble_enabled: bool | None = None
    worker_ensemble_members_configured: int | None = None
    worker_skipped_rows: int | None = None
    worker_skipped_sequences: int | None = None
    worker_skipped_shards: int | None = None
    z_val_quantiles: dict[str, float] | None = None
    val_returns_quantiles: dict[str, float] | None = None
    val_returns_positive_fraction: float | None = None
    val_returns_zero_fraction: float | None = None
    val_returns_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "plan_id": self.plan_id,
            "metric": self.metric_value,
            "metric_name": self.metric_name,
            "manifest_path": str(self.manifest_path),
            "logits_path": str(self.logits_path) if self.logits_path else None,
            "z_val_mean": self.z_val_mean,
            "z_val_std": self.z_val_std,
            "val_returns_mean": self.val_returns_mean,
            "val_returns_std": self.val_returns_std,
            "fallback_join": self.fallback_join,
            "mismatch_count": self.mismatch_count,
            "missing_count": self.missing_count,
            "dataset_seed": self.dataset_seed,
            "worker_seed": self.worker_seed,
            "worker_curriculum_enabled": self.worker_curriculum_enabled,
            "worker_train_fraction": self.worker_train_fraction,
            "worker_amp_enabled": self.worker_amp_enabled,
            "worker_ensemble_enabled": self.worker_ensemble_enabled,
            "worker_ensemble_members_configured": self.worker_ensemble_members_configured,
            "worker_skipped_rows": self.worker_skipped_rows,
            "worker_skipped_sequences": self.worker_skipped_sequences,
            "worker_skipped_shards": self.worker_skipped_shards,
            "z_val_quantiles": dict(self.z_val_quantiles) if self.z_val_quantiles is not None else None,
            "val_returns_quantiles": (
                dict(self.val_returns_quantiles) if self.val_returns_quantiles is not None else None
            ),
            "val_returns_positive_fraction": self.val_returns_positive_fraction,
            "val_returns_zero_fraction": self.val_returns_zero_fraction,
            "val_returns_count": self.val_returns_count,
        }


def _resolve_artifact_path(raw_path: str | None, manifest_dir: Path) -> Path | None:
    if raw_path is None or not raw_path.strip():
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("/app/"):
        host_relative = Path(normalized.removeprefix("/app/"))
        if host_relative.exists():
            return host_relative
    resolved = manifest_dir / Path(normalized).name
    if resolved.exists():
        return resolved
    return None


def _array_stats(array: np.ndarray | None) -> tuple[float | None, float | None]:
    if array is None or array.size == 0:
        return None, None
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return None, None
    return float(np.mean(valid)), float(np.std(valid))


def _quantiles(array: np.ndarray | None) -> dict[str, float] | None:
    if array is None or array.size == 0:
        return None
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return None
    percentiles = np.percentile(valid, (5, 50, 95))
    return {
        "p05": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p95": float(percentiles[2]),
    }


def _counting_metrics(
    array: np.ndarray | None,
) -> tuple[float | None, float | None, int | None]:
    if array is None or array.size == 0:
        return None, None, None
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return None, None, int(array.size)
    count = valid.size
    positive_fraction = float(np.mean(valid > 0.0))
    zero_fraction = float(np.mean(np.isclose(valid, 0.0)))
    return positive_fraction, zero_fraction, int(count)


def _load_logits_stats(
    logits_path: Path | None,
) -> tuple[
    float | None,
    float | None,
    dict[str, float] | None,
    float | None,
    float | None,
    dict[str, float] | None,
    float | None,
    float | None,
    int | None,
]:
    if logits_path is None or not logits_path.exists():
        return (None, None, None, None, None, None, None, None, None)
    with np.load(logits_path, allow_pickle=False) as npz:
        z_val = npz["z_val"] if "z_val" in npz.files else None
        val_returns = npz["val_returns"] if "val_returns" in npz.files else None
        z_mean, z_std = _array_stats(z_val if isinstance(z_val, np.ndarray) else None)
        ret_mean, ret_std = _array_stats(val_returns if isinstance(val_returns, np.ndarray) else None)
        z_quantiles = _quantiles(z_val if isinstance(z_val, np.ndarray) else None)
        val_quantiles = _quantiles(val_returns if isinstance(val_returns, np.ndarray) else None)
        val_positive_fraction, val_zero_fraction, val_count = _counting_metrics(
            val_returns if isinstance(val_returns, np.ndarray) else None
        )
    return (
        z_mean,
        z_std,
        z_quantiles,
        ret_mean,
        ret_std,
        val_quantiles,
        val_positive_fraction,
        val_zero_fraction,
        val_count,
    )


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        string_value = str(value).strip().lower()
        if string_value in {"1", "true", "yes"}:
            return True
        if string_value in {"0", "false", "no"}:
            return False
    return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_stats(manifest_path: Path, *, metric: str) -> CohortStats:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    cohort_block = data.get("cohort_run", {})
    if not isinstance(cohort_block, Mapping):
        raise ValueError(f"cohort_run block missing or malformed in {manifest_path}")
    metrics_block = cohort_block.get("metrics", {})
    if not isinstance(metrics_block, Mapping):
        metrics_block = {}
    metric_key = _resolve_metric_name(metric)
    telemetry = cohort_block.get("telemetry", {})
    caps_block = telemetry.get("caps", {}) if isinstance(telemetry, Mapping) else {}
    validation_returns = telemetry.get("validation_returns", {}) if isinstance(telemetry, Mapping) else {}
    artifact_paths = cohort_block.get("artifact_paths", {})
    if not isinstance(artifact_paths, Mapping):
        artifact_paths = {}
    logits_path = _resolve_artifact_path(
        str(artifact_paths.get("logits")) if isinstance(artifact_paths, Mapping) else None,
        manifest_path.parent,
    )
    (
        z_mean,
        z_std,
        z_quantiles,
        ret_mean,
        ret_std,
        val_quantiles,
        val_positive_fraction,
        val_zero_fraction,
        val_count,
    ) = _load_logits_stats(logits_path)
    metric_value = _resolve_metric_value(
        metric_key,
        metrics_block=metrics_block,
        z_val_mean=z_mean,
        z_val_std=z_std,
        val_returns_mean=ret_mean,
        val_returns_std=ret_std,
    )
    return CohortStats(
        plan_id=str(cohort_block.get("plan_id", manifest_path.stem.replace("_manifest", ""))),
        metric_value=metric_value,
        metric_name=metric_key,
        manifest_path=manifest_path,
        logits_path=logits_path,
        z_val_mean=z_mean,
        z_val_std=z_std,
        val_returns_mean=ret_mean,
        val_returns_std=ret_std,
        fallback_join=_coerce_bool(validation_returns.get("fallback_join"))
        if isinstance(validation_returns, Mapping)
        else None,
        mismatch_count=int(validation_returns["mismatch_count"])
        if isinstance(validation_returns, Mapping) and "mismatch_count" in validation_returns
        else None,
        missing_count=int(validation_returns["missing_count"])
        if isinstance(validation_returns, Mapping) and "missing_count" in validation_returns
        else None,
        dataset_seed=_coerce_int(caps_block.get("dataset_seed")) if isinstance(caps_block, Mapping) else None,
        worker_seed=_coerce_int(caps_block.get("worker_seed")) if isinstance(caps_block, Mapping) else None,
        worker_curriculum_enabled=_coerce_bool(caps_block.get("worker_curriculum_enabled"))
        if isinstance(caps_block, Mapping)
        else None,
        worker_train_fraction=_coerce_float(caps_block.get("worker_train_fraction"))
        if isinstance(caps_block, Mapping)
        else None,
        worker_amp_enabled=_coerce_bool(caps_block.get("worker_amp_enabled"))
        if isinstance(caps_block, Mapping)
        else None,
        worker_ensemble_enabled=_coerce_bool(caps_block.get("worker_ensemble_enabled"))
        if isinstance(caps_block, Mapping)
        else None,
        worker_ensemble_members_configured=_coerce_int(caps_block.get("worker_ensemble_members_configured"))
        if isinstance(caps_block, Mapping)
        else None,
        worker_skipped_rows=_coerce_int(caps_block.get("worker_skipped_rows"))
        if isinstance(caps_block, Mapping)
        else None,
        worker_skipped_sequences=_coerce_int(caps_block.get("worker_skipped_sequences"))
        if isinstance(caps_block, Mapping)
        else None,
        worker_skipped_shards=_coerce_int(caps_block.get("worker_skipped_shards"))
        if isinstance(caps_block, Mapping)
        else None,
        z_val_quantiles=z_quantiles,
        val_returns_quantiles=val_quantiles,
        val_returns_positive_fraction=val_positive_fraction,
        val_returns_zero_fraction=val_zero_fraction,
        val_returns_count=val_count,
    )


def compare_cohorts(
    manifest_dir: Path,
    *,
    metric: str = DEFAULT_METRIC,
    count: int = DEFAULT_COUNT,
) -> dict[str, Any]:
    manifests = sorted(manifest_dir.glob("*_manifest.json"))
    stats: list[CohortStats] = [collect_stats(path, metric=metric) for path in manifests]
    filtered = [entry for entry in stats if entry.metric_value is not None]
    if not filtered:
        raise ValueError(f"No manifest metrics found for metric={metric!r} in {manifest_dir}")
    limit = max(1, min(count, len(filtered)))
    sorted_stats = sorted(filtered, key=_metric_value)
    bottom_stats = sorted_stats[:limit]
    top_stats = list(reversed(sorted_stats[-limit:]))
    bottom = [entry.as_dict() for entry in bottom_stats]
    top = [entry.as_dict() for entry in top_stats]
    summary = {
        "top": _summarize_group(top_stats),
        "bottom": _summarize_group(bottom_stats),
    }
    summary["delta"] = _diff_summary(summary["top"], summary["bottom"])
    return {"metric": metric, "top": top, "bottom": bottom, "summary": summary}


def _resolve_metric_name(metric: str) -> str:
    normalized = metric.strip()
    lookup = normalized.lower()
    return _METRIC_ALIASES.get(lookup, normalized)


def _resolve_metric_value(
    metric_key: str,
    *,
    metrics_block: Mapping[str, Any],
    z_val_mean: float | None,
    z_val_std: float | None,
    val_returns_mean: float | None,
    val_returns_std: float | None,
) -> float | None:
    canonical = metric_key.lower()
    if canonical in _SPECIAL_METRICS:
        if canonical == "z_val_std":
            return z_val_std
        if canonical == "z_val_mean":
            return z_val_mean
        if canonical == "val_returns_std":
            return val_returns_std
        if canonical == "val_returns_mean":
            return val_returns_mean
    raw_value = metrics_block.get(metric_key)
    return _coerce_float(raw_value)


def _metric_value(entry: CohortStats) -> float:
    value = entry.metric_value
    if value is None:
        raise ValueError("metric_value must be populated before ordering cohorts")
    return float(value)


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if isinstance(value, (int, float))]
    return float(np.mean(finite)) if finite else None


def _fraction(values: Sequence[bool | None]) -> float | None:
    filtered = [value for value in values if isinstance(value, bool)]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _merge_quantiles(mappings: Sequence[dict[str, float] | None]) -> dict[str, float] | None:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for mapping in mappings:
        if not mapping:
            continue
        for key, value in mapping.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    if not totals:
        return None
    return {key: totals[key] / counts[key] for key in sorted(totals)}


def _summarize_group(entries: Sequence[CohortStats]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(entries),
    }
    summary["metric_mean"] = _mean([entry.metric_value for entry in entries])
    summary["z_val_std_mean"] = _mean([entry.z_val_std for entry in entries])
    summary["val_returns_std_mean"] = _mean([entry.val_returns_std for entry in entries])
    summary["val_returns_positive_fraction_mean"] = _mean(
        [entry.val_returns_positive_fraction for entry in entries]
    )
    summary["val_returns_zero_fraction_mean"] = _mean(
        [entry.val_returns_zero_fraction for entry in entries]
    )
    summary["val_returns_count_mean"] = _mean([entry.val_returns_count for entry in entries])
    summary["amp_enabled_fraction"] = _fraction([entry.worker_amp_enabled for entry in entries])
    summary["curriculum_enabled_fraction"] = _fraction(
        [entry.worker_curriculum_enabled for entry in entries]
    )
    summary["fallback_join_fraction"] = _fraction([entry.fallback_join for entry in entries])
    summary["train_fraction_mean"] = _mean([entry.worker_train_fraction for entry in entries])
    summary["dataset_seeds"] = sorted(
        {entry.dataset_seed for entry in entries if entry.dataset_seed is not None}
    )
    summary["worker_seeds"] = sorted(
        {entry.worker_seed for entry in entries if entry.worker_seed is not None}
    )
    summary["z_val_quantiles_mean"] = _merge_quantiles([entry.z_val_quantiles for entry in entries])
    summary["val_returns_quantiles_mean"] = _merge_quantiles(
        [entry.val_returns_quantiles for entry in entries]
    )
    return summary


def _diff_summary(top_summary: dict[str, Any], bottom_summary: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, top_value in top_summary.items():
        bottom_value = bottom_summary.get(key)
        if isinstance(top_value, (int, float)) and isinstance(bottom_value, (int, float)):
            delta[key] = float(top_value) - float(bottom_value)
        elif isinstance(top_value, dict) and isinstance(bottom_value, dict):
            nested: dict[str, float] = {}
            for nested_key, nested_value in top_value.items():
                if nested_key in bottom_value:
                    nested[nested_key] = float(nested_value) - float(bottom_value[nested_key])
            if nested:
                delta[key] = nested
    return delta


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare streaming cohorts using saved manifests.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing streaming manifest JSON files.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        help=f"Metric key used to rank cohorts (default: {DEFAULT_METRIC}).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of cohorts to include per side (default: {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = compare_cohorts(
        args.manifest_dir,
        metric=args.metric,
        count=max(1, args.count),
    )
    indent = 2 if bool(getattr(args, "pretty", False)) else None
    print(json.dumps(result, indent=indent, sort_keys=False))
    return 0


__all__ = ["CohortStats", "collect_stats", "compare_cohorts", "main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
