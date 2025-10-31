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
    return float(np.mean(array)), float(np.std(array))


def _load_logits_stats(logits_path: Path | None) -> tuple[float | None, float | None, float | None, float | None]:
    if logits_path is None or not logits_path.exists():
        return None, None, None, None
    with np.load(logits_path, allow_pickle=False) as npz:
        z_val = npz["z_val"] if "z_val" in npz.files else None
        val_returns = npz["val_returns"] if "val_returns" in npz.files else None
        z_mean, z_std = _array_stats(z_val if isinstance(z_val, np.ndarray) else None)
        ret_mean, ret_std = _array_stats(val_returns if isinstance(val_returns, np.ndarray) else None)
    return z_mean, z_std, ret_mean, ret_std


def collect_stats(manifest_path: Path, *, metric: str) -> CohortStats:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    cohort_block = data.get("cohort_run", {})
    if not isinstance(cohort_block, Mapping):
        raise ValueError(f"cohort_run block missing or malformed in {manifest_path}")
    metrics_block = cohort_block.get("metrics", {})
    if not isinstance(metrics_block, Mapping):
        metrics_block = {}
    metric_value = float(metrics_block[metric]) if metric in metrics_block else None
    telemetry = cohort_block.get("telemetry", {})
    validation_returns = telemetry.get("validation_returns", {}) if isinstance(telemetry, Mapping) else {}
    artifact_paths = cohort_block.get("artifact_paths", {})
    if not isinstance(artifact_paths, Mapping):
        artifact_paths = {}
    logits_path = _resolve_artifact_path(
        str(artifact_paths.get("logits")) if isinstance(artifact_paths, Mapping) else None,
        manifest_path.parent,
    )
    z_mean, z_std, ret_mean, ret_std = _load_logits_stats(logits_path)
    return CohortStats(
        plan_id=str(cohort_block.get("plan_id", manifest_path.stem.replace("_manifest", ""))),
        metric_value=metric_value,
        metric_name=metric,
        manifest_path=manifest_path,
        logits_path=logits_path,
        z_val_mean=z_mean,
        z_val_std=z_std,
        val_returns_mean=ret_mean,
        val_returns_std=ret_std,
        fallback_join=bool(validation_returns.get("fallback_join"))
        if isinstance(validation_returns, Mapping) and "fallback_join" in validation_returns
        else None,
        mismatch_count=int(validation_returns["mismatch_count"])
        if isinstance(validation_returns, Mapping) and "mismatch_count" in validation_returns
        else None,
        missing_count=int(validation_returns["missing_count"])
        if isinstance(validation_returns, Mapping) and "missing_count" in validation_returns
        else None,
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
    bottom = [entry.as_dict() for entry in sorted_stats[:limit]]
    top = [entry.as_dict() for entry in reversed(sorted_stats[-limit:])]
    return {"metric": metric, "top": top, "bottom": bottom}


def _metric_value(entry: CohortStats) -> float:
    value = entry.metric_value
    if value is None:
        raise ValueError("metric_value must be populated before ordering cohorts")
    return float(value)


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
