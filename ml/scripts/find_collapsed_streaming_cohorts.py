#!/usr/bin/env python3
"""Surface streaming cohorts with collapsed logits or weak economic metrics."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class CohortInspection:
    """Inspection payload for a single streaming cohort."""

    plan_id: str
    manifest_path: Path
    logits_path: Path | None
    z_val_std: float | None
    sharpe: float | None
    hit_rate: float | None
    worker_curriculum_enabled: bool | None
    worker_amp_enabled: bool | None
    is_collapsed: bool


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - guard for malformed files
        raise ValueError(f"failed to parse manifest {path}: {exc}") from exc
    cohort = payload.get("cohort_run")
    if not isinstance(cohort, dict):
        raise ValueError(f"manifest {path} missing cohort_run block")
    return cohort


def _resolve_logits_path(manifest_path: Path, cohort: dict[str, Any]) -> Path | None:
    artifacts = cohort.get("artifact_paths", {})
    if not isinstance(artifacts, dict):
        return None
    raw_path = artifacts.get("logits")
    if raw_path is None:
        return None
    candidate = Path(str(raw_path))
    if candidate.exists():
        return candidate
    normalized = str(raw_path).replace("\\", "/")
    if normalized.startswith("/app/"):
        host_relative = Path(normalized.removeprefix("/app/"))
        if host_relative.exists():
            return host_relative
    resolved = manifest_path.parent / Path(normalized).name
    return resolved if resolved.exists() else None


def _compute_z_std(logits_path: Path | None) -> float | None:
    if logits_path is None or not logits_path.exists():
        return None
    with np.load(logits_path, allow_pickle=False) as npz:
        if "z_val" not in npz.files:
            return None
        z_val = np.asarray(npz["z_val"], dtype=np.float64).reshape(-1)
        if z_val.size == 0:
            return None
        finite = z_val[np.isfinite(z_val)]
        if finite.size == 0:
            return None
        return float(np.std(finite))


def _get_metric(cohort: dict[str, Any], key: str) -> float | None:
    metrics = cohort.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    raw = metrics.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _get_cap(cohort: dict[str, Any], key: str) -> Any:
    telemetry = cohort.get("telemetry", {})
    if not isinstance(telemetry, dict):
        return None
    caps = telemetry.get("caps", {})
    if not isinstance(caps, dict):
        return None
    return caps.get(key)


def inspect_cohort(manifest_path: Path, *, z_threshold: float, sharpe_threshold: float) -> CohortInspection:
    """Inspect a single manifest for collapsed metrics."""
    cohort = _load_manifest(manifest_path)
    plan_id = str(cohort.get("plan_id", manifest_path.stem.replace("_manifest", "")))
    logits_path = _resolve_logits_path(manifest_path, cohort)
    z_std = _compute_z_std(logits_path)
    sharpe = _get_metric(cohort, "economic_slippage_adjusted_sharpe")
    hit_rate = _get_metric(cohort, "economic_hit_rate")
    curriculum_enabled = _get_cap(cohort, "worker_curriculum_enabled")
    amp_enabled = _get_cap(cohort, "worker_amp_enabled")
    collapsed = _is_collapsed(z_std, sharpe, z_threshold=z_threshold, sharpe_threshold=sharpe_threshold)
    return CohortInspection(
        plan_id=plan_id,
        manifest_path=manifest_path,
        logits_path=logits_path,
        z_val_std=z_std,
        sharpe=sharpe,
        hit_rate=hit_rate,
        worker_curriculum_enabled=bool(curriculum_enabled) if isinstance(curriculum_enabled, bool) else None,
        worker_amp_enabled=bool(amp_enabled) if isinstance(amp_enabled, bool) else None,
        is_collapsed=collapsed,
    )


def _is_collapsed(z_std: float | None, sharpe: float | None, *, z_threshold: float = 0.02, sharpe_threshold: float = 0.0) -> bool:
    if z_std is not None and z_std < z_threshold:
        return True
    if sharpe is not None and sharpe < sharpe_threshold:
        return True
    return False


def _collect_inspections(manifest_dir: Path, *, z_threshold: float, sharpe_threshold: float) -> list[dict[str, Any]]:
    inspections: list[dict[str, Any]] = []
    for manifest_path in sorted(manifest_dir.glob("*_manifest.json")):
        try:
            inspection = inspect_cohort(manifest_path, z_threshold=z_threshold, sharpe_threshold=sharpe_threshold)
        except ValueError as exc:
            logger.warning("skipping manifest", extra={"manifest": str(manifest_path), "error": str(exc)})
            continue
        record = {
            "plan_id": inspection.plan_id,
            "manifest_path": str(inspection.manifest_path),
            "logits_path": str(inspection.logits_path) if inspection.logits_path else None,
            "z_val_std": inspection.z_val_std,
            "sharpe": inspection.sharpe,
            "hit_rate": inspection.hit_rate,
            "worker_curriculum_enabled": inspection.worker_curriculum_enabled,
            "worker_amp_enabled": inspection.worker_amp_enabled,
            "collapsed": inspection.is_collapsed,
        }
        inspections.append(record)
    return inspections


def run_inspection(manifest_dir: Path, *, z_threshold: float, sharpe_threshold: float) -> dict[str, Any]:
    """Return collapsed cohort summary for the provided manifests."""
    inspections = _collect_inspections(manifest_dir, z_threshold=z_threshold, sharpe_threshold=sharpe_threshold)
    collapsed = [record for record in inspections if record["collapsed"]]
    return {
        "manifest_dir": str(manifest_dir),
        "z_val_std_threshold": z_threshold,
        "sharpe_threshold": sharpe_threshold,
        "collapsed_count": len(collapsed),
        "total_count": len(inspections),
        "collapsed": collapsed,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report streaming cohorts with collapsed logits or weak economics.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing streaming manifest JSON files.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=0.02,
        help="Std-dev threshold for z_val below which a cohort is considered collapsed.",
    )
    parser.add_argument(
        "--sharpe-threshold",
        type=float,
        default=0.0,
        help="Sharpe threshold below which a cohort is considered collapsed.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print output JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_inspection(
        Path(args.manifest_dir),
        z_threshold=float(args.z_threshold),
        sharpe_threshold=float(args.sharpe_threshold),
    )
    output = json.dumps(result, indent=2 if args.pretty else None, sort_keys=bool(args.pretty))
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
