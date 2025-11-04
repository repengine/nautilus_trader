#!/usr/bin/env python3
"""Audit streaming runner state snapshots for stale plans and manifests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class StreamingStateAudit:
    """Summarize discrepancies between runner state and persisted manifests."""

    missing_results: tuple[str, ...]
    missing_plans: tuple[str, ...]
    missing_manifests: tuple[str, ...]
    manifest_only: tuple[str, ...]
    total_plans: int
    total_results: int
    stream_cursor: str | None
    manifest_only_details: dict[str, Any] = field(default_factory=dict)
    missing_manifest_details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the audit result."""
        return {
            "missing_results": list(self.missing_results),
            "missing_plans": list(self.missing_plans),
            "missing_manifests": list(self.missing_manifests),
            "manifest_only": list(self.manifest_only),
            "total_plans": self.total_plans,
            "total_results": self.total_results,
            "stream_cursor": self.stream_cursor,
            "manifest_only_details": self.manifest_only_details,
            "missing_manifest_details": self.missing_manifest_details,
        }


def audit_streaming_state(state_path: Path, *, manifest_dir: Path | None = None) -> StreamingStateAudit:
    """Load the runner state snapshot and report inconsistencies."""
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    plans_block = payload.get("plans", {})
    results_block = payload.get("results", {})
    stream_cursor = payload.get("stream_cursor")

    plan_ids = _extract_keys(plans_block)
    result_ids = _extract_keys(results_block)

    missing_results = tuple(sorted(plan_ids - result_ids))
    missing_plans = tuple(sorted(result_ids - plan_ids))

    manifest_ids: set[str] = set()
    manifest_lookup: dict[str, Path] = {}
    if manifest_dir is not None:
        for path in manifest_dir.glob("*_manifest.json"):
            if not path.name.endswith("_manifest.json"):
                continue
            plan_id = path.name[: -len("_manifest.json")]
            manifest_ids.add(plan_id)
            manifest_lookup[plan_id] = path
    missing_manifests = tuple(sorted((plan_ids | result_ids) - manifest_ids)) if manifest_ids else tuple()
    manifest_only = tuple(sorted(manifest_ids - (plan_ids | result_ids))) if manifest_ids else tuple()
    manifest_only_details = _load_manifest_details(manifest_lookup, manifest_only)
    missing_manifest_details = _load_state_details(plans_block, results_block, missing_manifests)

    return StreamingStateAudit(
        missing_results=missing_results,
        missing_plans=missing_plans,
        missing_manifests=missing_manifests,
        manifest_only=manifest_only,
        total_plans=len(plan_ids),
        total_results=len(result_ids),
        stream_cursor=str(stream_cursor) if stream_cursor is not None else None,
        manifest_only_details=manifest_only_details,
        missing_manifest_details=missing_manifest_details,
    )


def _extract_keys(block: Any) -> set[str]:
    if isinstance(block, dict):
        return {str(key) for key in block}
    return set()


def _load_manifest_details(
    manifest_lookup: Mapping[str, Path],
    manifest_only: Sequence[str],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for plan_id in manifest_only:
        path = manifest_lookup.get(plan_id)
        if path is None or not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cohort = payload.get("cohort_run", {})
        if not isinstance(cohort, Mapping):
            continue
        metrics = cohort.get("metrics", {})
        telemetry = cohort.get("telemetry", {})
        caps = telemetry.get("caps", {}) if isinstance(telemetry, Mapping) else {}
        validation_returns = (
            telemetry.get("validation_returns", {}) if isinstance(telemetry, Mapping) else {}
        )
        detail_payload: dict[str, Any] = {
            "dataset_id": cohort.get("dataset_id"),
            "completed_at": cohort.get("completed_at"),
        }
        if isinstance(metrics, Mapping):
            detail_payload["roc_auc"] = metrics.get("roc_auc")
            detail_payload["pr_auc_multiple"] = metrics.get("pr_auc_multiple")
            detail_payload["economic_slippage_adjusted_sharpe"] = metrics.get(
                "economic_slippage_adjusted_sharpe",
            )
        if isinstance(caps, Mapping):
            detail_payload["dataset_seed"] = caps.get("dataset_seed")
            detail_payload["worker_seed"] = caps.get("worker_seed")
            detail_payload["worker_curriculum_enabled"] = caps.get("worker_curriculum_enabled")
            detail_payload["worker_amp_enabled"] = caps.get("worker_amp_enabled")
        if isinstance(validation_returns, Mapping):
            detail_payload["validation_returns_fallback_join"] = validation_returns.get("fallback_join")
            detail_payload["validation_returns_missing_count"] = validation_returns.get("missing_count")
            detail_payload["validation_returns_mismatch_count"] = validation_returns.get("mismatch_count")
        details[plan_id] = detail_payload
    return details


def _load_state_details(
    plans_block: Any,
    results_block: Any,
    missing_manifests: Sequence[str],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    plans_mapping = plans_block if isinstance(plans_block, Mapping) else {}
    results_mapping = results_block if isinstance(results_block, Mapping) else {}
    for plan_id in missing_manifests:
        record: Any = results_mapping.get(plan_id)
        if not isinstance(record, Mapping):
            record = plans_mapping.get(plan_id, {})
        if not isinstance(record, Mapping):
            continue
        metrics = record.get("metrics", {})
        telemetry = record.get("telemetry", {})
        detail_payload: dict[str, Any] = {
            "status": record.get("status"),
            "completed_at": record.get("completed_at"),
            "dataset_id": record.get("dataset_id"),
        }
        if isinstance(metrics, Mapping):
            detail_payload["roc_auc"] = metrics.get("roc_auc")
            detail_payload["pr_auc_multiple"] = metrics.get("pr_auc_multiple")
            detail_payload["economic_slippage_adjusted_sharpe"] = metrics.get(
                "economic_slippage_adjusted_sharpe",
            )
        if isinstance(telemetry, Mapping):
            validation_returns = telemetry.get("validation_returns", {})
            if isinstance(validation_returns, Mapping):
                detail_payload["validation_returns_fallback_join"] = validation_returns.get("fallback_join")
                detail_payload["validation_returns_missing_count"] = validation_returns.get("missing_count")
                detail_payload["validation_returns_mismatch_count"] = validation_returns.get("mismatch_count")
        details[plan_id] = detail_payload
    return details


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit streaming runner state snapshots.")
    parser.add_argument(
        "--state-path",
        type=Path,
        required=True,
        help="Path to streaming_training_state_snapshot.json.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Optional manifest directory for cross-validation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when inconsistencies are detected.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    audit = audit_streaming_state(args.state_path, manifest_dir=args.manifest_dir)
    print(json.dumps(audit.as_dict(), indent=2, sort_keys=True))
    has_issues = bool(audit.missing_results or audit.missing_plans or audit.missing_manifests)
    if args.strict and has_issues:
        return 1
    return 0


__all__ = ["StreamingStateAudit", "audit_streaming_state", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
