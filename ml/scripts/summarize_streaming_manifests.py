#!/usr/bin/env python3
r"""
Summarize streaming training manifests.

This utility scans a directory of runner manifests and prints a compact summary
containing validation metrics, resource telemetry, and cohort metadata. It is
designed for ad-hoc analysis and lightweight CI hooks during continuous
streaming iterations.

Example:
    $ python -m ml.scripts.summarize_streaming_manifests \
        --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, SupportsFloat


@dataclass(slots=True)
class ManifestSummary:
    """Structured summary of a single streaming manifest."""

    plan_id: str
    dataset_id: str
    completed_at: datetime
    roc_auc: float | None
    pr_auc: float | None
    pr_auc_multiple: float | None
    log_loss: float | None
    brier_score: float | None
    peak_gpu_mb: float | None
    train_rows: int | None
    validation_rows: int | None


def _load_manifest(path: Path) -> ManifestSummary | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cohort = payload.get("cohort_run", {})
    metrics = cohort.get("metrics", {})
    telemetry = cohort.get("telemetry", {})
    selected_rows = telemetry.get("selected_rows", {})
    resources = telemetry.get("resources", {})

    completed_raw = cohort.get("completed_at")
    completed_at = _parse_datetime(completed_raw) if isinstance(completed_raw, str) else None
    if completed_at is None:
        return None

    return ManifestSummary(
        plan_id=str(cohort.get("plan_id", "UNKNOWN")),
        dataset_id=str(cohort.get("dataset_id", "UNKNOWN")),
        completed_at=completed_at,
        roc_auc=_coerce_float(metrics.get("roc_auc")),
        pr_auc=_coerce_float(metrics.get("pr_auc")),
        pr_auc_multiple=_coerce_float(metrics.get("pr_auc_multiple")),
        log_loss=_coerce_float(metrics.get("log_loss")),
        brier_score=_coerce_float(metrics.get("brier_score")),
        peak_gpu_mb=_coerce_float(resources.get("max_gpu_memory_mb")),
        train_rows=_coerce_int(selected_rows.get("train")),
        validation_rows=_coerce_int(selected_rows.get("validation")),
    )


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(raw: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_value(value: SupportsFloat | None, *, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{precision}f}"


def summarize_manifests(manifest_dir: Path, *, limit: int | None = None) -> list[ManifestSummary]:
    """Collect and sort manifest summaries by completion time (descending)."""
    manifests = sorted(manifest_dir.glob("*_manifest.json"))
    summaries: list[ManifestSummary] = []
    for path in manifests:
        summary = _load_manifest(path)
        if summary is None:
            continue
        summaries.append(summary)
    summaries.sort(key=lambda item: item.completed_at, reverse=True)
    if limit is not None and limit > 0:
        return summaries[:limit]
    return summaries


def _print_markdown(summaries: list[ManifestSummary]) -> None:
    if not summaries:
        print("No manifests found", file=sys.stdout)
        return
    print("| Plan | Dataset | Completed | ROC-AUC | PR-AUC | PR multiple | LogLoss | Brier | Peak GPU (MB) | Train Rows | Val Rows |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in summaries:
        print(
            f"| {item.plan_id} | {item.dataset_id} | {item.completed_at.isoformat()} | "
            f"{_format_value(item.roc_auc)} | {_format_value(item.pr_auc)} | "
            f"{_format_value(item.pr_auc_multiple)} | {_format_value(item.log_loss)} | "
            f"{_format_value(item.brier_score)} | {_format_value(item.peak_gpu_mb, precision=1)} | "
            f"{_format_value(item.train_rows)} | {_format_value(item.validation_rows)} |"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize streaming training manifests.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("ml_out/tft_streaming_artifacts/full_tft_95"),
        help="Directory containing streaming manifest JSON files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of manifests to display (most recent first).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summaries = summarize_manifests(args.manifest_dir, limit=args.limit)
    _print_markdown(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
