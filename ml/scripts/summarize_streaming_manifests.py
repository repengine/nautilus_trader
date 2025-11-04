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
    temperature_calibration_log_loss: float | None
    temperature_calibration_log_loss_delta: float | None
    platt_calibration_log_loss: float | None
    platt_calibration_log_loss_delta: float | None
    isotonic_calibration_log_loss: float | None
    isotonic_calibration_log_loss_delta: float | None
    ensemble_members_misaligned: float | None
    economic_slippage_adjusted_sharpe: float | None
    economic_hit_rate: float | None
    economic_turnover: float | None
    economic_max_drawdown: float | None
    stability_ks_statistic: float | None
    validation_instrument_rows_total: dict[str, int] | None = None
    validation_instrument_rows_selected: dict[str, int] | None = None
    validation_instrument_sequences_total: dict[str, int] | None = None
    validation_instrument_sequences_selected: dict[str, int] | None = None
    validation_failure_reason: str | None = None
    validation_failure_details: dict[str, Any] | None = None
    validation_returns_fallback_join: bool | None = None
    validation_returns_mismatch_count: int | None = None
    validation_returns_missing_count: int | None = None
    worker_skipped_rows: int | None = None
    worker_skipped_sequences: int | None = None
    worker_skipped_shards: int | None = None


def _load_manifest(path: Path) -> ManifestSummary | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cohort = payload.get("cohort_run", {})
    metrics = cohort.get("metrics", {})
    telemetry = cohort.get("telemetry", {})
    validation_returns = telemetry.get("validation_returns", {}) if isinstance(telemetry, dict) else {}
    selected_rows = telemetry.get("selected_rows", {})
    resources = telemetry.get("resources", {})
    caps = telemetry.get("caps", {})

    completed_raw = cohort.get("completed_at")
    completed_at = _parse_datetime(completed_raw) if isinstance(completed_raw, str) else None
    if completed_at is None:
        return None

    temperature_ll = _coerce_float(metrics.get("temperature_calibration_log_loss"))
    platt_ll = _coerce_float(metrics.get("platt_calibration_log_loss"))
    isotonic_ll = _coerce_float(metrics.get("isotonic_calibration_log_loss"))
    log_loss_value = metrics.get("log_loss")

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
        temperature_calibration_log_loss=temperature_ll,
        temperature_calibration_log_loss_delta=_delta(temperature_ll, log_loss_value),
        platt_calibration_log_loss=platt_ll,
        platt_calibration_log_loss_delta=_delta(platt_ll, log_loss_value),
        isotonic_calibration_log_loss=isotonic_ll,
        isotonic_calibration_log_loss_delta=_delta(isotonic_ll, log_loss_value),
        ensemble_members_misaligned=_coerce_float(metrics.get("ensemble_members_misaligned")),
        economic_slippage_adjusted_sharpe=_coerce_float(
            metrics.get("economic_slippage_adjusted_sharpe"),
        ),
        economic_hit_rate=_coerce_float(metrics.get("economic_hit_rate")),
        economic_turnover=_coerce_float(metrics.get("economic_turnover")),
        economic_max_drawdown=_coerce_float(metrics.get("economic_max_drawdown")),
        stability_ks_statistic=_coerce_float(metrics.get("stability_ks_statistic")),
        validation_instrument_rows_total=_coerce_int_mapping(
            caps.get("worker_validation_instrument_rows_total"),
        ),
        validation_instrument_rows_selected=_coerce_int_mapping(
            caps.get("worker_validation_instrument_rows_selected"),
        ),
        validation_instrument_sequences_total=_coerce_int_mapping(
            caps.get("worker_validation_instrument_sequences_total"),
        ),
        validation_instrument_sequences_selected=_coerce_int_mapping(
            caps.get("worker_validation_instrument_sequences_selected"),
        ),
        validation_failure_reason=_coerce_str(caps.get("validation_failure_reason")),
        validation_failure_details=_coerce_dict(caps.get("validation_failure_details")),
        validation_returns_fallback_join=_coerce_bool(validation_returns.get("fallback_join"))
        if isinstance(validation_returns, dict)
        else None,
        validation_returns_mismatch_count=_coerce_int(validation_returns.get("mismatch_count"))
        if isinstance(validation_returns, dict)
        else None,
        validation_returns_missing_count=_coerce_int(validation_returns.get("missing_count"))
        if isinstance(validation_returns, dict)
        else None,
        worker_skipped_rows=_coerce_int(caps.get("worker_skipped_rows"))
        if isinstance(caps, dict)
        else None,
        worker_skipped_sequences=_coerce_int(caps.get("worker_skipped_sequences"))
        if isinstance(caps, dict)
        else None,
        worker_skipped_shards=_coerce_int(caps.get("worker_skipped_shards"))
        if isinstance(caps, dict)
        else None,
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


def _coerce_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _coerce_int_mapping(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, int] = {}
    for key, raw in value.items():
        coerced = _coerce_int(raw)
        if coerced is not None:
            result[str(key)] = coerced
    return result or None


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): val for key, val in value.items()}

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

def _delta(calibrated: Any, base: Any) -> float | None:
    try:
        calibrated_value = float(calibrated)
        base_value = float(base)
    except (TypeError, ValueError):
        return None
    return calibrated_value - base_value


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
    print(
        "| Plan | Dataset | Completed | ROC-AUC | PR-AUC | PR multiple | LogLoss | Temp LL | Temp Δ | "
        "Platt LL | Platt Δ | Iso LL | Iso Δ | Brier | Peak GPU (MB) | Train Rows | Val Rows | "
        "Ensemble x | Sharpeₛ | Hit Rate | Turnover | Drawdown | KS |",
    )
    print(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    for item in summaries:
        print(
            f"| {item.plan_id} | {item.dataset_id} | {item.completed_at.isoformat()} | "
            f"{_format_value(item.roc_auc)} | {_format_value(item.pr_auc)} | "
            f"{_format_value(item.pr_auc_multiple)} | {_format_value(item.log_loss)} | "
            f"{_format_value(item.temperature_calibration_log_loss)} | {_format_value(item.temperature_calibration_log_loss_delta)} | "
            f"{_format_value(item.platt_calibration_log_loss)} | {_format_value(item.platt_calibration_log_loss_delta)} | "
            f"{_format_value(item.isotonic_calibration_log_loss)} | {_format_value(item.isotonic_calibration_log_loss_delta)} | "
            f"{_format_value(item.brier_score)} | {_format_value(item.peak_gpu_mb, precision=1)} | "
            f"{_format_value(item.train_rows)} | {_format_value(item.validation_rows)} | "
            f"{_format_value(item.ensemble_members_misaligned)} | {_format_value(item.economic_slippage_adjusted_sharpe)} | "
            f"{_format_value(item.economic_hit_rate)} | {_format_value(item.economic_turnover)} | "
            f"{_format_value(item.economic_max_drawdown)} | {_format_value(item.stability_ks_statistic)} |"
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
