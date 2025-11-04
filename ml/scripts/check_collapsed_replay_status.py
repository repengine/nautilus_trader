#!/usr/bin/env python3
"""Validate collapsed replay plan completion against manifest telemetry."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class TaskCheckResult:
    """Evaluation result for a single collapsed replay task."""

    plan_id: str
    manifest_path: Path
    logits_path: Path | None
    manifest_exists: bool
    logits_exists: bool | None
    sharpe_baseline: float | None
    sharpe_current: float | None
    fallback_join: bool | None
    validation_failure_reason: str | None
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        """Return 'ok' when no issues were recorded."""
        return "ok" if not self.issues else "failed"

    def as_dict(self) -> dict[str, Any]:
        """Serialize the task evaluation result."""
        return {
            "plan_id": self.plan_id,
            "status": self.status,
            "manifest_path": str(self.manifest_path),
            "logits_path": str(self.logits_path) if self.logits_path is not None else None,
            "manifest_exists": self.manifest_exists,
            "logits_exists": self.logits_exists,
            "sharpe_baseline": self.sharpe_baseline,
            "sharpe_current": self.sharpe_current,
            "fallback_join": self.fallback_join,
            "validation_failure_reason": self.validation_failure_reason,
            "issues": list(self.issues),
        }


@dataclass(slots=True, frozen=True)
class PlanCheckReport:
    """Aggregated report for an entire collapsed replay plan."""

    plan_path: Path
    generated_at: datetime | None
    tasks: tuple[TaskCheckResult, ...]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the overall report."""
        return {
            "plan_path": str(self.plan_path),
            "generated_at": self.generated_at.isoformat() if self.generated_at is not None else None,
            "tasks": [task.as_dict() for task in self.tasks],
        }

    @property
    def has_failures(self) -> bool:
        """Return True when any task failed validation."""
        return any(task.status != "ok" for task in self.tasks)


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        logger.error("plan file missing", extra={"path": str(path)}, exc_info=True)
        raise RuntimeError(f"plan file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        logger.error("failed to decode JSON", extra={"path": str(path)}, exc_info=True)
        raise RuntimeError(f"invalid JSON in {path}: {exc}") from exc


def _parse_datetime_candidate(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            return None
    return None


def _extract_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return None


def _extract_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _resolve_relative(base: Path, raw: Any) -> Path:
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        return (base / candidate).resolve()
    return candidate.resolve()


def _coerce_plan_tasks(plan_payload: Any, *, plan_dir: Path) -> tuple[dict[str, Any], ...]:
    if not isinstance(plan_payload, dict):
        raise RuntimeError("replay plan must be a JSON object")
    tasks = plan_payload.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("replay plan must include a 'tasks' list")
    normalized: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise RuntimeError("each task entry must be a JSON object")
        manifest_path_raw = task.get("manifest_path")
        if manifest_path_raw is None or (isinstance(manifest_path_raw, str) and not manifest_path_raw.strip()):
            raise RuntimeError("task missing manifest_path field")
        manifest_path = _resolve_relative(plan_dir, manifest_path_raw)
        logits_path_raw = task.get("logits_path")
        logits_path = None
        if logits_path_raw not in (None, ""):
            logits_path = _resolve_relative(plan_dir, logits_path_raw)
        normalized.append(
            {
                "plan_id": str(task.get("plan_id") or ""),
                "manifest_path": manifest_path,
                "logits_path": logits_path,
                "sharpe": _extract_float(task.get("sharpe")),
            },
        )
    return tuple(normalized)


def _check_manifest(
    path: Path,
    *,
    expected_plan_id: str,
    sharpe_baseline: float | None,
    require_logits: bool,
    logits_path: Path | None,
) -> TaskCheckResult:
    manifest_exists = path.exists()
    issues: list[str] = []
    sharpe_current: float | None = None
    fallback_join: bool | None = None
    validation_failure_reason: str | None = None
    logits_exists = logits_path.exists() if logits_path is not None else None

    if not manifest_exists:
        issues.append("manifest_missing")
        return TaskCheckResult(
            plan_id=expected_plan_id or path.stem.replace("_manifest", ""),
            manifest_path=path,
            logits_path=logits_path,
            manifest_exists=False,
            logits_exists=logits_exists,
            sharpe_baseline=sharpe_baseline,
            sharpe_current=None,
            fallback_join=None,
            validation_failure_reason=None,
            issues=tuple(issues),
        )

    try:
        manifest_payload = _load_json_file(path)
    except RuntimeError:
        issues.append("manifest_read_error")
        logger.error(
            "failed to load manifest",
            extra={"path": str(path)},
            exc_info=True,
        )
        return TaskCheckResult(
            plan_id=expected_plan_id or path.stem.replace("_manifest", ""),
            manifest_path=path,
            logits_path=logits_path,
            manifest_exists=True,
            logits_exists=logits_exists,
            sharpe_baseline=sharpe_baseline,
            sharpe_current=None,
            fallback_join=None,
            validation_failure_reason=None,
            issues=tuple(issues),
        )
    if not isinstance(manifest_payload, dict):
        issues.append("manifest_root_not_object")
        return TaskCheckResult(
            plan_id=expected_plan_id or path.stem.replace("_manifest", ""),
            manifest_path=path,
            logits_path=logits_path,
            manifest_exists=True,
            logits_exists=logits_exists,
            sharpe_baseline=sharpe_baseline,
            sharpe_current=None,
            fallback_join=None,
            validation_failure_reason=None,
            issues=tuple(issues),
        )

    cohort = manifest_payload.get("cohort_run")
    if not isinstance(cohort, dict):
        issues.append("missing_cohort_run")
        return TaskCheckResult(
            plan_id=expected_plan_id or path.stem.replace("_manifest", ""),
            manifest_path=path,
            logits_path=logits_path,
            manifest_exists=True,
            logits_exists=logits_exists,
            sharpe_baseline=sharpe_baseline,
            sharpe_current=None,
            fallback_join=None,
            validation_failure_reason=None,
            issues=tuple(issues),
        )

    metrics = cohort.get("metrics")
    if isinstance(metrics, dict):
        sharpe_current = _extract_float(metrics.get("economic_slippage_adjusted_sharpe"))
        if sharpe_baseline is not None and sharpe_current is not None and sharpe_current < sharpe_baseline:
            issues.append("economic_sharpe_regressed")
    else:
        issues.append("missing_metrics_block")

    telemetry = cohort.get("telemetry")
    if isinstance(telemetry, dict):
        validation_returns = telemetry.get("validation_returns")
        if isinstance(validation_returns, dict):
            fallback_join = _extract_bool(validation_returns.get("fallback_join"))
            if fallback_join:
                issues.append("validation_returns_fallback_active")
        caps = telemetry.get("caps")
        if isinstance(caps, dict):
            reason = caps.get("validation_failure_reason")
            if isinstance(reason, str) and reason:
                validation_failure_reason = reason
                issues.append("validation_failure_recorded")
    else:
        issues.append("missing_telemetry_block")

    if require_logits and not logits_exists:
        issues.append("logits_missing")

    plan_id = str(cohort.get("plan_id") or expected_plan_id or path.stem.replace("_manifest", ""))

    return TaskCheckResult(
        plan_id=plan_id,
        manifest_path=path,
        logits_path=logits_path,
        manifest_exists=True,
        logits_exists=logits_exists,
        sharpe_baseline=sharpe_baseline,
        sharpe_current=sharpe_current,
        fallback_join=fallback_join,
        validation_failure_reason=validation_failure_reason,
        issues=tuple(issues),
    )


def _evaluate_tasks(tasks: tuple[dict[str, Any], ...], *, require_logits: bool) -> tuple[TaskCheckResult, ...]:
    results: list[TaskCheckResult] = []
    for task in tasks:
        manifest_path = task["manifest_path"]
        logits_path = task.get("logits_path")
        base_sharpe = _extract_float(task.get("sharpe"))
        result = _check_manifest(
            manifest_path,
            expected_plan_id=str(task.get("plan_id") or ""),
            sharpe_baseline=base_sharpe,
            require_logits=require_logits,
            logits_path=logits_path if isinstance(logits_path, Path) else None,
        )
        results.append(result)
    return tuple(results)


def check_collapsed_replay_plan(plan_path: Path, *, require_logits: bool) -> PlanCheckReport:
    plan_payload = _load_json_file(plan_path)
    generated_at = _parse_datetime_candidate(plan_payload.get("generated_at"))
    tasks = _coerce_plan_tasks(plan_payload, plan_dir=plan_path.parent)
    task_results = _evaluate_tasks(tasks, require_logits=require_logits)
    return PlanCheckReport(plan_path=plan_path, generated_at=generated_at, tasks=task_results)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check collapsed replay plan completion status.")
    parser.add_argument(
        "plan_path",
        type=Path,
        help="Path to the plan JSON produced by plan_collapsed_replays.py.",
    )
    parser.add_argument(
        "--require-logits",
        action="store_true",
        help="Fail when logits paths are missing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when any task fails validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = check_collapsed_replay_plan(args.plan_path, require_logits=args.require_logits)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    if args.strict and report.has_failures:
        return 1
    return 0


__all__ = ["PlanCheckReport", "TaskCheckResult", "check_collapsed_replay_plan"]


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
