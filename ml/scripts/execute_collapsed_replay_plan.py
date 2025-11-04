#!/usr/bin/env python3
"""Execute collapsed replay plans and capture manifest telemetry summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaskExecutionResult:
    """Execution payload captured for a single replay task."""

    plan_id: str | None
    status: str
    replay_command: list[str] | None
    command_stdout: str | None
    command_stderr: str | None
    returncode: int | None
    manifest: dict[str, Any]
    logits_path: str | None
    logits_exists: bool | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "plan_id": self.plan_id,
            "status": self.status,
            "replay_command": self.replay_command,
            "command_stdout": self.command_stdout,
            "command_stderr": self.command_stderr,
            "returncode": self.returncode,
            "manifest": self.manifest,
            "logits_path": self.logits_path,
            "logits_exists": self.logits_exists,
        }


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("replay plan must be a JSON object")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("replay plan must contain a list of tasks")
    return payload


def _resolve_path(base: Path, raw: Any) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _summarize_manifest(path: Path | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path) if path is not None else None,
        "exists": False,
    }
    if path is None or not path.exists():
        return summary

    summary["exists"] = True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        summary["error"] = f"invalid_json: {exc}"
        return summary

    if not isinstance(payload, dict):
        summary["error"] = "manifest_root_not_object"
        return summary

    cohort = payload.get("cohort_run")
    if not isinstance(cohort, dict):
        summary["error"] = "missing_cohort_run"
        return summary

    metrics_block = cohort.get("metrics")
    telemetry_block = cohort.get("telemetry")
    metrics_summary: dict[str, float] = {}
    if isinstance(metrics_block, dict):
        for key in (
            "roc_auc",
            "pr_auc",
            "pr_auc_multiple",
            "log_loss",
            "economic_slippage_adjusted_sharpe",
            "economic_turnover",
            "economic_hit_rate",
            "economic_max_drawdown",
            "worker_peak_instrument_share",
        ):
            value = metrics_block.get(key)
            try:
                metrics_summary[key] = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    summary["metrics"] = metrics_summary

    validation_failure: dict[str, Any] = {}
    caps_block = {}
    if isinstance(telemetry_block, dict):
        caps_candidate = telemetry_block.get("caps")
        if isinstance(caps_candidate, dict):
            caps_block = caps_candidate
    reason = caps_block.get("validation_failure_reason")
    if isinstance(reason, str):
        validation_failure["reason"] = reason
    details = caps_block.get("validation_failure_details")
    if isinstance(details, dict):
        validation_failure["details"] = {
            str(key): value for key, value in details.items()
        }
    if validation_failure:
        summary["validation_failure"] = validation_failure
    return summary


def _run_replay_command(command: Sequence[str]) -> tuple[str, int, str, str]:
    completed = subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    )
    return "succeeded", completed.returncode, completed.stdout, completed.stderr


def _execute_task(
    task: dict[str, Any],
    *,
    base_dir: Path,
    execute: bool,
    capture_manifest: bool,
) -> TaskExecutionResult:
    plan_id = task.get("plan_id")
    replay_command_raw = task.get("replay_command")
    replay_command = [str(token) for token in replay_command_raw] if isinstance(replay_command_raw, list) else None
    manifest_path = _resolve_path(base_dir, task.get("manifest_path"))
    logits_path = _resolve_path(base_dir, task.get("logits_path"))

    status = "dry_run"
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None

    if execute and replay_command:
        try:
            status, returncode, stdout, stderr = _run_replay_command(replay_command)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            returncode = exc.returncode
            stdout = exc.stdout or ""
            stderr = exc.stderr or str(exc)
        except FileNotFoundError as exc:
            status = "failed"
            returncode = -1
            stdout = ""
            stderr = str(exc)

    manifest_summary = _summarize_manifest(manifest_path) if capture_manifest else {
        "path": str(manifest_path) if manifest_path is not None else None,
        "exists": manifest_path.exists() if manifest_path is not None else False,
    }

    return TaskExecutionResult(
        plan_id=str(plan_id) if plan_id is not None else None,
        status=status if replay_command else "no_command",
        replay_command=replay_command,
        command_stdout=stdout,
        command_stderr=stderr,
        returncode=returncode,
        manifest=manifest_summary,
        logits_path=str(logits_path) if logits_path is not None else None,
        logits_exists=logits_path.exists() if logits_path is not None else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute collapsed replay plan commands and capture telemetry summaries.",
    )
    parser.add_argument(
        "plan_path",
        type=Path,
        help="Path to the JSON plan generated by plan_collapsed_replays.py.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute replay commands instead of performing a dry run.",
    )
    parser.add_argument(
        "--summaries-output",
        type=Path,
        default=None,
        help="Optional output path for execution summaries (JSON).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tasks from the plan.",
    )
    parser.add_argument(
        "--only-plan",
        action="append",
        default=[],
        help="Restrict execution to specific plan_id values (can be repeated).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing subsequent tasks even if a command fails.",
    )
    parser.add_argument(
        "--skip-manifest-summary",
        action="store_true",
        help="Skip manifest inspection when gathering task summaries.",
    )
    args = parser.parse_args(argv)

    plan_path = args.plan_path.resolve()
    plan_payload = _load_plan(plan_path)
    tasks_payload = plan_payload["tasks"]
    base_dir = plan_path.parent

    filtered_plan_ids = {str(value) for value in args.only_plan} if args.only_plan else None
    processed: list[TaskExecutionResult] = []
    exit_code = 0

    for index, raw_task in enumerate(tasks_payload):
        if not isinstance(raw_task, dict):
            raise ValueError(f"task at index {index} must be a JSON object")
        plan_id = str(raw_task.get("plan_id") or "")
        if filtered_plan_ids and plan_id not in filtered_plan_ids:
            continue
        if args.limit is not None and len(processed) >= args.limit:
            break

        result = _execute_task(
            raw_task,
            base_dir=base_dir,
            execute=args.execute,
            capture_manifest=not args.skip_manifest_summary,
        )
        processed.append(result)

        if result.status == "failed":
            exit_code = 1
            if not args.continue_on_error:
                break

    summary_payload = {
        "plan_path": str(plan_path),
        "execute": bool(args.execute),
        "tasks_processed": len(processed),
        "tasks": [entry.as_dict() for entry in processed],
    }

    print(json.dumps(summary_payload, indent=2, sort_keys=True))

    if args.summaries_output is not None:
        args.summaries_output.parent.mkdir(parents=True, exist_ok=True)
        args.summaries_output.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
