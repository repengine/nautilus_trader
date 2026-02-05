"""Validation bundle for streaming wave rollouts."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ml.common.subprocess_utils import run_command
from ml.features.canonical import find_legacy_feature_aliases


logger = logging.getLogger(__name__)

# Default paths and targets
DEFAULT_DOC_PATHS = (
    Path("ml/docs/architecture/event_driven_streaming_plan.md"),
    Path("ml/docs/ops/streaming_scaling_experiments.md"),
)
DEFAULT_STATE_PATH = Path("ml_out/streaming_training_state_snapshot.json")
ALERTS_PATH = Path("ml_out/streaming_alerts.json")
DEFAULT_PYTEST_TARGETS = (
    "ml/tests/unit/config/test_streaming_pipeline_config.py",
    "ml/tests/unit/training/event_driven/",
    "ml/tests/integration/training/event_driven/test_plan_to_result.py",
)

_LEGACY_EVENT_EXTRAS: set[str] = {
    "has_fed_event_today",
    "has_cpi_event_today",
    "has_earnings_today",
    "days_to_next_fed",
    "days_to_next_cpi",
    "days_to_next_earnings",
    "days_since_last_fed",
    "days_since_last_cpi",
    "days_since_last_earnings",
    "event_importance_score",
    "event_clustering_score",
}
_DEFAULT_META_COLUMNS: set[str] = {
    "time_index",
    "timestamp",
    "ts_event",
    "ts_init",
    "instrument_id",
    "y",
}


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for validation bundle.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Run validation bundle for streaming wave rollouts"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing streaming manifest JSON files",
    )
    parser.add_argument(
        "--manifest-limit",
        type=int,
        default=None,
        help="Limit number of manifests to validate (newest first)",
    )
    parser.add_argument(
        "--max-doc-age-hours",
        type=float,
        default=24.0,
        help="Maximum age in hours for documentation files",
    )
    parser.add_argument(
        "--alerts-only",
        action="store_true",
        help="Only check for alert files, don't run full validation",
    )
    return parser


def run_alerts_only() -> list[str]:
    """
    Check for presence of alert files.

    Returns:
        List of alert rule names that have files present
    """
    alerts = []
    if ALERTS_PATH.exists():
        alerts.append("streaming_alerts")
    return alerts


def validate_manifest_coverage(manifest_dir: Path, limit: int | None = None) -> None:
    """
    Validate that recent manifests have adequate coverage.

    Args:
        manifest_dir: Directory containing manifest files
        limit: Optional limit on number of manifests to check

    Raises:
        RuntimeError: If manifests are missing required fields or use legacy feature names
    """
    if not manifest_dir.exists():
        raise RuntimeError(f"Manifest directory does not exist: {manifest_dir}")
    manifests = sorted(
        manifest_dir.glob("*_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise RuntimeError(f"No manifests found in {manifest_dir}")
    if limit is not None and limit > 0:
        manifests = manifests[:limit]

    issues: list[str] = []
    for manifest_path in manifests:
        payload = _load_manifest_payload(manifest_path)
        if payload is None:
            issues.append(f"{manifest_path.name}: failed to load manifest JSON")
            continue
        feature_names = _extract_feature_names(payload, manifest_path)
        if not feature_names:
            issues.append(f"{manifest_path.name}: feature names unavailable")
            continue
        legacy = _find_legacy_features(feature_names)
        if legacy:
            formatted = ", ".join(
                f"{name} -> {suggestion}" for name, suggestion in sorted(legacy.items())
            )
            issues.append(f"{manifest_path.name}: legacy feature names detected: {formatted}")

    if issues:
        detail = "\n".join(f"- {issue}" for issue in issues)
        raise RuntimeError(f"Manifest coverage validation failed:\n{detail}")


def _load_manifest_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive for corrupt manifests
        logger.warning("failed to load manifest", extra={"path": str(path), "error": str(exc)})
        return None
    if not isinstance(payload, dict):
        logger.warning("manifest root is not a JSON object", extra={"path": str(path)})
        return None
    return payload


def _extract_feature_names(payload: Mapping[str, Any], manifest_path: Path) -> list[str]:
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        return []
    paths = dataset.get("paths")
    if not isinstance(paths, Mapping):
        return []

    metadata_path = _resolve_artifact_path(paths.get("metadata"), manifest_path)
    if metadata_path is not None:
        names = _extract_feature_names_from_metadata(metadata_path)
        if names:
            return names

    report_path = _resolve_artifact_path(paths.get("report_json"), manifest_path)
    if report_path is not None:
        names = _extract_feature_names_from_report(report_path)
        if names:
            return names

    return []


def _resolve_artifact_path(raw: Any, manifest_path: Path) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate
    raw_str = str(raw)
    if "/ml_out/" in raw_str:
        suffix = raw_str.split("/ml_out/", 1)[1]
        alt = Path("ml_out") / suffix
        if alt.exists():
            return alt
    if raw_str.startswith("/app/"):
        alt = Path(raw_str.replace("/app/", ""))
        if alt.exists():
            return alt
    alt = manifest_path.parent / Path(raw_str).name
    if alt.exists():
        return alt
    return None


def _extract_feature_names_from_metadata(path: Path) -> list[str]:
    payload = _load_manifest_payload(path)
    if payload is None:
        return []
    column_info = payload.get("column_info")
    if not isinstance(column_info, Mapping):
        return []
    features: list[str] = []
    for key in (
        "time_varying_known_reals",
        "time_varying_unknown_reals",
        "static_reals",
        "static_categoricals",
    ):
        values = column_info.get(key)
        features.extend(_coerce_string_list(values))

    meta_columns = _resolve_meta_columns(column_info)
    return [name for name in features if name not in meta_columns]


def _extract_feature_names_from_report(path: Path) -> list[str]:
    payload = _load_manifest_payload(path)
    if payload is None:
        return []
    feature_coverage = payload.get("feature_coverage")
    if not isinstance(feature_coverage, Mapping):
        return []
    by_symbol = feature_coverage.get("by_symbol")
    if not isinstance(by_symbol, Mapping) or not by_symbol:
        return []
    first = next(iter(by_symbol.values()), None)
    if not isinstance(first, Mapping):
        return []
    return [str(name) for name in first.keys()]


def _resolve_meta_columns(column_info: Mapping[str, Any]) -> set[str]:
    meta = set(_DEFAULT_META_COLUMNS)
    for key in ("group_id_col", "time_idx_col", "target_col"):
        value = column_info.get(key)
        if isinstance(value, str) and value:
            meta.add(value)
    return meta


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if item is not None]
    return []


def _find_legacy_features(feature_names: Sequence[str]) -> dict[str, str]:
    legacy = find_legacy_feature_aliases(feature_names)
    for name in feature_names:
        if name in _LEGACY_EVENT_EXTRAS:
            legacy[name] = "drop or map to canonical event_schedule outputs"
            continue
        if name.startswith("sma_") and name not in legacy:
            legacy[name] = "replace with canonical pipeline indicators"
    return legacy


def _check_doc_staleness(max_age_hours: float) -> None:
    """
    Check that documentation files are not stale.

    Args:
        max_age_hours: Maximum age in hours

    Raises:
        RuntimeError: If any docs are stale
    """
    now = time.time()
    max_age_seconds = max_age_hours * 3600

    for doc_path in DEFAULT_DOC_PATHS:
        if not doc_path.exists():
            continue

        age_seconds = now - doc_path.stat().st_mtime
        if age_seconds > max_age_seconds:
            age_hours = age_seconds / 3600
            raise RuntimeError(
                f"Documentation file {doc_path} is stale "
                f"({age_hours:.1f}h old, max {max_age_hours}h)"
            )


def run_validation(args: argparse.Namespace) -> None:
    """
    Run full validation bundle.

    Args:
        args: Parsed command-line arguments

    Raises:
        SubprocessExecutionError: If any validation command fails
        RuntimeError: If validation checks fail
    """
    # Check doc staleness first
    _check_doc_staleness(args.max_doc_age_hours)

    # Validate manifests
    validate_manifest_coverage(args.manifest_dir, args.manifest_limit)

    # Run mypy
    logger.info("Running mypy type checking...")
    run_command(["poetry", "run", "mypy", "ml", "--strict"])

    # Run ruff
    logger.info("Running ruff linter...")
    run_command(["poetry", "run", "ruff", "check", "ml"])

    # Run pytest on targeted test suites
    logger.info("Running pytest on streaming tests...")
    pytest_cmd = ["poetry", "run", "pytest"] + list(DEFAULT_PYTEST_TARGETS) + ["-q"]
    run_command(pytest_cmd)

    logger.info("✓ All validation checks passed")


__all__ = [
    "ALERTS_PATH",
    "DEFAULT_DOC_PATHS",
    "DEFAULT_PYTEST_TARGETS",
    "DEFAULT_STATE_PATH",
    "build_parser",
    "run_alerts_only",
    "run_validation",
    "validate_manifest_coverage",
]
