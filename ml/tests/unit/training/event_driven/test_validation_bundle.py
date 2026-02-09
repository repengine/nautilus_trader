from __future__ import annotations

import os
import json
from argparse import Namespace
from pathlib import Path

import pytest

from ml.training.event_driven.guardrails import validation_bundle


def _write_manifest(tmp_path: Path, *, feature_names: list[str]) -> Path:
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_payload = {
        "column_info": {
            "group_id_col": "instrument_id",
            "time_idx_col": "time_index",
            "target_col": "y",
            "static_categoricals": ["instrument_id", "exchange"],
            "static_reals": ["tick_size"],
            "time_varying_known_reals": ["time_index", *feature_names],
            "time_varying_unknown_reals": [],
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

    manifest_payload = {
        "cohort_run": {"plan_id": "plan-1", "completed_at": "2026-01-01T00:00:00Z"},
        "dataset": {"paths": {"metadata": str(metadata_path)}},
    }
    manifest_path = tmp_path / "plan-1_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    return manifest_path


def test_validate_manifest_coverage_accepts_canonical_names(tmp_path: Path) -> None:
    _write_manifest(tmp_path, feature_names=["return_1", "hour_sin", "hour_cos"])

    validation_bundle.validate_manifest_coverage(tmp_path, limit=1)


def test_validate_manifest_coverage_rejects_legacy_names(tmp_path: Path) -> None:
    _write_manifest(tmp_path, feature_names=["tod_sin", "return_1"])

    with pytest.raises(RuntimeError, match="tod_sin"):
        validation_bundle.validate_manifest_coverage(tmp_path, limit=1)


def test_build_parser_parses_expected_defaults(tmp_path: Path) -> None:
    parser = validation_bundle.build_parser()

    args = parser.parse_args(["--manifest-dir", str(tmp_path)])

    assert args.manifest_dir == tmp_path
    assert args.manifest_limit is None
    assert args.max_doc_age_hours == pytest.approx(24.0)
    assert args.alerts_only is False


def test_run_alerts_only_detects_streaming_alert_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alerts_path = tmp_path / "streaming_alerts.json"
    monkeypatch.setattr(validation_bundle, "ALERTS_PATH", alerts_path)

    assert validation_bundle.run_alerts_only() == []
    alerts_path.write_text("{}", encoding="utf-8")
    assert validation_bundle.run_alerts_only() == ["streaming_alerts"]


def test_validate_manifest_coverage_rejects_missing_manifest_dir(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(RuntimeError, match="Manifest directory does not exist"):
        validation_bundle.validate_manifest_coverage(missing_dir)


def test_validate_manifest_coverage_rejects_empty_manifest_dir(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No manifests found"):
        validation_bundle.validate_manifest_coverage(tmp_path)


def test_validate_manifest_coverage_uses_report_json_fallback(tmp_path: Path) -> None:
    report_path = tmp_path / "dataset_report.json"
    report_payload = {
        "feature_coverage": {
            "by_symbol": {
                "AAPL": {
                    "return_1": 1.0,
                    "hour_sin": 1.0,
                    "hour_cos": 1.0,
                },
            },
        },
    }
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    manifest_payload = {
        "dataset": {
            "paths": {
                "metadata": str(tmp_path / "missing_metadata.json"),
                "report_json": str(report_path),
            },
        },
    }
    (tmp_path / "report_manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    validation_bundle.validate_manifest_coverage(tmp_path)


def test_validate_manifest_coverage_reports_invalid_manifest_payload(tmp_path: Path) -> None:
    (tmp_path / "bad_manifest.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed to load manifest JSON"):
        validation_bundle.validate_manifest_coverage(tmp_path)


def test_validate_manifest_coverage_reports_missing_feature_names(tmp_path: Path) -> None:
    manifest_payload = {
        "dataset": {
            "paths": {
                "metadata": str(tmp_path / "missing_metadata.json"),
                "report_json": str(tmp_path / "missing_report.json"),
            },
        },
    }
    (tmp_path / "missing_features_manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="feature names unavailable"):
        validation_bundle.validate_manifest_coverage(tmp_path)


def test_resolve_artifact_path_handles_alternate_path_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "plan_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    ml_out_candidate = tmp_path / "ml_out" / "run-01" / "metadata.json"
    ml_out_candidate.parent.mkdir(parents=True, exist_ok=True)
    ml_out_candidate.write_text("{}", encoding="utf-8")

    app_candidate = tmp_path / "tmp" / "report.json"
    app_candidate.parent.mkdir(parents=True, exist_ok=True)
    app_candidate.write_text("{}", encoding="utf-8")

    sibling_candidate = manifest_dir / "artifact.json"
    sibling_candidate.write_text("{}", encoding="utf-8")

    resolved_ml_out = validation_bundle._resolve_artifact_path(
        "/app/ml_out/run-01/metadata.json",
        manifest_path,
    )
    resolved_app = validation_bundle._resolve_artifact_path("/app/tmp/report.json", manifest_path)
    resolved_sibling = validation_bundle._resolve_artifact_path(
        "/not/available/artifact.json",
        manifest_path,
    )
    missing = validation_bundle._resolve_artifact_path("", manifest_path)

    assert resolved_ml_out == Path("ml_out/run-01/metadata.json")
    assert resolved_app == Path("tmp/report.json")
    assert resolved_sibling == sibling_candidate
    assert missing is None


def test_extract_feature_name_helpers_cover_invalid_payload_shapes(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"column_info": "invalid"}), encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"feature_coverage": {"by_symbol": {"AAPL": "invalid"}}}),
        encoding="utf-8",
    )

    metadata_names = validation_bundle._extract_feature_names_from_metadata(metadata_path)
    report_names = validation_bundle._extract_feature_names_from_report(report_path)
    extracted_names = validation_bundle._extract_feature_names({}, tmp_path / "plan_manifest.json")

    assert metadata_names == []
    assert report_names == []
    assert extracted_names == []


def test_find_legacy_features_flags_event_extras_and_sma_columns() -> None:
    legacy = validation_bundle._find_legacy_features(
        ["has_fed_event_today", "sma_5", "return_1"],
    )

    assert "has_fed_event_today" in legacy
    assert "sma_5" in legacy


def test_check_doc_staleness_raises_for_old_docs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_path = tmp_path / "stale_doc.md"
    doc_path.write_text("# stale", encoding="utf-8")
    now = 20_000.0
    stale_ts = now - (3.0 * 3600.0)
    os.utime(doc_path, (stale_ts, stale_ts))
    monkeypatch.setattr(validation_bundle, "DEFAULT_DOC_PATHS", (doc_path,))
    monkeypatch.setattr(validation_bundle.time, "time", lambda: now)

    with pytest.raises(RuntimeError, match="is stale"):
        validation_bundle._check_doc_staleness(max_age_hours=1.0)


def test_run_validation_executes_expected_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_commands: list[list[str]] = []
    captured_manifest_args: list[tuple[Path, int | None]] = []

    def _fake_check_doc_staleness(max_age_hours: float) -> None:
        assert max_age_hours == pytest.approx(6.0)

    def _fake_validate_manifest_coverage(manifest_dir: Path, limit: int | None) -> None:
        captured_manifest_args.append((manifest_dir, limit))

    def _fake_run_command(command: list[str]) -> None:
        captured_commands.append(command)

    monkeypatch.setattr(validation_bundle, "_check_doc_staleness", _fake_check_doc_staleness)
    monkeypatch.setattr(
        validation_bundle,
        "validate_manifest_coverage",
        _fake_validate_manifest_coverage,
    )
    monkeypatch.setattr(validation_bundle, "run_command", _fake_run_command)
    args = Namespace(
        manifest_dir=tmp_path,
        manifest_limit=2,
        max_doc_age_hours=6.0,
    )

    validation_bundle.run_validation(args)

    assert captured_manifest_args == [(tmp_path, 2)]
    assert captured_commands == [
        ["poetry", "run", "mypy", "ml", "--strict"],
        ["poetry", "run", "ruff", "check", "ml"],
        ["poetry", "run", "pytest", *validation_bundle.DEFAULT_PYTEST_TARGETS, "-q"],
    ]
