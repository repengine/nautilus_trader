"""Tests for walk-forward monitoring helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog

from playground.backtest.monitoring import MetadataDrift
from playground.backtest.monitoring import log_walk_forward_metadata
from playground.backtest.monitoring import validate_walk_forward_metadata
from playground.scripts.check_walk_forward_metadata import main as metadata_cli_main


def test_validate_walk_forward_metadata_matches_defaults() -> None:
    defaults_metadata = {
        "risk_free_rate": 0.02,
        "turnover_smoothing": {"stable": 0.30, "rolling": 0.40},
        "liquidity_config": {
            "severe_threshold": -0.02,
            "moderate_threshold": -0.01,
            "severe_regime_multiplier": 0.85,
            "moderate_regime_multiplier": 0.92,
            "severe_liquidity_multiplier": 0.55,
            "moderate_liquidity_multiplier": 0.70,
            "neutral_liquidity_multiplier": 1.0,
            "floor": 0.40,
        },
        "baseline_strategies": ["Equal Weight", "60/40 Portfolio", "Risk Parity"],
    }

    drifts = validate_walk_forward_metadata(defaults_metadata)
    assert drifts == []


def test_validate_walk_forward_metadata_detects_drift() -> None:
    metadata = {
        "risk_free_rate": 0.025,
        "turnover_smoothing": {"stable": 0.30, "rolling": 0.50},
    }

    drifts = validate_walk_forward_metadata(metadata)
    drift_fields = {drift.field for drift in drifts}
    assert {"risk_free_rate", "turnover_smoothing"}.issubset(drift_fields)


def test_log_walk_forward_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record: dict[str, MetadataDrift] = {}

    def fake_warning(event: str, **kwargs: object) -> None:  # type: ignore[override]
        field = kwargs.get("field", "_none")
        record[str(field)] = MetadataDrift(str(field), kwargs.get("expected"), kwargs.get("actual"))

    logger = structlog.get_logger("playground.tests.backtest.test_monitoring")
    monkeypatch.setattr(logger, "warning", fake_warning)
    monkeypatch.setattr("playground.backtest.monitoring.LOGGER", logger)

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "risk_free_rate": 0.03,
        "turnover_smoothing": {"stable": 0.32, "rolling": 0.45},
        "liquidity_config": {"severe_threshold": -0.02, "moderate_threshold": -0.01},
        "baseline_strategies": ["Equal Weight", "60/40 Portfolio", "Risk Parity"],
    }), encoding="utf-8")

    log_walk_forward_metadata(metadata_path)
    assert "risk_free_rate" in record


def test_check_walk_forward_metadata_cli(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "risk_free_rate": 0.02,
        "turnover_smoothing": {"stable": 0.30, "rolling": 0.40},
        "liquidity_config": {
            "severe_threshold": -0.02,
            "moderate_threshold": -0.01,
            "severe_regime_multiplier": 0.85,
            "moderate_regime_multiplier": 0.92,
            "severe_liquidity_multiplier": 0.55,
            "moderate_liquidity_multiplier": 0.70,
            "neutral_liquidity_multiplier": 1.0,
            "floor": 0.40,
        },
        "baseline_strategies": ["Equal Weight", "60/40 Portfolio", "Risk Parity"],
    }), encoding="utf-8")

    exit_code = metadata_cli_main(["--metadata", str(metadata_path)])
    assert exit_code == 0

    metadata_path.write_text(json.dumps({
        "risk_free_rate": 0.05,
        "turnover_smoothing": {"stable": 0.20, "rolling": 0.40},
    }), encoding="utf-8")
    exit_code = metadata_cli_main(["--metadata", str(metadata_path)])
    assert exit_code == 1
