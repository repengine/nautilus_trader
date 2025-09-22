from pathlib import Path
from typing import Any

import os
import shlex

from pytest import MonkeyPatch

from ml.core.integration import MLIntegrationManager


def test_backfill_bootstrap_builds_cli_args(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # Arrange environment for catalog-mode backfill
    monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
    monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
    monkeypatch.setenv("BACKFILL_SCHEMA", "bars")
    monkeypatch.setenv("BACKFILL_INSTRUMENTS", "SPY.EQUS")
    monkeypatch.setenv("COVERAGE_MODE", "catalog")
    monkeypatch.setenv("INGEST_CLIENT_MODE", "catalog")
    monkeypatch.setenv("CATALOG_PATH", str(tmp_path))
    monkeypatch.setenv("BACKFILL_LOOKBACK_DAYS", "3")

    captured: dict[str, Any] = {"cmd": None}

    def fake_run(cmd: list[str], check: bool = False) -> None:
        captured["cmd"] = cmd

    monkeypatch.setattr("subprocess.run", fake_run)

    mgr = MLIntegrationManager.__new__(MLIntegrationManager)
    setattr(mgr, "db_connection", "postgresql://postgres:postgres@localhost:5432/nautilus")

    # Act
    MLIntegrationManager._maybe_run_backfill_on_start(mgr)

    # Assert
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:3] == ["python", "-m", "ml.cli.ingest_backfill"]
    joined = " ".join(shlex.quote(x) for x in cmd)
    assert "--dataset-id EQUS.MINI" in joined
    assert "--schema bars" in joined
    assert "--instruments SPY.EQUS" in joined
    assert "--lookback-days 3" in joined
    assert "--coverage-mode catalog" in joined
    assert "--client-mode catalog" in joined
    assert f"--catalog-path {tmp_path}" in joined
