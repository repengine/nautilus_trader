#!/usr/bin/env python3

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

from pathlib import Path
from typing import Any

import pytest

from ml.cli import pipeline_orchestrator
from ml.data import BuildResult


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

def test_pipeline_orchestrator_cli_attach_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    build_calls: list[Path] = []

    def _fake_build(cfg: Any, **kwargs: Any) -> BuildResult:
        assert kwargs.get("data_store") is None
        build_calls.append(Path(cfg.out_dir))
        output_dir = Path(cfg.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv = output_dir / "dataset.csv"
        dataset_parquet = output_dir / "dataset.parquet"
        features_npz = output_dir / "features_npz.npz"
        dataset_csv.write_text("time_index,y\n0,0\n", encoding="utf-8")
        dataset_parquet.write_bytes(b"PARQUET")
        features_npz.write_bytes(b"NPZ")
        return BuildResult(
            dataset_parquet=dataset_parquet,
            dataset_csv=dataset_csv,
            features_npz=features_npz,
            feature_names=["feature_0"],
            feature_set_id=None,
        )

    monkeypatch.setattr("ml.data.build_tft_dataset", _fake_build)

    manager_calls: list[dict[str, Any]] = []

    class _StubIntegrationManager:
        def __init__(self, **kwargs: Any) -> None:
            manager_calls.append(kwargs)
            self.data_registry = object()
            self.feature_registry = object()
            self.model_registry = object()
            self.strategy_registry = object()
            self.feature_store = object()
            self.model_store = object()
            self.strategy_store = object()
            self.data_store = None
            self.partition_manager = object()

    monkeypatch.setattr("ml.core.integration.MLIntegrationManager", _StubIntegrationManager)

    args = [
        "--data_dir",
        str(catalog_dir),
        "--symbols",
        "SPY.NYSE",
        "--out_dir",
        str(out_dir),
        "--catalog_path",
        str(catalog_dir),
        "--attach-runtime",
        "--runtime-db-connection",
        "postgresql://example",
        "--runtime-auto-start-db",
        "--runtime-no-ensure-healthy",
        "--runtime-skip-validators",
    ]

    rc = pipeline_orchestrator.main(args)
    assert rc == 0
    assert build_calls == [out_dir]
    # one invocation for writer bootstrap, one for runtime attachment
    assert len(manager_calls) == 2
    assert manager_calls[-1]["auto_start_postgres"] is True
    assert manager_calls[-1]["ensure_healthy"] is False
    assert manager_calls[-1]["db_connection"] == "postgresql://example"
