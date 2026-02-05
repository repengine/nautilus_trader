#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from ml.tests.utils.targets import build_default_target_semantics_payload

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

def test_orchestrator_cli_promotions(monkeypatch: object, tmp_path: Path) -> None:
    # Store original sys.modules state for cleanup
    original_modules: dict[str, Any] = {
        "ml.orchestration.promotions": sys.modules.get("ml.orchestration.promotions"),
        "ml.tasks.datasets.tft_cli": sys.modules.get("ml.tasks.datasets.tft_cli"),
        "ml.training.teacher.tft_cli": sys.modules.get("ml.training.teacher.tft_cli"),
    }

    try:
        # Prepare stub promotions module (capturing calls)
        prom = ModuleType("ml.orchestration.promotions")
        setattr(prom, "calls", {"model": [], "features": []})

        def _reg_and_promote(**kwargs: object) -> str:
            cast(dict[str, list[dict[str, object]]], getattr(prom, "calls"))["model"].append(
                dict(kwargs),
            )
            return "mid123"

        def _reg_or_refresh(**kwargs: object) -> str | None:
            cast(dict[str, list[dict[str, object]]], getattr(prom, "calls"))["features"].append(
                dict(kwargs),
            )
            return "fid123"

        setattr(prom, "register_and_promote_model", _reg_and_promote)
        setattr(prom, "register_or_refresh_features", _reg_or_refresh)
        sys.modules["ml.orchestration.promotions"] = prom

        # Stub dataset builder to write dataset.csv and return 0
        build = ModuleType("ml.tasks.datasets.tft_cli")

        def _build_main(argv: list[str] | None = None) -> int:
            out_dir = None
            if argv is not None and "--out_dir" in argv:
                out_dir = Path(argv[argv.index("--out_dir") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "dataset.csv").write_text("id,ts\n1,1\n", encoding="utf-8")
            return 0

        setattr(build, "main", _build_main)
        sys.modules["ml.tasks.datasets.tft_cli"] = build

        # Stub teacher CLI
        tft = ModuleType("ml.training.teacher.tft_cli")

        def _tft_main(argv: list[str] | None = None) -> int:
            return 0

        setattr(tft, "main", _tft_main)
        sys.modules["ml.training.teacher.tft_cli"] = tft

        # Feature metrics file for registration
        feat_metrics = tmp_path / "feature_metrics.json"
        feat_metrics.write_text('{"feature_set_id": "fs1", "metric": 1.0}', encoding="utf-8")

        # Call orchestrator CLI main
        from ml.cli.pipeline_orchestrator import main as orch_main
        from ml import cli as _ml_cli_pkg  # just to ensure package load for patching
        import ml.cli.pipeline_orchestrator as orch_cli_mod

        # Save original MLIntegrationManager classes before stubbing
        _original_orch_mgr = getattr(orch_cli_mod, "MLIntegrationManager", None)

        # Stub IntegrationManager used by CLI to avoid DB dependencies
        class _StubStore:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def write_ingestion(
                self,
                *,
                dataset_id: str,
                records: object,
                source: str,
                run_id: str,
                instrument_id: str,
            ) -> None:
                self.calls.append(
                    {
                        "dataset_id": dataset_id,
                        "records": records,
                        "source": source,
                        "run_id": run_id,
                        "instrument_id": instrument_id,
                    },
                )

        class _StubMgr:
            def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: D401 - simple stub
                self.data_registry = object()
                self.model_registry = object()
                self.feature_registry = object()
                self.strategy_registry = object()
                self.feature_store = object()
                self.model_store = object()
                self.strategy_store = object()
                self.partition_manager = object()
                self.data_store = _StubStore()

        cast(Any, orch_cli_mod).MLIntegrationManager = _StubMgr
        import ml.core.integration as core_integ

        _original_core_mgr = getattr(core_integ, "MLIntegrationManager", None)
        cast(Any, core_integ).MLIntegrationManager = _StubMgr

        out_dir = tmp_path / "out"
        target_semantics = build_default_target_semantics_payload()
        args = [
            "--data_dir",
            str(tmp_path),
            "--symbols",
            "SPY.NYSE",
            "--out_dir",
            str(out_dir),
            "--target_semantics",
            json.dumps(target_semantics),
            "--coverage_mode",
            "sql",
            "--write_mode",
            "datastore",
            # promotions flags
            "--auto_register_model",
            "--gates_json",
            "ml/config/promotion_gates_example.json",
            "--auto_promote",
            "--deploy_target",
            "ml_actor",
            "--auto_register_features",
            "--feature_metrics_json",
            str(feat_metrics),
        ]
        rc = orch_main(args)
        assert rc == 0
        # Confirm promotions called with expected keys
        assert prom.calls["model"], "model promotion function should be called"
        assert prom.calls["features"], "feature registration function should be called"
    finally:
        # Cleanup: restore original sys.modules state
        for module_name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        # Restore original MLIntegrationManager classes
        import ml.cli.pipeline_orchestrator as orch_cli_mod
        import ml.core.integration as core_integ

        if _original_orch_mgr is not None:
            cast(Any, orch_cli_mod).MLIntegrationManager = _original_orch_mgr
        if _original_core_mgr is not None:
            cast(Any, core_integ).MLIntegrationManager = _original_core_mgr
