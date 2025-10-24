#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast


def test_orchestrator_cli_refresh_features(monkeypatch: object, tmp_path: Path) -> None:
    # Store original sys.modules state for cleanup
    original_modules: dict[str, Any] = {
        "ml.common.event_emitter": sys.modules.get("ml.common.event_emitter"),
        "ml.scripts.build_tft_dataset": sys.modules.get("ml.scripts.build_tft_dataset"),
        "ml.training.teacher.tft_cli": sys.modules.get("ml.training.teacher.tft_cli"),
    }

    try:
        # Stub event emitter to capture calls
        emitter = ModuleType("ml.common.event_emitter")
        setattr(emitter, "calls", [])

        def _emit_dataset_event(*args: object, **kwargs: object) -> None:  # noqa: D401 - test stub
            emitter.calls.append(kwargs)

        setattr(emitter, "emit_dataset_event", _emit_dataset_event)
        sys.modules["ml.common.event_emitter"] = emitter

        # Stub IntegrationManager to avoid DB
        import ml.cli.pipeline_orchestrator as orch_mod
        import ml.core.integration as core_integ

        # Save original MLIntegrationManager classes before stubbing
        _original_orch_mgr = getattr(orch_mod, "MLIntegrationManager", None)
        _original_core_mgr = getattr(core_integ, "MLIntegrationManager", None)

        class _Store:
            def write_ingestion(
                self,
                *,
                dataset_id: str,
                records: object,
                source: str,
                run_id: str,
                instrument_id: str,
            ) -> None:
                return None

        class _Mgr:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.data_registry = object()
                self.model_registry = object()
                self.feature_registry = object()
                self.strategy_registry = object()
                self.feature_store = object()
                self.model_store = object()
                self.strategy_store = object()
                self.partition_manager = object()
                self.data_store = _Store()

        cast(Any, orch_mod).MLIntegrationManager = _Mgr
        cast(Any, core_integ).MLIntegrationManager = _Mgr

        # Stub dataset builder and teacher mains
        build = ModuleType("ml.scripts.build_tft_dataset")

        def _build_main(argv: list[str] | None = None) -> int:  # noqa: D401 - test stub
            if argv and "--out_dir" in argv:
                out = Path(argv[argv.index("--out_dir") + 1])
                out.mkdir(parents=True, exist_ok=True)
                (out / "dataset.csv").write_text("id,ts\n1,1\n", encoding="utf-8")
            return 0

        setattr(build, "main", _build_main)
        sys.modules["ml.scripts.build_tft_dataset"] = build

        teacher = ModuleType("ml.training.teacher.tft_cli")
        setattr(teacher, "main", lambda argv=None: 0)
        sys.modules["ml.training.teacher.tft_cli"] = teacher

        from ml.cli.pipeline_orchestrator import main as orch_main

        out_dir = tmp_path / "out"
        rc = orch_main(
            [
                "--coverage_mode",
                "sql",
                "--write_mode",
                "datastore",
                "--data_dir",
                str(tmp_path),
                "--symbols",
                "SPY.NYSE",
                "--out_dir",
                str(out_dir),
                "--refresh_features",
            ],
        )
        assert rc == 0
        # One event should be emitted with FEATURE_COMPUTED stage
        assert emitter.calls, "emit_dataset_event should have been called"
        assert emitter.calls[-1]["stage"].value == "FEATURE_COMPUTED"
    finally:
        # Cleanup: restore original sys.modules state
        for module_name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module

        # Restore original MLIntegrationManager classes
        import ml.cli.pipeline_orchestrator as orch_mod
        import ml.core.integration as core_integ

        if _original_orch_mgr is not None:
            cast(Any, orch_mod).MLIntegrationManager = _original_orch_mgr
        if _original_core_mgr is not None:
            cast(Any, core_integ).MLIntegrationManager = _original_core_mgr
