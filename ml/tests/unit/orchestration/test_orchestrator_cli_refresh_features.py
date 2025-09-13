#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def test_orchestrator_cli_refresh_features(monkeypatch: object, tmp_path: Path) -> None:
    # Stub event emitter to capture calls
    emitter = ModuleType("ml.common.event_emitter")
    emitter.calls = []  # type: ignore[attr-defined]

    def _emit_dataset_event(*args: object, **kwargs: object) -> None:  # noqa: D401 - test stub
        emitter.calls.append(kwargs)

    emitter.emit_dataset_event = _emit_dataset_event  # type: ignore[attr-defined]
    sys.modules["ml.common.event_emitter"] = emitter

    # Stub IntegrationManager to avoid DB
    import ml.cli.pipeline_orchestrator as orch_mod
    import ml.core.integration as core_integ

    class _Mgr:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.data_registry = object()
            self.model_registry = object()
            self.feature_registry = object()
            self.data_store = object()

    orch_mod.MLIntegrationManager = _Mgr  # type: ignore[assignment]
    core_integ.MLIntegrationManager = _Mgr  # type: ignore[assignment]

    # Stub dataset builder and teacher mains
    build = ModuleType("ml.scripts.build_tft_dataset")

    def _build_main(argv: list[str] | None = None) -> int:  # noqa: D401 - test stub
        if argv and "--out_dir" in argv:
            out = Path(argv[argv.index("--out_dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "dataset.csv").write_text("id,ts\n1,1\n", encoding="utf-8")
        return 0

    build.main = _build_main  # type: ignore[attr-defined]
    sys.modules["ml.scripts.build_tft_dataset"] = build

    teacher = ModuleType("ml.training.teacher.tft_cli")
    teacher.main = lambda argv=None: 0  # type: ignore[assignment]
    sys.modules["ml.training.teacher.tft_cli"] = teacher

    from ml.cli.pipeline_orchestrator import main as orch_main

    out_dir = tmp_path / "out"
    rc = orch_main([
        "--coverage_mode", "sql",
        "--write_mode", "datastore",
        "--data_dir", str(tmp_path),
        "--symbols", "SPY.NYSE",
        "--out_dir", str(out_dir),
        "--refresh_features",
    ])
    assert rc == 0
    # One event should be emitted with FEATURE_COMPUTED stage
    assert emitter.calls, "emit_dataset_event should have been called"
    assert emitter.calls[-1]["stage"].value == "FEATURE_COMPUTED"
