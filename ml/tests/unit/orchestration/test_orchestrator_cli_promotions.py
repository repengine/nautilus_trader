#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def test_orchestrator_cli_promotions(monkeypatch: object, tmp_path: Path) -> None:
    # Prepare stub promotions module (capturing calls)
    prom = ModuleType("ml.orchestration.promotions")
    prom.calls: dict[str, list[dict[str, object]]] = {"model": [], "features": []}

    def _reg_and_promote(**kwargs: object) -> str:  # type: ignore[no-redef]
        prom.calls["model"].append(dict(kwargs))
        return "mid123"

    def _reg_or_refresh(**kwargs: object) -> str | None:  # type: ignore[no-redef]
        prom.calls["features"].append(dict(kwargs))
        return "fid123"

    prom.register_and_promote_model = _reg_and_promote  # type: ignore[attr-defined]
    prom.register_or_refresh_features = _reg_or_refresh  # type: ignore[attr-defined]
    sys.modules["ml.orchestration.promotions"] = prom

    # Stub dataset builder to write dataset.csv and return 0
    build = ModuleType("ml.scripts.build_tft_dataset")

    def _build_main(argv: list[str] | None = None) -> int:  # type: ignore[no-redef]
        out_dir = None
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dataset.csv").write_text("id,ts\n1,1\n", encoding="utf-8")
        return 0

    build.main = _build_main  # type: ignore[attr-defined]
    sys.modules["ml.scripts.build_tft_dataset"] = build

    # Stub teacher CLI
    tft = ModuleType("ml.training.teacher.tft_cli")

    def _tft_main(argv: list[str] | None = None) -> int:  # type: ignore[no-redef]
        return 0

    tft.main = _tft_main  # type: ignore[attr-defined]
    sys.modules["ml.training.teacher.tft_cli"] = tft

    # Feature metrics file for registration
    feat_metrics = tmp_path / "feature_metrics.json"
    feat_metrics.write_text('{"feature_set_id": "fs1", "metric": 1.0}', encoding="utf-8")

    # Call orchestrator CLI main
    from ml.cli.pipeline_orchestrator import main as orch_main
    from ml import cli as _ml_cli_pkg  # just to ensure package load for patching
    import ml.cli.pipeline_orchestrator as orch_cli_mod

    # Stub IntegrationManager used by CLI to avoid DB dependencies
    class _StubMgr:
        def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: D401 - simple stub
            self.data_registry = object()
            self.model_registry = object()
            self.feature_registry = object()
            self.data_store = object()

    orch_cli_mod.MLIntegrationManager = _StubMgr  # type: ignore[assignment]
    import ml.core.integration as core_integ
    core_integ.MLIntegrationManager = _StubMgr  # type: ignore[assignment]

    out_dir = tmp_path / "out"
    args = [
        "--data_dir",
        str(tmp_path),
        "--symbols",
        "SPY.NYSE",
        "--out_dir",
        str(out_dir),
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
