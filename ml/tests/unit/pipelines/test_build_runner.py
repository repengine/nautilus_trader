from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.pipelines.build_runner import BuildConfig
from ml.pipelines.build_runner import execute
from ml.pipelines.build_runner import load_config
from ml.pipelines.build_runner import plan_tasks
from ml.tests.utils.targets import build_default_target_semantics_payload


def test_load_config_and_plan(tmp_path: Path) -> None:
    target_semantics = build_default_target_semantics_payload()
    cfg_obj = {
        "data_dir": str(tmp_path / "data"),
        "out_dir": str(tmp_path / "out"),
        "symbols": ["spy", "qqq"],
        "include_macro": True,
        "macro_lag_days": 1,
        "target_semantics": target_semantics,
        "workers": 1,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg_obj), encoding="utf-8")

    cfg = load_config(cfg_path)
    assert isinstance(cfg, BuildConfig)
    tasks = plan_tasks(cfg)
    assert [t.symbol for t in tasks] == ["SPY", "QQQ"]


def test_execute_monkeypatched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub builder main to write a marker file and succeed
    calls: list[list[str]] = []

    def _fake_build_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        calls.append(list(argv))
        # Find --out_dir value
        out_idx = argv.index("--out_dir") + 1
        out_dir = Path(argv[out_idx])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dataset.parquet").write_bytes(b"")
        return 0

    import ml.tasks.datasets.tft_cli as build_cli

    monkeypatch.setattr(build_cli, "main", _fake_build_main)

    cfg = BuildConfig(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=build_default_target_semantics_payload(),
    )
    res = execute(cfg)
    assert res["total"] == 1 and res["succeeded"] == 1 and res["failed"] == 0
    # Verify artifact and progress
    assert (tmp_path / "out" / "SPY" / "dataset.parquet").exists()
    progress = tmp_path / "out" / "progress.jsonl"
    lines = progress.read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(line).get("event") == "success" for line in lines)
