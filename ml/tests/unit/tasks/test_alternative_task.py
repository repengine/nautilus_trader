from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from ml.data.loaders.alternative import PopulateAlternativeDataTaskConfig
from ml.data.loaders.alternative import populate_alternative_data_task

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@pytest.fixture(autouse=True)
def _patch_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.data.loaders import alternative as alt

    def _fake_populate(config: Any) -> alt.AlternativeDataResult:
        frames = {"demo": alt.PL.DataFrame({"value": [1]})}
        return alt.AlternativeDataResult(frames=frames)

    monkeypatch.setattr(alt, "populate_alternative_data", _fake_populate)
    monkeypatch.setattr(
        alt,
        "save_alternative_data",
        lambda result, output_dir: (output_dir / "demo.parquet",),
    )


def test_populate_alternative_data_task_uses_tier1_symbols(
    tmp_path: Path,
    tier1_symbol_loader_stub: tuple[str, ...],
) -> None:
    config = PopulateAlternativeDataTaskConfig(
        output_dir=tmp_path,
        populate_all=True,
    )
    result = populate_alternative_data_task(config)
    assert "demo" in result.frames
    assert result.frames["demo"].select("value").height == 1
    assert tier1_symbol_loader_stub  # ensure stub applied for visibility


def test_task_alternative_module_is_retired() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ml.tasks.ingest.alternative")
