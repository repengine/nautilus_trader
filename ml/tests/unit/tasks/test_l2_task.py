from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from ml.data.ingest.l2_efficient import PopulateL2TaskConfig
from ml.data.ingest.l2_efficient import populate_l2_efficient

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@pytest.fixture(autouse=True)
def _stub_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ml.data.ingest.api.ensure_service", lambda: object())


def test_populate_l2_efficient_builds_loader_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tier1_symbol_loader_stub: tuple[str, ...],
) -> None:
    recorded: dict[str, Any] = {}

    def _fake_populate(config: Any, *, service: object) -> object:
        recorded["config"] = config
        recorded["service"] = service
        from ml.data.ingest.l2_efficient import L2PopulateResult

        return L2PopulateResult(total_records=0, total_size_mb=0.0, symbols_processed=1)

    monkeypatch.setattr("ml.data.ingest.l2_efficient.populate_l2_data", _fake_populate)
    config = PopulateL2TaskConfig(
        data_dir=tmp_path,
        progress_file=tmp_path / "progress.json",
        tier=1,
        days=2,
        start_date=datetime(2024, 1, 2, 0, 0, 0),
        end_date=None,
        rate_limit=30,
        shuffle=True,
    )

    populate_l2_efficient(config)

    assert recorded["config"].dataset == "DBEQ.BASIC"
    assert recorded["config"].symbols == tuple(tier1_symbol_loader_stub)
    assert recorded["config"].rate_limit == 30
    assert recorded["config"].shuffle is True


def test_task_l2_module_is_retired() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ml.tasks.ingest.l2")
