from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from ml.tasks.ingest import PopulateL2TaskConfig
from ml.tasks.ingest import populate_l2_efficient


@pytest.fixture(autouse=True)
def _stub_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ml.tasks.ingest.l2.ensure_service", lambda: object())


def test_populate_l2_efficient_builds_loader_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, Any] = {}

    def _fake_populate(config: Any, *, service: object) -> object:
        recorded["config"] = config
        recorded["service"] = service
        from ml.data.loaders.l2_efficient import L2PopulateResult

        return L2PopulateResult(total_records=0, total_size_mb=0.0, symbols_processed=1)

    monkeypatch.setattr("ml.tasks.ingest.l2.populate_l2_data", _fake_populate)
    monkeypatch.setattr("ml.tasks.ingest.l2.get_tier1_symbols", lambda: ["SPY"])

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
    assert recorded["config"].symbols == ("SPY",)
    assert recorded["config"].rate_limit == 30
    assert recorded["config"].shuffle is True
