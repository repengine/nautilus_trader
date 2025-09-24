from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ml.core.integration import MLIntegrationManager
from ml.preprocessing.event_ingestion import EventIngestionConfig


def test_ingest_events_creates_artifact(tmp_path: Path) -> None:
    mgr = object.__new__(MLIntegrationManager)
    cfg = EventIngestionConfig(
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 31, tzinfo=UTC),
        out_dir=tmp_path / "events",
    )

    result = MLIntegrationManager.ingest_events(mgr, cfg)

    assert result.exists()
    assert result.name == "events.parquet"
    assert result.parent == tmp_path / "events"
