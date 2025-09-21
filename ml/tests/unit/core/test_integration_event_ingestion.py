from __future__ import annotations

from datetime import UTC, datetime

from ml.core.integration import MLIntegrationManager
from ml.preprocessing.event_ingestion import EventIngestionConfig


def test_ingest_events_creates_artifact(tmp_path) -> None:
    mgr = MLIntegrationManager.__new__(MLIntegrationManager)  # type: ignore[misc]
    cfg = EventIngestionConfig(
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 31, tzinfo=UTC),
        out_dir=tmp_path / "events",
    )

    result = MLIntegrationManager.ingest_events(mgr, cfg)

    assert result.exists()
    assert result.name == "events.parquet"
    assert result.parent == tmp_path / "events"
