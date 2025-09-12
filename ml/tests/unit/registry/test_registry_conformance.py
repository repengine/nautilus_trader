import json
import os
from pathlib import Path

import pytest

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


def test_data_registry_json_backend(tmp_path: Path) -> None:
    reg_dir = tmp_path / "registry"
    registry = DataRegistry(
        registry_path=reg_dir,
        persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
    )

    # Emit an event and update a watermark
    registry.emit_event(
        dataset_id="features",
        instrument_id="EUR/USD",
        stage=Stage.FEATURE_COMPUTED,
        source=Source.HISTORICAL,
        run_id="r1",
        ts_min=1,
        ts_max=2,
        count=1,
        status=EventStatus.SUCCESS,
    )
    registry.update_watermark(
        dataset_id="features",
        instrument_id="EUR/USD",
        source=Source.HISTORICAL,
        last_success_ns=2,
        count=1,
        completeness_pct=100.0,
    )

    # Ensure JSON file exists and contains entries
    reg_file = reg_dir / "data_registry.json"
    assert reg_file.exists()
    data = json.loads(reg_file.read_text())
    assert data.get("events"), "events should not be empty"
    assert data.get("watermarks"), "watermarks should not be empty"


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.skipif(
    not os.getenv("NAUTILUS_REGISTRY_DB_URL") and not os.getenv("DATABASE_URL"),
    reason="No Postgres URL provided for registry conformance",
)
def test_data_registry_postgres_backend_smoke(tmp_path: Path) -> None:
    db_url = os.getenv("NAUTILUS_REGISTRY_DB_URL") or os.getenv("DATABASE_URL")
    registry = DataRegistry(
        registry_path=tmp_path / "registry",
        persistence_config=PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string=db_url,
        ),
    )
    # Smoke: ensure emit_event doesn't raise; migrations must be applied in env
    registry.emit_event(
        dataset_id="features",
        instrument_id="EUR/USD",
        stage=Stage.FEATURE_COMPUTED,
        source=Source.HISTORICAL,
        run_id="r1",
        ts_min=1,
        ts_max=1,
        count=1,
        status=EventStatus.SUCCESS,
    )
