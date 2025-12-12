from __future__ import annotations

from pathlib import Path
import pytest

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.serial,
    pytest.mark.usefixtures(
        "cloned_test_database",
        "isolated_prometheus_registry",
        "mock_tracing_backend",
        "isolated_orchestrator_env",
    ),
]


def test_data_registry_postgres_backend_smoke(
    tmp_path: Path,
    cloned_test_database: str,
) -> None:
    """
    Smoke test that POSTGRES-backed DataRegistry can emit an event.

    Relies on migrations applied via TestDatabase fixture in integration suite.

    """
    registry = DataRegistry(
        registry_path=tmp_path / "registry",
        persistence_config=PersistenceConfig(
            backend=BackendType.POSTGRES,
            connection_string=cloned_test_database,
        ),
    )
    # Emit event should not raise
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
