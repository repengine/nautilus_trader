from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pytest

from ml.deployment import entrypoint_pipeline
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

if TYPE_CHECKING:
    from ml.tests.fixtures.model_factory import TestDataFactory

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

class _StubStore:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - simple stub
        """No-op store placeholder for tests."""


class _StubScheduler:
    def __init__(
        self,
        *,
        catalog: Any,
        config: Any,
        use_orchestrator: bool,
        dual_write: bool,
        **_: Any,
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.use_orchestrator = use_orchestrator
        self.dual_write = dual_write
        self.run_count = 0

    def run_daily_update(self) -> None:
        self.run_count += 1

    def stop(self) -> None:  # pragma: no cover - noop for interface compatibility
        return None


def _populate_catalog(
    catalog_path: Path,
    *,
    instrument: str,
    data_factory: TestDataFactory,
) -> None:
    catalog = ParquetDataCatalog(str(catalog_path))
    start = datetime.now(tz=UTC) - timedelta(hours=1)
    bars = data_factory.bars(
        n=16,
        instrument_id=instrument,
        bar_type=f"{instrument}-1-MINUTE-LAST-EXTERNAL",
        start_date=start,
    )
    catalog.write_data(bars)


@pytest.mark.serial
def test_pipeline_runner_performs_catalog_rehydration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_data_factory: TestDataFactory,
) -> None:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    instrument = "NVDA.EQUS"
    _populate_catalog(
        catalog_dir,
        instrument=instrument,
        data_factory=test_data_factory,
    )

    db_path = tmp_path / "rehydrate.db"
    db_uri = f"sqlite:///{db_path}"

    monkeypatch.setattr(entrypoint_pipeline, "FeatureStore", _StubStore)
    monkeypatch.setattr(entrypoint_pipeline, "ModelStore", _StubStore)
    monkeypatch.setattr(entrypoint_pipeline, "DataScheduler", _StubScheduler)
    monkeypatch.setattr(entrypoint_pipeline, "check_ml_dependencies", lambda *args, **kwargs: None)

    env_overrides = {
        "PIPELINE_MODE": "backfill",
        "CATALOG_REHYDRATE_ENABLED": "1",
        "CATALOG_PATH": str(catalog_dir),
        "DB_CONNECTION": db_uri,
        "UNIVERSE_SYMBOLS": instrument,
        # Keep universe deterministic; EQUS.MINI normalization already enforces suffixes.
        "UNIVERSE_EXPAND": "0",
        "DATABENTO_DATASET": "EQUS.MINI",
        "DATABENTO_SCHEMA": "ohlcv-1m",
        "CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE": "{instrument_id}-1-MINUTE-LAST-EXTERNAL",
    }
    for name, value in env_overrides.items():
        monkeypatch.setenv(name, value)

    entrypoint_pipeline.pipeline_status.update(
        {
            "healthy": False,
            "last_run": None,
            "errors": [],
            "last_rehydrate": None,
        }
    )

    runner = entrypoint_pipeline.PipelineRunner()
    runner.run()

    assert entrypoint_pipeline.pipeline_status["last_rehydrate"] is not None

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data")
        row_count = cursor.fetchone()[0]
    assert row_count > 0

    # clean up env overrides
    for name in env_overrides:
        monkeypatch.delenv(name, raising=False)

    entrypoint_pipeline.pipeline_status.update(
        {
            "healthy": False,
            "last_run": None,
            "errors": [],
            "last_rehydrate": None,
        }
    )
