"""Tests for cross-asset beta persistence utilities."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table

from ml.core.db_engine import EngineManager
from playground.exposure.persistence import CrossAssetBetaPersistenceConfig
from playground.exposure.persistence import persist_cross_asset_betas


@pytest.fixture()
def _sample_exposures() -> pl.DataFrame:
    ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    return pl.DataFrame(
        {
            "feature_set_id": ["fs"],
            "asset_id": ["XLF"],
            "benchmark_id": ["factor_duration"],
            "ts_event": [ts],
            "ts_init": [ts],
            "ewma_beta": [0.42],
            "ewma_cov": [0.1],
            "ewma_var_market": [0.3],
            "n_observations": [15],
            "alpha": [0.94],
            "source": ["historical"],
        },
    )


def test_persist_cross_asset_betas_disabled(_sample_exposures: pl.DataFrame) -> None:
    config = CrossAssetBetaPersistenceConfig(enabled=False)
    persisted = persist_cross_asset_betas(_sample_exposures, config)
    assert persisted == 0


def test_persist_cross_asset_betas_sqlite_roundtrip(
    _sample_exposures: pl.DataFrame,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "betas.db"
    connection = f"sqlite:///{db_path}"
    engine = EngineManager.get_engine(connection)

    metadata = MetaData()
    Table(
        "ml_cross_asset_betas",
        metadata,
        Column("feature_set_id", String, nullable=False),
        Column("asset_id", String, nullable=False),
        Column("benchmark_id", String, nullable=False),
        Column("ts_event", Integer, nullable=False),
        Column("ts_init", Integer, nullable=False),
        Column("ewma_beta", Float, nullable=False),
        Column("ewma_cov", Float, nullable=False),
        Column("ewma_var_market", Float, nullable=False),
        Column("n_observations", Integer, nullable=False),
        Column("alpha", Float, nullable=False),
        Column("source", String, nullable=False),
    )
    metadata.create_all(engine)

    config = CrossAssetBetaPersistenceConfig(
        enabled=True,
        connection_url=connection,
        chunk_size=1,
        upsert=False,
    )

    persisted = persist_cross_asset_betas(_sample_exposures, config)
    assert persisted == 1

    with engine.connect() as conn:
        result = conn.execute(
            Table("ml_cross_asset_betas", metadata, autoload_with=engine).select(),
        )
        rows = list(result)
        assert len(rows) == 1
        row = rows[0]._mapping
        assert row["asset_id"] == "XLF"
        assert pytest.approx(row["ewma_beta"]) == 0.42
