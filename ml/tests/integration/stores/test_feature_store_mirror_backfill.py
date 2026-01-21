"""
Integration tests for feature store mirror backfill.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from ml.config.feature_store_mirror import FeatureStoreMirrorBackfillConfig
from ml.config.feature_store_mirror import FeatureStoreMirrorConfig
from ml.data.cache_common import day_partition_path
from ml.stores.feature_store_mirror_backfill import backfill_feature_store_mirror


pytestmark = [
    pytest.mark.database,
    pytest.mark.serial,
    pytest.mark.usefixtures("isolated_prometheus_registry"),
]


def test_feature_store_mirror_backfill_writes_parquet(
    tmp_path: Path,
    test_database: Any,
) -> None:
    ts_event = 1_700_000_000_000_000_000
    ts_init = ts_event
    feature_set_id = "test_set"
    instrument_id = "AAPL"
    values = json.dumps({"feature": 1.23})

    with test_database.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO ml_feature_values
                    (feature_set_id, instrument_id, ts_event, ts_init, values, source)
                VALUES
                    (:feature_set_id, :instrument_id, :ts_event, :ts_init, :values, :source)
                """,
            ),
            {
                "feature_set_id": feature_set_id,
                "instrument_id": instrument_id,
                "ts_event": ts_event,
                "ts_init": ts_init,
                "values": values,
                "source": "test",
            },
        )

    mirror_config = FeatureStoreMirrorConfig(
        enabled=True,
        base_dir=tmp_path / "feature_values",
    )
    config = FeatureStoreMirrorBackfillConfig(
        db_connection=test_database.connection_string,
        batch_size=10,
    )
    result = backfill_feature_store_mirror(config, mirror_config=mirror_config)

    assert result.rows_written == 1

    day = datetime.fromtimestamp(ts_event / 1_000_000_000, tz=UTC).date()
    expected_path = day_partition_path(mirror_config.base_dir, instrument_id, day)
    assert expected_path.exists()
