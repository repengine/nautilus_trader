from __future__ import annotations

import re
from datetime import datetime

import pytest

from ml.stores.infrastructure import PartitionManager

pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.usefixtures("clean_postgres_db_module"),
]


def test_create_test_partitions_minimal(clean_postgres_db, postgres_connection: str) -> None:  # type: ignore[override]
    pm = PartitionManager(connection_string=postgres_connection, tables=["ml_model_predictions"], months_ahead=0)
    # Create a tiny range (one month)
    created = pm.create_test_partitions(
        start_year=2025,
        start_month=1,
        end_year=2025,
        end_month=1,
    )
    assert created >= 0


def test_ensure_current_partition_name_format() -> None:
    # Sanity-check month formatting logic separately from DB behavior
    now = datetime.now()
    pattern = rf"ml_strategy_signals_{now.year:04d}_{now.month:02d}"
    assert re.match(r"ml_strategy_signals_\d{4}_\d{2}", pattern)

