"""
Unit tests for CrossAssetFeatureService.

Tests the modular service layer for cross-asset relationship features (beta, spreads,
correlations) following Protocol-First Interface Design (Pattern 2).

Coverage Target: ≥90%

Test Categories:
- Happy path: Write/read operations with correct namespacing
- Upsert behavior: Conflict resolution on duplicate keys
- Empty results: Graceful handling when no data exists
- Edge cases: Time range boundaries, JSON serialization
- Protocol isolation: Minimal mock-based dependencies

Design: These tests are initially marked @pytest.mark.skip and define the contract
for implementation. Once CrossAssetFeatureService is implemented, remove skip markers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import uuid
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table

from ml.stores.table_factory import get_schema_name

if TYPE_CHECKING:
    from ml.stores.services.feature_services import CrossAssetFeatureService
    from ml.tests.conftest import TestDatabase


pytestmark = pytest.mark.serial


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def cross_asset_service(test_database: TestDatabase) -> CrossAssetFeatureService:
    """
    Provide initialized CrossAssetFeatureService for testing.

    Uses ComponentFeatureStore to ensure proper table setup and dependency injection.
    """
    from ml.stores.feature_store import ComponentFeatureStore

    store = ComponentFeatureStore(connection_string=test_database.connection_string)
    return store.cross_asset


@pytest.fixture
def mock_cross_asset_deps():
    """
    Provide mock _CrossAssetDeps for protocol-based testing.

    Enables fast unit tests without database by mocking engine and table.
    """
    mock_deps = MagicMock()
    mock_deps.engine = MagicMock()

    # Create mock table with required columns
    metadata = MetaData()
    mock_deps.feature_values_table = Table(
        "ml_feature_values",
        metadata,
        Column("feature_set_id", String),
        Column("instrument_id", String),
        Column("ts_event", Integer),
        Column("ts_init", Integer),
        Column("values", JSON),
    )

    return mock_deps


@pytest.fixture
def sample_beta_data() -> dict:
    """Provide sample beta data for tests."""
    asset_id = f"AAPL_{uuid.uuid4().hex[:8]}"
    benchmark_id = f"SPY_{uuid.uuid4().hex[:8]}"
    return {
        "asset_id": asset_id,
        "benchmark_id": benchmark_id,
        "ts_event": 1234567890000000000,
        "ts_init": 1234567890000000000,
        "beta": 1.25,
        "lookback_periods": 60,
        "ewma_span": 30,
    }


@pytest.fixture
def sample_spread_data() -> dict:
    """Provide sample spread data for tests."""
    asset_1 = f"AAPL_{uuid.uuid4().hex[:8]}"
    asset_2 = f"MSFT_{uuid.uuid4().hex[:8]}"
    return {
        "asset_1_id": asset_1,
        "asset_2_id": asset_2,
        "ts_event": 1234567890000000000,
        "ts_init": 1234567890000000000,
        "z_score": 2.5,
        "spread_value": 10.5,
        "lookback_periods": 60,
    }


@pytest.fixture
def sample_correlation_data() -> dict:
    """Provide sample correlation data for tests."""
    asset_1 = f"GOOGL_{uuid.uuid4().hex[:8]}"
    asset_2 = f"AMZN_{uuid.uuid4().hex[:8]}"
    return {
        "asset_1_id": asset_1,
        "asset_2_id": asset_2,
        "ts_event": 1234567890000000000,
        "ts_init": 1234567890000000000,
        "correlation": 0.85,
        "lookback_periods": 30,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _query_beta_row(
    engine,
    asset_id: str,
    benchmark_id: str,
    ts_event: int,
) -> dict | None:
    """
    Query database for beta row.

    Returns row as dict or None if not found.
    """
    from sqlalchemy import select
    from sqlalchemy import table as _table
    from sqlalchemy import column as _column

    feature_table = _table(
        "ml_feature_values",
        _column("feature_set_id"),
        _column("instrument_id"),
        _column("ts_event"),
        _column("ts_init"),
        _column("values"),
        schema=get_schema_name(engine),
    )

    feature_set_id = f"cross_asset:beta:{asset_id}:{benchmark_id}"

    stmt = (
        select(
            feature_table.c.feature_set_id,
            feature_table.c.instrument_id,
            feature_table.c.ts_event,
            feature_table.c.ts_init,
            feature_table.c["values"],
        )
        .where(feature_table.c.feature_set_id == feature_set_id)
        .where(feature_table.c.ts_event == ts_event)
    )

    with engine.connect() as conn:
        result = conn.execute(stmt).fetchone()
        if result is None:
            return None
        return {
            "feature_set_id": result[0],
            "instrument_id": result[1],
            "ts_event": result[2],
            "ts_init": result[3],
            "values": result[4],
        }


def _count_rows_for_beta(
    engine,
    asset_id: str,
    benchmark_id: str,
    ts_event: int,
) -> int:
    """Count rows for specific beta key."""
    from sqlalchemy import func
    from sqlalchemy import select
    from sqlalchemy import table as _table
    from sqlalchemy import column as _column

    feature_table = _table(
        "ml_feature_values",
        _column("feature_set_id"),
        _column("instrument_id"),
        _column("ts_event"),
        schema=get_schema_name(engine),
    )

    feature_set_id = f"cross_asset:beta:{asset_id}:{benchmark_id}"

    stmt = (
        select(func.count())
        .select_from(feature_table)
        .where(feature_table.c.feature_set_id == feature_set_id)
        .where(feature_table.c.instrument_id == asset_id)
        .where(feature_table.c.ts_event == ts_event)
    )

    with engine.connect() as conn:
        result = conn.execute(stmt).scalar()
        return result or 0


def _count_all_rows_for_instruments(
    engine,
    asset_1: str,
    asset_2: str,
) -> int:
    """Count all cross-asset rows for given instrument pair."""
    from sqlalchemy import func
    from sqlalchemy import select
    from sqlalchemy import table as _table
    from sqlalchemy import column as _column

    feature_table = _table(
        "ml_feature_values",
        _column("feature_set_id"),
        _column("instrument_id"),
        schema=get_schema_name(engine),
    )

    stmt = (
        select(func.count())
        .select_from(feature_table)
        .where(feature_table.c.feature_set_id.like(f"%{asset_1}%{asset_2}%"))
    )

    with engine.connect() as conn:
        result = conn.execute(stmt).scalar()
        return result or 0


# ============================================================================
# HAPPY PATH TESTS
# ============================================================================



@pytest.mark.usefixtures("clean_postgres_db")
def test_write_beta_inserts_with_namespaced_id(
    cross_asset_service: CrossAssetFeatureService,
    sample_beta_data: dict,
    test_database: TestDatabase,
) -> None:
    """
    Verify that write_beta() correctly inserts a beta value with namespaced feature_set_id.

    Contract:
    - feature_set_id format: "cross_asset:beta:{asset_id}:{benchmark_id}"
    - instrument_id is primary asset
    - values stored as JSON with all metadata
    """
    cross_asset_service.write_beta(
        asset_id=sample_beta_data["asset_id"],
        benchmark_id=sample_beta_data["benchmark_id"],
        ts_event=sample_beta_data["ts_event"],
        ts_init=sample_beta_data["ts_init"],
        beta=sample_beta_data["beta"],
        lookback_periods=sample_beta_data["lookback_periods"],
        ewma_span=sample_beta_data["ewma_span"],
    )

    # Query database for inserted row
    row = _query_beta_row(
        test_database.engine,
        sample_beta_data["asset_id"],
        sample_beta_data["benchmark_id"],
        sample_beta_data["ts_event"],
    )

    assert row is not None, "Beta row should exist in database"
    assert row["feature_set_id"] == f"cross_asset:beta:{sample_beta_data['asset_id']}:{sample_beta_data['benchmark_id']}"
    assert row["instrument_id"] == sample_beta_data["asset_id"]
    assert row["ts_event"] == sample_beta_data["ts_event"]
    assert row["ts_init"] == sample_beta_data["ts_init"]
    assert row["values"]["beta"] == sample_beta_data["beta"]
    assert row["values"]["lookback_periods"] == sample_beta_data["lookback_periods"]
    assert row["values"]["ewma_span"] == sample_beta_data["ewma_span"]



@pytest.mark.usefixtures("clean_postgres_db")
def test_write_beta_upserts_on_conflict(
    cross_asset_service: CrossAssetFeatureService,
    sample_beta_data: dict,
    test_database: TestDatabase,
) -> None:
    """
    Verify that write_beta() correctly updates existing row on conflict.

    Contract:
    - Conflict keys: (feature_set_id, instrument_id, ts_event)
    - Updates: values, ts_init
    - Only one row should exist after upsert
    """
    # First write
    cross_asset_service.write_beta(
        asset_id=sample_beta_data["asset_id"],
        benchmark_id=sample_beta_data["benchmark_id"],
        ts_event=sample_beta_data["ts_event"],
        ts_init=1000000000000000000,
        beta=1.2,
        lookback_periods=sample_beta_data["lookback_periods"],
        ewma_span=sample_beta_data["ewma_span"],
    )

    # Second write (same keys, different values)
    cross_asset_service.write_beta(
        asset_id=sample_beta_data["asset_id"],
        benchmark_id=sample_beta_data["benchmark_id"],
        ts_event=sample_beta_data["ts_event"],
        ts_init=2000000000000000000,
        beta=1.3,
        lookback_periods=sample_beta_data["lookback_periods"],
        ewma_span=sample_beta_data["ewma_span"],
    )

    # Verify only one row exists
    count = _count_rows_for_beta(
        test_database.engine,
        sample_beta_data["asset_id"],
        sample_beta_data["benchmark_id"],
        sample_beta_data["ts_event"],
    )
    assert count == 1, "Only one row should exist after upsert"

    # Verify updated values
    row = _query_beta_row(
        test_database.engine,
        sample_beta_data["asset_id"],
        sample_beta_data["benchmark_id"],
        sample_beta_data["ts_event"],
    )
    assert row is not None
    assert row["values"]["beta"] == 1.3, "Beta should be updated"
    assert row["ts_init"] == 2000000000000000000, "ts_init should be updated"



@pytest.mark.usefixtures("clean_postgres_db")
def test_get_beta_history_returns_values_in_time_range(
    cross_asset_service: CrossAssetFeatureService,
    sample_beta_data: dict,
) -> None:
    """
    Verify that get_beta_history() correctly retrieves beta values within time range.

    Contract:
    - Time range: [start_ts, end_ts) - inclusive start, exclusive end
    - Results ordered by ts_event ascending
    - Each dict contains: ts_event, beta, lookback_periods, ewma_span
    """
    # Write 5 beta values with different timestamps
    timestamps = [1000000000000000000, 2000000000000000000, 3000000000000000000, 4000000000000000000, 5000000000000000000]

    for i, ts in enumerate(timestamps):
        cross_asset_service.write_beta(
            asset_id=sample_beta_data["asset_id"],
            benchmark_id=sample_beta_data["benchmark_id"],
            ts_event=ts,
            ts_init=ts,
            beta=1.0 + i * 0.1,
            lookback_periods=sample_beta_data["lookback_periods"],
            ewma_span=sample_beta_data["ewma_span"],
        )

    # Query for middle range
    results = cross_asset_service.get_beta_history(
        asset_id=sample_beta_data["asset_id"],
        benchmark_id=sample_beta_data["benchmark_id"],
        start_ts=2000000000000000000,
        end_ts=4000000000000000000,
    )

    # Verify results
    assert len(results) == 2, "Should return 2 values in range [2000, 4000)"
    assert results[0]["ts_event"] == 2000000000000000000
    assert results[1]["ts_event"] == 3000000000000000000
    assert all("beta" in r and "lookback_periods" in r and "ewma_span" in r for r in results)
    assert results == sorted(results, key=lambda r: r["ts_event"]), "Results should be ordered by ts_event"



@pytest.mark.usefixtures("clean_postgres_db")
def test_get_beta_history_returns_empty_when_no_data(
    cross_asset_service: CrossAssetFeatureService,
) -> None:
    """
    Verify that get_beta_history() returns empty list when no data exists.

    Contract:
    - Returns [] when no data found
    - No exceptions raised
    """
    results = cross_asset_service.get_beta_history(
        asset_id="NONEXISTENT",
        benchmark_id="SPY",
        start_ts=1000000000000000000,
        end_ts=2000000000000000000,
    )

    assert results == [], "Should return empty list when no data exists"
    assert isinstance(results, list), "Result should be a list"



@pytest.mark.usefixtures("clean_postgres_db")
def test_get_beta_history_excludes_end_timestamp(
    cross_asset_service: CrossAssetFeatureService,
    sample_beta_data: dict,
) -> None:
    """
    Verify that get_beta_history() excludes end timestamp (half-open interval).

    Contract:
    - Time range is [start_ts, end_ts) - inclusive start, exclusive end
    """
    timestamps = [1000000000000000000, 2000000000000000000, 3000000000000000000]

    for ts in timestamps:
        cross_asset_service.write_beta(
            asset_id=sample_beta_data["asset_id"],
            benchmark_id=sample_beta_data["benchmark_id"],
            ts_event=ts,
            ts_init=ts,
            beta=1.0,
            lookback_periods=sample_beta_data["lookback_periods"],
            ewma_span=sample_beta_data["ewma_span"],
        )

    # Query [1000, 2000) - should return only ts=1000
    results = cross_asset_service.get_beta_history(
        asset_id=sample_beta_data["asset_id"],
        benchmark_id=sample_beta_data["benchmark_id"],
        start_ts=1000000000000000000,
        end_ts=2000000000000000000,
    )

    assert len(results) == 1, "Should return only start timestamp"
    assert results[0]["ts_event"] == 1000000000000000000



@pytest.mark.usefixtures("clean_postgres_db")
def test_write_spread_with_correct_namespace(
    cross_asset_service: CrossAssetFeatureService,
    sample_spread_data: dict,
    test_database: TestDatabase,
) -> None:
    """
    Verify that write_spread() uses correct namespace format.

    Contract:
    - feature_set_id format: "cross_asset:spread:{asset_1_id}:{asset_2_id}"
    - instrument_id is primary asset (asset_1_id)
    - values contains: z_score, spread_value, lookback_periods
    """
    cross_asset_service.write_spread(
        asset_1_id=sample_spread_data["asset_1_id"],
        asset_2_id=sample_spread_data["asset_2_id"],
        ts_event=sample_spread_data["ts_event"],
        ts_init=sample_spread_data["ts_init"],
        z_score=sample_spread_data["z_score"],
        spread_value=sample_spread_data["spread_value"],
        lookback_periods=sample_spread_data["lookback_periods"],
    )

    # Query database
    from sqlalchemy import select
    from sqlalchemy import table as _table
    from sqlalchemy import column as _column

    feature_table = _table(
        "ml_feature_values",
        _column("feature_set_id"),
        _column("instrument_id"),
        _column("values"),
        schema=get_schema_name(test_database.engine),
    )

    feature_set_id = f"cross_asset:spread:{sample_spread_data['asset_1_id']}:{sample_spread_data['asset_2_id']}"

    stmt = select(
        feature_table.c.feature_set_id,
        feature_table.c.instrument_id,
        feature_table.c["values"],
    ).where(feature_table.c.feature_set_id == feature_set_id)

    with test_database.engine.connect() as conn:
        result = conn.execute(stmt).fetchone()

    assert result is not None, "Spread row should exist"
    assert result[0] == feature_set_id
    assert result[1] == sample_spread_data["asset_1_id"]
    assert result[2]["z_score"] == sample_spread_data["z_score"]
    assert result[2]["spread_value"] == sample_spread_data["spread_value"]
    assert result[2]["lookback_periods"] == sample_spread_data["lookback_periods"]



@pytest.mark.usefixtures("clean_postgres_db")
def test_write_correlation_with_correct_namespace(
    cross_asset_service: CrossAssetFeatureService,
    sample_correlation_data: dict,
    test_database: TestDatabase,
) -> None:
    """
    Verify that write_correlation() uses correct namespace format.

    Contract:
    - feature_set_id format: "cross_asset:correlation:{asset_1_id}:{asset_2_id}"
    - instrument_id is primary asset (asset_1_id)
    - values contains: correlation, lookback_periods
    """
    cross_asset_service.write_correlation(
        asset_1_id=sample_correlation_data["asset_1_id"],
        asset_2_id=sample_correlation_data["asset_2_id"],
        ts_event=sample_correlation_data["ts_event"],
        ts_init=sample_correlation_data["ts_init"],
        correlation=sample_correlation_data["correlation"],
        lookback_periods=sample_correlation_data["lookback_periods"],
    )

    # Query database
    from sqlalchemy import select
    from sqlalchemy import table as _table
    from sqlalchemy import column as _column

    feature_table = _table(
        "ml_feature_values",
        _column("feature_set_id"),
        _column("instrument_id"),
        _column("values"),
        schema=get_schema_name(test_database.engine),
    )

    feature_set_id = f"cross_asset:correlation:{sample_correlation_data['asset_1_id']}:{sample_correlation_data['asset_2_id']}"

    stmt = select(
        feature_table.c.feature_set_id,
        feature_table.c.instrument_id,
        feature_table.c["values"],
    ).where(feature_table.c.feature_set_id == feature_set_id)

    with test_database.engine.connect() as conn:
        result = conn.execute(stmt).fetchone()

    assert result is not None, "Correlation row should exist"
    assert result[0] == feature_set_id
    assert result[1] == sample_correlation_data["asset_1_id"]
    assert result[2]["correlation"] == sample_correlation_data["correlation"]
    assert result[2]["lookback_periods"] == sample_correlation_data["lookback_periods"]


# ============================================================================
# ERROR CONDITION TESTS
# ============================================================================



def test_service_initialization_with_minimal_deps(
    mock_cross_asset_deps,
) -> None:
    """
    Verify that CrossAssetFeatureService can be initialized with minimal protocol deps.

    Contract:
    - Service depends only on _CrossAssetDeps protocol
    - No database required for instantiation
    - All methods are accessible
    """
    from ml.stores.services.feature_services import CrossAssetFeatureService

    service = CrossAssetFeatureService(deps=mock_cross_asset_deps)

    assert service is not None
    assert hasattr(service, "write_beta")
    assert hasattr(service, "get_beta_history")
    assert hasattr(service, "write_spread")
    assert hasattr(service, "write_correlation")



@pytest.mark.usefixtures("clean_postgres_db")
def test_namespace_collision_prevention(
    cross_asset_service: CrossAssetFeatureService,
    test_database: TestDatabase,
) -> None:
    """
    Verify that different feature types never collide even with same asset pairs.

    Contract:
    - Beta, spread, and correlation for same assets create separate rows
    - feature_set_id namespaces prevent collisions
    """
    unique_asset = f"COLLISION_{uuid.uuid4().hex[:8]}"
    unique_benchmark = f"BENCH_{uuid.uuid4().hex[:8]}"
    ts_event = 1234567890000000000
    ts_init = 1234567890000000000

    # Write beta
    cross_asset_service.write_beta(
        asset_id=unique_asset,
        benchmark_id=unique_benchmark,
        ts_event=ts_event,
        ts_init=ts_init,
        beta=1.0,
        lookback_periods=60,
        ewma_span=30,
    )

    # Write spread
    cross_asset_service.write_spread(
        asset_1_id=unique_asset,
        asset_2_id=unique_benchmark,
        ts_event=ts_event,
        ts_init=ts_init,
        z_score=2.0,
        spread_value=10.0,
        lookback_periods=60,
    )

    # Write correlation
    cross_asset_service.write_correlation(
        asset_1_id=unique_asset,
        asset_2_id=unique_benchmark,
        ts_event=ts_event,
        ts_init=ts_init,
        correlation=0.8,
        lookback_periods=60,
    )

    # Count total rows
    count = _count_all_rows_for_instruments(
        test_database.engine,
        unique_asset,
        unique_benchmark,
    )
    assert count == 3, "Three distinct rows should exist"

    # Verify distinct feature_set_ids
    from sqlalchemy import select
    from sqlalchemy import table as _table
    from sqlalchemy import column as _column

    feature_table = _table(
        "ml_feature_values",
        _column("feature_set_id"),
        schema=get_schema_name(test_database.engine),
    )

    stmt = select(feature_table.c.feature_set_id).where(
        feature_table.c.feature_set_id.like(f"%{unique_asset}%"),
    )

    with test_database.engine.connect() as conn:
        results = conn.execute(stmt).fetchall()

    feature_set_ids = {row[0] for row in results}
    assert len(feature_set_ids) == 3, "Three distinct feature_set_ids should exist"
    assert f"cross_asset:beta:{unique_asset}:{unique_benchmark}" in feature_set_ids
    assert f"cross_asset:spread:{unique_asset}:{unique_benchmark}" in feature_set_ids
    assert f"cross_asset:correlation:{unique_asset}:{unique_benchmark}" in feature_set_ids


# ============================================================================
# EDGE CASE TESTS
# ============================================================================



@pytest.mark.usefixtures("clean_postgres_db")
def test_json_serialization_of_metadata(
    cross_asset_service: CrossAssetFeatureService,
    sample_beta_data: dict,
    test_database: TestDatabase,
) -> None:
    """
    Verify that metadata is correctly serialized as JSON and retrieved as proper types.

    Contract:
    - lookback_periods, ewma_span stored as int
    - beta stored as float
    - JSON round-trip preserves types
    """
    cross_asset_service.write_beta(
        asset_id=sample_beta_data["asset_id"],
        benchmark_id=sample_beta_data["benchmark_id"],
        ts_event=sample_beta_data["ts_event"],
        ts_init=sample_beta_data["ts_init"],
        beta=sample_beta_data["beta"],
        lookback_periods=100,
        ewma_span=50,
    )

    row = _query_beta_row(
        test_database.engine,
        sample_beta_data["asset_id"],
        sample_beta_data["benchmark_id"],
        sample_beta_data["ts_event"],
    )

    assert row is not None
    assert isinstance(row["values"]["lookback_periods"], int)
    assert isinstance(row["values"]["ewma_span"], int)
    assert isinstance(row["values"]["beta"], float)
    assert row["values"]["lookback_periods"] == 100
    assert row["values"]["ewma_span"] == 50
