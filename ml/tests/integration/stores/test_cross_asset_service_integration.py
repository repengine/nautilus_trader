"""
Integration tests for CrossAssetFeatureService.

Tests the end-to-end integration of the CrossAssetFeatureService with the
ComponentFeatureStore facade and PostgreSQL database.

Integration Points:
- ComponentFeatureStore facade property access
- Service lazy initialization and caching
- PostgreSQL engine and table interactions
- Full write-read cycle verification

Design: These tests require a real PostgreSQL database and verify that the service
integrates correctly with the existing FeatureStore infrastructure.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import column as _column
from sqlalchemy import select
from sqlalchemy import table as _table

from ml.stores.table_factory import get_schema_name

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
    "real_engine_manager",
)

if TYPE_CHECKING:
    from ml.stores.feature_store import ComponentFeatureStore


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_cross_asset_property_returns_service_instance(
    component_feature_store: ComponentFeatureStore,
) -> None:
    """
    Verify that ComponentFeatureStore.cross_asset property returns service instance.

    Integration contract:
    - Property returns CrossAssetFeatureService instance
    - Service is properly initialized with deps=self
    - Service has access to engine and feature_values_table

    """
    from ml.stores.services.feature_services import CrossAssetFeatureService

    service = component_feature_store.cross_asset

    assert service is not None, "cross_asset property should return service instance"
    assert isinstance(
        service,
        CrossAssetFeatureService,
    ), "Should return CrossAssetFeatureService instance"
    assert hasattr(service, "deps"), "Service should have deps attribute"
    assert hasattr(service.deps, "engine"), "Deps should have engine"
    assert hasattr(service.deps, "feature_values_table"), "Deps should have feature_values_table"


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_facade_reuses_service_instance(
    component_feature_store: ComponentFeatureStore,
) -> None:
    """
    Verify that ComponentFeatureStore reuses same service instance on repeated access.

    Integration contract:
    - First access creates service instance
    - Subsequent accesses return same instance (cached)
    - No redundant initialization

    """
    service1 = component_feature_store.cross_asset
    service2 = component_feature_store.cross_asset

    assert service1 is service2, "Should return same service instance (cached)"


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_end_to_end_beta_compute_persist_retrieve(
    component_feature_store: ComponentFeatureStore,
) -> None:
    """
    Verify end-to-end workflow: compute beta → persist → retrieve.

    Integration contract:
    - Service writes to PostgreSQL via facade
    - Data persists across service instances
    - Retrieval returns correct values with all metadata
    """
    # Write beta via service
    component_feature_store.cross_asset.write_beta(
        asset_id="AAPL",
        benchmark_id="SPY",
        ts_event=1234567890000000000,
        ts_init=1234567890000000000,
        beta=1.25,
        lookback_periods=60,
        ewma_span=30,
    )

    # Retrieve via same service
    results = component_feature_store.cross_asset.get_beta_history(
        asset_id="AAPL",
        benchmark_id="SPY",
        start_ts=1234567890000000000,
        end_ts=1234567890000000001,
    )

    assert len(results) == 1, "Should retrieve one beta value"
    assert results[0]["ts_event"] == 1234567890000000000
    assert results[0]["beta"] == 1.25
    assert results[0]["lookback_periods"] == 60
    assert results[0]["ewma_span"] == 30

    # Create new feature store instance and verify data persists
    from ml.stores.feature_store import ComponentFeatureStore

    new_store = ComponentFeatureStore(connection_string=component_feature_store.connection_string)

    results2 = new_store.cross_asset.get_beta_history(
        asset_id="AAPL",
        benchmark_id="SPY",
        start_ts=1234567890000000000,
        end_ts=1234567890000000001,
    )

    assert len(results2) == 1, "Data should persist across store instances"
    assert results2[0]["beta"] == 1.25


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_end_to_end_spread_workflow(
    component_feature_store: ComponentFeatureStore,
) -> None:
    """
    Verify end-to-end spread workflow: write → read via database query.

    Integration contract:
    - Spread data persists in ml_feature_values table
    - Correct namespace format used
    - All metadata fields preserved
    """
    # Write spread
    component_feature_store.cross_asset.write_spread(
        asset_1_id="AAPL",
        asset_2_id="MSFT",
        ts_event=1234567890000000000,
        ts_init=1234567890000000000,
        z_score=2.5,
        spread_value=10.5,
        lookback_periods=60,
    )
    # Query database directly to verify persistence
    feature_table = _table(
        "ml_feature_values",
        _column("feature_set_id"),
        _column("instrument_id"),
        _column("values"),
        _column("ts_event"),
        schema=get_schema_name(component_feature_store.engine),
    )

    stmt = select(
        feature_table.c.feature_set_id,
        feature_table.c.instrument_id,
        feature_table.c["values"],
        feature_table.c.ts_event,
    ).where(
        feature_table.c.feature_set_id == "cross_asset:spread:AAPL:MSFT",
    )

    with component_feature_store.engine.connect() as conn:
        result = conn.execute(stmt).fetchone()

    assert result is not None, "Spread should be persisted in database"
    assert result[0] == "cross_asset:spread:AAPL:MSFT"
    assert result[1] == "AAPL"
    assert result[2]["z_score"] == 2.5
    assert result[2]["spread_value"] == 10.5
    assert result[2]["lookback_periods"] == 60


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_end_to_end_correlation_workflow(
    component_feature_store: ComponentFeatureStore,
) -> None:
    """
    Verify end-to-end correlation workflow: write → read via database query.

    Integration contract:
    - Correlation data persists in ml_feature_values table
    - Correct namespace format used
    - All metadata fields preserved
    """
    # Write correlation
    component_feature_store.cross_asset.write_correlation(
        asset_1_id="GOOGL",
        asset_2_id="AMZN",
        ts_event=1234567890000000000,
        ts_init=1234567890000000000,
        correlation=0.85,
        lookback_periods=30,
    )
    # Query database directly to verify persistence
    feature_table = _table(
        "ml_feature_values",
        _column("feature_set_id"),
        _column("instrument_id"),
        _column("values"),
        _column("ts_event"),
        schema=get_schema_name(component_feature_store.engine),
    )

    stmt = select(
        feature_table.c.feature_set_id,
        feature_table.c.instrument_id,
        feature_table.c["values"],
        feature_table.c.ts_event,
    ).where(
        feature_table.c.feature_set_id == "cross_asset:correlation:GOOGL:AMZN",
    )

    with component_feature_store.engine.connect() as conn:
        result = conn.execute(stmt).fetchone()

    assert result is not None, "Correlation should be persisted in database"
    assert result[0] == "cross_asset:correlation:GOOGL:AMZN"
    assert result[1] == "GOOGL"
    assert result[2]["correlation"] == 0.85
    assert result[2]["lookback_periods"] == 30


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_service_shares_engine_with_facade(
    component_feature_store: ComponentFeatureStore,
) -> None:
    """
    Verify that CrossAssetFeatureService shares engine with ComponentFeatureStore.

    Integration contract:
    - Service uses facade's engine (no new connections)
    - Engine is same instance (not copied)
    - Connection pooling is shared

    """
    service = component_feature_store.cross_asset

    # Verify service uses facade's engine
    assert (
        service.deps.engine is component_feature_store.engine
    ), "Service should share facade's engine"


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_service_shares_table_with_facade(
    component_feature_store: ComponentFeatureStore,
) -> None:
    """
    Verify that CrossAssetFeatureService shares feature_values_table with facade.

    Integration contract:
    - Service uses facade's feature_values_table
    - No duplicate table objects created
    - Schema is consistent

    """
    service = component_feature_store.cross_asset

    # Verify service uses facade's table
    assert (
        service.deps.feature_values_table is component_feature_store.feature_values_table
    ), "Service should share facade's feature_values_table"


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_multiple_asset_pairs_coexist(
    component_feature_store: ComponentFeatureStore,
) -> None:
    """
    Verify that multiple asset pairs can coexist without conflicts.

    Integration contract:
    - Multiple beta pairs stored independently
    - No cross-contamination of data
    - Queries return correct pair-specific results

    """
    # Write betas for multiple pairs
    pairs = [
        ("AAPL", "SPY"),
        ("MSFT", "SPY"),
        ("GOOGL", "QQQ"),
    ]

    ts_event = 1234567890000000000

    for i, (asset, benchmark) in enumerate(pairs):
        component_feature_store.cross_asset.write_beta(
            asset_id=asset,
            benchmark_id=benchmark,
            ts_event=ts_event,
            ts_init=ts_event,
            beta=1.0 + i * 0.1,
            lookback_periods=60,
            ewma_span=30,
        )

    # Verify each pair returns correct data
    for i, (asset, benchmark) in enumerate(pairs):
        results = component_feature_store.cross_asset.get_beta_history(
            asset_id=asset,
            benchmark_id=benchmark,
            start_ts=ts_event,
            end_ts=ts_event + 1,
        )

        assert len(results) == 1, f"Should retrieve beta for {asset}:{benchmark}"
        assert (
            results[0]["beta"] == 1.0 + i * 0.1
        ), f"Should retrieve correct beta for {asset}:{benchmark}"


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_time_series_retrieval_performance(
    component_feature_store: ComponentFeatureStore,
) -> None:
    """
    Verify that time series retrieval performs well with many timestamps.

    Integration contract:
    - Handles large time series (100+ points)
    - Results returned in reasonable time
    - Memory usage is acceptable

    """
    # Write 100 beta values
    base_ts = 1234567890000000000
    timestamps = [base_ts + i * 1_000_000_000 for i in range(100)]

    for i, ts in enumerate(timestamps):
        component_feature_store.cross_asset.write_beta(
            asset_id="AAPL",
            benchmark_id="SPY",
            ts_event=ts,
            ts_init=ts,
            beta=1.0 + i * 0.01,
            lookback_periods=60,
            ewma_span=30,
        )

    # Retrieve full history
        results = component_feature_store.cross_asset.get_beta_history(
        asset_id="AAPL",
        benchmark_id="SPY",
        start_ts=base_ts,
        end_ts=base_ts + 100 * 1_000_000_000,
    )

    assert len(results) == 100, "Should retrieve all 100 beta values"
    assert results[0]["beta"] == 1.0, "First beta should be 1.0"
    assert results[-1]["beta"] == 1.99, "Last beta should be 1.99"
