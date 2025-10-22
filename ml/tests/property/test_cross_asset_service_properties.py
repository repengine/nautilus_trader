"""
Property-based tests for CrossAssetFeatureService.

Tests invariants that must hold regardless of specific input values using Hypothesis.

Invariants Tested:
- Timestamp monotonicity: Retrieved results maintain ascending order
- Upsert idempotence: Writing same data twice produces single row
- Namespace uniqueness: Different feature types never collide

Design: These tests use Hypothesis to generate test data and verify properties hold
for all valid inputs. Initially marked @pytest.mark.skip until implementation exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from sqlalchemy import column as _column
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import table as _table

from ml.stores.table_factory import get_schema_name

if TYPE_CHECKING:
    from ml.stores.services.feature_services import CrossAssetFeatureService
    from ml.tests.conftest import TestDatabase

pytestmark = pytest.mark.serial



# ============================================================================
# HYPOTHESIS STRATEGIES
# ============================================================================


@st.composite
def nanosecond_timestamps(draw):
    """Generate valid nanosecond timestamps."""
    return draw(
        st.integers(
            min_value=1_000_000_000_000_000_000,  # 2001-09-09
            max_value=2_000_000_000_000_000_000,  # 2033-05-18
        ),
    )


@st.composite
def valid_instrument_ids(draw):
    """Generate realistic instrument IDs."""
    return draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=1,
            max_size=10,
        ).filter(lambda x: x.isalnum()),
    )


@st.composite
def beta_values(draw):
    """Generate realistic beta values."""
    return draw(
        st.floats(
            min_value=-5.0,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


@st.composite
def correlation_values(draw):
    """Generate valid correlation values."""
    return draw(
        st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


@st.composite
def lookback_periods(draw):
    """Generate valid lookback periods."""
    return draw(st.integers(min_value=10, max_value=500))


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def cross_asset_service(test_database: TestDatabase) -> CrossAssetFeatureService:
    """Provide initialized CrossAssetFeatureService."""
    from ml.stores.feature_store import ComponentFeatureStore

    store = ComponentFeatureStore(connection_string=test_database.connection_string)
    return store.cross_asset


# ============================================================================
# PROPERTY TESTS
# ============================================================================



@pytest.mark.usefixtures("clean_postgres_db")
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    timestamps=st.lists(
        nanosecond_timestamps(),
        min_size=1,
        max_size=20,
        unique=True,
    ),
    asset_id=valid_instrument_ids(),
    benchmark_id=valid_instrument_ids(),
    beta=beta_values(),
)
def test_timestamp_monotonicity_invariant(
    cross_asset_service: CrossAssetFeatureService,
    timestamps: list[int],
    asset_id: str,
    benchmark_id: str,
    beta: float,
) -> None:
    """
    INVARIANT: Retrieved beta history maintains ascending timestamp order.

    Property: For any sequence of timestamps, get_beta_history() returns results
    sorted by ts_event in ascending order.

    This property must hold regardless of:
    - Number of timestamps written
    - Order timestamps are written
    - Values of beta/metadata
    """
    # Assume valid inputs
    if asset_id == benchmark_id:
        return  # Skip degenerate case

    # Write betas in random order
    for ts in timestamps:
        cross_asset_service.write_beta(
            asset_id=asset_id,
            benchmark_id=benchmark_id,
            ts_event=ts,
            ts_init=ts,
            beta=beta,
            lookback_periods=60,
            ewma_span=30,
        )

    # Retrieve full history
    start_ts = min(timestamps)
    end_ts = max(timestamps) + 1

    results = cross_asset_service.get_beta_history(
        asset_id=asset_id,
        benchmark_id=benchmark_id,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    # INVARIANT: Results must be sorted by ts_event
    result_timestamps = [r["ts_event"] for r in results]
    assert result_timestamps == sorted(result_timestamps), "Results must be ordered by ts_event"

    # INVARIANT: All written timestamps should be retrieved
    assert len(results) == len(timestamps), "All written timestamps should be retrieved"



@pytest.mark.usefixtures("clean_postgres_db")
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    ts_event=nanosecond_timestamps(),
    asset_id=valid_instrument_ids(),
    benchmark_id=valid_instrument_ids(),
    beta1=beta_values(),
    beta2=beta_values(),
)
def test_upsert_idempotence_invariant(
    cross_asset_service: CrossAssetFeatureService,
    test_database: TestDatabase,
    ts_event: int,
    asset_id: str,
    benchmark_id: str,
    beta1: float,
    beta2: float,
) -> None:
    """
    INVARIANT: Writing to same key multiple times produces single row with latest value.

    Property: For any (asset_id, benchmark_id, ts_event) tuple, writing N times
    results in exactly 1 row with the last written value.

    This property must hold regardless of:
    - Number of writes
    - Values written
    - Order of writes
    """
    # Assume valid inputs
    if asset_id == benchmark_id:
        return  # Skip degenerate case

    # Write first beta
    cross_asset_service.write_beta(
        asset_id=asset_id,
        benchmark_id=benchmark_id,
        ts_event=ts_event,
        ts_init=ts_event,
        beta=beta1,
        lookback_periods=60,
        ewma_span=30,
    )

    # Write second beta (same key, different value)
    cross_asset_service.write_beta(
        asset_id=asset_id,
        benchmark_id=benchmark_id,
        ts_event=ts_event,
        ts_init=ts_event,
        beta=beta2,
        lookback_periods=60,
        ewma_span=30,
    )

    # Count rows
    feature_table = _table(
        "ml_feature_values",
        _column("feature_set_id"),
        _column("instrument_id"),
        _column("ts_event"),
        schema=get_schema_name(test_database.engine),
    )

    feature_set_id = f"cross_asset:beta:{asset_id}:{benchmark_id}"

    stmt = (
        select(func.count())
        .select_from(feature_table)
        .where(feature_table.c.feature_set_id == feature_set_id)
        .where(feature_table.c.ts_event == ts_event)
    )

    with test_database.engine.connect() as conn:
        count = conn.execute(stmt).scalar()

    # INVARIANT: Exactly one row exists
    assert count == 1, "Upsert should produce exactly one row"

    # INVARIANT: Row contains last written value
    results = cross_asset_service.get_beta_history(
        asset_id=asset_id,
        benchmark_id=benchmark_id,
        start_ts=ts_event,
        end_ts=ts_event + 1,
    )

    assert len(results) == 1
    assert results[0]["beta"] == beta2, "Row should contain last written value"



@pytest.mark.usefixtures("clean_postgres_db")
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    ts_event=nanosecond_timestamps(),
    asset_1=valid_instrument_ids(),
    asset_2=valid_instrument_ids(),
    beta=beta_values(),
    correlation=correlation_values(),
)
def test_namespace_uniqueness_invariant(
    cross_asset_service: CrossAssetFeatureService,
    test_database: TestDatabase,
    ts_event: int,
    asset_1: str,
    asset_2: str,
    beta: float,
    correlation: float,
) -> None:
    """
    INVARIANT: Different feature types never collide in storage.

    Property: Writing beta, spread, and correlation for same asset pair produces
    three distinct rows with unique feature_set_id values.

    This property must hold regardless of:
    - Asset IDs chosen
    - Timestamp values
    - Feature values
    """
    # Assume valid inputs
    if asset_1 == asset_2:
        return  # Skip degenerate case

    # Write beta
    cross_asset_service.write_beta(
        asset_id=asset_1,
        benchmark_id=asset_2,
        ts_event=ts_event,
        ts_init=ts_event,
        beta=beta,
        lookback_periods=60,
        ewma_span=30,
    )

    # Write spread
    cross_asset_service.write_spread(
        asset_1_id=asset_1,
        asset_2_id=asset_2,
        ts_event=ts_event,
        ts_init=ts_event,
        z_score=0.0,
        spread_value=0.0,
        lookback_periods=60,
    )

    # Write correlation
    cross_asset_service.write_correlation(
        asset_1_id=asset_1,
        asset_2_id=asset_2,
        ts_event=ts_event,
        ts_init=ts_event,
        correlation=correlation,
        lookback_periods=60,
    )

    # Query all rows for this asset pair
    feature_table = _table(
        "ml_feature_values",
        _column("feature_set_id"),
        _column("ts_event"),
        schema=get_schema_name(test_database.engine),
    )

    stmt = (
        select(feature_table.c.feature_set_id)
        .where(feature_table.c.ts_event == ts_event)
        .where(
            feature_table.c.feature_set_id.like(f"%{asset_1}%{asset_2}%"),
        )
    )

    with test_database.engine.connect() as conn:
        results = conn.execute(stmt).fetchall()

    feature_set_ids = {row[0] for row in results}

    # INVARIANT: Three distinct feature_set_ids exist
    assert len(feature_set_ids) == 3, "Three distinct feature types should exist"

    # INVARIANT: Expected namespaces are present
    expected_beta = f"cross_asset:beta:{asset_1}:{asset_2}"
    expected_spread = f"cross_asset:spread:{asset_1}:{asset_2}"
    expected_correlation = f"cross_asset:correlation:{asset_1}:{asset_2}"

    assert expected_beta in feature_set_ids, "Beta namespace should exist"
    assert expected_spread in feature_set_ids, "Spread namespace should exist"
    assert expected_correlation in feature_set_ids, "Correlation namespace should exist"



@pytest.mark.usefixtures("clean_postgres_db")
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    timestamps=st.lists(
        nanosecond_timestamps(),
        min_size=5,
        max_size=20,
        unique=True,
    ),
    asset_id=valid_instrument_ids(),
    benchmark_id=valid_instrument_ids(),
)
def test_time_range_boundary_invariant(
    cross_asset_service: CrossAssetFeatureService,
    timestamps: list[int],
    asset_id: str,
    benchmark_id: str,
) -> None:
    """
    INVARIANT: Time range queries use half-open interval [start, end).

    Property: For any time range [start_ts, end_ts), results include timestamps
    >= start_ts and < end_ts (inclusive start, exclusive end).

    This property must hold regardless of:
    - Number of timestamps
    - Distribution of timestamps
    - Values of start/end boundaries
    """
    # Assume valid inputs
    if asset_id == benchmark_id or len(timestamps) < 5:
        return  # Skip degenerate cases

    # Write betas for all timestamps
    for ts in timestamps:
        cross_asset_service.write_beta(
            asset_id=asset_id,
            benchmark_id=benchmark_id,
            ts_event=ts,
            ts_init=ts,
            beta=1.0,
            lookback_periods=60,
            ewma_span=30,
        )

    # Sort timestamps for boundary testing
    sorted_ts = sorted(timestamps)

    # Query middle range [ts[2], ts[-2])
    start_ts = sorted_ts[2]
    end_ts = sorted_ts[-2]

    results = cross_asset_service.get_beta_history(
        asset_id=asset_id,
        benchmark_id=benchmark_id,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    result_timestamps = {r["ts_event"] for r in results}

    # INVARIANT: Start timestamp is included
    assert start_ts in result_timestamps, "Start timestamp should be included (inclusive)"

    # INVARIANT: End timestamp is excluded
    assert end_ts not in result_timestamps, "End timestamp should be excluded (exclusive)"

    # INVARIANT: Timestamps before start are excluded
    for ts in sorted_ts[:2]:
        assert ts not in result_timestamps, f"Timestamp {ts} < start_ts should be excluded"

    # INVARIANT: Timestamps at/after end are excluded
    for ts in sorted_ts[-2:]:
        assert ts not in result_timestamps, f"Timestamp {ts} >= end_ts should be excluded"
