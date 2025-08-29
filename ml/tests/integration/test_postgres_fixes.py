#!/usr/bin/env python3
"""
Test script to verify PostgreSQL fixtures work correctly.
"""

import pytest

from ml.stores.feature_store import FeatureStore
from ml.tests.fixtures.database_fixtures import TestDatabase


@pytest.mark.database
@pytest.mark.serial
def test_feature_store_with_postgres(test_database: TestDatabase):
    """Test that FeatureStore can connect to PostgreSQL."""
    # Create FeatureStore with test database
    store = FeatureStore(connection_string=test_database.connection_string)

    # Verify connection string is set correctly
    assert store.connection_string == test_database.connection_string

    # Verify store is initialized
    assert store is not None
    assert store.feature_engineer is not None


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
def test_clean_db_fixture(test_database: TestDatabase):
    """Test that clean_postgres_db fixture works."""
    # Should have clean database
    with test_database.get_session() as session:
        from sqlalchemy import text
        result = session.execute(text("SELECT 1"))
        assert result.scalar() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
