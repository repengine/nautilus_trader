"""
Unit tests for the database engine manager.

Tests the singleton pattern, thread safety, and connection pooling functionality
of the EngineManager class.

"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from ml.core.db_engine import EngineManager


if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.flaky
@pytest.mark.slow
@pytest.mark.unit
class TestEngineManager:
    """Test cases for the EngineManager singleton."""

    def setup_method(self) -> None:
        """Clean up any existing engines before each test."""
        EngineManager.dispose_all()

    def teardown_method(self) -> None:
        """Clean up engines after each test."""
        EngineManager.dispose_all()

    @pytest.mark.database
    @pytest.mark.serial
    def test_singleton_pattern(self) -> None:
        """Test that the same connection string returns the same engine instance."""
        # Arrange
        conn_str = "sqlite:///:memory:"

        # Act
        engine1 = EngineManager.get_engine(conn_str)
        engine2 = EngineManager.get_engine(conn_str)

        # Assert
        assert engine1 is engine2
        assert EngineManager.get_engine_count() == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_different_connections_get_different_engines(self) -> None:
        """Test that different connection strings get different engine instances."""
        # Arrange
        conn_str1 = "sqlite:///:memory:?db1"
        conn_str2 = "sqlite:///:memory:?db2"

        # Act
        engine1 = EngineManager.get_engine(conn_str1)
        engine2 = EngineManager.get_engine(conn_str2)

        # Assert
        assert engine1 is not engine2
        assert EngineManager.get_engine_count() == 2

    @pytest.mark.database
    @pytest.mark.serial
    def test_thread_safety(self) -> None:
        """Test that the manager is thread-safe under concurrent access."""
        # Arrange
        conn_str = "sqlite:///:memory:"
        engines: list[Engine] = []

        def get_engine() -> None:
            engine = EngineManager.get_engine(conn_str)
            engines.append(engine)

        # Act
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_engine)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # Assert
        assert len(engines) == 10
        # All should be the same instance
        assert all(e is engines[0] for e in engines)
        assert EngineManager.get_engine_count() == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_concurrent_different_connections(self) -> None:
        """Test thread safety with different connection strings."""
        # Arrange
        results = {}

        def get_engine(conn_str: str, index: int) -> None:
            engine = EngineManager.get_engine(conn_str)
            results[index] = engine

        # Act
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                conn_str = f"sqlite:///:memory:?db{i % 3}"  # 3 different DBs
                future = executor.submit(get_engine, conn_str, i)
                futures.append(future)

            # Wait for all to complete
            for future in futures:
                future.result()

        # Assert
        assert len(results) == 10
        assert EngineManager.get_engine_count() == 3  # 3 unique connection strings

    @pytest.mark.database
    @pytest.mark.serial
    def test_dispose_engine(self) -> None:
        """Test disposing a specific engine."""
        # Arrange
        conn_str = "sqlite:///:memory:"
        engine = EngineManager.get_engine(conn_str)

        # Act
        assert EngineManager.has_engine(conn_str)
        EngineManager.dispose_engine(conn_str)

        # Assert
        assert not EngineManager.has_engine(conn_str)
        assert EngineManager.get_engine_count() == 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_dispose_all(self) -> None:
        """Test disposing all engines."""
        # Arrange
        for i in range(5):
            EngineManager.get_engine(f"sqlite:///:memory:?db{i}")

        assert EngineManager.get_engine_count() == 5

        # Act
        EngineManager.dispose_all()

        # Assert
        assert EngineManager.get_engine_count() == 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_test_environment_detection(self) -> None:
        """Test that test environments get conservative pool settings."""
        # Arrange
        test_conn_str = "postgresql://user:pass@localhost/test_db"

        # Act
        engine = EngineManager.get_engine(
            test_conn_str,
            pool_size=10,
            max_overflow=20,
        )

        # Assert - can't directly check pool settings but engine should be created
        assert engine is not None
        assert EngineManager.has_engine(test_conn_str)

    @pytest.mark.database
    @pytest.mark.serial
    def test_empty_connection_string_raises_error(self) -> None:
        """Test that empty connection string raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Connection string cannot be empty"):
            EngineManager.get_engine("")

    @pytest.mark.database
    @pytest.mark.serial
    def test_none_connection_string_raises_error(self) -> None:
        """Test that None connection string raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Connection string cannot be empty"):
            EngineManager.get_engine(None)  # type: ignore

    @pytest.mark.database
    @pytest.mark.serial
    def test_pool_status_for_existing_engine(self) -> None:
        """Test getting pool status for an existing engine."""
        # Arrange
        conn_str = "sqlite:///:memory:"
        EngineManager.get_engine(conn_str)

        # Act
        status = EngineManager.get_pool_status(conn_str)

        # Assert
        assert status is not None
        assert "pool_type" in status or "size" in status

    @pytest.mark.database
    @pytest.mark.serial
    def test_pool_status_for_nonexistent_engine(self) -> None:
        """Test getting pool status for a nonexistent engine."""
        # Act
        status = EngineManager.get_pool_status("postgresql://nonexistent")

        # Assert
        assert status is None

    @pytest.mark.database
    @pytest.mark.serial
    def test_reuse_after_disposal(self) -> None:
        """Test that a new engine is created after disposal."""
        # Arrange
        conn_str = "sqlite:///:memory:"
        engine1 = EngineManager.get_engine(conn_str)

        # Act
        EngineManager.dispose_engine(conn_str)
        engine2 = EngineManager.get_engine(conn_str)

        # Assert
        assert engine1 is not engine2  # Different instances
        assert EngineManager.get_engine_count() == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_dispose_nonexistent_engine_is_safe(self) -> None:
        """Test that disposing a nonexistent engine doesn't raise an error."""
        # Act - should not raise
        EngineManager.dispose_engine("postgresql://nonexistent")

        # Assert
        assert EngineManager.get_engine_count() == 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_dispose_all_when_empty_is_safe(self) -> None:
        """Test that dispose_all when no engines exist is safe."""
        # Act - should not raise
        EngineManager.dispose_all()

        # Assert
        assert EngineManager.get_engine_count() == 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_high_concurrency_stress(self) -> None:
        """Stress test with high concurrency to ensure no race conditions."""
        # Arrange
        errors = []

        def stress_test(thread_id: int) -> None:
            try:
                for i in range(10):
                    conn_str = f"sqlite:///:memory:?db{thread_id % 5}"
                    engine = EngineManager.get_engine(conn_str)
                    assert engine is not None

                    # Simulate some work
                    time.sleep(0.001)

                    # Randomly dispose
                    if i % 3 == 0:
                        EngineManager.dispose_engine(conn_str)
            except Exception as e:
                errors.append((thread_id, e))

        # Act
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for thread_id in range(20):
                future = executor.submit(stress_test, thread_id)
                futures.append(future)

            for future in futures:
                future.result()

        # Assert
        assert len(errors) == 0, f"Errors during stress test: {errors}"
