"""
Unit tests for DatabaseLifecycleComponent.

This module tests the database lifecycle management component extracted from
MLIntegrationManager (Phase 3.6.1). Tests cover:

- Happy path: connection probing, migrations, container startup
- Error conditions: all candidates fail, operational errors, helpers unavailable
- Edge cases: empty candidates, already exists errors

"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.core.common.database_lifecycle import DatabaseLifecycleComponent
from ml.core.db_engine import EngineManager
from ml.tests.utils.db import build_postgres_url, get_test_db_port

TEST_DB_CONNECTION = build_postgres_url()
ALT_DB_CONNECTION = build_postgres_url(port="5433")
DEFAULT_CANDIDATES = (TEST_DB_CONNECTION, ALT_DB_CONNECTION)


if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_candidates() -> tuple[str, ...]:
    """Provide default connection candidates for tests."""
    return DEFAULT_CANDIDATES


@pytest.fixture
def mock_database_lifecycle(default_candidates: tuple[str, ...]) -> DatabaseLifecycleComponent:
    """Provide a DatabaseLifecycleComponent for unit tests."""
    return DatabaseLifecycleComponent(
        connection_candidates=default_candidates,
        auto_start_postgres=False,
        auto_migrate=False,
        allow_dummy=False,
    )


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestHappyPath:
    """Tests for successful operation paths."""

    def test_is_postgres_running_when_primary_reachable_returns_true(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify primary PostgreSQL connection detected.

        Input: Valid connection string, mocked can_connect returns True for primary.
        Expected Behavior: Returns True, db_connection unchanged.
        """
        original_connection = mock_database_lifecycle.db_connection

        # Mock can_connect to return True for primary
        monkeypatch.setattr(
            DatabaseLifecycleComponent,
            "can_connect",
            lambda self, conn: conn == original_connection,
        )

        result = mock_database_lifecycle.is_postgres_running()

        assert result is True
        assert mock_database_lifecycle.db_connection == original_connection

    def test_is_postgres_running_when_alternate_reachable_updates_connection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify fallback to alternate port works.

        Input: Connection candidates with primary unreachable, secondary reachable.
        Expected Behavior: Returns True, db_connection updated to secondary.
        """
        candidates = DEFAULT_CANDIDATES
        component = DatabaseLifecycleComponent(
            connection_candidates=candidates,
            auto_start_postgres=False,
        )

        # Mock can_connect to fail for primary, succeed for secondary
        def fake_can_connect(self: DatabaseLifecycleComponent, conn: str) -> bool:
            return conn == ALT_DB_CONNECTION

        monkeypatch.setattr(DatabaseLifecycleComponent, "can_connect", fake_can_connect)

        # Track dispose_engine calls
        disposed: list[str] = []
        monkeypatch.setattr(
            EngineManager, "dispose_engine", lambda conn: disposed.append(conn)
        )

        result = component.is_postgres_running()

        assert result is True
        assert component.db_connection.endswith(":5433/nautilus")
        assert len(disposed) > 0
        assert disposed[0].endswith(f":{get_test_db_port()}/nautilus")

    def test_can_connect_when_valid_connection_returns_true(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify database probe works with valid connection.

        Input: Valid PostgreSQL connection string (mocked).
        Expected Behavior: Execute SELECT 1 successfully.
        """
        # Create mock engine with context manager
        mock_conn = MagicMock()
        mock_conn.execute.return_value = None

        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=None)

        monkeypatch.setattr(
            EngineManager, "get_engine", lambda conn, **kwargs: mock_engine
        )

        result = mock_database_lifecycle.can_connect(
            TEST_DB_CONNECTION
        )

        assert result is True
        mock_conn.execute.assert_called_once()

    def test_init_database_runs_migrations_when_auto_migrate_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify migrations applied when enabled.

        Input: auto_migrate=True, valid connection (mocked).
        Expected Behavior: run_migrations is called.
        """
        component = DatabaseLifecycleComponent(
            connection_candidates=(
                TEST_DB_CONNECTION,
            ),
            auto_migrate=True,
            allow_dummy=False,
        )

        # Mock is_postgres_running and run_migrations
        monkeypatch.setattr(
            DatabaseLifecycleComponent, "is_postgres_running", lambda self: True
        )

        migrations_called = []
        monkeypatch.setattr(
            DatabaseLifecycleComponent,
            "run_migrations",
            lambda self: migrations_called.append(True),
        )

        component.init_database()

        assert len(migrations_called) == 1

    def test_start_postgres_container_with_docker_compose(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify Docker Compose invocation path.

        Input: Docker Compose file exists.
        Expected Behavior: docker compose up -d postgres executed.
        """
        # Create a temporary compose file
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3'\n")

        # Mock shutil.which to return docker path
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/docker")

        # Mock subprocess.run
        run_calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
            run_calls.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            return result

        monkeypatch.setattr("subprocess.run", fake_run)

        # Mock Path.exists to return True for our compose file
        original_exists = Path.exists

        def fake_exists(self: Path) -> bool:
            if str(self) == str(compose_file):
                return True
            # Check actual filesystem for the compose file candidates
            if "docker-compose" in str(self):
                return str(self) == str(compose_file)
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)

        # Set the env variable to our compose file
        monkeypatch.setenv("ML_COMPOSE_FILE", str(compose_file))

        # Mock is_postgres_running to return True after "start"
        call_count = [0]

        def fake_is_running(self: DatabaseLifecycleComponent) -> bool:
            call_count[0] += 1
            return call_count[0] > 0

        monkeypatch.setattr(
            DatabaseLifecycleComponent, "is_postgres_running", fake_is_running
        )

        mock_database_lifecycle.start_postgres_container()

        # Verify docker compose was called
        assert any("compose" in str(call) for call in run_calls)


# =============================================================================
# Error Condition Tests
# =============================================================================


class TestErrorConditions:
    """Tests for error handling paths."""

    def test_is_postgres_running_when_all_candidates_fail_returns_false(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify failure handling when no PostgreSQL reachable.

        Input: All connection candidates unreachable.
        Expected Behavior: Returns False, logs debug message.
        """
        import logging

        # Mock can_connect to always return False
        monkeypatch.setattr(
            DatabaseLifecycleComponent, "can_connect", lambda self, conn: False
        )

        with caplog.at_level(logging.DEBUG, logger="ml.core.common.database_lifecycle"):
            result = mock_database_lifecycle.is_postgres_running()

        assert result is False
        # Check that debug message was logged (message includes "postgres_unreachable")
        assert any(
            "postgres_unreachable" in record.getMessage()
            for record in caplog.records
            if record.name == "ml.core.common.database_lifecycle"
        )

    def test_can_connect_when_operational_error_returns_false(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify graceful handling of connection errors.

        Input: Invalid connection string (raises OperationalError).
        Expected Behavior: Returns False, disposes engine.
        """
        from sqlalchemy.exc import OperationalError

        # Mock get_engine to raise OperationalError
        def raise_operational_error(conn: str, **kwargs: object) -> MagicMock:
            mock_engine = MagicMock()
            mock_engine.connect.side_effect = OperationalError("statement", {}, None)
            return mock_engine

        monkeypatch.setattr(EngineManager, "get_engine", raise_operational_error)

        # Track dispose calls
        disposed: list[str] = []
        monkeypatch.setattr(
            EngineManager, "dispose_engine", lambda conn: disposed.append(conn)
        )

        result = mock_database_lifecycle.can_connect(
            TEST_DB_CONNECTION
        )

        assert result is False
        assert len(disposed) == 1

    def test_run_migrations_fallback_when_cli_unavailable(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Verify inline migration fallback path.

        Input: task helpers raise ImportError.
        Expected Behavior: Warning logged about task helpers unavailable.
        """
        # Create mock engine
        mock_conn = MagicMock()
        mock_engine = MagicMock()
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=None)

        monkeypatch.setattr(
            EngineManager, "get_engine", lambda conn, **kwargs: mock_engine
        )

        # Force ImportError for task helpers by patching the import
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if "ml.stores.migrations_runner" in name:
                raise ImportError("No module named 'ml.stores.migrations_runner'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with caplog.at_level("WARNING"):
            mock_database_lifecycle.run_migrations()

        assert any(
            "Migration helpers unavailable" in record.message
            for record in caplog.records
        )

    def test_start_postgres_container_raises_when_docker_not_found(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify clear error when Docker unavailable.

        Input: shutil.which("docker") returns None.
        Expected Behavior: RuntimeError raised.
        """
        monkeypatch.setattr("shutil.which", lambda cmd: None)

        with pytest.raises(RuntimeError, match="docker executable not found"):
            mock_database_lifecycle.start_postgres_container()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_constructor_raises_with_empty_candidates(self) -> None:
        """Verify handling of empty candidate list.

        Input: connection_candidates = ().
        Expected Behavior: ValueError raised.
        """
        with pytest.raises(ValueError, match="At least one connection candidate required"):
            DatabaseLifecycleComponent(
                connection_candidates=(),
                auto_start_postgres=False,
            )

    def test_run_migrations_handles_already_exists_errors(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Verify idempotent migration handling.

        Input: Migration with "already exists" error.
        Expected Behavior: Debug logged, migration continues without exception.
        """
        import logging

        # Create mock engine that raises "already exists" error
        mock_conn = MagicMock()

        def execute_with_error(stmt: object) -> None:
            raise Exception("relation already exists")

        mock_conn.execute.side_effect = execute_with_error

        mock_engine = MagicMock()
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=None)

        monkeypatch.setattr(
            EngineManager, "get_engine", lambda conn, **kwargs: mock_engine
        )

        # Force fallback path by raising ImportError for task helpers
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if "ml.stores.migrations_runner" in name:
                raise ImportError("No module")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # We need to check logs at DEBUG level for our specific logger
        with caplog.at_level(logging.DEBUG, logger="ml.core.common.database_lifecycle"):
            # Should not raise exception
            mock_database_lifecycle.run_migrations()

        # Verify debug log contains "already exists" - check getMessage() for formatted message
        db_lifecycle_logs = [
            record.getMessage()
            for record in caplog.records
            if record.name == "ml.core.common.database_lifecycle"
        ]
        # The migration runs and hits "already exists" errors which get logged
        assert any(
            "already exists" in msg.lower() for msg in db_lifecycle_logs
        ), f"Expected 'already exists' in logs, got: {db_lifecycle_logs}"

    def test_init_database_skips_in_dummy_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify init_database skips quietly in dummy mode.

        Input: allow_dummy=True, PostgreSQL not running.
        Expected Behavior: No migrations attempted.
        """
        component = DatabaseLifecycleComponent(
            connection_candidates=(
                TEST_DB_CONNECTION,
            ),
            auto_migrate=True,
            allow_dummy=True,
        )

        # Mock is_postgres_running to return False
        monkeypatch.setattr(
            DatabaseLifecycleComponent, "is_postgres_running", lambda self: False
        )

        # Track if run_migrations is called
        migrations_called = []
        monkeypatch.setattr(
            DatabaseLifecycleComponent,
            "run_migrations",
            lambda self: migrations_called.append(True),
        )

        component.init_database()

        # Should not call run_migrations in dummy mode when postgres not running
        assert len(migrations_called) == 0

    def test_db_connection_set_from_first_candidate(self) -> None:
        """Verify db_connection initialized from first candidate.

        Input: Multiple connection candidates.
        Expected Behavior: db_connection equals first candidate.
        """
        candidates = (
            "postgresql://postgres:postgres@localhost:5433/nautilus",
            TEST_DB_CONNECTION,
        )
        component = DatabaseLifecycleComponent(
            connection_candidates=candidates,
        )

        assert component.db_connection == candidates[0]

    def test_auto_start_postgres_default_false(self) -> None:
        """Verify auto_start_postgres defaults to False."""
        component = DatabaseLifecycleComponent(
            connection_candidates=(TEST_DB_CONNECTION,),
        )
        assert component.auto_start_postgres is False

    def test_auto_migrate_default_false(self) -> None:
        """Verify auto_migrate defaults to False."""
        component = DatabaseLifecycleComponent(
            connection_candidates=(TEST_DB_CONNECTION,),
        )
        assert component.auto_migrate is False

    def test_allow_dummy_default_false(self) -> None:
        """Verify allow_dummy defaults to False."""
        component = DatabaseLifecycleComponent(
            connection_candidates=(TEST_DB_CONNECTION,),
        )
        assert component.allow_dummy is False


# =============================================================================
# Integration-Like Tests (still unit-level with mocks)
# =============================================================================


class TestIntegrationLike:
    """Tests that verify component interactions with mocked dependencies."""

    def test_start_postgres_container_fallback_to_docker_run(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify fallback to docker run when compose unavailable.

        Input: No compose file exists, container doesn't exist.
        Expected Behavior: docker run command executed.
        """
        # Mock shutil.which
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/docker")

        # Mock all Path.exists to return False (no compose files)
        monkeypatch.setattr(Path, "exists", lambda self: False)

        # Track subprocess.run calls
        run_calls: list[list[str]] = []

        def fake_run(
            cmd: list[str],
            check: bool = False,
            capture_output: bool = False,
            text: bool = False,
        ) -> MagicMock:
            run_calls.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""  # No existing container
            return result

        monkeypatch.setattr("subprocess.run", fake_run)

        # Mock is_postgres_running to succeed after container start
        call_count = [0]

        def fake_is_running(self: DatabaseLifecycleComponent) -> bool:
            call_count[0] += 1
            return call_count[0] > 0

        monkeypatch.setattr(
            DatabaseLifecycleComponent, "is_postgres_running", fake_is_running
        )

        mock_database_lifecycle.start_postgres_container()

        # Verify docker run was called
        assert any("run" in cmd for cmd in run_calls)

    def test_start_postgres_container_starts_existing_container(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify existing container is started instead of created.

        Input: nautilus-postgres container exists but stopped.
        Expected Behavior: docker start called instead of docker run.
        """
        # Mock shutil.which
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/docker")

        # Mock Path.exists to return False (no compose files)
        monkeypatch.setattr(Path, "exists", lambda self: False)

        # Track subprocess.run calls
        run_calls: list[list[str]] = []

        def fake_run(
            cmd: list[str],
            check: bool = False,
            capture_output: bool = False,
            text: bool = False,
        ) -> MagicMock:
            run_calls.append(cmd)
            result = MagicMock()
            result.returncode = 0
            # Return container name when listing
            if "ps" in cmd:
                result.stdout = "nautilus-postgres"
            else:
                result.stdout = ""
            return result

        monkeypatch.setattr("subprocess.run", fake_run)

        # Mock is_postgres_running to succeed after start
        call_count = [0]

        def fake_is_running(self: DatabaseLifecycleComponent) -> bool:
            call_count[0] += 1
            return call_count[0] > 0

        monkeypatch.setattr(
            DatabaseLifecycleComponent, "is_postgres_running", fake_is_running
        )

        mock_database_lifecycle.start_postgres_container()

        # Verify docker start was called (not run)
        assert any("start" in cmd for cmd in run_calls)
        assert not any(cmd[1] == "run" for cmd in run_calls if len(cmd) > 1)

    def test_start_postgres_container_timeout_raises(
        self,
        mock_database_lifecycle: DatabaseLifecycleComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify timeout raises RuntimeError.

        Input: PostgreSQL never becomes ready.
        Expected Behavior: RuntimeError raised after timeout.
        """
        # Mock shutil.which
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/docker")

        # Mock Path.exists
        monkeypatch.setattr(Path, "exists", lambda self: False)

        # Mock subprocess.run to succeed
        def fake_run(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            return result

        monkeypatch.setattr("subprocess.run", fake_run)

        # Mock is_postgres_running to always return False
        monkeypatch.setattr(
            DatabaseLifecycleComponent, "is_postgres_running", lambda self: False
        )

        # Mock time.sleep to skip waiting
        monkeypatch.setattr("time.sleep", lambda x: None)

        with pytest.raises(RuntimeError, match="PostgreSQL failed to start within 30 seconds"):
            mock_database_lifecycle.start_postgres_container()
