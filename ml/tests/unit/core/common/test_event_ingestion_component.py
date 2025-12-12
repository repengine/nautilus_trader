"""
Unit tests for EventIngestionComponent.

This module tests the event ingestion component extracted from MLIntegrationManager
(Phase 3.6.7). Tests cover:

- Happy path: event ingestion returns path, backfill invokes CLI, metrics emitted
- Error conditions: ingestion failure metrics, missing required env vars
- Edge cases: backfill disabled, CLI failure with partition maintenance fallback

"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.core.common.event_ingestion import EventIngestionComponent
from ml.tests.utils.db import build_postgres_url


TEST_DB_CONNECTION = build_postgres_url()


if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event_ingestion_component() -> EventIngestionComponent:
    """Provide a basic EventIngestionComponent."""
    return EventIngestionComponent(
        db_connection=TEST_DB_CONNECTION,
    )


@pytest.fixture
def mock_partition_manager() -> MagicMock:
    """Provide a mock partition manager."""
    manager = MagicMock()
    manager.run_maintenance = MagicMock(return_value={"created": 2, "dropped": 0})
    return manager


@pytest.fixture
def component_with_partition_manager(
    mock_partition_manager: MagicMock,
) -> EventIngestionComponent:
    """Provide a component with partition manager."""
    return EventIngestionComponent(
        db_connection=TEST_DB_CONNECTION,
        partition_manager=mock_partition_manager,
    )


@dataclass
class MockEventIngestionConfig:
    """Mock EventIngestionConfig for testing."""

    start: datetime = datetime(2024, 1, 1, tzinfo=UTC)
    end: datetime = datetime(2024, 1, 31, tzinfo=UTC)
    out_dir: Path = Path("./data/events")


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestHappyPath:
    """Tests for successful operation paths."""

    def test_ingest_events_returns_path(
        self,
        event_ingestion_component: EventIngestionComponent,
    ) -> None:
        """Verify event ingestion workflow returns Path.

        Input: Valid EventIngestionConfig.
        Expected Behavior: Returns Path to events.parquet.
        """
        config = MockEventIngestionConfig()
        expected_path = Path("./data/events/events.parquet")

        # Mock the EventIngestionUtility at the source module
        mock_utility_class = MagicMock()
        mock_utility_instance = MagicMock()
        mock_utility_instance.ingest.return_value = expected_path
        mock_utility_class.return_value = mock_utility_instance

        # Create a mock module with the class
        mock_module = MagicMock()
        mock_module.EventIngestionUtility = mock_utility_class

        with patch.dict(
            sys.modules, {"ml.preprocessing.event_ingestion": mock_module}
        ):
            result = event_ingestion_component.ingest_events(config)

        assert isinstance(result, Path)
        assert result == expected_path

    def test_ingest_events_success_metric_emitted(
        self,
        event_ingestion_component: EventIngestionComponent,
    ) -> None:
        """Verify success metric is emitted on successful ingestion.

        Input: Valid EventIngestionConfig.
        Expected Behavior: Success metric incremented.
        """
        config = MockEventIngestionConfig()
        expected_path = Path("./data/events/events.parquet")

        # Mock the utility
        mock_utility_class = MagicMock()
        mock_utility_instance = MagicMock()
        mock_utility_instance.ingest.return_value = expected_path
        mock_utility_class.return_value = mock_utility_instance

        mock_module = MagicMock()
        mock_module.EventIngestionUtility = mock_utility_class

        with patch.dict(
            sys.modules, {"ml.preprocessing.event_ingestion": mock_module}
        ):
            with patch(
                "ml.core.common.event_ingestion._EVENT_INGEST_COUNTER"
            ) as mock_counter:
                mock_labels = MagicMock()
                mock_counter.labels.return_value = mock_labels

                event_ingestion_component.ingest_events(config)

                # Verify success metric was incremented
                mock_counter.labels.assert_called_with(status="success")
                mock_labels.inc.assert_called_once()

    def test_maybe_run_backfill_on_start_invokes_cli(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify backfill CLI invocation when enabled.

        Input: Required env vars set.
        Expected Behavior: CLI subprocess invoked with correct arguments.
        """
        # Set required environment variables
        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL,MSFT")
        # Use noop client mode to avoid CATALOG_PATH requirement
        monkeypatch.setenv("INGEST_CLIENT_MODE", "noop")

        with patch(
            "ml.core.common.event_ingestion.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            event_ingestion_component.maybe_run_backfill_on_start()

            # Verify subprocess was called
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]

            # Check key arguments are present
            assert "python" in call_args
            assert "-m" in call_args
            assert "ml.cli.ingest_backfill" in call_args
            assert "--dataset-id" in call_args
            assert "EQUS.MINI" in call_args
            assert "--instruments" in call_args
            assert "AAPL,MSFT" in call_args

    def test_ingest_events_logs_start_and_completion(
        self,
        event_ingestion_component: EventIngestionComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify logging of ingestion start and completion.

        Input: Valid config.
        Expected Behavior: Start and completion messages logged.
        """
        import logging

        config = MockEventIngestionConfig()
        expected_path = Path("./data/events/events.parquet")

        mock_utility_class = MagicMock()
        mock_utility_instance = MagicMock()
        mock_utility_instance.ingest.return_value = expected_path
        mock_utility_class.return_value = mock_utility_instance

        mock_module = MagicMock()
        mock_module.EventIngestionUtility = mock_utility_class

        with patch.dict(
            sys.modules, {"ml.preprocessing.event_ingestion": mock_module}
        ):
            with caplog.at_level(logging.INFO):
                event_ingestion_component.ingest_events(config)

        # Check for start message
        assert any(
            "Starting event ingestion" in record.getMessage()
            for record in caplog.records
        )

        # Check for completion message
        assert any(
            "Completed event ingestion" in record.getMessage()
            for record in caplog.records
        )


# =============================================================================
# Error Condition Tests
# =============================================================================


class TestErrorConditions:
    """Tests for error handling paths."""

    def test_ingest_events_increments_error_metric_on_failure(
        self,
        event_ingestion_component: EventIngestionComponent,
    ) -> None:
        """Verify error metric emission on ingestion failure.

        Input: Ingestion utility raises exception.
        Expected Behavior: Error metric incremented, exception re-raised.
        """
        config = MockEventIngestionConfig()

        mock_utility_class = MagicMock()
        mock_utility_instance = MagicMock()
        mock_utility_instance.ingest.side_effect = RuntimeError("Ingestion failed!")
        mock_utility_class.return_value = mock_utility_instance

        mock_module = MagicMock()
        mock_module.EventIngestionUtility = mock_utility_class

        with patch.dict(
            sys.modules, {"ml.preprocessing.event_ingestion": mock_module}
        ):
            with patch(
                "ml.core.common.event_ingestion._EVENT_INGEST_COUNTER"
            ) as mock_counter:
                mock_labels = MagicMock()
                mock_counter.labels.return_value = mock_labels

                with pytest.raises(RuntimeError, match="Ingestion failed"):
                    event_ingestion_component.ingest_events(config)

                # Verify error metric was incremented
                mock_counter.labels.assert_called_with(status="error")
                mock_labels.inc.assert_called_once()

    def test_ingest_events_increments_error_metric_on_import_failure(
        self,
        event_ingestion_component: EventIngestionComponent,
    ) -> None:
        """Verify error metric on utility import failure.

        Input: EventIngestionUtility import fails.
        Expected Behavior: Error metric incremented, exception re-raised.
        """
        config = MockEventIngestionConfig()

        # Remove the module from cache if present and make import fail
        with patch.dict(
            sys.modules, {"ml.preprocessing.event_ingestion": None}
        ):
            with patch(
                "ml.core.common.event_ingestion._EVENT_INGEST_COUNTER"
            ) as mock_counter:
                mock_labels = MagicMock()
                mock_counter.labels.return_value = mock_labels

                with pytest.raises((ImportError, TypeError)):
                    event_ingestion_component.ingest_events(config)

                # Verify error metric was incremented
                mock_counter.labels.assert_called_with(status="error")
                mock_labels.inc.assert_called()

    def test_maybe_run_backfill_raises_when_required_env_missing(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify required env var validation.

        Input: ML_BACKFILL_ON_START=1 but no BACKFILL_DATASET_ID.
        Expected Behavior: RuntimeError raised.
        """
        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        # Explicitly unset required vars
        monkeypatch.delenv("BACKFILL_DATASET_ID", raising=False)
        monkeypatch.delenv("BACKFILL_INSTRUMENTS", raising=False)

        with pytest.raises(
            RuntimeError,
            match="BACKFILL_DATASET_ID and BACKFILL_INSTRUMENTS are required",
        ):
            event_ingestion_component.maybe_run_backfill_on_start()

    def test_maybe_run_backfill_raises_when_dataset_id_missing(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify error when BACKFILL_DATASET_ID missing but instruments provided.

        Input: ML_BACKFILL_ON_START=1, instruments set, no dataset_id.
        Expected Behavior: RuntimeError raised.
        """
        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.delenv("BACKFILL_DATASET_ID", raising=False)
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL,MSFT")

        with pytest.raises(
            RuntimeError,
            match="BACKFILL_DATASET_ID and BACKFILL_INSTRUMENTS are required",
        ):
            event_ingestion_component.maybe_run_backfill_on_start()

    def test_maybe_run_backfill_raises_when_catalog_mode_without_path(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify error when catalog mode enabled without CATALOG_PATH.

        Input: coverage_mode='catalog' but no CATALOG_PATH.
        Expected Behavior: RuntimeError raised.
        """
        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL")
        monkeypatch.setenv("COVERAGE_MODE", "catalog")
        monkeypatch.delenv("CATALOG_PATH", raising=False)

        with pytest.raises(
            RuntimeError,
            match="CATALOG_PATH required for catalog coverage/client",
        ):
            event_ingestion_component.maybe_run_backfill_on_start()

    def test_maybe_run_backfill_raises_when_also_write_catalog_without_path(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify error when ALSO_WRITE_CATALOG without CATALOG_PATH.

        Input: ALSO_WRITE_CATALOG=1 but no CATALOG_PATH.
        Expected Behavior: RuntimeError raised.
        """
        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL")
        monkeypatch.setenv("ALSO_WRITE_CATALOG", "1")
        monkeypatch.setenv("COVERAGE_MODE", "sql")  # Avoid catalog coverage check
        monkeypatch.setenv("INGEST_CLIENT_MODE", "noop")  # Avoid client mode check
        monkeypatch.delenv("CATALOG_PATH", raising=False)

        with pytest.raises(
            RuntimeError,
            match="ALSO_WRITE_CATALOG set but CATALOG_PATH is missing",
        ):
            event_ingestion_component.maybe_run_backfill_on_start()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_backfill_skipped_when_disabled(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify backfill is skipped when not enabled.

        Input: ML_BACKFILL_ON_START not set or '0'.
        Expected Behavior: Returns early without invoking subprocess.
        """
        # Explicitly unset or set to '0'
        monkeypatch.delenv("ML_BACKFILL_ON_START", raising=False)

        with patch(
            "ml.core.common.event_ingestion.subprocess.run"
        ) as mock_run:
            event_ingestion_component.maybe_run_backfill_on_start()
            mock_run.assert_not_called()

    def test_backfill_skipped_when_disabled_false(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify backfill skipped when explicitly set to false.

        Input: ML_BACKFILL_ON_START='false'.
        Expected Behavior: Returns early without invoking subprocess.
        """
        monkeypatch.setenv("ML_BACKFILL_ON_START", "false")

        with patch(
            "ml.core.common.event_ingestion.subprocess.run"
        ) as mock_run:
            event_ingestion_component.maybe_run_backfill_on_start()
            mock_run.assert_not_called()

    def test_backfill_handles_cli_failure(
        self,
        component_with_partition_manager: EventIngestionComponent,
        mock_partition_manager: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify partition maintenance fallback on CLI failure.

        Input: Subprocess raises exception.
        Expected Behavior: Warning logged, partition maintenance attempted.
        """
        import logging

        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL")
        # Use noop client mode to avoid CATALOG_PATH requirement
        monkeypatch.setenv("INGEST_CLIENT_MODE", "noop")

        with patch(
            "ml.core.common.event_ingestion.subprocess.run",
            side_effect=RuntimeError("CLI failed"),
        ):
            with caplog.at_level(logging.WARNING):
                component_with_partition_manager.maybe_run_backfill_on_start()

        # Should have logged warning about CLI failure
        assert any(
            "Backfill CLI failed" in record.getMessage() for record in caplog.records
        )

        # Partition maintenance should have been called
        mock_partition_manager.run_maintenance.assert_called_once()

    def test_backfill_handles_cli_failure_initializes_partition_manager(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify partition manager initialized via callback on CLI failure.

        Input: No partition manager, callback provided.
        Expected Behavior: Callback invoked to initialize partition manager.
        """
        import logging

        mock_manager = MagicMock()
        mock_manager.run_maintenance.return_value = {"created": 1}

        init_called = []

        def init_pm() -> MagicMock:
            init_called.append(True)
            return mock_manager

        component = EventIngestionComponent(
            db_connection=TEST_DB_CONNECTION,
            partition_manager=None,
            init_partition_manager=init_pm,
        )

        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL")
        # Use noop client mode to avoid CATALOG_PATH requirement
        monkeypatch.setenv("INGEST_CLIENT_MODE", "noop")

        with patch(
            "ml.core.common.event_ingestion.subprocess.run",
            side_effect=RuntimeError("CLI failed"),
        ):
            with caplog.at_level(logging.INFO):
                component.maybe_run_backfill_on_start()

        # Callback should have been invoked
        assert len(init_called) == 1

        # Maintenance should have been called on the initialized manager
        mock_manager.run_maintenance.assert_called_once()

    def test_backfill_partition_maintenance_failure_handled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify partition maintenance failure is gracefully handled.

        Input: Partition manager raises on run_maintenance.
        Expected Behavior: Warning logged, no exception propagates.
        """
        import logging

        mock_manager = MagicMock()
        mock_manager.run_maintenance.side_effect = RuntimeError("Maintenance failed")

        component = EventIngestionComponent(
            db_connection=TEST_DB_CONNECTION,
            partition_manager=mock_manager,
        )

        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL")
        # Use noop client mode to avoid CATALOG_PATH requirement
        monkeypatch.setenv("INGEST_CLIENT_MODE", "noop")

        with patch(
            "ml.core.common.event_ingestion.subprocess.run",
            side_effect=RuntimeError("CLI failed"),
        ):
            with caplog.at_level(logging.WARNING):
                # Should not raise
                component.maybe_run_backfill_on_start()

        # Should have logged warning about maintenance skip
        assert any(
            "Partition maintenance skipped" in record.getMessage()
            for record in caplog.records
        )

    def test_component_default_values(self) -> None:
        """Verify component initializes with correct defaults."""
        component = EventIngestionComponent()

        assert component.db_connection is None
        assert component.partition_manager is None

    def test_backfill_with_all_optional_env_vars(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify backfill CLI includes all optional arguments when set.

        Input: All optional env vars set.
        Expected Behavior: CLI command includes all optional arguments.
        """
        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL,MSFT,GOOG")
        monkeypatch.setenv("BACKFILL_SCHEMA", "tbbo")
        monkeypatch.setenv("COVERAGE_MODE", "catalog")
        monkeypatch.setenv("WRITE_MODE", "sql")
        monkeypatch.setenv("INGEST_CLIENT_MODE", "databento")
        monkeypatch.setenv("BACKFILL_LOOKBACK_DAYS", "14")
        monkeypatch.setenv("TABLE_NAME", "custom_market_data")
        monkeypatch.setenv("CATALOG_PATH", "/data/catalog")
        monkeypatch.setenv("DATABENTO_API_KEY", "test_api_key")
        monkeypatch.setenv("ALSO_WRITE_CATALOG", "1")

        with patch(
            "ml.core.common.event_ingestion.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            event_ingestion_component.maybe_run_backfill_on_start()

            call_args = mock_run.call_args[0][0]

            # Verify optional arguments
            assert "--schema" in call_args
            assert "tbbo" in call_args
            assert "--lookback-days" in call_args
            assert "14" in call_args
            assert "--table-name" in call_args
            assert "custom_market_data" in call_args
            assert "--catalog-path" in call_args
            assert "/data/catalog" in call_args
            assert "--api-key" in call_args
            assert "test_api_key" in call_args
            assert "--also-write-catalog" in call_args

    def test_backfill_with_catalog_client_mode(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify backfill with catalog client mode requires CATALOG_PATH.

        Input: client_mode='catalog' with CATALOG_PATH set.
        Expected Behavior: CLI command includes catalog path.
        """
        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL")
        monkeypatch.setenv("INGEST_CLIENT_MODE", "catalog")
        monkeypatch.setenv("CATALOG_PATH", "/data/catalog")

        with patch(
            "ml.core.common.event_ingestion.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            event_ingestion_component.maybe_run_backfill_on_start()

            call_args = mock_run.call_args[0][0]
            assert "--catalog-path" in call_args
            assert "/data/catalog" in call_args

    def test_backfill_enabled_values(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify various truthy values for ML_BACKFILL_ON_START.

        Input: '1', 'true', 'yes' (case-insensitive).
        Expected Behavior: Backfill is enabled for all truthy values.
        """
        truthy_values = ["1", "true", "TRUE", "yes", "YES", "True"]

        for value in truthy_values:
            monkeypatch.setenv("ML_BACKFILL_ON_START", value)
            monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
            monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL")
            # Use noop client mode to avoid CATALOG_PATH requirement
            monkeypatch.setenv("INGEST_CLIENT_MODE", "noop")

            with patch(
                "ml.core.common.event_ingestion.subprocess.run"
            ) as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                event_ingestion_component.maybe_run_backfill_on_start()

                # Should have been called for each truthy value
                assert mock_run.called, f"Backfill not triggered for value: {value}"


# =============================================================================
# Parity Tests (verify behavior matches legacy MLIntegrationManager)
# =============================================================================


class TestParityWithLegacy:
    """Tests to verify behavior matches the legacy MLIntegrationManager."""

    def test_ingest_events_logging_format(
        self,
        event_ingestion_component: EventIngestionComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify logging format matches legacy implementation.

        Expected: Log includes start, end, and out_dir parameters.
        """
        import logging

        config = MockEventIngestionConfig()
        expected_path = Path("./data/events/events.parquet")

        mock_utility_class = MagicMock()
        mock_utility_instance = MagicMock()
        mock_utility_instance.ingest.return_value = expected_path
        mock_utility_class.return_value = mock_utility_instance

        mock_module = MagicMock()
        mock_module.EventIngestionUtility = mock_utility_class

        with patch.dict(
            sys.modules, {"ml.preprocessing.event_ingestion": mock_module}
        ):
            with caplog.at_level(logging.INFO):
                event_ingestion_component.ingest_events(config)

        # Check logging format matches legacy
        start_logs = [
            r
            for r in caplog.records
            if "Starting event ingestion" in r.getMessage()
        ]
        assert len(start_logs) == 1
        msg = start_logs[0].getMessage()
        assert "start=" in msg
        assert "end=" in msg
        assert "out_dir=" in msg

    def test_backfill_cli_command_structure(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify CLI command structure matches legacy implementation.

        Expected: Command uses 'python -m ml.cli.ingest_backfill' with all flags.
        """
        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL")
        # Use noop client mode to avoid CATALOG_PATH requirement
        monkeypatch.setenv("INGEST_CLIENT_MODE", "noop")

        with patch(
            "ml.core.common.event_ingestion.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            event_ingestion_component.maybe_run_backfill_on_start()

            call_args = mock_run.call_args[0][0]

            # Verify command structure matches legacy
            assert call_args[0] == "python"
            assert call_args[1] == "-m"
            assert call_args[2] == "ml.cli.ingest_backfill"

            # Verify required flags are present (in pairs)
            expected_flags = [
                "--db",
                "--dataset-id",
                "--schema",
                "--instruments",
                "--lookback-days",
                "--coverage-mode",
                "--write-mode",
                "--table-name",
                "--client-mode",
            ]
            for flag in expected_flags:
                assert flag in call_args, f"Missing expected flag: {flag}"

    def test_backfill_logs_cli_command(
        self,
        event_ingestion_component: EventIngestionComponent,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify backfill logs the CLI command as in legacy.

        Expected: "Running backfill bootstrap: <command>" logged.
        """
        import logging

        monkeypatch.setenv("ML_BACKFILL_ON_START", "1")
        monkeypatch.setenv("BACKFILL_DATASET_ID", "EQUS.MINI")
        monkeypatch.setenv("BACKFILL_INSTRUMENTS", "AAPL")
        # Use noop client mode to avoid CATALOG_PATH requirement
        monkeypatch.setenv("INGEST_CLIENT_MODE", "noop")

        with patch(
            "ml.core.common.event_ingestion.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with caplog.at_level(logging.INFO):
                event_ingestion_component.maybe_run_backfill_on_start()

        assert any(
            "Running backfill bootstrap:" in record.getMessage()
            for record in caplog.records
        )

    def test_ingest_events_error_logging_with_exc_info(
        self,
        event_ingestion_component: EventIngestionComponent,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify error logging includes exc_info as in legacy.

        Expected: Error logs include exception info.
        """
        import logging

        config = MockEventIngestionConfig()

        mock_utility_class = MagicMock()
        mock_utility_instance = MagicMock()
        mock_utility_instance.ingest.side_effect = RuntimeError("Test error")
        mock_utility_class.return_value = mock_utility_instance

        mock_module = MagicMock()
        mock_module.EventIngestionUtility = mock_utility_class

        with patch.dict(
            sys.modules, {"ml.preprocessing.event_ingestion": mock_module}
        ):
            with caplog.at_level(logging.ERROR):
                with pytest.raises(RuntimeError):
                    event_ingestion_component.ingest_events(config)

        # Check that error was logged
        error_logs = [
            r for r in caplog.records if "Event ingestion failed" in r.getMessage()
        ]
        assert len(error_logs) == 1
        assert error_logs[0].exc_info is not None


# =============================================================================
# Component Integration Tests
# =============================================================================


class TestComponentIntegration:
    """Tests for component integration with dependencies."""

    def test_component_can_be_imported_from_package(self) -> None:
        """Verify component is properly exported from package."""
        from ml.core.common import EventIngestionComponent as ImportedComponent

        component = ImportedComponent()
        assert component is not None

    def test_component_dataclass_fields(self) -> None:
        """Verify dataclass fields are properly defined."""
        component = EventIngestionComponent(
            db_connection="test_connection",
            partition_manager=MagicMock(),
            init_partition_manager=lambda: None,
        )

        assert component.db_connection == "test_connection"
        assert component.partition_manager is not None
        assert callable(component.init_partition_manager)

    def test_init_partition_manager_callback_signature(self) -> None:
        """Verify init_partition_manager callback accepts no arguments."""
        call_count = []

        def callback() -> None:
            call_count.append(1)
            return None

        component = EventIngestionComponent(
            init_partition_manager=callback,
        )

        # Invoke the callback
        result = component.init_partition_manager()
        assert result is None
        assert len(call_count) == 1
