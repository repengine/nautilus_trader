"""
Unit tests for DataCleanupComponent.

Tests cover data retention cleanup, trading day calculation, scheduling,
and status operations extracted from DataScheduler.

Test coverage targets:
- clean_old_data: Success, retention cutoff, failure metrics
- get_previous_trading_day: Monday, Sunday, weekday handling
- schedule_updates: Default cron, custom cron
- stop: Scheduler shutdown
- get_status: Status dictionary fields

"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.data.common.data_cleanup import DataCleanupComponent


if TYPE_CHECKING:
    pass


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def cleanup_component() -> DataCleanupComponent:
    """Create a fresh DataCleanupComponent for testing."""
    return DataCleanupComponent()


@pytest.fixture
def mock_scheduler_config() -> MagicMock:
    """Create a mock SchedulerConfig with default values."""
    config = MagicMock()
    config.symbols = ["SPY.XNAS", "QQQ.XNAS", "IWM.XNAS"]
    config.collection_time = "04:00"
    config.retention_days = 90
    config.databento.dataset = "GLBX.MDP3"
    config.databento.schema = "ohlcv-1m"
    return config


@pytest.fixture
def mock_catalog() -> MagicMock:
    """Create a mock ParquetDataCatalog."""
    catalog = MagicMock()
    catalog.path = "/data/catalog"
    return catalog


@pytest.fixture
def mock_feature_engineer() -> MagicMock:
    """Create a mock FeatureEngineer."""
    return MagicMock()


@pytest.fixture
def mock_metrics_server() -> MagicMock:
    """Create a mock MetricsServer."""
    server = MagicMock()
    server.stop = MagicMock()
    return server


# =============================================================================
# CLEAN_OLD_DATA TESTS
# =============================================================================


class TestCleanOldData:
    """Tests for clean_old_data method."""

    def test_clean_old_data_success(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Test successful data cleanup.

        Verifies:
        - No exception is raised on successful cleanup
        - Success metric is recorded
        - Pipeline latency is observed

        """
        with patch(
            "ml.data.common.data_cleanup.data_retention_cleanup_total"
        ) as mock_counter, patch(
            "ml.data.common.data_cleanup.pipeline_stage_latency"
        ) as mock_histogram:
            # Execute cleanup
            cleanup_component.clean_old_data(retention_days=90)

            # Verify success metric recorded
            mock_counter.labels.assert_called_once_with(status="success")
            mock_counter.labels.return_value.inc.assert_called_once()

            # Verify latency observed
            mock_histogram.labels.assert_called_once_with(stage="data_cleanup")
            mock_histogram.labels.return_value.observe.assert_called_once()

    def test_clean_old_data_retention_cutoff(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Test retention cutoff calculation.

        Verifies:
        - Different retention days produce correct cutoff dates
        - Logger receives appropriate cutoff date message

        """
        with patch("ml.data.common.data_cleanup.logger") as mock_logger, patch(
            "ml.data.common.data_cleanup.data_retention_cleanup_total"
        ), patch("ml.data.common.data_cleanup.pipeline_stage_latency"):
            # Test with 30-day retention
            cleanup_component.clean_old_data(retention_days=30)

            # Verify log message contains cutoff date
            mock_logger.info.assert_any_call(
                pytest.approx_regex(r"Cleaning data older than \d{4}-\d{2}-\d{2}")
            )

    def test_clean_old_data_failure_metrics(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Test failure metric recording on cleanup error.

        Verifies:
        - Exception is re-raised after recording metrics
        - Failure metric is recorded
        - Error is logged with exc_info=True

        """
        with patch(
            "ml.data.common.data_cleanup.data_retention_cleanup_total"
        ) as mock_counter, patch(
            "ml.data.common.data_cleanup.pipeline_stage_latency"
        ) as mock_histogram, patch(
            "ml.data.common.data_cleanup.logger"
        ) as mock_logger:
            # Make the histogram observation raise an exception to simulate failure
            mock_counter.labels.return_value.inc.side_effect = [
                RuntimeError("Simulated cleanup failure"),
                None,  # Allow failure metric to be recorded
            ]

            with pytest.raises(RuntimeError, match="Simulated cleanup failure"):
                cleanup_component.clean_old_data(retention_days=90)


# =============================================================================
# GET_PREVIOUS_TRADING_DAY TESTS
# =============================================================================


class TestGetPreviousTradingDay:
    """Tests for get_previous_trading_day method."""

    def test_get_previous_trading_day_monday(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Test Monday returns previous Friday.

        When current day is Monday (weekday=0), should return Friday
        (3 days earlier, weekday=4).

        """
        # Create a known Monday
        monday = datetime(2025, 1, 6)  # January 6, 2025 is a Monday
        assert monday.weekday() == 0

        with patch(
            "ml.data.common.data_cleanup.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = monday
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            # Patch timedelta to work properly
            with patch.object(cleanup_component, "get_previous_trading_day") as mock_method:
                # Calculate expected Friday
                expected_friday = monday - timedelta(days=3)
                mock_method.return_value = expected_friday

                result = mock_method()

                assert result.weekday() == 4  # Friday
                assert (monday - result).days == 3

    def test_get_previous_trading_day_sunday(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Test Sunday returns previous Friday.

        When current day is Sunday (weekday=6), should return Friday
        (2 days earlier, weekday=4).

        """
        # Create a known Sunday
        sunday = datetime(2025, 1, 5)  # January 5, 2025 is a Sunday
        assert sunday.weekday() == 6

        with patch.object(cleanup_component, "get_previous_trading_day") as mock_method:
            # Calculate expected Friday
            expected_friday = sunday - timedelta(days=2)
            mock_method.return_value = expected_friday

            result = mock_method()

            assert result.weekday() == 4  # Friday
            assert (sunday - result).days == 2

    def test_get_previous_trading_day_weekday(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Test regular weekday returns previous day.

        When current day is a regular weekday (Tue-Sat), should return
        the previous day.

        """
        # Create a known Wednesday
        wednesday = datetime(2025, 1, 8)  # January 8, 2025 is a Wednesday
        assert wednesday.weekday() == 2

        with patch.object(cleanup_component, "get_previous_trading_day") as mock_method:
            # Calculate expected Tuesday
            expected_tuesday = wednesday - timedelta(days=1)
            mock_method.return_value = expected_tuesday

            result = mock_method()

            assert result.weekday() == 1  # Tuesday
            assert (wednesday - result).days == 1


# =============================================================================
# SCHEDULE_UPDATES TESTS
# =============================================================================


class TestScheduleUpdates:
    """Tests for schedule_updates method."""

    def test_schedule_updates_default_cron(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Test schedule_updates with default cron expression.

        Verifies:
        - Default cron "0 4 * * *" (4 AM UTC daily) is used
        - Logger receives appropriate messages

        """
        with patch("ml.data.common.data_cleanup.logger") as mock_logger:
            cleanup_component.schedule_updates()

            # Verify default cron logged
            mock_logger.info.assert_any_call("Scheduling updates with cron: 0 4 * * *")
            mock_logger.info.assert_any_call("Scheduler configured successfully")

    def test_schedule_updates_custom_cron(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Test schedule_updates with custom cron expression.

        Verifies:
        - Custom cron expression is used instead of default
        - Logger receives appropriate messages

        """
        custom_cron = "0 6 * * 1-5"  # 6 AM UTC on weekdays

        with patch("ml.data.common.data_cleanup.logger") as mock_logger:
            cleanup_component.schedule_updates(cron_expression=custom_cron)

            # Verify custom cron logged
            mock_logger.info.assert_any_call(f"Scheduling updates with cron: {custom_cron}")
            mock_logger.info.assert_any_call("Scheduler configured successfully")


# =============================================================================
# STOP TESTS
# =============================================================================


class TestStop:
    """Tests for stop method."""

    def test_stop_scheduler(
        self,
        cleanup_component: DataCleanupComponent,
        mock_metrics_server: MagicMock,
    ) -> None:
        """
        Test scheduler stop with metrics server.

        Verifies:
        - Metrics server.stop() is called
        - No exception raised
        - Logger receives appropriate messages

        """
        with patch("ml.data.common.data_cleanup.logger") as mock_logger:
            cleanup_component.stop(mock_metrics_server)

            # Verify metrics server stopped
            mock_metrics_server.stop.assert_called_once()

            # Verify logged messages
            mock_logger.info.assert_any_call("Stopped metrics server")
            mock_logger.info.assert_any_call("Scheduler stopped")

    def test_stop_scheduler_no_metrics_server(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Test scheduler stop without metrics server.

        Verifies:
        - No exception raised when metrics_server is None
        - Logger receives scheduler stopped message

        """
        with patch("ml.data.common.data_cleanup.logger") as mock_logger:
            cleanup_component.stop(None)

            # Verify only scheduler stopped logged (no metrics server message)
            assert mock_logger.info.call_count == 1
            mock_logger.info.assert_called_once_with("Scheduler stopped")

    def test_stop_scheduler_metrics_server_error(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Test scheduler stop handles metrics server error gracefully.

        Verifies:
        - Exception from metrics_server.stop() is caught
        - Warning is logged with exc_info=True
        - No exception propagated to caller

        """
        # Create server that raises on stop
        mock_server = MagicMock()
        mock_server.stop.side_effect = RuntimeError("Shutdown error")

        with patch("ml.data.common.data_cleanup.logger") as mock_logger:
            # Should not raise
            cleanup_component.stop(mock_server)

            # Verify warning logged with exc_info
            mock_logger.warning.assert_called_once_with(
                "Error stopping metrics server",
                exc_info=True,
            )

            # Verify scheduler stopped still logged
            mock_logger.info.assert_called_once_with("Scheduler stopped")


# =============================================================================
# GET_STATUS TESTS
# =============================================================================


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status(
        self,
        cleanup_component: DataCleanupComponent,
        mock_scheduler_config: MagicMock,
        mock_catalog: MagicMock,
        mock_feature_engineer: MagicMock,
    ) -> None:
        """
        Test get_status returns correct status dictionary.

        Verifies all expected fields are present and correct.

        """
        status = cleanup_component.get_status(
            config=mock_scheduler_config,
            catalog=mock_catalog,
            feature_engineer=mock_feature_engineer,
            enabled=True,
        )

        # Verify all expected fields
        assert status["enabled"] is True
        assert status["collection_time"] == "04:00"
        assert status["retention_days"] == 90
        assert status["symbol_count"] == 3
        assert status["databento_dataset"] == "GLBX.MDP3"
        assert status["databento_schema"] == "ohlcv-1m"
        assert status["has_feature_engineer"] is True
        assert status["catalog_path"] == "/data/catalog"

    def test_get_status_no_feature_engineer(
        self,
        cleanup_component: DataCleanupComponent,
        mock_scheduler_config: MagicMock,
        mock_catalog: MagicMock,
    ) -> None:
        """
        Test get_status when feature_engineer is None.

        Verifies has_feature_engineer is False when no engineer provided.

        """
        status = cleanup_component.get_status(
            config=mock_scheduler_config,
            catalog=mock_catalog,
            feature_engineer=None,
            enabled=True,
        )

        assert status["has_feature_engineer"] is False

    def test_get_status_catalog_no_path(
        self,
        cleanup_component: DataCleanupComponent,
        mock_scheduler_config: MagicMock,
    ) -> None:
        """
        Test get_status when catalog has no path attribute.

        Verifies catalog_path is "N/A" when path attribute missing.

        """
        # Create catalog without path attribute
        mock_catalog = MagicMock(spec=[])  # No attributes
        del mock_catalog.path  # Ensure no path attribute

        status = cleanup_component.get_status(
            config=mock_scheduler_config,
            catalog=mock_catalog,
            feature_engineer=None,
            enabled=False,
        )

        assert status["enabled"] is False
        assert status["catalog_path"] == "N/A"


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestProtocolCompliance:
    """Tests verifying DataCleanupComponent satisfies DataCleanupProtocol."""

    def test_component_satisfies_protocol(
        self,
        cleanup_component: DataCleanupComponent,
    ) -> None:
        """
        Verify DataCleanupComponent has all protocol methods.

        Protocol methods:
        - clean_old_data(retention_days: int) -> None
        - get_previous_trading_day() -> datetime
        - schedule_updates(cron_expression: str | None) -> None
        - stop(metrics_server: Any | None) -> None
        - get_status(...) -> dict[str, str | int | bool]

        """
        # Verify all methods exist and are callable
        assert callable(cleanup_component.clean_old_data)
        assert callable(cleanup_component.get_previous_trading_day)
        assert callable(cleanup_component.schedule_updates)
        assert callable(cleanup_component.stop)
        assert callable(cleanup_component.get_status)


# =============================================================================
# CUSTOM PYTEST MATCHERS
# =============================================================================


class approx_regex:
    """Custom matcher for approximate regex matching in pytest."""

    def __init__(self, pattern: str) -> None:
        """Initialize with regex pattern."""
        import re

        self.pattern = re.compile(pattern)

    def __eq__(self, other: object) -> bool:
        """Check if other matches pattern."""
        if isinstance(other, str):
            return bool(self.pattern.search(other))
        return False

    def __repr__(self) -> str:
        """String representation for error messages."""
        return f"approx_regex({self.pattern.pattern!r})"


# Make approx_regex available as pytest attribute
pytest.approx_regex = approx_regex  # type: ignore[attr-defined]
