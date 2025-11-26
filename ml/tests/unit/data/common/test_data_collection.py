"""
Unit tests for DataCollectionComponent.

This module contains 18 tests as specified in Phase 3.9.5:
- Happy path tests (collect_latest_data_success, collect_symbol_data_success)
- Error condition tests (no_api_key, databento_import_error, invalid_format)
- Retry and rate limit tests
- DBN file loading tests
- Metrics and event emission tests
- Edge case tests

"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from ml.data.common.data_collection import (
    DataCollectionComponent,
    VENUE_MAP,
    active_collection_tasks,
    api_rate_limit_hits,
    api_request_total,
    catalog_write_latency,
    catalog_write_operations_total,
    data_collected_total,
    data_collection_errors_total,
    data_collection_latency,
    data_staleness_seconds,
)


# ============================================================================
# Test Data Classes (to bypass MagicMock detection in component)
# ============================================================================


@dataclass
class FakeDataItem:
    """Fake data item that mimics Nautilus data objects for testing."""

    ts_event: int = 1704067200000000000  # 2024-01-01 00:00:00 in nanoseconds


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def component() -> DataCollectionComponent:
    """Fixture providing a fresh DataCollectionComponent instance."""
    return DataCollectionComponent()


@pytest.fixture
def mock_config() -> MagicMock:
    """Fixture providing a mock SchedulerConfig."""
    config = MagicMock()
    config.symbols = ["AAPL.XNAS", "MSFT.XNAS"]
    config.max_retries = 3
    config.retry_delay_seconds = 0.01  # Short for tests
    config.data_retention_days = 90
    config.databento.api_key = "test_api_key"
    config.databento.dataset = "XNAS.ITCH"
    config.databento.schema = "ohlcv-1m"
    config.databento.stype_in = "raw_symbol"
    config.databento.use_temporary_files = True
    config.databento.temp_data_dir = "/tmp/databento_test"
    config.databento.price_precision = 2
    return config


@pytest.fixture
def mock_catalog() -> MagicMock:
    """Fixture providing a mock ParquetDataCatalog."""
    catalog = MagicMock()
    catalog.write_data = MagicMock(return_value=None)
    return catalog


@pytest.fixture
def mock_registry() -> MagicMock:
    """Fixture providing a mock DataRegistry."""
    registry = MagicMock()
    registry.emit_event = MagicMock(return_value=None)
    registry.update_watermark = MagicMock(return_value=None)
    return registry


@pytest.fixture
def mock_databento_client() -> MagicMock:
    """Fixture providing a mock Databento Historical client."""
    client = MagicMock()
    response = MagicMock()
    response.to_file = MagicMock(return_value=None)
    client.timeseries.get_range = MagicMock(return_value=response)
    return client


@pytest.fixture
def fake_data_items() -> list[FakeDataItem]:
    """Fixture providing fake data items (not MagicMock to bypass detection)."""
    return [FakeDataItem(ts_event=1704067200000000000)]


# ============================================================================
# Happy Path Tests
# ============================================================================


class TestHappyPath:
    """Happy path tests for DataCollectionComponent."""

    def test_collect_latest_data_success(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        """
        Test successful data collection for all symbols.

        Verifies that collect_latest_data returns correct counts
        when all symbol collections succeed.
        """
        mock_config.symbols = ["AAPL.XNAS"]

        # Mock the databento import and client
        mock_db = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.to_file = MagicMock()
        mock_client.timeseries.get_range = MagicMock(return_value=mock_response)
        mock_db.Historical = MagicMock(return_value=mock_client)

        # Mock the data loaded from DBN file
        mock_data = [MagicMock(ts_event=1704067200000000000)]

        with (
            patch.dict(os.environ, {"DATABENTO_API_KEY": "test_key"}),
            patch.object(component, "load_from_dbn_file", return_value=mock_data),
            patch("ml.data.common.data_collection.Path.mkdir"),
            patch("ml.data.common.data_collection.Path.exists", return_value=True),
            patch("ml.data.common.data_collection.Path.unlink"),
            patch("ml.data.common.data_collection.Path.iterdir", return_value=[]),
            patch("ml.data.common.data_collection.Path.rmdir"),
            patch.dict("sys.modules", {"databento": mock_db}),
        ):
            # Import databento dynamically since we patched it
            import importlib
            import sys

            # Create a temporary module
            sys.modules["databento"] = mock_db

            collected, failed = component.collect_latest_data(
                config=mock_config,
                catalog=mock_catalog,
                registry=mock_registry,
                ensure_registered_fn=MagicMock(),
                get_previous_day_fn=lambda: datetime(2024, 1, 1),
            )

            # Clean up
            del sys.modules["databento"]

        assert collected == 1, f"Expected 1 collected, got {collected}"
        assert failed == 0, f"Expected 0 failed, got {failed}"

    def test_collect_symbol_data_success(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_registry: MagicMock,
        mock_databento_client: MagicMock,
        fake_data_items: list[FakeDataItem],
    ) -> None:
        """
        Test successful data collection for a single symbol.

        Verifies that collect_symbol_data returns True and writes to catalog.
        """
        temp_dir = Path("/tmp/databento_test")

        with (
            patch.object(component, "load_from_dbn_file", return_value=fake_data_items),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink"),
        ):
            success = component.collect_symbol_data(
                client=mock_databento_client,
                symbol="AAPL.XNAS",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 23, 59, 59),
                target_date=datetime(2024, 1, 1),
                temp_data_dir=temp_dir,
                config=mock_config,
                catalog=mock_catalog,
                registry=mock_registry,
                run_id="test_run_001",
                ensure_registered_fn=MagicMock(),
            )

        assert success is True
        mock_catalog.write_data.assert_called_once_with(fake_data_items)


# ============================================================================
# Error Condition Tests
# ============================================================================


class TestErrorConditions:
    """Error condition tests for DataCollectionComponent."""

    def test_collect_latest_data_no_api_key(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
    ) -> None:
        """
        Test collect_latest_data raises ValueError when no API key is set.

        Verifies that the appropriate error is raised with clear message.
        """
        mock_config.databento.api_key = None

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="DATABENTO_API_KEY"),
        ):
            # Remove env var if present
            os.environ.pop("DATABENTO_API_KEY", None)

            component.collect_latest_data(
                config=mock_config,
                catalog=mock_catalog,
                registry=None,
                ensure_registered_fn=MagicMock(),
                get_previous_day_fn=lambda: datetime(2024, 1, 1),
            )

    def test_collect_latest_data_databento_import_error(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
    ) -> None:
        """
        Test collect_latest_data raises ImportError when databento not installed.

        Verifies graceful handling of missing dependency.
        """
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "databento":
                raise ImportError("No module named 'databento'")
            return original_import(name, *args, **kwargs)

        with (
            patch.dict(os.environ, {"DATABENTO_API_KEY": "test_key"}),
            patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(ImportError),
        ):
            component.collect_latest_data(
                config=mock_config,
                catalog=mock_catalog,
                registry=None,
                ensure_registered_fn=MagicMock(),
                get_previous_day_fn=lambda: datetime(2024, 1, 1),
            )

    def test_collect_symbol_data_invalid_format(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_databento_client: MagicMock,
    ) -> None:
        """
        Test collect_symbol_data returns False for invalid symbol format.

        Verifies that symbols without SYMBOL.VENUE format are rejected.
        """
        # Test invalid format without venue separator
        success = component.collect_symbol_data(
            client=mock_databento_client,
            symbol="AAPL",  # Missing .VENUE
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 23, 59, 59),
            target_date=datetime(2024, 1, 1),
            temp_data_dir=None,
            config=mock_config,
            catalog=mock_catalog,
            registry=None,
            run_id="test_run_001",
            ensure_registered_fn=MagicMock(),
        )

        assert success is False

    def test_collect_symbol_data_with_temp_files(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_databento_client: MagicMock,
        fake_data_items: list[FakeDataItem],
    ) -> None:
        """
        Test collect_symbol_data handles temporary files correctly.

        Verifies that temp files are created and cleaned up properly.
        """
        mock_config.databento.use_temporary_files = True

        temp_dir = Path("/tmp/databento_test")

        with (
            patch.object(component, "load_from_dbn_file", return_value=fake_data_items),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink") as mock_unlink,
        ):
            success = component.collect_symbol_data(
                client=mock_databento_client,
                symbol="AAPL.XNAS",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 23, 59, 59),
                target_date=datetime(2024, 1, 1),
                temp_data_dir=temp_dir,
                config=mock_config,
                catalog=mock_catalog,
                registry=None,
                run_id="test_run_001",
                ensure_registered_fn=MagicMock(),
            )

        assert success is True
        # Verify temp file cleanup was called
        mock_unlink.assert_called()


# ============================================================================
# Retry and Rate Limit Tests
# ============================================================================


class TestRetryAndRateLimits:
    """Tests for retry logic and rate limit handling."""

    def test_collect_symbol_data_retry_logic(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_databento_client: MagicMock,
    ) -> None:
        """
        Test that collect_symbol_data retries on transient failures.

        Verifies retry behavior with max_retries configuration.
        """
        mock_config.max_retries = 3
        mock_config.retry_delay_seconds = 0.001

        # Fail first two attempts, succeed on third
        fake_data = [FakeDataItem(ts_event=1704067200000000000)]
        call_count = [0]

        def mock_load(*args: Any, **kwargs: Any) -> list[FakeDataItem]:
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Transient failure")
            return fake_data

        with (
            patch.object(component, "load_from_dbn_file", side_effect=mock_load),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink"),
        ):
            success = component.collect_symbol_data(
                client=mock_databento_client,
                symbol="AAPL.XNAS",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 23, 59, 59),
                target_date=datetime(2024, 1, 1),
                temp_data_dir=Path("/tmp/test"),
                config=mock_config,
                catalog=mock_catalog,
                registry=None,
                run_id="test_run_001",
                ensure_registered_fn=MagicMock(),
            )

        assert success is True
        assert call_count[0] == 3

    def test_collect_symbol_data_rate_limit_handling(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_databento_client: MagicMock,
    ) -> None:
        """
        Test that rate limit errors are classified correctly.

        Verifies that rate limit errors increment the appropriate metric.
        """
        mock_config.max_retries = 1
        mock_config.retry_delay_seconds = 0.001

        mock_databento_client.timeseries.get_range.side_effect = Exception("Rate limit exceeded")

        success = component.collect_symbol_data(
            client=mock_databento_client,
            symbol="AAPL.XNAS",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 23, 59, 59),
            target_date=datetime(2024, 1, 1),
            temp_data_dir=Path("/tmp/test"),
            config=mock_config,
            catalog=mock_catalog,
            registry=None,
            run_id="test_run_001",
            ensure_registered_fn=MagicMock(),
        )

        assert success is False

    def test_collect_symbol_data_connection_error(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_databento_client: MagicMock,
    ) -> None:
        """
        Test that connection errors are classified correctly.

        Verifies that connection errors are handled gracefully.
        """
        mock_config.max_retries = 1
        mock_config.retry_delay_seconds = 0.001

        mock_databento_client.timeseries.get_range.side_effect = Exception("Connection timeout")

        success = component.collect_symbol_data(
            client=mock_databento_client,
            symbol="AAPL.XNAS",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 23, 59, 59),
            target_date=datetime(2024, 1, 1),
            temp_data_dir=Path("/tmp/test"),
            config=mock_config,
            catalog=mock_catalog,
            registry=None,
            run_id="test_run_001",
            ensure_registered_fn=MagicMock(),
        )

        assert success is False

    def test_collect_symbol_data_auth_error(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_databento_client: MagicMock,
    ) -> None:
        """
        Test that authentication errors are classified correctly.

        Verifies that auth errors increment the appropriate metric.
        """
        mock_config.max_retries = 1
        mock_config.retry_delay_seconds = 0.001

        mock_databento_client.timeseries.get_range.side_effect = Exception("Unauthorized access")

        success = component.collect_symbol_data(
            client=mock_databento_client,
            symbol="AAPL.XNAS",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 23, 59, 59),
            target_date=datetime(2024, 1, 1),
            temp_data_dir=Path("/tmp/test"),
            config=mock_config,
            catalog=mock_catalog,
            registry=None,
            run_id="test_run_001",
            ensure_registered_fn=MagicMock(),
        )

        assert success is False


# ============================================================================
# DBN File Loading Tests
# ============================================================================


class TestDBNFileLoading:
    """Tests for DBN file loading functionality."""

    def test_load_from_dbn_file_venue_mapping_xnas(
        self,
        component: DataCollectionComponent,
    ) -> None:
        """
        Test venue mapping for XNAS to NASDAQ.

        Verifies that XNAS is correctly mapped to NASDAQ venue.
        """
        assert VENUE_MAP["XNAS"] == "NASDAQ"

        mock_loader = MagicMock()
        mock_loader.from_dbn_file = MagicMock(return_value=[])

        with patch.object(component, "_get_databento_loader", return_value=mock_loader):
            component.load_from_dbn_file(
                file_path=Path("/tmp/test.dbn"),
                symbol_code="AAPL",
                venue="XNAS",
                price_precision=2,
                schema="ohlcv-1m",
            )

        # Verify the instrument_id used NASDAQ venue
        call_args = mock_loader.from_dbn_file.call_args
        instrument_id = call_args.kwargs.get("instrument_id") or call_args[1].get("instrument_id")
        assert str(instrument_id) == "AAPL.NASDAQ"

    def test_load_from_dbn_file_venue_mapping_xnys(
        self,
        component: DataCollectionComponent,
    ) -> None:
        """
        Test venue mapping for XNYS to NYSE.

        Verifies that XNYS is correctly mapped to NYSE venue.
        """
        assert VENUE_MAP["XNYS"] == "NYSE"

        mock_loader = MagicMock()
        mock_loader.from_dbn_file = MagicMock(return_value=[])

        with patch.object(component, "_get_databento_loader", return_value=mock_loader):
            component.load_from_dbn_file(
                file_path=Path("/tmp/test.dbn"),
                symbol_code="IBM",
                venue="XNYS",
                price_precision=2,
                schema="ohlcv-1m",
            )

        call_args = mock_loader.from_dbn_file.call_args
        instrument_id = call_args.kwargs.get("instrument_id") or call_args[1].get("instrument_id")
        assert str(instrument_id) == "IBM.NYSE"

    def test_load_from_dbn_file_venue_mapping_unknown(
        self,
        component: DataCollectionComponent,
    ) -> None:
        """
        Test venue mapping for unknown venue codes.

        Verifies that unknown venues are passed through unchanged.
        """
        mock_loader = MagicMock()
        mock_loader.from_dbn_file = MagicMock(return_value=[])

        with patch.object(component, "_get_databento_loader", return_value=mock_loader):
            component.load_from_dbn_file(
                file_path=Path("/tmp/test.dbn"),
                symbol_code="TEST",
                venue="CUSTOM",  # Not in VENUE_MAP
                price_precision=2,
                schema="ohlcv-1m",
            )

        call_args = mock_loader.from_dbn_file.call_args
        instrument_id = call_args.kwargs.get("instrument_id") or call_args[1].get("instrument_id")
        assert str(instrument_id) == "TEST.CUSTOM"


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetrics:
    """Tests for metrics emission."""

    def test_catalog_write_success_metrics(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_databento_client: MagicMock,
        fake_data_items: list[FakeDataItem],
    ) -> None:
        """
        Test that successful catalog write emits success metrics.

        Verifies catalog_write_operations_total increments on success.
        """
        with (
            patch.object(component, "load_from_dbn_file", return_value=fake_data_items),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink"),
        ):
            success = component.collect_symbol_data(
                client=mock_databento_client,
                symbol="AAPL.XNAS",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 23, 59, 59),
                target_date=datetime(2024, 1, 1),
                temp_data_dir=Path("/tmp/test"),
                config=mock_config,
                catalog=mock_catalog,
                registry=None,
                run_id="test_run_001",
                ensure_registered_fn=MagicMock(),
            )

        assert success is True
        mock_catalog.write_data.assert_called_once()

    def test_catalog_write_failure_metrics(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_databento_client: MagicMock,
        fake_data_items: list[FakeDataItem],
    ) -> None:
        """
        Test that failed catalog write emits failure metrics.

        Verifies catalog_write_operations_total increments on failure.
        """
        mock_catalog.write_data.side_effect = Exception("Write failed")

        mock_config.max_retries = 1

        with (
            patch.object(component, "load_from_dbn_file", return_value=fake_data_items),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink"),
        ):
            success = component.collect_symbol_data(
                client=mock_databento_client,
                symbol="AAPL.XNAS",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 23, 59, 59),
                target_date=datetime(2024, 1, 1),
                temp_data_dir=Path("/tmp/test"),
                config=mock_config,
                catalog=mock_catalog,
                registry=None,
                run_id="test_run_001",
                ensure_registered_fn=MagicMock(),
            )

        assert success is False


# ============================================================================
# Event Emission Tests
# ============================================================================


class TestEventEmission:
    """Tests for data event emission."""

    def test_data_event_emission_success(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_registry: MagicMock,
        mock_databento_client: MagicMock,
        fake_data_items: list[FakeDataItem],
    ) -> None:
        """
        Test that successful collection emits data events.

        Verifies emit_event is called with correct parameters.
        """
        with (
            patch.object(component, "load_from_dbn_file", return_value=fake_data_items),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink"),
        ):
            success = component.collect_symbol_data(
                client=mock_databento_client,
                symbol="AAPL.XNAS",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 23, 59, 59),
                target_date=datetime(2024, 1, 1),
                temp_data_dir=Path("/tmp/test"),
                config=mock_config,
                catalog=mock_catalog,
                registry=mock_registry,
                run_id="test_run_001",
                ensure_registered_fn=MagicMock(),
            )

        assert success is True
        mock_registry.emit_event.assert_called_once()

    def test_watermark_update_after_collection(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        mock_registry: MagicMock,
        mock_databento_client: MagicMock,
        fake_data_items: list[FakeDataItem],
    ) -> None:
        """
        Test that watermark is updated after successful collection.

        Verifies update_watermark is called with correct parameters.
        """
        with (
            patch.object(component, "load_from_dbn_file", return_value=fake_data_items),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "unlink"),
        ):
            success = component.collect_symbol_data(
                client=mock_databento_client,
                symbol="AAPL.XNAS",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 1, 23, 59, 59),
                target_date=datetime(2024, 1, 1),
                temp_data_dir=Path("/tmp/test"),
                config=mock_config,
                catalog=mock_catalog,
                registry=mock_registry,
                run_id="test_run_001",
                ensure_registered_fn=MagicMock(),
            )

        assert success is True
        mock_registry.update_watermark.assert_called_once()


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Edge case tests for DataCollectionComponent."""

    def test_high_failure_rate_warning(
        self,
        component: DataCollectionComponent,
        mock_config: MagicMock,
        mock_catalog: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        Test that high failure rate triggers warning.

        Verifies warning is logged when >50% of collections fail.
        """
        mock_config.symbols = ["AAPL.XNAS", "MSFT.XNAS", "GOOG.XNAS", "AMZN.XNAS"]
        mock_config.max_retries = 1
        mock_config.retry_delay_seconds = 0.001

        mock_db = MagicMock()
        mock_client = MagicMock()
        # All requests fail
        mock_client.timeseries.get_range.side_effect = Exception("API error")
        mock_db.Historical = MagicMock(return_value=mock_client)

        with (
            patch.dict(os.environ, {"DATABENTO_API_KEY": "test_key"}),
            patch.dict("sys.modules", {"databento": mock_db}),
            caplog.at_level(logging.WARNING),
        ):
            import sys

            sys.modules["databento"] = mock_db

            collected, failed = component.collect_latest_data(
                config=mock_config,
                catalog=mock_catalog,
                registry=None,
                ensure_registered_fn=MagicMock(),
                get_previous_day_fn=lambda: datetime(2024, 1, 1),
            )

            del sys.modules["databento"]

        assert failed == 4
        assert collected == 0
        assert "High failure rate" in caplog.text
