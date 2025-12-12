"""
Unit tests for DataSchedulerFacade.

Tests verify that the facade correctly:
- Initializes all 8 components
- Delegates methods to appropriate components
- Implements the feature flag correctly
- Provides factory function for scheduler creation

"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest


if TYPE_CHECKING:
    from ml.config.scheduler_config import SchedulerConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_catalog() -> MagicMock:
    """Create a mock ParquetDataCatalog."""
    catalog = MagicMock()
    catalog.path = Path("/data/catalog")
    return catalog


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock SchedulerConfig."""
    config = MagicMock()
    config.symbols = ["SPY.XNAS", "QQQ.XNAS"]
    config.retention_days = 90
    config.feature_store_enabled = False
    config.collection_time = "04:00"
    config.databento = MagicMock()
    config.databento.dataset = "GLBX.MDP3"
    config.databento.schema = "ohlcv-1m"
    config.databento.api_key = None
    config.databento.use_temporary_files = True
    config.databento.temp_data_dir = "/tmp/databento"
    config.databento.price_precision = 2
    config.databento.stype_in = "raw_symbol"
    config.max_retries = 3
    config.retry_delay_seconds = 5
    config.market_inputs = None
    config.market_dataset_id = None
    return config


@pytest.fixture
def mock_feature_engineer() -> MagicMock:
    """Create a mock FeatureEngineer."""
    engineer = MagicMock()
    engineer.config = MagicMock()
    return engineer


# =============================================================================
# TEST: FACADE INITIALIZATION
# =============================================================================


@patch("ml.data.scheduler_facade.DatabentoDataLoader")
@patch("ml.data.scheduler_facade.SchedulerInitComponent")
@patch("ml.data.scheduler_facade.MetricsServerComponent")
@patch("ml.data.scheduler_facade.HAS_PROMETHEUS", False)
def test_facade_init_creates_components(
    mock_metrics_cls: MagicMock,
    mock_init_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_catalog: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Test that facade initialization creates all required components."""
    from ml.data.scheduler_facade import DataSchedulerFacade

    # Setup mocks
    mock_init = mock_init_cls.return_value
    mock_init.resolve_connection.return_value = None
    mock_init.init_data_registry.return_value = None
    mock_init.init_feature_store.return_value = None

    # Create facade
    facade = DataSchedulerFacade(
        catalog=mock_catalog,
        config=mock_config,
        start_metrics_server=False,
    )

    # Verify all components are created
    assert facade._init_component is not None
    assert facade._cleanup_component is not None
    assert facade._metrics_component is not None
    assert facade._daily_update_component is not None
    assert facade._registration_component is not None
    assert facade._feature_computation_component is not None
    assert facade._orchestrator_component is not None
    assert facade._data_collection_component is not None


@patch("ml.data.scheduler_facade.DatabentoDataLoader")
@patch("ml.data.scheduler_facade.SchedulerInitComponent")
@patch("ml.data.scheduler_facade.HAS_PROMETHEUS", False)
def test_facade_init_stores_public_attributes(
    mock_init_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_catalog: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Test that facade stores all public attributes matching legacy."""
    from ml.data.scheduler_facade import DataSchedulerFacade

    # Setup mocks
    mock_init = mock_init_cls.return_value
    mock_init.resolve_connection.return_value = "postgresql://test"
    mock_init.init_data_registry.return_value = MagicMock()
    mock_init.init_feature_store.return_value = None

    # Create facade
    facade = DataSchedulerFacade(
        catalog=mock_catalog,
        config=mock_config,
        use_orchestrator=True,
        dual_write=True,
        start_metrics_server=False,
    )

    # Verify public attributes
    assert facade.catalog is mock_catalog
    assert facade.config is mock_config
    assert facade.collector is not None
    assert facade.feature_engineer is None
    assert facade.enabled is True
    assert facade._use_orchestrator is True
    assert facade._dual_write is True
    assert facade._databento_loader is not None
    assert facade._current_run_id == ""
    assert facade._feature_store_connection == "postgresql://test"


# =============================================================================
# TEST: DELEGATION TO COMPONENTS
# =============================================================================


@patch("ml.data.scheduler_facade.DatabentoDataLoader")
@patch("ml.data.scheduler_facade.SchedulerInitComponent")
@patch("ml.data.scheduler_facade.HAS_PROMETHEUS", False)
def test_facade_run_daily_update_delegates(
    mock_init_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_catalog: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Test that run_daily_update delegates to DailyUpdateOrchestratorComponent."""
    from ml.data.scheduler_facade import DataSchedulerFacade

    # Setup mocks
    mock_init = mock_init_cls.return_value
    mock_init.resolve_connection.return_value = None
    mock_init.init_data_registry.return_value = None
    mock_init.init_feature_store.return_value = None

    # Create facade
    facade = DataSchedulerFacade(
        catalog=mock_catalog,
        config=mock_config,
        start_metrics_server=False,
    )

    # Mock the daily update component
    mock_daily_update = MagicMock()
    facade._daily_update_component = mock_daily_update

    # Call method
    facade.run_daily_update()

    # Verify delegation
    mock_daily_update.run_daily_update.assert_called_once()
    call_kwargs = mock_daily_update.run_daily_update.call_args
    assert call_kwargs.kwargs["use_orchestrator"] is False
    assert call_kwargs.kwargs["feature_engineer"] is None


@patch("ml.data.scheduler_facade.DatabentoDataLoader")
@patch("ml.data.scheduler_facade.SchedulerInitComponent")
@patch("ml.data.scheduler_facade.HAS_PROMETHEUS", False)
def test_facade_get_status_delegates(
    mock_init_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_catalog: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Test that get_status delegates to DataCleanupComponent."""
    from ml.data.scheduler_facade import DataSchedulerFacade

    # Setup mocks
    mock_init = mock_init_cls.return_value
    mock_init.resolve_connection.return_value = None
    mock_init.init_data_registry.return_value = None
    mock_init.init_feature_store.return_value = None

    # Create facade
    facade = DataSchedulerFacade(
        catalog=mock_catalog,
        config=mock_config,
        start_metrics_server=False,
    )

    # Mock the cleanup component
    mock_cleanup = MagicMock()
    expected_status = {
        "enabled": True,
        "collection_time": "04:00",
        "retention_days": 90,
        "symbol_count": 2,
    }
    mock_cleanup.get_status.return_value = expected_status
    facade._cleanup_component = mock_cleanup

    # Call method
    status = facade.get_status()

    # Verify delegation
    mock_cleanup.get_status.assert_called_once_with(
        config=mock_config,
        catalog=mock_catalog,
        feature_engineer=None,
        enabled=True,
    )
    assert status == expected_status


@patch("ml.data.scheduler_facade.DatabentoDataLoader")
@patch("ml.data.scheduler_facade.SchedulerInitComponent")
@patch("ml.data.scheduler_facade.HAS_PROMETHEUS", False)
def test_facade_stop_delegates(
    mock_init_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_catalog: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Test that stop delegates to DataCleanupComponent."""
    from ml.data.scheduler_facade import DataSchedulerFacade

    # Setup mocks
    mock_init = mock_init_cls.return_value
    mock_init.resolve_connection.return_value = None
    mock_init.init_data_registry.return_value = None
    mock_init.init_feature_store.return_value = None

    # Create facade
    facade = DataSchedulerFacade(
        catalog=mock_catalog,
        config=mock_config,
        start_metrics_server=False,
    )

    # Mock the cleanup component
    mock_cleanup = MagicMock()
    facade._cleanup_component = mock_cleanup

    # Call method
    facade.stop()

    # Verify delegation
    mock_cleanup.stop.assert_called_once_with(metrics_server=None)
    assert facade.enabled is False


@patch("ml.data.scheduler_facade.DatabentoDataLoader")
@patch("ml.data.scheduler_facade.SchedulerInitComponent")
@patch("ml.data.scheduler_facade.HAS_PROMETHEUS", False)
def test_facade_schedule_updates_delegates(
    mock_init_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_catalog: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Test that schedule_updates delegates to DataCleanupComponent."""
    from ml.data.scheduler_facade import DataSchedulerFacade

    # Setup mocks
    mock_init = mock_init_cls.return_value
    mock_init.resolve_connection.return_value = None
    mock_init.init_data_registry.return_value = None
    mock_init.init_feature_store.return_value = None

    # Create facade
    facade = DataSchedulerFacade(
        catalog=mock_catalog,
        config=mock_config,
        start_metrics_server=False,
    )

    # Mock the cleanup component
    mock_cleanup = MagicMock()
    facade._cleanup_component = mock_cleanup

    # Call method with custom cron
    facade.schedule_updates(cron_expression="0 6 * * *")

    # Verify delegation
    mock_cleanup.schedule_updates.assert_called_once_with(cron_expression="0 6 * * *")


# =============================================================================
# TEST: FEATURE FLAG
# =============================================================================


def test_use_legacy_scheduler_returns_true() -> None:
    """Test that use_legacy_scheduler returns True when env var is '1'."""
    from ml.data.scheduler_facade import use_legacy_scheduler

    with patch.dict(os.environ, {"ML_USE_LEGACY_SCHEDULER": "1"}):
        assert use_legacy_scheduler() is True


def test_use_legacy_scheduler_returns_false() -> None:
    """Test that use_legacy_scheduler returns False when env var is not '1'."""
    from ml.data.scheduler_facade import use_legacy_scheduler

    # Test with "0"
    with patch.dict(os.environ, {"ML_USE_LEGACY_SCHEDULER": "0"}):
        assert use_legacy_scheduler() is False

    # Test with unset
    with patch.dict(os.environ, {}, clear=True):
        # Remove if exists
        os.environ.pop("ML_USE_LEGACY_SCHEDULER", None)
        assert use_legacy_scheduler() is False


def test_use_legacy_scheduler_returns_false_for_other_values() -> None:
    """Test that use_legacy_scheduler returns False for non-'1' values."""
    from ml.data.scheduler_facade import use_legacy_scheduler

    # Test with other values
    for value in ["true", "yes", "false", "no", "2", ""]:
        with patch.dict(os.environ, {"ML_USE_LEGACY_SCHEDULER": value}):
            assert use_legacy_scheduler() is False


# =============================================================================
# TEST: FACTORY FUNCTION
# =============================================================================


@patch("ml.data.scheduler_facade.DatabentoDataLoader")
@patch("ml.data.scheduler_facade.SchedulerInitComponent")
@patch("ml.data.scheduler_facade.HAS_PROMETHEUS", False)
def test_create_data_scheduler_factory(
    mock_init_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_catalog: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Test that create_data_scheduler factory returns correct type based on flag."""
    from ml.data.scheduler import DataScheduler
    from ml.data.scheduler_facade import DataSchedulerFacade, create_data_scheduler

    # Setup mocks
    mock_init = mock_init_cls.return_value
    mock_init.resolve_connection.return_value = None
    mock_init.init_data_registry.return_value = None
    mock_init.init_feature_store.return_value = None

    # Test facade mode (default)
    with patch.dict(os.environ, {"ML_USE_LEGACY_SCHEDULER": "0"}):
        scheduler = create_data_scheduler(
            catalog=mock_catalog,
            config=mock_config,
            start_metrics_server=False,
        )
        assert isinstance(scheduler, DataSchedulerFacade)

    # Test legacy mode
    with patch.dict(os.environ, {"ML_USE_LEGACY_SCHEDULER": "1"}):
        # Patch the actual scheduler module since factory imports from there
        with patch("ml.data.scheduler.DataScheduler") as mock_ds_cls:
            mock_instance = MagicMock(spec=DataScheduler)
            mock_ds_cls.return_value = mock_instance
            scheduler = create_data_scheduler(
                catalog=mock_catalog,
                config=mock_config,
                start_metrics_server=False,
            )
            mock_ds_cls.assert_called_once()
            assert scheduler is mock_instance


# =============================================================================
# TEST: INTERNAL METHOD DELEGATION
# =============================================================================


@patch("ml.data.scheduler_facade.DatabentoDataLoader")
@patch("ml.data.scheduler_facade.SchedulerInitComponent")
@patch("ml.data.scheduler_facade.HAS_PROMETHEUS", False)
def test_facade_get_previous_trading_day_delegates(
    mock_init_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_catalog: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Test that _get_previous_trading_day delegates correctly."""
    from ml.data.scheduler_facade import DataSchedulerFacade

    # Setup mocks
    mock_init = mock_init_cls.return_value
    mock_init.resolve_connection.return_value = None
    mock_init.init_data_registry.return_value = None
    mock_init.init_feature_store.return_value = None

    # Create facade
    facade = DataSchedulerFacade(
        catalog=mock_catalog,
        config=mock_config,
        start_metrics_server=False,
    )

    # Mock the cleanup component
    mock_cleanup = MagicMock()
    expected_date = datetime(2024, 1, 15)
    mock_cleanup.get_previous_trading_day.return_value = expected_date
    facade._cleanup_component = mock_cleanup

    # Call method
    result = facade._get_previous_trading_day()

    # Verify delegation
    mock_cleanup.get_previous_trading_day.assert_called_once()
    assert result == expected_date


@patch("ml.data.scheduler_facade.DatabentoDataLoader")
@patch("ml.data.scheduler_facade.SchedulerInitComponent")
@patch("ml.data.scheduler_facade.HAS_PROMETHEUS", False)
def test_facade_ensure_dataset_registered_delegates(
    mock_init_cls: MagicMock,
    mock_loader_cls: MagicMock,
    mock_catalog: MagicMock,
    mock_config: MagicMock,
) -> None:
    """Test that _ensure_dataset_registered delegates correctly."""
    from ml.data.scheduler_facade import DataSchedulerFacade

    # Setup mocks
    mock_init = mock_init_cls.return_value
    mock_init.resolve_connection.return_value = None
    mock_init.init_data_registry.return_value = MagicMock()
    mock_init.init_feature_store.return_value = None

    # Create facade
    facade = DataSchedulerFacade(
        catalog=mock_catalog,
        config=mock_config,
        start_metrics_server=False,
    )

    # Mock the registration component
    mock_registration = MagicMock()
    facade._registration_component = mock_registration

    # Call method
    facade._ensure_dataset_registered(
        dataset_id="ohlcv_spy_xnas",
        dataset_type_label="bars",
        location="/data/catalog",
    )

    # Verify delegation
    mock_registration.ensure_dataset_registered.assert_called_once_with(
        registry=facade._data_registry,
        dataset_id="ohlcv_spy_xnas",
        dataset_type_label="bars",
        location="/data/catalog",
        retention_days=90,
    )


__all__ = [
    "test_create_data_scheduler_factory",
    "test_facade_get_status_delegates",
    "test_facade_init_creates_components",
    "test_facade_init_stores_public_attributes",
    "test_facade_run_daily_update_delegates",
    "test_facade_schedule_updates_delegates",
    "test_facade_stop_delegates",
    "test_use_legacy_scheduler_returns_false",
    "test_use_legacy_scheduler_returns_true",
]
