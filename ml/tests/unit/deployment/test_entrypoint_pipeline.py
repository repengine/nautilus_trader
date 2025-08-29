#!/usr/bin/env python3
"""
Unit tests for ML Pipeline deployment entrypoint.

Tests pipeline modes (backfill, daily, realtime), health checks, and store initialization.
"""

from __future__ import annotations

import os
import signal
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.deployment.entrypoint_pipeline import PipelineRunner
from ml.deployment.entrypoint_pipeline import PipelineStatus
from ml.deployment.entrypoint_pipeline import app
from ml.deployment.entrypoint_pipeline import health_check
from ml.deployment.entrypoint_pipeline import main
from ml.deployment.entrypoint_pipeline import pipeline_status


# Import test database fixture if not already available
try:
    from ml.tests.conftest import clean_postgres_db
    from ml.tests.conftest import test_database
except ImportError:
    # Fixtures should be available via pytest
    pass


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.unit
class TestHealthEndpoint:
    """Test Flask health check endpoint."""

    def test_health_check_healthy(self):
        """Test health check returns 200 when healthy."""
        # Set pipeline status to healthy
        pipeline_status["healthy"] = True
        pipeline_status["last_run"] = datetime.now().isoformat()
        pipeline_status["errors"] = []

        with app.test_client() as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.get_json()
            assert data["healthy"] is True
            assert data["last_run"] is not None
            assert len(data["errors"]) == 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_health_check_unhealthy(self):
        """Test health check returns 503 when unhealthy."""
        # Set pipeline status to unhealthy
        pipeline_status["healthy"] = False
        pipeline_status["last_run"] = None
        pipeline_status["errors"] = ["Database connection failed"]

        with app.test_client() as client:
            response = client.get("/health")
            assert response.status_code == 503
            data = response.get_json()
            assert data["healthy"] is False
            assert "Database connection failed" in data["errors"]


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")  # Ensure clean PostgreSQL state
class TestPipelineRunner:
    """Test PipelineRunner class."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        """Clean environment for isolated testing."""
        for key in list(os.environ.keys()):
            if key.startswith(("UNIVERSE_", "DATABENTO_", "DATABASE_", "FEATURE_", "MODEL_", "CATALOG_", "PIPELINE_", "REALTIME_", "HEALTH_")):
                monkeypatch.delenv(key, raising=False)

    @pytest.fixture
    def valid_env(self, monkeypatch, tmp_path, test_database):
        """Set up valid environment variables for testing with PostgreSQL."""
        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir(exist_ok=True)

        monkeypatch.setenv("UNIVERSE_SYMBOLS", "SPY.XNAS,QQQ.XNAS,IWM.XNAS")
        monkeypatch.setenv("DATABENTO_DATASET", "XNAS.ITCH")
        monkeypatch.setenv("DATABENTO_SCHEMA", "ohlcv-1m")
        monkeypatch.setenv("DATABASE_URL", test_database.connection_string)
        monkeypatch.setenv("CATALOG_PATH", str(catalog_path))
        monkeypatch.setenv("PIPELINE_MODE", "daily")
        monkeypatch.setenv("PIPELINE_SCHEDULE", "0 17 * * *")

        return catalog_path

    @pytest.mark.database
    @pytest.mark.serial
    def test_init(self):
        """Test PipelineRunner initialization."""
        runner = PipelineRunner()
        assert runner.scheduler is None
        assert runner.running is False
        assert isinstance(runner._shutdown_event, threading.Event)

    @pytest.mark.database
    @pytest.mark.serial
    def test_signal_handler(self):
        """Test signal handler sets shutdown event."""
        runner = PipelineRunner()
        runner.running = True

        # Mock scheduler
        mock_scheduler = Mock()
        runner.scheduler = mock_scheduler

        # Call signal handler
        runner._signal_handler(signal.SIGTERM, None)

        assert runner.running is False
        assert runner._shutdown_event.is_set()
        mock_scheduler.stop.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_signal_handler_no_scheduler(self):
        """Test signal handler handles missing scheduler gracefully."""
        runner = PipelineRunner()
        runner.running = True
        runner.scheduler = None

        # Should not raise
        runner._signal_handler(signal.SIGINT, None)

        assert runner.running is False
        assert runner._shutdown_event.is_set()

    @pytest.mark.database
    @pytest.mark.serial
    def test_create_config(self, valid_env):
        """Test configuration creation from environment."""
        runner = PipelineRunner()
        config = runner._create_config()

        assert "SPY.XNAS" in config.symbols
        assert "QQQ.XNAS" in config.symbols
        assert "IWM.XNAS" in config.symbols
        assert config.databento.dataset == "XNAS.ITCH"
        assert config.databento.schema == "ohlcv-1m"
        assert config.databento.stype_in == "raw_symbol"
        assert config.feature_store_enabled is True

    @pytest.mark.database
    @pytest.mark.serial
    def test_create_config_with_universe_expansion(self, valid_env, monkeypatch):
        """Test configuration with universe expansion."""
        monkeypatch.setenv("UNIVERSE_MODE", "aggressive")

        runner = PipelineRunner()
        config = runner._create_config()

        # Should include original symbols plus expanded universe
        assert "SPY.XNAS" in config.symbols
        assert len(config.symbols) > 3  # More than just the 3 provided

    @pytest.mark.database
    @pytest.mark.serial
    def test_initialize_stores(self, valid_env, test_database):
        """Test store initialization with PostgreSQL."""
        runner = PipelineRunner()
        config = runner._create_config()

        with patch("ml.deployment.entrypoint_pipeline.FeatureStore") as mock_fs_class:
            with patch("ml.deployment.entrypoint_pipeline.ModelStore") as mock_ms_class:
                mock_fs = Mock()
                mock_ms = Mock()
                mock_fs_class.return_value = mock_fs
                mock_ms_class.return_value = mock_ms

                feature_store, model_store = runner._initialize_stores(config)

                # Verify stores were created with PostgreSQL connections
                mock_fs_class.assert_called_once()
                mock_ms_class.assert_called_once()
                assert feature_store == mock_fs
                assert model_store == mock_ms
                # Verify PostgreSQL connection string was used
                fs_call_args = mock_fs_class.call_args
                if fs_call_args and fs_call_args[1].get("connection_string"):
                    assert "postgresql://" in fs_call_args[1]["connection_string"]

    @pytest.mark.database
    @pytest.mark.serial
    def test_initialize_catalog(self, valid_env):
        """Test catalog initialization."""
        runner = PipelineRunner()
        config = runner._create_config()

        with patch("ml.deployment.entrypoint_pipeline.ParquetDataCatalog") as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog_class.return_value = mock_catalog

            catalog = runner._initialize_catalog(config)

            # Verify catalog was created with correct path
            mock_catalog_class.assert_called_once()
            call_args = mock_catalog_class.call_args[0]
            assert "catalog" in call_args[0]
            assert catalog == mock_catalog

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_backfill_mode(self, valid_env, monkeypatch):
        """Test pipeline run in backfill mode."""
        monkeypatch.setenv("PIPELINE_MODE", "backfill")

        runner = PipelineRunner()

        with patch.object(runner, "_run_backfill") as mock_backfill:
            with patch("ml.deployment.entrypoint_pipeline.check_ml_dependencies"):
                with patch("ml.deployment.entrypoint_pipeline.FeatureStore"):
                    with patch("ml.deployment.entrypoint_pipeline.ModelStore"):
                        with patch("ml.deployment.entrypoint_pipeline.ParquetDataCatalog"):
                            with patch("ml.deployment.entrypoint_pipeline.DataScheduler"):
                                runner.run()

                                mock_backfill.assert_called_once()
                                assert pipeline_status["healthy"] is True
                                assert pipeline_status["last_run"] is not None

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_daily_mode(self, valid_env, monkeypatch):
        """Test pipeline run in daily mode."""
        monkeypatch.setenv("PIPELINE_MODE", "daily")

        runner = PipelineRunner()

        with patch.object(runner, "_run_daily") as mock_daily:
            with patch("ml.deployment.entrypoint_pipeline.check_ml_dependencies"):
                with patch("ml.deployment.entrypoint_pipeline.FeatureStore"):
                    with patch("ml.deployment.entrypoint_pipeline.ModelStore"):
                        with patch("ml.deployment.entrypoint_pipeline.ParquetDataCatalog"):
                            with patch("ml.deployment.entrypoint_pipeline.DataScheduler"):
                                runner.run()

                                mock_daily.assert_called_once()
                                assert pipeline_status["healthy"] is True

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_realtime_mode(self, valid_env, monkeypatch):
        """Test pipeline run in realtime mode."""
        monkeypatch.setenv("PIPELINE_MODE", "realtime")

        runner = PipelineRunner()

        with patch.object(runner, "_run_realtime") as mock_realtime:
            with patch("ml.deployment.entrypoint_pipeline.check_ml_dependencies"):
                with patch("ml.deployment.entrypoint_pipeline.FeatureStore"):
                    with patch("ml.deployment.entrypoint_pipeline.ModelStore"):
                        with patch("ml.deployment.entrypoint_pipeline.ParquetDataCatalog"):
                            with patch("ml.deployment.entrypoint_pipeline.DataScheduler"):
                                runner.run()

                                mock_realtime.assert_called_once()
                                assert pipeline_status["healthy"] is True

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_invalid_mode(self, valid_env, monkeypatch):
        """Test pipeline run with invalid mode."""
        monkeypatch.setenv("PIPELINE_MODE", "invalid_mode")

        runner = PipelineRunner()

        with patch("ml.deployment.entrypoint_pipeline.check_ml_dependencies"):
            with patch("ml.deployment.entrypoint_pipeline.FeatureStore"):
                with patch("ml.deployment.entrypoint_pipeline.ModelStore"):
                    with patch("ml.deployment.entrypoint_pipeline.ParquetDataCatalog"):
                        with patch("ml.deployment.entrypoint_pipeline.DataScheduler"):
                            with pytest.raises(SystemExit) as exc_info:
                                runner.run()

                            assert exc_info.value.code == 1
                            assert pipeline_status["healthy"] is False
                            assert len(pipeline_status["errors"]) > 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_handles_exception(self, valid_env):
        """Test pipeline run handles exceptions."""
        runner = PipelineRunner()

        with patch("ml.deployment.entrypoint_pipeline.check_ml_dependencies", side_effect=ImportError("Missing dependency")):
            with pytest.raises(SystemExit) as exc_info:
                runner.run()

            assert exc_info.value.code == 1
            assert pipeline_status["healthy"] is False
            assert "Missing dependency" in str(pipeline_status["errors"])

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_backfill_executes_daily_update(self, valid_env):
        """Test backfill mode executes daily update."""
        runner = PipelineRunner()

        # Mock scheduler
        mock_scheduler = Mock()
        runner.scheduler = mock_scheduler

        runner._run_backfill()

        mock_scheduler.run_daily_update.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_daily_schedules_updates(self, valid_env, monkeypatch):
        """Test daily mode schedules updates."""
        monkeypatch.setenv("PIPELINE_SCHEDULE", "0 18 * * *")

        runner = PipelineRunner()
        runner.running = True

        # Mock scheduler
        mock_scheduler = Mock()
        runner.scheduler = mock_scheduler

        # Set shutdown event to break the loop
        runner._shutdown_event.set()

        runner._run_daily()

        mock_scheduler.schedule_updates.assert_called_once_with("0 18 * * *")
        assert runner.running is True  # Should stay running until shutdown

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_realtime_continuous_updates(self, valid_env, monkeypatch):
        """Test realtime mode runs continuous updates."""
        monkeypatch.setenv("REALTIME_INTERVAL", "60")  # 1 minute

        runner = PipelineRunner()
        runner.running = True

        # Mock scheduler
        mock_scheduler = Mock()
        runner.scheduler = mock_scheduler

        # Set shutdown event to break the loop immediately
        runner._shutdown_event.set()

        runner._run_realtime()

        # Should attempt at least one update
        mock_scheduler.run_daily_update.assert_called()

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_realtime_handles_errors(self, valid_env, monkeypatch):
        """Test realtime mode handles update errors."""
        monkeypatch.setenv("REALTIME_INTERVAL", "60")

        runner = PipelineRunner()
        runner.running = True

        # Mock scheduler to raise error
        mock_scheduler = Mock()
        mock_scheduler.run_daily_update.side_effect = [RuntimeError("Update failed"), None]
        runner.scheduler = mock_scheduler

        # Set shutdown after first error
        def stop_after_error(*args):
            runner._shutdown_event.set()

        mock_scheduler.run_daily_update.side_effect = [RuntimeError("Update failed"), stop_after_error()]

        runner._run_realtime()

        # Should have error in pipeline status
        assert "Update failed" in str(pipeline_status["errors"])

    @pytest.mark.database
    @pytest.mark.serial
    def test_environment_variable_defaults(self, monkeypatch, tmp_path, test_database):
        """Test default environment variable values with PostgreSQL."""
        # Set minimal required env vars with PostgreSQL
        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir(exist_ok=True)
        monkeypatch.setenv("CATALOG_PATH", str(catalog_path))
        monkeypatch.setenv("DATABASE_URL", test_database.connection_string)

        runner = PipelineRunner()
        config = runner._create_config()

        # Check defaults
        assert config.databento.dataset == "EQUS.MINI"
        assert config.databento.schema == "ohlcv-1m"
        assert config.databento.stype_in == "raw_symbol"
        assert "SPY.XNAS" in config.symbols  # Default symbol

    @pytest.mark.database
    @pytest.mark.serial
    def test_signal_registration(self):
        """Test signal handlers are registered."""
        runner = PipelineRunner()

        # Verify signal handlers were registered
        with patch("signal.signal") as mock_signal:
            PipelineRunner()
            assert mock_signal.call_count == 2
            calls = mock_signal.call_args_list
            signals_registered = [call[0][0] for call in calls]
            assert signal.SIGINT in signals_registered
            assert signal.SIGTERM in signals_registered


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
class TestMainFunction:
    """Test the main entry point function with PostgreSQL."""

    @pytest.fixture
    def valid_env(self, monkeypatch, tmp_path, test_database):
        """Set up valid environment for main function with PostgreSQL."""
        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir(exist_ok=True)
        monkeypatch.setenv("CATALOG_PATH", str(catalog_path))
        monkeypatch.setenv("PIPELINE_MODE", "backfill")
        monkeypatch.setenv("HEALTH_CHECK_PORT", "8081")
        monkeypatch.setenv("DATABASE_URL", test_database.connection_string)

    @pytest.mark.database
    @pytest.mark.serial
    def test_main_starts_health_server(self, valid_env):
        """Test main starts health check server in background."""
        with patch("threading.Thread") as mock_thread_class:
            with patch("ml.deployment.entrypoint_pipeline.PipelineRunner") as mock_runner_class:
                mock_thread = Mock()
                mock_thread_class.return_value = mock_thread
                mock_runner = Mock()
                mock_runner_class.return_value = mock_runner

                main()

                # Verify health thread was created and started
                mock_thread_class.assert_called_once()
                assert mock_thread.daemon is True
                mock_thread.start.assert_called_once()

                # Verify runner was created and run
                mock_runner_class.assert_called_once()
                mock_runner.run.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_main_configures_health_port(self, valid_env, monkeypatch):
        """Test main uses configured health check port."""
        monkeypatch.setenv("HEALTH_CHECK_PORT", "9090")

        with patch("threading.Thread") as mock_thread_class:
            with patch("ml.deployment.entrypoint_pipeline.PipelineRunner"):
                with patch("ml.deployment.entrypoint_pipeline.app.run") as mock_app_run:
                    mock_thread = Mock()
                    mock_thread_class.return_value = mock_thread

                    # Get the target function from Thread constructor
                    main()
                    target_func = mock_thread_class.call_args[1]["target"]

                    # Call the target function to verify app.run arguments
                    with patch("ml.deployment.entrypoint_pipeline.app.run") as mock_app_run_inner:
                        target_func()
                        mock_app_run_inner.assert_called_once_with(
                            host="0.0.0.0",
                            port=9090,
                            debug=False,
                        )
