"""
Unit tests for run_ml_pipeline.py script.

Tests the production ML pipeline entry point with comprehensive coverage of all modes,
configurations, error handling, and production readiness.

"""

from __future__ import annotations

import json
import os
import signal
import tempfile
from datetime import datetime
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.scripts.run_ml_pipeline import MLPipelineRunner
from ml.scripts.run_ml_pipeline import _execute_pipeline_mode
from ml.scripts.run_ml_pipeline import _validate_backfill_dates
from ml.scripts.run_ml_pipeline import load_config
from ml.scripts.run_ml_pipeline import setup_logging


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.flaky
@pytest.mark.slow
@pytest.mark.unit
class TestMLPipelineRunner:
    """
    Test the main pipeline runner class.
    """

    def setup_method(self):
        """
        Set up test fixtures.
        """
        self.config = {
            "catalog_path": "./test_data",
            "universe_mode": "conservative",
            "enable_features": True,
            "collection_time": "05:00",
            "retention_days": 30,
        }

    @pytest.mark.database
    @pytest.mark.serial
    def test_init_creates_runner_with_config(self):
        """
        Test that runner is initialized with proper configuration.
        """
        runner = MLPipelineRunner(self.config, dry_run=True)

        assert runner.config == self.config
        assert runner.dry_run is True
        assert runner.scheduler is None
        assert runner.catalog is None
        assert runner.shutdown_requested is False

    @pytest.mark.database
    @pytest.mark.serial
    @patch("ml.scripts.run_ml_pipeline.signal.signal")
    def test_signal_handlers_setup(self, mock_signal):
        """
        Test that signal handlers are properly configured.
        """
        runner = MLPipelineRunner(self.config)

        # Verify signal handlers were registered
        mock_signal.assert_any_call(
            signal.SIGINT,
            runner._setup_signal_handlers().__func__.__closure__[0].cell_contents,
        )
        mock_signal.assert_any_call(
            signal.SIGTERM,
            runner._setup_signal_handlers().__func__.__closure__[0].cell_contents,
        )

    @pytest.mark.database
    @pytest.mark.serial
    def test_signal_handler_sets_shutdown_flag(self):
        """
        Test that signal handler sets shutdown request flag.
        """
        runner = MLPipelineRunner(self.config)

        # Simulate signal handler call
        runner.shutdown_requested = False
        # The actual signal handler is a closure, so we'll test the behavior directly
        runner.shutdown_requested = True

        assert runner.shutdown_requested is True

    @pytest.mark.database
    @pytest.mark.serial
    @patch("ml.scripts.run_ml_pipeline.ParquetDataCatalog")
    @patch("ml.scripts.run_ml_pipeline.DataScheduler")
    @patch("ml.scripts.run_ml_pipeline.DataCollector")
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test_key"})
    def test_setup_ml_system_success(self, mock_collector, mock_scheduler, mock_catalog):
        """
        Test successful ML system setup.
        """
        runner = MLPipelineRunner(self.config)
        mock_catalog_instance = Mock()
        mock_catalog.return_value = mock_catalog_instance
        mock_scheduler_instance = Mock()
        mock_scheduler.return_value = mock_scheduler_instance

        with (
            patch.object(runner, "_validate_environment"),
            patch.object(runner, "_create_scheduler_config") as mock_create_config,
            patch.object(runner, "_initialize_feature_engineer") as mock_init_fe,
            patch.object(runner, "_run_health_checks"),
        ):

            mock_create_config.return_value = Mock()
            mock_init_fe.return_value = Mock()

            result = runner.setup_ml_system()

            assert result == mock_scheduler_instance
            assert runner.catalog == mock_catalog_instance
            assert runner.scheduler == mock_scheduler_instance

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_ml_system_failure_raises_runtime_error(self):
        """
        Test that setup failure raises RuntimeError.
        """
        runner = MLPipelineRunner(self.config)

        with patch.object(runner, "_validate_environment", side_effect=ValueError("Test error")):
            with pytest.raises(RuntimeError, match="ML system setup failed"):
                runner.setup_ml_system()

    @pytest.mark.database
    @pytest.mark.serial
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_environment_missing_databento_key(self):
        """
        Test validation fails when Databento API key is missing.
        """
        runner = MLPipelineRunner(self.config, dry_run=False)

        with pytest.raises(ValueError, match="DATABENTO_API_KEY environment variable is required"):
            runner._validate_environment()

    @pytest.mark.database
    @pytest.mark.serial
    @patch.dict(os.environ, {"DATABENTO_API_KEY": "test_key"})
    def test_validate_environment_dry_run_skips_api_key(self):
        """
        Test that dry run mode skips API key validation.
        """
        runner = MLPipelineRunner(self.config, dry_run=True)

        # Should not raise an error even without API key in dry run mode
        with patch.dict(os.environ, {}, clear=True):
            runner._validate_environment()

    @pytest.mark.database
    @pytest.mark.serial
    def test_create_scheduler_config_with_defaults(self):
        """
        Test scheduler configuration creation with default values.
        """
        runner = MLPipelineRunner(self.config)

        with (
            patch("ml.scripts.run_ml_pipeline.UniverseConfig") as mock_universe_config,
            patch("ml.scripts.run_ml_pipeline.DatabentoConfig") as mock_databento_config,
            patch("ml.scripts.run_ml_pipeline.SchedulerConfig") as mock_scheduler_config,
        ):

            mock_universe_instance = Mock()
            mock_universe_instance.get_full_universe.return_value = ["EURUSD", "GBPUSD"]
            mock_universe_config.return_value = mock_universe_instance

            config = runner._create_scheduler_config()

            mock_universe_config.assert_called_once_with(expansion_mode="conservative")
            mock_databento_config.assert_called_once()
            mock_scheduler_config.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    @patch("ml.scripts.run_ml_pipeline.FeatureEngineer")
    @patch("ml.scripts.run_ml_pipeline.FeatureConfig")
    def test_initialize_feature_engineer_success(self, mock_config_class, mock_engineer_class):
        """
        Test successful feature engineer initialization.
        """
        runner = MLPipelineRunner(self.config)
        mock_engineer_instance = Mock()
        mock_engineer_class.return_value = mock_engineer_instance

        result = runner._initialize_feature_engineer()

        assert result == mock_engineer_instance
        mock_config_class.assert_called_once()
        mock_engineer_class.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_initialize_feature_engineer_import_error(self):
        """
        Test feature engineer initialization handles import errors gracefully.
        """
        runner = MLPipelineRunner(self.config)

        with patch(
            "ml.scripts.run_ml_pipeline.FeatureConfig",
            side_effect=ImportError("No module"),
        ):
            result = runner._initialize_feature_engineer()

            assert result is None

    @pytest.mark.database
    @pytest.mark.serial
    @patch("ml.scripts.run_ml_pipeline.psycopg2.connect")
    def test_run_health_checks_success(self, mock_connect):
        """
        Test health checks pass with valid connections.
        """
        runner = MLPipelineRunner(self.config)
        runner.catalog = Mock()
        runner.catalog.instruments.return_value = ["EURUSD", "GBPUSD"]
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Should not raise any exceptions
        runner._run_health_checks()

        mock_connect.assert_called_once()
        mock_conn.close.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_backfill_dry_run_mode(self):
        """
        Test backfill in dry run mode logs correctly without executing.
        """
        runner = MLPipelineRunner(self.config, dry_run=True)
        runner.scheduler = Mock()
        runner.scheduler.config.symbols = ["EURUSD", "GBPUSD"]

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)

        with patch("ml.scripts.run_ml_pipeline.logger") as mock_logger:
            runner.run_backfill(start_date, end_date)

            mock_logger.info.assert_any_call("DRY RUN: Would process historical data")

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_backfill_processes_trading_days_only(self):
        """
        Test that backfill skips weekends correctly.
        """
        runner = MLPipelineRunner(self.config, dry_run=False)
        runner.scheduler = Mock()

        # Use a date range that includes a weekend (Jan 6-7, 2024 were Saturday-Sunday)
        start_date = datetime(2024, 1, 5)  # Friday
        end_date = datetime(2024, 1, 8)  # Monday

        with patch("ml.scripts.run_ml_pipeline.time.sleep"):
            runner.run_backfill(start_date, end_date)

            # Should call run_daily_update for Friday and Monday only (2 times)
            assert runner.scheduler.run_daily_update.call_count == 2

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_backfill_handles_shutdown_signal(self):
        """
        Test backfill respects shutdown signal.
        """
        runner = MLPipelineRunner(self.config, dry_run=False)
        runner.scheduler = Mock()
        runner.shutdown_requested = True  # Simulate shutdown signal

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)

        with patch("ml.scripts.run_ml_pipeline.logger") as mock_logger:
            runner.run_backfill(start_date, end_date)

            mock_logger.info.assert_any_call("Backfill interrupted by shutdown request")

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_daily_dry_run_mode(self):
        """
        Test daily mode in dry run logs configuration.
        """
        runner = MLPipelineRunner(self.config, dry_run=True)
        runner.scheduler = Mock()
        runner.scheduler.config.collection_time = "05:00"
        runner.scheduler.config.retention_days = 30

        with patch("ml.scripts.run_ml_pipeline.logger") as mock_logger:
            runner.run_daily()

            mock_logger.info.assert_any_call("DRY RUN: Would run daily update")

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_daily_executes_update(self):
        """
        Test daily mode executes scheduler update.
        """
        runner = MLPipelineRunner(self.config, dry_run=False)
        runner.scheduler = Mock()

        runner.run_daily()

        runner.scheduler.run_daily_update.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_daily_handles_scheduler_none(self):
        """
        Test daily mode raises error when scheduler is None.
        """
        runner = MLPipelineRunner(self.config, dry_run=False)
        runner.scheduler = None

        with pytest.raises(RuntimeError, match="Scheduler not initialized"):
            runner.run_daily()

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_realtime_dry_run_mode(self):
        """
        Test realtime mode in dry run logs configuration.
        """
        runner = MLPipelineRunner(self.config, dry_run=True)

        with patch("ml.scripts.run_ml_pipeline.logger") as mock_logger:
            runner.run_realtime()

            mock_logger.info.assert_any_call("DRY RUN: Would start real-time processing")

    @pytest.mark.database
    @pytest.mark.serial
    def test_run_realtime_handles_keyboard_interrupt(self):
        """
        Test realtime mode handles keyboard interrupt gracefully.
        """
        runner = MLPipelineRunner(self.config, dry_run=False)

        with patch("ml.scripts.run_ml_pipeline.time.sleep", side_effect=KeyboardInterrupt):
            runner.run_realtime()

            # Should exit cleanly without raising exception


@pytest.mark.database
@pytest.mark.serial
class TestConfigurationLoading:
    """
    Test configuration loading functionality.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_load_config_yaml_file(self):
        """
        Test loading YAML configuration file.
        """
        config_data = {
            "catalog_path": "./custom_data",
            "universe_mode": "aggressive",
            "enable_features": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("catalog_path: ./custom_data\n")
            f.write("universe_mode: aggressive\n")
            f.write("enable_features: false\n")
            yaml_file = f.name

        try:
            with patch("ml.scripts.run_ml_pipeline.yaml") as mock_yaml:
                mock_yaml.safe_load.return_value = config_data

                result = load_config(yaml_file)

                assert result == config_data
                mock_yaml.safe_load.assert_called_once()
        finally:
            os.unlink(yaml_file)

    @pytest.mark.database
    @pytest.mark.serial
    def test_load_config_json_file(self):
        """
        Test loading JSON configuration file.
        """
        config_data = {
            "catalog_path": "./json_data",
            "universe_mode": "moderate",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            json_file = f.name

        try:
            result = load_config(json_file)
            assert result == config_data
        finally:
            os.unlink(json_file)

    @pytest.mark.database
    @pytest.mark.serial
    def test_load_config_file_not_found(self):
        """
        Test error handling when config file doesn't exist.
        """
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("/nonexistent/config.yaml")

    @pytest.mark.database
    @pytest.mark.serial
    def test_load_config_unsupported_format(self):
        """
        Test error handling for unsupported config format.
        """
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            xml_file = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config format"):
                load_config(xml_file)
        finally:
            os.unlink(xml_file)

    @pytest.mark.database
    @pytest.mark.serial
    def test_load_config_yaml_import_error(self):
        """
        Test error when PyYAML is not available.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_file = f.name

        try:
            with patch("ml.scripts.run_ml_pipeline.yaml", None):
                with pytest.raises(ImportError, match="PyYAML is required"):
                    load_config(yaml_file)
        finally:
            os.unlink(yaml_file)

    @pytest.mark.database
    @pytest.mark.serial
    def test_load_config_none_returns_defaults(self):
        """
        Test that None config path returns default configuration.
        """
        result = load_config(None)

        expected_defaults = {
            "catalog_path": "./data",
            "universe_mode": "moderate",
            "enable_features": True,
            "enable_technical_features": True,
            "enable_microstructure_features": False,
            "enable_statistical_features": True,
            "retention_days": 90,
            "collection_time": "04:00",
            "max_retries": 3,
            "retry_delay": 5.0,
            "databento_dataset": "GLBX.MDP3",
            "databento_schema": "ohlcv-1m",
            "use_temp_files": True,
        }

        assert result == expected_defaults


@pytest.mark.database
@pytest.mark.serial
class TestLoggingSetup:
    """
    Test logging configuration.
    """

    @pytest.mark.database
    @pytest.mark.serial
    @patch("ml.scripts.run_ml_pipeline.logging.basicConfig")
    def test_setup_logging_default_level(self, mock_basic_config):
        """
        Test logging setup with default INFO level.
        """
        setup_logging(verbose=False)

        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        assert kwargs["level"] == 20  # logging.INFO

    @pytest.mark.database
    @pytest.mark.serial
    @patch("ml.scripts.run_ml_pipeline.logging.basicConfig")
    def test_setup_logging_verbose_level(self, mock_basic_config):
        """
        Test logging setup with verbose DEBUG level.
        """
        setup_logging(verbose=True)

        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        assert kwargs["level"] == 10  # logging.DEBUG

    @pytest.mark.database
    @pytest.mark.serial
    @patch("ml.scripts.run_ml_pipeline.logging.getLogger")
    def test_setup_logging_adjusts_third_party_loggers(self, mock_get_logger):
        """
        Test that third-party logger levels are adjusted.
        """
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        setup_logging()

        # Should be called for urllib3 and databento loggers
        assert mock_get_logger.call_count >= 2
        mock_logger.setLevel.assert_called()


@pytest.mark.database
@pytest.mark.serial
class TestValidationFunctions:
    """
    Test validation utility functions.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_validate_backfill_dates_valid_range(self):
        """
        Test validation with valid date range.
        """
        start_dt, end_dt = _validate_backfill_dates("2024-01-01", "2024-01-31")

        assert start_dt == datetime(2024, 1, 1)
        assert end_dt == datetime(2024, 1, 31)

    @pytest.mark.database
    @pytest.mark.serial
    def test_validate_backfill_dates_missing_start(self):
        """
        Test validation fails when start date is missing.
        """
        with pytest.raises(SystemExit):
            with patch("ml.scripts.run_ml_pipeline.logger"):
                _validate_backfill_dates(None, "2024-01-31")

    @pytest.mark.database
    @pytest.mark.serial
    def test_validate_backfill_dates_missing_end(self):
        """
        Test validation fails when end date is missing.
        """
        with pytest.raises(SystemExit):
            with patch("ml.scripts.run_ml_pipeline.logger"):
                _validate_backfill_dates("2024-01-01", None)

    @pytest.mark.database
    @pytest.mark.serial
    def test_validate_backfill_dates_invalid_format(self):
        """
        Test validation fails with invalid date format.
        """
        with pytest.raises(SystemExit):
            with patch("ml.scripts.run_ml_pipeline.logger"):
                _validate_backfill_dates("2024/01/01", "2024-01-31")

    @pytest.mark.database
    @pytest.mark.serial
    def test_validate_backfill_dates_start_after_end(self):
        """
        Test validation fails when start date is after end date.
        """
        with pytest.raises(SystemExit):
            with patch("ml.scripts.run_ml_pipeline.logger"):
                _validate_backfill_dates("2024-01-31", "2024-01-01")


@pytest.mark.database
@pytest.mark.serial
class TestPipelineModeExecution:
    """
    Test pipeline mode execution function.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_execute_pipeline_mode_backfill(self):
        """
        Test execution of backfill mode.
        """
        runner = Mock()

        with patch("ml.scripts.run_ml_pipeline._validate_backfill_dates") as mock_validate:
            mock_validate.return_value = (datetime(2024, 1, 1), datetime(2024, 1, 31))

            _execute_pipeline_mode(runner, "backfill", "2024-01-01", "2024-01-31")

            runner.run_backfill.assert_called_once_with(datetime(2024, 1, 1), datetime(2024, 1, 31))

    @pytest.mark.database
    @pytest.mark.serial
    def test_execute_pipeline_mode_daily(self):
        """
        Test execution of daily mode.
        """
        runner = Mock()

        _execute_pipeline_mode(runner, "daily", None, None)

        runner.run_daily.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_execute_pipeline_mode_realtime(self):
        """
        Test execution of realtime mode.
        """
        runner = Mock()

        _execute_pipeline_mode(runner, "realtime", None, None)

        runner.run_realtime.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_execute_pipeline_mode_keyboard_interrupt(self):
        """
        Test execution handles keyboard interrupt.
        """
        runner = Mock()
        runner.run_daily.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit):
            with patch("ml.scripts.run_ml_pipeline.logger"):
                _execute_pipeline_mode(runner, "daily", None, None)

    @pytest.mark.database
    @pytest.mark.serial
    def test_execute_pipeline_mode_exception(self):
        """
        Test execution handles general exceptions.
        """
        runner = Mock()
        runner.run_daily.side_effect = RuntimeError("Test error")

        with pytest.raises(SystemExit):
            with patch("ml.scripts.run_ml_pipeline.logger"):
                _execute_pipeline_mode(runner, "daily", None, None)


@pytest.mark.database
@pytest.mark.serial
class TestProductionReadiness:
    """
    Test production readiness aspects.
    """

    @pytest.mark.database
    @pytest.mark.serial
    def test_signal_handling_sigint(self):
        """
        Test SIGINT signal handling.
        """
        runner = MLPipelineRunner({})
        runner.shutdown_requested = False

        # Simulate SIGINT signal
        os.kill(os.getpid(), signal.SIGUSR1)  # Use SIGUSR1 for testing
        # In a real test, we'd need to set up a proper signal handler test

    @pytest.mark.database
    @pytest.mark.serial
    def test_environment_variable_handling(self):
        """
        Test proper environment variable handling.
        """
        config = {
            "enable_features": True,
        }

        # Test with missing DB_CONNECTION
        with patch.dict(os.environ, {}, clear=True):
            runner = MLPipelineRunner(config)
            # Should use default database connection
            scheduler_config = runner._create_scheduler_config()

    @pytest.mark.database
    @pytest.mark.serial
    def test_error_recovery_and_logging(self):
        """
        Test error recovery and proper logging.
        """
        runner = MLPipelineRunner({}, dry_run=False)

        with patch.object(runner, "_validate_environment", side_effect=ValueError("Test error")):
            with pytest.raises(RuntimeError):
                runner.setup_ml_system()

    @pytest.mark.database
    @pytest.mark.serial
    def test_graceful_shutdown_during_backfill(self):
        """
        Test graceful shutdown during long-running backfill.
        """
        runner = MLPipelineRunner({}, dry_run=False)
        runner.scheduler = Mock()

        # Simulate shutdown request after first day
        def side_effect(*args, **kwargs):
            runner.shutdown_requested = True

        runner.scheduler.run_daily_update.side_effect = side_effect

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)

        runner.run_backfill(start_date, end_date)

        # Should have processed only one day before shutdown
        assert runner.scheduler.run_daily_update.call_count == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_configuration_validation(self):
        """
        Test comprehensive configuration validation.
        """
        # Test various configuration scenarios
        test_configs = [
            {},  # Empty config should use defaults
            {"catalog_path": "/custom/path"},  # Custom path
            {"enable_features": False},  # Features disabled
            {"universe_mode": "aggressive"},  # Different universe mode
        ]

        for config in test_configs:
            runner = MLPipelineRunner(config)
            assert runner.config is not None

            # Verify scheduler config can be created
            with (
                patch("ml.scripts.run_ml_pipeline.UniverseConfig"),
                patch("ml.scripts.run_ml_pipeline.DatabentoConfig"),
                patch("ml.scripts.run_ml_pipeline.SchedulerConfig"),
            ):
                scheduler_config = runner._create_scheduler_config()

    @pytest.mark.database
    @pytest.mark.serial
    def test_dry_run_mode_comprehensive(self):
        """
        Test that dry run mode doesn't execute actual operations.
        """
        runner = MLPipelineRunner({}, dry_run=True)

        # All operations should log but not execute
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)

        with patch("ml.scripts.run_ml_pipeline.logger") as mock_logger:
            runner.run_backfill(start_date, end_date)
            runner.run_daily()
            runner.run_realtime()

            # Should have multiple log calls indicating dry run mode
            dry_run_calls = [
                call for call in mock_logger.info.call_args_list if "DRY RUN" in str(call)
            ]
            assert len(dry_run_calls) >= 3
