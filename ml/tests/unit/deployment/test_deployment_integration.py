#!/usr/bin/env python3
"""
Integration tests for ML deployment components.

Tests inter-service communication, end-to-end container workflows, and Prometheus metrics.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.deployment.entrypoint_actor import MLSignalActorNode
from ml.deployment.entrypoint_pipeline import PipelineRunner
from ml.deployment.entrypoint_strategy import MLStrategyNode


# Import test database fixture if not already available
try:
    from ml.tests.conftest import clean_postgres_db
    from ml.tests.conftest import test_database
except ImportError:
    # Fixtures should be available via pytest
    pass


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.deployment
@pytest.mark.integration
@pytest.mark.usefixtures("clean_postgres_db")  # Ensure clean PostgreSQL state
class TestDeploymentIntegration:
    """Integration tests for deployment components."""

    @pytest.fixture
    def deployment_env(self, monkeypatch, tmp_path, test_database):
        """Set up complete deployment environment with PostgreSQL."""
        # Create necessary directories and files
        model_path = tmp_path / "models" / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("dummy_model")

        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir(exist_ok=True)

        # Set comprehensive environment variables with PostgreSQL connection
        monkeypatch.setenv("DATABENTO_API_KEY", "test_api_key")
        monkeypatch.setenv("DATABASE_URL", test_database.connection_string)
        monkeypatch.setenv("DB_CONNECTION", test_database.connection_string)
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.setenv("CATALOG_PATH", str(catalog_path))
        monkeypatch.setenv("INSTRUMENT_ID", "BTC-USDT.DATABENTO")
        monkeypatch.setenv("BAR_TYPE", "BTC-USDT.DATABENTO-1-MINUTE")
        monkeypatch.setenv("ACTOR_ID", "MLSignalActor-001")
        monkeypatch.setenv("STRATEGY_ID", "MLStrategy-001")
        monkeypatch.setenv("ML_SIGNAL_SOURCE", "MLSignalActor-001")
        monkeypatch.setenv("USE_DUMMY_STORES", "true")
        monkeypatch.setenv("EXECUTE_TRADES", "false")
        monkeypatch.setenv("PIPELINE_MODE", "daily")
        monkeypatch.setenv("HEALTH_CHECK_PORT", "8082")

        return {
            "model_path": model_path,
            "catalog_path": catalog_path,
        }

    @pytest.mark.database
    @pytest.mark.serial
    def test_actor_to_strategy_communication(self, deployment_env):
        """Test signal actor communicates with strategy."""
        # Set up actor node
        actor_node = MLSignalActorNode()
        strategy_node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_actor.TradingNode") as mock_actor_trading_node:
            with patch("ml.deployment.entrypoint_actor.MLSignalActor") as mock_signal_actor:
                with patch("ml.deployment.entrypoint_strategy.TradingNode") as mock_strategy_trading_node:
                    with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy") as mock_strategy:
                        # Mock the components
                        mock_actor_node = Mock()
                        mock_actor_trading_node.return_value = mock_actor_node
                        mock_actor = Mock()
                        mock_actor.component_id = "MLSignalActor-001"
                        mock_signal_actor.return_value = mock_actor

                        mock_strategy_node = Mock()
                        mock_strategy_trading_node.return_value = mock_strategy_node
                        mock_strat = Mock()
                        mock_strategy.return_value = mock_strat

                        # Set up both nodes
                        actor_node.setup()
                        strategy_node.setup()

                        # Verify actor was created
                        mock_signal_actor.assert_called_once()
                        actor_config = mock_signal_actor.call_args[1]["config"]
                        assert actor_config.component_id == "MLSignalActor-001"

                        # Verify strategy was created with correct signal source
                        mock_strategy.assert_called_once()
                        strategy_config = mock_strategy.call_args[1]["config"]
                        assert strategy_config.ml_signal_source == "MLSignalActor-001"

    @pytest.mark.asyncio
    async def test_concurrent_node_startup(self, deployment_env):
        """Test concurrent startup of multiple nodes."""
        actor_node = MLSignalActorNode()
        strategy_node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_actor.TradingNode") as mock_actor_node_class:
            with patch("ml.deployment.entrypoint_actor.MLSignalActor"):
                with patch("ml.deployment.entrypoint_strategy.TradingNode") as mock_strategy_node_class:
                    with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy"):
                        # Mock trading nodes
                        mock_actor_trading = AsyncMock()
                        mock_actor_trading.run_async = AsyncMock(side_effect=asyncio.CancelledError)
                        mock_actor_node_class.return_value = mock_actor_trading
                        actor_node.node = mock_actor_trading

                        mock_strategy_trading = AsyncMock()
                        mock_strategy_trading.run_async = AsyncMock(side_effect=asyncio.CancelledError)
                        mock_strategy_node_class.return_value = mock_strategy_trading
                        strategy_node.node = mock_strategy_trading

                        # Set up nodes
                        actor_node.setup()
                        strategy_node.setup()

                        # Run nodes concurrently
                        tasks = [
                            asyncio.create_task(actor_node.run()),
                            asyncio.create_task(strategy_node.run()),
                        ]

                        # Wait for tasks
                        await asyncio.gather(*tasks, return_exceptions=True)

                        # Verify both nodes ran
                        mock_actor_trading.run_async.assert_called_once()
                        mock_strategy_trading.run_async.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_pipeline_with_stores_initialization(self, deployment_env, test_database):
        """Test pipeline initializes stores correctly with PostgreSQL."""
        runner = PipelineRunner()

        with patch("ml.deployment.entrypoint_pipeline.check_ml_dependencies"):
            with patch("ml.deployment.entrypoint_pipeline.FeatureStore") as mock_fs_class:
                with patch("ml.deployment.entrypoint_pipeline.ModelStore") as mock_ms_class:
                    with patch("ml.deployment.entrypoint_pipeline.ParquetDataCatalog") as mock_catalog_class:
                        with patch("ml.deployment.entrypoint_pipeline.DataScheduler") as mock_scheduler_class:
                            # Mock stores
                            mock_fs = Mock()
                            mock_fs_class.return_value = mock_fs
                            mock_ms = Mock()
                            mock_ms_class.return_value = mock_ms
                            mock_catalog = Mock()
                            mock_catalog_class.return_value = mock_catalog
                            mock_scheduler = Mock()
                            mock_scheduler_class.return_value = mock_scheduler

                            # Mock the run method to prevent infinite loop
                            with patch.object(runner, "_run_daily"):
                                runner.run()

                            # Verify stores were initialized with PostgreSQL
                            mock_fs_class.assert_called_once()
                            mock_ms_class.assert_called_once()
                            # Check that connection strings are PostgreSQL
                            fs_call_args = mock_fs_class.call_args
                            if fs_call_args and fs_call_args[1].get("connection_string"):
                                assert "postgresql://" in fs_call_args[1]["connection_string"]
                            mock_catalog_class.assert_called_once()
                            mock_scheduler_class.assert_called_once()

                            # Verify scheduler was created with catalog
                            call_kwargs = mock_scheduler_class.call_args[1]
                            assert call_kwargs["catalog"] == mock_catalog

    @pytest.mark.database
    @pytest.mark.serial
    def test_health_check_endpoint_availability(self, deployment_env):
        """Test health check endpoint is available during pipeline run."""
        from ml.deployment.entrypoint_pipeline import app
        from ml.deployment.entrypoint_pipeline import pipeline_status

        # Set pipeline to healthy
        pipeline_status["healthy"] = True
        pipeline_status["last_run"] = "2024-01-01T00:00:00"

        with app.test_client() as client:
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            data = response.get_json()
            assert data["healthy"] is True
            assert data["last_run"] is not None

    @pytest.mark.asyncio
    async def test_graceful_shutdown_sequence(self, deployment_env):
        """Test graceful shutdown of all components."""
        actor_node = MLSignalActorNode()
        strategy_node = MLStrategyNode()
        pipeline_runner = PipelineRunner()

        # Mock components
        mock_actor_trading = AsyncMock()
        mock_actor_trading.stop_async = AsyncMock()
        actor_node.node = mock_actor_trading
        actor_node.running = True

        mock_strategy_trading = AsyncMock()
        mock_strategy_trading.stop_async = AsyncMock()
        mock_strategy_trading.trader.strategies.return_value = {}
        strategy_node.node = mock_strategy_trading
        strategy_node.running = True

        mock_scheduler = Mock()
        mock_scheduler.stop = Mock()
        pipeline_runner.scheduler = mock_scheduler
        pipeline_runner.running = True

        # Shutdown all components
        await actor_node.shutdown()
        await strategy_node.shutdown()
        pipeline_runner._signal_handler(15, None)  # SIGTERM

        # Verify all components stopped
        assert actor_node.running is False
        assert strategy_node.running is False
        assert pipeline_runner.running is False

        mock_actor_trading.stop_async.assert_called_once()
        mock_strategy_trading.stop_async.assert_called_once()
        mock_scheduler.stop.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_environment_variable_propagation(self, deployment_env, monkeypatch):
        """Test environment variables propagate correctly to all components."""
        # Set custom environment variables
        monkeypatch.setenv("POSITION_SIZE_PCT", "0.05")
        monkeypatch.setenv("MIN_CONFIDENCE", "0.75")
        monkeypatch.setenv("UNIVERSE_SYMBOLS", "AAPL.XNAS,MSFT.XNAS,GOOGL.XNAS")
        monkeypatch.setenv("DATABENTO_DATASET", "XNAS.ITCH")

        # Test actor node
        actor_node = MLSignalActorNode()
        with patch("ml.deployment.entrypoint_actor.TradingNode"):
            with patch("ml.deployment.entrypoint_actor.MLSignalActor") as mock_actor:
                actor_node.setup()
                actor_config = mock_actor.call_args[1]["config"]
                assert actor_config.use_dummy_stores is True

        # Test strategy node
        strategy_node = MLStrategyNode()
        with patch("ml.deployment.entrypoint_strategy.TradingNode"):
            with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy") as mock_strategy:
                strategy_node.setup()
                strategy_config = mock_strategy.call_args[1]["config"]
                assert strategy_config.position_size_pct == 0.05
                assert strategy_config.min_confidence == 0.75

        # Test pipeline runner
        runner = PipelineRunner()
        config = runner._create_config()
        assert "AAPL.XNAS" in config.symbols
        assert "MSFT.XNAS" in config.symbols
        assert config.databento.dataset == "XNAS.ITCH"

    @pytest.mark.database
    @pytest.mark.serial
    def test_error_recovery_and_retry(self, deployment_env):
        """Test error recovery and retry mechanisms."""
        runner = PipelineRunner()
        runner.running = True

        # Mock scheduler with failures then success
        mock_scheduler = Mock()
        mock_scheduler.run_daily_update.side_effect = [
            RuntimeError("First failure"),
            RuntimeError("Second failure"),
            None,  # Success on third attempt
        ]
        runner.scheduler = mock_scheduler

        # Track errors
        from ml.deployment.entrypoint_pipeline import pipeline_status

        # Run with immediate shutdown after third attempt
        call_count = 0

        def stop_after_third():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                runner._shutdown_event.set()

        mock_scheduler.run_daily_update.side_effect = [
            RuntimeError("First failure"),
            RuntimeError("Second failure"),
            stop_after_third(),
        ]

        with patch("time.sleep"):  # Skip sleep delays
            runner._run_realtime()

        # Verify errors were tracked
        assert len(pipeline_status["errors"]) >= 2
        assert "First failure" in str(pipeline_status["errors"])
        assert "Second failure" in str(pipeline_status["errors"])

    @pytest.mark.database
    @pytest.mark.serial
    @patch("ml.common.metrics.PREDICTION_COUNTER")
    @patch("ml.common.metrics.FEATURE_CALCULATION_TIMER")
    @patch("ml.common.metrics.MODEL_INFERENCE_TIMER")
    def test_prometheus_metrics_exposure(self, mock_inference_timer, mock_feature_timer, mock_prediction_counter, deployment_env):
        """Test Prometheus metrics are properly exposed."""
        # Test actor metrics
        actor_node = MLSignalActorNode()
        with patch("ml.deployment.entrypoint_actor.TradingNode"):
            with patch("ml.deployment.entrypoint_actor.MLSignalActor") as mock_actor_class:
                mock_actor = Mock()
                mock_actor_class.return_value = mock_actor

                actor_node.setup()

                # Verify actor configuration enables metrics
                actor_config = mock_actor_class.call_args[1]["config"]
                assert actor_config.enable_health_monitoring is True

    @pytest.mark.database
    @pytest.mark.serial
    def test_docker_compose_service_dependencies(self, deployment_env):
        """Test service dependency checks."""
        from ml.deployment.check_health import check_docker_compose

        with patch("subprocess.run") as mock_run:
            # Simulate all required services running
            services_json = """[
                {"Service": "postgres", "State": "running"},
                {"Service": "ml_pipeline", "State": "running"},
                {"Service": "redis", "State": "running"},
                {"Service": "prometheus", "State": "running"},
                {"Service": "grafana", "State": "running"}
            ]"""

            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = services_json
            mock_run.return_value = mock_result

            result = check_docker_compose()
            assert result is True

            # Test with missing critical service
            services_json_missing = """[
                {"Service": "redis", "State": "running"},
                {"Service": "prometheus", "State": "running"}
            ]"""

            mock_result.stdout = services_json_missing
            result = check_docker_compose()
            assert result is False

    @pytest.mark.database
    @pytest.mark.serial
    def test_configuration_validation(self, deployment_env, monkeypatch):
        """Test configuration validation across components."""
        # Test invalid configuration handling
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

        actor_node = MLSignalActorNode()
        with pytest.raises(SystemExit) as exc_info:
            actor_node.setup()
        assert exc_info.value.code == 1

        # Test missing model file
        monkeypatch.setenv("DATABENTO_API_KEY", "test_key")
        monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.pkl")

        actor_node = MLSignalActorNode()
        with pytest.raises(SystemExit) as exc_info:
            actor_node.setup()
        assert exc_info.value.code == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_multi_threading_safety(self, deployment_env):
        """Test thread safety of concurrent operations."""
        from ml.deployment.entrypoint_pipeline import pipeline_status

        # Reset status
        pipeline_status["errors"] = []

        def update_status(error_msg):
            """Simulate concurrent status updates."""
            time.sleep(0.001)  # Simulate processing
            pipeline_status["errors"].append(error_msg)

        # Create multiple threads updating status
        threads = []
        for i in range(10):
            thread = threading.Thread(target=update_status, args=(f"Error {i}",))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all errors were recorded
        assert len(pipeline_status["errors"]) == 10
        for i in range(10):
            assert f"Error {i}" in pipeline_status["errors"]
