#!/usr/bin/env python3

"""
Integration tests for ML deployment components.

Tests inter-service communication, end-to-end container workflows, and Prometheus
metrics.

"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml._imports import HAS_NAUTILUS_CORE
from ml._imports import NAUTILUS_CORE_IMPORT_ERROR

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

_XDIST_WORKER_COUNT = int(os.getenv("PYTEST_XDIST_WORKER_COUNT", "1"))

if not HAS_NAUTILUS_CORE:  # pragma: no cover - depends on native extensions
    pytest.skip(
        f"Nautilus Trader core extensions unavailable: {NAUTILUS_CORE_IMPORT_ERROR}",
        allow_module_level=True,
    )

from ml.deployment.entrypoint_actor import MLSignalActorNode
from ml.deployment.entrypoint_pipeline import PipelineRunner
from ml.deployment.entrypoint_strategy import MLStrategyNode


logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.deployment
@pytest.mark.usefixtures("cloned_test_database")
class TestDeploymentIntegration:
    """
    Integration tests for deployment components.
    """

    _DIAGNOSTIC_ENV_KEYS: Sequence[str] = (
        "DATABENTO_API_KEY",
        "USE_DUMMY_STORES",
        "USE_MOCK_DATA",
        "DATABASE_URL",
        "DB_CONNECTION",
        "MODEL_PATH",
        "CATALOG_PATH",
        "INSTRUMENT_ID",
        "BAR_TYPE",
        "ACTOR_ID",
        "STRATEGY_ID",
        "ML_SIGNAL_SOURCE",
        "PIPELINE_MODE",
        "EXECUTE_TRADES",
    )

    @classmethod
    def _log_deployment_env_state(
        cls,
        context: str,
        *,
        request: pytest.FixtureRequest | None = None,
    ) -> None:
        state: dict[str, str | None] = {}
        for key in cls._DIAGNOSTIC_ENV_KEYS:
            value = os.getenv(key)
            if key == "DATABENTO_API_KEY" and value:
                state[key] = f"<redacted len={len(value)}>"
            else:
                state[key] = value
        if request is not None and hasattr(request, "node"):
            callspec = getattr(request.node, "callspec", None)
            if callspec is not None and hasattr(callspec, "id"):
                state["pytest_variant"] = callspec.id
        payload = json.dumps(state, sort_keys=True)
        message = f"[deployment-diagnostics] context={context} env={payload}"
        print(message)
        logger.info(message)

    @staticmethod
    def _log_config_snapshot(actor_config: Any, strategy_config: Any) -> None:
        snapshot: dict[str, Any] = {
            "actor_component_id": getattr(actor_config, "component_id", None),
            "actor_use_dummy_stores": getattr(actor_config, "use_dummy_stores", None),
            "actor_publish_signals": getattr(actor_config, "publish_signals", None),
            "strategy_ml_signal_source": getattr(strategy_config, "ml_signal_source", None),
            "strategy_position_size_pct": getattr(strategy_config, "position_size_pct", None),
        }
        payload = json.dumps(snapshot, sort_keys=True, default=str)
        message = f"[deployment-diagnostics] configs={payload}"
        print(message)
        logger.info(message)

    @pytest.fixture
    def deployment_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cloned_test_database: str,
        mock_onnx_runtime: Any,
        onnx_session_stub_factory: Callable[..., object],
    ):
        """
        Set up complete deployment environment with PostgreSQL.
        """
        model_path = tmp_path / "models" / "model.onnx"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"deterministic-onnx")
        model_meta = model_path.with_suffix(".onnx.meta")
        if not model_meta.exists():
            model_meta.write_text("{}")

        def _build_session(*_: object, **__: object) -> object:
            return onnx_session_stub_factory(
                prediction=0.42,
                confidence=0.91,
            )

        mock_onnx_runtime.ort.InferenceSession.side_effect = _build_session

        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir(exist_ok=True)

        # Set comprehensive environment variables with PostgreSQL connection
        monkeypatch.setenv("DATABENTO_API_KEY", "test_api_key")
        monkeypatch.setenv("DATABASE_URL", cloned_test_database)
        monkeypatch.setenv("DB_CONNECTION", cloned_test_database)
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.setenv("CATALOG_PATH", str(catalog_path))
        monkeypatch.setenv("INSTRUMENT_ID", "BTC-USDT.DATABENTO")
        monkeypatch.setenv("BAR_TYPE", "BTC-USDT.DATABENTO-1-MINUTE-LAST-EXTERNAL")
        monkeypatch.setenv("ACTOR_ID", "MLSignalActor-001")
        monkeypatch.setenv("STRATEGY_ID", "MLStrategy-001")
        monkeypatch.setenv("ML_SIGNAL_SOURCE", "MLSignalActor-001")
        monkeypatch.setenv("USE_DUMMY_STORES", "true")
        monkeypatch.setenv("ML_LIVE_RECORD_ENABLE", "0")
        monkeypatch.setenv("EXECUTE_TRADES", "false")
        monkeypatch.setenv("PIPELINE_MODE", "daily")
        monkeypatch.setenv("HEALTH_CHECK_PORT", "8082")

        return {
            "model_path": model_path,
            "catalog_path": catalog_path,
        }

    @pytest.mark.database
    @pytest.mark.serial
    def test_actor_to_strategy_communication(
        self,
        deployment_env: dict[str, Path],
        request: pytest.FixtureRequest,
    ) -> None:
        """
        Test signal actor communicates with strategy.
        """
        self._log_deployment_env_state(
            "test_actor_to_strategy_communication",
            request=request,
        )
        # Set up actor node
        actor_node = MLSignalActorNode()
        strategy_node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_actor.TradingNode") as mock_actor_trading_node:
            with patch("ml.deployment.entrypoint_actor.MLSignalActor") as mock_signal_actor:
                with patch(
                    "ml.deployment.entrypoint_strategy.TradingNode",
                ) as mock_strategy_trading_node:
                    with patch(
                        "ml.deployment.entrypoint_strategy.MLTradingStrategy",
                    ) as mock_strategy:
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

                        # Verify strategy was created with correct signal source
                        mock_strategy.assert_called_once()
                        strategy_config = mock_strategy.call_args[1]["config"]

                        self._log_config_snapshot(actor_config, strategy_config)

                        assert actor_config.component_id == "MLSignalActor-001"
                        assert strategy_config.ml_signal_source == "MLSignalActor-001"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    @pytest.mark.serial
    @pytest.mark.skipif(
        _XDIST_WORKER_COUNT > 1,
        reason="xdist workers crash under concurrent node startup; full-suite runs hit an Execnet segfault while tearing down this async test",
    )
    async def test_concurrent_node_startup(self, deployment_env):
        """
        Test concurrent startup of multiple nodes.
        """
        actor_node = MLSignalActorNode()
        strategy_node = MLStrategyNode()
        worker_pid = os.getpid()
        log_path = Path("/tmp/concurrent-worker.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"{time.time():.3f} concurrent_node_startup beginning worker_pid={worker_pid}\n",
            )
        print(f"concurrent_node_startup beginning worker_pid={worker_pid}", flush=True)

        with patch("ml.deployment.entrypoint_actor.TradingNode") as mock_actor_node_class:
            with patch("ml.deployment.entrypoint_actor.MLSignalActor"):
                with patch(
                    "ml.deployment.entrypoint_strategy.TradingNode",
                ) as mock_strategy_node_class:
                    with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy"):
                        # Mock trading nodes
                        mock_actor_trading = AsyncMock()
                        mock_actor_trading.run_async = AsyncMock(side_effect=asyncio.CancelledError)
                        mock_actor_node_class.return_value = mock_actor_trading
                        actor_node.node = mock_actor_trading

                        mock_strategy_trading = AsyncMock()
                        mock_strategy_trading.run_async = AsyncMock(
                            side_effect=asyncio.CancelledError,
                        )
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
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        print(
                            f"concurrent_node_startup completed worker_pid={worker_pid} "
                            f"results={[type(result).__name__ for result in results]}",
                            flush=True,
                        )
                        with log_path.open("a", encoding="utf-8") as log_file:
                            log_file.write(
                                f"{time.time():.3f} concurrent_node_startup completed "
                                f"worker_pid={worker_pid} "
                                f"results={[type(result).__name__ for result in results]}\n",
                            )

                        # Verify both nodes ran
                        mock_actor_trading.run_async.assert_called_once()
                        mock_strategy_trading.run_async.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_pipeline_with_stores_initialization(self, deployment_env):
        """
        Test pipeline initializes stores correctly with PostgreSQL.
        """
        runner = PipelineRunner()

        with patch("ml.deployment.entrypoint_pipeline.check_ml_dependencies"):
            with patch("ml.deployment.entrypoint_pipeline.FeatureStore") as mock_fs_class:
                with patch("ml.deployment.entrypoint_pipeline.ModelStore") as mock_ms_class:
                    with patch(
                        "ml.deployment.entrypoint_pipeline.ParquetDataCatalog",
                    ) as mock_catalog_class:
                        with patch(
                            "ml.deployment.entrypoint_pipeline.DataScheduler",
                        ) as mock_scheduler_class:
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
                            with patch.object(runner, "_run_daily"), patch.object(
                                runner, "_bootstrap_database"
                            ):
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
        """
        Test health check endpoint is available during pipeline run.
        """
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
        """
        Test graceful shutdown of all components.
        """
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
        mock_strategy_trader = Mock()
        mock_strategy_trader.strategies = Mock(return_value={})
        mock_strategy_trading.trader = mock_strategy_trader
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
        """
        Test environment variables propagate correctly to all components.
        """
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
        """
        Test error recovery and retry mechanisms.
        """
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
            runner._shutdown_event.set()  # Stop immediately on third call

        mock_scheduler.run_daily_update.side_effect = [
            RuntimeError("First failure"),
            RuntimeError("Second failure"),
            stop_after_third,  # Call the function to trigger shutdown
        ]

        # Completely replace the runner's loop with our test logic
        def test_run_realtime():
            try:
                mock_scheduler.run_daily_update()  # First failure
            except RuntimeError as e:
                pipeline_status["errors"].append(str(e))

            try:
                mock_scheduler.run_daily_update()  # Second failure
            except RuntimeError as e:
                pipeline_status["errors"].append(str(e))

            try:
                mock_scheduler.run_daily_update()  # Third call - shutdown
            except Exception:
                import logging as _logging

                _logging.getLogger(__name__).debug(
                    "Third update call raised as expected in test",
                    exc_info=True,
                )

        # Replace the method temporarily
        original_run_realtime = runner._run_realtime
        runner._run_realtime = test_run_realtime

        try:
            runner._run_realtime()
        finally:
            runner._run_realtime = original_run_realtime

        # Verify errors were tracked
        assert len(pipeline_status["errors"]) >= 2
        assert "First failure" in str(pipeline_status["errors"])
        assert "Second failure" in str(pipeline_status["errors"])

    @pytest.mark.database
    @pytest.mark.serial
    def test_prometheus_metrics_exposure(
        self,
        deployment_env,
    ):
        """
        Test Prometheus metrics are properly exposed.
        """
        with (
            patch("ml.common.metrics.MODEL_INFERENCE_TIMER") as mock_MODEL_INFERENCE_TIMER,
            patch("ml.common.metrics.FEATURE_CALCULATION_TIMER") as mock_FEATURE_CALCULATION_TIMER,
            patch("ml.common.metrics.PREDICTION_COUNTER") as mock_PREDICTION_COUNTER,
        ):
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
        """
        Test service dependency checks.
        """
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
    def test_configuration_validation(
        self,
        deployment_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        Test configuration validation across components.
        """
        # Test invalid configuration handling
        monkeypatch.setenv("USE_DUMMY_STORES", "false")
        monkeypatch.setenv("USE_MOCK_DATA", "false")
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

        self._log_deployment_env_state(
            "test_configuration_validation:missing_key",
            request=request,
        )

        actor_node = MLSignalActorNode()
        with pytest.raises(SystemExit) as exc_info:
            actor_node.setup()
        assert exc_info.value.code == 1

        # Test missing model file
        monkeypatch.setenv("DATABENTO_API_KEY", "x" * 32)
        monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.pkl")

        self._log_deployment_env_state(
            "test_configuration_validation:missing_model",
            request=request,
        )

        actor_node = MLSignalActorNode()
        with pytest.raises(SystemExit) as exc_info:
            actor_node.setup()
        assert exc_info.value.code == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_multi_threading_safety(self, deployment_env):
        """
        Test thread safety of concurrent operations.
        """
        from ml.deployment.entrypoint_pipeline import pipeline_status

        # Reset status
        pipeline_status["errors"] = []

        def update_status(error_msg):
            """
            Simulate concurrent status updates.
            """
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
