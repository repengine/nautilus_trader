#!/usr/bin/env python3
"""
Unit tests for ML Signal Actor deployment entrypoint.

Tests container startup, configuration, health checks, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.deployment.entrypoint_actor import MLSignalActorNode
from ml.deployment.entrypoint_actor import main


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.unit
class TestMLSignalActorNode:
    """Test MLSignalActorNode container entrypoint."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        """Clean environment for isolated testing."""
        # Remove any existing ML environment variables
        for key in list(os.environ.keys()):
            if key.startswith(("ML_", "DATABENTO_", "DB_")):
                monkeypatch.delenv(key, raising=False)

    @pytest.fixture
    def valid_env(self, monkeypatch, tmp_path):
        """Set up valid environment variables for testing."""
        # Create a temporary model file
        model_path = tmp_path / "model.pkl"
        model_path.write_text("dummy_model")

        monkeypatch.setenv("DATABENTO_API_KEY", "test_api_key")
        monkeypatch.setenv("DB_CONNECTION", "postgresql://test:test@localhost:5432/test")
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.setenv("INSTRUMENT_ID", "BTC-USDT.DATABENTO")
        monkeypatch.setenv("BAR_TYPE", "BTC-USDT.DATABENTO-1-MINUTE-LAST-EXTERNAL")
        monkeypatch.setenv("ACTOR_ID", "MLSignalActor-TEST")
        monkeypatch.setenv("USE_DUMMY_STORES", "true")

        return model_path

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_with_valid_config(self, valid_env):
        """Test setup with valid configuration."""
        node = MLSignalActorNode()

        with patch("ml.deployment.entrypoint_actor.TradingNode") as mock_node_class:
            with patch("ml.deployment.entrypoint_actor.MLSignalActor") as mock_actor_class:
                mock_node = Mock()
                mock_node_class.return_value = mock_node
                mock_actor = Mock()
                mock_actor_class.return_value = mock_actor

                node.setup()

                # Verify node was created with correct config
                mock_node_class.assert_called_once()
                config = mock_node_class.call_args[1]["config"]
                assert config.trader_id.value == "ML-ACTOR-001"

                # Verify actor was created and added
                mock_actor_class.assert_called_once()
                actor_config = mock_actor_class.call_args[1]["config"]
                assert actor_config.model_path == str(valid_env)
                assert actor_config.component_id == "MLSignalActor-TEST"
                assert actor_config.use_dummy_stores is True

                # Verify actor was added to trader
                mock_node.trader.add_actor.assert_called_once_with(mock_actor)

                # Verify subscription
                mock_actor.subscribe_bars.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_without_api_key_exits(self, monkeypatch, tmp_path):
        """Test that setup exits when DATABENTO_API_KEY is missing."""
        model_path = tmp_path / "model.pkl"
        model_path.write_text("dummy")
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

        node = MLSignalActorNode()

        with pytest.raises(SystemExit) as exc_info:
            node.setup()

        assert exc_info.value.code == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_without_model_exits(self, monkeypatch):
        """Test that setup exits when model file doesn't exist."""
        monkeypatch.setenv("DATABENTO_API_KEY", "test_key")
        monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.pkl")

        node = MLSignalActorNode()

        with pytest.raises(SystemExit) as exc_info:
            node.setup()

        assert exc_info.value.code == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_with_database_connection(self, valid_env, monkeypatch):
        """Test setup with database connection (non-dummy stores)."""
        monkeypatch.setenv("USE_DUMMY_STORES", "false")
        monkeypatch.setenv("DB_CONNECTION", "postgresql://user:pass@host:5432/db")

        node = MLSignalActorNode()

        with patch("ml.deployment.entrypoint_actor.TradingNode") as mock_node_class:
            with patch("ml.deployment.entrypoint_actor.MLSignalActor") as mock_actor_class:
                mock_node = Mock()
                mock_node_class.return_value = mock_node
                mock_actor = Mock()
                mock_actor_class.return_value = mock_actor

                node.setup()

                # Verify actor config includes database connection
                actor_config = mock_actor_class.call_args[1]["config"]
                assert actor_config.db_connection == "postgresql://user:pass@host:5432/db"
                assert actor_config.use_dummy_stores is False

    @pytest.mark.asyncio
    async def test_run_successful(self, valid_env):
        """Test successful run of actor node."""
        node = MLSignalActorNode()

        # Mock the trading node
        mock_node = AsyncMock()
        mock_node.run_async = AsyncMock(side_effect=asyncio.CancelledError)
        node.node = mock_node

        # Run should handle CancelledError gracefully
        await node.run()

        # Verify node was run
        mock_node.run_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_handles_error(self, valid_env):
        """Test run handles errors gracefully."""
        node = MLSignalActorNode()

        # Mock the trading node to raise an error
        mock_node = AsyncMock()
        mock_node.run_async = AsyncMock(side_effect=RuntimeError("Test error"))
        mock_node.stop_async = AsyncMock()
        node.node = mock_node

        # Run should handle error and shutdown
        await node.run()

        # Verify shutdown was called
        mock_node.stop_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_without_node_raises(self):
        """Test run raises error when node not initialized."""
        node = MLSignalActorNode()
        node.node = None

        with patch.object(node, "shutdown", new_callable=AsyncMock) as mock_shutdown:
            await node.run()
            mock_shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_signal(self, valid_env):
        """Test graceful shutdown with signal."""
        node = MLSignalActorNode()

        # Mock the trading node
        mock_node = AsyncMock()
        mock_node.stop_async = AsyncMock()
        node.node = mock_node
        node.running = True

        # Test shutdown with SIGTERM
        await node.shutdown(signal.SIGTERM)

        assert node.running is False
        mock_node.stop_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_without_signal(self, valid_env):
        """Test graceful shutdown without signal."""
        node = MLSignalActorNode()

        # Mock the trading node
        mock_node = AsyncMock()
        mock_node.stop_async = AsyncMock()
        node.node = mock_node
        node.running = True

        # Test shutdown without signal
        await node.shutdown()

        assert node.running is False
        mock_node.stop_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_dispose(self, valid_env):
        """Test shutdown calls dispose_async if available."""
        node = MLSignalActorNode()

        # Mock the trading node with dispose_async
        mock_node = AsyncMock()
        mock_node.stop_async = AsyncMock()
        mock_node.dispose_async = AsyncMock()
        node.node = mock_node

        await node.shutdown()

        mock_node.stop_async.assert_called_once()
        mock_node.dispose_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_signal_handlers_setup(self, valid_env):
        """Test signal handlers are properly set up."""
        node = MLSignalActorNode()

        # Mock the trading node
        mock_node = AsyncMock()
        mock_node.run_async = AsyncMock(side_effect=asyncio.CancelledError)
        node.node = mock_node

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop

            await node.run()

            # Verify signal handlers were added
            assert mock_loop.add_signal_handler.call_count == 2
            calls = mock_loop.add_signal_handler.call_args_list
            signals_registered = [call[0][0] for call in calls]
            assert signal.SIGTERM in signals_registered
            assert signal.SIGINT in signals_registered

    @pytest.mark.database
    @pytest.mark.serial
    def test_environment_variable_parsing(self, monkeypatch, tmp_path):
        """Test correct parsing of environment variables."""
        model_path = tmp_path / "model.pkl"
        model_path.write_text("dummy")

        # Set various environment variable formats
        monkeypatch.setenv("DATABENTO_API_KEY", "test_key")
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.setenv("INSTRUMENT_ID", "ETH-USD.BINANCE")
        monkeypatch.setenv("BAR_TYPE", "ETH-USD.BINANCE-5-MINUTE-LAST-EXTERNAL")
        monkeypatch.setenv("ACTOR_ID", "CustomActor-123")
        monkeypatch.setenv("USE_DUMMY_STORES", "TRUE")  # Test case insensitive

        node = MLSignalActorNode()

        with patch("ml.deployment.entrypoint_actor.TradingNode") as mock_node_class:
            with patch("ml.deployment.entrypoint_actor.MLSignalActor") as mock_actor_class:
                mock_node = Mock()
                mock_node_class.return_value = mock_node
                mock_actor = Mock()
                mock_actor_class.return_value = mock_actor

                node.setup()

                # Verify parsed values
                actor_config = mock_actor_class.call_args[1]["config"]
                assert str(actor_config.instrument_id) == "ETH-USD.BINANCE"
                assert "ETH-USD.BINANCE" in str(actor_config.bar_type)
                assert "5-MINUTE" in str(actor_config.bar_type)
                assert actor_config.component_id == "CustomActor-123"
                assert actor_config.use_dummy_stores is True

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_config_creation(self, valid_env):
        """Test feature configuration is properly created."""
        node = MLSignalActorNode()

        with patch("ml.deployment.entrypoint_actor.TradingNode") as mock_node_class:
            with patch("ml.deployment.entrypoint_actor.MLSignalActor") as mock_actor_class:
                mock_node = Mock()
                mock_node_class.return_value = mock_node
                mock_actor = Mock()
                mock_actor_class.return_value = mock_actor

                node.setup()

                # Verify feature config
                actor_config = mock_actor_class.call_args[1]["config"]
                feature_config = actor_config.feature_config
                assert feature_config.lookback_window == 20
                assert feature_config.normalize_features is True
                assert feature_config.fill_missing_with == 0.0
                assert "sma" in feature_config.indicators
                assert "rsi" in feature_config.indicators
                assert "bbands" in feature_config.indicators

    @pytest.mark.database
    @pytest.mark.serial
    def test_databento_config_creation(self, valid_env):
        """Test Databento configuration is properly created."""
        node = MLSignalActorNode()

        with patch("ml.deployment.entrypoint_actor.TradingNode") as mock_node_class:
            with patch("ml.deployment.entrypoint_actor.MLSignalActor"):
                mock_node = Mock()
                mock_node_class.return_value = mock_node

                node.setup()

                # Verify Databento config
                node_config = mock_node_class.call_args[1]["config"]
                data_config = node_config.data_clients["DATABENTO"]
                assert data_config.api_key == "test_api_key"
                assert data_config.http_gateway == "https://hist.databento.com"
                assert data_config.live_gateway == "wss://stream.databento.com"


@pytest.mark.database
@pytest.mark.serial
class TestMainFunction:
    """Test the main entry point function."""

    @pytest.fixture
    def valid_env(self, monkeypatch, tmp_path):
        """Set up valid environment variables for testing."""
        # Create a temporary model file
        model_path = tmp_path / "model.pkl"
        model_path.write_text("dummy_model")

        monkeypatch.setenv("DATABENTO_API_KEY", "test_api_key")
        monkeypatch.setenv("DB_CONNECTION", "postgresql://test:test@localhost:5432/test")
        monkeypatch.setenv("MODEL_PATH", str(model_path))
        monkeypatch.setenv("INSTRUMENT_ID", "BTC-USDT.DATABENTO")
        monkeypatch.setenv("BAR_TYPE", "BTC-USDT.DATABENTO-1-MINUTE-LAST-EXTERNAL")
        monkeypatch.setenv("ACTOR_ID", "MLSignalActor-TEST")
        monkeypatch.setenv("USE_DUMMY_STORES", "true")

        return model_path

    @pytest.mark.database
    @pytest.mark.serial
    def test_main_successful_run(self, valid_env):
        """Test successful main function execution."""
        with patch("ml.deployment.entrypoint_actor.MLSignalActorNode") as mock_node_class:
            with patch("asyncio.run") as mock_asyncio_run:
                mock_node = Mock()
                mock_node_class.return_value = mock_node

                main()

                # Verify node was created and set up
                mock_node_class.assert_called_once()
                mock_node.setup.assert_called_once()
                mock_asyncio_run.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_main_handles_keyboard_interrupt(self, valid_env):
        """Test main handles KeyboardInterrupt gracefully."""
        with patch("ml.deployment.entrypoint_actor.MLSignalActorNode") as mock_node_class:
            with patch("asyncio.run", side_effect=KeyboardInterrupt):
                mock_node = Mock()
                mock_node_class.return_value = mock_node

                # Should not raise, just print message
                main()

                mock_node.setup.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_main_handles_fatal_error(self, valid_env):
        """Test main handles fatal errors with sys.exit."""
        with patch("ml.deployment.entrypoint_actor.MLSignalActorNode") as mock_node_class:
            with patch("asyncio.run", side_effect=RuntimeError("Fatal error")):
                mock_node = Mock()
                mock_node_class.return_value = mock_node

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_main_prints_startup_info(self, valid_env, capsys):
        """Test main prints startup information."""
        with patch("ml.deployment.entrypoint_actor.MLSignalActorNode") as mock_node_class:
            with patch("asyncio.run"):
                mock_node = Mock()
                mock_node_class.return_value = mock_node

                node = MLSignalActorNode()
                node.setup()

                captured = capsys.readouterr()
                assert "ML SIGNAL ACTOR - CONTAINER MODE" in captured.out
                assert "Actor ID:" in captured.out
                assert "Instrument:" in captured.out
                assert "ML Signal Actor configured and ready" in captured.out
