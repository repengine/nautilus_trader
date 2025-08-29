#!/usr/bin/env python3
"""
Unit tests for ML Trading Strategy deployment entrypoint.

Tests container startup, dry run mode, signal consumption, and risk management.
"""

from __future__ import annotations

import asyncio
import os
import signal
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.deployment.entrypoint_strategy import MLStrategyNode
from ml.deployment.entrypoint_strategy import main


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.unit
class TestMLStrategyNode:
    """Test MLStrategyNode container entrypoint."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        """Clean environment for isolated testing."""
        for key in list(os.environ.keys()):
            if key.startswith(("ML_", "DATABENTO_", "DB_", "STRATEGY_", "POSITION_", "MIN_", "MAX_", "STOP_", "TAKE_", "USE_", "PERSIST_", "EXECUTE_")):
                monkeypatch.delenv(key, raising=False)

    @pytest.fixture
    def valid_env(self, monkeypatch):
        """Set up valid environment variables for testing."""
        monkeypatch.setenv("DB_CONNECTION", "postgresql://test:test@localhost:5432/test")
        monkeypatch.setenv("STRATEGY_ID", "MLStrategy-TEST-001")
        monkeypatch.setenv("ML_SIGNAL_SOURCE", "MLSignalActor-001")
        monkeypatch.setenv("INSTRUMENT_ID", "BTC-USDT.DATABENTO")
        monkeypatch.setenv("EXECUTE_TRADES", "false")  # Dry run by default
        monkeypatch.setenv("POSITION_SIZE_PCT", "0.02")
        monkeypatch.setenv("MIN_CONFIDENCE", "0.6")
        monkeypatch.setenv("MAX_POSITIONS", "3")
        monkeypatch.setenv("STOP_LOSS_PCT", "0.02")
        monkeypatch.setenv("TAKE_PROFIT_PCT", "0.04")
        monkeypatch.setenv("USE_STRATEGY_STORE", "true")
        monkeypatch.setenv("PERSIST_ALL_SIGNALS", "true")

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_with_dry_run_mode(self, valid_env):
        """Test setup in dry run mode (default)."""
        node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_strategy.TradingNode") as mock_node_class:
            with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy") as mock_strategy_class:
                mock_node = Mock()
                mock_node_class.return_value = mock_node
                mock_strategy = Mock()
                mock_strategy_class.return_value = mock_strategy

                node.setup()

                # Verify strategy was created with dry run mode
                strategy_config = mock_strategy_class.call_args[1]["config"]
                assert strategy_config.execute_trades is False
                assert strategy_config.strategy_id == "MLStrategy-TEST-001"
                assert strategy_config.ml_signal_source == "MLSignalActor-001"

                # Verify strategy was added to trader
                mock_node.trader.add_strategy.assert_called_once_with(mock_strategy)

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_with_live_mode(self, valid_env, monkeypatch):
        """Test setup in live trading mode."""
        monkeypatch.setenv("EXECUTE_TRADES", "true")

        node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_strategy.TradingNode") as mock_node_class:
            with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy") as mock_strategy_class:
                mock_node = Mock()
                mock_node_class.return_value = mock_node
                mock_strategy = Mock()
                mock_strategy_class.return_value = mock_strategy

                node.setup()

                # Verify strategy was created with live mode
                strategy_config = mock_strategy_class.call_args[1]["config"]
                assert strategy_config.execute_trades is True

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_risk_parameters(self, valid_env):
        """Test risk parameters are correctly parsed."""
        node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_strategy.TradingNode"):
            with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy") as mock_strategy_class:
                mock_strategy = Mock()
                mock_strategy_class.return_value = mock_strategy

                node.setup()

                # Verify risk parameters
                strategy_config = mock_strategy_class.call_args[1]["config"]
                assert strategy_config.position_size_pct == 0.02
                assert strategy_config.min_confidence == 0.6
                assert strategy_config.max_positions == 3
                assert strategy_config.stop_loss_pct == 0.02
                assert strategy_config.take_profit_pct == 0.04

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_with_strategy_store(self, valid_env):
        """Test setup with strategy store enabled."""
        node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_strategy.TradingNode"):
            with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy") as mock_strategy_class:
                mock_strategy = Mock()
                mock_strategy_class.return_value = mock_strategy

                node.setup()

                # Verify strategy store configuration
                strategy_config = mock_strategy_class.call_args[1]["config"]
                assert strategy_config.use_strategy_store is True
                assert strategy_config.strategy_store_config is not None
                assert strategy_config.strategy_store_config["connection_string"] == "postgresql://test:test@localhost:5432/test"
                assert strategy_config.strategy_store_config["batch_size"] == 100
                assert strategy_config.strategy_store_config["flush_interval_ms"] == 1000
                assert strategy_config.persist_all_signals is True

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_without_strategy_store(self, valid_env, monkeypatch):
        """Test setup with strategy store disabled."""
        monkeypatch.setenv("USE_STRATEGY_STORE", "false")

        node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_strategy.TradingNode"):
            with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy") as mock_strategy_class:
                mock_strategy = Mock()
                mock_strategy_class.return_value = mock_strategy

                node.setup()

                # Verify no strategy store config
                strategy_config = mock_strategy_class.call_args[1]["config"]
                assert strategy_config.use_strategy_store is False
                assert strategy_config.strategy_store_config is None

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_with_databento_api_key(self, valid_env, monkeypatch):
        """Test setup with Databento API key for market data."""
        monkeypatch.setenv("DATABENTO_API_KEY", "test_api_key")

        node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_strategy.TradingNode") as mock_node_class:
            with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy"):
                mock_node = Mock()
                mock_node_class.return_value = mock_node

                node.setup()

                # Verify Databento client was configured
                node_config = mock_node_class.call_args[1]["config"]
                assert "DATABENTO" in node_config.data_clients
                data_config = node_config.data_clients["DATABENTO"]
                assert data_config.api_key == "test_api_key"

    @pytest.mark.database
    @pytest.mark.serial
    def test_setup_without_databento_api_key(self, valid_env):
        """Test setup without Databento API key."""
        node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_strategy.TradingNode") as mock_node_class:
            with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy"):
                mock_node = Mock()
                mock_node_class.return_value = mock_node

                node.setup()

                # Verify no data clients configured
                node_config = mock_node_class.call_args[1]["config"]
                assert len(node_config.data_clients) == 0

    @pytest.mark.asyncio
    async def test_run_successful(self, valid_env):
        """Test successful run of strategy node."""
        node = MLStrategyNode()

        # Mock the trading node
        mock_node = AsyncMock()
        mock_node.run_async = AsyncMock(side_effect=asyncio.CancelledError)
        node.node = mock_node

        await node.run()

        # Verify node was run
        mock_node.run_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_handles_error(self, valid_env):
        """Test run handles errors gracefully."""
        node = MLStrategyNode()

        # Mock the trading node to raise an error
        mock_node = AsyncMock()
        mock_node.run_async = AsyncMock(side_effect=RuntimeError("Test error"))
        mock_node.stop_async = AsyncMock()
        node.node = mock_node

        await node.run()

        # Verify shutdown was called
        mock_node.stop_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_without_node_raises(self):
        """Test run raises error when node not initialized."""
        node = MLStrategyNode()
        node.node = None

        with patch.object(node, "shutdown", new_callable=AsyncMock) as mock_shutdown:
            await node.run()
            mock_shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_signal(self, valid_env):
        """Test graceful shutdown with signal."""
        node = MLStrategyNode()

        # Mock the trading node and strategy
        mock_node = AsyncMock()
        mock_node.stop_async = AsyncMock()
        mock_strategy = Mock()
        mock_strategy._signals_received = 10
        mock_strategy._dry_run_trades = 5
        mock_strategy._config = Mock(execute_trades=False)
        mock_node.trader.strategies.return_value = {"test": mock_strategy}
        node.node = mock_node
        node.running = True

        await node.shutdown(signal.SIGTERM)

        assert node.running is False
        mock_node.stop_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_prints_statistics(self, valid_env, capsys):
        """Test shutdown prints final statistics."""
        node = MLStrategyNode()

        # Mock the trading node and strategy with statistics
        mock_node = AsyncMock()
        mock_node.stop_async = AsyncMock()
        mock_strategy = Mock()
        mock_strategy._signals_received = 15
        mock_strategy._dry_run_trades = 8
        mock_strategy._config = Mock(execute_trades=False)
        mock_node.trader.strategies.return_value = {"test": mock_strategy}
        node.node = mock_node

        await node.shutdown()

        captured = capsys.readouterr()
        assert "FINAL STATISTICS" in captured.out
        assert "Signals Received: 15" in captured.out
        assert "Dry Run Trades: 8" in captured.out
        assert "Execute Trades Setting: False" in captured.out

    @pytest.mark.asyncio
    async def test_signal_handlers_setup(self, valid_env):
        """Test signal handlers are properly set up."""
        node = MLStrategyNode()

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
    def test_environment_variable_parsing(self, monkeypatch):
        """Test correct parsing of various environment variable formats."""
        # Set custom values
        monkeypatch.setenv("STRATEGY_ID", "CustomStrategy-999")
        monkeypatch.setenv("ML_SIGNAL_SOURCE", "CustomActor-456")
        monkeypatch.setenv("INSTRUMENT_ID", "ETH-USD.BINANCE")
        monkeypatch.setenv("EXECUTE_TRADES", "FALSE")  # Test case insensitive
        monkeypatch.setenv("POSITION_SIZE_PCT", "0.05")
        monkeypatch.setenv("MIN_CONFIDENCE", "0.75")
        monkeypatch.setenv("MAX_POSITIONS", "5")
        monkeypatch.setenv("STOP_LOSS_PCT", "0.03")
        monkeypatch.setenv("TAKE_PROFIT_PCT", "0.06")

        node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_strategy.TradingNode"):
            with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy") as mock_strategy_class:
                mock_strategy = Mock()
                mock_strategy_class.return_value = mock_strategy

                node.setup()

                # Verify parsed values
                strategy_config = mock_strategy_class.call_args[1]["config"]
                assert strategy_config.strategy_id == "CustomStrategy-999"
                assert strategy_config.ml_signal_source == "CustomActor-456"
                assert str(strategy_config.instrument_id) == "ETH-USD.BINANCE"
                assert strategy_config.execute_trades is False
                assert strategy_config.position_size_pct == 0.05
                assert strategy_config.min_confidence == 0.75
                assert strategy_config.max_positions == 5
                assert strategy_config.stop_loss_pct == 0.03
                assert strategy_config.take_profit_pct == 0.06

    @pytest.mark.database
    @pytest.mark.serial
    def test_dry_run_mode_output(self, valid_env, capsys):
        """Test dry run mode warning is displayed."""
        node = MLStrategyNode()

        with patch("ml.deployment.entrypoint_strategy.TradingNode"):
            with patch("ml.deployment.entrypoint_strategy.MLTradingStrategy"):
                node.setup()

                captured = capsys.readouterr()
                assert "DRY RUN MODE ACTIVE" in captured.out
                assert "will NOT submit actual orders" in captured.out


@pytest.mark.database
@pytest.mark.serial
class TestMainFunction:
    """Test the main entry point function."""

    @pytest.mark.database
    @pytest.mark.serial
    def test_main_successful_run(self, valid_env):
        """Test successful main function execution."""
        with patch("ml.deployment.entrypoint_strategy.MLStrategyNode") as mock_node_class:
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
        with patch("ml.deployment.entrypoint_strategy.MLStrategyNode") as mock_node_class:
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
        with patch("ml.deployment.entrypoint_strategy.MLStrategyNode") as mock_node_class:
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
        with patch("ml.deployment.entrypoint_strategy.MLStrategyNode") as mock_node_class:
            with patch("asyncio.run"):
                mock_node = Mock()
                mock_node_class.return_value = mock_node

                node = MLStrategyNode()
                node.setup()

                captured = capsys.readouterr()
                assert "ML TRADING STRATEGY - CONTAINER MODE" in captured.out
                assert "Strategy ID:" in captured.out
                assert "Signal Source:" in captured.out
                assert "Execute Trades:" in captured.out
