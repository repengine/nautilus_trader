from __future__ import annotations

from unittest.mock import Mock
from unittest.mock import patch

from ml.deployment.entrypoint_actor import MLSignalActorNode
from ml.deployment.entrypoint_strategy import MLStrategyNode


def test_setup_falls_back_to_mock_data_when_key_invalid(monkeypatch, tmp_path) -> None:
    """Ensure dummy-store deployments ignore malformed Databento keys."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"\0")  # Minimal ONNX sentinel

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("DATABENTO_API_KEY", "short-key")
    monkeypatch.setenv("USE_DUMMY_STORES", "true")
    monkeypatch.setenv("ML_LIVE_RECORD_ENABLE", "0")

    actor_node = MLSignalActorNode()

    with patch("ml.deployment.entrypoint_actor.TradingNode") as mock_trading_node, patch(
        "ml.deployment.entrypoint_actor.MLSignalActor",
    ) as mock_signal_actor:
        mock_node = Mock()
        mock_node.trader = Mock()
        mock_trading_node.return_value = mock_node
        mock_signal_actor.return_value = Mock()

        actor_node.setup()

    args, kwargs = mock_trading_node.call_args
    node_config = kwargs.get("config") or args[0]
    assert node_config.data_clients == {}
    mock_node.add_data_client_factory.assert_not_called()


def test_strategy_does_not_register_databento_with_invalid_key(monkeypatch) -> None:
    """Strategy node should not attach live data clients when the key is malformed."""
    monkeypatch.setenv("DATABENTO_API_KEY", "short-key")
    monkeypatch.setenv("USE_STRATEGY_STORE", "false")
    monkeypatch.delenv("EXECUTE_TRADES", raising=False)

    strategy_node = MLStrategyNode()

    with patch("ml.deployment.entrypoint_strategy.TradingNode") as mock_trading_node, patch(
        "ml.deployment.entrypoint_strategy.MLTradingStrategy",
    ) as mock_strategy:
        mock_node = Mock()
        mock_node.trader = Mock()
        mock_trading_node.return_value = mock_node
        mock_strategy.return_value = Mock()

        strategy_node.setup()

    args, kwargs = mock_trading_node.call_args
    node_config = kwargs.get("config") or args[0]
    assert node_config.data_clients == {}
