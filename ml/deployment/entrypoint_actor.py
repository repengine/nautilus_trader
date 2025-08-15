#!/usr/bin/env python
"""
Entrypoint for ML Signal Actor container.

This script runs the ML Signal Actor with Databento data feed
and PostgreSQL persistence in a containerized environment.
"""

import os
import sys
import asyncio
import signal
from pathlib import Path

from nautilus_trader.adapters.databento.config import DatabentoDataClientConfig
from nautilus_trader.adapters.databento.factories import DatabentoLiveDataClientFactory
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId, TraderId

from ml.actors.signal import MLSignalActor
from ml.config.actors import MLSignalActorConfig
from ml.config.base import MLFeatureConfig


class MLSignalActorNode:
    """
    Container-ready ML Signal Actor node.
    """
    
    def __init__(self):
        self.node = None
        self.running = False
        
    def setup(self):
        """
        Set up the trading node with ML Signal Actor.
        """
        # Get configuration from environment
        db_connection = os.getenv("DB_CONNECTION", "postgresql://postgres:postgres@localhost:5432/nautilus")
        databento_api_key = os.getenv("DATABENTO_API_KEY")
        databento_dataset = os.getenv("DATABENTO_DATASET", "GLBX.MDP3")
        
        model_path = os.getenv("MODEL_PATH", "/app/models/model.pkl")
        instrument_str = os.getenv("INSTRUMENT_ID", "BTC-USDT.DATABENTO")
        bar_type_str = os.getenv("BAR_TYPE", "BTC-USDT.DATABENTO-1-MINUTE")
        actor_id = os.getenv("ACTOR_ID", "MLSignalActor-001")
        use_dummy_stores = os.getenv("USE_DUMMY_STORES", "false").lower() == "true"
        
        # Parse identifiers
        instrument_id = InstrumentId.from_str(instrument_str)
        bar_type = BarType.from_str(bar_type_str)
        
        # Check for API key
        if not databento_api_key:
            print("ERROR: DATABENTO_API_KEY environment variable not set")
            print("Please set your Databento API key to connect to market data")
            sys.exit(1)
        
        # Check model exists
        if not Path(model_path).exists():
            print(f"WARNING: Model not found at {model_path}")
            print("Using dummy model for demonstration...")
            # Create a dummy model if needed
            self._create_dummy_model(model_path)
        
        print("=" * 80)
        print("ML SIGNAL ACTOR - CONTAINER MODE")
        print("=" * 80)
        print(f"Database: {db_connection.split('@')[1] if '@' in db_connection else 'local'}")
        print(f"Model: {model_path}")
        print(f"Instrument: {instrument_id}")
        print(f"Bar Type: {bar_type}")
        print(f"Actor ID: {actor_id}")
        print(f"Dummy Stores: {use_dummy_stores}")
        print("=" * 80)
        
        # Feature configuration
        feature_config = MLFeatureConfig(
            lookback_window=20,
            indicators={
                "sma": {"period": 10},
                "rsi": {"period": 14},
                "bbands": {"period": 20, "std": 2},
            },
            normalize_features=True,
            fill_missing_with=0.0,
        )
        
        # Actor configuration
        actor_config = MLSignalActorConfig(
            model_id=actor_id.replace("Actor", "Model"),
            component_id=actor_id,
            model_path=model_path,
            bar_type=bar_type,
            instrument_id=instrument_id,
            db_connection=db_connection if not use_dummy_stores else None,
            prediction_threshold=0.5,
            max_inference_latency_ms=5.0,
            feature_config=feature_config,
            warm_up_period=20,
            publish_signals=True,
            log_predictions=True,
            use_dummy_stores=use_dummy_stores,
            enable_health_monitoring=True,
        )
        
        # Databento data client configuration
        data_config = DatabentoDataClientConfig(
            api_key=databento_api_key,
            http_gateway="https://hist.databento.com",
            streaming_gateway="wss://stream.databento.com",
            historical_gateway="https://hist.databento.com",
            dataset=databento_dataset,
        )
        
        # Trading node configuration
        node_config = TradingNodeConfig(
            trader_id=TraderId("ML-ACTOR-001"),
            data_clients={
                "DATABENTO": data_config,
            },
            exec_clients={},  # No execution for signal actor
            default_data_client="DATABENTO",
            logging={
                "level": os.getenv("LOG_LEVEL", "INFO"),
            },
            timeout_connection=30.0,
            timeout_reconciliation=10.0,
            timeout_portfolio=10.0,
            timeout_disconnection=10.0,
        )
        
        # Create trading node
        self.node = TradingNode(config=node_config)
        
        # Add ML Signal Actor
        actor = MLSignalActor(config=actor_config)
        self.node.add_actor(actor)
        
        # Subscribe to market data
        actor.subscribe_bars(bar_type)
        
        print("\nML Signal Actor configured and ready")
        print("Waiting for market data...")
        
    def _create_dummy_model(self, model_path: str):
        """
        Create a dummy model for testing if none exists.
        """
        import pickle
        import numpy as np
        
        class DummyModel:
            def predict(self, X):
                return np.random.rand(len(X) if len(X.shape) > 1 else 1)
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(DummyModel(), f)
        print(f"Created dummy model at {model_path}")
    
    async def run(self):
        """
        Run the actor node.
        """
        self.running = True
        
        # Set up graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))
        
        # Run the node
        try:
            await self.node.run_async()
        except Exception as e:
            print(f"Error running node: {e}")
            await self.shutdown()
    
    async def shutdown(self, sig=None):
        """
        Gracefully shutdown the node.
        """
        if sig:
            print(f"\nReceived signal {sig.name}, shutting down...")
        else:
            print("\nShutting down...")
        
        self.running = False
        
        if self.node:
            await self.node.stop_async()
            await self.node.dispose_async()
        
        print("ML Signal Actor shutdown complete")


def main():
    """
    Main entry point.
    """
    # Create and run the actor node
    actor_node = MLSignalActorNode()
    actor_node.setup()
    
    # Run async event loop
    try:
        asyncio.run(actor_node.run())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()