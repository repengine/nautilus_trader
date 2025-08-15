"""
Integration tests for store persistence.

Tests that data is actually persisted to stores and can be retrieved.
"""

import tempfile
from pathlib import Path

import pytest

from ml.registry.persistence import PersistenceConfig
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


class TestStorePersistence:
    """
    Test that stores actually persist data.
    """

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def persistence_config(self, temp_db_path):
        """Create persistence config for testing."""
        return PersistenceConfig(
            backend="sqlite",
            connection_string=f"sqlite:///{temp_db_path}",
        )

    def test_feature_store_persistence(self, temp_db_path):
        """
        Test that FeatureStore actually persists and retrieves features.
        """
        # Create store
        store = FeatureStore(
            connection_string=f"sqlite:///{temp_db_path}",
        )

        # Write features
        features = {
            "feature_0": 0.5,
            "feature_1": 0.7,
            "feature_2": -0.3,
        }

        store.write_features(
            feature_set_id="test_set",
            instrument_id="EUR/USD",
            features=features,
            ts_event=1000000000,
            ts_init=1000000001,
        )

        # Flush (should be no-op but test it)
        store.flush()

        # Create new store instance to verify persistence
        store2 = FeatureStore(
            connection_string=f"sqlite:///{temp_db_path}",
        )

        # Read back features
        with store2.engine.connect() as conn:
            from sqlalchemy import select
            result = conn.execute(
                select([store2.feature_values_table]).where(
                    store2.feature_values_table.c.feature_set_id == "test_set"
                )
            ).fetchone()

        assert result is not None
        assert result["instrument_id"] == "EUR/USD"
        assert result["ts_event"] == 1000000000

        # Check health
        assert store2.is_healthy()

    def test_model_store_persistence(self, persistence_config, temp_db_path):
        """
        Test that ModelStore actually persists and retrieves predictions.
        """
        # Create store
        store = ModelStore(persistence_config=persistence_config)

        # Write predictions
        store.write_prediction(
            model_id="test_model",
            instrument_id="EUR/USD",
            prediction=0.65,
            confidence=0.8,
            features={"feature_0": 0.5},
            inference_time_ms=2.5,
            ts_event=1000000000,
        )

        # Write more to trigger batch
        for i in range(10):
            store.write_prediction(
                model_id="test_model",
                instrument_id="EUR/USD",
                prediction=0.5 + i * 0.01,
                confidence=0.7 + i * 0.01,
                features={"feature_0": 0.5 + i * 0.1},
                inference_time_ms=2.0 + i * 0.1,
                ts_event=1000000000 + i * 1000,
            )

        # Force flush
        store.flush()

        # Create new store instance to verify persistence
        store2 = ModelStore(persistence_config=persistence_config)

        # Read back predictions
        predictions = store2.get_predictions(
            model_id="test_model",
            start_ns=999999999,
            end_ns=2000000000,
        )

        assert len(predictions) >= 11  # Should have all predictions

        # Check health
        assert store2.is_healthy()

    def test_strategy_store_persistence(self, persistence_config, temp_db_path):
        """
        Test that StrategyStore actually persists and retrieves signals.
        """
        # Create store
        store = StrategyStore(persistence_config=persistence_config)

        # Write signals
        store.write_signal(
            strategy_id="test_strategy",
            instrument_id="EUR/USD",
            signal_type="buy",
            strength=0.8,
            model_predictions={"model_1": 0.7},
            risk_metrics={"sharpe": 1.5},
            execution_params={"threshold": 0.6},
            ts_event=1000000000,
        )

        # Write more signals
        for i in range(5):
            store.write_signal(
                strategy_id="test_strategy",
                instrument_id="EUR/USD",
                signal_type="sell" if i % 2 else "buy",
                strength=0.5 + i * 0.1,
                model_predictions={"model_1": 0.4 + i * 0.1},
                risk_metrics={"sharpe": 1.0 + i * 0.2},
                execution_params={"threshold": 0.5},
                ts_event=1000000000 + i * 1000,
            )

        # Force flush
        store.flush()

        # Create new store instance to verify persistence
        store2 = StrategyStore(persistence_config=persistence_config)

        # Read back signals
        signals = store2.get_signals(
            strategy_id="test_strategy",
            start_ns=999999999,
            end_ns=2000000000,
        )

        assert len(signals) >= 6  # Should have all signals

        # Check health
        assert store2.is_healthy()

    def test_store_failure_handling(self):
        """
        Test that stores handle connection failures gracefully.
        """
        # Create config with invalid connection
        bad_config = PersistenceConfig(
            backend="postgres",
            connection_string="postgresql://invalid:invalid@nonexistent:5432/none",
        )

        # ModelStore should raise on bad connection in production mode
        with pytest.raises(Exception):
            ModelStore(persistence_config=bad_config)

    def test_dummy_store_in_test_mode(self):
        """
        Test that DummyStore works in test mode.
        """
        from ml.stores.base import DummyStore

        store = DummyStore()

        # All operations should work without errors
        store.write_features(
            feature_set_id="test",
            instrument_id="EUR/USD",
            features={"test": 1.0},
            ts_event=1000000000,
            ts_init=1000000001,
        )

        store.write_prediction(
            model_id="test",
            instrument_id="EUR/USD",
            prediction=0.5,
            confidence=0.8,
            features={},
            inference_time_ms=1.0,
            ts_event=1000000000,
        )

        store.write_signal(
            strategy_id="test",
            instrument_id="EUR/USD",
            signal_type="buy",
            strength=0.8,
            model_predictions={},
            risk_metrics={},
            execution_params={},
            ts_event=1000000000,
        )

        store.flush()

        # Health check should always return True
        assert store.is_healthy()

        # Stats should return dummy indicator
        stats = store.get_stats()
        assert stats["dummy"] is True
