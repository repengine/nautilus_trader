"""
Integration tests for store persistence.

Tests that data is actually persisted to stores and can be retrieved.

"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from ml.stores.model_store import ModelStore


@pytest.mark.database
@pytest.mark.serial
class TestStorePersistence:
    """Test that stores actually persist data with PostgreSQL."""

    @pytest.mark.database
    @pytest.mark.serial
    def test_feature_store_persistence(self, feature_store, store_bundle, default_instrument_id) -> None:
        """Test that FeatureStore actually persists and retrieves features."""

        features = {
            "feature_0": 0.5,
            "feature_1": 0.7,
            "feature_2": -0.3,
        }

        feature_store.write_features(
            feature_set_id="test_set",
            instrument_id="EUR/USD",
            features=features,
            ts_event=1_000_000_000,
            ts_init=1_000_000_001,
        )
        feature_store.flush()

        with store_bundle.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                        SELECT instrument_id, ts_event
                        FROM public.ml_feature_values
                        WHERE feature_set_id = :feature_set_id
                        LIMIT 1
                        """,
                ),
                {"feature_set_id": "test_set"},
            ).fetchone()

        assert row is not None
        assert row[0] == "EUR/USD"
        assert int(row[1]) == 1_000_000_000_000_000_000
        assert feature_store.is_healthy()

    @pytest.mark.database
    @pytest.mark.serial
    def test_model_store_persistence(self, model_store, store_bundle) -> None:
        """Test that ModelStore actually persists and retrieves predictions."""

        model_store.write_prediction(
            model_id="test_model",
            instrument_id="EUR/USD",
            prediction=0.65,
            confidence=0.8,
            features={"feature_0": 0.5},
            inference_time_ms=2.5,
            ts_event=1_000_000_000,
        )

        for i in range(10):
            model_store.write_prediction(
                model_id="test_model",
                instrument_id="EUR/USD",
                prediction=0.5 + i * 0.01,
                confidence=0.7 + i * 0.01,
                features={"feature_0": 0.5 + i * 0.1},
                inference_time_ms=2.0 + i * 0.1,
                ts_event=1_000_000_000 + i * 1_000,
            )
        model_store.flush()

        with store_bundle.engine.connect() as conn:
            count = conn.execute(
                text(
                    """
                        SELECT COUNT(*)
                        FROM public.ml_model_predictions
                        WHERE model_id = :model_id
                    """,
                ),
                {"model_id": "test_model"},
            ).scalar_one()

        assert count >= 10
        assert model_store.is_healthy()

    @pytest.mark.database
    @pytest.mark.serial
    def test_strategy_store_persistence(self, strategy_store, store_bundle) -> None:
        """Test that StrategyStore actually persists and retrieves signals."""

        strategy_store.write_signal(
            strategy_id="test_strategy",
            instrument_id="EUR/USD",
            signal_type="buy",
            strength=0.8,
            model_predictions={"model_1": 0.7},
            risk_metrics={"sharpe": 1.5},
            execution_params={"threshold": 0.6},
            ts_event=1_000_000_000,
        )

        for i in range(5):
            strategy_store.write_signal(
                strategy_id="test_strategy",
                instrument_id="EUR/USD",
                signal_type="sell" if i % 2 else "buy",
                strength=0.5 + i * 0.1,
                model_predictions={"model_1": 0.4 + i * 0.1},
                risk_metrics={"sharpe": 1.0 + i * 0.2},
                execution_params={"threshold": 0.5},
                ts_event=1_000_000_000 + i * 1_000,
            )
        strategy_store.flush()

        with store_bundle.engine.connect() as conn:
            count = conn.execute(
                text(
                    """
                        SELECT COUNT(*)
                        FROM public.ml_strategy_signals
                        WHERE strategy_id = :sid
                    """,
                ),
                {"sid": "test_strategy"},
            ).scalar_one()

        assert count >= 5
        assert strategy_store.is_healthy()

    @pytest.mark.database
    @pytest.mark.serial
    def test_store_failure_handling(self) -> None:
        """Test that stores handle connection failures gracefully."""

        bad_connection = "postgresql://invalid:invalid@nonexistent:5432/none"

        with pytest.raises(Exception):
            ModelStore(connection_string=bad_connection)

    @pytest.mark.database
    @pytest.mark.serial
    def test_store_is_healthy(self, store_bundle) -> None:
        """Test that stores properly report health status with PostgreSQL."""

        assert store_bundle.feature_store.is_healthy()
        assert store_bundle.model_store.is_healthy()
        assert store_bundle.strategy_store.is_healthy()
