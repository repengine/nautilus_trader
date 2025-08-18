"""
Simple tests for ML stores without full dependencies.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.data_processor import DataProcessor
from ml.stores.data_processor import QualityFlags


class TestDataProcessor:
    """
    Test DataProcessor functionality.
    """

    def test_process_market_data(self) -> None:
        """
        Test processing market data.
        """
        # Mock the database engine
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            ts_event = int(time.time() * 1e9)

            data = {
                "bid": 150.0,
                "ask": 150.1,
                "bid_size": 100,
                "ask_size": 200,
                "volume": 10000,
            }

            processed, metrics = processor.process_market_data(
                instrument_id="AAPL",
                data=data,
                ts_event=ts_event,
            )

            assert processed["bid"] == 150.0
            assert processed["ask"] == 150.1
            assert processed["quality_score"] >= 0.0
            assert metrics.records_processed == 1

    def test_process_market_data_with_crossed_market(self) -> None:
        """
        Test processing crossed market.
        """
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            ts_event = int(time.time() * 1e9)

            data = {
                "bid": 150.1,  # Bid > Ask (crossed)
                "ask": 150.0,
                "bid_size": 100,
                "ask_size": 200,
            }

            processed, metrics = processor.process_market_data(
                instrument_id="AAPL",
                data=data,
                ts_event=ts_event,
            )

            # Should fix the crossed market
            assert processed["bid"] < processed["ask"]
            assert processed["quality_flags"] & QualityFlags.INVALID_RANGE

    def test_process_features_with_nan(self) -> None:
        """
        Test processing features with NaN values.
        """
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            ts_event = int(time.time() * 1e9)

            features = {
                "sma_20": 150.5,
                "rsi_14": np.nan,  # NaN value
                "volume": 10000,
            }

            feature_data, metrics = processor.process_features(
                feature_set_id="test_features",
                instrument_id="AAPL",
                features=features,
                ts_event=ts_event,
            )

            # NaN should be imputed
            assert feature_data.values["rsi_14"] == 0.0
            assert metrics.missing_imputed == 1

    def test_process_prediction(self) -> None:
        """
        Test processing model predictions.
        """
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            ts_event = int(time.time() * 1e9)

            pred_data, metrics = processor.process_prediction(
                model_id="xgboost_v1",
                instrument_id="AAPL",
                prediction=0.75,
                confidence=0.85,
                features={"sma_20": 150.5},
                inference_time_ms=2.5,
                ts_event=ts_event,
            )

            assert pred_data.prediction == 0.75
            assert pred_data.confidence <= 0.85  # May be adjusted
            assert metrics.records_processed == 1

    def test_process_signal_with_risk_limits(self) -> None:
        """
        Test processing signals with risk limits.
        """
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            ts_event = int(time.time() * 1e9)

            signal_data, metrics = processor.process_signal(
                strategy_id="momentum_v1",
                instrument_id="AAPL",
                signal_type="BUY",
                strength=1.0,
                model_predictions={"xgboost": 0.75},
                ts_event=ts_event,
            )

            assert signal_data.strength <= 1.0  # May be adjusted by risk limits
            assert "order_size" in signal_data.execution_params
            assert metrics.records_processed == 1

    def test_quality_score_calculation(self) -> None:
        """
        Test quality score calculation.
        """
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            # Clean data
            score = processor._calculate_quality_score(QualityFlags.CLEAN)
            assert score == 1.0

            # Data with issues
            flags = QualityFlags.MISSING_DATA | QualityFlags.OUTLIER_DETECTED
            score = processor._calculate_quality_score(flags)
            assert 0 < score < 1.0

            # Severe issues
            flags = QualityFlags.TIMESTAMP_ERROR | QualityFlags.NAN_VALUES | QualityFlags.INF_VALUES
            score = processor._calculate_quality_score(flags)
            assert score < 0.5

    def test_batch_processing(self) -> None:
        """
        Test batch processing.
        """
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            batch = [
                {
                    "instrument_id": "AAPL",
                    "data": {"bid": 150.0, "ask": 150.1},
                    "ts_event": int(time.time() * 1e9),
                },
                {
                    "instrument_id": "GOOGL",
                    "data": {"bid": 2800.0, "ask": 2800.1},
                    "ts_event": int(time.time() * 1e9) + 1000,
                },
            ]

            processed, metrics = processor.process_batch(
                data_type="market",
                batch=batch,
            )

            assert len(processed) == 2
            assert metrics.records_processed == 2

    def test_invalid_prediction_handling(self) -> None:
        """
        Test handling of invalid predictions.
        """
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            ts_event = int(time.time() * 1e9)

            # Test with out-of-range prediction
            pred_data, metrics = processor.process_prediction(
                model_id="xgboost_v1",
                instrument_id="AAPL",
                prediction=100.0,  # Out of range
                confidence=2.0,  # Out of range
                features={"sma_20": 150.5},
                inference_time_ms=2.5,
                ts_event=ts_event,
            )

            # Should fallback to safe values
            assert pred_data.prediction == 0.0
            assert pred_data.confidence == 0.0
            assert metrics.records_failed == 1

    def test_feature_drift_calculation(self) -> None:
        """
        Test feature drift calculation.
        """
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            # Mock feature statistics
            processor._get_feature_statistics = lambda x: {  # type: ignore[method-assign, assignment]
                "sma_20": {"mean": 150.0, "std": 10.0},
                "rsi_14": {"mean": 50.0, "std": 20.0},
            }

            features = {
                "sma_20": 180.0,  # 3 std devs away
                "rsi_14": 90.0,  # 2 std devs away
            }

            drift_score = processor._calculate_feature_drift("test_features", features)

            # Average drift should be (3 + 2) / 2 = 2.5
            assert 2.4 < drift_score < 2.6


class TestDataTypes:
    """
    Test data type classes.
    """

    def test_feature_data_creation(self) -> None:
        """
        Test FeatureData creation.
        """
        ts_event = int(time.time() * 1e9)
        ts_init = ts_event + 1000

        data = FeatureData(
            feature_set_id="test_features",
            instrument_id="AAPL",
            values={"sma_20": 150.5, "rsi_14": 65.2},
            _ts_event=ts_event,
            _ts_init=ts_init,
        )

        assert data.feature_set_id == "test_features"
        assert data.instrument_id == "AAPL"
        assert data.values["sma_20"] == 150.5
        assert data._ts_event == ts_event
        assert data._ts_init == ts_init

    def test_model_prediction_creation(self) -> None:
        """
        Test ModelPrediction creation.
        """
        ts_event = int(time.time() * 1e9)
        ts_init = ts_event + 1000

        data = ModelPrediction(
            model_id="xgboost_v1",
            instrument_id="AAPL",
            prediction=0.75,
            confidence=0.85,
            features_used={"sma_20": 150.5},
            inference_time_ms=2.5,
            _ts_event=ts_event,
            _ts_init=ts_init,
        )

        assert data.model_id == "xgboost_v1"
        assert data.prediction == 0.75
        assert data.confidence == 0.85
        assert data.inference_time_ms == 2.5

    def test_strategy_signal_creation(self) -> None:
        """
        Test StrategySignal creation.
        """
        ts_event = int(time.time() * 1e9)
        ts_init = ts_event + 1000

        data = StrategySignal(
            strategy_id="momentum_v1",
            instrument_id="AAPL",
            signal_type="BUY",
            strength=0.8,
            model_predictions={"xgboost": 0.75},
            risk_metrics={"position_size": 100},
            execution_params={"order_type": "LIMIT"},
            _ts_event=ts_event,
            _ts_init=ts_init,
        )

        assert data.strategy_id == "momentum_v1"
        assert data.signal_type == "BUY"
        assert data.strength == 0.8
        assert data.model_predictions["xgboost"] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
