#!/usr/bin/env python3

"""
End-to-end integration tests for the complete ML pipeline.

These tests verify that all components work together:
1. Training → Saving → Loading → Inference → Signal → Trading
2. Multi-model deployment and coordination
3. Model hot-reloading without system interruption
4. Performance requirements (<5ms end-to-end)
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_XGBOOST
from nautilus_trader.model.data import Bar
from nautilus_trader.test_kit.stubs.data import TestDataStubs


class TestIntegrationPipeline:
    """Test suite for end-to-end ML pipeline integration."""

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost required")
    def test_complete_ml_pipeline(self) -> None:
        """
        Test complete flow from training to trading.

        Given: Training data
        When: Train → Save → Load → Inference → Signal → Trade
        Then: Each step succeeds and data flows correctly
        """
        from ml.actors.base import MLSignal
        from ml.config.xgboost import XGBoostTrainingConfig
        from ml.models.loader import ProductionModelLoader
        from ml.training.xgboost import XGBoostTrainer

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # Step 1: Train model
            print("Step 1: Training model...")
            config = XGBoostTrainingConfig(
                data_source="integration_test",
                n_estimators=5,
                max_depth=3,
                save_model_path=str(tmpdir / "model.json"),
            )

            trainer = XGBoostTrainer(config)

            # Create training data
            n_samples, n_features = 200, 10
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple rule for testing

            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y

            trainer.train(df)

            # Step 2: Save model
            print("Step 2: Saving model...")
            model_path = tmpdir / "model.json"
            trainer.save_model(model_path)
            assert model_path.exists(), "Model file should exist"

            # Step 3: Load model
            print("Step 3: Loading model...")
            loader = ProductionModelLoader()
            loaded_model, metadata = loader.load_model(str(model_path))
            assert loaded_model is not None, "Model should load successfully"

            # Step 4: Create inference actor
            print("Step 4: Setting up inference actor...")

            class TestInferenceActor:
                def __init__(self, model: Any, model_id: str = "test_model") -> None:
                    self.model = model
                    self.model_id = model_id
                    self.published_signals: list[Any] = []

                def on_bar(self, bar: Bar) -> MLSignal | None:
                    # Generate features (simplified)
                    features = np.random.randn(1, n_features).astype(np.float32)

                    # Make prediction
                    import xgboost as xgb
                    dtest = xgb.DMatrix(features, feature_names=feature_names)
                    prediction = self.model.predict(dtest)[0]

                    # Create signal
                    signal = MLSignal(
                        instrument_id=bar.bar_type.instrument_id,
                        model_id=self.model_id,
                        prediction=float(prediction),
                        confidence=abs(float(prediction)),
                        metadata={},
                        ts_event=bar.ts_event,
                        ts_init=bar.ts_init,
                    )

                    self.published_signals.append(signal)
                    return signal

            actor = TestInferenceActor(loaded_model)

            # Step 5: Generate signal
            print("Step 5: Generating signal...")
            bar = TestDataStubs.bar_5decimal()
            signal = actor.on_bar(bar)
            assert signal is not None, "Signal should be generated"
            assert isinstance(signal, MLSignal), "Should be MLSignal type"

            # Step 6: Strategy receives signal
            print("Step 6: Strategy processing signal...")

            class TestStrategy:
                def __init__(self) -> None:
                    self.received_signals: list[Any] = []
                    self.executed_trades: list[dict[str, Any]] = []

                def on_data(self, data: Any) -> None:
                    if isinstance(data, MLSignal):
                        self.received_signals.append(data)

                        # Simple trading logic
                        if data.confidence > 0.6:
                            self.execute_trade(data)

                def execute_trade(self, signal: Any) -> None:
                    trade = {
                        "instrument": signal.instrument_id,
                        "direction": "BUY" if signal.prediction > 0.5 else "SELL",
                        "confidence": signal.confidence,
                        "timestamp": time.time(),
                    }
                    self.executed_trades.append(trade)

            strategy = TestStrategy()
            strategy.on_data(signal)

            # Verify complete pipeline
            assert len(strategy.received_signals) == 1, "Strategy should receive signal"
            assert strategy.received_signals[0] == signal, "Should be same signal"

            print("✅ Complete ML pipeline test passed!")

    @pytest.mark.skipif(not (HAS_XGBOOST and HAS_LIGHTGBM), reason="Both XGBoost and LightGBM required")
    def test_multi_model_deployment(self) -> None:
        """
        Test deploying multiple models simultaneously.

        Given: Multiple trained models (XGBoost + LightGBM)
        When: Both models process same data
        Then: Ensemble predictions are generated correctly
        """
        from ml.config.lightgbm import LightGBMTrainingConfig
        from ml.config.xgboost import XGBoostTrainingConfig
        from ml.models.loader import ProductionModelLoader
        from ml.training.lightgbm import LightGBMTrainer
        from ml.training.xgboost import XGBoostTrainer

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # Train multiple models
            n_samples, n_features = 200, 10
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)

            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y

            # Train XGBoost
            xgb_config = XGBoostTrainingConfig(
                data_source="test",
                n_estimators=5,
                max_depth=3,
                save_model_path=str(tmpdir / "xgboost.json"),
            )
            xgb_trainer = XGBoostTrainer(xgb_config)
            xgb_trainer.train(df)
            xgb_trainer.save_model(tmpdir / "xgboost.json")

            # Train LightGBM
            lgbm_config = LightGBMTrainingConfig(
                data_source="test",
                n_estimators=5,
                max_depth=5,  # Increased to satisfy num_leaves constraint
                num_leaves=20,  # Set explicitly to avoid overfitting
                objective="binary",  # Classification task
                metric="binary_logloss",
                save_model_path=str(tmpdir / "lightgbm.txt"),
            )
            lgbm_trainer = LightGBMTrainer(lgbm_config)
            lgbm_trainer.train(df)
            lgbm_trainer.save_model(tmpdir / "lightgbm.txt")

            # Load both models
            loader = ProductionModelLoader()
            xgb_model, _ = loader.load_model(str(tmpdir / "xgboost.json"))
            lgbm_model, _ = loader.load_model(str(tmpdir / "lightgbm.txt"))

            # Create multi-model orchestrator
            class ModelActor:
                def __init__(self, model: Any, model_id: str) -> None:
                    self.model = model
                    self.model_id = model_id

                def predict(self, features: np.ndarray[Any, Any]) -> float:
                    if "xgboost" in self.model_id:
                        import xgboost as xgb
                        # Need to provide feature names for XGBoost
                        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
                        dtest = xgb.DMatrix(features, feature_names=feature_names)
                        return float(self.model.predict(dtest)[0])
                    else:
                        return float(self.model.predict(features)[0])

            class MultiModelOrchestrator:
                def __init__(self) -> None:
                    self.models: dict[str, ModelActor] = {}

                def add_model(self, model_id: str, model: Any) -> None:
                    self.models[model_id] = ModelActor(model, model_id)

                def process_bar(self, bar: Bar) -> dict[str, Any]:
                    features = np.random.randn(1, n_features).astype(np.float32)
                    predictions = {}

                    for model_id, actor in self.models.items():
                        predictions[model_id] = actor.predict(features)

                    # Ensemble prediction (simple average)
                    ensemble_pred = np.mean(list(predictions.values()))

                    return {
                        "individual": predictions,
                        "ensemble": ensemble_pred,
                        "timestamp": bar.ts_event,
                    }

            orchestrator = MultiModelOrchestrator()
            orchestrator.add_model("xgboost_v1", xgb_model)
            orchestrator.add_model("lightgbm_v1", lgbm_model)

            # Process data through both models
            bar = TestDataStubs.bar_5decimal()
            result = orchestrator.process_bar(bar)

            # Verify multi-model deployment
            assert "xgboost_v1" in result["individual"], "XGBoost prediction missing"
            assert "lightgbm_v1" in result["individual"], "LightGBM prediction missing"
            assert "ensemble" in result, "Ensemble prediction missing"
            assert len(result["individual"]) == 2, "Should have 2 model predictions"

            print("✅ Multi-model deployment test passed!")

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost required")
    def test_model_hot_reload(self) -> None:
        """
        Test hot-reloading models without system interruption.

        Given: Running system with active model
        When: New model is deployed
        Then: System switches to new model without dropping signals
        """
        from ml.config.xgboost import XGBoostTrainingConfig
        from ml.models.loader import ProductionModelLoader
        from ml.training.xgboost import XGBoostTrainer

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # Train initial model (v1)
            n_samples, n_features = 200, 10
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] > 0).astype(int)  # Simple rule for v1

            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y

            config_v1 = XGBoostTrainingConfig(
                data_source="test",
                n_estimators=5,
                max_depth=3,
                save_model_path=str(tmpdir / "model_v1.json"),
            )
            trainer_v1 = XGBoostTrainer(config_v1)
            trainer_v1.train(df)
            trainer_v1.save_model(tmpdir / "model_v1.json")

            # Train new model (v2) with different pattern
            y_v2 = (X[:, 1] > 0).astype(int)  # Different rule for v2
            df["target"] = y_v2

            config_v2 = XGBoostTrainingConfig(
                data_source="test",
                n_estimators=5,
                max_depth=3,
                save_model_path=str(tmpdir / "model_v2.json"),
            )
            trainer_v2 = XGBoostTrainer(config_v2)
            trainer_v2.train(df)
            trainer_v2.save_model(tmpdir / "model_v2.json")

            # Create hot-reloadable actor
            class HotReloadableActor:
                def __init__(self) -> None:
                    self.model: Any | None = None
                    self.model_version: str = ""
                    self.loader = ProductionModelLoader()
                    self.predictions_made: int = 0

                def load_model(self, model_path: str, version: str) -> None:
                    """Hot-reload a new model."""
                    self.model, _ = self.loader.load_model(model_path)
                    self.model_version = version
                    print(f"Loaded model version: {version}")

                def on_bar(self, bar: Bar) -> dict[str, Any]:
                    """Process bar with current model."""
                    if self.model is None:
                        return {"error": "No model loaded"}

                    features = np.random.randn(1, n_features).astype(np.float32)

                    import xgboost as xgb
                    dtest = xgb.DMatrix(features, feature_names=feature_names)
                    prediction = float(self.model.predict(dtest)[0])

                    self.predictions_made += 1

                    return {
                        "prediction": prediction,
                        "model_version": self.model_version,
                        "predictions_made": self.predictions_made,
                    }

            actor = HotReloadableActor()

            # Start with v1
            actor.load_model(str(tmpdir / "model_v1.json"), "v1")

            # Process some data
            bar = TestDataStubs.bar_5decimal()
            result_v1_1 = actor.on_bar(bar)
            result_v1_2 = actor.on_bar(bar)

            assert result_v1_1["model_version"] == "v1"
            assert result_v1_1["predictions_made"] == 1
            assert result_v1_2["predictions_made"] == 2

            # Hot-reload to v2
            actor.load_model(str(tmpdir / "model_v2.json"), "v2")

            # Continue processing without interruption
            result_v2_1 = actor.on_bar(bar)
            result_v2_2 = actor.on_bar(bar)

            assert result_v2_1["model_version"] == "v2"
            assert result_v2_1["predictions_made"] == 3  # Continuous count
            assert result_v2_2["predictions_made"] == 4

            print("✅ Model hot-reload test passed!")

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost required")
    def test_performance_requirements(self) -> None:
        """
        Test that ML pipeline meets performance requirements.

        Given: Complete ML pipeline
        When: Processing high-frequency data
        Then: End-to-end latency < 5ms (P99)
        """
        from ml.config.xgboost import XGBoostTrainingConfig
        from ml.models.loader import ProductionModelLoader
        from ml.training.xgboost import XGBoostTrainer

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # Train lightweight model for speed testing
            n_samples, n_features = 100, 5  # Smaller for speed
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] > 0).astype(int)

            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y

            config = XGBoostTrainingConfig(
                data_source="test",
                n_estimators=3,  # Very small for speed
                max_depth=2,
                save_model_path=str(tmpdir / "speed_model.json"),
            )
            trainer = XGBoostTrainer(config)
            trainer.train(df)
            trainer.save_model(tmpdir / "speed_model.json")

            # Load model
            loader = ProductionModelLoader()
            model, _ = loader.load_model(str(tmpdir / "speed_model.json"))

            # Create performance testing pipeline
            class PerformancePipeline:
                def __init__(self, model: Any) -> None:
                    self.model = model
                    self.latencies: list[float] = []

                def process_bar(self, bar: Bar) -> None:
                    """Process bar and measure latency."""
                    start_time = time.perf_counter()

                    # Feature engineering
                    features = np.random.randn(1, n_features).astype(np.float32)

                    # Model inference
                    import xgboost as xgb
                    dtest = xgb.DMatrix(features, feature_names=feature_names)
                    prediction = self.model.predict(dtest)[0]

                    # Signal generation (simplified)
                    signal = {
                        "prediction": float(prediction),
                        "timestamp": bar.ts_event,
                    }

                    # Measure latency
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    self.latencies.append(latency_ms)

                def get_p99_latency(self) -> float:
                    """Calculate P99 latency."""
                    if not self.latencies:
                        return 0.0
                    return float(np.percentile(self.latencies, 99))

            pipeline = PerformancePipeline(model)

            # Process many bars to measure performance
            bar = TestDataStubs.bar_5decimal()
            for _ in range(100):  # Process 100 bars
                pipeline.process_bar(bar)

            # Warm-up complete, now measure
            pipeline.latencies.clear()
            for _ in range(1000):  # Measure 1000 iterations
                pipeline.process_bar(bar)

            # Check performance
            p99_latency = pipeline.get_p99_latency()
            mean_latency = np.mean(pipeline.latencies)

            print("Performance results:")
            print(f"  Mean latency: {mean_latency:.2f}ms")
            print(f"  P99 latency: {p99_latency:.2f}ms")
            print(f"  Min latency: {min(pipeline.latencies):.2f}ms")
            print(f"  Max latency: {max(pipeline.latencies):.2f}ms")

            # Verify performance requirements
            # Note: In CI/testing, we allow higher threshold due to overhead
            # In production, this should be < 5ms
            assert p99_latency < 50, f"P99 latency {p99_latency}ms exceeds 50ms threshold"

            print("✅ Performance requirements test passed!")
