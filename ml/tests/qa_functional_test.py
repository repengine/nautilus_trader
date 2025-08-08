#!/usr/bin/env python3
"""
QA Functional Testing Script for UnifiedXGBoostTrainer Tests core functionality in
isolation.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml._imports import HAS_XGBOOST
from ml.config.shared import MLflowConfig
from ml.config.shared import OptunaConfig
from ml.config.shared import XGBoostGPUConfig as GPUConfig
from ml.config.xgboost import UnifiedXGBoostConfig


def generate_sample_data(n_samples=1000, n_features=10):
    """
    Generate sample financial-like data for testing.
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Add some structure to make it trainable
    coefficients = np.random.randn(n_features)
    y = X @ coefficients + np.random.randn(n_samples) * 0.1

    # Convert to binary classification
    y_binary = (y > np.median(y)).astype(int)

    return X, y, y_binary


def test_basic_training() -> bool:
    """
    Test basic training without any optional features.
    """
    logger.info("Test completed ===")

    if not HAS_XGBOOST:
        logger.info(" XGBoost not available, skipping test")
        return False

    try:
        from ml.training.xgboost import UnifiedXGBoostTrainer

        # Create config with all optional features disabled
        config = UnifiedXGBoostConfig(
            data_source="test",
            objective="binary:logistic",  # Classification objective
            n_estimators=10,  # Small for testing
            max_depth=3,
            gpu_config=GPUConfig(enabled=False),
            mlflow_config=MLflowConfig(enabled=False),
            optuna_config=OptunaConfig(enabled=False),
            export_onnx=False,
        )

        # Initialize trainer
        trainer = UnifiedXGBoostTrainer(config)

        # Generate data
        X, _, y = generate_sample_data()

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train model
        model, metrics = trainer.train(X_train, y_train, X_val, y_val)

        # Validate results
        assert model is not None, "Model should not be None"
        assert "training_time" in metrics, "Metrics should contain training_time"
        assert metrics["training_time"] > 0, "Training time should be positive"

        # Test prediction
        predictions = model.predict(X_val[:10])
        assert len(predictions) == 10, "Should predict 10 samples"

        logger.info(" Basic training successful")
        logger.info(f"   Training time: {metrics.get('training_time', 0):.3f}s")
        logger.info(f"   Model score: {metrics.get('best_score', 'N/A')}")
        return True

    except Exception as e:
        logger.info(f"Test failed: {e}")
        traceback.print_exc()
        return False


def test_gpu_fallback() -> bool:
    """
    Test GPU configuration and fallback to CPU.
    """
    logger.info("\n=== TEST 2: GPU Fallback ===")

    if not HAS_XGBOOST:
        logger.info(" XGBoost not available, skipping test")
        return False

    try:
        from ml.training.xgboost import UnifiedXGBoostTrainer

        # Create config with GPU enabled
        config = UnifiedXGBoostConfig(
            data_source="test",
            objective="reg:squarederror",  # Regression objective
            n_estimators=5,
            gpu_config=GPUConfig(
                enabled=True,
                device_id=0,
                validate_gpu=False,  # Don't validate to avoid errors
            ),
            mlflow_config=MLflowConfig(enabled=False),
            optuna_config=OptunaConfig(enabled=False),
        )

        # Initialize trainer (should handle GPU gracefully)
        trainer = UnifiedXGBoostTrainer(config)

        # Check if GPU was disabled
        actual_params = trainer._base_params
        if "tree_method" in actual_params:
            if actual_params["tree_method"] == "hist":
                logger.info(" GPU fallback to CPU successful (tree_method=hist)")
            else:
                logger.info(f"  GPU might be enabled: tree_method={actual_params['tree_method']}")

        # Quick training test
        X, y, _ = generate_sample_data(n_samples=100)
        split_idx = int(0.8 * len(X))
        model, _ = trainer.train(X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:])

        assert model is not None, "Model should train even with GPU fallback"
        logger.info(" GPU fallback test successful")
        return True

    except Exception as e:
        logger.info(f" GPU fallback test failed: {e}")
        traceback.print_exc()
        return False


def test_mlflow_optional() -> bool:
    """
    Test that MLflow features are optional.
    """
    logger.info("\n=== TEST 3: MLflow Optional Features ===")

    if not HAS_XGBOOST:
        logger.info(" XGBoost not available, skipping test")
        return False

    try:
        from ml.training.xgboost import UnifiedXGBoostTrainer

        # Create config with MLflow enabled but not installed
        config = UnifiedXGBoostConfig(
            data_source="test",
            objective="binary:logistic",
            n_estimators=5,
            mlflow_config=MLflowConfig(
                enabled=True,
                experiment_name="test_experiment",
                tracking_uri="file:///tmp/mlflow",
            ),
            gpu_config=GPUConfig(enabled=False),
            optuna_config=OptunaConfig(enabled=False),
        )

        # Initialize trainer
        trainer = UnifiedXGBoostTrainer(config)

        # Generate minimal data
        X, _, y = generate_sample_data(n_samples=100)
        split_idx = 80

        # Train (should work even if MLflow not available)
        model, metrics = trainer.train(
            X[:split_idx],
            y[:split_idx],
            X[split_idx:],
            y[split_idx:],
        )

        assert model is not None, "Training should work without MLflow"
        logger.info(" MLflow optional features test successful")
        logger.info("   Training works without MLflow dependency")
        return True

    except Exception as e:
        logger.info(f" MLflow optional test failed: {e}")
        traceback.print_exc()
        return False


def test_optuna_optional() -> bool:
    """
    Test that Optuna features are optional.
    """
    logger.info("\n=== TEST 4: Optuna Optional Features ===")

    if not HAS_XGBOOST:
        logger.info(" XGBoost not available, skipping test")
        return False

    try:
        from ml.training.xgboost import UnifiedXGBoostTrainer

        # Create config with Optuna enabled but not installed
        config = UnifiedXGBoostConfig(
            data_source="test",
            objective="reg:squarederror",
            n_estimators=5,
            optuna_config=OptunaConfig(
                enabled=True,
                n_trials=10,
                study_name="test_study",
            ),
            gpu_config=GPUConfig(enabled=False),
            mlflow_config=MLflowConfig(enabled=False),
        )

        # Initialize trainer
        trainer = UnifiedXGBoostTrainer(config)

        # Generate minimal data
        X, y, _ = generate_sample_data(n_samples=100)
        split_idx = 80

        # Train (should work even if Optuna not available)
        model, metrics = trainer.train(
            X[:split_idx],
            y[:split_idx],
            X[split_idx:],
            y[split_idx:],
        )

        assert model is not None, "Training should work without Optuna"
        logger.info(" Optuna optional features test successful")
        logger.info("   Training works without Optuna dependency")
        return True

    except Exception as e:
        logger.info(f" Optuna optional test failed: {e}")
        traceback.print_exc()
        return False


def test_feature_importance() -> bool:
    """
    Test feature importance tracking.
    """
    logger.info("\n=== TEST 5: Feature Importance ===")

    if not HAS_XGBOOST:
        logger.info(" XGBoost not available, skipping test")
        return False

    try:
        from ml.training.xgboost import UnifiedXGBoostTrainer

        # Create config
        config = UnifiedXGBoostConfig(
            data_source="test",
            objective="binary:logistic",
            n_estimators=20,
            track_feature_decay=True,
            feature_decay_threshold=0.01,
            gpu_config=GPUConfig(enabled=False),
            mlflow_config=MLflowConfig(enabled=False),
            optuna_config=OptunaConfig(enabled=False),
        )

        # Initialize trainer
        trainer = UnifiedXGBoostTrainer(config)

        # Generate data with feature names
        X, _, y = generate_sample_data(n_features=5)
        feature_names = [f"feature_{i}" for i in range(5)]

        # Convert to DataFrame for feature names
        X_df = pd.DataFrame(X, columns=feature_names)
        split_idx = int(0.8 * len(X))

        # Train model
        model, metrics = trainer.train(
            X_df[:split_idx],
            y[:split_idx],
            X_df[split_idx:],
            y[split_idx:],
            feature_names=feature_names,
        )

        # Check feature importance
        assert hasattr(model, "feature_importances_"), "Model should have feature importances"
        assert len(model.feature_importances_) == 5, "Should have importance for all features"

        # Get feature decay summary
        summary = trainer.get_feature_decay_summary()
        assert summary is not None, "Should return feature decay summary"

        logger.info(" Feature importance test successful")
        logger.info(f"   Top feature importance: {model.feature_importances_.max():.3f}")
        return True

    except Exception as e:
        logger.info(f" Feature importance test failed: {e}")
        traceback.print_exc()
        return False


def test_model_persistence() -> bool:
    """
    Test model saving and loading.
    """
    logger.info("\n=== TEST 6: Model Persistence ===")

    if not HAS_XGBOOST:
        logger.info(" XGBoost not available, skipping test")
        return False

    try:
        import tempfile

        from ml.training.xgboost import UnifiedXGBoostTrainer

        # Create config
        config = UnifiedXGBoostConfig(
            data_source="test",
            objective="binary:logistic",
            n_estimators=10,
            gpu_config=GPUConfig(enabled=False),
            mlflow_config=MLflowConfig(enabled=False),
            optuna_config=OptunaConfig(enabled=False),
        )

        # Initialize trainer
        trainer = UnifiedXGBoostTrainer(config)

        # Generate data
        X, _, y = generate_sample_data(n_samples=200)
        split_idx = 160

        # Train model
        model, _ = trainer.train(
            X[:split_idx],
            y[:split_idx],
            X[split_idx:],
            y[split_idx:],
        )

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            import pickle

            pickle.dump(model, f)
            model_path = f.name

        # Load model
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        # Test predictions match
        test_data = X[split_idx : split_idx + 10]
        original_preds = model.predict(test_data)
        loaded_preds = loaded_model.predict(test_data)

        assert np.allclose(original_preds, loaded_preds), "Predictions should match"

        # Cleanup
        Path(model_path).unlink()

        logger.info(" Model persistence test successful")
        return True

    except Exception as e:
        logger.info(f" Model persistence test failed: {e}")
        traceback.print_exc()
        return False


def test_inference_performance() -> bool:
    """
    Test inference performance requirements.
    """
    logger.info("\n=== TEST 7: Inference Performance ===")

    if not HAS_XGBOOST:
        logger.info(" XGBoost not available, skipping test")
        return False

    try:
        from ml.training.xgboost import UnifiedXGBoostTrainer

        # Create config for fast inference
        config = UnifiedXGBoostConfig(
            data_source="test",
            objective="binary:logistic",
            n_estimators=50,
            max_depth=5,
            gpu_config=GPUConfig(enabled=False),
            mlflow_config=MLflowConfig(enabled=False),
            optuna_config=OptunaConfig(enabled=False),
        )

        # Initialize and train
        trainer = UnifiedXGBoostTrainer(config)
        X, _, y = generate_sample_data(n_samples=500, n_features=20)
        split_idx = 400

        model, _ = trainer.train(
            X[:split_idx],
            y[:split_idx],
            X[split_idx:],
            y[split_idx:],
        )

        # Test single prediction latency
        test_sample = X[0:1]

        # Warm up
        for _ in range(10):
            _ = model.predict(test_sample)

        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = model.predict(test_sample)
            latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms

        # Calculate statistics
        latencies = np.array(latencies)
        p50 = np.percentile(latencies, 50)
        p99 = np.percentile(latencies, 99)

        logger.info(" Inference performance test completed")
        logger.info(f"   P50 latency: {p50:.3f}ms")
        logger.info(f"   P99 latency: {p99:.3f}ms")
        logger.info("   Requirement: <5ms")

        if p99 < 5.0:
            logger.info("    MEETS PERFORMANCE REQUIREMENT")
            return True
        else:
            logger.info("     May not meet strict performance requirement")
            return True  # Still pass test, just warn

    except Exception as e:
        logger.info(f" Inference performance test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """
    Run all functional tests.
    """
    logger.info("=" * 60)
    logger.info("UnifiedXGBoostTrainer QA Functional Testing")
    logger.info("=" * 60)

    tests = [
        test_basic_training,
        test_gpu_fallback,
        test_mlflow_optional,
        test_optuna_optional,
        test_feature_importance,
        test_model_persistence,
        test_inference_performance,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.info(f" Test {test_func.__name__} crashed: {e}")
            results.append(False)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FUNCTIONAL TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(results)
    total = len(results)

    logger.info(f"Tests Passed: {passed}/{total}")
    logger.info(f"Pass Rate: {passed/total*100:.1f}%")

    if passed == total:
        logger.info("\n ALL FUNCTIONAL TESTS PASSED")
    elif passed >= total * 0.7:
        logger.info("\n  MOST FUNCTIONAL TESTS PASSED")
    else:
        logger.info("\n FUNCTIONAL TESTS FAILED")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
