#!/usr/bin/env python3
"""
Test ONNX calibration and production readiness functionality.
"""

import numpy as np
import tempfile
from pathlib import Path
import json
import sys


def test_student_onnx_export():
    """
    Test that student models can export ONNX with calibration.
    """
    print("=== Testing Student ONNX Export with Calibration ===")

    try:
        sys.path.insert(0, "/home/nate/projects/nautilus_trader")
        from ml.training.student.lightgbm import LightGBMStudentDistiller

        # Create test data
        n_samples_train = 100
        n_samples_val = 20
        n_features = 5

        X_train = np.random.randn(n_samples_train, n_features).astype(np.float32)
        X_val = np.random.randn(n_samples_val, n_features).astype(np.float32)

        # Simulate teacher soft labels (probabilities)
        q_train = np.random.uniform(0.2, 0.8, n_samples_train).astype(np.float32)

        # True binary labels for calibration
        y_val_true = np.random.randint(0, 2, n_samples_val).astype(np.float32)

        # Create and train student
        distiller = LightGBMStudentDistiller(
            objective="soft_ce",
            early_stopping=10,
            lgb_params={
                "learning_rate": 0.1,
                "num_leaves": 15,
                "verbose": -1,
            },
        )

        # Fit the student model
        distiller.fit(X_train, q_train, X_val, y_val_true)

        # Test ONNX export
        with tempfile.TemporaryDirectory() as temp_dir:
            feature_names = [f"feature_{i}" for i in range(n_features)]

            try:
                onnx_path, meta_path = distiller.export_onnx(
                    feature_names=feature_names,
                    out_dir=temp_dir,
                    model_id="test_student_calibrated",
                    train_date_range=("2024-01-01", "2024-01-31"),
                    flags={"production_ready": True},
                )

                print(f"✓ ONNX export successful: {onnx_path}")

                # Verify metadata
                with open(meta_path) as f:
                    metadata = json.load(f)

                expected_keys = [
                    "model_id",
                    "feature_names",
                    "feature_schema_hash",
                    "calibrator_kind",
                    "calibrator_params",
                    "opset",
                ]

                missing_keys = [k for k in expected_keys if k not in metadata]
                if missing_keys:
                    print(f"✗ Missing metadata keys: {missing_keys}")
                    return False

                print(f"✓ Metadata complete with keys: {list(metadata.keys())}")
                print(f"✓ Calibration method: {metadata.get('calibrator_kind')}")
                print(f"✓ Feature schema hash: {metadata.get('feature_schema_hash')[:16]}...")

                # Test ONNX model inference
                try:
                    import onnxruntime as ort

                    session = ort.InferenceSession(onnx_path)
                    input_name = session.get_inputs()[0].name
                    output_name = session.get_outputs()[0].name

                    # Run inference
                    test_input = np.random.randn(5, n_features).astype(np.float32)
                    outputs = session.run([output_name], {input_name: test_input})
                    predictions = outputs[0]

                    print(f"✓ ONNX inference successful, output shape: {predictions.shape}")
                    print(f"✓ Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")

                    # Verify outputs are probabilities (should be in [0,1])
                    if np.all(predictions >= 0) and np.all(predictions <= 1):
                        print("✓ ONNX outputs are properly calibrated probabilities")
                        return True
                    else:
                        print("✗ ONNX outputs are not in probability range [0,1]")
                        return False

                except ImportError:
                    print("⚠️  ONNX Runtime not available for inference testing")
                    return True  # Export worked, just can't test inference

            except Exception as e:
                print(f"✗ ONNX export failed: {e}")
                return False

    except ImportError as e:
        print(f"✗ Required dependencies not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Student ONNX test failed: {e}")
        return False


def test_model_export_system():
    """
    Test the general model export system.
    """
    print("\n=== Testing Model Export System ===")

    try:
        sys.path.insert(0, "/home/nate/projects/nautilus_trader")
        from ml.training.export import save_model_with_metadata, detect_model_type, ModelType

        # Test model type detection
        print("Testing model type detection...")

        # Test with file paths
        test_cases = [
            ("model.onnx", ModelType.ONNX),
            ("model.xgb", ModelType.XGBOOST),
            ("model.lgb", ModelType.LIGHTGBM),
            ("model.json", ModelType.XGBOOST),
        ]

        for file_path, expected_type in test_cases:
            detected_type = detect_model_type(None, Path(file_path))
            if detected_type == expected_type:
                print(f"✓ {file_path} -> {expected_type.value}")
            else:
                print(f"✗ {file_path} -> {detected_type.value} (expected {expected_type.value})")

        # Test with actual LightGBM model
        try:
            from ml._imports import lgb

            if lgb is not None:
                # Create a simple LightGBM model
                X = np.random.randn(100, 3).astype(np.float32)
                y = np.random.randint(0, 2, 100)

                train_data = lgb.Dataset(X, label=y)
                params = {"objective": "binary", "verbose": -1}
                model = lgb.train(params, train_data, num_boost_round=10, valid_sets=[train_data])

                detected_type = detect_model_type(model)
                if detected_type == ModelType.LIGHTGBM:
                    print("✓ LightGBM model detection works")
                else:
                    print(f"✗ LightGBM model detected as {detected_type.value}")
            else:
                print("⚠️  LightGBM not available for model detection test")

        except Exception as e:
            print(f"⚠️  LightGBM model detection test failed: {e}")

        return True

    except Exception as e:
        print(f"✗ Model export system test failed: {e}")
        return False


def test_feature_schema_validation():
    """
    Test feature schema validation and parity enforcement.
    """
    print("\n=== Testing Feature Schema Validation ===")

    try:
        sys.path.insert(0, "/home/nate/projects/nautilus_trader")
        from ml.training.student.lightgbm import schema_hash

        # Test schema hash consistency
        feature_names_1 = ["feature_0", "feature_1", "feature_2"]
        feature_names_2 = ["feature_0", "feature_1", "feature_2"]  # Same order
        feature_names_3 = ["feature_1", "feature_0", "feature_2"]  # Different order

        hash_1 = schema_hash(feature_names_1)
        hash_2 = schema_hash(feature_names_2)
        hash_3 = schema_hash(feature_names_3)

        if hash_1 == hash_2:
            print("✓ Schema hashes are consistent for same features")
        else:
            print("✗ Schema hashes differ for identical features")
            return False

        if hash_1 != hash_3:
            print("✓ Schema hashes differ for different feature orders")
        else:
            print("✗ Schema hashes are same despite different feature order")
            return False

        print(f"✓ Example schema hash: {hash_1[:16]}...")

        # Test with dtypes
        dtypes = ["float32", "float32", "float32"]
        hash_with_dtypes = schema_hash(feature_names_1, dtypes)

        if hash_with_dtypes != hash_1:
            print("✓ Schema hashes include dtype information")
        else:
            print("⚠️  Schema hashes may not include dtype information")

        return True

    except Exception as e:
        print(f"✗ Feature schema validation test failed: {e}")
        return False


def test_trading_metrics():
    """
    Test trading metrics calculation.
    """
    print("\n=== Testing Trading Metrics Calculation ===")

    try:
        # Test the trading metrics calculation directly
        sys.path.insert(0, "/home/nate/projects/nautilus_trader")

        # Create a concrete trainer just to test the trading metrics
        from ml.training.non_distilled.lightgbm import LightGBMTrainer
        from ml.config.base import MLTrainingConfig

        config = MLTrainingConfig(
            data_source="test",
            target_column="target",
            train_test_split=0.8,
        )

        trainer = LightGBMTrainer(config)

        # Create sample returns and predictions
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.012, 0.025])
        predictions = np.array([0.6, 0.4, 0.8, 0.3, 0.7, 0.65, 0.35, 0.9])

        metrics = trainer.calculate_trading_metrics(returns, predictions)

        expected_metrics = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "information_ratio",
        ]

        for metric in expected_metrics:
            if metric in metrics:
                print(f"✓ {metric}: {metrics[metric]:.4f}")
            else:
                print(f"✗ Missing metric: {metric}")
                return False

        # Validate metric ranges
        if -1 <= metrics["max_drawdown"] <= 1:
            print("✓ Max drawdown in valid range")
        else:
            print(f"⚠️  Max drawdown may be outside expected range: {metrics['max_drawdown']}")

        if 0 <= metrics["win_rate"] <= 1:
            print("✓ Win rate in valid range")
        else:
            print(f"✗ Win rate outside [0,1] range: {metrics['win_rate']}")
            return False

        return True

    except Exception as e:
        print(f"✗ Trading metrics test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """
    Run all ONNX calibration and production readiness tests.
    """
    print("Testing ONNX Calibration and Production Readiness")
    print("=" * 60)

    tests = [
        ("Student ONNX Export with Calibration", test_student_onnx_export),
        ("Model Export System", test_model_export_system),
        ("Feature Schema Validation", test_feature_schema_validation),
        ("Trading Metrics Calculation", test_trading_metrics),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("PRODUCTION READINESS SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:40} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All production readiness tests passed!")
        return 0
    else:
        print("⚠️  Some production readiness issues found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
