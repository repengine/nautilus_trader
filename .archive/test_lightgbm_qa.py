#!/usr/bin/env python
"""
Comprehensive QA test suite for UnifiedLightGBMTrainer.

Tests all functionality without requiring LightGBM installation.

"""
import sys

import numpy as np
import pandas as pd


# Add project to path
sys.path.insert(0, "/home/nate/projects/nautilus_trader")

from ml._imports import HAS_LIGHTGBM
from ml.config.lightgbm_unified import DARTConfig
from ml.config.lightgbm_unified import EFBConfig
from ml.config.lightgbm_unified import GOSSConfig
from ml.config.lightgbm_unified import GPUConfig
from ml.config.lightgbm_unified import OptunaConfig
from ml.config.lightgbm_unified import UnifiedLightGBMConfig
from ml.data.loader import MLDataLoader
from ml.monitoring.collector import MLMetricsCollector


def print_section(title: str) -> None:
    """
    Print a formatted section header.
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_config_creation() -> bool:
    """
    Test configuration object creation and validation.
    """
    print_section("Testing Configuration Creation")

    try:
        # Test basic config
        config = UnifiedLightGBMConfig(
            data_source="test_source",
            boosting_type="gbdt",
            num_leaves=31,
            max_depth=10,
            learning_rate=0.05,
            n_estimators=100,
        )
        print("✓ Basic config created successfully")

        # Test GOSS config
        goss_config = UnifiedLightGBMConfig(
            data_source="test_source",
            boosting_type="goss",
            goss_config=GOSSConfig(top_rate=0.2, other_rate=0.1),
        )
        print("✓ GOSS config created successfully")

        # Test DART config
        dart_config = UnifiedLightGBMConfig(
            data_source="test_source",
            boosting_type="dart",
            dart_config=DARTConfig(
                drop_rate=0.15,
                max_drop=50,
                skip_drop=0.5,
                uniform_drop=False,
            ),
        )
        print("✓ DART config created successfully")

        # Test GPU config
        gpu_config = UnifiedLightGBMConfig(
            data_source="test_source",
            gpu_config=GPUConfig(enabled=True, device_id=0),
        )
        print("✓ GPU config created successfully")

        # Test categorical features
        cat_config = UnifiedLightGBMConfig(
            data_source="test_source",
            categorical_features=["feature1", "feature2"],
            categorical_feature_indices=[0, 1],
        )
        print("✓ Categorical features config created successfully")

        return True

    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False


def test_parameter_generation() -> bool:
    """
    Test LightGBM parameter generation.
    """
    print_section("Testing Parameter Generation")

    try:
        # Test GBDT params
        config = UnifiedLightGBMConfig(
            data_source="test",
            boosting_type="gbdt",
        )
        params = config.get_lgb_params()
        assert "objective" in params
        assert params.get("num_leaves") == 31
        print("✓ GBDT parameters generated correctly")

        # Test GOSS params
        config_goss = UnifiedLightGBMConfig(
            data_source="test",
            boosting_type="goss",
            goss_config=GOSSConfig(enabled=True, top_rate=0.3, other_rate=0.2),
        )
        params_goss = config_goss.get_unified_lgb_params()
        assert params_goss.get("boosting_type") == "goss"
        assert params_goss.get("top_rate") == 0.3
        assert params_goss.get("other_rate") == 0.2
        print("✓ GOSS parameters generated correctly")

        # Test DART params
        config_dart = UnifiedLightGBMConfig(
            data_source="test",
            boosting_type="dart",
            dart_config=DARTConfig(enabled=True, drop_rate=0.1),
        )
        params_dart = config_dart.get_unified_lgb_params()
        assert params_dart.get("boosting_type") == "dart"
        assert params_dart.get("drop_rate") == 0.1
        print("✓ DART parameters generated correctly")

        # Test EFB params
        config_efb = UnifiedLightGBMConfig(
            data_source="test",
            efb_config=EFBConfig(enabled=True, max_conflict_rate=0.05),
        )
        params_efb = config_efb.get_unified_lgb_params()
        assert params_efb.get("enable_bundle") is True
        assert params_efb.get("max_conflict_rate") == 0.05
        print("✓ EFB parameters generated correctly")

        return True

    except Exception as e:
        print(f"✗ Parameter generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_validation_rules() -> bool:
    """
    Test configuration validation rules.
    """
    print_section("Testing Validation Rules")

    try:
        # Test mutual exclusivity of GOSS and DART
        config = UnifiedLightGBMConfig(
            data_source="test",
            goss_config=GOSSConfig(enabled=True),
            dart_config=DARTConfig(enabled=True),
        )
        warnings = config.validate_config()
        assert any("GOSS and DART cannot be enabled simultaneously" in w for w in warnings)
        print("✓ GOSS/DART mutual exclusivity validated")

        # Test ONNX export validation
        config = UnifiedLightGBMConfig(
            data_source="test",
            export_onnx=True,
            onnx_output_path="",
        )
        warnings = config.validate_config()
        assert any("onnx_output_path cannot be empty" in w for w in warnings)
        print("✓ ONNX export validation working")

        # Test GPU + Optuna warning
        config = UnifiedLightGBMConfig(
            data_source="test",
            gpu_config=GPUConfig(enabled=True),
            optuna_config=OptunaConfig(enabled=True),
        )
        warnings = config.validate_config()
        assert any("GPU + Optuna" in w for w in warnings)
        print("✓ GPU + Optuna warning generated")

        # Test DART + early stopping warning
        config = UnifiedLightGBMConfig(
            data_source="test",
            dart_config=DARTConfig(enabled=True),
            early_stopping_rounds=10,
        )
        warnings = config.validate_config()
        assert any("Early stopping may not work well with DART" in w for w in warnings)
        print("✓ DART + early stopping warning generated")

        return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mldata_loader_integration() -> bool:
    """
    Test integration with MLDataLoader.
    """
    print_section("Testing MLDataLoader Integration")

    try:
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            },
        )

        # Initialize MLDataLoader
        loader = MLDataLoader(
            data_source="test_data",
            feature_columns=["feature1", "feature2", "feature3"],
            target_column="target",
        )

        # Load data
        X, y = loader.load_training_data(data)
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        print("✓ MLDataLoader integration successful")

        # Test with categorical features
        data_cat = pd.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C"], 100),
                "cat2": np.random.choice(["X", "Y"], 100),
                "num1": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            },
        )

        loader_cat = MLDataLoader(
            data_source="test_categorical",
            feature_columns=["cat1", "cat2", "num1"],
            target_column="target",
            categorical_features=["cat1", "cat2"],
        )

        X_cat, y_cat = loader_cat.load_training_data(data_cat)
        assert X_cat.shape == (100, 3)
        print("✓ Categorical features handled correctly")

        return True

    except Exception as e:
        print(f"✗ MLDataLoader integration failed: {e}")
        return False


def test_monitoring_integration() -> bool:
    """
    Test integration with monitoring collector.
    """
    print_section("Testing Monitoring Integration")

    try:
        # Initialize monitoring collector
        collector = MLMetricsCollector(
            enable_prometheus=False,  # Don't start server for test
            enable_wandb=False,
        )

        # Test metric logging
        collector.log_training_metrics(
            model_type="lightgbm",
            metrics={
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.93,
                "f1_score": 0.925,
            },
            epoch=1,
        )
        print("✓ Training metrics logged successfully")

        # Test feature importance logging
        feature_importance = {
            "feature1": 0.45,
            "feature2": 0.35,
            "feature3": 0.20,
        }
        collector.log_feature_importance("lightgbm", feature_importance)
        print("✓ Feature importance logged successfully")

        # Test inference latency logging
        collector.log_inference_latency("lightgbm", 0.003)  # 3ms
        print("✓ Inference latency logged successfully")

        return True

    except Exception as e:
        print(f"✗ Monitoring integration failed: {e}")
        return False


def test_trainer_initialization() -> bool:
    """
    Test UnifiedLightGBMTrainer initialization without LightGBM.
    """
    print_section("Testing Trainer Initialization")

    if HAS_LIGHTGBM:
        print("⚠ LightGBM is installed, skipping import test")
        return True

    try:
        from ml.training.lightgbm_unified import UnifiedLightGBMTrainer

        # Trainer should import but fail gracefully when used without LightGBM
        config = UnifiedLightGBMConfig(
            data_source="test",
            boosting_type="gbdt",
        )

        # This should not fail at import/init time
        print("✓ UnifiedLightGBMTrainer imported successfully without LightGBM")

        # But should fail with helpful message when trying to train
        try:
            trainer = UnifiedLightGBMTrainer(config)
            trainer.train(np.random.randn(10, 5), np.random.randint(0, 2, 10))
            print("✗ Should have raised dependency error")
            return False
        except ImportError as e:
            if "lightgbm" in str(e).lower():
                print("✓ Correct dependency error raised when training without LightGBM")
                return True
            else:
                print(f"✗ Unexpected error: {e}")
                return False

    except Exception as e:
        print(f"✗ Trainer initialization failed: {e}")
        return False


def test_performance_requirements() -> bool:
    """
    Test performance-related configurations.
    """
    print_section("Testing Performance Requirements")

    try:
        # Test GOSS for large dataset efficiency
        config_goss = UnifiedLightGBMConfig(
            data_source="large_dataset",
            boosting_type="goss",
            goss_config=GOSSConfig(
                enabled=True,
                top_rate=0.2,  # Keep 20% of large gradients
                other_rate=0.1,  # Sample 10% of small gradients
            ),
        )
        params = config_goss.get_unified_lgb_params()
        assert params["boosting_type"] == "goss"
        print("✓ GOSS configured for large dataset efficiency")

        # Test GPU acceleration config
        config_gpu = UnifiedLightGBMConfig(
            data_source="gpu_test",
            gpu_config=GPUConfig(
                enabled=True,
                device_id=0,
                gpu_use_dp=False,  # Single precision for speed
            ),
        )
        params = config_gpu.get_unified_lgb_params()
        assert params["device_type"] == "gpu"
        assert params["gpu_use_dp"] is False
        print("✓ GPU acceleration configured for performance")

        # Test EFB for memory efficiency
        config_efb = UnifiedLightGBMConfig(
            data_source="memory_test",
            efb_config=EFBConfig(
                enabled=True,
                max_conflict_rate=0.0,  # Strict bundling for memory savings
                bundle_size=256,
            ),
        )
        params = config_efb.get_unified_lgb_params()
        assert params["enable_bundle"] is True
        assert params["max_bundle"] == 256
        print("✓ EFB configured for memory efficiency")

        # Test early stopping for training efficiency
        config_early = UnifiedLightGBMConfig(
            data_source="early_stop_test",
            early_stopping_rounds=10,
            n_estimators=1000,  # Will stop early if no improvement
        )
        assert config_early.early_stopping_rounds == 10
        print("✓ Early stopping configured for training efficiency")

        return True

    except Exception as e:
        print(f"✗ Performance configuration failed: {e}")
        return False


def compare_with_xgboost() -> bool:
    """
    Compare configuration with XGBoost trainer.
    """
    print_section("Comparing with XGBoost Trainer")

    try:
        from ml.config.xgboost import XGBoostTrainingConfig

        # Create comparable configs
        lgb_config = UnifiedLightGBMConfig(
            data_source="comparison",
            num_leaves=31,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        xgb_config = XGBoostTrainingConfig(
            data_source="comparison",
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        print("✓ Both configs created with similar parameters")

        # Compare parameter ranges
        lgb_params = lgb_config.get_lgb_params()
        xgb_params = xgb_config.get_xgb_params()

        assert abs(lgb_params["learning_rate"] - xgb_params["learning_rate"]) < 0.001
        assert lgb_params["max_depth"] == xgb_params["max_depth"]
        print("✓ Parameter compatibility verified")

        # LightGBM advantages
        print("\nLightGBM Advantages over XGBoost:")
        print("  • Native categorical feature support (no encoding needed)")
        print("  • GOSS for efficient large dataset training")
        print("  • EFB for memory efficiency with sparse features")
        print("  • Leaf-wise growth (more accurate than XGBoost's level-wise)")
        print("  • Generally faster training speed")

        return True

    except Exception as e:
        print(f"✗ XGBoost comparison failed: {e}")
        return False


def main():
    """
    Run all QA tests.
    """
    print("\n" + "=" * 60)
    print("  UnifiedLightGBMTrainer QA Test Suite")
    print("=" * 60)

    # Track results
    results = {}

    # Run all tests
    results["Config Creation"] = test_config_creation()
    results["Parameter Generation"] = test_parameter_generation()
    results["Validation Rules"] = test_validation_rules()
    results["MLDataLoader Integration"] = test_mldata_loader_integration()
    results["Monitoring Integration"] = test_monitoring_integration()
    results["Trainer Initialization"] = test_trainer_initialization()
    results["Performance Requirements"] = test_performance_requirements()
    results["XGBoost Comparison"] = compare_with_xgboost()

    # Print summary
    print_section("QA Test Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")
    print("\nDetailed Results:")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    # Overall assessment
    print_section("Overall Assessment")

    if passed == total:
        print("✓ All tests passed successfully!")
        print("\nDeployment Readiness: READY")
        print("  • All configurations validate correctly")
        print("  • Parameter generation works as expected")
        print("  • Integrations are functional")
        print("  • Performance optimizations are in place")
    else:
        print(f"⚠ {total - passed} tests failed")
        print("\nDeployment Readiness: NOT READY")
        print("  • Fix failing tests before deployment")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
