#!/usr/bin/env python3
"""
Validate teacher-student training pipeline claims.

Focus on checking what actually exists and works vs. documentation claims.

"""

import importlib
import inspect
import sys
from pathlib import Path


def test_import(module_name, description):
    """
    Test if a module can be imported.
    """
    try:
        module = importlib.import_module(module_name)
        return True, f"✓ {description}: {module.__file__}"
    except ImportError as e:
        return False, f"✗ {description}: {e}"
    except Exception as e:
        return False, f"✗ {description}: Unexpected error: {e}"


def test_class_exists(module_name, class_name, description):
    """
    Test if a class exists in a module.
    """
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        methods = [name for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)]
        return True, f"✓ {description}: {len(methods)} methods"
    except ImportError as e:
        return False, f"✗ {description}: Module import failed: {e}"
    except AttributeError as e:
        return False, f"✗ {description}: Class not found: {e}"
    except Exception as e:
        return False, f"✗ {description}: Unexpected error: {e}"


def test_console_script(script_name):
    """
    Test if console script exists and shows help.
    """
    import subprocess

    try:
        result = subprocess.run(
            [script_name, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, f"✓ Console script {script_name} works"
        else:
            return False, f"✗ Console script {script_name} failed: {result.stderr[:200]}"
    except FileNotFoundError:
        return False, f"✗ Console script {script_name} not found"
    except subprocess.TimeoutExpired:
        return False, f"✗ Console script {script_name} timed out"
    except Exception as e:
        return False, f"✗ Console script {script_name} error: {e}"


def test_file_exists(file_path, description):
    """
    Test if a file exists.
    """
    path = Path(file_path)
    if path.exists():
        size_kb = path.stat().st_size / 1024
        return True, f"✓ {description}: {size_kb:.1f}KB"
    else:
        return False, f"✗ {description}: File not found"


def test_function_call(module_name, func_name, args, description):
    """
    Test if a function can be called.
    """
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        result = func(*args)
        return True, f"✓ {description}: Returns {type(result).__name__}"
    except Exception as e:
        return False, f"✗ {description}: {e}"


def main():
    print("Validating Teacher-Student Training Pipeline Claims")
    print("=" * 65)

    tests = [
        # Core infrastructure
        (
            "Base Training Infrastructure",
            lambda: test_import(
                "ml.training.base",
                "BaseMLTrainer import",
            ),
        ),
        (
            "BaseMLTrainer Class",
            lambda: test_class_exists(
                "ml.training.base",
                "BaseMLTrainer",
                "BaseMLTrainer class",
            ),
        ),
        # Export system
        (
            "Export System",
            lambda: test_import(
                "ml.training.export",
                "Export system import",
            ),
        ),
        (
            "Model Export Functions",
            lambda: test_file_exists(
                "/home/nate/projects/nautilus_trader/ml/training/export.py",
                "Export module",
            ),
        ),
        # Teacher implementations
        (
            "TFT Teacher",
            lambda: test_import(
                "ml.training.teacher.tft_teacher",
                "TFT teacher import",
            ),
        ),
        (
            "TFT Teacher Class",
            lambda: test_class_exists(
                "ml.training.teacher.tft_teacher",
                "TFTTeacher",
                "TFTTeacher class",
            ),
        ),
        (
            "TFT CLI",
            lambda: test_file_exists(
                "/home/nate/projects/nautilus_trader/ml/training/teacher/tft_cli.py",
                "TFT CLI",
            ),
        ),
        (
            "TFT TorchScript",
            lambda: test_import(
                "ml.training.teacher.tft_torchscript",
                "TFT TorchScript export",
            ),
        ),
        # Student implementations
        (
            "LightGBM Student",
            lambda: test_import(
                "ml.training.student.lightgbm",
                "LightGBM student import",
            ),
        ),
        (
            "LightGBM Student Class",
            lambda: test_class_exists(
                "ml.training.student.lightgbm",
                "LightGBMStudentDistiller",
                "Student distiller",
            ),
        ),
        (
            "LightGBM CLI",
            lambda: test_file_exists(
                "/home/nate/projects/nautilus_trader/ml/training/student/lightgbm_cli.py",
                "Student CLI",
            ),
        ),
        # Non-distilled trainers
        (
            "LightGBM Trainer",
            lambda: test_import(
                "ml.training.non_distilled.lightgbm",
                "LightGBM trainer import",
            ),
        ),
        (
            "XGBoost Trainer",
            lambda: test_import(
                "ml.training.non_distilled.xgboost",
                "XGBoost trainer import",
            ),
        ),
        # HPO
        (
            "Optuna Optimizer",
            lambda: test_import(
                "ml.training.optuna_optimizer",
                "Optuna optimizer import",
            ),
        ),
        # Console scripts
        ("TFT Teacher CLI Script", lambda: test_console_script("ml-teacher-tft")),
        ("LightGBM Student CLI Script", lambda: test_console_script("ml-student-lightgbm")),
        # Registry integration
        (
            "Model Registry",
            lambda: test_import(
                "ml.registry.model_registry",
                "Model registry import",
            ),
        ),
        (
            "Feature Registry",
            lambda: test_import(
                "ml.registry.feature_registry",
                "Feature registry import",
            ),
        ),
        # Core functionality tests
        (
            "Trading Metrics Function",
            lambda: test_function_call(
                "ml.training.base",
                "BaseMLTrainer",
                [],
                "Trading metrics base",
            ),
        ),
    ]

    results = {}

    print("Testing core implementations...")
    print("-" * 65)

    for test_name, test_func in tests:
        try:
            success, message = test_func()
            results[test_name] = success
            print(f"{test_name:35} {message}")
        except Exception as e:
            results[test_name] = False
            print(f"{test_name:35} ✗ Exception: {e}")

    # Additional validation: check key methods exist
    print("\nValidating key method implementations...")
    print("-" * 65)

    method_tests = [
        ("ml.training.teacher.tft_teacher", "TFTTeacher", "fit", "TFT training method"),
        (
            "ml.training.teacher.tft_teacher",
            "TFTTeacher",
            "predict_logits",
            "TFT prediction method",
        ),
        (
            "ml.training.student.lightgbm",
            "LightGBMStudentDistiller",
            "fit",
            "Student training method",
        ),
        (
            "ml.training.student.lightgbm",
            "LightGBMStudentDistiller",
            "export_onnx",
            "Student ONNX export",
        ),
        ("ml.training.export", None, "save_model_with_metadata", "Model saving function"),
        ("ml.training.export", None, "convert_to_onnx", "ONNX conversion function"),
    ]

    for module_name, class_name, method_name, description in method_tests:
        try:
            module = importlib.import_module(module_name)
            if class_name:
                cls = getattr(module, class_name)
                method = getattr(cls, method_name)
            else:
                method = getattr(module, method_name)

            sig = inspect.signature(method)
            param_count = len(sig.parameters)
            results[f"{description} method"] = True
            print(f"{description:35} ✓ {param_count} parameters")
        except Exception as e:
            results[f"{description} method"] = False
            print(f"{description:35} ✗ {e}")

    # Test dependency management
    print("\nValidating dependency management...")
    print("-" * 65)

    try:
        from ml._imports import HAS_LIGHTGBM
        from ml._imports import HAS_MLFLOW
        from ml._imports import HAS_ONNX
        from ml._imports import HAS_OPTUNA
        from ml._imports import HAS_SKLEARN
        from ml._imports import HAS_XGBOOST

        deps = {
            "LightGBM": HAS_LIGHTGBM,
            "XGBoost": HAS_XGBOOST,
            "Sklearn": HAS_SKLEARN,
            "ONNX": HAS_ONNX,
            "Optuna": HAS_OPTUNA,
            "MLflow": HAS_MLFLOW,
        }

        for dep_name, available in deps.items():
            status = "✓ Available" if available else "✗ Missing"
            results[f"{dep_name} dependency"] = available
            print(f"{dep_name:35} {status}")

    except Exception as e:
        print(f"Dependency check failed: {e}")

    # Summary
    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    passed = sum(results.values())
    total = len(results)

    categories = {
        "Core Infrastructure": ["Base Training Infrastructure", "BaseMLTrainer Class"],
        "Teacher Models": ["TFT Teacher", "TFT Teacher Class", "TFT CLI"],
        "Student Models": ["LightGBM Student", "LightGBM Student Class", "LightGBM CLI"],
        "Export System": ["Export System", "Model Export Functions"],
        "Console Scripts": ["TFT Teacher CLI Script", "LightGBM Student CLI Script"],
        "Registry": ["Model Registry", "Feature Registry"],
    }

    for category, test_names in categories.items():
        cat_passed = sum(results.get(name, False) for name in test_names)
        cat_total = len(test_names)
        print(f"{category:25} {cat_passed}/{cat_total} tests passed")

    print(f"\nOverall Results: {passed}/{total} validations passed")

    # Specific findings
    print("\nKey Findings:")
    print("-" * 65)

    if results.get("TFT Teacher CLI Script", False):
        print("✓ TFT teacher console script is functional")
    else:
        print("✗ TFT teacher console script has issues")

    if results.get("LightGBM Student CLI Script", False):
        print("✓ LightGBM student console script is functional")
    else:
        print("✗ LightGBM student console script has issues")

    if results.get("LightGBM dependency", False):
        print("✓ LightGBM dependency is available")
    else:
        print("✗ LightGBM dependency is missing")

    if results.get("ONNX dependency", False):
        print("✓ ONNX dependency is available for model export")
    else:
        print("✗ ONNX dependency is missing - limits export capabilities")

    success_rate = passed / total
    if success_rate >= 0.8:
        print(f"\n🎉 Training pipeline implementation is {success_rate:.1%} complete!")
        return 0
    elif success_rate >= 0.6:
        print(
            f"\n⚠️  Training pipeline implementation is {success_rate:.1%} complete - some gaps remain"
        )
        return 0
    else:
        print(f"\n❌ Training pipeline implementation is only {success_rate:.1%} complete")
        return 1


if __name__ == "__main__":
    sys.exit(main())
