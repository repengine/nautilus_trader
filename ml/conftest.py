"""
Pytest configuration for ML module.

This file configures pytest to avoid collecting training modules as test files, which
would cause naming conflicts with installed packages.

"""

from typing import Any


def pytest_ignore_collect(collection_path: Any, config: Any) -> bool:
    """
    Ignore training modules during test collection to avoid naming conflicts.

    These files have the same names as installed packages (lightgbm, xgboost) which
    causes import conflicts when pytest tries to collect them.

    """
    # Get the path as string for comparison
    path_str = str(collection_path)

    # Ignore training modules that conflict with package names
    ignore_patterns = [
        "ml/training/non_distilled/lightgbm.py",
        "ml/training/non_distilled/xgboost.py",
        "ml/training/student/lightgbm.py",
        "ml/training/student/lightgbm_student.py",  # Compatibility shim
        "ml/training/lightgbm.py",  # Compatibility shim
        "ml/training/xgboost.py",  # If it exists
    ]

    for pattern in ignore_patterns:
        if path_str.endswith(pattern):
            return True

    return False
