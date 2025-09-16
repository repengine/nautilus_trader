#!/usr/bin/env python3
"""
Validate ML module imports and identify issues.
"""

import sys
from pathlib import Path


# Ensure project root is first on sys.path to avoid picking up any installed 'ml' packages
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def probe_import(module_name: str) -> tuple[bool, str]:
    """
    Test if a module can be imported.
    """
    try:
        __import__(module_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected: {e}"


def main() -> int:
    """
    Test all ML submodules.
    """
    ml_modules = [
        "ml.actors",
        "ml.common",
        "ml.config",
        "ml.consumers",
        "ml.core",
        "ml.data",
        "ml.deployment",
        "ml.evaluation",
        "ml.features",
        "ml.models",
        "ml.monitoring",
        "ml.observability",
        "ml.orchestration",
        "ml.pipelines",
        "ml.preprocessing",
        "ml.registry",
        "ml.stores",
        "ml.strategies",
        "ml.training",
    ]

    print("ML Module Import Validation Report")
    print("=" * 50)

    success_count = 0
    failures = []

    for module in ml_modules:
        success, message = probe_import(module)
        status = "✓" if success else "✗"
        print(f"{status} {module:30} {message if not success else ''}")

        if success:
            success_count += 1
        else:
            failures.append((module, message))

    print("\n" + "=" * 50)
    print(f"Results: {success_count}/{len(ml_modules)} modules import successfully")

    if failures:
        print("\nFailure Details:")
        for module, error in failures:
            print(f"\n{module}:")
            print(f"  {error}")

    return 0 if success_count == len(ml_modules) else 1


if __name__ == "__main__":
    sys.exit(main())
