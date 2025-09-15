#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Pre-commit hook to check test coverage for new ML modules.

This hook enforces a minimum coverage on new ML modules. The minimum is read from
tools/coverage_target.txt when available (the project-wide ratcheting baseline). If not
present, it falls back to 90% for strong coverage on new code.

"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tools" / "coverage_target.txt"


def read_dynamic_threshold(default: int = 90) -> int:
    try:
        value = int(BASELINE_PATH.read_text().strip())
        # Use the project baseline for new modules as well. If teams prefer a
        # stricter bar for new code, you can add an offset here (e.g., +5).
        return value
    except Exception:
        return default


def is_new_file(file_path):
    """
    Check if file is newly added (not in git history).
    """
    cmd = ["git", "ls-files", "--error-unmatch", file_path]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode != 0


def get_test_file_for_module(module_path):
    """
    Get the test file path for a module.
    """
    path = Path(module_path)

    if path.name.startswith("test_"):
        return None  # Skip test files

    # Look for test file
    test_patterns = [
        path.parent / "tests" / f"test_{path.name}",
        Path("tests") / "unit_tests" / "ml" / f"test_{path.name}",
        Path("tests") / "integration_tests" / "ml" / f"test_{path.name}",
    ]

    for test_path in test_patterns:
        if test_path.exists():
            return test_path

    return None


def check_coverage(module_path, test_path):
    """
    Check test coverage for the module.
    """
    module_name = Path(module_path).stem

    threshold = read_dynamic_threshold()

    # Use uv-managed environment to ensure pytest + plugins are present
    cmd = [
        "uv",
        "run",
        "--active",
        "--no-sync",
        "pytest",
        str(test_path),
        f"--cov={module_name}",
        "--cov-report=term-missing",
        "--cov-config=/dev/null",
        f"--cov-fail-under={threshold}",
        "--no-header",
        "-q",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return True, "Coverage ≥90%"
    else:
        # Extract coverage percentage
        for line in result.stdout.split("\n"):
            if module_name in line and "%" in line:
                return False, line.strip()
        return False, "Coverage check failed"


def main():
    """
    Execute the main test coverage checking process.
    """
    changed_files = sys.argv[1:]

    # Only check ML files
    ml_files = [f for f in changed_files if "ml/" in f and f.endswith(".py")]

    if not ml_files:
        return 0

    # Only check new files
    new_ml_files = [f for f in ml_files if is_new_file(f)]

    if not new_ml_files:
        return 0

    print(f"Checking test coverage for {len(new_ml_files)} new ML file(s)...")

    failed = False
    for file_path in new_ml_files:
        if Path(file_path).name.startswith("test_"):
            continue  # Skip test files

        test_path = get_test_file_for_module(file_path)

        if not test_path:
            print(f"✗ {file_path}: No test file found!")
            failed = True
            continue

        passed, message = check_coverage(file_path, test_path)

        if passed:
            print(f"✓ {file_path}: {message}")
        else:
            print(f"✗ {file_path}: {message}")
            failed = True

    if failed:
        print("\n❌ Some files have insufficient test coverage (<90%)")
        print("Please add more tests before committing.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
