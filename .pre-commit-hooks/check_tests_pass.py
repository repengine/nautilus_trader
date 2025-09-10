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
Pre-commit hook to run tests on changed files and ml/ folder.
"""

import subprocess
import sys
from pathlib import Path


# Note: Pre-commit manages its own environments, so we don't need to check for venv
# The hook will run in pre-commit's isolated environment with all dependencies installed


def get_test_files_for_changed_files(changed_files):
    """
    Get corresponding test files for changed source files.
    """
    test_files = []

    for file_path in changed_files:
        path = Path(file_path)

        # Skip if not a Python file
        if path.suffix != ".py":
            continue

        # Skip if it's already a test file
        if path.name.startswith("test_"):
            test_files.append(str(path))
            continue

        # Find corresponding test file
        if "ml" in path.parts:
            # For ml/ folder, tests might be in tests/unit_tests/ml/ or tests/integration_tests/ml/
            test_patterns = [
                Path("tests") / "unit_tests" / "ml" / f"test_{path.name}",
                Path("tests") / "integration_tests" / "ml" / f"test_{path.name}",
                path.parent / "tests" / f"test_{path.name}",
            ]
        else:
            # For other files, follow standard pattern
            try:
                if "nautilus_trader" in path.parts:
                    rel_path = path.relative_to("nautilus_trader")
                    test_patterns = [
                        Path("tests") / "unit_tests" / rel_path.parent / f"test_{path.name}",
                        Path("tests") / "integration_tests" / rel_path.parent / f"test_{path.name}",
                    ]
                else:
                    # For files outside nautilus_trader (like pre-commit hooks)
                    test_patterns = []
            except ValueError:
                # If path operations fail, skip this file
                test_patterns = []

        for test_path in test_patterns:
            if test_path.exists():
                test_files.append(str(test_path))
                break

    return list(set(test_files))  # Remove duplicates


def run_tests(test_files):
    """
    Run pytest on the specified test files.
    """
    if not test_files:
        print("No test files to run.")
        return True

    print(f"Running tests for {len(test_files)} file(s) (fast subset)...")

    MARKERS = (
        "unit and not slow and not requires_data and not requires_gpu and not requires_network "
        "and not database and not redis and not docker and not integration"
    )

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *test_files,
        "-q",
        "--tb=short",
        # Fast subset marker expression
        "-m",
        MARKERS,
        "-k",
        "not database and not hypothesis and not postgres",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ All tests passed!")
            return True
        else:
            print("✗ Tests failed!")
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            return False

    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def main():
    """
    Execute the main test running process.
    """
    print("check_tests_pass.py started", flush=True)

    # Since pass_filenames is false, we just run ML tests
    # In a real scenario, you'd use git to detect changed files
    # For now, just run all ML tests
    print("Running all ML tests...")
    test_files_to_run = ["ml/tests/unit"]

    # Run the tests
    if test_files_to_run:
        if not run_tests(test_files_to_run):
            return 1
    else:
        print("Warning: No test files found for changed source files!")
        # You might want to make this fail to enforce test writing
        # return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
