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

    print(f"Running tests for {len(test_files)} file(s)...")

    cmd = ["pytest", *test_files, "-xvs", "--tb=short"]

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
    # Get changed files from command line arguments
    changed_files = sys.argv[1:]

    if not changed_files:
        print("No files changed.")
        return 0

    # Filter for Python files
    python_files = [f for f in changed_files if f.endswith(".py")]

    if not python_files:
        print("No Python files changed.")
        return 0

    # Check if any ml/ files are changed
    ml_changed = any("ml/" in f for f in python_files)

    test_files_to_run = []

    # If ml/ files changed, run all ml/ tests
    if ml_changed:
        print("ML files changed, will run all ML tests...")
        ml_test_paths = [
            "tests/unit_tests/ml",
            "tests/integration_tests/ml",
            "ml/tests",
        ]

        for test_dir in ml_test_paths:
            test_path = Path(test_dir)
            if test_path.exists():
                test_files_to_run.append(str(test_path))

    # Add tests for other changed files
    other_test_files = get_test_files_for_changed_files(
        [f for f in python_files if "ml/" not in f],
    )
    test_files_to_run.extend(other_test_files)

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
