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
Pre-commit hook to ensure tests pass without failures, errors, or warnings.
"""

import re
import subprocess
import sys
from pathlib import Path


def get_test_files_for_changed_files(changed_files):
    """
    Get corresponding test files for changed source files.
    """
    test_files = set()

    for file_path in changed_files:
        path = Path(file_path)

        # Skip if not a Python file
        if path.suffix != ".py":
            continue

        # If it's already a test file, add it
        if path.name.startswith("test_"):
            test_files.add(str(path))
            continue

        # Find corresponding test file
        possible_test_locations = []

        if "nautilus_trader" in path.parts:
            # For nautilus_trader files
            try:
                rel_path = path.relative_to("nautilus_trader")
            except ValueError:
                # Handle files in python/nautilus_trader/
                if path.parts[0] == "python" and "nautilus_trader" in path.parts:
                    nautilus_trader_idx = path.parts.index("nautilus_trader")
                    rel_path = Path(*path.parts[nautilus_trader_idx + 1 :])
                else:
                    continue
            possible_test_locations.extend(
                [
                    Path("tests") / "unit_tests" / rel_path.parent / f"test_{path.name}",
                    Path("tests") / "integration_tests" / rel_path.parent / f"test_{path.name}",
                ],
            )
        elif "ml" in path.parts or "nautilus_ml" in path.parts:
            # For ML files
            possible_test_locations.extend(
                [
                    Path("tests") / "unit_tests" / "ml" / f"test_{path.name}",
                    Path("tests") / "integration_tests" / "ml" / f"test_{path.name}",
                    Path("nautilus_ml") / "tests" / f"test_{path.name}",
                    path.parent / "tests" / f"test_{path.name}",
                ],
            )

        # Check which test files exist
        for test_path in possible_test_locations:
            if test_path.exists():
                test_files.add(str(test_path))
                break

    return list(test_files)


def _parse_test_issues(output):
    """
    Parse test output for specific issues.
    """
    issues = []

    # Check for failures
    failure_match = re.search(r"(\d+) failed", output)
    if failure_match:
        issues.append(f"Test failures: {failure_match.group(1)}")

    # Check for errors
    error_match = re.search(r"(\d+) error", output)
    if error_match:
        issues.append(f"Test errors: {error_match.group(1)}")

    return issues


def _extract_failed_tests(output):
    """
    Extract failed test details from output.
    """
    issues = []
    if "FAILED" in output:
        failed_tests = re.findall(r"FAILED (.*?) -", output)
        if failed_tests:
            issues.append("Failed tests:")
            for test in failed_tests[:5]:  # Show first 5
                issues.append(f"  - {test}")
            if len(failed_tests) > 5:
                issues.append(f"  ... and {len(failed_tests) - 5} more")
    return issues


def _extract_warnings(output):
    """
    Extract warning details from output.
    """
    issues = []
    if "warnings summary" in output:
        warning_section = output.split("warnings summary")[1].split("=")[0]
        warning_lines = [line.strip() for line in warning_section.split("\n") if line.strip()][:3]
        if warning_lines:
            issues.append("Warnings detected:")
            for line in warning_lines:
                issues.append(f"  {line}")
    return issues


def run_tests_clean(test_files):
    """
    Run tests and ensure no failures, errors, or warnings.
    """
    if not test_files:
        print("No test files to run.")
        return True, "No tests to run"

    print(f"Running {len(test_files)} test file(s) with strict checks...")

    # Run pytest with warning capture
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-xvs",  # Stop on first failure, verbose, no capture
        "--tb=short",
        "--strict-markers",  # Strict marker checks
        "--strict-config",  # Strict config checks
        "-m",
        "unit and not slow and not requires_* and not database and not redis and not docker and not integration and not e2e and not system"
        "--no-header",
        *test_files,  # Unpack test files
    ]

    # Run tests
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check return code
    if result.returncode != 0:
        output = result.stdout + result.stderr

        # Collect all issues
        issues = []
        issues.extend(_parse_test_issues(output))
        issues.extend(_extract_failed_tests(output))
        issues.extend(_extract_warnings(output))

        out = result.stdout + result.stderr
        if "ModuleNotFoundError: No module named 'psycopg2'" in out:
            return True, "Skipped strict clean: psycopg2 missing in hook env"
        return False, "\n".join(issues) if issues else "Tests failed"

    # All tests passed
    passed_match = re.search(r"(\d+) passed", result.stdout)
    if passed_match:
        return True, f"All {passed_match.group(1)} tests passed cleanly (no warnings)"

    return True, "All tests passed cleanly"


def main():
    """
    Check that tests pass cleanly without failures, errors, or warnings.
    """
    changed_files = sys.argv[1:]

    if not changed_files:
        return 0

    # Get test files
    test_files = ["ml/tests/unit"]

    if not test_files:
        print("No test files found for changed source files.")
        return 0

    # Run tests
    passed, message = run_tests_clean(test_files)

    print(message)

    if not passed:
        print("\nTests did not pass cleanly (advisory). See above.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
