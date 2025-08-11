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
Pre-commit hook to check test coverage for all Python files.

Ensures minimum 80% test coverage across the codebase.

"""

import subprocess
import sys
from pathlib import Path


def get_module_name(file_path):
    """
    Get the module name from file path.
    """
    path = Path(file_path)

    # Skip test files
    if path.name.startswith("test_") or "test" in path.parts:
        return None

    # Convert path to module name
    parts = []
    for part in path.parts:
        if part.endswith(".py"):
            parts.append(part[:-3])
        else:
            parts.append(part)

    # Find the root module (nautilus_trader or nautilus_ml)
    if "nautilus_trader" in parts:
        idx = parts.index("nautilus_trader")
        return ".".join(parts[idx:])
    elif "nautilus_ml" in parts:
        idx = parts.index("nautilus_ml")
        return ".".join(parts[idx:])
    elif "ml" in parts:
        idx = parts.index("ml")
        return ".".join(parts[idx:])

    return None


def check_coverage(changed_files):
    """
    Check test coverage for changed files.
    """
    # Filter for Python files that need coverage
    python_files = []
    modules = set()

    for file_path in changed_files:
        if file_path.endswith(".py"):
            module = get_module_name(file_path)
            if module:
                modules.add(module)
                python_files.append(file_path)

    if not python_files:
        return True, "No Python source files to check"

    print(f"Checking test coverage for {len(python_files)} file(s)...")

    # Build coverage command
    cov_modules = ",".join(modules)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=" + cov_modules,
        "--cov-report=term-missing:skip-covered",
        "--cov-fail-under=80",
        "--no-header",
        "-q",
        "--tb=no",
        "-x",  # Stop on first failure
    ]

    # Run coverage check
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output for coverage info
    coverage_info = []
    failed = False

    for line in result.stdout.split("\n"):
        if "TOTAL" in line and "%" in line:
            # Extract total coverage
            parts = line.split()
            for i, part in enumerate(parts):
                if part.endswith("%"):
                    try:
                        coverage = float(part.rstrip("%"))
                        if coverage < 80:
                            failed = True
                        coverage_info.append(f"Total coverage: {coverage}%")
                    except ValueError:
                        pass
        elif any(module in line for module in modules) and "%" in line:
            # Module-specific coverage
            parts = line.split()
            module_name = parts[0]
            for part in parts:
                if part.endswith("%"):
                    try:
                        coverage = float(part.rstrip("%"))
                        status = "✅" if coverage >= 80 else "❌"
                        coverage_info.append(f"{status} {module_name}: {coverage}%")
                        if coverage < 80:
                            failed = True
                    except ValueError:
                        pass
                    break

    if result.returncode != 0 or failed:
        message = "\n".join(coverage_info) if coverage_info else "Coverage check failed"
        return False, message

    return True, "\n".join(coverage_info) if coverage_info else "All files have ≥80% coverage"


def main():
    """
    Execute the main general test coverage checking process.
    """
    changed_files = sys.argv[1:]

    if not changed_files:
        return 0

    passed, message = check_coverage(changed_files)

    print(message)

    if not passed:
        print("\n❌ Test coverage below 80% threshold!")
        print("Please add more tests to improve coverage.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
