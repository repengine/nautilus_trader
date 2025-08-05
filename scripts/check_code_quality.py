#!/usr/bin/env python
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
Code quality checker for new Nautilus Trader modules.

This script enforces:
- 90% test coverage minimum
- 0 mypy errors
- Ruff linting passes with 100 char line limit
- All tests pass

"""

import argparse
import subprocess
import sys
from pathlib import Path


class Colors:
    """
    ANSI color codes for terminal output.
    """

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    """
    Print a formatted header.
    """
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_success(text: str) -> None:
    """
    Print success message.
    """
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """
    Print error message.
    """
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """
    Print warning message.
    """
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def check_file_exists(filepath: Path) -> bool:
    """
    Check if file exists.
    """
    if not filepath.exists():
        print_error(f"File not found: {filepath}")
        return False
    return True


def run_mypy(filepath: Path) -> bool:
    """
    Run mypy type checking.
    """
    print_header("Running MyPy Type Checking")

    cmd = ["mypy", str(filepath), "--config", "pyproject.toml"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print_success("MyPy: No type errors found!")
        return True
    else:
        print_error("MyPy found type errors:")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return False


def run_ruff(filepath: Path) -> bool:
    """
    Run ruff linting with 100 char line limit.
    """
    print_header("Running Ruff Linting")

    # Check with 100 char line limit
    cmd = ["ruff", "check", str(filepath), "--line-length", "100"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print_success("Ruff: No linting violations found!")
        return True
    else:
        print_error("Ruff found violations:")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return False


def run_tests(test_filepath: Path) -> bool:
    """
    Run pytest for the test file.
    """
    print_header("Running Tests")

    if not test_filepath.exists():
        print_error(f"Test file not found: {test_filepath}")
        print_warning("Expected test file naming: test_<module_name>.py")
        return False

    cmd = ["pytest", str(test_filepath), "-v"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print_success("All tests passed!")
        return True
    else:
        print_error("Tests failed:")
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return False


def check_test_coverage(module_path: Path, test_path: Path) -> bool:
    """
    Check test coverage for the module.
    """
    print_header("Checking Test Coverage")

    if not test_path.exists():
        print_error(f"Test file not found: {test_path}")
        return False

    # Run coverage
    cmd = [
        "pytest",
        str(test_path),
        f"--cov={module_path.stem}",
        "--cov-report=term-missing",
        "--cov-fail-under=90",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print_success("Test coverage ≥90%!")
        print(result.stdout.split("\n")[-3])  # Print coverage summary
        return True
    else:
        print_error("Test coverage < 90%:")
        # Extract coverage percentage from output
        output_lines = result.stdout.split("\n")
        for line in output_lines:
            if "TOTAL" in line or module_path.stem in line:
                print(line)
        return False


def main():
    """
    Run code quality checks for Nautilus Trader modules.
    """
    parser = argparse.ArgumentParser(
        description="Check code quality for Nautilus Trader modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_code_quality.py path/to/my_module.py
  python scripts/check_code_quality.py path/to/my_module.py --test-file tests/test_my_module.py
        """,
    )

    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to the Python module to check",
    )

    parser.add_argument(
        "--test-file",
        type=Path,
        help="Path to the test file (defaults to tests/test_<module>.py)",
    )

    args = parser.parse_args()

    # Check if file exists
    if not check_file_exists(args.filepath):
        sys.exit(1)

    # Determine test file path
    if args.test_file:
        test_path = args.test_file
    else:
        # Default test path
        test_path = Path("tests") / f"test_{args.filepath.stem}.py"

    print(f"{Colors.BOLD}Checking code quality for: {args.filepath}{Colors.ENDC}")
    print(f"{Colors.BOLD}Test file: {test_path}{Colors.ENDC}")

    # Run all checks
    checks = [
        ("MyPy", lambda: run_mypy(args.filepath)),
        ("Ruff", lambda: run_ruff(args.filepath)),
        ("Tests", lambda: run_tests(test_path)),
        ("Coverage", lambda: check_test_coverage(args.filepath, test_path)),
    ]

    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()

    # Summary
    print_header("Summary")

    all_passed = all(results.values())

    for check_name, passed in results.items():
        if passed:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")

    if all_passed:
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}🎉 All quality checks passed! Ready to commit.{Colors.ENDC}",
        )
        sys.exit(0)
    else:
        print(
            f"\n{Colors.RED}{Colors.BOLD}❌ Some checks failed. Please fix before committing.{Colors.ENDC}",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
