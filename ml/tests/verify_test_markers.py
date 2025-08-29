#!/usr/bin/env python3
"""
Verify test markers are correctly applied.

This script checks:
1. All files have valid Python syntax
2. Database tests have @pytest.mark.serial markers
3. Import order is correct
4. Success rate of marker implementation
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple


def check_syntax(filepath: Path) -> tuple[bool, str]:
    """Check Python syntax of a file."""
    try:
        with open(filepath) as f:
            compile(f.read(), str(filepath), "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def check_markers(filepath: Path) -> dict[str, bool]:
    """Check test markers in a file."""
    result = {
        "has_syntax_error": False,
        "has_database_tests": False,
        "database_tests_have_serial": True,
        "import_order_correct": True,
    }

    # Check syntax first
    valid, error = check_syntax(filepath)
    if not valid:
        result["has_syntax_error"] = True
        return result

    with open(filepath) as f:
        content = f.read()
        lines = content.splitlines()

    # Check import order (future imports before other imports)
    future_line = -1
    pytest_line = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("from __future__ import"):
            future_line = i
        elif "import pytest" in line:
            pytest_line = i
            if future_line >= 0 and pytest_line < future_line:
                result["import_order_correct"] = False
                break

    # Check for database tests and serial markers
    if "@pytest.mark.database" in content:
        result["has_database_tests"] = True

        # Find all database test functions/classes
        database_markers = []
        for i, line in enumerate(lines):
            if "@pytest.mark.database" in line:
                database_markers.append(i)

        # Check if each database marker has a serial marker nearby
        for db_line in database_markers:
            # Check within 5 lines before and after for serial marker
            start = max(0, db_line - 5)
            end = min(len(lines), db_line + 5)

            found_serial = False
            for j in range(start, end):
                if "@pytest.mark.serial" in lines[j]:
                    found_serial = True
                    break

            if not found_serial:
                result["database_tests_have_serial"] = False
                break

    return result


def main():
    """Verify test markers in all test files."""
    test_dir = Path("/home/nate/projects/nautilus_trader/ml/tests")

    total_files = 0
    syntax_errors = 0
    import_order_errors = 0
    database_tests_without_serial = 0
    files_with_issues = []

    # Check all Python test files
    for filepath in test_dir.rglob("*.py"):
        if filepath.name.startswith("test_") or filepath.name == "conftest.py":
            total_files += 1

            result = check_markers(filepath)

            if result["has_syntax_error"]:
                syntax_errors += 1
                files_with_issues.append((filepath, "syntax error"))

            if not result["import_order_correct"]:
                import_order_errors += 1
                files_with_issues.append((filepath, "import order"))

            if result["has_database_tests"] and not result["database_tests_have_serial"]:
                database_tests_without_serial += 1
                files_with_issues.append((filepath, "missing serial marker"))

    # Calculate success rate
    issues_count = syntax_errors + import_order_errors + database_tests_without_serial
    success_rate = ((total_files - len({f[0] for f in files_with_issues})) / total_files) * 100 if total_files > 0 else 0

    # Print report
    print("=" * 60)
    print("TEST MARKER VERIFICATION REPORT")
    print("=" * 60)
    print(f"Total test files checked: {total_files}")
    print(f"Files with syntax errors: {syntax_errors}")
    print(f"Files with import order issues: {import_order_errors}")
    print(f"Database tests missing serial marker: {database_tests_without_serial}")
    print(f"Total unique files with issues: {len({f[0] for f in files_with_issues})}")
    print(f"\nSUCCESS RATE: {success_rate:.1f}%")
    print("=" * 60)

    if files_with_issues:
        print("\nFiles requiring attention (first 20):")
        for filepath, issue in files_with_issues[:20]:
            print(f"  - {filepath.relative_to(test_dir)}: {issue}")

    # Detailed check for database tests
    print("\n" + "=" * 60)
    print("DATABASE TEST VERIFICATION")
    print("=" * 60)

    db_test_files = 0
    db_tests_with_serial = 0

    for filepath in test_dir.rglob("*.py"):
        if filepath.name.startswith("test_"):
            with open(filepath) as f:
                content = f.read()
                if "@pytest.mark.database" in content:
                    db_test_files += 1
                    if "@pytest.mark.serial" in content:
                        db_tests_with_serial += 1

    print(f"Files with database tests: {db_test_files}")
    print(f"Files with both database and serial markers: {db_tests_with_serial}")
    print(f"Database test compliance: {(db_tests_with_serial/db_test_files*100) if db_test_files > 0 else 100:.1f}%")

    return success_rate >= 95  # Target 95% success rate


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
