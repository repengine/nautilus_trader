#!/usr/bin/env python3
"""
Fix test marker issues in ML test files.

This script fixes:
1. Import order issues (pytest import before __future__ imports)
2. Indentation issues with decorators
3. Missing @pytest.mark.serial on database tests
4. Missing pytest imports
"""

import re
from pathlib import Path
from typing import List, Tuple


def fix_import_order(lines: list[str]) -> list[str]:
    """Fix import order - __future__ imports must come first."""
    result = []
    future_imports = []
    pytest_imports = []
    other_lines = []

    in_header = True
    for line in lines:
        if in_header:
            if line.strip().startswith("from __future__ import"):
                future_imports.append(line)
            elif line.strip() == "import pytest" or line.strip().startswith("import pytest"):
                pytest_imports.append(line)
            elif line.strip().startswith("#!") or line.strip().startswith('"""') or line.strip().startswith("'''"):
                result.append(line)
            elif not line.strip() and not other_lines:  # Empty line in header
                continue
            else:
                in_header = False
                other_lines.append(line)
        else:
            other_lines.append(line)

    # Reconstruct with correct order
    final = result.copy()

    # Add future imports first (after docstring)
    if future_imports:
        final.extend(future_imports)
        if not any(line.strip() == "" for line in future_imports):
            final.append("\n")

    # Add pytest import if we have decorators but no pytest import
    has_pytest_decorators = any("@pytest.mark" in line for line in other_lines)
    has_pytest_import = bool(pytest_imports) or any("import pytest" in line for line in other_lines)

    if has_pytest_decorators and not has_pytest_import:
        final.append("import pytest\n")
        if not future_imports:
            final.append("\n")
    elif pytest_imports:
        final.extend(pytest_imports)

    # Add the rest
    final.extend(other_lines)

    return final


def fix_decorator_indentation(lines: list[str]) -> list[str]:
    """Fix indentation issues with pytest decorators."""
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is an indented decorator (wrong)
        if line.strip().startswith("@pytest.mark") and line.startswith("    "):
            # This decorator is indented when it shouldn't be
            # Look for the function definition to get correct indentation
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("@"):
                j += 1

            # Now j points to the line after decorators
            if j < len(lines) and lines[j].strip().startswith("def "):
                # Get the indentation of the def line
                def_line = lines[j]
                def_indent = len(def_line) - len(def_line.lstrip())

                # Fix all decorators to have same indentation as def
                for k in range(i, j):
                    decorator = lines[k].strip()
                    result.append(" " * def_indent + decorator + "\n")

                # Add the def line and continue
                result.append(lines[j])
                i = j + 1
                continue
            elif j < len(lines) and lines[j].strip().startswith("class "):
                # Same for class definitions
                class_line = lines[j]
                class_indent = len(class_line) - len(class_line.lstrip())

                # Fix all decorators to have same indentation as class
                for k in range(i, j):
                    decorator = lines[k].strip()
                    result.append(" " * class_indent + decorator + "\n")

                # Add the class line and continue
                result.append(lines[j])
                i = j + 1
                continue

        result.append(line)
        i += 1

    return result


def add_serial_marker_to_database_tests(lines: list[str]) -> list[str]:
    """Add @pytest.mark.serial to database tests that are missing it."""
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a @pytest.mark.database decorator
        if "@pytest.mark.database" in line:
            # Collect all decorators for this function/class
            decorators = [line]
            j = i + 1

            while j < len(lines) and lines[j].strip().startswith("@"):
                decorators.append(lines[j])
                j += 1

            # Check if @pytest.mark.serial is already present
            has_serial = any("@pytest.mark.serial" in d for d in decorators)

            # Add the database decorator
            result.append(line)

            # Add serial if missing
            if not has_serial:
                indent = len(line) - len(line.lstrip())
                result.append(" " * indent + "@pytest.mark.serial\n")

            # Add other decorators
            for k in range(1, len(decorators)):
                result.append(decorators[k])

            i = j
        else:
            result.append(line)
            i += 1

    return result


def fix_test_file(filepath: Path) -> bool:
    """Fix a single test file. Returns True if changes were made."""
    try:
        with open(filepath) as f:
            content = f.read()
            original_content = content
            lines = content.splitlines(keepends=True)

        # Apply fixes
        lines = fix_import_order(lines)
        lines = fix_decorator_indentation(lines)
        lines = add_serial_marker_to_database_tests(lines)

        # Write back if changed
        new_content = "".join(lines)
        if new_content != original_content:
            with open(filepath, "w") as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Fix all test files in the ml/tests directory."""
    test_dir = Path("/home/nate/projects/nautilus_trader/ml/tests")

    fixed_files = []
    error_files = []

    # Process all Python test files
    for filepath in test_dir.rglob("*.py"):
        if filepath.name.startswith("test_") or filepath.name == "conftest.py":
            if fix_test_file(filepath):
                fixed_files.append(filepath)
                print(f"Fixed: {filepath.relative_to(test_dir)}")

    # Verify syntax of fixed files
    print("\nVerifying fixed files...")
    for filepath in fixed_files:
        try:
            with open(filepath) as f:
                compile(f.read(), filepath, "exec")
            print(f"✓ {filepath.relative_to(test_dir)}")
        except SyntaxError as e:
            error_files.append((filepath, str(e)))
            print(f"✗ {filepath.relative_to(test_dir)}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Fixed {len(fixed_files)} files")
    if error_files:
        print(f"Files with remaining errors: {len(error_files)}")
        for filepath, error in error_files:
            print(f"  - {filepath.relative_to(test_dir)}: {error}")
    else:
        print("All fixed files have valid syntax!")

    return len(error_files) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
