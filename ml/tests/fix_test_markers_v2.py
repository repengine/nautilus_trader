#!/usr/bin/env python3
"""
Enhanced fix script for test marker issues in ML test files.

This script fixes:
1. Import order issues (pytest import before __future__ imports)
2. Indentation issues with decorators
3. Missing @pytest.mark.serial on database tests
4. Missing pytest imports
5. Handles complex cases with docstrings and comments
"""

import ast
import re
from pathlib import Path
from typing import List, Tuple


def parse_file_structure(content: str) -> dict:
    """Parse file structure to understand its components."""
    lines = content.splitlines(keepends=True)

    structure = {
        "shebang": [],
        "docstring": [],
        "future_imports": [],
        "regular_imports": [],
        "rest": []
    }

    i = 0
    # Check for shebang
    if lines and lines[0].startswith("#!"):
        structure["shebang"].append(lines[0])
        i = 1

    # Check for module docstring
    if i < len(lines):
        # Skip empty lines and comments before docstring
        while i < len(lines) and (lines[i].strip() == "" or lines[i].strip().startswith("#")):
            if lines[i].strip().startswith("#"):
                structure["shebang"].append(lines[i])  # Add comments to shebang section
            i += 1

        # Check for docstring
        if i < len(lines) and (lines[i].strip().startswith('"""') or lines[i].strip().startswith("'''")):
            quote = '"""' if '"""' in lines[i] else "'''"
            structure["docstring"].append(lines[i])
            i += 1
            # Multi-line docstring
            if quote not in lines[i-1][3:]:
                while i < len(lines) and quote not in lines[i]:
                    structure["docstring"].append(lines[i])
                    i += 1
                if i < len(lines):
                    structure["docstring"].append(lines[i])
                    i += 1

    # Skip empty lines after docstring
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    # Collect imports
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("from __future__ import"):
            structure["future_imports"].append(lines[i])
        elif line.startswith(("import ", "from ")):
            structure["regular_imports"].append(lines[i])
        elif line == "" and i + 1 < len(lines):
            # Check if next line is still an import
            next_line = lines[i + 1].strip()
            if next_line.startswith(("import ", "from ")):
                i += 1
                continue
            else:
                # End of imports section
                break
        else:
            # Non-import line found
            break
        i += 1

    # Rest of the file
    structure["rest"] = lines[i:]

    return structure


def fix_file_comprehensive(filepath: Path) -> bool:
    """Comprehensively fix a single test file."""
    try:
        with open(filepath) as f:
            content = f.read()
            original_content = content

        # Parse file structure
        structure = parse_file_structure(content)

        # Check if pytest imports are needed
        rest_content = "".join(structure["rest"])
        has_pytest_decorators = "@pytest.mark" in rest_content

        # Check if pytest import exists
        pytest_import_exists = any("import pytest" in line for line in structure["regular_imports"])

        # Add pytest import if needed
        if has_pytest_decorators and not pytest_import_exists:
            structure["regular_imports"].insert(0, "import pytest\n")

        # Fix decorator indentation in the rest of the file
        fixed_rest = fix_decorator_indentation_comprehensive(structure["rest"])

        # Add serial markers to database tests
        fixed_rest = add_serial_markers_comprehensive(fixed_rest)

        # Reconstruct file with correct order
        new_lines = []

        # 1. Shebang and top comments
        new_lines.extend(structure["shebang"])

        # 2. Docstring
        if structure["docstring"]:
            new_lines.extend(structure["docstring"])
            if not structure["docstring"][-1].endswith("\n"):
                new_lines.append("\n")

        # 3. Future imports (must be first after docstring)
        if structure["future_imports"]:
            if new_lines and not new_lines[-1].strip() == "":
                new_lines.append("\n")
            new_lines.extend(structure["future_imports"])

        # 4. Regular imports
        if structure["regular_imports"]:
            if new_lines and not new_lines[-1].strip() == "":
                new_lines.append("\n")
            new_lines.extend(structure["regular_imports"])

        # 5. Add blank line before rest if needed
        if new_lines and not new_lines[-1].strip() == "":
            pass  # Already has blank line
        else:
            new_lines.append("\n")

        # 6. Rest of file
        new_lines.extend(fixed_rest)

        # Join and write back
        new_content = "".join(new_lines)

        if new_content != original_content:
            with open(filepath, "w") as f:
                f.write(new_content)
            return True
        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def fix_decorator_indentation_comprehensive(lines: list[str]) -> list[str]:
    """Fix decorator indentation issues comprehensively."""
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line has a decorator
        if "@pytest.mark" in line:
            # Collect all consecutive decorators
            decorator_lines = []
            decorator_start = i

            # Find the actual indentation by looking for the def/class
            target_indent = None
            j = i
            while j < len(lines):
                if lines[j].strip().startswith("@"):
                    decorator_lines.append(lines[j])
                    j += 1
                elif lines[j].strip().startswith("def ") or lines[j].strip().startswith("class "):
                    # Found the target
                    target_line = lines[j]
                    target_indent = len(target_line) - len(target_line.lstrip())
                    break
                elif lines[j].strip() == "":
                    # Skip empty lines between decorators and def/class
                    j += 1
                else:
                    # Something else - might be a method inside a class
                    break

            if target_indent is not None:
                # Fix all decorator indentations
                for dec_line in decorator_lines:
                    dec_content = dec_line.strip()
                    result.append(" " * target_indent + dec_content + "\n")

                # Skip to after decorators
                i = j
            else:
                # No def/class found, keep as is
                result.append(line)
                i += 1
        else:
            result.append(line)
            i += 1

    return result


def add_serial_markers_comprehensive(lines: list[str]) -> list[str]:
    """Add @pytest.mark.serial to database tests comprehensively."""
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a database marker
        if "@pytest.mark.database" in line:
            # Get indentation
            indent = len(line) - len(line.lstrip())

            # Collect all decorators for this function/class
            decorators = [line]
            j = i + 1

            # Collect consecutive decorators
            while j < len(lines):
                if lines[j].strip().startswith("@"):
                    decorators.append(lines[j])
                    j += 1
                elif lines[j].strip() == "":
                    # Empty line, might have more decorators after
                    j += 1
                else:
                    # End of decorators
                    break

            # Check if serial marker exists
            has_serial = any("@pytest.mark.serial" in d for d in decorators)

            # Add database marker
            result.append(line)

            # Add serial marker if missing (right after database)
            if not has_serial:
                result.append(" " * indent + "@pytest.mark.serial\n")

            # Add rest of decorators (skip the database one we already added)
            for k in range(1, len(decorators)):
                if decorators[k].strip():  # Skip empty lines
                    result.append(decorators[k])

            i = j
        else:
            result.append(line)
            i += 1

    return result


def validate_python_syntax(filepath: Path) -> tuple[bool, str]:
    """Validate Python syntax of a file."""
    try:
        with open(filepath) as f:
            compile(f.read(), str(filepath), "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def main():
    """Fix all test files in the ml/tests directory."""
    test_dir = Path("/home/nate/projects/nautilus_trader/ml/tests")

    # Files that need special attention (from error output)
    problem_files = [
        "contracts/test_registry_behavioral.py",
        "e2e/test_ml_pipeline_integration.py",
        "e2e/test_data_registry_e2e.py",
        "unit/registry/test_enhanced_registry.py",
        "unit/registry/test_registry_statistics.py",
        "unit/registry/test_unified_registry.py",
        "unit/registry/test_model_contracts.py",
        "unit/registry/test_registry_performance.py",
        "unit/registry/test_deployment_manager.py",
        "unit/stores/test_data_store_validation.py",
        "unit/stores/test_live_data_recorder.py",
        "unit/config/test_config.py",
        # Add more problem files from initial scan
        "integration/test_scheduler_feature_store.py",
        "integration/test_stores_concurrency.py",
        "integration/test_end_to_end_pipeline.py",
        "integration/test_scheduler_databento.py",
        "integration/test_registry_store_l2_integration.py",
    ]

    fixed_files = []
    error_files = []

    # First fix the known problem files
    print("Fixing known problem files...")
    for rel_path in problem_files:
        filepath = test_dir / rel_path
        if filepath.exists():
            if fix_file_comprehensive(filepath):
                fixed_files.append(filepath)
                print(f"Fixed: {rel_path}")

    # Then process all other Python test files
    print("\nProcessing all test files...")
    for filepath in test_dir.rglob("*.py"):
        rel_path = filepath.relative_to(test_dir)
        if str(rel_path) not in problem_files:
            if filepath.name.startswith("test_") or filepath.name == "conftest.py":
                if fix_file_comprehensive(filepath):
                    fixed_files.append(filepath)
                    print(f"Fixed: {rel_path}")

    # Verify syntax of all fixed files
    print("\nVerifying syntax...")
    for filepath in fixed_files:
        valid, error = validate_python_syntax(filepath)
        if valid:
            print(f"✓ {filepath.relative_to(test_dir)}")
        else:
            error_files.append((filepath, error))
            print(f"✗ {filepath.relative_to(test_dir)}: {error}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Fixed {len(fixed_files)} files")
    if error_files:
        print(f"Files with remaining errors: {len(error_files)}")
        for filepath, error in error_files[:10]:  # Show first 10 errors
            print(f"  - {filepath.relative_to(test_dir)}: {error}")
    else:
        print("All fixed files have valid syntax!")

    return len(error_files) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
