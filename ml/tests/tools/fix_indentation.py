#!/usr/bin/env python3
"""
Fix indentation issues caused by incorrect marker placement.
"""

from __future__ import annotations

import re
from pathlib import Path


def fix_test_file_indentation(filepath: Path) -> bool:
    """
    Fix indentation in a test file.

    Parameters
    ----------
    filepath : Path
        Path to test file

    Returns
    -------
    bool
        True if file was modified
    """
    try:
        content = filepath.read_text()
        lines = content.split("\n")

        fixed = False
        new_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this is a marker placed at wrong indentation
            if re.match(r"^@pytest\.mark\.\w+$", line.strip()) and i < len(lines) - 1:
                # Look ahead to see what follows
                next_line = lines[i + 1] if i + 1 < len(lines) else ""

                # Check if next line is a method/function definition
                if re.match(r"^\s*(def|async def)\s+\w+", next_line):
                    # Get the indentation of the method
                    match = re.match(r"^(\s*)", next_line)
                    indent = match.group(1) if match else ""

                    # Apply same indentation to marker
                    new_lines.append(indent + line.strip())
                    fixed = True
                elif re.match(r"^@pytest\.mark\.\w+$", next_line.strip()):
                    # Another marker follows, check further ahead
                    j = i + 1
                    while j < len(lines) and re.match(r"^@pytest\.mark\.\w+$", lines[j].strip()):
                        j += 1

                    if j < len(lines) and re.match(r"^\s*(def|async def)\s+\w+", lines[j]):
                        # Get the indentation of the method
                        match = re.match(r"^(\s*)", lines[j])
                        indent = match.group(1) if match else ""

                        # Apply same indentation to this marker
                        new_lines.append(indent + line.strip())
                        fixed = True
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

            i += 1

        if fixed:
            new_content = "\n".join(new_lines)
            filepath.write_text(new_content)
            return True

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

    return False


def main():
    """Fix indentation in all test files."""
    test_dir = Path(__file__).parent.parent

    print(f"Fixing indentation in test files under: {test_dir}")

    test_files = list(test_dir.rglob("test_*.py"))
    fixed_count = 0

    for filepath in test_files:
        if "/__pycache__/" in str(filepath):
            continue

        if fix_test_file_indentation(filepath):
            relative_path = filepath.relative_to(test_dir.parent)
            print(f"  ✓ Fixed {relative_path}")
            fixed_count += 1

    print(f"\nFixed indentation in {fixed_count} files")


if __name__ == "__main__":
    main()
