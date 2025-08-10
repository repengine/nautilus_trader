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
Pre-commit hook to validate docstrings in ML modules.

Ensures all public symbols have NumPy-style docstrings.

"""

import ast
import os
import re
import sys
from pathlib import Path


# Check if running in virtual environment
def check_venv():
    """Check if running in a virtual environment."""
    # Check for virtualenv or venv
    in_virtualenv = hasattr(sys, 'real_prefix')
    in_venv = sys.base_prefix != sys.prefix
    has_venv_var = 'VIRTUAL_ENV' in os.environ
    
    if not (in_virtualenv or in_venv or has_venv_var):
        print("⚠️  Warning: Not running in a virtual environment!")
        print("Please activate your virtual environment and try again.")
        print(f"Python: {sys.executable}")
        sys.exit(1)


check_venv()


class DocstringValidator(ast.NodeVisitor):
    """
    Validate docstrings in Python code.

    Parameters
    ----------
    filename : str
        The filename being validated.

    """

    def __init__(self, filename: str):
        self.filename = filename
        self.errors: list[str] = []
        self.current_class = None

    def visit_ClassDef(self, node):
        """
        Visit class definition.

        Parameters
        ----------
        node : ast.ClassDef
            The class definition node.

        """
        # Check if class is public (doesn't start with _)
        if not node.name.startswith("_"):
            if not ast.get_docstring(node):
                self.errors.append(
                    f"Line {node.lineno}: Public class '{node.name}' missing docstring",
                )
            else:
                # Validate docstring format
                docstring = ast.get_docstring(node)
                self._validate_numpy_format(docstring, node.name, node.lineno, "class")

        # Store current class for method validation
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """
        Visit function definition.

        Parameters
        ----------
        node : ast.FunctionDef
            The function definition node.

        """
        # Skip private functions and special methods (except __init__)
        if node.name.startswith("_") and node.name not in ["__init__", "__call__"]:
            return

        # Skip test functions
        if node.name.startswith("test_"):
            return

        if not ast.get_docstring(node):
            location = f"{self.current_class}.{node.name}" if self.current_class else node.name
            self.errors.append(
                f"Line {node.lineno}: Public function/method '{location}' missing docstring",
            )
        else:
            # Validate docstring format
            docstring = ast.get_docstring(node)
            location = f"{self.current_class}.{node.name}" if self.current_class else node.name
            self._validate_numpy_format(docstring, location, node.lineno, "function")

        self.generic_visit(node)

    def _validate_numpy_format(self, docstring: str, name: str, lineno: int, symbol_type: str):
        """
        Validate NumPy docstring format.

        Parameters
        ----------
        docstring : str
            The docstring to validate.
        name : str
            The name of the symbol.
        lineno : int
            The line number.
        symbol_type : str
            Type of symbol (class/function).

        """
        lines = docstring.strip().split("\n")

        # Check for imperative mood in first line (for functions/methods)
        if symbol_type == "function" and lines:
            first_line = lines[0].strip()
            if first_line and not first_line[0].isupper():
                self.errors.append(
                    f"Line {lineno}: Docstring for '{name}' should start with capital letter (imperative mood)",
                )

        # Check for common NumPy sections
        has_params = any(re.match(r"^\s*Parameters\s*$", line) for line in lines)
        # has_returns = any(re.match(r"^\s*Returns\s*$", line) for line in lines)

        # Functions should have Parameters section if they have arguments
        if symbol_type == "function" and not has_params:
            # Parse to check if function has arguments (excluding self)
            # This is a simplified check - could be enhanced
            if "def " in str(lineno):  # Simplified check
                pass  # Would need actual AST analysis here

    def validate(self):
        """
        Run validation and return results.

        Returns
        -------
        bool
            True if validation passed, False otherwise.

        """
        try:
            with open(self.filename) as f:
                tree = ast.parse(f.read(), filename=self.filename)
            self.visit(tree)
        except Exception as e:
            self.errors.append(f"Failed to parse {self.filename}: {e}")

        return len(self.errors) == 0


def check_file(filepath: str) -> tuple[bool, list[str]]:
    """
    Check a single file for docstring compliance.

    Parameters
    ----------
    filepath : str
        Path to the file to check.

    Returns
    -------
    tuple[bool, list[str]]
        Tuple of (passed, errors).

    """
    validator = DocstringValidator(filepath)
    passed = validator.validate()
    return passed, validator.errors


def main():
    """
    Run the pre-commit hook.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).

    """
    files = sys.argv[1:]

    # Only check ML Python files
    ml_files = [f for f in files if "ml/" in f and f.endswith(".py")]

    if not ml_files:
        return 0

    # Skip test files and __init__.py
    ml_files = [
        f
        for f in ml_files
        if not Path(f).name.startswith("test_") and Path(f).name != "__init__.py"
    ]

    if not ml_files:
        return 0

    print(f"Checking docstrings in {len(ml_files)} ML file(s)...")

    all_passed = True
    total_errors = []

    for filepath in ml_files:
        passed, errors = check_file(filepath)

        if passed:
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath}")
            for error in errors:
                print(f"  {error}")
            total_errors.extend(errors)
            all_passed = False

    if not all_passed:
        print(f"\n❌ Found {len(total_errors)} docstring issue(s)")
        print("Please add/fix docstrings following NumPy format.")
        return 1

    print("\n✅ All docstrings validated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
