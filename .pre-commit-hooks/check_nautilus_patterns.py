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
Pre-commit hook to validate Nautilus Trader patterns in ML code.

Ensures ML code follows Nautilus architectural patterns and best practices.

"""

import ast
import sys
from pathlib import Path


class NautilusPatternValidator(ast.NodeVisitor):
    """
    Validate Nautilus patterns in ML code.

    Parameters
    ----------
    filename : str
        The filename being validated.
    filepath : Path
        The Path object for the file.

    """

    def __init__(self, filename: str, filepath: Path):
        self.filename = filename
        self.filepath = filepath
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.imports: dict[str, str] = {}
        self.current_class = None
        self.in_init = False
        self.has_on_start = False

    def visit_Import(self, node):
        """
        Track imports to validate hot/cold path separation.

        Parameters
        ----------
        node : ast.Import
            Import node.

        """
        for alias in node.names:
            self.imports[alias.name] = alias.asname or alias.name

    def visit_ImportFrom(self, node):
        """
        Track from imports.

        Parameters
        ----------
        node : ast.ImportFrom
            ImportFrom node.

        """
        if node.module:
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                self.imports[alias.name] = alias.asname or alias.name
                self.imports[full_name] = full_name

    def visit_ClassDef(self, node):
        """
        Validate class definitions.

        Parameters
        ----------
        node : ast.ClassDef
            Class definition node.

        """
        self.current_class = node.name
        self.has_on_start = False

        # Check Actor patterns
        if self._is_actor_class(node):
            self._validate_actor_patterns(node)

        # Check Strategy patterns
        if self._is_strategy_class(node):
            self._validate_strategy_patterns(node)

        # Check Config patterns
        if node.name.endswith("Config"):
            self._validate_config_patterns(node)

        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        """
        Validate function definitions.

        Parameters
        ----------
        node : ast.FunctionDef
            Function definition node.

        """
        old_in_init = self.in_init

        if node.name == "__init__":
            self.in_init = True
            if self.current_class and self._is_strategy_class_name(self.current_class):
                self._validate_strategy_init(node)

        elif node.name == "on_start":
            self.has_on_start = True

        elif node.name == "on_bar" or node.name == "on_data":
            self._validate_event_handler(node)

        self.generic_visit(node)
        self.in_init = old_in_init

    def visit_Attribute(self, node):
        """
        Check for prohibited attribute access.

        Parameters
        ----------
        node : ast.Attribute
            Attribute access node.

        """
        if self.in_init and self.current_class:
            # Check for clock/logger access in __init__
            if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                if node.value.value.id == "self" and node.value.attr in ["clock", "logger"]:
                    self.errors.append(
                        f"Line {node.lineno}: Accessing self.{node.value.attr} in __init__ is prohibited",
                    )

    def _is_actor_class(self, node):
        """
        Check if class inherits from Actor.
        """
        return any(
            (isinstance(base, ast.Name) and base.id == "Actor")
            or (isinstance(base, ast.Attribute) and base.attr == "Actor")
            for base in node.bases
        )

    def _is_strategy_class(self, node):
        """
        Check if class inherits from Strategy.
        """
        return any(
            (isinstance(base, ast.Name) and base.id == "Strategy")
            or (isinstance(base, ast.Attribute) and base.attr == "Strategy")
            for base in node.bases
        )

    def _is_strategy_class_name(self, name):
        """
        Check if class name suggests it's a strategy.
        """
        return "Strategy" in name

    def _validate_actor_patterns(self, node):
        """
        Validate Actor-specific patterns.

        Parameters
        ----------
        node : ast.ClassDef
            Actor class node.

        """
        # Check for inference/feature paths
        if "inference" in str(self.filepath) or "actors" in str(self.filepath):
            # Hot path validations
            if "pandas" in self.imports or "pd" in self.imports:
                self.errors.append(
                    f"Line {node.lineno}: Actor '{node.name}' uses pandas in hot path (inference)",
                )

        # More actor validations can be added here

    def _validate_strategy_patterns(self, node):
        """
        Validate Strategy-specific patterns.

        Parameters
        ----------
        node : ast.ClassDef
            Strategy class node.

        """
        # Strategies should have on_start for initialization
        method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

        if "__init__" in method_names and "on_start" not in method_names:
            self.warnings.append(
                f"Line {node.lineno}: Strategy '{node.name}' has __init__ but no on_start() method",
            )

    def _validate_config_patterns(self, node):  # noqa: C901
        """
        Validate Config class patterns.

        Parameters
        ----------
        node : ast.ClassDef
            Config class node.

        """
        # Check for frozen=True in class decorators or bases
        has_frozen = False

        # Check bases for frozen parameter
        for base in node.bases:
            # Check if base is a tuple with frozen=True
            if isinstance(base, ast.Name) and base.id.endswith("Config"):
                # Look for frozen in keywords
                continue
            if isinstance(base, ast.Call):
                for keyword in base.keywords:
                    if keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant):
                        if keyword.value.value is True:
                            has_frozen = True

        # Also check keywords directly on the class
        if hasattr(node, "keywords"):
            for keyword in node.keywords:
                if keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant):
                    if keyword.value.value is True:
                        has_frozen = True

        # For inheritance syntax like class Foo(Bar, frozen=True)
        # The frozen appears as a keyword, not in bases
        # So we shouldn't error if we have a Config base class
        for base in node.bases:
            if isinstance(base, ast.Name) and "Config" in base.id:
                # Assume configs with Config base are properly frozen
                has_frozen = True
                break

        if (
            not has_frozen
            and "Config" in node.name
            and not any(isinstance(base, ast.Name) and "Config" in base.id for base in node.bases)
        ):
            self.errors.append(
                f"Line {node.lineno}: Config class '{node.name}' should use frozen=True",
            )

    def _validate_strategy_init(self, node):
        """
        Validate Strategy __init__ method.

        Parameters
        ----------
        node : ast.FunctionDef
            __init__ method node.

        """
        # Already handled by visit_Attribute for clock/logger access

    def _validate_event_handler(self, node):
        """
        Validate event handler patterns.

        Parameters
        ----------
        node : ast.FunctionDef
            Event handler method node.

        """
        # Check for blocking operations
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    # Check for synchronous I/O operations
                    if child.func.attr in ["read", "write", "open"]:
                        self.warnings.append(
                            f"Line {child.lineno}: Potential blocking I/O in event handler '{node.name}'",
                        )

                    # Check for sleep/wait
                    if child.func.attr in ["sleep", "wait"]:
                        self.errors.append(
                            f"Line {child.lineno}: Blocking operation in event handler '{node.name}'",
                        )

    def validate_hot_cold_separation(self):
        """
        Validate hot/cold path separation rules.
        """
        # Additional validation based on file path
        path_str = str(self.filepath)

        if "inference" in path_str or "actors" in path_str:
            # Hot path checks
            if "polars" in self.imports or "pl" in self.imports:
                self.warnings.append(
                    "Polars should be used in cold path only (training), not in inference/actors",
                )

        elif "training" in path_str:
            # Cold path checks
            if "pandas" in self.imports or "pd" in self.imports:
                self.warnings.append(
                    "Consider using Polars instead of pandas for better performance in training",
                )


def check_file(filepath: str) -> tuple[bool, list[str], list[str]]:
    """
    Check a single file for Nautilus pattern compliance.

    Parameters
    ----------
    filepath : str
        Path to the file to check.

    Returns
    -------
    tuple[bool, list[str], list[str]]
        Tuple of (passed, errors, warnings).

    """
    path = Path(filepath)

    try:
        with open(filepath) as f:
            content = f.read()
            tree = ast.parse(content, filename=filepath)

        validator = NautilusPatternValidator(filepath, path)
        validator.visit(tree)
        validator.validate_hot_cold_separation()

        passed = len(validator.errors) == 0
        return passed, validator.errors, validator.warnings

    except Exception as e:
        return False, [f"Failed to parse {filepath}: {e}"], []


def main():  # noqa: C901
    """
    Main entry point for the pre-commit hook.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).

    """
    files = sys.argv[1:]

    # Only check ML Python files
    ml_files = [f for f in files if f.startswith("ml/") and f.endswith(".py")]

    if not ml_files:
        return 0

    # Skip test files
    ml_files = [f for f in ml_files if not Path(f).name.startswith("test_")]

    if not ml_files:
        return 0

    print(f"Checking Nautilus patterns in {len(ml_files)} ML file(s)...")

    all_passed = True
    total_errors = []
    total_warnings = []

    for filepath in ml_files:
        passed, errors, warnings = check_file(filepath)

        if passed and not warnings:
            print(f"✓ {filepath}")
        elif passed and warnings:
            print(f"⚠ {filepath}")
            for warning in warnings:
                print(f"  Warning: {warning}")
            total_warnings.extend(warnings)
        else:
            print(f"✗ {filepath}")
            for error in errors:
                print(f"  Error: {error}")
            for warning in warnings:
                print(f"  Warning: {warning}")
            total_errors.extend(errors)
            total_warnings.extend(warnings)
            all_passed = False

    if not all_passed:
        print(f"\n❌ Found {len(total_errors)} pattern violation(s)")
        print("Please fix the errors to follow Nautilus patterns.")
        return 1

    if total_warnings:
        print(f"\n⚠️  Found {len(total_warnings)} warning(s)")
        print("Consider addressing the warnings for better code quality.")

    print("\n✅ All Nautilus patterns validated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
