#!/usr/bin/env python3
"""
Fix pytest markers for all test files.

This script ensures all database tests are properly marked with both
@pytest.mark.database and @pytest.mark.serial markers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Database-related imports and fixtures that indicate a test needs database access
DATABASE_INDICATORS = {
    "imports": {
        "psycopg2",
        "psycopg",
        "sqlalchemy",
        "asyncpg",
        "ml.stores.feature_store",
        "ml.stores.model_store",
        "ml.stores.strategy_store",
        "ml.stores.data_store",
        "ml.registry.postgres_backend",
        "ml.core.db_engine",
        "ml.core.integration",
        "ml.stores.partition_manager",
        "ml.stores.data_processor",
        "ml.stores.live_data_recorder",
    },
    "fixtures": {
        "test_database",
        "pg_connection",
        "postgres_backend",
        "db_session",
        "database_connection",
        "persistence_manager",
        "mock_persistence_manager",
        "feature_store",
        "model_store",
        "strategy_store",
        "data_store",
        "clean_postgres_db",
        "data_processor",
    },
    "content_patterns": [
        r"PostgreSQL",
        r"postgres",
        r"connection_string",
        r"CREATE TABLE",
        r"INSERT INTO",
        r"SELECT.*FROM",
        r"\.execute\(",
        r"\.query\(",
        r"psql",
    ],
}


class TestMarkerFixer:
    """Fix pytest markers in test files."""

    def __init__(self) -> None:
        """Initialize fixer."""
        self.stats = {
            "files_processed": 0,
            "files_fixed": 0,
            "classes_marked": 0,
            "functions_marked": 0,
            "database_tests_fixed": 0,
        }

    def fix_file(self, filepath: Path) -> bool:
        """
        Fix markers in a single test file.

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
            original_content = content

            # Check if file needs database markers
            if not self._needs_database_markers(content):
                return False

            # Parse the file
            tree = ast.parse(content)
            lines = content.split("\n")

            # Track modifications
            modifications = []

            # Find all test classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef | ast.FunctionDef):
                    if self._is_test_node(node):
                        markers_to_add = self._get_required_markers(node, lines)
                        if markers_to_add:
                            modifications.append((node.lineno - 1, markers_to_add, node))

            if not modifications:
                return False

            # Apply modifications in reverse order to preserve line numbers
            modifications.sort(reverse=True)

            for line_idx, markers, node in modifications:
                # Find where to insert markers (before any existing decorators)
                insert_idx = line_idx
                while insert_idx > 0 and lines[insert_idx - 1].strip().startswith("@"):
                    insert_idx -= 1

                # Add markers
                for marker in reversed(markers):
                    lines.insert(insert_idx, f"@pytest.mark.{marker}")

                # Update stats
                if isinstance(node, ast.ClassDef):
                    self.stats["classes_marked"] += 1
                else:
                    self.stats["functions_marked"] += 1

            # Ensure pytest is imported
            if not self._has_pytest_import(lines):
                self._add_pytest_import(lines)

            # Write back modified content
            new_content = "\n".join(lines)
            if new_content != original_content:
                filepath.write_text(new_content)
                self.stats["files_fixed"] += 1
                self.stats["database_tests_fixed"] += 1
                return True

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

        return False

    def _needs_database_markers(self, content: str) -> bool:
        """
        Check if file needs database markers.

        Parameters
        ----------
        content : str
            File content

        Returns
        -------
        bool
            True if file needs database markers

        """
        # Check imports
        for indicator in DATABASE_INDICATORS["imports"]:
            if indicator in content:
                return True

        # Check fixtures
        for fixture in DATABASE_INDICATORS["fixtures"]:
            if f'"{fixture}"' in content or f"'{fixture}'" in content:
                return True
            if f"def {fixture}" in content:
                return True

        # Check content patterns
        for pattern in DATABASE_INDICATORS["content_patterns"]:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _is_test_node(self, node: ast.AST) -> bool:
        """
        Check if AST node is a test class or function.

        Parameters
        ----------
        node : ast.AST
            AST node

        Returns
        -------
        bool
            True if node is a test

        """
        if isinstance(node, ast.ClassDef):
            return node.name.startswith("Test")
        elif isinstance(node, ast.FunctionDef):
            return node.name.startswith("test_")
        return False

    def _get_required_markers(self, node: ast.AST, lines: list[str]) -> list[str]:
        """
        Get markers that need to be added to a node.

        Parameters
        ----------
        node : ast.AST
            AST node
        lines : List[str]
            File lines

        Returns
        -------
        List[str]
            List of markers to add

        """
        required = ["database", "serial"]
        existing = self._get_existing_markers(node, lines)

        # Also add integration marker for integration tests
        if "/integration/" in str(lines):
            required.append("integration")

        return [m for m in required if m not in existing]

    def _get_existing_markers(self, node: ast.AST, lines: list[str]) -> set[str]:
        """
        Get existing markers on a node.

        Parameters
        ----------
        node : ast.AST
            AST node
        lines : List[str]
            File lines

        Returns
        -------
        Set[str]
            Set of existing marker names

        """
        markers = set()

        # Check decorators on the node
        for decorator in node.decorator_list:
            marker_name = self._extract_marker_name(decorator)
            if marker_name:
                markers.add(marker_name)

        # Also check lines above the node for markers
        start_line = max(0, node.lineno - 10)
        end_line = node.lineno
        for i in range(start_line, end_line):
            if i < len(lines):
                line = lines[i]
                match = re.search(r"@pytest\.mark\.(\w+)", line)
                if match:
                    markers.add(match.group(1))

        return markers

    def _extract_marker_name(self, decorator: ast.AST) -> str | None:
        """
        Extract marker name from decorator node.

        Parameters
        ----------
        decorator : ast.AST
            Decorator node

        Returns
        -------
        str | None
            Marker name or None

        """
        if isinstance(decorator, ast.Attribute):
            if (isinstance(decorator.value, ast.Attribute) and
                isinstance(decorator.value.value, ast.Name) and
                decorator.value.value.id == "pytest" and
                decorator.value.attr == "mark"):
                return decorator.attr
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                return self._extract_marker_name(decorator.func)

        return None

    def _has_pytest_import(self, lines: list[str]) -> bool:
        """
        Check if pytest is imported.

        Parameters
        ----------
        lines : List[str]
            File lines

        Returns
        -------
        bool
            True if pytest is imported

        """
        for line in lines[:50]:  # Check first 50 lines
            if "import pytest" in line:
                return True
        return False

    def _add_pytest_import(self, lines: list[str]) -> None:
        """
        Add pytest import to file.

        Parameters
        ----------
        lines : List[str]
            File lines (modified in place)

        """
        # Find where to add import (after other imports)
        insert_idx = 0
        for i, line in enumerate(lines[:100]):
            if line.startswith(("import ", "from ")):
                insert_idx = i + 1
            elif insert_idx > 0 and line.strip() and not line.startswith("#"):
                # Found first non-import, non-comment line after imports
                break

        # Add import
        if insert_idx > 0:
            lines.insert(insert_idx, "import pytest")
            lines.insert(insert_idx + 1, "")

    def fix_all_tests(self, test_dir: Path) -> None:
        """
        Fix all test files in directory.

        Parameters
        ----------
        test_dir : Path
            Test directory

        """
        test_files = list(test_dir.rglob("test_*.py"))

        print(f"Processing {len(test_files)} test files...")

        for filepath in test_files:
            if "/__pycache__/" in str(filepath):
                continue

            self.stats["files_processed"] += 1

            if self.fix_file(filepath):
                relative_path = filepath.relative_to(test_dir.parent)
                print(f"  ✓ Fixed {relative_path}")

    def print_summary(self) -> None:
        """Print summary of fixes applied."""
        print("\n" + "=" * 60)
        print("MARKER FIX SUMMARY")
        print("=" * 60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files fixed: {self.stats['files_fixed']}")
        print(f"Database tests fixed: {self.stats['database_tests_fixed']}")
        print(f"Classes marked: {self.stats['classes_marked']}")
        print(f"Functions marked: {self.stats['functions_marked']}")
        print("=" * 60)


def main():
    """Run the marker fixer."""
    import sys

    # Get test directory
    test_dir = Path(__file__).parent.parent

    print(f"Fixing test markers in: {test_dir}")
    print()

    # Run fixer
    fixer = TestMarkerFixer()
    fixer.fix_all_tests(test_dir)
    fixer.print_summary()

    print("\nDone! All database tests should now have proper markers.")
    print("Run verify_test_markers.py to confirm all issues are resolved.")


if __name__ == "__main__":
    main()
