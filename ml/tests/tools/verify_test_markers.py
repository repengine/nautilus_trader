#!/usr/bin/env python3
"""
Verification script for pytest markers in ML test suite.

This script verifies that:
1. All database tests are marked as serial
2. Test markers are correctly applied
3. No conflicting markers exist
4. Import statements are properly added

Author: Test Infrastructure Engineer
Date: 2025-08-28
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


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
    },
    "content_patterns": [
        r"PostgreSQL",
        r"postgres",
        r"connection_string",
        r"CREATE TABLE",
        r"INSERT INTO",
        r"SELECT.*FROM",
        r"database",
        r"psql",
    ],
}


class TestMarkerVerifier:
    """Verifies pytest markers are correctly applied to test files."""

    def __init__(self, test_dir: Path) -> None:
        """
        Initialize verifier.

        Parameters
        ----------
        test_dir : Path
            Root directory containing test files

        """
        self.test_dir = test_dir
        self.issues: list[dict[str, Any]] = []
        self.statistics: dict[str, int] = defaultdict(int)

    def verify_all_tests(self) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """
        Verify all test files in the directory.

        Returns
        -------
        Tuple[List[Dict[str, Any]], Dict[str, int]]
            List of issues found and statistics

        """
        test_files = list(self.test_dir.rglob("test_*.py"))

        for test_file in test_files:
            if "/__pycache__/" in str(test_file):
                continue

            self._verify_file(test_file)

        return self.issues, dict(self.statistics)

    def _verify_file(self, file_path: Path) -> None:
        """
        Verify markers in a single test file.

        Parameters
        ----------
        file_path : Path
            Path to test file

        """
        try:
            content = file_path.read_text()
            tree = ast.parse(content)

            # Check if file has database indicators
            has_database = self._has_database_indicators(content, tree)

            # Extract markers from file
            markers = self._extract_markers(tree)

            # Update statistics
            for marker in markers:
                self.statistics[f"tests_with_{marker}"] += 1

            self.statistics["total_files"] += 1

            # Verify critical requirements
            if has_database:
                self.statistics["database_tests"] += 1

                if "database" not in markers:
                    self.issues.append({
                        "file": str(file_path.relative_to(self.test_dir.parent)),
                        "issue": "Database test missing @pytest.mark.database",
                        "severity": "HIGH",
                    })

                if "serial" not in markers:
                    self.issues.append({
                        "file": str(file_path.relative_to(self.test_dir.parent)),
                        "issue": "Database test missing @pytest.mark.serial (CRITICAL)",
                        "severity": "CRITICAL",
                    })
                elif "serial" in markers:
                    self.statistics["database_tests_marked_serial"] += 1

            # Check for conflicting markers
            if "serial" in markers and "parallel_safe" in markers:
                self.issues.append({
                    "file": str(file_path.relative_to(self.test_dir.parent)),
                    "issue": "Test has conflicting markers: serial and parallel_safe",
                    "severity": "HIGH",
                })

            # Check for pytest import
            if markers and "import pytest" not in content:
                self.issues.append({
                    "file": str(file_path.relative_to(self.test_dir.parent)),
                    "issue": "Test has markers but missing 'import pytest'",
                    "severity": "MEDIUM",
                })

        except Exception as e:
            self.issues.append({
                "file": str(file_path.relative_to(self.test_dir.parent)),
                "issue": f"Failed to parse file: {e}",
                "severity": "ERROR",
            })

    def _has_database_indicators(self, content: str, tree: ast.AST) -> bool:
        """
        Check if file has database-related indicators.

        Parameters
        ----------
        content : str
            File content
        tree : ast.AST
            Parsed AST

        Returns
        -------
        bool
            True if database indicators found

        """
        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(ind in alias.name for ind in DATABASE_INDICATORS["imports"]):
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(ind in node.module for ind in DATABASE_INDICATORS["imports"]):
                    return True

        # Check fixtures
        for fixture in DATABASE_INDICATORS["fixtures"]:
            if f"def {fixture}" in content or ("@pytest.fixture" in content and fixture in content):
                return True
            if f'"{fixture}"' in content or f"'{fixture}'" in content:
                return True

        # Check content patterns
        for pattern in DATABASE_INDICATORS["content_patterns"]:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _extract_markers(self, tree: ast.AST) -> set[str]:
        """
        Extract pytest markers from AST.

        Parameters
        ----------
        tree : ast.AST
            Parsed AST

        Returns
        -------
        Set[str]
            Set of marker names found

        """
        markers = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                for decorator in node.decorator_list:
                    marker_name = self._extract_marker_name(decorator)
                    if marker_name:
                        markers.add(marker_name)

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

    def generate_report(self) -> str:
        """
        Generate verification report.

        Returns
        -------
        str
            Formatted report

        """
        report = []
        report.append("=" * 80)
        report.append("ML TEST SUITE MARKER VERIFICATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Statistics section
        report.append("STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total test files analyzed: {self.statistics['total_files']}")
        report.append(f"Database tests found: {self.statistics['database_tests']}")
        report.append(f"Database tests with serial marker: {self.statistics['database_tests_marked_serial']}")

        if self.statistics["database_tests"] > 0:
            coverage = (self.statistics["database_tests_marked_serial"] /
                       self.statistics["database_tests"] * 100)
            report.append(f"Database serial marker coverage: {coverage:.1f}%")

        report.append("")
        report.append("MARKER DISTRIBUTION:")
        report.append("-" * 40)

        # Sort markers by count
        marker_stats = [(k.replace("tests_with_", ""), v)
                       for k, v in self.statistics.items()
                       if k.startswith("tests_with_")]
        marker_stats.sort(key=lambda x: x[1], reverse=True)

        for marker, count in marker_stats:
            report.append(f"  {marker:<20} {count:>4} files")

        report.append("")

        # Issues section
        if self.issues:
            report.append("ISSUES FOUND:")
            report.append("-" * 40)

            # Group by severity
            critical = [i for i in self.issues if i["severity"] == "CRITICAL"]
            high = [i for i in self.issues if i["severity"] == "HIGH"]
            medium = [i for i in self.issues if i["severity"] == "MEDIUM"]
            errors = [i for i in self.issues if i["severity"] == "ERROR"]

            if critical:
                report.append(f"\nCRITICAL ({len(critical)} issues):")
                for issue in critical:
                    report.append(f"  - {issue['file']}")
                    report.append(f"    {issue['issue']}")

            if high:
                report.append(f"\nHIGH ({len(high)} issues):")
                for issue in high:
                    report.append(f"  - {issue['file']}")
                    report.append(f"    {issue['issue']}")

            if medium:
                report.append(f"\nMEDIUM ({len(medium)} issues):")
                for issue in medium:
                    report.append(f"  - {issue['file']}")
                    report.append(f"    {issue['issue']}")

            if errors:
                report.append(f"\nERRORS ({len(errors)} issues):")
                for issue in errors:
                    report.append(f"  - {issue['file']}")
                    report.append(f"    {issue['issue']}")

            report.append("")
            report.append("SUMMARY:")
            report.append(f"  Total issues: {len(self.issues)}")
            report.append(f"  Critical: {len(critical)}")
            report.append(f"  High: {len(high)}")
            report.append(f"  Medium: {len(medium)}")
            report.append(f"  Errors: {len(errors)}")
        else:
            report.append("✓ NO ISSUES FOUND")
            report.append("")
            report.append("All database tests are properly marked as serial.")
            report.append("All test markers are correctly applied.")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    """Run verification and generate report."""
    import sys

    # Get test directory
    test_dir = Path(__file__).parent.parent

    print(f"Verifying test markers in: {test_dir}")
    print()

    # Run verification
    verifier = TestMarkerVerifier(test_dir)
    issues, stats = verifier.verify_all_tests()

    # Generate and print report
    report = verifier.generate_report()
    print(report)

    # Write report to file
    report_file = test_dir / "MARKER_VERIFICATION_REPORT.txt"
    report_file.write_text(report)
    print(f"\nReport saved to: {report_file}")

    # Exit with error code if critical issues found
    if any(i["severity"] == "CRITICAL" for i in issues):
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
