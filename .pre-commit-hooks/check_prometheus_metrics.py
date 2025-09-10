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
Pre-commit hook to validate Prometheus metrics in ML components.

Ensures ML actors and strategies have proper monitoring instrumentation.

"""

import ast
import re
import sys
from pathlib import Path


class PrometheusMetricsValidator(ast.NodeVisitor):
    """
    Validate Prometheus metrics in ML code.

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
        self.current_class = None
        self.is_actor = False
        self.is_strategy = False
        self.has_metrics = False
        self.metric_names: set[str] = set()
        self.required_metrics = {
            "actor": [
                "inference_latency",
                "inference_count",
                "inference_errors",
                "feature_computation_time",
            ],
            "strategy": [
                "signals_received",
                "orders_submitted",
                "position_count",
            ],
        }

    def visit_ClassDef(self, node):
        """
        Check class definitions for Actor/Strategy inheritance.

        Parameters
        ----------
        node : ast.ClassDef
            Class definition node.

        """
        self.current_class = node.name

        # Skip config classes
        if node.name.endswith("Config"):
            self.generic_visit(node)
            return

        # Check if it's an Actor or Strategy
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id == "Actor":
                    self.is_actor = True
                elif base.id == "Strategy":
                    self.is_strategy = True
            elif isinstance(base, ast.Attribute):
                if base.attr == "Actor":
                    self.is_actor = True
                elif base.attr == "Strategy":
                    self.is_strategy = True

        self.generic_visit(node)

        # Validate after visiting all methods
        if self.is_actor or self.is_strategy:
            self._validate_required_metrics()

        self.current_class = None
        self.is_actor = False
        self.is_strategy = False
        self.has_metrics = False
        self.metric_names.clear()

    def visit_FunctionDef(self, node):
        """
        Check function definitions for metric initialization and usage.

        Parameters
        ----------
        node : ast.FunctionDef
            Function definition node.

        """
        # Check for metric initialization in __init__
        if node.name == "__init__":
            self._check_metric_initialization(node)

        # Check for metric updates in other methods
        else:
            self._check_metric_usage(node)

        self.generic_visit(node)

    def _check_metric_initialization(self, node):
        """
        Check if metrics are initialized in __init__.

        Parameters
        ----------
        node : ast.FunctionDef
            The __init__ method node.

        """
        for stmt in ast.walk(node):
            # Look for metric initialization patterns
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self" and "_metric" in target.attr:
                            self.has_metrics = True

            # Look for prometheus metric creation
            elif isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Name):
                    if stmt.func.id in ["Counter", "Histogram", "Gauge", "Summary"]:
                        self.has_metrics = True
                        self._extract_metric_name(stmt)
                elif isinstance(stmt.func, ast.Attribute):
                    if stmt.func.attr in ["Counter", "Histogram", "Gauge", "Summary"]:
                        self.has_metrics = True
                        self._extract_metric_name(stmt)

    def _check_metric_usage(self, node):
        """
        Check if metrics are being updated in methods.

        Parameters
        ----------
        node : ast.FunctionDef
            Method node to check.

        """
        # Look for metric update patterns
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Attribute):
                    # Check for metric methods like inc(), observe(), set()
                    if stmt.func.attr in ["inc", "increment", "observe", "set", "time"]:
                        # Check if it's on a metric attribute
                        if isinstance(stmt.func.value, ast.Attribute):
                            if isinstance(stmt.func.value.value, ast.Name):
                                if stmt.func.value.value.id == "self":
                                    attr_name = stmt.func.value.attr
                                    if (
                                        "metric" in attr_name
                                        or "counter" in attr_name
                                        or "histogram" in attr_name
                                    ):
                                        self.metric_names.add(attr_name)

    def _extract_metric_name(self, call_node):
        """
        Extract metric name from metric creation call.

        Parameters
        ----------
        call_node : ast.Call
            The metric creation call node.

        """
        # Look for name argument
        for keyword in call_node.keywords:
            if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                metric_name = keyword.value.value
                self.metric_names.add(metric_name)

                # Validate metric naming convention
                if not self._validate_metric_name(metric_name):
                    self.warnings.append(
                        f"Line {call_node.lineno}: Metric name '{metric_name}' doesn't follow Prometheus naming conventions",
                    )

    def _validate_metric_name(self, name):
        """
        Validate Prometheus metric naming conventions.

        Parameters
        ----------
        name : str
            Metric name to validate.

        Returns
        -------
        bool
            True if valid, False otherwise.

        """
        # Prometheus naming conventions:
        # - Must match [a-zA-Z_:][a-zA-Z0-9_:]*
        # - Should use snake_case
        # - Should have a unit suffix where applicable (_seconds, _bytes, _total)
        pattern = r"^[a-zA-Z_:][a-zA-Z0-9_:]*$"

        if not re.match(pattern, name):
            return False

        # Check for common unit suffixes
        if any(unit in name for unit in ["time", "duration", "latency"]):
            if not name.endswith(("_seconds", "_ms", "_us")):
                self.warnings.append(
                    f"Time-based metric '{name}' should have a unit suffix (_seconds, _ms, _us)",
                )

        return True

    def _validate_required_metrics(self):
        """
        Validate that required metrics are present.
        """
        if not self.has_metrics:
            component_type = "Actor" if self.is_actor else "Strategy"
            self.errors.append(
                f"{component_type} '{self.current_class}' has no Prometheus metrics defined",
            )
            return

        # Check for required metrics based on component type
        if self.is_actor:
            required = self.required_metrics["actor"]
            component_type = "Actor"
        elif self.is_strategy:
            required = self.required_metrics["strategy"]
            component_type = "Strategy"
        else:
            return

        # Convert metric names to check for partial matches
        metric_names_lower = {name.lower() for name in self.metric_names}

        missing_metrics = []
        for req_metric in required:
            # Check if any metric contains the required metric concept
            found = any(
                req_metric.replace("_", "") in name.replace("_", "") for name in metric_names_lower
            )
            if not found:
                missing_metrics.append(req_metric)

        if missing_metrics:
            self.warnings.append(
                f"{component_type} '{self.current_class}' missing recommended metrics: {', '.join(missing_metrics)}",
            )


def check_file(filepath: str) -> tuple[bool, list[str], list[str]]:
    """
    Check a single file for Prometheus metrics compliance.

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

        validator = PrometheusMetricsValidator(filepath, path)
        validator.visit(tree)

        passed = len(validator.errors) == 0
        return passed, validator.errors, validator.warnings

    except Exception as e:
        return False, [f"Failed to parse {filepath}: {e}"], []


def main():
    """
    Run the pre-commit hook for Prometheus metrics validation.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).

    """
    files = sys.argv[1:]

    # Only check ML actor and strategy files
    ml_files = [
        f
        for f in files
        if f.startswith("ml/") and f.endswith(".py") and ("actors" in f or "strategies" in f)
    ]

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

    print(f"Checking Prometheus metrics in {len(ml_files)} ML file(s)...")

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
        print(f"\n❌ Found {len(total_errors)} metric issue(s)")
        print("Please add Prometheus metrics for monitoring.")
        return 1

    if total_warnings:
        print(f"\n⚠️  Found {len(total_warnings)} warning(s)")
        print("Consider addressing the warnings for better observability.")

    print("\n✅ All Prometheus metrics validated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
