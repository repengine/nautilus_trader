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
Validate Grafana dashboard JSON files.

This script validates ML monitoring dashboard JSON files for structural integrity,
PromQL query syntax, and adherence to ML monitoring standards.

Usage:
    python validate_dashboards.py [options]

Example:
    # Validate all dashboards in directory
    python validate_dashboards.py --input ./dashboards/

    # Validate specific dashboard
    python validate_dashboards.py --file ml-overview.json

    # Validate with detailed output
    python validate_dashboards.py --input ./dashboards/ --detailed

"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class DashboardValidator:
    """
    Validator for Grafana dashboard JSON files.
    """

    def __init__(self, detailed: bool = False) -> None:
        """
        Initialize validator.

        Parameters
        ----------
        detailed : bool, optional
            Whether to provide detailed validation output

        """
        self.detailed = detailed
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_dashboard(
        self,
        dashboard_data: dict[str, Any],
        filename: str = "",
    ) -> tuple[bool, list[str], list[str]]:
        """
        Validate a complete dashboard.

        Parameters
        ----------
        dashboard_data : dict[str, Any]
            Dashboard JSON data
        filename : str, optional
            Filename for error reporting

        Returns
        -------
        tuple[bool, list[str], list[str]]
            Tuple of (is_valid, errors, warnings)

        """
        self.errors = []
        self.warnings = []
        context = f" in {filename}" if filename else ""

        # Basic structure validation
        self._validate_basic_structure(dashboard_data, context)

        # Dashboard metadata validation
        self._validate_metadata(dashboard_data, context)

        # Panels validation
        self._validate_panels(dashboard_data, context)

        # Template variables validation
        self._validate_templating(dashboard_data, context)

        # ML monitoring specific validation
        self._validate_ml_standards(dashboard_data, context)

        return len(self.errors) == 0, self.errors.copy(), self.warnings.copy()

    def _validate_basic_structure(self, dashboard: dict[str, Any], context: str) -> None:
        """
        Validate basic dashboard structure.
        """
        required_fields = ["title", "panels", "templating", "time"]

        for field in required_fields:
            if field not in dashboard:
                self.errors.append(f"Missing required field '{field}'{context}")

        # Check data types
        if "panels" in dashboard and not isinstance(dashboard["panels"], list):
            self.errors.append(f"Field 'panels' must be a list{context}")

        if "templating" in dashboard and not isinstance(dashboard["templating"], dict):
            self.errors.append(f"Field 'templating' must be an object{context}")

    def _validate_metadata(self, dashboard: dict[str, Any], context: str) -> None:
        """
        Validate dashboard metadata.
        """
        # Title validation
        title = dashboard.get("title", "")
        if not title or len(title.strip()) == 0:
            self.errors.append(f"Dashboard title cannot be empty{context}")
        elif len(title) > 255:
            self.warnings.append(f"Dashboard title is very long ({len(title)} characters){context}")

        # UID validation
        uid = dashboard.get("uid")
        if uid:
            if not isinstance(uid, str) or len(uid) == 0:
                self.errors.append(f"UID must be a non-empty string{context}")
            elif not re.match(r"^[a-zA-Z0-9_-]+$", uid):
                self.errors.append(f"UID contains invalid characters: '{uid}'{context}")
            elif len(uid) > 40:
                self.warnings.append(f"UID is very long ({len(uid)} characters){context}")

        # Tags validation
        tags = dashboard.get("tags", [])
        if tags and not isinstance(tags, list):
            self.errors.append(f"Tags must be a list{context}")
        elif tags:
            for tag in tags:
                if not isinstance(tag, str):
                    self.errors.append(f"All tags must be strings{context}")

        # Check for ML monitoring tag
        if "ml-monitoring" not in tags:
            self.warnings.append(f"Dashboard should include 'ml-monitoring' tag{context}")

        # Refresh rate validation
        refresh = dashboard.get("refresh")
        if refresh:
            valid_rates = ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"]
            if refresh not in valid_rates:
                self.warnings.append(f"Non-standard refresh rate: '{refresh}'{context}")

    def _validate_panels(self, dashboard: dict[str, Any], context: str) -> None:
        """
        Validate dashboard panels.
        """
        panels = dashboard.get("panels", [])
        panel_ids = set()

        for i, panel in enumerate(panels):
            panel_context = f"{context} panel {i}"

            if not isinstance(panel, dict):
                self.errors.append(f"Panel must be an object{panel_context}")
                continue

            # Required panel fields
            required_panel_fields = ["id", "type", "title", "gridPos"]
            for field in required_panel_fields:
                if field not in panel:
                    self.errors.append(f"Missing required panel field '{field}'{panel_context}")

            # Panel ID uniqueness
            panel_id = panel.get("id")
            if panel_id is not None:
                if panel_id in panel_ids:
                    self.errors.append(f"Duplicate panel ID: {panel_id}{panel_context}")
                panel_ids.add(panel_id)

            # Grid position validation
            grid_pos = panel.get("gridPos", {})
            if isinstance(grid_pos, dict):
                required_grid_fields = ["h", "w", "x", "y"]
                for field in required_grid_fields:
                    if field not in grid_pos:
                        self.errors.append(f"Missing grid position field '{field}'{panel_context}")
                    elif not isinstance(grid_pos.get(field), int):
                        self.errors.append(
                            f"Grid position '{field}' must be integer{panel_context}",
                        )

                # Validate grid constraints
                if "w" in grid_pos and grid_pos["w"] > 24:
                    self.errors.append(f"Panel width cannot exceed 24 units{panel_context}")
                if "x" in grid_pos and grid_pos["x"] >= 24:
                    self.errors.append(f"Panel x position must be less than 24{panel_context}")

            # Panel-specific validation
            panel_type = panel.get("type")
            if panel_type == "stat":
                self._validate_stat_panel(panel, panel_context)
            elif panel_type == "timeseries":
                self._validate_timeseries_panel(panel, panel_context)
            elif panel_type == "table":
                self._validate_table_panel(panel, panel_context)
            elif panel_type in ["heatmap", "piechart"]:
                self._validate_visualization_panel(panel, panel_context)
            elif panel_type == "row":
                # Row panels are simpler
                pass
            else:
                self.warnings.append(f"Unknown panel type: '{panel_type}'{panel_context}")

            # Query validation for panels with targets
            if "targets" in panel:
                self._validate_panel_queries(panel["targets"], panel_context)

    def _validate_stat_panel(self, panel: dict[str, Any], context: str) -> None:
        """
        Validate stat panel specific configuration.
        """
        # Check for proper options
        options = panel.get("options", {})
        if not options.get("reduceOptions", {}).get("calcs"):
            self.warnings.append(f"Stat panel should specify calculation method{context}")

        # Check thresholds
        field_config = panel.get("fieldConfig", {}).get("defaults", {})
        thresholds = field_config.get("thresholds", {}).get("steps")
        if not thresholds:
            self.warnings.append(f"Stat panel should have thresholds configured{context}")

    def _validate_timeseries_panel(self, panel: dict[str, Any], context: str) -> None:
        """
        Validate timeseries panel specific configuration.
        """
        # Check legend configuration
        options = panel.get("options", {})
        legend = options.get("legend", {})
        if not legend.get("displayMode"):
            self.warnings.append(f"Timeseries panel should configure legend display{context}")

        # Check for appropriate unit
        field_config = panel.get("fieldConfig", {}).get("defaults", {})
        unit = field_config.get("unit")
        if not unit or unit == "short":
            self.warnings.append(f"Timeseries panel should specify appropriate unit{context}")

    def _validate_table_panel(self, panel: dict[str, Any], context: str) -> None:
        """
        Validate table panel specific configuration.
        """
        # Check for transformations
        transformations = panel.get("transformations")
        if not transformations:
            self.warnings.append(
                f"Table panel typically needs transformations for proper display{context}",
            )

    def _validate_visualization_panel(self, panel: dict[str, Any], context: str) -> None:
        """
        Validate visualization panels (heatmap, piechart).
        """
        # Basic validation for visualization panels
        panel_type = panel.get("type")
        if panel_type == "heatmap":
            # Heatmap should have proper bucket configuration
            pass
        elif panel_type == "piechart":
            # Pie chart should have proper legend
            options = panel.get("options", {})
            if not options.get("legend", {}).get("displayMode"):
                self.warnings.append(f"Pie chart should configure legend{context}")

    def _validate_panel_queries(self, targets: list[dict[str, Any]], context: str) -> None:
        """
        Validate PromQL queries in panel targets.
        """
        if not isinstance(targets, list):
            self.errors.append(f"Panel targets must be a list{context}")
            return

        for i, target in enumerate(targets):
            target_context = f"{context} target {i}"

            if not isinstance(target, dict):
                self.errors.append(f"Target must be an object{target_context}")
                continue

            # Check for expression
            expr = target.get("expr")
            if not expr:
                self.warnings.append(f"Target missing PromQL expression{target_context}")
                continue

            # Basic PromQL syntax validation
            self._validate_promql_syntax(expr, target_context)

            # Check for proper legend format
            legend_format = target.get("legendFormat")
            if not legend_format and "rate(" in expr:
                self.warnings.append(f"Rate queries should include legend format{target_context}")

    def _validate_promql_syntax(self, expr: str, context: str) -> None:
        """
        Basic PromQL syntax validation.
        """
        # Check for balanced parentheses
        if expr.count("(") != expr.count(")"):
            self.errors.append(f"Unbalanced parentheses in PromQL: '{expr}'{context}")

        # Check for ML metrics naming convention
        if not re.search(r"ml_[a-z_]+", expr):
            self.warnings.append(f"Query doesn't appear to use ML metrics: '{expr}'{context}")

        # Check for proper aggregation
        if "by (" in expr and not expr.endswith(")"):
            agg_match = re.search(r"by \(([^)]+)\)", expr)
            if agg_match:
                labels = agg_match.group(1).split(",")
                for label in labels:
                    label = label.strip()
                    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", label):
                        self.warnings.append(
                            f"Invalid label name in aggregation: '{label}'{context}",
                        )

    def _validate_templating(self, dashboard: dict[str, Any], context: str) -> None:
        """
        Validate template variables.
        """
        templating = dashboard.get("templating", {})
        variables = templating.get("list", [])

        # Check for required ML variables
        variable_names = [var.get("name") for var in variables if isinstance(var, dict)]

        required_vars = ["datasource"]
        recommended_vars = ["model", "interval"]

        for var in required_vars:
            if var not in variable_names:
                self.errors.append(f"Missing required template variable: '{var}'{context}")

        for var in recommended_vars:
            if var not in variable_names:
                self.warnings.append(f"Missing recommended template variable: '{var}'{context}")

        # Validate individual variables
        for i, variable in enumerate(variables):
            if not isinstance(variable, dict):
                self.errors.append(f"Template variable {i} must be an object{context}")
                continue

            var_name = variable.get("name", f"variable_{i}")
            var_context = f"{context} variable '{var_name}'"

            # Required variable fields
            required_var_fields = ["name", "type"]
            for field in required_var_fields:
                if field not in variable:
                    self.errors.append(f"Missing required variable field '{field}'{var_context}")

            # Variable type validation
            var_type = variable.get("type")
            if var_type == "query" and not variable.get("query"):
                self.errors.append(f"Query variable missing query{var_context}")

    def _validate_ml_standards(self, dashboard: dict[str, Any], context: str) -> None:
        """
        Validate ML monitoring specific standards.
        """
        title = dashboard.get("title", "").lower()
        panels = dashboard.get("panels", [])

        # Dashboard-specific validation based on type
        if "data quality" in title or "data-quality" in dashboard.get("uid", ""):
            self._validate_data_quality_dashboard(panels, context)
        elif "feature engineering" in title or "feature-engineering" in dashboard.get("uid", ""):
            self._validate_feature_dashboard(panels, context)
        elif "model lifecycle" in title or "model-lifecycle" in dashboard.get("uid", ""):
            self._validate_model_dashboard(panels, context)
        elif "performance" in title or "performance-degradation" in dashboard.get("uid", ""):
            self._validate_performance_dashboard(panels, context)
        elif "resource" in title or "resource-utilization" in dashboard.get("uid", ""):
            self._validate_resource_dashboard(panels, context)

    def _validate_data_quality_dashboard(self, panels: list[dict[str, Any]], context: str) -> None:
        """
        Validate data quality dashboard specifics.
        """
        # Look for key metrics
        expected_metrics = [
            "ml_data_missing_values_ratio",
            "ml_data_outliers_detected",
            "ml_data_staleness_seconds",
            "ml_data_cache_hit_ratio",
        ]
        self._check_for_metrics(panels, expected_metrics, "data quality", context)

    def _validate_feature_dashboard(self, panels: list[dict[str, Any]], context: str) -> None:
        """
        Validate feature engineering dashboard specifics.
        """
        expected_metrics = [
            "ml_feature_computation_latency",
            "ml_feature_drift_score",
            "ml_feature_cache_hit_ratio",
            "ml_feature_importance",
        ]
        self._check_for_metrics(panels, expected_metrics, "feature engineering", context)

    def _validate_model_dashboard(self, panels: list[dict[str, Any]], context: str) -> None:
        """
        Validate model lifecycle dashboard specifics.
        """
        expected_metrics = [
            "ml_model_version",
            "ml_model_size_bytes",
            "ml_model_training_duration",
            "ml_model_deployments_total",
        ]
        self._check_for_metrics(panels, expected_metrics, "model lifecycle", context)

    def _validate_performance_dashboard(self, panels: list[dict[str, Any]], context: str) -> None:
        """
        Validate performance degradation dashboard specifics.
        """
        expected_metrics = [
            "ml_model_accuracy_rolling",
            "ml_prediction_distribution_shift",
            "ml_prediction_timeouts_total",
            "ml_model_retraining_required",
        ]
        self._check_for_metrics(panels, expected_metrics, "performance degradation", context)

    def _validate_resource_dashboard(self, panels: list[dict[str, Any]], context: str) -> None:
        """
        Validate resource utilization dashboard specifics.
        """
        expected_metrics = [
            "ml_cpu_usage_percent",
            "ml_memory_usage_percent",
            "ml_gpu_utilization_percent",
            "ml_disk_io_bytes_total",
        ]
        self._check_for_metrics(panels, expected_metrics, "resource utilization", context)

    def _check_for_metrics(
        self,
        panels: list[dict[str, Any]],
        expected_metrics: list[str],
        dashboard_type: str,
        context: str,
    ) -> None:
        """
        Check if expected metrics are present in panels.
        """
        # Extract all expressions from panels
        all_expressions = []
        for panel in panels:
            targets = panel.get("targets", [])
            for target in targets:
                if isinstance(target, dict) and "expr" in target:
                    all_expressions.append(target["expr"])

        dashboard_content = " ".join(all_expressions).lower()

        missing_metrics = []
        for metric in expected_metrics:
            if metric not in dashboard_content:
                missing_metrics.append(metric)

        if missing_metrics:
            self.warnings.append(
                f"{dashboard_type.title()} dashboard missing key metrics: {', '.join(missing_metrics)}{context}",
            )


def validate_file(file_path: Path, detailed: bool = False) -> tuple[bool, list[str], list[str]]:
    """
    Validate a single dashboard file.

    Parameters
    ----------
    file_path : Path
        Path to dashboard JSON file
    detailed : bool, optional
        Whether to provide detailed validation output

    Returns
    -------
    tuple[bool, list[str], list[str]]
        Tuple of (is_valid, errors, warnings)

    """
    validator = DashboardValidator(detailed=detailed)

    try:
        with open(file_path, encoding="utf-8") as f:
            dashboard_data = json.load(f)

        return validator.validate_dashboard(dashboard_data, file_path.name)

    except FileNotFoundError:
        return False, [f"File not found: {file_path}"], []

    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in {file_path.name}: {e}"], []

    except Exception as e:
        return False, [f"Validation error in {file_path.name}: {e}"], []


def validate_directory(
    input_dir: Path,
    detailed: bool = False,
) -> dict[str, tuple[bool, list[str], list[str]]]:
    """
    Validate all dashboard files in a directory.

    Parameters
    ----------
    input_dir : Path
        Directory containing dashboard JSON files
    detailed : bool, optional
        Whether to provide detailed validation output

    Returns
    -------
    dict[str, tuple[bool, list[str], list[str]]]
        Dictionary mapping filenames to validation results

    """
    results: dict[str, tuple[bool, list[str], list[str]]] = {}
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in: {input_dir}")
        return results

    logger.info(f"Validating {len(json_files)} dashboard files")

    for json_file in sorted(json_files):
        is_valid, errors, warnings = validate_file(json_file, detailed)
        results[json_file.name] = (is_valid, errors, warnings)

    return results


def main() -> int:
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Validate Grafana dashboard JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Input directory containing dashboard JSON files",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Validate specific dashboard file",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Provide detailed validation output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show errors and summary",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.file:
        logger.error("Must specify either --input directory or --file")
        return 1

    if args.input and args.file:
        logger.error("Cannot specify both --input and --file")
        return 1

    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Validate dashboards
    results = {}

    try:
        if args.file:
            # Validate single file
            is_valid, errors, warnings = validate_file(args.file, args.detailed)
            results[args.file.name] = (is_valid, errors, warnings)
        else:
            # Validate directory
            results = validate_directory(args.input, args.detailed)

        # Print results
        total_files = len(results)
        valid_files = 0
        total_errors = 0
        total_warnings = 0

        for filename, (is_valid, errors, warnings) in results.items():
            if is_valid and (not args.strict or len(warnings) == 0):
                valid_files += 1
                if not args.quiet:
                    logger.info(f" {filename}: Valid")
            else:
                logger.error(f" {filename}: Invalid")

            total_errors += len(errors)
            total_warnings += len(warnings)

            # Show errors
            for error in errors:
                logger.error(f"  ERROR: {error}")

            # Show warnings
            for warning in warnings:
                if args.strict:
                    logger.error(f"  ERROR (strict): {warning}")
                else:
                    logger.warning(f"  WARNING: {warning}")

        # Print summary
        logger.info("Validation completed:")
        logger.info(f"  Total files: {total_files}")
        logger.info(f"  Valid files: {valid_files}")
        logger.info(f"  Invalid files: {total_files - valid_files}")
        logger.info(f"  Total errors: {total_errors}")
        logger.info(f"  Total warnings: {total_warnings}")

        # Return appropriate exit code
        if args.strict:
            return 0 if total_errors == 0 and total_warnings == 0 else 1
        else:
            return 0 if total_errors == 0 else 1

    except KeyboardInterrupt:
        logger.info("Validation cancelled by user")
        return 1

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
