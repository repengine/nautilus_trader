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
Integration test for Grafana dashboard system.

This script tests the complete dashboard integration system including
dashboard validation, client connectivity, and rendering performance.

Usage:
    python test_integration.py [options]

Example:
    # Run all tests
    python test_integration.py --all

    # Test dashboard validation only
    python test_integration.py --test-validation

    # Test with live Grafana instance
    python test_integration.py --test-live --grafana-url http://localhost:3000 --api-token <token>

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TypedDict


class TestResult(TypedDict):
    """
    Type definition for test results.
    """

    success: bool
    errors: list[str]
    warnings: list[str]
    info: list[str]


# Set up path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dashboard_factory import GrafanaDashboardFactory
    from dashboard_factory import GrafanaPanelFactory
    from grafana_client import GrafanaAPIError
    from grafana_client import GrafanaClient

    from scripts.validate_config import ConfigValidator
    from scripts.validate_dashboards import validate_directory
    from scripts.validate_dashboards import validate_file
except ImportError as e:
    logger.error(f"Test failed: {e}")
    logger.info("Make sure you're running from the ml/monitoring directory")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """
    Integration tester for Grafana dashboard system.
    """

    def __init__(self) -> None:
        """
        Initialize integration tester.
        """
        self.test_results: dict[str, TestResult] = {}
        self.temp_files: list[Path] = []

    def run_all_tests(self) -> dict[str, TestResult]:
        """
        Run all integration tests.

        Returns
        -------
        dict[str, TestResult]
            Test results

        """
        logger.info("Running complete integration test suite...")

        # Test dashboard factory
        self.test_results["dashboard_factory"] = self.test_dashboard_factory()

        # Test dashboard validation
        self.test_results["dashboard_validation"] = self.test_dashboard_validation()

        # Test configuration validation
        self.test_results["config_validation"] = self.test_config_validation()

        # Test dashboard JSON files
        self.test_results["dashboard_files"] = self.test_dashboard_files()

        return self.test_results

    def test_dashboard_factory(self) -> TestResult:
        """
        Test dashboard factory functionality.
        """
        logger.info("Testing dashboard factory...")
        result: TestResult = {"success": True, "errors": [], "warnings": [], "info": []}

        try:
            # Test panel factory
            panel_factory = GrafanaPanelFactory()

            # Test stat panel creation
            stat_panel = panel_factory.create_stat_panel(
                title="Test Stat Panel",
                expr="test_metric",
                panel_id=1,
                grid_pos={"h": 4, "w": 6, "x": 0, "y": 0},
                unit="short",
            )

            if not isinstance(stat_panel, dict) or stat_panel.get("type") != "stat":
                result["errors"].append("Stat panel creation failed")
                result["success"] = False
            else:
                result["info"].append("Stat panel created successfully")

            # Test timeseries panel creation
            targets = [
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "expr": "test_metric",
                    "legendFormat": "Test Metric",
                    "refId": "A",
                },
            ]

            timeseries_panel = panel_factory.create_timeseries_panel(
                title="Test Timeseries Panel",
                targets=targets,
                panel_id=2,
                grid_pos={"h": 8, "w": 12, "x": 0, "y": 4},
                unit="short",
            )

            if (
                not isinstance(timeseries_panel, dict)
                or timeseries_panel.get("type") != "timeseries"
            ):
                result["errors"].append("Timeseries panel creation failed")
                result["success"] = False
            else:
                result["info"].append("Timeseries panel created successfully")

            # Test dashboard factory
            dashboard_factory = GrafanaDashboardFactory()

            dashboard = dashboard_factory.create_base_dashboard(
                title="Test Dashboard",
                uid="test-dashboard",
                tags=["test", "ml-monitoring"],
            )

            if not isinstance(dashboard, dict) or dashboard.get("title") != "Test Dashboard":
                result["errors"].append("Dashboard creation failed")
                result["success"] = False
            else:
                result["info"].append("Dashboard created successfully")

            # Test alert configuration
            alert_config = dashboard_factory.create_alert_config(
                alert_name="Test Alert",
                condition_value=0.8,
                condition_type="gt",
                duration="5m",
            )

            if not isinstance(alert_config, dict) or not alert_config.get("conditions"):
                result["errors"].append("Alert configuration creation failed")
                result["success"] = False
            else:
                result["info"].append("Alert configuration created successfully")

        except Exception as e:
            result["errors"].append(f"Dashboard factory test failed: {e}")
            result["success"] = False

        return result

    def test_dashboard_validation(self) -> TestResult:
        """
        Test dashboard validation functionality.
        """
        logger.info("Testing dashboard validation...")
        result: TestResult = {"success": True, "errors": [], "warnings": [], "info": []}

        try:
            # Create a test dashboard
            factory = GrafanaDashboardFactory()
            test_dashboard = factory.create_base_dashboard(
                title="Validation Test Dashboard",
                uid="validation-test",
                tags=["test", "ml-monitoring"],
            )

            # Add a test panel
            panel_factory = GrafanaPanelFactory()
            test_panel = panel_factory.create_stat_panel(
                title="Test Panel",
                expr="ml_test_metric",
                panel_id=1,
                grid_pos={"h": 4, "w": 6, "x": 0, "y": 0},
            )
            test_dashboard["panels"].append(test_panel)

            # Save test dashboard to temp file
            test_file = Path("/tmp/test_dashboard.json")
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(test_dashboard, f, indent=2)

            self.temp_files.append(test_file)

            # Validate the test dashboard
            is_valid, errors, warnings = validate_file(test_file)

            if not is_valid:
                result["errors"].extend([f"Validation error: {e}" for e in errors])
                result["success"] = False
            else:
                result["info"].append("Dashboard validation passed")

            if warnings:
                result["warnings"].extend([f"Validation warning: {w}" for w in warnings])

        except Exception as e:
            result["errors"].append(f"Dashboard validation test failed: {e}")
            result["success"] = False

        return result

    def test_config_validation(self) -> TestResult:
        """
        Test configuration validation.
        """
        logger.info("Testing configuration validation...")
        result: TestResult = {"success": True, "errors": [], "warnings": [], "info": []}

        try:
            # Create a minimal test config
            test_config = {
                "GRAFANA_URL": "http://localhost:3000",
                "GRAFANA_API_TOKEN": "test-token-1234567890",
                "LOG_LEVEL": "INFO",
                "DEFAULT_DASHBOARD_REFRESH": "30s",
            }

            # Create test .env file
            test_env_file = Path("/tmp/test.env")
            with open(test_env_file, "w", encoding="utf-8") as f:
                for key, value in test_config.items():
                    f.write(f"{key}={value}\n")

            self.temp_files.append(test_env_file)

            # Validate configuration
            validator = ConfigValidator(env_file=test_env_file, check_connectivity=False)
            is_valid, errors, warnings, info = validator.validate()

            if not is_valid:
                result["errors"].extend([f"Config error: {e}" for e in errors])
                result["success"] = False
            else:
                result["info"].append("Configuration validation passed")

            if warnings:
                result["warnings"].extend([f"Config warning: {w}" for w in warnings])

        except Exception as e:
            result["errors"].append(f"Configuration validation test failed: {e}")
            result["success"] = False

        return result

    def test_dashboard_files(self) -> TestResult:
        """
        Test existing dashboard JSON files.
        """
        logger.info("Testing dashboard JSON files...")
        result: TestResult = {"success": True, "errors": [], "warnings": [], "info": []}

        try:
            # Find dashboard directory
            dashboard_dir = Path(__file__).parent.parent / "grafana" / "dashboards"

            if not dashboard_dir.exists():
                result["errors"].append(f"Dashboard directory not found: {dashboard_dir}")
                result["success"] = False
                return result

            # Validate all dashboard files
            validation_results = validate_directory(dashboard_dir)

            total_files = len(validation_results)
            valid_files = sum(1 for is_valid, _, _ in validation_results.values() if is_valid)

            result["info"].append(f"Validated {total_files} dashboard files")
            result["info"].append(f"Valid files: {valid_files}/{total_files}")

            for filename, (is_valid, errors, warnings) in validation_results.items():
                if not is_valid:
                    result["errors"].extend([f"{filename}: {e}" for e in errors])
                    result["success"] = False

                if warnings:
                    result["warnings"].extend([f"{filename}: {w}" for w in warnings])

            if valid_files == total_files:
                result["info"].append("All dashboard files are valid")

        except Exception as e:
            result["errors"].append(f"Dashboard files test failed: {e}")
            result["success"] = False

        return result

    def test_live_grafana(self, grafana_url: str, api_token: str) -> TestResult:
        """
        Test connection to live Grafana instance.
        """
        logger.info("Testing live Grafana connection...")
        result: TestResult = {"success": True, "errors": [], "warnings": [], "info": []}

        try:
            # Create Grafana client
            client = GrafanaClient(
                base_url=grafana_url,
                api_token=api_token,
                timeout=10,
            )

            # Test health check
            if not client.health_check():
                result["errors"].append("Grafana health check failed")
                result["success"] = False
                return result

            result["info"].append("Grafana health check passed")

            # Test dashboard search
            try:
                dashboards = client.search_dashboards(tag="ml-monitoring", limit=10)
                result["info"].append(f"Found {len(dashboards)} ML monitoring dashboards")
            except GrafanaAPIError as e:
                result["warnings"].append(f"Dashboard search failed: {e}")

            # Test folder listing
            try:
                folders = client.get_folders()
                result["info"].append(f"Found {len(folders)} folders")
            except GrafanaAPIError as e:
                result["warnings"].append(f"Folder listing failed: {e}")

            # Test data source listing
            try:
                datasources = client.get_datasources()
                result["info"].append(f"Found {len(datasources)} data sources")

                # Check for Prometheus
                prometheus_sources = [ds for ds in datasources if ds.get("type") == "prometheus"]
                if prometheus_sources:
                    result["info"].append(
                        f"Found {len(prometheus_sources)} Prometheus data sources",
                    )
                else:
                    result["warnings"].append("No Prometheus data sources found")

            except GrafanaAPIError as e:
                result["warnings"].append(f"Data source listing failed: {e}")

        except Exception as e:
            result["errors"].append(f"Live Grafana test failed: {e}")
            result["success"] = False

        return result

    def test_dashboard_performance(self, dashboard_file: Path) -> TestResult:
        """
        Test dashboard rendering performance.
        """
        logger.info(f"Testing dashboard performance: {dashboard_file}")
        result: TestResult = {"success": True, "errors": [], "warnings": [], "info": []}

        try:
            # Load and parse dashboard
            start_time = time.time()

            with open(dashboard_file, encoding="utf-8") as f:
                dashboard_data = json.load(f)

            parse_time = time.time() - start_time

            # Check dashboard size
            file_size = dashboard_file.stat().st_size
            if file_size > 1024 * 1024:  # 1MB
                result["warnings"].append(f"Dashboard file is large: {file_size / 1024:.1f} KB")

            # Check number of panels
            panels = dashboard_data.get("panels", [])
            panel_count = len(panels)

            if panel_count > 50:
                result["warnings"].append(f"Dashboard has many panels: {panel_count}")
            elif panel_count > 30:
                result["info"].append(f"Dashboard has {panel_count} panels")

            # Check query complexity
            total_queries = 0
            complex_queries = 0

            for panel in panels:
                targets = panel.get("targets", [])
                total_queries += len(targets)

                for target in targets:
                    expr = target.get("expr", "")
                    if expr.count("(") > 5 or len(expr) > 200:
                        complex_queries += 1

            if complex_queries > 0:
                result["warnings"].append(f"Found {complex_queries} complex queries")

            # Performance metrics
            result["info"].append(f"Parse time: {parse_time * 1000:.1f}ms")
            result["info"].append(f"File size: {file_size / 1024:.1f} KB")
            result["info"].append(f"Panel count: {panel_count}")
            result["info"].append(f"Query count: {total_queries}")

            # Performance thresholds
            if parse_time > 0.1:
                result["warnings"].append(f"Slow parse time: {parse_time * 1000:.1f}ms")

        except Exception as e:
            result["errors"].append(f"Performance test failed: {e}")
            result["success"] = False

        return result

    def cleanup(self) -> None:
        """
        Clean up temporary files.
        """
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except OSError:
                pass

    def print_results(self, results: dict[str, TestResult]) -> None:
        """
        Print test results in a formatted way.
        """
        logger.info("\n" + "=" * 60)
        logger.info("INTEGRATION TEST RESULTS")
        logger.info("=" * 60)

        overall_success = True
        total_errors = 0
        total_warnings = 0

        for test_name, result in results.items():
            success = result.get("success", False)
            errors = result.get("errors", [])
            warnings = result.get("warnings", [])
            info = result.get("info", [])

            overall_success &= success
            total_errors += len(errors)
            total_warnings += len(warnings)

            status = "PASS" if success else "FAIL"
            logger.info(f"\n{test_name.upper()}: {status}")

            if errors:
                logger.info("  ERRORS:")
                for error in errors:
                    logger.info(f"    • {error}")

            if warnings:
                logger.info("  WARNINGS:")
                for warning in warnings:
                    logger.info(f"    • {warning}")

            if info:
                logger.info("  INFO:")
                for info_item in info:
                    logger.info(f"    • {info_item}")

        # Overall summary
        logger.info("\n" + "=" * 60)
        overall_status = " ALL TESTS PASSED" if overall_success else " SOME TESTS FAILED"
        logger.info(f"OVERALL: {overall_status}")
        logger.info(f"Total Errors: {total_errors}, Total Warnings: {total_warnings}")
        logger.info("=" * 60)


def main() -> int:
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Integration test for Grafana dashboard system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all integration tests",
    )
    parser.add_argument(
        "--test-factory",
        action="store_true",
        help="Test dashboard factory",
    )
    parser.add_argument(
        "--test-validation",
        action="store_true",
        help="Test dashboard validation",
    )
    parser.add_argument(
        "--test-config",
        action="store_true",
        help="Test configuration validation",
    )
    parser.add_argument(
        "--test-files",
        action="store_true",
        help="Test dashboard JSON files",
    )
    parser.add_argument(
        "--test-live",
        action="store_true",
        help="Test live Grafana connection",
    )
    parser.add_argument(
        "--test-performance",
        action="store_true",
        help="Test dashboard performance",
    )
    parser.add_argument(
        "--grafana-url",
        default="http://localhost:3000",
        help="Grafana server URL for live testing",
    )
    parser.add_argument(
        "--api-token",
        help="Grafana API token for live testing",
    )
    parser.add_argument(
        "--dashboard-file",
        type=Path,
        help="Specific dashboard file for performance testing",
    )

    args = parser.parse_args()

    # Create tester
    tester = IntegrationTester()

    try:
        results = {}

        if args.all or args.test_factory:
            results["dashboard_factory"] = tester.test_dashboard_factory()

        if args.all or args.test_validation:
            results["dashboard_validation"] = tester.test_dashboard_validation()

        if args.all or args.test_config:
            results["config_validation"] = tester.test_config_validation()

        if args.all or args.test_files:
            results["dashboard_files"] = tester.test_dashboard_files()

        if args.test_live:
            if not args.api_token:
                logger.error("--api-token required for live testing")
                return 1
            results["live_grafana"] = tester.test_live_grafana(args.grafana_url, args.api_token)

        if args.test_performance:
            if not args.dashboard_file:
                # Test all dashboard files
                dashboard_dir = Path(__file__).parent.parent / "grafana" / "dashboards"
                for dashboard_file in dashboard_dir.glob("*.json"):
                    test_name = f"performance_{dashboard_file.stem}"
                    results[test_name] = tester.test_dashboard_performance(dashboard_file)
            else:
                results["dashboard_performance"] = tester.test_dashboard_performance(
                    args.dashboard_file,
                )

        if not results:
            logger.error("No tests specified. Use --all or specific test flags.")
            return 1

        # Print results
        tester.print_results(results)

        # Return success/failure
        overall_success = all(result.get("success", False) for result in results.values())
        return 0 if overall_success else 1

    except KeyboardInterrupt:
        logger.info("Testing cancelled by user")
        return 1

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return 1

    finally:
        # Cleanup
        tester.cleanup()


if __name__ == "__main__":
    sys.exit(main())
