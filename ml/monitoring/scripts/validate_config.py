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
Configuration validation for ML monitoring dashboard integration.

This script validates environment configuration and system prerequisites
for the Grafana dashboard integration system.

Usage:
    python validate_config.py [options]

Example:
    # Validate default configuration
    python validate_config.py

    # Validate specific .env file
    python validate_config.py --env-file /path/to/custom.env

    # Check system connectivity
    python validate_config.py --check-connectivity

    # Detailed validation report
    python validate_config.py --detailed

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validator for ML monitoring configuration.
    """

    def __init__(self, env_file: Path | None = None, check_connectivity: bool = False) -> None:
        """
        Initialize config validator.

        Parameters
        ----------
        env_file : Path, optional
            Path to .env file
        check_connectivity : bool, optional
            Whether to test network connectivity

        """
        self.env_file = env_file
        self.check_connectivity = check_connectivity
        self.config: dict[str, str] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def load_config(self) -> bool:
        """
        Load configuration from environment and .env file.

        Returns
        -------
        bool
            True if config loaded successfully

        """
        try:
            # Load from .env file if specified
            if self.env_file and self.env_file.exists():
                self._load_env_file(self.env_file)
                self.info.append(f"Loaded configuration from: {self.env_file}")
            elif self.env_file:
                self.warnings.append(f".env file not found: {self.env_file}")
            else:
                # Look for .env in current directory
                default_env = Path(".env")
                if default_env.exists():
                    self._load_env_file(default_env)
                    self.info.append(f"Loaded configuration from: {default_env}")

            # Load from environment variables (overrides .env)
            for key, value in os.environ.items():
                self.config[key] = value

            return True

        except Exception as e:
            self.errors.append(f"Failed to load configuration: {e}")
            return False

    def _load_env_file(self, env_file: Path) -> None:
        """
        Load environment variables from .env file.
        """
        with open(env_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse key=value
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    self.config[key] = value
                else:
                    self.warnings.append(f"Invalid line {line_num} in {env_file}: {line}")

    def validate(self) -> tuple[bool, list[str], list[str], list[str]]:
        """
        Validate complete configuration.

        Returns
        -------
        tuple[bool, list[str], list[str], list[str]]
            Tuple of (is_valid, errors, warnings, info)

        """
        self.errors = []
        self.warnings = []
        self.info = []

        # Load configuration
        if not self.load_config():
            return False, self.errors, self.warnings, self.info

        # Validate different configuration sections
        self._validate_grafana_config()
        self._validate_prometheus_config()
        self._validate_alertmanager_config()
        self._validate_dashboard_config()
        self._validate_logging_config()
        self._validate_paths_and_directories()
        self._validate_performance_settings()
        self._validate_security_settings()

        # Test connectivity if requested
        if self.check_connectivity:
            self._test_connectivity()

        return len(self.errors) == 0, self.errors, self.warnings, self.info

    def _validate_grafana_config(self) -> None:
        """
        Validate Grafana configuration.
        """
        # Check Grafana URL
        grafana_url = self.config.get("GRAFANA_URL", "")
        if not grafana_url:
            self.errors.append("GRAFANA_URL is required")
        elif not self._validate_url(grafana_url):
            self.errors.append(f"Invalid GRAFANA_URL: {grafana_url}")
        else:
            self.info.append(f"Grafana URL: {grafana_url}")

        # Check authentication
        api_token = self.config.get("GRAFANA_API_TOKEN")
        username = self.config.get("GRAFANA_USERNAME")
        password = self.config.get("GRAFANA_PASSWORD")

        if not api_token and not (username and password):
            self.errors.append(
                "Must provide either GRAFANA_API_TOKEN or GRAFANA_USERNAME/GRAFANA_PASSWORD",
            )
        elif api_token:
            if len(api_token) < 10:
                self.warnings.append("GRAFANA_API_TOKEN seems too short")
            self.info.append("Using API token authentication")
        else:
            self.info.append("Using username/password authentication")

        # Check timeout
        timeout = self.config.get("GRAFANA_TIMEOUT", "30")
        if not timeout.isdigit() or int(timeout) <= 0:
            self.warnings.append(f"Invalid GRAFANA_TIMEOUT: {timeout}")

        # Check SSL verification
        verify_ssl = self.config.get("GRAFANA_VERIFY_SSL", "true").lower()
        if verify_ssl not in ["true", "false"]:
            self.warnings.append(f"Invalid GRAFANA_VERIFY_SSL: {verify_ssl}")

    def _validate_prometheus_config(self) -> None:
        """
        Validate Prometheus configuration.
        """
        prometheus_url = self.config.get("PROMETHEUS_URL", "")
        if not prometheus_url:
            self.warnings.append("PROMETHEUS_URL not set (required for some features)")
        elif not self._validate_url(prometheus_url):
            self.errors.append(f"Invalid PROMETHEUS_URL: {prometheus_url}")
        else:
            self.info.append(f"Prometheus URL: {prometheus_url}")

        # Check timeout
        timeout = self.config.get("PROMETHEUS_TIMEOUT", "30")
        if not timeout.isdigit() or int(timeout) <= 0:
            self.warnings.append(f"Invalid PROMETHEUS_TIMEOUT: {timeout}")

    def _validate_alertmanager_config(self) -> None:
        """
        Validate Alertmanager configuration.
        """
        alertmanager_url = self.config.get("ALERTMANAGER_URL", "")
        if alertmanager_url and not self._validate_url(alertmanager_url):
            self.errors.append(f"Invalid ALERTMANAGER_URL: {alertmanager_url}")
        elif alertmanager_url:
            self.info.append(f"Alertmanager URL: {alertmanager_url}")

    def _validate_dashboard_config(self) -> None:
        """
        Validate dashboard management configuration.
        """
        # Check folder name
        folder = self.config.get("ML_DASHBOARD_FOLDER", "")
        if not folder:
            self.warnings.append("ML_DASHBOARD_FOLDER not set, using default folder")

        # Check refresh rate
        refresh = self.config.get("DEFAULT_DASHBOARD_REFRESH", "30s")
        if not self._validate_refresh_rate(refresh):
            self.warnings.append(f"Invalid DEFAULT_DASHBOARD_REFRESH: {refresh}")

        # Check time range
        time_from = self.config.get("DEFAULT_TIME_FROM", "now-6h")
        time_to = self.config.get("DEFAULT_TIME_TO", "now")
        if not self._validate_time_range(time_from, time_to):
            self.warnings.append(f"Invalid time range: {time_from} to {time_to}")

        # Check boolean settings
        bool_settings = [
            "AUTO_CREATE_FOLDER",
            "BACKUP_DASHBOARDS",
            "STRICT_VALIDATION",
            "DETAILED_VALIDATION",
        ]

        for setting in bool_settings:
            value = self.config.get(setting, "").lower()
            if value and value not in ["true", "false"]:
                self.warnings.append(f"Invalid boolean value for {setting}: {value}")

    def _validate_logging_config(self) -> None:
        """
        Validate logging configuration.
        """
        # Check log level
        log_level = self.config.get("LOG_LEVEL", "INFO").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_levels:
            self.warnings.append(f"Invalid LOG_LEVEL: {log_level}")

        # Check log format
        log_format = self.config.get("LOG_FORMAT", "text").lower()
        if log_format not in ["json", "text"]:
            self.warnings.append(f"Invalid LOG_FORMAT: {log_format}")

        # Check log file settings
        log_file = self.config.get("LOG_FILE", "")
        if log_file:
            log_dir = Path(log_file).parent
            if not log_dir.exists():
                self.warnings.append(f"Log directory does not exist: {log_dir}")

        # Check numeric settings
        numeric_settings = [
            ("LOG_MAX_SIZE", "10"),
            ("LOG_BACKUP_COUNT", "5"),
        ]

        for setting, default in numeric_settings:
            value = self.config.get(setting, default)
            if not value.isdigit() or int(value) < 0:
                self.warnings.append(f"Invalid {setting}: {value}")

    def _validate_paths_and_directories(self) -> None:
        """
        Validate file paths and directories.
        """
        # Check backup directory
        backup_dir = self.config.get("BACKUP_DIR", "./backups")
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            self.warnings.append(f"Backup directory does not exist: {backup_path}")

        # Check test data directory
        test_data_dir = self.config.get("TEST_DATA_DIR", "./test_data")
        test_path = Path(test_data_dir)
        if not test_path.exists():
            self.info.append(f"Test data directory does not exist: {test_path}")

        # Check custom template directory
        template_dir = self.config.get("CUSTOM_TEMPLATE_DIR", "")
        if template_dir:
            template_path = Path(template_dir)
            if not template_path.exists():
                self.warnings.append(f"Custom template directory does not exist: {template_path}")

    def _validate_performance_settings(self) -> None:
        """
        Validate performance-related settings.
        """
        numeric_settings = [
            ("MAX_CONCURRENT_OPERATIONS", "5", 1, 50),
            ("MAX_RETRIES", "3", 0, 10),
            ("RATE_LIMIT", "10", 0, 1000),
            ("MAX_DASHBOARD_SIZE", "1024", 1, 10240),
            ("MAX_PANELS_PER_DASHBOARD", "50", 1, 200),
            ("HEALTH_CHECK_INTERVAL", "300", 30, 3600),
        ]

        for setting, default, min_val, max_val in numeric_settings:
            value = self.config.get(setting, default)
            if not value.isdigit():
                self.warnings.append(f"Invalid {setting}: {value} (must be numeric)")
                continue

            int_value = int(value)
            if int_value < min_val or int_value > max_val:
                self.warnings.append(
                    f"{setting} out of range: {int_value} (valid: {min_val}-{max_val})",
                )

        # Check backoff factor
        backoff = self.config.get("RETRY_BACKOFF_FACTOR", "0.3")
        try:
            float_value = float(backoff)
            if float_value < 0 or float_value > 2.0:
                self.warnings.append(
                    f"RETRY_BACKOFF_FACTOR out of range: {float_value} (valid: 0.0-2.0)",
                )
        except ValueError:
            self.warnings.append(f"Invalid RETRY_BACKOFF_FACTOR: {backoff}")

    def _validate_security_settings(self) -> None:
        """
        Validate security settings.
        """
        # Check allowed hosts
        allowed_hosts = self.config.get("ALLOWED_GRAFANA_HOSTS", "")
        if allowed_hosts:
            hosts = [host.strip() for host in allowed_hosts.split(",")]
            for host in hosts:
                if not host or not self._validate_hostname(host):
                    self.warnings.append(f"Invalid hostname in ALLOWED_GRAFANA_HOSTS: {host}")

        # Check encryption key
        encryption_key = self.config.get("API_TOKEN_ENCRYPTION_KEY", "")
        if encryption_key and len(encryption_key) < 32:
            self.warnings.append("API_TOKEN_ENCRYPTION_KEY should be at least 32 characters")

    def _test_connectivity(self) -> None:
        """
        Test connectivity to configured services.
        """
        services = [
            ("Grafana", self.config.get("GRAFANA_URL")),
            ("Prometheus", self.config.get("PROMETHEUS_URL")),
            ("Alertmanager", self.config.get("ALERTMANAGER_URL")),
        ]

        for service_name, url in services:
            if not url:
                continue

            try:
                # Test basic connectivity with timeout
                timeout = int(self.config.get(f"{service_name.upper()}_TIMEOUT", "30"))
                response = requests.get(f"{url}/api/health", timeout=timeout, verify=False)

                if response.status_code == 200:
                    self.info.append(f"✓ {service_name} connectivity OK ({url})")
                else:
                    self.warnings.append(
                        f"⚠ {service_name} responded with status {response.status_code}",
                    )

            except requests.RequestException as e:
                self.warnings.append(f"✗ {service_name} connectivity failed: {e}")

            except ValueError as e:
                self.errors.append(f"Invalid timeout for {service_name}: {e}")

    def _validate_url(self, url: str) -> bool:
        """
        Validate URL format.
        """
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def _validate_hostname(self, hostname: str) -> bool:
        """
        Validate hostname format.
        """
        if not hostname:
            return False

        # Simple hostname validation
        if hostname in ["localhost", "127.0.0.1", "::1"]:
            return True

        # Basic domain name validation
        parts = hostname.split(".")
        if len(parts) < 2:
            return False

        for part in parts:
            if not part or not part.replace("-", "").isalnum():
                return False

        return True

    def _validate_refresh_rate(self, refresh: str) -> bool:
        """
        Validate dashboard refresh rate.
        """
        valid_rates = [
            "5s",
            "10s",
            "30s",
            "1m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "1d",
        ]
        return refresh in valid_rates

    def _validate_time_range(self, time_from: str, time_to: str) -> bool:
        """
        Validate time range format.
        """
        # Basic validation for Grafana time format
        valid_patterns = ["now", "now-", "YYYY-MM-DD"]

        def check_time(time_str: str) -> bool:
            if time_str == "now":
                return True
            if time_str.startswith("now-"):
                return True
            return len(time_str) >= 10  # Basic length check

        return check_time(time_from) and check_time(time_to)


def print_validation_results(
    is_valid: bool,
    errors: list[str],
    warnings: list[str],
    info: list[str],
    detailed: bool = False,
) -> None:
    """
    Print validation results in a formatted way.
    """
    # Print summary
    status = "✓ VALID" if is_valid else "✗ INVALID"
    print(f"\nConfiguration Status: {status}")
    print(f"Errors: {len(errors)}, Warnings: {len(warnings)}, Info: {len(info)}")

    # Print errors
    if errors:
        print("\n🔴 ERRORS:")
        for error in errors:
            print(f"  • {error}")

    # Print warnings
    if warnings and (detailed or not errors):
        print("\n🟡 WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")

    # Print info in detailed mode
    if info and detailed:
        print("\n🔵 INFORMATION:")
        for info_item in info:
            print(f"  • {info_item}")


def main() -> int:
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Validate ML monitoring configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        help="Path to .env file (default: .env in current directory)",
    )
    parser.add_argument(
        "--check-connectivity",
        action="store_true",
        help="Test network connectivity to configured services",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed validation output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show errors",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    args = parser.parse_args()

    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    try:
        # Create validator
        validator = ConfigValidator(
            env_file=args.env_file,
            check_connectivity=args.check_connectivity,
        )

        # Validate configuration
        is_valid, errors, warnings, info = validator.validate()

        # Output results
        if args.output_format == "json":
            import json

            result = {
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "info": info,
            }
            print(json.dumps(result, indent=2))
        else:
            print_validation_results(is_valid, errors, warnings, info, args.detailed)

        # Return appropriate exit code
        return 0 if is_valid else 1

    except KeyboardInterrupt:
        logger.info("Validation cancelled by user")
        return 1

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
