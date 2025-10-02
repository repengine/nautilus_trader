"""
Terminal & Settings Service for ML Dashboard.

Provides secure command execution and configuration management with sandboxing,
validation, and Prometheus metrics integration.

Key Features:
- Whitelisted ML command execution
- Command history tracking
- Hierarchical configuration management
- Security validation and sandboxing
- Prometheus metrics for all operations

Performance Targets: Cold path only (no hot path requirements)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram
from ml.dashboard.services.base_service import BaseIntegrationService


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager


logger = logging.getLogger(__name__)


# ============================================================================
# METRICS
# ============================================================================

terminal_commands_total = get_counter(
    "ml_terminal_commands_total",
    "Total terminal commands executed",
    labelnames=["command_type", "status"],
)

terminal_command_duration = get_histogram(
    "ml_terminal_command_duration_seconds",
    "Terminal command execution duration",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    labelnames=["command_type"],
)

settings_updates_total = get_counter(
    "ml_settings_updates_total",
    "Total settings update operations",
    labelnames=["section", "status"],
)

settings_validation_errors_total = get_counter(
    "ml_settings_validation_errors_total",
    "Total settings validation errors",
    labelnames=["section", "error_type"],
)

terminal_history_size = get_gauge(
    "ml_terminal_history_size",
    "Current terminal command history size",
)


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_HISTORY_SIZE: Final[int] = 1000
MAX_COMMAND_LENGTH: Final[int] = 10000
COMMAND_TIMEOUT_SECONDS: Final[int] = 300


# ============================================================================
# ENUMS
# ============================================================================


class CommandType(str, Enum):
    """Allowed command types for terminal execution."""

    ML_DATA = "ml.data"
    ML_MODEL = "ml.model"
    ML_FEATURE = "ml.feature"
    ML_PIPELINE = "ml.pipeline"
    ML_REGISTRY = "ml.registry"
    SYSTEM_STATUS = "system.status"
    SYSTEM_LOGS = "system.logs"
    SYSTEM_CONFIG = "system.config"


class ValidationErrorType(str, Enum):
    """Settings validation error types."""

    SCHEMA_VIOLATION = "schema_violation"
    TYPE_MISMATCH = "type_mismatch"
    RANGE_VIOLATION = "range_violation"
    SECURITY_POLICY = "security_policy"
    BUSINESS_LOGIC = "business_logic"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass(slots=True, frozen=True)
class CommandResult:
    """Result of a terminal command execution."""

    command: str
    output: str
    exit_code: int
    duration_seconds: float
    timestamp: int  # nanoseconds
    command_type: str
    success: bool
    error: str | None = None


@dataclass(slots=True)
class CommandHistoryEntry:
    """Entry in command history."""

    command: str
    timestamp: int  # nanoseconds
    command_type: str
    success: bool
    duration_seconds: float
    output_preview: str  # First 200 chars
    error: str | None = None


@dataclass(slots=True)
class SettingsValidationResult:
    """Result of settings validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ============================================================================
# CONFIGURATION SCHEMAS
# ============================================================================

CONFIG_SCHEMA: Final[dict[str, Any]] = {
    "system": {
        "dashboard": {
            "type": "object",
            "properties": {
                "theme": {"type": "string", "enum": ["light", "dark", "auto"]},
                "refresh_interval": {"type": "integer", "minimum": 1, "maximum": 300},
                "max_concurrent_sessions": {"type": "integer", "minimum": 1, "maximum": 10},
                "enable_websockets": {"type": "boolean"},
                "terminal_history_size": {"type": "integer", "minimum": 100, "maximum": 10000},
            },
        },
        "security": {
            "type": "object",
            "properties": {
                "session_timeout_minutes": {"type": "integer", "minimum": 5, "maximum": 1440},
                "require_token_for_terminal": {"type": "boolean"},
                "max_command_execution_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3600,
                },
            },
        },
    },
    "ml": {
        "models": {
            "type": "object",
            "properties": {
                "max_inference_latency_ms": {"type": "number", "minimum": 1.0, "maximum": 100.0},
                "prediction_confidence_threshold": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.99,
                },
                "enable_model_hot_reload": {"type": "boolean"},
                "model_check_interval_seconds": {
                    "type": "integer",
                    "minimum": 30,
                    "maximum": 3600,
                },
            },
        },
        "features": {
            "type": "object",
            "properties": {
                "feature_computation_timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60,
                },
                "enable_feature_caching": {"type": "boolean"},
                "cache_ttl_seconds": {"type": "integer", "minimum": 60, "maximum": 86400},
            },
        },
        "data": {
            "type": "object",
            "properties": {
                "ingestion_batch_size": {"type": "integer", "minimum": 100, "maximum": 10000},
                "max_backfill_days": {"type": "integer", "minimum": 1, "maximum": 365},
                "enable_data_validation": {"type": "boolean"},
                "data_retention_days": {"type": "integer", "minimum": 7, "maximum": 3650},
            },
        },
    },
}

DEFAULT_CONFIG: Final[dict[str, Any]] = {
    "system": {
        "dashboard": {
            "theme": "light",
            "refresh_interval": 30,
            "max_concurrent_sessions": 5,
            "enable_websockets": True,
            "terminal_history_size": 1000,
        },
        "security": {
            "session_timeout_minutes": 60,
            "require_token_for_terminal": True,
            "max_command_execution_seconds": 300,
        },
    },
    "ml": {
        "models": {
            "max_inference_latency_ms": 5.0,
            "prediction_confidence_threshold": 0.7,
            "enable_model_hot_reload": False,
            "model_check_interval_seconds": 300,
        },
        "features": {
            "feature_computation_timeout": 30,
            "enable_feature_caching": True,
            "cache_ttl_seconds": 3600,
        },
        "data": {
            "ingestion_batch_size": 1000,
            "max_backfill_days": 30,
            "enable_data_validation": True,
            "data_retention_days": 365,
        },
    },
}


# ============================================================================
# ALLOWED COMMANDS
# ============================================================================

ALLOWED_COMMANDS: Final[dict[str, dict[str, list[str]]]] = {
    "ml": {
        "data": ["ingest", "backfill", "validate", "coverage", "list", "show"],
        "model": [
            "train",
            "deploy",
            "rollback",
            "evaluate",
            "list",
            "show",
            "performance",
        ],
        "feature": [
            "compute",
            "promote",
            "deprecate",
            "validate",
            "list",
            "show",
            "lineage",
        ],
        "pipeline": ["run", "status", "stop", "schedule", "list", "show"],
        "registry": ["list", "show", "search", "health", "watermarks"],
    },
    "system": {
        "status": ["services", "health", "metrics", "stores"],
        "logs": ["tail", "search", "export"],
        "config": ["show", "validate", "reload"],
    },
}


# ============================================================================
# SERVICE IMPLEMENTATION
# ============================================================================


class TerminalService(BaseIntegrationService):
    """
    Terminal & Settings Service for ML Dashboard.

    Provides secure command execution with whitelisting, command history tracking,
    and hierarchical configuration management.
    """

    def __init__(
        self,
        integration_manager: MLIntegrationManager | None,
        history_file: Path | None = None,
        config_file: Path | None = None,
    ) -> None:
        """
        Initialize terminal service.

        Parameters
        ----------
        integration_manager : MLIntegrationManager | None
            Integration manager for accessing ML components
        history_file : Path | None
            Path to command history file (for persistence)
        config_file : Path | None
            Path to settings configuration file

        """
        super().__init__(integration_manager)
        self._history: list[CommandHistoryEntry] = []
        self._history_file = history_file
        self._config_file = config_file
        self._current_config = self._load_or_create_config()
        self._load_history()

    def get_service_name(self) -> str:
        """Return service name for metrics labels."""
        return "terminal"

    # ========================================================================
    # COMMAND EXECUTION
    # ========================================================================

    def execute_command(self, command: str, user_id: str | None = None) -> CommandResult:
        """
        Execute a terminal command with validation and sandboxing.

        Parameters
        ----------
        command : str
            Command to execute
        user_id : str | None
            User ID for tracking

        Returns
        -------
        CommandResult
            Result of command execution

        """
        start_time = time.perf_counter()
        timestamp_ns = time.time_ns()

        # Validate command
        validation_result = self._validate_command(command)
        if not validation_result["valid"]:
            error_msg = validation_result.get("error", "Invalid command")
            terminal_commands_total.labels(command_type="invalid", status="rejected").inc()
            result = CommandResult(
                command=command,
                output="",
                exit_code=1,
                duration_seconds=0.0,
                timestamp=timestamp_ns,
                command_type="invalid",
                success=False,
                error=error_msg,
            )
            # Add invalid command to history
            self._add_to_history(result)
            return result

        command_type = validation_result["command_type"]

        # Execute command in sandbox
        try:
            output, exit_code = self._execute_sandbox(command, command_type)
            success = exit_code == 0
            error = None if success else f"Command failed with exit code {exit_code}"
            status = "success" if success else "failed"
        except Exception as e:
            output = ""
            exit_code = 1
            success = False
            error = f"Execution error: {e!s}"
            status = "error"
            logger.exception("Command execution failed", extra={"command": command})

        duration = time.perf_counter() - start_time

        # Record metrics
        terminal_commands_total.labels(command_type=command_type, status=status).inc()
        terminal_command_duration.labels(command_type=command_type).observe(duration)
        self._track_operation(operation="execute_command", status=status)

        # Create result
        result = CommandResult(
            command=command,
            output=output,
            exit_code=exit_code,
            duration_seconds=duration,
            timestamp=timestamp_ns,
            command_type=command_type,
            success=success,
            error=error,
        )

        # Add to history
        self._add_to_history(result)

        return result

    def _validate_command(self, command: str) -> dict[str, Any]:
        """
        Validate command against whitelist.

        Parameters
        ----------
        command : str
            Command to validate

        Returns
        -------
        dict[str, Any]
            Validation result with 'valid', 'command_type', and optional 'error'

        """
        # Check length
        if len(command) > MAX_COMMAND_LENGTH:
            return {
                "valid": False,
                "error": f"Command exceeds maximum length of {MAX_COMMAND_LENGTH}",
            }

        # Parse command
        parts = command.strip().split()
        if not parts:
            return {"valid": False, "error": "Empty command"}

        # Check against whitelist
        if len(parts) >= 2:
            category = parts[0]
            subcategory = parts[1]

            if category in ALLOWED_COMMANDS:
                if subcategory in ALLOWED_COMMANDS[category]:
                    if len(parts) >= 3:
                        action = parts[2]
                        if action in ALLOWED_COMMANDS[category][subcategory]:
                            command_type = f"{category}.{subcategory}"
                            return {"valid": True, "command_type": command_type}
                        else:
                            return {
                                "valid": False,
                                "error": f"Action '{action}' not allowed for {category} {subcategory}",
                            }
                    else:
                        # Command with just category and subcategory (e.g., "ml data")
                        command_type = f"{category}.{subcategory}"
                        return {"valid": True, "command_type": command_type}

        return {
            "valid": False,
            "error": f"Command not in whitelist. Use allowed commands: {list(ALLOWED_COMMANDS.keys())}",
        }

    def _execute_sandbox(self, command: str, command_type: str) -> tuple[str, int]:
        """
        Execute command in sandboxed environment.

        For this implementation, we return simulated results.
        Real implementation would integrate with actual ML components.

        Parameters
        ----------
        command : str
            Validated command to execute
        command_type : str
            Type of command

        Returns
        -------
        tuple[str, int]
            Output string and exit code

        """
        # Simulated execution for different command types
        parts = command.strip().split()

        if command_type == "ml.data":
            return self._handle_ml_data_command(parts)
        elif command_type == "ml.model":
            return self._handle_ml_model_command(parts)
        elif command_type == "ml.feature":
            return self._handle_ml_feature_command(parts)
        elif command_type == "ml.pipeline":
            return self._handle_ml_pipeline_command(parts)
        elif command_type == "ml.registry":
            return self._handle_ml_registry_command(parts)
        elif command_type == "system.status":
            return self._handle_system_status_command(parts)
        elif command_type == "system.logs":
            return self._handle_system_logs_command(parts)
        elif command_type == "system.config":
            return self._handle_system_config_command(parts)
        else:
            return f"Unknown command type: {command_type}", 1

    def _handle_ml_data_command(self, parts: list[str]) -> tuple[str, int]:
        """Handle ML data commands."""
        if len(parts) < 3:
            return "Usage: ml data <action> [options]", 1

        action = parts[2]
        if action == "list":
            output = "Available datasets:\n  - features_v1\n  - ohlcv_daily\n  - tick_data"
            return output, 0
        elif action == "show":
            if len(parts) >= 4:
                dataset = parts[3]
                output = f"Dataset: {dataset}\nRecords: 1,234,567\nLast Updated: 2025-10-01"
                return output, 0
            return "Usage: ml data show <dataset_id>", 1
        elif action == "coverage":
            output = "Data coverage report:\n  EUR/USD: 99.8%\n  SPY: 99.5%\n  BTC-USD: 98.2%"
            return output, 0
        else:
            return f"Data action '{action}' simulated successfully", 0

    def _handle_ml_model_command(self, parts: list[str]) -> tuple[str, int]:
        """Handle ML model commands."""
        if len(parts) < 3:
            return "Usage: ml model <action> [options]", 1

        action = parts[2]
        if action == "list":
            output = "Registered models:\n  - lgb_v1_2_3 (deployed)\n  - xgb_v2_1_0 (staging)\n  - nn_v1_0_0 (development)"
            return output, 0
        elif action == "show":
            if len(parts) >= 4:
                model_id = parts[3]
                output = f"Model: {model_id}\nAccuracy: 0.85\nLatency: 2.3ms\nStatus: deployed"
                return output, 0
            return "Usage: ml model show <model_id>", 1
        elif action == "performance":
            output = "Model performance:\n  Precision: 0.82\n  Recall: 0.79\n  F1: 0.80"
            return output, 0
        else:
            return f"Model action '{action}' simulated successfully", 0

    def _handle_ml_feature_command(self, parts: list[str]) -> tuple[str, int]:
        """Handle ML feature commands."""
        if len(parts) < 3:
            return "Usage: ml feature <action> [options]", 1

        action = parts[2]
        if action == "list":
            output = "Available features:\n  - core_v1\n  - technical_v2\n  - microstructure_v1"
            return output, 0
        elif action == "show":
            if len(parts) >= 4:
                feature_set = parts[3]
                output = f"Feature Set: {feature_set}\nFeatures: 42\nStage: production"
                return output, 0
            return "Usage: ml feature show <feature_set_id>", 1
        else:
            return f"Feature action '{action}' simulated successfully", 0

    def _handle_ml_pipeline_command(self, parts: list[str]) -> tuple[str, int]:
        """Handle ML pipeline commands."""
        if len(parts) < 3:
            return "Usage: ml pipeline <action> [options]", 1

        action = parts[2]
        if action == "list":
            output = "Active pipelines:\n  - full_pipeline (running)\n  - backfill_job (queued)"
            return output, 0
        elif action == "status":
            output = "Pipeline status: 2 running, 1 queued, 5 completed"
            return output, 0
        else:
            return f"Pipeline action '{action}' simulated successfully", 0

    def _handle_ml_registry_command(self, parts: list[str]) -> tuple[str, int]:
        """Handle ML registry commands."""
        if len(parts) < 3:
            return "Usage: ml registry <action> [options]", 1

        action = parts[2]
        if action == "list":
            output = "Registry summary:\n  Models: 12\n  Features: 8\n  Datasets: 15\n  Strategies: 5"
            return output, 0
        elif action == "health":
            output = "Registry health: OK\n  Feature Registry: healthy\n  Model Registry: healthy\n  Data Registry: healthy"
            return output, 0
        else:
            return f"Registry action '{action}' simulated successfully", 0

    def _handle_system_status_command(self, parts: list[str]) -> tuple[str, int]:
        """Handle system status commands."""
        if len(parts) < 3:
            return "Usage: system status <action>", 1

        action = parts[2]
        if action == "services":
            output = "System services:\n  PostgreSQL: running\n  Redis: running\n  Message Bus: running"
            return output, 0
        elif action == "health":
            output = "System health: OK\n  CPU: 45%\n  Memory: 60%\n  Disk: 72%"
            return output, 0
        elif action == "metrics":
            output = "Metrics summary:\n  Requests: 1,234\n  Errors: 5\n  Avg Latency: 23ms"
            return output, 0
        else:
            return f"Status action '{action}' simulated successfully", 0

    def _handle_system_logs_command(self, parts: list[str]) -> tuple[str, int]:
        """Handle system logs commands."""
        if len(parts) < 3:
            return "Usage: system logs <action> [options]", 1

        action = parts[2]
        if action == "tail":
            output = "[2025-10-01 10:15:23] INFO: Pipeline completed\n[2025-10-01 10:15:24] DEBUG: Feature cache hit"
            return output, 0
        else:
            return f"Logs action '{action}' simulated successfully", 0

    def _handle_system_config_command(self, parts: list[str]) -> tuple[str, int]:
        """Handle system config commands."""
        if len(parts) < 3:
            return "Usage: system config <action>", 1

        action = parts[2]
        if action == "show":
            output = json.dumps(self._current_config, indent=2)
            return output, 0
        elif action == "validate":
            validation = self._validate_config_recursive(self._current_config, CONFIG_SCHEMA)
            if validation.valid:
                return "Configuration is valid", 0
            else:
                return "Configuration errors:\n" + "\n".join(validation.errors), 1
        else:
            return f"Config action '{action}' simulated successfully", 0

    # ========================================================================
    # COMMAND HISTORY
    # ========================================================================

    def get_command_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get command history.

        Parameters
        ----------
        limit : int | None
            Maximum number of entries to return

        Returns
        -------
        list[dict[str, Any]]
            Command history entries

        """
        entries = self._history[-limit:] if limit else self._history
        return [
            {
                "command": entry.command,
                "timestamp": entry.timestamp,
                "timestamp_iso": datetime.fromtimestamp(
                    entry.timestamp / 1e9,
                    tz=UTC,
                ).isoformat(),
                "command_type": entry.command_type,
                "success": entry.success,
                "duration_seconds": entry.duration_seconds,
                "output_preview": entry.output_preview,
                "error": entry.error,
            }
            for entry in entries
        ]

    def _add_to_history(self, result: CommandResult) -> None:
        """Add command result to history."""
        entry = CommandHistoryEntry(
            command=result.command,
            timestamp=result.timestamp,
            command_type=result.command_type,
            success=result.success,
            duration_seconds=result.duration_seconds,
            output_preview=result.output[:200] if result.output else "",
            error=result.error,
        )

        self._history.append(entry)

        # Trim history if needed
        if len(self._history) > MAX_HISTORY_SIZE:
            self._history = self._history[-MAX_HISTORY_SIZE:]

        # Update metrics
        terminal_history_size.set(len(self._history))

        # Persist if file is configured
        if self._history_file:
            self._save_history()

    def _load_history(self) -> None:
        """Load command history from file."""
        if not self._history_file or not self._history_file.exists():
            return

        try:
            with self._history_file.open("r") as f:
                data = json.load(f)
                self._history = [
                    CommandHistoryEntry(
                        command=entry["command"],
                        timestamp=entry["timestamp"],
                        command_type=entry["command_type"],
                        success=entry["success"],
                        duration_seconds=entry["duration_seconds"],
                        output_preview=entry["output_preview"],
                        error=entry.get("error"),
                    )
                    for entry in data
                ]
                terminal_history_size.set(len(self._history))
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")

    def _save_history(self) -> None:
        """Save command history to file."""
        if not self._history_file:
            return

        try:
            # Ensure directory exists
            self._history_file.parent.mkdir(parents=True, exist_ok=True)

            with self._history_file.open("w") as f:
                data = [
                    {
                        "command": entry.command,
                        "timestamp": entry.timestamp,
                        "command_type": entry.command_type,
                        "success": entry.success,
                        "duration_seconds": entry.duration_seconds,
                        "output_preview": entry.output_preview,
                        "error": entry.error,
                    }
                    for entry in self._history
                ]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    # ========================================================================
    # SETTINGS MANAGEMENT
    # ========================================================================

    def get_settings(self, section: str | None = None) -> dict[str, Any]:
        """
        Get current settings.

        Parameters
        ----------
        section : str | None
            Optional section to retrieve (e.g., "system.dashboard")

        Returns
        -------
        dict[str, Any]
            Settings configuration

        """
        if section is None:
            return dict(self._current_config)

        # Navigate to section
        parts = section.split(".")
        current = self._current_config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return {}

        return dict(current) if isinstance(current, dict) else {}

    def update_settings(
        self,
        section: str,
        updates: Mapping[str, Any],
        validate: bool = True,
    ) -> dict[str, Any]:
        """
        Update settings configuration.

        Parameters
        ----------
        section : str
            Section to update (e.g., "system.dashboard")
        updates : Mapping[str, Any]
            Settings updates
        validate : bool
            Whether to validate against schema

        Returns
        -------
        dict[str, Any]
            Result with 'success', optional 'errors' and 'warnings'

        """
        start_time = time.perf_counter()

        # Validate if requested
        if validate:
            validation = self._validate_section_updates(section, updates)
            if not validation.valid:
                settings_updates_total.labels(section=section, status="rejected").inc()
                for error in validation.errors:
                    settings_validation_errors_total.labels(
                        section=section,
                        error_type=ValidationErrorType.SCHEMA_VIOLATION.value,
                    ).inc()
                return {
                    "success": False,
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                }

        # Apply updates
        try:
            self._apply_updates(section, updates)
            settings_updates_total.labels(section=section, status="success").inc()
            self._track_operation(operation="update_settings", status="success")

            # Persist to file
            if self._config_file:
                self._save_config()

            duration = time.perf_counter() - start_time
            return {
                "success": True,
                "warnings": validation.warnings if validate else [],
                "duration_seconds": duration,
            }

        except Exception as e:
            settings_updates_total.labels(section=section, status="error").inc()
            self._track_operation(operation="update_settings", status="error")
            logger.exception("Settings update failed", extra={"section": section})
            return {"success": False, "errors": [f"Update failed: {e!s}"]}

    def _validate_section_updates(
        self,
        section: str,
        updates: Mapping[str, Any],
    ) -> SettingsValidationResult:
        """Validate settings updates against schema."""
        errors: list[str] = []
        warnings: list[str] = []

        # Get schema for section
        schema = self._get_section_schema(section)
        if not schema:
            errors.append(f"Unknown section: {section}")
            return SettingsValidationResult(valid=False, errors=errors)

        # Validate each update
        for key, value in updates.items():
            if key not in schema.get("properties", {}):
                errors.append(f"Unknown setting: {key}")
                continue

            prop_schema = schema["properties"][key]

            # Type validation
            expected_type = prop_schema.get("type")
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"{key}: expected string, got {type(value).__name__}")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"{key}: expected integer, got {type(value).__name__}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"{key}: expected number, got {type(value).__name__}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"{key}: expected boolean, got {type(value).__name__}")

            # Range validation
            if "minimum" in prop_schema and isinstance(value, (int, float)):
                if value < prop_schema["minimum"]:
                    errors.append(f"{key}: value {value} below minimum {prop_schema['minimum']}")

            if "maximum" in prop_schema and isinstance(value, (int, float)):
                if value > prop_schema["maximum"]:
                    errors.append(f"{key}: value {value} above maximum {prop_schema['maximum']}")

            # Enum validation
            if "enum" in prop_schema:
                if value not in prop_schema["enum"]:
                    errors.append(f"{key}: value '{value}' not in allowed values {prop_schema['enum']}")

        # Business logic warnings
        if section == "system.security":
            if "session_timeout_minutes" in updates:
                timeout = updates["session_timeout_minutes"]
                if timeout > 720:  # 12 hours
                    warnings.append("Session timeout exceeds 12 hours - consider security implications")

        valid = len(errors) == 0
        return SettingsValidationResult(valid=valid, errors=errors, warnings=warnings)

    def _validate_config_recursive(
        self,
        config: dict[str, Any],
        schema: dict[str, Any],
    ) -> SettingsValidationResult:
        """Recursively validate entire configuration."""
        errors: list[str] = []
        warnings: list[str] = []

        for section_key, section_config in config.items():
            if section_key not in schema:
                errors.append(f"Unknown section: {section_key}")
                continue

            section_schema = schema[section_key]
            for subsection_key, subsection_config in section_config.items():
                if subsection_key not in section_schema:
                    errors.append(f"Unknown subsection: {section_key}.{subsection_key}")
                    continue

                section_path = f"{section_key}.{subsection_key}"

                # Validate subsection
                result = self._validate_section_updates(section_path, subsection_config)
                errors.extend(result.errors)
                warnings.extend(result.warnings)

        valid = len(errors) == 0
        return SettingsValidationResult(valid=valid, errors=errors, warnings=warnings)

    def _get_section_schema(self, section: str) -> dict[str, Any] | None:
        """Get schema for a specific section."""
        parts = section.split(".")
        current = CONFIG_SCHEMA
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current if isinstance(current, dict) else None

    def _apply_updates(self, section: str, updates: Mapping[str, Any]) -> None:
        """Apply settings updates to current configuration."""
        parts = section.split(".")
        current = self._current_config

        # Navigate to section
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Update final section
        last_part = parts[-1]
        if last_part not in current:
            current[last_part] = {}

        current[last_part].update(updates)

    def _load_or_create_config(self) -> dict[str, Any]:
        """Load configuration from file or create default."""
        if self._config_file and self._config_file.exists():
            try:
                with self._config_file.open("r") as f:
                    config: dict[str, Any] = json.load(f)
                    logger.info(f"Loaded configuration from {self._config_file}")
                    return config
            except Exception as e:
                logger.warning(f"Failed to load config, using defaults: {e}")

        # Return deep copy of defaults
        result: dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))
        return result

    def _save_config(self) -> None:
        """Save configuration to file."""
        if not self._config_file:
            return

        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            with self._config_file.open("w") as f:
                json.dump(self._current_config, f, indent=2)
                logger.info(f"Saved configuration to {self._config_file}")
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    async def health_check(self) -> dict[str, Any]:
        """
        Return health information for terminal service.

        Returns
        -------
        dict[str, Any]
            Health status information

        """
        return {
            "service": self.get_service_name(),
            "status": "healthy",
            "history_size": len(self._history),
            "config_sections": len(self._current_config),
            "integration_available": self._integration is not None,
        }


__all__ = [
    "CommandResult",
    "CommandType",
    "SettingsValidationResult",
    "TerminalService",
    "ValidationErrorType",
]
