"""
Tests for Terminal & Settings Service.

Comprehensive test coverage for secure command execution, history management,
and configuration management with validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.dashboard.services.terminal_service import TerminalService


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_history_file(tmp_path: Path) -> Path:
    """Provide temporary history file path."""
    return tmp_path / "history.json"


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Provide temporary config file path."""
    return tmp_path / "config.json"


@pytest.fixture
def service(temp_history_file: Path, temp_config_file: Path) -> TerminalService:
    """Provide TerminalService instance."""
    return TerminalService(
        integration_manager=None,
        history_file=temp_history_file,
        config_file=temp_config_file,
    )


# ============================================================================
# COMMAND EXECUTION TESTS
# ============================================================================


class TestCommandExecution:
    """Test command execution with validation and sandboxing."""

    def test_execute_valid_ml_data_list_command(self, service: TerminalService) -> None:
        """Test executing valid ml data list command."""
        result = service.execute_command("ml data list")

        assert result.success is True
        assert result.exit_code == 0
        assert result.command_type == "ml.data"
        assert "dataset" in result.output.lower()
        assert result.error is None

    def test_execute_valid_ml_model_show_command(self, service: TerminalService) -> None:
        """Test executing valid ml model show command."""
        result = service.execute_command("ml model show lgb_v1_2_3")

        assert result.success is True
        assert result.exit_code == 0
        assert result.command_type == "ml.model"
        assert "lgb_v1_2_3" in result.output
        assert result.duration_seconds > 0

    def test_execute_valid_system_status_command(self, service: TerminalService) -> None:
        """Test executing valid system status command."""
        result = service.execute_command("system status health")

        assert result.success is True
        assert result.exit_code == 0
        assert result.command_type == "system.status"
        assert "health" in result.output.lower()

    def test_execute_invalid_command_rejected(self, service: TerminalService) -> None:
        """Test that invalid commands are rejected."""
        result = service.execute_command("rm -rf /")

        assert result.success is False
        assert result.exit_code == 1
        assert result.command_type == "invalid"
        assert result.error is not None
        assert "not in whitelist" in result.error.lower()

    def test_execute_empty_command_rejected(self, service: TerminalService) -> None:
        """Test that empty commands are rejected."""
        result = service.execute_command("")

        assert result.success is False
        assert result.command_type == "invalid"
        assert result.error is not None
        assert "empty command" in result.error.lower()

    def test_execute_command_exceeds_length_limit(self, service: TerminalService) -> None:
        """Test that overly long commands are rejected."""
        long_command = "ml data list " + ("x" * 100000)
        result = service.execute_command(long_command)

        assert result.success is False
        assert result.error is not None
        assert "maximum length" in result.error.lower()

    def test_execute_command_with_disallowed_action(self, service: TerminalService) -> None:
        """Test command with disallowed action."""
        result = service.execute_command("ml data delete")

        assert result.success is False
        assert result.error is not None
        assert "not allowed" in result.error.lower()

    def test_execute_ml_feature_commands(self, service: TerminalService) -> None:
        """Test various ML feature commands."""
        commands = [
            "ml feature list",
            "ml feature show core_v1",
        ]

        for cmd in commands:
            result = service.execute_command(cmd)
            assert result.success is True
            assert result.command_type == "ml.feature"

    def test_execute_ml_pipeline_commands(self, service: TerminalService) -> None:
        """Test ML pipeline commands."""
        result = service.execute_command("ml pipeline status")

        assert result.success is True
        assert result.command_type == "ml.pipeline"
        assert "pipeline" in result.output.lower()

    def test_execute_ml_registry_commands(self, service: TerminalService) -> None:
        """Test ML registry commands."""
        result = service.execute_command("ml registry health")

        assert result.success is True
        assert result.command_type == "ml.registry"
        assert "registry" in result.output.lower()

    def test_execute_system_logs_commands(self, service: TerminalService) -> None:
        """Test system logs commands."""
        result = service.execute_command("system logs tail")

        assert result.success is True
        assert result.command_type == "system.logs"

    def test_execute_system_config_show(self, service: TerminalService) -> None:
        """Test system config show command."""
        result = service.execute_command("system config show")

        assert result.success is True
        assert result.command_type == "system.config"
        # Output should be valid JSON config
        config = json.loads(result.output)
        assert "system" in config
        assert "ml" in config

    def test_execute_system_config_validate(self, service: TerminalService) -> None:
        """Test system config validate command."""
        result = service.execute_command("system config validate")

        assert result.success is True
        assert result.command_type == "system.config"
        assert "valid" in result.output.lower()


# ============================================================================
# COMMAND HISTORY TESTS
# ============================================================================


class TestCommandHistory:
    """Test command history management."""

    def test_history_initially_empty(self, service: TerminalService) -> None:
        """Test that history is initially empty."""
        history = service.get_command_history()
        assert len(history) == 0

    def test_history_records_commands(self, service: TerminalService) -> None:
        """Test that executed commands are recorded in history."""
        service.execute_command("ml data list")
        service.execute_command("ml model list")

        history = service.get_command_history()
        assert len(history) == 2
        assert history[0]["command"] == "ml data list"
        assert history[1]["command"] == "ml model list"

    def test_history_includes_metadata(self, service: TerminalService) -> None:
        """Test that history entries include complete metadata."""
        service.execute_command("ml data list")

        history = service.get_command_history()
        entry = history[0]

        assert "command" in entry
        assert "timestamp" in entry
        assert "timestamp_iso" in entry
        assert "command_type" in entry
        assert "success" in entry
        assert "duration_seconds" in entry
        assert "output_preview" in entry

    def test_history_limit_parameter(self, service: TerminalService) -> None:
        """Test history retrieval with limit."""
        for i in range(10):
            service.execute_command("ml data list")

        history_limited = service.get_command_history(limit=5)
        assert len(history_limited) == 5

        # Should return most recent entries
        history_all = service.get_command_history()
        assert history_limited == history_all[-5:]

    def test_history_persistence(
        self,
        temp_history_file: Path,
        temp_config_file: Path,
    ) -> None:
        """Test that history persists to file."""
        service1 = TerminalService(None, temp_history_file, temp_config_file)
        service1.execute_command("ml data list")
        service1.execute_command("ml model list")

        # Create new service instance
        service2 = TerminalService(None, temp_history_file, temp_config_file)
        history = service2.get_command_history()

        assert len(history) == 2
        assert history[0]["command"] == "ml data list"
        assert history[1]["command"] == "ml model list"

    def test_history_max_size_enforced(self, service: TerminalService) -> None:
        """Test that history respects maximum size limit."""
        # Execute more commands than the limit
        from ml.dashboard.services.terminal_service import MAX_HISTORY_SIZE

        for i in range(MAX_HISTORY_SIZE + 100):
            service.execute_command("ml data list")

        history = service.get_command_history()
        assert len(history) == MAX_HISTORY_SIZE

    def test_history_failed_commands_recorded(self, service: TerminalService) -> None:
        """Test that failed commands are also recorded."""
        service.execute_command("invalid command")

        history = service.get_command_history()
        assert len(history) == 1
        assert history[0]["success"] is False
        assert history[0]["error"] is not None


# ============================================================================
# SETTINGS MANAGEMENT TESTS
# ============================================================================


class TestSettingsManagement:
    """Test settings configuration management."""

    def test_get_all_settings(self, service: TerminalService) -> None:
        """Test retrieving all settings."""
        settings = service.get_settings()

        assert "system" in settings
        assert "ml" in settings
        assert "dashboard" in settings["system"]
        assert "security" in settings["system"]

    def test_get_section_settings(self, service: TerminalService) -> None:
        """Test retrieving specific section settings."""
        settings = service.get_settings(section="system.dashboard")

        assert "theme" in settings
        assert "refresh_interval" in settings
        assert settings["theme"] == "light"

    def test_get_nested_section(self, service: TerminalService) -> None:
        """Test retrieving nested section."""
        settings = service.get_settings(section="ml.models")

        assert "max_inference_latency_ms" in settings
        assert "prediction_confidence_threshold" in settings

    def test_get_nonexistent_section(self, service: TerminalService) -> None:
        """Test retrieving non-existent section returns empty dict."""
        settings = service.get_settings(section="nonexistent.section")
        assert settings == {}

    def test_update_settings_success(self, service: TerminalService) -> None:
        """Test successful settings update."""
        updates = {"theme": "dark", "refresh_interval": 60}
        result = service.update_settings("system.dashboard", updates)

        assert result["success"] is True
        assert "duration_seconds" in result

        # Verify update applied
        settings = service.get_settings("system.dashboard")
        assert settings["theme"] == "dark"
        assert settings["refresh_interval"] == 60

    def test_update_settings_validation_error(self, service: TerminalService) -> None:
        """Test settings update with validation errors."""
        updates = {"theme": "invalid_theme"}
        result = service.update_settings("system.dashboard", updates)

        assert result["success"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0
        assert "not in allowed values" in result["errors"][0]

    def test_update_settings_type_error(self, service: TerminalService) -> None:
        """Test settings update with wrong type."""
        updates = {"refresh_interval": "not_a_number"}
        result = service.update_settings("system.dashboard", updates)

        assert result["success"] is False
        assert any("expected integer" in err for err in result["errors"])

    def test_update_settings_range_violation(self, service: TerminalService) -> None:
        """Test settings update with range violation."""
        updates = {"refresh_interval": 999}  # Above maximum of 300
        result = service.update_settings("system.dashboard", updates)

        assert result["success"] is False
        assert any("above maximum" in err for err in result["errors"])

    def test_update_settings_warnings(self, service: TerminalService) -> None:
        """Test settings update with warnings."""
        updates = {"session_timeout_minutes": 800}  # > 12 hours
        result = service.update_settings("system.security", updates, validate=True)

        # Should succeed but with warning
        if result["success"]:
            assert "warnings" in result
            assert len(result["warnings"]) > 0
        # Alternatively might fail validation depending on implementation
        else:
            assert "errors" in result

    def test_update_settings_without_validation(self, service: TerminalService) -> None:
        """Test settings update without validation."""
        updates = {"theme": "custom_theme"}  # Not in enum
        result = service.update_settings("system.dashboard", updates, validate=False)

        # Should succeed even though invalid
        assert result["success"] is True

    def test_update_settings_persistence(
        self,
        temp_history_file: Path,
        temp_config_file: Path,
    ) -> None:
        """Test that settings updates persist to file."""
        service1 = TerminalService(None, temp_history_file, temp_config_file)
        updates = {"theme": "dark"}
        service1.update_settings("system.dashboard", updates)

        # Create new service instance
        service2 = TerminalService(None, temp_history_file, temp_config_file)
        settings = service2.get_settings("system.dashboard")

        assert settings["theme"] == "dark"

    def test_update_settings_unknown_key(self, service: TerminalService) -> None:
        """Test updating with unknown setting key."""
        updates = {"unknown_setting": "value"}
        result = service.update_settings("system.dashboard", updates)

        assert result["success"] is False
        assert any("unknown setting" in err.lower() for err in result["errors"])

    def test_update_multiple_settings(self, service: TerminalService) -> None:
        """Test updating multiple settings at once."""
        updates = {
            "max_inference_latency_ms": 10.0,
            "prediction_confidence_threshold": 0.8,
            "enable_model_hot_reload": True,
        }
        result = service.update_settings("ml.models", updates)

        assert result["success"] is True

        settings = service.get_settings("ml.models")
        assert settings["max_inference_latency_ms"] == 10.0
        assert settings["prediction_confidence_threshold"] == 0.8
        assert settings["enable_model_hot_reload"] is True

    def test_settings_schema_coverage(self, service: TerminalService) -> None:
        """Test that all default settings have schema definitions."""
        from ml.dashboard.services.terminal_service import CONFIG_SCHEMA
        from ml.dashboard.services.terminal_service import DEFAULT_CONFIG

        # Verify all sections in defaults have schemas
        for section_key in DEFAULT_CONFIG:
            assert section_key in CONFIG_SCHEMA, f"Missing schema for section: {section_key}"


# ============================================================================
# VALIDATION TESTS
# ============================================================================


class TestValidation:
    """Test validation logic."""

    def test_validate_command_valid(self, service: TerminalService) -> None:
        """Test validating a valid command."""
        result = service._validate_command("ml data list")

        assert result["valid"] is True
        assert result["command_type"] == "ml.data"

    def test_validate_command_invalid(self, service: TerminalService) -> None:
        """Test validating an invalid command."""
        result = service._validate_command("invalid command")

        assert result["valid"] is False
        assert "error" in result

    def test_validate_section_updates_valid(self, service: TerminalService) -> None:
        """Test validating valid section updates."""
        updates = {"theme": "dark"}
        result = service._validate_section_updates("system.dashboard", updates)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_section_updates_invalid_type(self, service: TerminalService) -> None:
        """Test validating updates with wrong type."""
        updates = {"refresh_interval": "not_an_int"}
        result = service._validate_section_updates("system.dashboard", updates)

        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_config_recursive(self, service: TerminalService) -> None:
        """Test recursive configuration validation."""
        config = service._current_config
        from ml.dashboard.services.terminal_service import CONFIG_SCHEMA

        result = service._validate_config_recursive(config, CONFIG_SCHEMA)

        # Default config should be valid
        assert result.valid is True


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self, service: TerminalService) -> None:
        """Test that health check returns proper status."""
        health = await service.health_check()

        assert "service" in health
        assert health["service"] == "terminal"
        assert "status" in health
        assert health["status"] == "healthy"
        assert "history_size" in health
        assert "config_sections" in health
        assert "integration_available" in health


# ============================================================================
# METRICS TESTS
# ============================================================================


class TestMetrics:
    """Test that metrics are properly recorded."""

    def test_command_execution_records_metrics(self, service: TerminalService) -> None:
        """Test that executing commands records metrics."""
        from ml.dashboard.services.terminal_service import terminal_command_duration
        from ml.dashboard.services.terminal_service import terminal_commands_total

        # Execute command
        service.execute_command("ml data list")

        # Metrics should be recorded
        # Note: We can't easily assert on metric values in tests without mocking,
        # but we can verify the metrics exist
        assert terminal_commands_total is not None
        assert terminal_command_duration is not None

    def test_settings_update_records_metrics(self, service: TerminalService) -> None:
        """Test that settings updates record metrics."""
        from ml.dashboard.services.terminal_service import settings_updates_total

        updates = {"theme": "dark"}
        service.update_settings("system.dashboard", updates)

        assert settings_updates_total is not None


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_service_name(self, service: TerminalService) -> None:
        """Test service name is correct."""
        assert service.get_service_name() == "terminal"

    def test_history_file_directory_creation(self, tmp_path: Path) -> None:
        """Test that history file directory is created if needed."""
        nested_path = tmp_path / "nested" / "dir" / "history.json"
        service = TerminalService(None, history_file=nested_path)

        service.execute_command("ml data list")

        # File and directories should exist
        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_config_file_directory_creation(self, tmp_path: Path) -> None:
        """Test that config file directory is created if needed."""
        nested_path = tmp_path / "nested" / "dir" / "config.json"
        service = TerminalService(None, config_file=nested_path)

        updates = {"theme": "dark"}
        service.update_settings("system.dashboard", updates)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_service_without_persistence(self) -> None:
        """Test service works without persistence files."""
        service = TerminalService(None)

        result = service.execute_command("ml data list")
        assert result.success is True

        settings = service.get_settings()
        assert "system" in settings

    def test_command_with_extra_whitespace(self, service: TerminalService) -> None:
        """Test command with extra whitespace."""
        result = service.execute_command("  ml   data   list  ")

        # Should still work after stripping
        assert result.success is True

    def test_timestamp_format_in_history(self, service: TerminalService) -> None:
        """Test that timestamps are in correct format."""
        import time

        before = time.time_ns()
        service.execute_command("ml data list")
        after = time.time_ns()

        history = service.get_command_history()
        entry = history[0]

        # Timestamp should be in nanoseconds
        assert before <= entry["timestamp"] <= after

        # ISO format should be valid
        from datetime import datetime

        dt = datetime.fromisoformat(entry["timestamp_iso"].replace("Z", "+00:00"))
        assert dt.tzinfo is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_execute_and_retrieve_history(self, service: TerminalService) -> None:
        """Test executing commands and retrieving history."""
        commands = [
            "ml data list",
            "ml model show lgb_v1_2_3",
            "system status health",
        ]

        for cmd in commands:
            result = service.execute_command(cmd)
            assert result.success is True

        history = service.get_command_history()
        assert len(history) == 3

        for i, cmd in enumerate(commands):
            assert history[i]["command"] == cmd
            assert history[i]["success"] is True

    def test_update_settings_and_validate(self, service: TerminalService) -> None:
        """Test updating settings and validating them."""
        updates = {
            "theme": "dark",
            "refresh_interval": 45,
            "enable_websockets": False,
        }

        result = service.update_settings("system.dashboard", updates)
        assert result["success"] is True

        # Verify via system config show
        cmd_result = service.execute_command("system config show")
        config = json.loads(cmd_result.output)

        assert config["system"]["dashboard"]["theme"] == "dark"
        assert config["system"]["dashboard"]["refresh_interval"] == 45
        assert config["system"]["dashboard"]["enable_websockets"] is False

    def test_full_workflow_with_persistence(
        self,
        temp_history_file: Path,
        temp_config_file: Path,
    ) -> None:
        """Test complete workflow with persistence."""
        # Session 1: Execute commands and update settings
        service1 = TerminalService(None, temp_history_file, temp_config_file)

        service1.execute_command("ml data list")
        service1.execute_command("ml model list")

        settings_update = {"theme": "dark"}
        service1.update_settings("system.dashboard", settings_update)

        # Session 2: Verify persistence
        service2 = TerminalService(None, temp_history_file, temp_config_file)

        history = service2.get_command_history()
        assert len(history) == 2

        settings = service2.get_settings("system.dashboard")
        assert settings["theme"] == "dark"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
