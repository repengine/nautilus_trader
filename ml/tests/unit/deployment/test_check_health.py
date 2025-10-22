#!/usr/bin/env python3
"""
Unit tests for ML deployment health check script.

Tests health checks for all services and inter-service communication.

"""

from __future__ import annotations

import json
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import requests

from ml.deployment.check_health import check_docker_compose
from ml.deployment.check_health import check_grafana
from ml.deployment.check_health import check_ml_pipeline
from ml.deployment.check_health import check_postgres
from ml.deployment.check_health import check_prometheus
from ml.deployment.check_health import check_redis
from ml.deployment.check_health import check_service_health
from ml.deployment.check_health import main


@pytest.mark.redis
@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.unit
class TestServiceHealthChecks:
    """
    Test individual service health check functions.
    """

    def test_check_service_health_success(self):
        """
        Test check_service_health with successful check.
        """

        def successful_check():
            return True

        result, message = check_service_health("TestService", successful_check)
        assert result is True
        assert message == "OK"

    def test_check_service_health_failure(self):
        """
        Test check_service_health with failed check.
        """

        def failed_check():
            return False

        result, message = check_service_health("TestService", failed_check)
        assert result is False
        assert message == "UNHEALTHY"

    def test_check_service_health_exception(self):
        """
        Test check_service_health with exception.
        """

        def error_check():
            raise RuntimeError("Connection failed")

        result, message = check_service_health("TestService", error_check)
        assert result is False
        assert "ERROR: Connection failed" in message

    @patch("ml.deployment.check_health.run_command")
    @patch("ml.deployment.check_health._compose_command")
    def test_check_postgres_healthy(self, mock_compose, mock_run):
        """
        Test PostgreSQL health check when healthy.
        """
        command = [
            "docker-compose",
            "exec",
            "-T",
            "postgres",
            "pg_isready",
            "-U",
            "postgres",
        ]
        mock_compose.return_value = command
        mock_run.return_value = Mock(returncode=0, stdout="")

        result = check_postgres()

        assert result is True
        mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    @patch("ml.deployment.check_health.run_command")
    @patch("ml.deployment.check_health._compose_command")
    def test_check_postgres_unhealthy(self, mock_compose, mock_run):
        """
        Test PostgreSQL health check when unhealthy.
        """
        mock_compose.return_value = ["docker-compose", "exec", "-T", "postgres", "pg_isready", "-U", "postgres"]
        mock_run.return_value = Mock(returncode=1, stdout="")

        result = check_postgres()

        assert result is False

    @patch("ml.deployment.check_health.run_command")
    @patch("ml.deployment.check_health._compose_command")
    def test_check_redis_healthy(self, mock_compose, mock_run):
        """
        Test Redis health check when healthy.
        """
        command = [
            "docker-compose",
            "exec",
            "-T",
            "redis",
            "redis-cli",
            "ping",
        ]
        mock_compose.return_value = command
        mock_run.return_value = Mock(returncode=0, stdout="PONG")

        result = check_redis()

        assert result is True
        mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    @patch("ml.deployment.check_health.run_command")
    @patch("ml.deployment.check_health._compose_command")
    def test_check_redis_unhealthy(self, mock_compose, mock_run):
        """
        Test Redis health check when unhealthy.
        """
        mock_compose.return_value = ["docker-compose", "exec", "-T", "redis", "redis-cli", "ping"]
        mock_run.return_value = Mock(returncode=0, stdout="Error: Connection refused")

        result = check_redis()

        assert result is False

    @patch("requests.get")
    def test_check_ml_pipeline_healthy(self, mock_get):
        """
        Test ML Pipeline health check when healthy.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = check_ml_pipeline()

        assert result is True
        mock_get.assert_called_once_with("http://localhost:8080/health", timeout=5)

    @patch("requests.get")
    def test_check_ml_pipeline_unhealthy(self, mock_get):
        """
        Test ML Pipeline health check when unhealthy.
        """
        mock_response = Mock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        result = check_ml_pipeline()

        assert result is False

    @patch("requests.get")
    def test_check_ml_pipeline_connection_error(self, mock_get):
        """
        Test ML Pipeline health check with connection error.
        """
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        result = check_ml_pipeline()

        assert result is False

    @patch("requests.get")
    def test_check_ml_pipeline_timeout(self, mock_get):
        """
        Test ML Pipeline health check with timeout.
        """
        mock_get.side_effect = requests.Timeout("Request timed out")

        result = check_ml_pipeline()

        assert result is False

    @patch("requests.get")
    def test_check_prometheus_healthy(self, mock_get):
        """
        Test Prometheus health check when healthy.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = check_prometheus()

        assert result is True
        mock_get.assert_called_once_with("http://localhost:9090/-/healthy", timeout=5)

    @patch("requests.get")
    def test_check_prometheus_unhealthy(self, mock_get):
        """
        Test Prometheus health check when unhealthy.
        """
        mock_response = Mock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        result = check_prometheus()

        assert result is False

    @patch("requests.get")
    def test_check_grafana_healthy(self, mock_get):
        """
        Test Grafana health check when healthy.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = check_grafana()

        assert result is True
        mock_get.assert_called_once_with("http://localhost:3000/api/health", timeout=5)

    @patch("requests.get")
    def test_check_grafana_unhealthy(self, mock_get):
        """
        Test Grafana health check when unhealthy.
        """
        mock_response = Mock()
        mock_response.status_code = 401  # Unauthorized but service is running
        mock_get.return_value = mock_response

        result = check_grafana()

        assert result is False

    @patch("ml.deployment.check_health.run_command")
    @patch("ml.deployment.check_health._compose_command")
    def test_check_docker_compose_all_running(self, mock_compose, mock_run):
        """
        Test Docker Compose check when all services are running.
        """
        services_json = json.dumps(
            [
                {"Service": "postgres", "State": "running"},
                {"Service": "ml_pipeline", "State": "running"},
                {"Service": "redis", "State": "running"},
            ],
        )

        command = ["docker-compose", "ps", "--format", "json"]
        mock_compose.return_value = command
        mock_run.return_value = Mock(returncode=0, stdout=services_json)

        result = check_docker_compose()

        assert result is True
        mock_run.assert_called_once_with(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    @patch("ml.deployment.check_health.run_command")
    @patch("ml.deployment.check_health._compose_command")
    def test_check_docker_compose_missing_service(self, mock_compose, mock_run):
        """
        Test Docker Compose check when required service is missing.
        """
        services_json = json.dumps(
            [
                {"Service": "postgres", "State": "running"},
                {"Service": "redis", "State": "running"},
                # ml_pipeline is missing
            ],
        )

        mock_compose.return_value = ["docker-compose", "ps", "--format", "json"]
        mock_run.return_value = Mock(returncode=0, stdout=services_json)

        result = check_docker_compose()

        assert result is False

    @patch("ml.deployment.check_health.run_command")
    @patch("ml.deployment.check_health._compose_command")
    def test_check_docker_compose_service_not_running(self, mock_compose, mock_run):
        """
        Test Docker Compose check when service is not running.
        """
        services_json = json.dumps(
            [
                {"Service": "postgres", "State": "running"},
                {"Service": "ml_pipeline", "State": "exited"},
                {"Service": "redis", "State": "running"},
            ],
        )

        mock_compose.return_value = ["docker-compose", "ps", "--format", "json"]
        mock_run.return_value = Mock(returncode=0, stdout=services_json)

        result = check_docker_compose()

        assert result is False

    @patch("ml.deployment.check_health.run_command")
    @patch("ml.deployment.check_health._compose_command")
    def test_check_docker_compose_command_error(self, mock_compose, mock_run):
        """
        Test Docker Compose check when command fails.
        """
        mock_compose.return_value = ["docker-compose", "ps", "--format", "json"]
        mock_run.return_value = Mock(returncode=1, stdout="")

        result = check_docker_compose()

        assert result is False

    @patch("ml.deployment.check_health.run_command")
    @patch("ml.deployment.check_health._compose_command")
    def test_check_docker_compose_invalid_json(self, mock_compose, mock_run):
        """
        Test Docker Compose check with invalid JSON output.
        """
        mock_compose.return_value = ["docker-compose", "ps", "--format", "json"]
        mock_run.return_value = Mock(returncode=0, stdout="invalid json")

        result = check_docker_compose()

        assert result is False


class TestMainFunction:
    """
    Test the main health check function.
    """

    @patch("ml.deployment.check_health.check_docker_compose", return_value=True)
    @patch("ml.deployment.check_health.check_postgres", return_value=True)
    @patch("ml.deployment.check_health.check_redis", return_value=True)
    @patch("ml.deployment.check_health.check_ml_pipeline", return_value=True)
    @patch("ml.deployment.check_health.check_prometheus", return_value=True)
    @patch("ml.deployment.check_health.check_grafana", return_value=True)
    def test_main_all_healthy(
        self,
        mock_grafana,
        mock_prometheus,
        mock_ml_pipeline,
        mock_redis,
        mock_postgres,
        mock_docker_compose,
        capsys,
    ):
        """
        Test main function when all services are healthy.
        """
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "ML Pipeline Health Check" in captured.out
        assert "All services are healthy!" in captured.out
        assert "[✓]" in captured.out
        assert "[✗]" not in captured.out

    @patch("ml.deployment.check_health.check_docker_compose", return_value=True)
    @patch("ml.deployment.check_health.check_postgres", return_value=False)
    @patch("ml.deployment.check_health.check_redis", return_value=True)
    @patch("ml.deployment.check_health.check_ml_pipeline", return_value=True)
    @patch("ml.deployment.check_health.check_prometheus", return_value=True)
    @patch("ml.deployment.check_health.check_grafana", return_value=True)
    def test_main_some_unhealthy(
        self,
        mock_grafana,
        mock_prometheus,
        mock_ml_pipeline,
        mock_redis,
        mock_postgres,
        mock_docker_compose,
        capsys,
    ):
        """
        Test main function when some services are unhealthy.
        """
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "ML Pipeline Health Check" in captured.out
        assert "Some services are unhealthy" in captured.out
        assert "[✓]" in captured.out  # Some healthy
        assert "[✗]" in captured.out  # Some unhealthy
        assert "make logs SERVICE=<service_name>" in captured.out

    @patch("ml.deployment.check_health.check_docker_compose", return_value=False)
    @patch("ml.deployment.check_health.check_postgres", return_value=False)
    @patch("ml.deployment.check_health.check_redis", return_value=False)
    @patch("ml.deployment.check_health.check_ml_pipeline", return_value=False)
    @patch("ml.deployment.check_health.check_prometheus", return_value=False)
    @patch("ml.deployment.check_health.check_grafana", return_value=False)
    def test_main_all_unhealthy(
        self,
        mock_grafana,
        mock_prometheus,
        mock_ml_pipeline,
        mock_redis,
        mock_postgres,
        mock_docker_compose,
        capsys,
    ):
        """
        Test main function when all services are unhealthy.
        """
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "ML Pipeline Health Check" in captured.out
        assert "[✗]" in captured.out
        assert "[✓]" not in captured.out  # None healthy

    @patch(
        "ml.deployment.check_health.check_docker_compose",
        side_effect=Exception("Unexpected error"),
    )
    @patch("ml.deployment.check_health.check_postgres", return_value=True)
    @patch("ml.deployment.check_health.check_redis", return_value=True)
    @patch("ml.deployment.check_health.check_ml_pipeline", return_value=True)
    @patch("ml.deployment.check_health.check_prometheus", return_value=True)
    @patch("ml.deployment.check_health.check_grafana", return_value=True)
    def test_main_handles_exceptions(
        self,
        mock_grafana,
        mock_prometheus,
        mock_ml_pipeline,
        mock_redis,
        mock_postgres,
        mock_docker_compose,
        capsys,
    ):
        """
        Test main function handles exceptions in checks.
        """
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "[✗]" in captured.out  # Docker Compose check failed
        assert "ERROR:" in captured.out

    def test_main_output_formatting(self, capsys):
        """
        Test main function output formatting.
        """
        with patch("ml.deployment.check_health.check_docker_compose", return_value=True):
            with patch("ml.deployment.check_health.check_postgres", return_value=True):
                with patch("ml.deployment.check_health.check_redis", return_value=False):
                    with patch("ml.deployment.check_health.check_ml_pipeline", return_value=True):
                        with patch(
                            "ml.deployment.check_health.check_prometheus",
                            return_value=False,
                        ):
                            with patch(
                                "ml.deployment.check_health.check_grafana",
                                return_value=True,
                            ):
                                with pytest.raises(SystemExit):
                                    main()

        captured = capsys.readouterr()
        lines = captured.out.split("\n")

        # Check header
        assert any("=" * 60 in line for line in lines)
        assert any("ML Pipeline Health Check" in line for line in lines)

        # Check service checks
        assert any("Docker Compose" in line for line in lines)
        assert any("PostgreSQL" in line for line in lines)
        assert any("Redis" in line for line in lines)
        assert any("ML Pipeline" in line for line in lines)
        assert any("Prometheus" in line for line in lines)
        assert any("Grafana" in line for line in lines)

        # Check status indicators
        assert any("[✓]" in line for line in lines)  # Healthy services
        assert any("[✗]" in line for line in lines)  # Unhealthy services
