from __future__ import annotations

from unittest import mock

import pytest

from ml.cli.dashboard_welcome import parse_args
from ml.dashboard_bootstrap import welcome


def test_start_services_invokes_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> None:
        calls.append(cmd)

    monkeypatch.setattr("ml.dashboard_bootstrap.welcome.COMMAND_RUNNER", fake_run)

    welcome.start_services(compose_file="compose.yml", services=("grafana",), detach=True)

    assert calls == [["docker", "compose", "-f", "compose.yml", "up", "-d", "grafana"]]


def test_build_welcome_summary_status_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response = mock.Mock(status_code=200)
    monkeypatch.setattr(
        "ml.dashboard_bootstrap.welcome.requests.get",
        lambda url, timeout: fake_response,
    )
    summary = welcome.build_welcome_summary(
        compose_file="compose.yml",
        services=("grafana",),
        checks=(welcome.HealthCheck(name="Grafana", kind="dependency", url="http://example"),),
        timeout_seconds=0.1,
        retries=0,
        retry_interval_seconds=0.01,
        start=False,
    )
    assert "Grafana" in summary
    assert "✅" in summary


def test_cli_parse_args_default_services() -> None:
    args = parse_args([])
    assert args.compose_file == welcome.DEFAULT_COMPOSE_FILE
    assert args.services is None
    assert args.status_only is False
