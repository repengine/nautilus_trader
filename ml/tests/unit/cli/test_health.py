from __future__ import annotations

import json

import pytest

import ml.cli.health as cli


def test_main_emits_sorted_json_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def _fake_aggregate(
        db_connection: str | None = None,
        *,
        strict_protocol_validation: bool = False,
    ) -> dict[str, object]:
        captured["db_connection"] = db_connection
        captured["strict_protocol_validation"] = strict_protocol_validation
        return {
            "system": {"healthy": True, "unhealthy": []},
            "domains": {},
            "components": {},
        }

    monkeypatch.setattr(cli, "aggregate_integration_health", _fake_aggregate)

    rc = cli.main(["--db-connection", "postgresql://example", "--strict"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert captured["db_connection"] == "postgresql://example"
    assert captured["strict_protocol_validation"] is True
    assert payload["system"]["healthy"] is True
    assert payload["system"]["unhealthy"] == []
