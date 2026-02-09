from __future__ import annotations

from pathlib import Path

import pytest

import ml.cli.observability_backfill as observability_backfill_cli


def test_main_passes_config_to_backfill_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def _fake_backfill(config: object, *, emit: object | None = None) -> dict[str, int]:
        captured["config"] = config
        emit_fn = emit
        if callable(emit_fn):
            emit_fn("metrics: 4")
        return {"metrics": 4}

    monkeypatch.setattr(
        observability_backfill_cli,
        "backfill_observability_tables",
        _fake_backfill,
    )

    rc = observability_backfill_cli.main(
        [
            "--src",
            str(tmp_path),
            "--db-url",
            "sqlite:///observability.db",
        ],
    )

    out = capsys.readouterr().out
    config = captured["config"]
    assert rc == 0
    assert getattr(config, "src") == tmp_path
    assert getattr(config, "db_url") == "sqlite:///observability.db"
    assert "metrics: 4" in out


def test_main_maps_invalid_config_to_parser_exit(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        observability_backfill_cli.main(
            [
                "--src",
                str(tmp_path),
                "--db-url",
                "",
            ],
        )

    assert exc_info.value.code == 2
