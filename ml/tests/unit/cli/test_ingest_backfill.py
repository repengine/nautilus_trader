from __future__ import annotations

from pathlib import Path

import pytest

import ml.cli.ingest_backfill as ingest_backfill_cli
from ml.orchestration.ingestion_coordinator import IngestBackfillRuntimeResult


def test_main_passes_parsed_comma_instruments_to_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def _run(config: object, *, emit: object | None = None) -> IngestBackfillRuntimeResult:
        del emit
        captured.append(config)
        return IngestBackfillRuntimeResult(total_windows_planned=0, state_saved=False)

    monkeypatch.setattr(ingest_backfill_cli, "run_ingest_backfill", _run)

    rc = ingest_backfill_cli.main(
        [
            "--db",
            "postgresql://user:pass@localhost:5432/ml",
            "--dataset-id",
            "EQUS.MINI",
            "--schema",
            "ohlcv-1m",
            "--instruments",
            "SPY.XNAS,QQQ.XNAS",
            "--client-mode",
            "noop",
        ],
    )

    assert rc == 0
    assert len(captured) == 1
    config = captured[0]
    assert getattr(config, "instruments") == ("SPY.XNAS", "QQQ.XNAS")


def test_main_reads_instruments_from_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[object] = []
    instruments_file = tmp_path / "instruments.txt"
    instruments_file.write_text("SPY.XNAS\nQQQ.XNAS\n", encoding="utf-8")

    def _run(config: object, *, emit: object | None = None) -> IngestBackfillRuntimeResult:
        del emit
        captured.append(config)
        return IngestBackfillRuntimeResult(total_windows_planned=0, state_saved=False)

    monkeypatch.setattr(ingest_backfill_cli, "run_ingest_backfill", _run)

    rc = ingest_backfill_cli.main(
        [
            "--db",
            "postgresql://user:pass@localhost:5432/ml",
            "--dataset-id",
            "EQUS.MINI",
            "--schema",
            "ohlcv-1m",
            "--instruments",
            str(instruments_file),
            "--client-mode",
            "noop",
        ],
    )

    assert rc == 0
    assert len(captured) == 1
    config = captured[0]
    assert getattr(config, "instruments") == ("SPY.XNAS", "QQQ.XNAS")


def test_main_maps_runtime_value_error_to_parser_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _run(config: object, *, emit: object | None = None) -> IngestBackfillRuntimeResult:
        del config
        del emit
        raise ValueError("bad config")

    monkeypatch.setattr(ingest_backfill_cli, "run_ingest_backfill", _run)

    with pytest.raises(SystemExit) as exc_info:
        ingest_backfill_cli.main(
            [
                "--db",
                "postgresql://user:pass@localhost:5432/ml",
                "--dataset-id",
                "EQUS.MINI",
                "--schema",
                "ohlcv-1m",
                "--instruments",
                "SPY.XNAS",
            ],
        )

    assert exc_info.value.code == 2
