from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import ml.observability.backfill as backfill_module
from ml.observability.backfill import ObservabilityBackfillConfig
from ml.observability.backfill import backfill_observability_tables
from ml.observability.backfill import collect_observability_backfill_tables


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_json(path, orient="records", lines=True)


def test_collect_observability_backfill_tables_reads_day_and_root_shards(
    tmp_path: Path,
) -> None:
    day_dir = tmp_path / "2025-01-01"
    day_dir.mkdir(parents=True)
    _write_jsonl(day_dir / "metrics.jsonl", [{"value": 1}, {"value": 2}])
    _write_jsonl(tmp_path / "metrics_shard.jsonl", [{"value": 3}])

    tables = collect_observability_backfill_tables(
        tmp_path,
        table_names=("metrics",),
    )

    assert tuple(tables) == ("metrics",)
    assert len(tables["metrics"]) == 3


def test_backfill_observability_tables_when_no_files_emits_notice(
    tmp_path: Path,
) -> None:
    emitted: list[str] = []
    config = ObservabilityBackfillConfig(
        src=tmp_path,
        db_url="sqlite:///observability.db",
        table_names=("metrics",),
    )

    result = backfill_observability_tables(config, emit=emitted.append)

    assert result == {}
    assert emitted == ["No observability files found"]


def test_backfill_observability_tables_persists_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_jsonl(tmp_path / "metrics.jsonl", [{"value": 10}, {"value": 11}])
    captured: dict[str, object] = {}

    class _DummyPersistor:
        def __init__(self, connection_string: str) -> None:
            captured["connection_string"] = connection_string

        def persist(self, tables: Mapping[str, pd.DataFrame | None]) -> dict[str, int]:
            metrics_table = cast(pd.DataFrame, tables["metrics"])
            captured["table_names"] = tuple(tables)
            captured["row_count"] = len(metrics_table)
            return {"metrics": len(metrics_table)}

    monkeypatch.setattr(backfill_module, "ObservabilityDBPersistor", _DummyPersistor)

    config = ObservabilityBackfillConfig(
        src=tmp_path,
        db_url="sqlite:///observability.db",
        table_names=("metrics",),
    )
    emitted: list[str] = []

    result = backfill_observability_tables(config, emit=emitted.append)

    assert result == {"metrics": 2}
    assert captured["connection_string"] == "sqlite:///observability.db"
    assert captured["table_names"] == ("metrics",)
    assert captured["row_count"] == 2
    assert emitted == ["metrics: 2"]
