from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml.observability.persistence import ObservabilityPersistor


def _ts_for(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> int:
    dt = datetime(year, month, day, hour, minute, second, tzinfo=UTC)
    return int(dt.timestamp() * 1e9)


def test_rotate_daily_writes_to_day_directory(tmp_path: Path) -> None:
    day_ts = _ts_for(2024, 1, 2, 12, 0, 0)
    df = pd.DataFrame(
        [
            {
                "correlation_id": "00000000-0000-0000-0000-000000000001",
                "instrument_id": "EURUSD.SIM",
                "pipeline_stage": "data_ingestion",
                "ts_stage_start": day_ts,
                "ts_stage_end": day_ts + 1_000_000,
            },
        ],
    )

    per = ObservabilityPersistor(base_path=tmp_path, file_format="jsonl", rotate_daily=True)
    out = per.persist({"latency": df})
    written = out["latency"]
    # Path should include day folder and a shard suffix
    assert written.parent.name == "2024-01-02"
    assert written.name.startswith("latency-") and written.name.endswith(".jsonl")


def test_compact_daily_concatenates_shards(tmp_path: Path) -> None:
    day = "2024-01-03"
    day_dir = tmp_path / day
    day_dir.mkdir(parents=True, exist_ok=True)
    # Create two shards
    (day_dir / "metrics-120000000000.jsonl").write_text(
        '{"metric_name":"a","metric_type":"counter","value":1.0,"timestamp":1,"labels":"{}"}\n',
    )
    (day_dir / "metrics-130000000000.jsonl").write_text(
        '{"metric_name":"b","metric_type":"counter","value":2.0,"timestamp":2,"labels":"{}"}\n',
    )

    per = ObservabilityPersistor(base_path=tmp_path, file_format="jsonl", rotate_daily=True)
    out = per.compact_daily(day)
    compacted = out["metrics"]
    assert compacted.exists()
    # Two lines from two shards
    assert len(compacted.read_text().strip().splitlines()) == 2
