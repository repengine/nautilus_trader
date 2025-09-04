from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from ml.observability.persistence import ObservabilityPersistor
from ml.observability.pipeline import build_latency_watermarks


def test_dto_builders_micro_bench() -> None:
    rows = [
        {
            "correlation_id": "00000000-0000-0000-0000-000000000001",
            "instrument_id": "EURUSD.SIM",
            "pipeline_stage": "data_ingestion",
            "ts_stage_start": i,
            "ts_stage_end": i + 1000,
        }
        for i in range(1000)
    ]
    t0 = time.perf_counter()
    df = build_latency_watermarks(rows)
    t1 = time.perf_counter()
    assert len(df) == 1000
    # Ensure build runs quickly in test environment (< 0.1s)
    assert (t1 - t0) < 0.1


def test_persist_micro_bench(tmp_path: Path) -> None:
    df = pd.DataFrame([
        {
            "correlation_id": "00000000-0000-0000-0000-000000000001",
            "instrument_id": "EURUSD.SIM",
            "pipeline_stage": "data_ingestion",
            "ts_stage_start": 1,
            "ts_stage_end": 2,
            "stage_latency_ns": 1,
            "cumulative_latency_ns": 1,
        }
        for _ in range(1000)
    ])
    per = ObservabilityPersistor(base_path=tmp_path, file_format="jsonl")
    t0 = time.perf_counter()
    out = per.persist({"latency": df})
    t1 = time.perf_counter()
    assert out["latency"].exists()
    assert (t1 - t0) < 0.2
