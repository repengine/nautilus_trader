from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.training.teacher.streaming_loader import TFTStreamingSummary
from ml.training.teacher.streaming_telemetry import StreamingLoaderTelemetry
from ml.training.teacher.streaming_telemetry import StreamingRunTelemetry
from ml.training.teacher.tft_cli import _persist_teacher_outputs


def _make_telemetry() -> StreamingRunTelemetry:
    summary = TFTStreamingSummary(total_shards=2, total_rows=16, max_shard_rows=9)
    train = StreamingLoaderTelemetry(
        loader="train",
        total_shards=2,
        selected_shards=2,
        skipped_shards=0,
        total_rows=16,
        selected_rows=16,
        skipped_rows=0,
        total_sequences=8,
        selected_sequences=8,
        skipped_sequences=0,
    )
    val = StreamingLoaderTelemetry(
        loader="validation",
        total_shards=1,
        selected_shards=1,
        skipped_shards=0,
        total_rows=8,
        selected_rows=8,
        skipped_rows=0,
        total_sequences=4,
        selected_sequences=4,
        skipped_sequences=0,
    )
    return StreamingRunTelemetry(
        metadata_summary=summary,
        caps={"max_shards": 4, "max_total_rows": 500_000, "max_total_sequences": 300_000},
        train=train,
        validation=val,
    )


def test_persist_teacher_outputs_writes_all_files(tmp_path: Path) -> None:
    telemetry = _make_telemetry()
    q_train = np.linspace(0.1, 0.9, num=4, dtype=np.float32)
    q_val = np.linspace(0.2, 0.8, num=4, dtype=np.float32)
    y_val_true = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    meta = {"model_id": "m1", "feature_set_id": "fs", "feature_schema_hash": "abc123"}

    paths = _persist_teacher_outputs(
        tmp_path,
        q_train=q_train,
        q_val=q_val,
        y_val_true=y_val_true,
        meta=meta,
        streaming_telemetry=telemetry,
    )

    preds_path = paths["preds_path"]
    meta_path = paths["meta_path"]
    summary_path = paths["streaming_summary_path"]
    assert preds_path.exists()
    assert meta_path.exists()
    assert summary_path.exists()

    data = np.load(preds_path)
    assert np.allclose(data["q_train"], q_train)
    assert np.allclose(data["q_val"], q_val)
    assert np.allclose(data["y_val_true"].astype(np.float64), y_val_true)

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["caps"]["max_shards"] == 4
    assert summary["train"]["selected_sequences"] == 8


def test_persist_teacher_outputs_skips_summary_when_none(tmp_path: Path) -> None:
    q_val = np.array([0.4, 0.6], dtype=np.float32)
    y_val_true = np.array([0.0, 1.0], dtype=np.float64)
    meta = {"model_id": "m2", "feature_set_id": "fs", "feature_schema_hash": "xyz"}

    paths = _persist_teacher_outputs(
        tmp_path,
        q_train=None,
        q_val=q_val,
        y_val_true=y_val_true,
        meta=meta,
        streaming_telemetry=None,
    )

    assert "streaming_summary_path" not in paths
    assert paths["preds_path"].exists()
    assert paths["meta_path"].exists()
