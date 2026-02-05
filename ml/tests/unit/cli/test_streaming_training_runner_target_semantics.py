from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.cli.streaming_training_runner import _resolve_dataset_spec
from ml.tests.utils.targets import build_default_target_semantics
from ml.training.datasets.target_generator import build_target_semantics_metadata


def _write_metadata(
    dataset_dir: Path,
    *,
    target_col: str,
    target_semantics: dict[str, object] | None,
) -> None:
    payload: dict[str, object] = {
        "dataset_id": "streaming-test",
        "build_ts": "2025-01-01T00:00:00Z",
        "column_info": {
            "target_col": target_col,
            "categorical_columns": ["instrument_id"],
            "time_idx_col": "time_index",
            "group_id_col": "instrument_id",
        },
    }
    if target_semantics is not None:
        payload["target_semantics"] = target_semantics
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_report(dataset_dir: Path) -> None:
    (dataset_dir / "report.json").write_text("{}", encoding="utf-8")


def _call_resolve(dataset_dir: Path) -> None:
    _resolve_dataset_spec(
        dataset_dir,
        batch_size=1,
        dataloader_workers=0,
        max_total_rows=None,
        max_total_sequences=None,
        max_shards=None,
        max_encoder_length=2,
        max_prediction_length=1,
        include_macro=False,
        include_calendar=False,
        include_events=False,
        include_earnings=False,
        include_micro=False,
        include_l2=False,
        include_macro_revisions=False,
        include_macro_deltas=False,
        include_calendar_lags=False,
        include_clustering_tags=False,
        include_context_features=False,
        dataset_seed=None,
    )


def test_streaming_runner_requires_target_semantics_metadata(tmp_path: Path) -> None:
    _write_metadata(tmp_path, target_col="y", target_semantics=None)
    _write_report(tmp_path)

    with pytest.raises(ValueError, match="dataset metadata missing target_semantics"):
        _call_resolve(tmp_path)


def test_streaming_runner_requires_target_col_declared(tmp_path: Path) -> None:
    semantics = build_target_semantics_metadata(build_default_target_semantics())
    _write_metadata(tmp_path, target_col="missing_target", target_semantics=semantics)
    _write_report(tmp_path)

    with pytest.raises(ValueError, match="target_col 'missing_target'"):
        _call_resolve(tmp_path)
