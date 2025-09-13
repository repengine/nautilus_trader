#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ml.orchestration.pipeline_orchestrator import (
    DatasetBuildConfig,
    HPOConfig,
    MLPipelineOrchestrator,
    OrchestratorConfig,
    TeacherTrainConfig,
)


@dataclass
class _Coverage:
    def read_bucket_coverage(
        self, *, dataset_id: str, schema: str, instrument_id: str, start_ns: int, end_ns: int
    ) -> set[int]:
        return set()


@dataclass
class _Writer:
    def write(self, *, dataset_id: str, schema: str, instrument_id: str, df: pd.DataFrame) -> int:
        return len(df.index) if df is not None and not df.empty else 0


@dataclass
class _Registry:
    def emit_event(self, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None

    def update_watermark(self, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None


@dataclass
class _Ingestor:
    def ingest_time_window(self, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"ts_event": [1, 2]})


def _ok(_: list[str] | None = None) -> int:
    return 0


def test_pipeline_orchestrator_runs_all_phases(tmp_path: Path) -> None:
    coverage = _Coverage()
    writer = _Writer()
    registry = _Registry()
    ingestor = _Ingestor()

    # record calls
    called: dict[str, int] = {"build": 0, "hpo": 0, "train": 0}

    def _build(argv: list[str] | None = None) -> int:
        called["build"] += 1
        # simulate dataset.csv emitted
        out_dir = None
        if argv is not None and "--out_dir" in argv:
            out_dir = Path(argv[argv.index("--out_dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dataset.csv").write_text("id,ts_event\n1,1\n")
        return 0

    def _hpo(argv: list[str] | None = None) -> int:
        called["hpo"] += 1
        return 0

    def _train(argv: list[str] | None = None) -> int:
        called["train"] += 1
        return 0

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        registry=registry,
        ingestor=ingestor,
        build_main=_build,
        hpo_main=_hpo,
        teacher_main=_train,
    )

    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path), symbols="SPY.NYSE", out_dir=str(tmp_path / "out"), include_micro=True
        ),
        hpo=HPOConfig(enabled=True, epochs=1, batch_size=8, tail_rows=100, limit_groups=10),
        teacher=TeacherTrainConfig(enabled=True, model_id="teacher_X", max_epochs=1),
    )
    rc = orch.run(cfg)
    assert rc == 0
    assert called == {"build": 1, "hpo": 1, "train": 1}

