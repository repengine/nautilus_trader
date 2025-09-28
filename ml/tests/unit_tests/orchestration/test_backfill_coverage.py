from typing import Any

import pytest

from ml.config.coverage import CoveragePolicy
from ml.data.ingest.orchestrator import BackfillWindowList
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator


class _DummyCoverage:
    def read_bucket_coverage(self, **kwargs: Any) -> set[int]:  # pragma: no cover - not exercised
        return set()


class _DummyWriter:
    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    def write(
        self, *, dataset_id: str, schema: str, instrument_id: str, df: Any
    ) -> int:  # pragma: no cover - not exercised
        self.writes.append(
            {"dataset_id": dataset_id, "schema": schema, "instrument_id": instrument_id}
        )
        return 0


def test_backfill_coverage_uses_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeIngestion:
        def __init__(self, **_: Any) -> None:
            pass

        def backfill_gaps(
            self, *, dataset_id: str, schema: str, instrument_id: str, lookback_days: int
        ) -> BackfillWindowList:
            captured.update(
                {
                    "dataset_id": dataset_id,
                    "schema": schema,
                    "instrument_id": instrument_id,
                    "lookback_days": lookback_days,
                },
            )
            return BackfillWindowList((), requested=())

    monkeypatch.setattr(
        "ml.orchestration.pipeline_orchestrator.IngestionOrchestrator",
        _FakeIngestion,
    )

    orchestrator = MLPipelineOrchestrator(
        coverage=_DummyCoverage(),
        writer=_DummyWriter(),
        registry=object(),
        ingestor=object(),
        build_main=lambda _args=None: 0,
        hpo_main=None,
        teacher_main=lambda _args=None: 0,
    )

    policy = CoveragePolicy(
        l0_max_lookback_days=10,
        l1_max_lookback_days=20,
        l2_max_lookback_days=30,
        l3_max_lookback_days=40,
    )
    orchestrator.backfill_coverage(
        dataset_id="bars", schema="ohlcv", instrument_id="SPY.EQUS", policy=policy
    )
    assert captured["lookback_days"] == 10

    orchestrator.backfill_coverage(
        dataset_id="quotes", schema="tick", instrument_id="SPY.EQUS", policy=policy
    )
    assert captured["lookback_days"] == 20

    orchestrator.backfill_coverage(
        dataset_id="mbp1", schema="book", instrument_id="SPY.EQUS", policy=policy
    )
    assert captured["lookback_days"] == 30
