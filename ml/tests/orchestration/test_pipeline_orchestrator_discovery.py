from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ml.config.market_data import MarketDatasetInput
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator


class _CoverageStub:
    def read_bucket_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        return set()


class _WriterStub:
    def write(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        df: pd.DataFrame,
    ) -> int:
        return len(df)


class _DiscoveryStub:
    def __init__(self, mapping: dict[str, MarketDatasetInput]) -> None:
        self._mapping = mapping

    def discover(
        self,
        *,
        requests: Sequence[object],
        dataset_hint: str | None = None,
    ) -> tuple[MarketDatasetInput, ...]:
        return tuple(self._mapping[getattr(request, "symbol")] for request in requests)


def test_prepare_dataset_config_applies_discovery() -> None:
    coverage = _CoverageStub()
    writer = _WriterStub()
    discovery = _DiscoveryStub(
        {
            "INTC": MarketDatasetInput(
                dataset_id="XNAS.ITCH",
                symbols=("INTC",),
                schema_override="ohlcv-1m",
            ),
            "INTC.XNAS": MarketDatasetInput(
                dataset_id="XNAS.ITCH",
                symbols=("INTC",),
                schema_override="ohlcv-1m",
            ),
        }
    )
    orchestrator = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        build_main=lambda argv=None: 0,
        teacher_main=lambda argv=None: 0,
        dataset_discovery=discovery,  # type: ignore[arg-type]
    )
    cfg = DatasetBuildConfig(
        data_dir="data",
        symbols="INTC.XNAS",
        out_dir="out",
    )
    prepared = orchestrator._prepare_dataset_config(cfg)
    assert prepared.market_inputs is not None
    assert prepared.market_inputs[0].dataset_id == "XNAS.ITCH"
    assert prepared.instrument_ids is not None
    assert "INTC.XNAS" in prepared.instrument_ids
