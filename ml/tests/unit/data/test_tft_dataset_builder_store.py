#!/usr/bin/env python3

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from ml.data.tft_dataset_builder import TFTDatasetBuilder


@pytest.mark.usefixtures("monkeypatch")
def test_prepare_training_data_from_store_uses_datastore(monkeypatch: pytest.MonkeyPatch) -> None:
    pl = pytest.importorskip("polars")

    start_dt = datetime(2024, 1, 1, tzinfo=UTC)
    end_dt = start_dt + timedelta(minutes=2)
    timestamps = [
        int(start_dt.timestamp() * 1_000_000_000),
        int((start_dt + timedelta(minutes=1)).timestamp() * 1_000_000_000),
    ]

    class _FeatureStore:
        def get_training_data(
            self,
            *,
            instrument_id: str,
            start: datetime,
            end: datetime,
            include_bars: bool = False,
        ) -> tuple[np.ndarray, list[int], list[str]]:
            features = np.array([[0.1], [0.2]], dtype=np.float32)
            return features, timestamps, ["feat_0"]

    class _Store:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def read_range(
            self,
            dataset_id: str,
            instrument_id: str,
            start_ns: int,
            end_ns: int,
        ) -> Any:
            self.calls.append((dataset_id, instrument_id))
            return pl.DataFrame(
                {
                    "instrument_id": [instrument_id, instrument_id],
                    "ts_event": timestamps,
                    "ts_init": timestamps,
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.5, 100.5],
                    "close": [100.5, 101.5],
                    "volume": [1000.0, 1100.0],
                },
            )

    # Provide a minimal catalog stub that should never be used
    class _Catalog:
        def bars(self, *args: object, **kwargs: object) -> Any:  # pragma: no cover - defensive
            raise AssertionError("Catalog fallback should not be invoked when DataStore is available")

    builder = TFTDatasetBuilder(
        catalog=_Catalog(),
        symbols=["SPY"],
        instrument_ids=["SPY.NYSE"],
        feature_store=_FeatureStore(),
        data_store=_Store(),
        market_dataset_id="EQUS.MINI",
        include_macro=False,
        include_micro=False,
        include_l2=False,
        include_events=False,
        include_calendar=False,
    )

    dataset = builder.prepare_training_data_from_store(
        instrument_ids=["SPY.NYSE"],
        start=start_dt,
        end=end_dt,
        horizon_minutes=1,
        min_return_threshold=0.0,
    )

    assert not dataset.is_empty()
    assert "instrument_id" in dataset.columns
    assert "feat_0" in dataset.columns
