from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import cast

import pandas as pd
import pytest
from ml.tests.fixtures.pandera import DataFrame, Series, ensure_pandera_available

from ml.config.events import Source

pa = ensure_pandera_available()


class MLWatermarkProgressSchema(pa.DataFrameModel):
    """
    Schema for event-driven watermark progression at the dataset boundary.

    Validates non-decreasing watermark_ts per (dataset_id, instrument_id, source) in
    update_ts order.

    """

    dataset_id: Series[str] = pa.Field(nullable=False)
    instrument_id: Series[str] = pa.Field(nullable=False)
    source: Series[str] = pa.Field(nullable=False, isin=[s.value for s in Source])
    watermark_ts: Series[int] = pa.Field(nullable=False, ge=0)
    update_ts: Series[int] = pa.Field(nullable=False, ge=0)

    @pa.check("instrument_id", name="instrument_id_format")
    def check_instrument_id_format(cls, s: Series[str]) -> Series[bool]:  # noqa: N805
        return cast(Series[bool], s.str.match(r"^[A-Z0-9]+\.[A-Z]+$"))

    @pa.dataframe_check()
    def check_monotonic_by_key(
        cls,
        df: DataFrame[Any],
    ) -> Series[bool]:  # noqa: N805 - pandera signature
        try:
            for key, group in df.groupby(["dataset_id", "instrument_id", "source"]):
                ordered = group.sort_values("update_ts")
                values = ordered["watermark_ts"].to_numpy()
                if any(values[i] > values[i + 1] for i in range(len(values) - 1)):
                    return cast(Series[bool], pd.Series([False] * len(df)))
            return cast(Series[bool], pd.Series([True] * len(df)))
        except Exception:
            return cast(Series[bool], pd.Series([False] * len(df)))


@pytest.mark.contracts
def test_watermark_progression_valid() -> None:
    df = pd.DataFrame(
        {
            "dataset_id": ["features", "features", "features"],
            "instrument_id": ["EURUSD.SIM", "EURUSD.SIM", "EURUSD.SIM"],
            "source": [Source.LIVE.value, Source.LIVE.value, Source.LIVE.value],
            "watermark_ts": [100, 200, 200],
            "update_ts": [int(datetime(2024, 1, 1).timestamp() * 1e9) + i for i in (1, 2, 3)],
        },
    )
    validated = MLWatermarkProgressSchema.validate(df)
    assert len(validated) == 3


@pytest.mark.contracts
def test_watermark_progression_regression_fails() -> None:
    df = pd.DataFrame(
        {
            "dataset_id": ["features", "features"],
            "instrument_id": ["EURUSD.SIM", "EURUSD.SIM"],
            "source": [Source.HISTORICAL.value, Source.HISTORICAL.value],
            "watermark_ts": [200, 150],  # regression
            "update_ts": [
                int(datetime(2024, 1, 1).timestamp() * 1e9),
                int(datetime(2024, 1, 1, 1).timestamp() * 1e9),
            ],
        },
    )
    with pytest.raises(pa.errors.SchemaError):
        MLWatermarkProgressSchema.validate(df)
