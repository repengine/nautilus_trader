from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.data.ingest.nautilus_adapters import NautilusBarsToDataFrame
from ml.data.ingest.nautilus_adapters import to_df_bars


@dataclass
class _FakeBar:
    ts_event: int
    open: str | None
    high: str | None
    low: str | None
    close: str | None
    volume: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "ts_event": self.ts_event,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


def test_nautilus_bars_to_df_handles_empty_iterable() -> None:
    adapter = NautilusBarsToDataFrame()

    df = adapter.to_df([])

    assert df.empty
    assert list(df.columns) == ["ts_event", "open", "high", "low", "close", "volume"]
    assert str(df["ts_event"].dtype) == "int64"


def test_nautilus_bars_to_df_coerces_numeric_fields() -> None:
    adapter = NautilusBarsToDataFrame()
    bars = [
        _FakeBar(100, "1.0", "2.0", "0.5", "1.5", "10"),
        _FakeBar(200, None, "2.5", "1.0", None, None),
    ]

    df = adapter.to_df(bars)

    assert df["ts_event"].tolist() == [100, 200]
    assert df["open"].tolist()[0] == 1.0
    assert df["high"].tolist()[0] == 2.0
    assert df["low"].tolist()[0] == 0.5
    assert df["close"].tolist()[0] == 1.5
    assert df["volume"].tolist()[0] == 10.0
    assert pd.isna(df.loc[1, "open"])
    assert pd.isna(df.loc[1, "close"])
    assert pd.isna(df.loc[1, "volume"])


def test_to_df_bars_delegates_to_adapter() -> None:
    bars = [_FakeBar(123, "1.0", "1.5", "0.5", "1.2", "4")]

    direct = NautilusBarsToDataFrame().to_df(bars)
    delegated = to_df_bars(bars)

    pd.testing.assert_frame_equal(direct, delegated)
