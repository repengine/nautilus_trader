"""
Utilities to adapt Nautilus data objects to pandas DataFrames for ingestion.

Focused on Bars (OHLCV) where the canonical SQL writer expects columns: `ts_event, open,
high, low, close, volume`.

"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd


if TYPE_CHECKING:  # Import for type checking only
    from nautilus_trader.model.data import Bar as NautilusBar  # pragma: no cover
else:  # At runtime, avoid hard dependency for tooling
    NautilusBar = Any


@dataclass(slots=True)
class NautilusBarsToDataFrame:
    """
    Convert a sequence of Nautilus `Bar` objects to a pandas DataFrame with canonical
    columns for the SQL writer.
    """

    def to_df(self, bars: Iterable[NautilusBar]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for bar in bars:
            # Use to_dict() to avoid relying on Cython price types; numeric fields are strings
            d = bar.to_dict()
            rows.append(
                {
                    "ts_event": int(d.get("ts_event")),
                    "open": float(d.get("open")) if d.get("open") is not None else None,
                    "high": float(d.get("high")) if d.get("high") is not None else None,
                    "low": float(d.get("low")) if d.get("low") is not None else None,
                    "close": float(d.get("close")) if d.get("close") is not None else None,
                    "volume": float(d.get("volume")) if d.get("volume") is not None else None,
                },
            )
        if not rows:
            return pd.DataFrame(
                columns=["ts_event", "open", "high", "low", "close", "volume"],
            ).astype(
                {"ts_event": "int64"},
                errors="ignore",
            )
        df = pd.DataFrame.from_records(rows)
        # Ensure dtype for ts_event
        if "ts_event" in df:
            df["ts_event"] = df["ts_event"].astype("int64", copy=False)
        return df


def to_df_bars(bars: Iterable[NautilusBar]) -> pd.DataFrame:
    return NautilusBarsToDataFrame().to_df(bars)
