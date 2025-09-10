from __future__ import annotations

"""
Deterministic Databento-like fixtures (TBBO, MBP-10, Trades) with manifests.

These generators produce small, reproducible DataFrames suitable for unit and
property tests without external dependencies. They are not intended for model
training, only for adapter/contract validation and ingestion tests.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ml.data.fixtures.manifest import FixtureManifest
from ml.data.fixtures.manifest import compute_bytes_sha256
from ml.data.fixtures.manifest import compute_schema_hash


Kind = Literal["tbbo", "mbp10", "trades"]


@dataclass(slots=True)
class _FixtureSpec:
    kind: Kind
    instrument_id: str
    start_ns: int
    end_ns: int
    rows: int


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _ns_range(start_ns: int, end_ns: int, rows: int) -> npt.NDArray[np.int64]:
    step = max((end_ns - start_ns) // max(rows, 1), 1)
    return np.arange(start_ns, start_ns + step * rows, step, dtype=np.int64)


def _manifest(
    df: pd.DataFrame, *, dataset: str, instrument_id: str, start_ns: int, end_ns: int
) -> FixtureManifest:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return FixtureManifest(
        dataset=dataset,
        instrument_id=instrument_id,
        start_ns=int(start_ns),
        end_ns=int(end_ns),
        rows=len(df),
        schema_hash=compute_schema_hash(df),
        content_sha256=compute_bytes_sha256(csv_bytes),
    )


def make_tbbo_fixture(
    *,
    instrument_id: str = "EURUSD.SIM",
    start_ns: int = 1_600_000_000_000_000_000,
    end_ns: int = 1_600_000_000_060_000_000,
    rows: int = 60,
    seed: int = 42,
) -> tuple[pd.DataFrame, FixtureManifest]:
    """
    Create deterministic TBBO DataFrame with columns:
    ts_event, instrument_id, bid_px, bid_sz, ask_px, ask_sz
    """
    gen = _rng(seed)
    ts = _ns_range(start_ns, end_ns, rows)
    mid: npt.NDArray[np.float64] = 1.1000 + gen.normal(0.0, 0.0001, size=rows).cumsum()
    spread: npt.NDArray[np.float64] = np.full(rows, 0.00005)
    bid_px: npt.NDArray[np.float64] = mid - spread / 2
    ask_px: npt.NDArray[np.float64] = mid + spread / 2
    bid_sz: npt.NDArray[np.int64] = gen.integers(100, 300, size=rows, dtype=np.int64)
    ask_sz: npt.NDArray[np.int64] = gen.integers(100, 300, size=rows, dtype=np.int64)
    df = pd.DataFrame(
        {
            "ts_event": ts,
            "instrument_id": instrument_id,
            "bid_px": bid_px.astype("float64"),
            "bid_sz": bid_sz.astype("int64"),
            "ask_px": ask_px.astype("float64"),
            "ask_sz": ask_sz.astype("int64"),
        },
    )
    return df, _manifest(
        df, dataset="tbbo", instrument_id=instrument_id, start_ns=start_ns, end_ns=end_ns
    )


def make_trades_fixture(
    *,
    instrument_id: str = "EURUSD.SIM",
    start_ns: int = 1_600_000_000_000_000_000,
    end_ns: int = 1_600_000_000_060_000_000,
    rows: int = 120,
    seed: int = 1337,
) -> tuple[pd.DataFrame, FixtureManifest]:
    """
    Create deterministic trades DataFrame with columns:
    ts_event, instrument_id, price, size, side
    """
    gen = _rng(seed)
    ts: npt.NDArray[np.int64] = np.sort(
        _rng(seed + 1).integers(start_ns, end_ns, size=rows, dtype=np.int64)
    )
    price: npt.NDArray[np.float64] = 1.1000 + gen.normal(0.0, 0.0002, size=rows).cumsum()
    size: npt.NDArray[np.int64] = gen.integers(1, 10, size=rows, dtype=np.int64)
    side: npt.NDArray[np.object_] = gen.choice(["buy", "sell"], size=rows).astype(object)
    df = pd.DataFrame(
        {
            "ts_event": ts,
            "instrument_id": instrument_id,
            "price": price.astype("float64"),
            "size": size.astype("int64"),
            "side": side.astype("object"),
        },
    )
    return df, _manifest(
        df, dataset="trades", instrument_id=instrument_id, start_ns=start_ns, end_ns=end_ns
    )


def make_mbp10_fixture(
    *,
    instrument_id: str = "EURUSD.SIM",
    start_ns: int = 1_600_000_000_000_000_000,
    end_ns: int = 1_600_000_000_003_000_000,
    rows: int = 30,
    seed: int = 7,
) -> tuple[pd.DataFrame, FixtureManifest]:
    """
    Create deterministic MBP-10 snapshot DataFrame with columns:
    ts_event, instrument_id, bid_px_i, bid_sz_i, ask_px_i, ask_sz_i for i=1..10
    """
    gen = _rng(seed)
    ts = _ns_range(start_ns, end_ns, rows)
    mid: npt.NDArray[np.float64] = 1.1000 + gen.normal(0.0, 0.0001, size=rows).cumsum()
    tick = 0.00001
    data: dict[str, npt.NDArray[np.int64] | npt.NDArray[np.float64] | npt.NDArray[np.object_]] = {
        "ts_event": ts,
        "instrument_id": np.array([instrument_id] * rows, dtype=object),
    }
    for level in range(1, 11):
        spread_lv = tick * (2 * level - 1)
        data[f"bid_px_{level}"] = (mid - spread_lv).astype("float64")
        data[f"ask_px_{level}"] = (mid + spread_lv).astype("float64")
        data[f"bid_sz_{level}"] = gen.integers(50, 200, size=rows, dtype=np.int64)
        data[f"ask_sz_{level}"] = gen.integers(50, 200, size=rows, dtype=np.int64)
    df = pd.DataFrame(data)
    return df, _manifest(
        df, dataset="mbp10", instrument_id=instrument_id, start_ns=start_ns, end_ns=end_ns
    )


__all__: Final[list[str]] = [
    "make_mbp10_fixture",
    "make_tbbo_fixture",
    "make_trades_fixture",
]
