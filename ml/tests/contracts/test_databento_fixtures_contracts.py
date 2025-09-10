from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest
from pandera.typing import Series

from ml.data.fixtures import make_tbbo_fixture, make_trades_fixture, make_mbp10_fixture
from ml.data.fixtures.manifest import compute_schema_hash


class TBBOSchema(pa.DataFrameModel):
    ts_event: Series[int] = pa.Field(ge=0)
    instrument_id: Series[str] = pa.Field()
    bid_px: Series[float] = pa.Field()
    bid_sz: Series[int] = pa.Field(ge=0)
    ask_px: Series[float] = pa.Field()
    ask_sz: Series[int] = pa.Field(ge=0)

    @pa.dataframe_check()
    def ts_monotonic(cls, df: pd.DataFrame) -> bool:  # noqa: N805 (pandera signature)
        return df["ts_event"].is_monotonic_increasing


class TradesSchema(pa.DataFrameModel):
    ts_event: Series[int] = pa.Field(ge=0)
    instrument_id: Series[str] = pa.Field()
    price: Series[float] = pa.Field()
    size: Series[int] = pa.Field(ge=1)
    side: Series[str] = pa.Field(isin=["buy", "sell"])

    @pa.dataframe_check()
    def ts_monotonic(cls, df: pd.DataFrame) -> bool:  # noqa: N805 (pandera signature)
        return df["ts_event"].is_monotonic_increasing


class MBP10Schema(pa.DataFrameModel):
    ts_event: Series[int] = pa.Field(ge=0)
    instrument_id: Series[str] = pa.Field()

    @pa.dataframe_check()
    def shape_and_monotonic(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        # 10 levels for both bid/ask px/sz
        for level in range(1, 11):
            for col in (f"bid_px_{level}", f"ask_px_{level}", f"bid_sz_{level}", f"ask_sz_{level}"):
                if col not in df.columns:
                    return False
        return df["ts_event"].is_monotonic_increasing


@pytest.mark.contracts
def test_tbbo_fixture_contract() -> None:
    df, manifest = make_tbbo_fixture()
    validated = TBBOSchema.validate(df)
    assert len(validated) == len(df)
    assert manifest.schema_hash == compute_schema_hash(df)
    # Schema stable over re-generation
    df2, man2 = make_tbbo_fixture()
    assert compute_schema_hash(df2) == manifest.schema_hash == man2.schema_hash
    assert manifest.content_sha256 == man2.content_sha256


@pytest.mark.contracts
def test_trades_fixture_contract() -> None:
    df, manifest = make_trades_fixture()
    validated = TradesSchema.validate(df)
    assert len(validated) == len(df)
    assert manifest.schema_hash == compute_schema_hash(df)


@pytest.mark.contracts
def test_mbp10_fixture_contract() -> None:
    df, manifest = make_mbp10_fixture()
    validated = MBP10Schema.validate(df)
    assert len(validated) == len(df)
    assert manifest.schema_hash == compute_schema_hash(df)


@pytest.mark.contracts
def test_idempotent_replay_deduplication() -> None:
    df, _ = make_trades_fixture(rows=50)
    # Simulate duplicates by appending the same rows
    doubled = pd.concat([df, df], ignore_index=True)
    # Deduplicate by (ts_event, price, size, side) for fixture
    deduped = doubled.drop_duplicates(subset=["ts_event", "price", "size", "side"], keep="first")
    assert len(deduped) == len(df)
