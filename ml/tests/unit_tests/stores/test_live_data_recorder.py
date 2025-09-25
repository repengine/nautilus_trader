import asyncio
from typing import Any

import pytest

from ml.stores.writers import LiveDataRecorder


class _FakePrice:
    def __init__(self, v: float) -> None:
        self._v = float(v)

    def as_double(self) -> float:
        return self._v


class _FakeBarType:
    def __init__(self, instrument_id: str) -> None:
        self.instrument_id = instrument_id


class _FakeBar:
    def __init__(
        self,
        instrument_id: str,
        ts: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
    ) -> None:
        self.bar_type = _FakeBarType(instrument_id)
        self.ts_event = ts
        self.ts_init = ts
        self.open = _FakePrice(open_price)
        self.high = _FakePrice(high_price)
        self.low = _FakePrice(low_price)
        self.close = _FakePrice(close_price)
        self.volume = _FakePrice(volume)


class _FakeStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def write_ingestion(
        self,
        *,
        dataset_id: str,
        records: list[dict[str, Any]] | Any,
        source: str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> Any:
        self.calls.append(
            {
                "dataset_id": dataset_id,
                "records": records,
                "source": source,
                "run_id": run_id,
                "instrument_id": instrument_id,
            }
        )
        return object()


class _FakeRegistry:
    pass


@pytest.mark.asyncio
async def test_live_data_recorder_persists_bars_via_datastore() -> None:
    store = _FakeStore()
    registry = _FakeRegistry()

    recorder = LiveDataRecorder(
        data_store=store,  # type: ignore[arg-type]
        data_registry=registry,  # type: ignore[arg-type]
        buffer_size=1000,
        flush_interval_ms=10_000,
    )

    # Feed two bars for the same instrument
    recorder.on_bar(_FakeBar("SPY.EQUS", 1_000_000_000, 1.0, 2.0, 0.5, 1.5, 100))
    recorder.on_bar(_FakeBar("SPY.EQUS", 2_000_000_000, 1.5, 2.5, 1.0, 2.0, 200))

    # Explicitly flush
    await recorder.flush_all()

    # Verify DataStore.write_ingestion was called once per instrument
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["dataset_id"] == "bars"
    assert call["instrument_id"] == "SPY.EQUS"
    records = call["records"]
    assert isinstance(records, list)
    assert len(records) == 2
    assert records[0]["close"] == 1.5
    assert records[1]["close"] == 2.0


class _FakeQuoteTick:
    def __init__(self, instrument_id: str, ts: int, bid: float, ask: float, bsz: float, asz: float) -> None:
        self.instrument_id = instrument_id
        self.ts_event = ts
        self.ts_init = ts
        self.bid = _FakePrice(bid)
        self.ask = _FakePrice(ask)
        self.bid_size = _FakePrice(bsz)
        self.ask_size = _FakePrice(asz)


class _FakeTradeTick:
    def __init__(self, instrument_id: str, ts: int, price: float, size: float) -> None:
        self.instrument_id = instrument_id
        self.ts_event = ts
        self.ts_init = ts
        self.price = _FakePrice(price)
        self.size = _FakePrice(size)


@pytest.mark.asyncio
async def test_live_data_recorder_persists_quotes_and_trades_via_datastore() -> None:
    store = _FakeStore()
    registry = _FakeRegistry()

    recorder = LiveDataRecorder(
        data_store=store,  # type: ignore[arg-type]
        data_registry=registry,  # type: ignore[arg-type]
        buffer_size=1000,
        flush_interval_ms=10_000,
    )

    recorder.on_quote(_FakeQuoteTick("SPY.EQUS", 1_000_000_000, 1.0, 1.1, 100, 200))
    recorder.on_trade(_FakeTradeTick("SPY.EQUS", 1_000_001_000, 1.05, 10))

    await recorder.flush_all()

    # Two write_ingestion calls expected: one for quotes, one for trades
    assert any(call["dataset_id"] == "quotes" for call in store.calls)
    assert any(call["dataset_id"] == "trades" for call in store.calls)
