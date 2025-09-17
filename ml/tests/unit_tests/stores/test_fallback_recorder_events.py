from pathlib import Path
from typing import Any

import pytest

from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType, PersistenceConfig
from ml.stores.protocols import MarketDataWriterProtocol
from ml.stores.writers import CatalogWriteFacade, LiveDataRecorder


class _StubWriter(MarketDataWriterProtocol):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def write(self, *, dataset_id: str, schema: str, instrument_id: str, df: Any) -> int:  # type: ignore[override]
        self.calls.append({"dataset_id": dataset_id, "schema": schema, "instrument_id": instrument_id, "rows": getattr(df, "shape", (len(df), 0))[0] if hasattr(df, "shape") else len(df)})
        return getattr(df, "shape", (len(df), 0))[0] if hasattr(df, "shape") else len(df)


class _FakePrice:
    def __init__(self, v: float) -> None:
        self._v = float(v)

    def as_double(self) -> float:
        return self._v


class _FakeBarType:
    def __init__(self, instrument_id: str) -> None:
        self.instrument_id = instrument_id


class _FakeBar:
    def __init__(self, instrument_id: str, ts: int, close: float) -> None:
        self.bar_type = _FakeBarType(instrument_id)
        self.ts_event = ts
        self.ts_init = ts
        self.open = _FakePrice(close)
        self.high = _FakePrice(close)
        self.low = _FakePrice(close)
        self.close = _FakePrice(close)
        self.volume = _FakePrice(0.0)


@pytest.mark.asyncio
async def test_fallback_recorder_emits_events_with_json_registry(tmp_path: Path) -> None:
    # JSON registry
    persistence = PersistenceConfig(backend=BackendType.JSON, json_path=tmp_path)
    registry = DataRegistry(registry_path=tmp_path, persistence_config=persistence)

    # Catalog facade using stub writer
    stub_writer = _StubWriter()
    facade = CatalogWriteFacade(stub_writer)

    # Recorder
    recorder = LiveDataRecorder(
        data_store=facade,  # type: ignore[arg-type]
        data_registry=registry,  # type: ignore[arg-type]
        buffer_size=1000,
        flush_interval_ms=1000,
    )

    # Feed bars and flush
    recorder.on_bar(_FakeBar("SPY.EQUS", 1_000_000, 1.23))
    await recorder.flush_all()

    # Verify writer called
    assert len(stub_writer.calls) == 1
    assert stub_writer.calls[0]["dataset_id"] == "bars"

    # Verify JSON registry recorded at least one event
    # Access internal event list (test-only)
    assert len(registry._events) >= 1  # type: ignore[attr-defined]

