from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Mapping
from typing import Sequence

from ml.data.ingest.symbology import SymbolResolution
from ml.data.ingest.symbology import SymbologyResolutionError

import pytest

from ml.data.ingest.discovery import DatasetDiscoveryError
from ml.data.ingest.discovery import DatasetDiscoveryService
from ml.data.ingest.discovery import DiscoveryPolicy
from ml.data.ingest.discovery import DiscoveryRequest


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


class _FakeMetadataClient:
    def __init__(self) -> None:
        self.datasets = ("XNAS.ITCH",)
        base_start = datetime(2018, 1, 1, tzinfo=UTC)
        base_end = datetime(2030, 1, 1, tzinfo=UTC)
        self.ranges = {
            "XNAS.ITCH": {
                "start": _iso(base_start),
                "end": _iso(base_end),
                "schema": {
                    "ohlcv-1m": {
                        "start": _iso(base_start),
                        "end": _iso(base_end),
                    },
                },
            },
        }
        self.schemas = {"XNAS.ITCH": ("ohlcv-1m",)}
        self.costs: dict[tuple[str, str], float] = {
            ("XNAS.ITCH", "INTC"): 0.0,
        }
        self.cost_calls: list[tuple[str, str]] = []

    def list_datasets(self) -> Sequence[str]:
        return self.datasets

    def list_schemas(self, *, dataset: str) -> Sequence[str]:
        return self.schemas[dataset]

    def get_dataset_range(self, dataset: str) -> dict[str, object]:
        return self.ranges[dataset]

    def get_cost(
        self,
        *,
        dataset: str,
        symbols: Sequence[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
    ) -> float:
        symbol = symbols[0]
        key = (dataset, symbol)
        self.cost_calls.append(key)
        if key not in self.costs:
            raise ValueError(f"unsupported symbol {symbol}")
        return self.costs[key]


class _StubResolver:
    def __init__(self, mapping: Mapping[str, SymbolResolution]) -> None:
        self._mapping = mapping

    def resolve(
        self,
        *,
        dataset: str,
        symbol: str,
        schema: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> SymbolResolution:
        del schema, start, end
        try:
            resolution = self._mapping[dataset]
        except KeyError as exc:
            raise SymbologyResolutionError(f"unsupported dataset {dataset}") from exc
        return resolution


def test_discover_selects_lowest_cost_dataset() -> None:
    metadata = _FakeMetadataClient()
    policy = DiscoveryPolicy.from_env({})
    resolver = _StubResolver(
        {
            "XNAS.ITCH": SymbolResolution(
                original="INTC.XNAS",
                dataset="XNAS.ITCH",
                schema="ohlcv-1m",
                candidates=("INTC",),
                preferred="INTC",
                instrument_id="4182",
            ),
        },
    )
    service = DatasetDiscoveryService(metadata=metadata, policy=policy, resolver=resolver)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=7)
    requests = (
        DiscoveryRequest(symbol="INTC.XNAS", schema="ohlcv-1m", start=start, end=end),
    )
    inputs = service.discover(requests=requests)
    assert len(inputs) == 1
    assert inputs[0].dataset_id == "XNAS.ITCH"
    assert metadata.cost_calls == [("XNAS.ITCH", "INTC")]


def test_discover_one_returns_symbol_resolution() -> None:
    metadata = _FakeMetadataClient()
    policy = DiscoveryPolicy.from_env({})
    resolver = _StubResolver(
        {
            "XNAS.ITCH": SymbolResolution(
                original="INTC.XNAS",
                dataset="XNAS.ITCH",
                schema="ohlcv-1m",
                candidates=("INTC",),
                preferred="INTC",
                instrument_id="4182",
            ),
        },
    )
    service = DatasetDiscoveryService(metadata=metadata, policy=policy, resolver=resolver)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=7)
    request = DiscoveryRequest(symbol="INTC.XNAS", schema="ohlcv-1m", start=start, end=end)
    discovered = service.discover_one(request=request)
    assert discovered.dataset_id == "XNAS.ITCH"
    assert discovered.symbol == "INTC"
    assert discovered.requested_symbol == "INTC.XNAS"
    market_input = discovered.to_market_input()
    assert market_input.symbols == ("INTC",)


def test_discover_respects_cost_limit() -> None:
    metadata = _FakeMetadataClient()
    policy = DiscoveryPolicy.from_env({"DATABENTO_DISCOVERY_MAX_COST_USD": "1.0"})
    resolver = _StubResolver(
        {
            "XNAS.ITCH": SymbolResolution(
                original="INTC.XNAS",
                dataset="XNAS.ITCH",
                schema="ohlcv-1m",
                candidates=("INTC",),
                preferred="INTC",
                instrument_id="4182",
            ),
        },
    )
    service = DatasetDiscoveryService(metadata=metadata, policy=policy, resolver=resolver)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=7)
    requests = (
        DiscoveryRequest(symbol="INTC.XNAS", schema="ohlcv-1m", start=start, end=end),
    )
    inputs = service.discover(requests=requests)
    assert inputs[0].dataset_id == "XNAS.ITCH"
    assert ("XNAS.ITCH", "INTC") in metadata.cost_calls


def test_discover_raises_when_no_schema_matches() -> None:
    metadata = _FakeMetadataClient()
    metadata.schemas = {"XNAS.ITCH": ("tbbo",)}
    policy = DiscoveryPolicy.from_env({})
    resolver = _StubResolver(
        {
            "XNAS.ITCH": SymbolResolution(
                original="INTC.XNAS",
                dataset="XNAS.ITCH",
                schema="ohlcv-1m",
                candidates=("INTC",),
                preferred="INTC",
                instrument_id="4182",
            ),
        },
    )
    service = DatasetDiscoveryService(metadata=metadata, policy=policy, resolver=resolver)
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=7)
    requests = (
        DiscoveryRequest(symbol="INTC.XNAS", schema="ohlcv-1m", start=start, end=end),
    )
    with pytest.raises(DatasetDiscoveryError):
        service.discover(requests=requests)


def test_discover_skips_zero_coverage_dataset() -> None:
    metadata = _FakeMetadataClient()
    policy = DiscoveryPolicy.from_env({"DATABENTO_DISCOVERY_DATASETS": "XNAS.ITCH,EQUS.MINI"})
    metadata.datasets = ("XNAS.ITCH", "EQUS.MINI")
    metadata.schemas["EQUS.MINI"] = ("ohlcv-1m",)
    base_start = datetime(2018, 1, 1, tzinfo=UTC)
    base_end = datetime(2030, 1, 1, tzinfo=UTC)
    metadata.ranges["EQUS.MINI"] = {
        "start": _iso(base_start),
        "end": _iso(base_end),
        "schema": {
            "ohlcv-1m": {
                "start": _iso(base_start),
                "end": _iso(base_end),
            },
        },
    }
    now = datetime(2025, 1, 1, tzinfo=UTC)
    zero_start = now.isoformat().replace("+00:00", "Z")
    metadata.ranges["XNAS.ITCH"] = {
        "start": zero_start,
        "end": zero_start,
        "schema": {"ohlcv-1m": {"start": zero_start, "end": zero_start}},
    }
    metadata.costs[("EQUS.MINI", "INTC")] = 0.0
    resolver = _StubResolver(
        {
            "EQUS.MINI": SymbolResolution(
                original="INTC.XNAS",
                dataset="EQUS.MINI",
                schema="ohlcv-1m",
                candidates=("INTC",),
                preferred="INTC",
                instrument_id="100",
            ),
            "XNAS.ITCH": SymbolResolution(
                original="INTC.XNAS",
                dataset="XNAS.ITCH",
                schema="ohlcv-1m",
                candidates=("INTC",),
                preferred="INTC",
                instrument_id="4182",
            ),
        },
    )
    service = DatasetDiscoveryService(metadata=metadata, policy=policy, resolver=resolver)
    end = now
    start = end - timedelta(days=7)
    request = DiscoveryRequest(symbol="INTC.XNAS", schema="ohlcv-1m", start=start, end=end)
    discovered = service.discover_one(request=request)
    assert discovered.dataset_id == "EQUS.MINI"
