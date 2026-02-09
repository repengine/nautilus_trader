from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import Any, cast

import pytest

from ml.data.ingest.discovery import DatasetDiscoveryError
from ml.data.ingest.discovery import DatasetDiscoveryService
from ml.data.ingest.discovery import DiscoveryPolicy
from ml.data.ingest.discovery import DiscoveryRequest
from ml.data.ingest.discovery import _datetime_to_ns
from ml.data.ingest.discovery import _iso_to_ns
from ml.data.ingest.discovery import _ns_to_iso
from ml.data.ingest.symbology import SymbolResolution
from ml.data.ingest.symbology import SymbologyResolutionError
from ml.registry.dataclasses import StorageKind


class _CoverageStub:
    def __init__(self, *, invert: bool = False) -> None:
        self.invert = invert

    def clamp_range(
        self,
        start: datetime,
        end: datetime,
        *,
        dataset: str | None = None,
        schema: str | None = None,
    ) -> tuple[datetime, datetime]:
        del dataset, schema
        if self.invert:
            return end, start
        return start, end


class _ResolverStub:
    def __init__(self, *, raise_error: bool = False) -> None:
        self.raise_error = raise_error
        self.calls: list[tuple[str, str]] = []

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
        self.calls.append((dataset, symbol))
        if self.raise_error:
            raise SymbologyResolutionError("resolver rejected symbol")
        preferred = symbol.strip().upper()
        return SymbolResolution(
            original=symbol.strip().upper(),
            dataset=dataset,
            schema="trades",
            candidates=(preferred, f"{preferred}.ALT"),
            preferred=preferred,
            instrument_id="1234",
        )


class _MetadataStub:
    def __init__(
        self,
        *,
        datasets: Any,
        schemas: dict[str, Any] | None = None,
        ranges: dict[str, dict[str, Any]] | None = None,
        costs: dict[tuple[str, str], float] | None = None,
        failing_symbols: set[str] | None = None,
    ) -> None:
        self._datasets = datasets
        self._schemas = schemas or {}
        self._ranges = ranges or {}
        self._costs = costs or {}
        self._failing_symbols = failing_symbols or set()

    def list_datasets(self) -> Any:
        return self._datasets

    def list_schemas(self, *, dataset: str) -> Any:
        value = self._schemas.get(dataset, ())
        if isinstance(value, Exception):
            raise value
        return value

    def get_dataset_range(self, dataset: str) -> dict[str, Any]:
        return self._ranges[dataset]

    def get_cost(
        self,
        *,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: str,
        end: str,
    ) -> float:
        del schema, start, end
        symbol = symbols[0]
        if symbol in self._failing_symbols:
            raise RuntimeError("cost lookup failed")
        return self._costs.get((dataset, symbol), 0.0)


@dataclass
class _JsonListingObject:
    payload: str

    def __str__(self) -> str:
        return self.payload


def _build_policy(
    *,
    max_cost_usd: float | None = None,
    max_candidates: int | None = None,
    allowlist: tuple[str, ...] | None = ("XNAS.ITCH",),
    denylist: frozenset[str] | None = None,
    invert_window: bool = False,
) -> DiscoveryPolicy:
    return DiscoveryPolicy(
        coverage=cast(Any, _CoverageStub(invert=invert_window)),
        dataset_allowlist=allowlist,
        dataset_denylist=denylist,
        max_cost_usd=max_cost_usd,
        max_candidates=max_candidates,
    )


def test_discovery_policy_from_env_and_candidate_filtering() -> None:
    env = {
        "DATABENTO_DISCOVERY_DATASETS": "XNAS.ITCH,EQUS.MINI",
        "DATABENTO_DISCOVERY_DENYLIST": "EQUS.MINI",
        "DATABENTO_DISCOVERY_MAX_COST_USD": "2.5",
        "DATABENTO_DISCOVERY_MAX_CANDIDATES": "1",
    }
    policy = DiscoveryPolicy.from_env(env)

    assert policy.dataset_allowlist == ("XNAS.ITCH", "EQUS.MINI")
    assert policy.dataset_denylist == frozenset({"EQUS.MINI"})
    assert policy.max_cost_usd == pytest.approx(2.5)
    assert policy.max_candidates == 1
    assert tuple(policy.candidates(("EQUS.MINI", "XNAS.ITCH", "GLBX.MDP3"))) == ("XNAS.ITCH",)
    assert policy.cost_allowed(2.0)
    assert not policy.cost_allowed(3.0)

    no_allowlist = _build_policy(allowlist=None, max_candidates=None)
    assert tuple(no_allowlist.candidates(("A", "B"))) == ("A", "B")


def test_dataset_discovery_service_requires_resolver() -> None:
    metadata = _MetadataStub(datasets=("XNAS.ITCH",))
    with pytest.raises(ValueError):
        DatasetDiscoveryService(
            metadata=cast(Any, metadata),
            policy=_build_policy(),
            resolver=None,
        )


def test_discover_and_discover_one_return_market_inputs() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 2, tzinfo=UTC)
    metadata = _MetadataStub(
        datasets=("XNAS.ITCH", "EQUS.MINI"),
        schemas={"XNAS.ITCH": ("trades",), "EQUS.MINI": ("trades",)},
        ranges={
            "XNAS.ITCH": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2026-01-01T00:00:00Z",
                "schema": {"trades": {"start": "2024-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"}},
            },
            "EQUS.MINI": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2026-01-01T00:00:00Z",
                "schema": {"trades": {"start": "2024-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"}},
            },
        },
        costs={("XNAS.ITCH", "AAPL"): 1.2, ("EQUS.MINI", "AAPL"): 0.0},
    )
    resolver = _ResolverStub()
    service = DatasetDiscoveryService(
        metadata=cast(Any, metadata),
        policy=_build_policy(allowlist=None),
        resolver=cast(Any, resolver),
    )
    request = DiscoveryRequest(symbol="aapl", schema="trades", start=start, end=end)

    discovered_one = service.discover_one(request=request, dataset_hint=" EQUS.MINI ")
    assert discovered_one.dataset_id == "EQUS.MINI"
    assert discovered_one.symbol == "AAPL"
    assert discovered_one.storage_kind == StorageKind.POSTGRES

    discovered_many = service.discover(requests=(request,), dataset_hint="EQUS.MINI")
    assert len(discovered_many) == 1
    assert discovered_many[0].dataset_id == "EQUS.MINI"
    assert discovered_many[0].symbols == ("AAPL",)
    assert service.discover(requests=()) == ()


def test_discovery_rejects_empty_window_and_cost_and_symbology_errors() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 2, tzinfo=UTC)
    metadata = _MetadataStub(
        datasets=("", "XNAS.ITCH"),
        schemas={"XNAS.ITCH": ("trades",)},
        ranges={
            "XNAS.ITCH": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2026-01-01T00:00:00Z",
                "schema": {"trades": {"start": "2024-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"}},
            },
        },
        costs={("XNAS.ITCH", "MSFT"): 5.0},
    )
    request = DiscoveryRequest(symbol="msft", schema="trades", start=start, end=end)

    inverted = DatasetDiscoveryService(
        metadata=cast(Any, metadata),
        policy=_build_policy(invert_window=True),
        resolver=cast(Any, _ResolverStub()),
    )
    with pytest.raises(DatasetDiscoveryError):
        inverted.discover_one(request=request)

    cost_limited = DatasetDiscoveryService(
        metadata=cast(Any, metadata),
        policy=_build_policy(max_cost_usd=0.5),
        resolver=cast(Any, _ResolverStub()),
    )
    with pytest.raises(DatasetDiscoveryError):
        cost_limited.discover_one(request=request)

    resolver_error = DatasetDiscoveryService(
        metadata=cast(Any, metadata),
        policy=_build_policy(),
        resolver=cast(Any, _ResolverStub(raise_error=True)),
    )
    with pytest.raises(DatasetDiscoveryError):
        resolver_error.discover_one(request=request)


def test_list_dataset_and_schema_parsing_variants_and_failures() -> None:
    base_ranges = {
        "XNAS.ITCH": {"start": "2024-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z", "schema": {}},
    }

    service_from_lines = DatasetDiscoveryService(
        metadata=cast(
            Any,
            _MetadataStub(
                datasets="XNAS.ITCH\nEQUS.MINI\n",
                schemas={"XNAS.ITCH": json.dumps(["trades"])},
                ranges=base_ranges,
            ),
        ),
        policy=_build_policy(),
        resolver=cast(Any, _ResolverStub()),
    )
    assert service_from_lines._list_datasets() == ("XNAS.ITCH", "EQUS.MINI")
    assert service_from_lines._list_schemas("XNAS.ITCH") == frozenset({"trades"})

    service_from_json = DatasetDiscoveryService(
        metadata=cast(
            Any,
            _MetadataStub(
                datasets=_JsonListingObject('["XNAS.ITCH", "EQUS.MINI"]'),
                schemas={"XNAS.ITCH": "not-json"},
                ranges=base_ranges,
            ),
        ),
        policy=_build_policy(),
        resolver=cast(Any, _ResolverStub()),
    )
    assert service_from_json._list_datasets() == ("XNAS.ITCH", "EQUS.MINI")
    assert service_from_json._list_schemas("XNAS.ITCH") == frozenset()

    bad_listing_service = DatasetDiscoveryService(
        metadata=cast(
            Any,
            _MetadataStub(
                datasets=object(),
                schemas={"XNAS.ITCH": RuntimeError("boom")},
                ranges=base_ranges,
            ),
        ),
        policy=_build_policy(),
        resolver=cast(Any, _ResolverStub()),
    )
    with pytest.raises(Exception):
        _ = bad_listing_service._list_datasets()
    assert bad_listing_service._list_schemas("XNAS.ITCH") == frozenset()


def test_time_conversion_and_intersection_helpers_cover_edge_cases() -> None:
    dt = datetime(2025, 1, 1, 0, 0, 0)
    ns = _datetime_to_ns(dt)
    assert _iso_to_ns("2025-01-01T00:00:00Z") == ns
    assert _iso_to_ns(None) is None
    assert _iso_to_ns("not-a-date") is None
    assert _ns_to_iso(ns).endswith("Z")

    metadata = _MetadataStub(
        datasets=("XNAS.ITCH",),
        schemas={"XNAS.ITCH": ("trades",)},
        ranges={"XNAS.ITCH": {"start": "2024-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z", "schema": {}}},
    )
    service = DatasetDiscoveryService(
        metadata=cast(Any, metadata),
        policy=_build_policy(),
        resolver=cast(Any, _ResolverStub()),
    )
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 2, tzinfo=UTC)
    start_ns = _datetime_to_ns(start)
    end_ns = _datetime_to_ns(end)
    assert service._intersect_bounds(start, end, end_ns, None) == (None, None)
    assert service._intersect_bounds(start, end, None, start_ns) == (None, None)
    assert service._intersect_bounds(start, end, start_ns, start_ns) == (None, None)
    assert service._intersect_bounds(start, end, None, None) == (start_ns, end_ns)
