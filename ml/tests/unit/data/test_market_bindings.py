from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import load_market_feed_descriptors
from ml.data import DatasetMetadata
from ml.data.ingest.market_bindings import MarketBindingStats
from ml.data.ingest.market_bindings import resolve_market_dataset_bindings
from ml.data import MarketBindingMetadata
from ml.data import VintagePolicy
from ml.data import _binding_stats_to_metadata
from ml.data import _metadata_to_dict
from ml.data import load_dataset_metadata


def test_resolve_market_bindings_prefers_descriptor_and_fallback() -> None:
    descriptors = load_market_feed_descriptors().as_mapping()
    assert "EQUS.MINI" in descriptors

    inputs = (
        MarketDatasetInput(descriptor_id="EQUS.MINI", symbols=("AAPL",)),
    )

    bindings = resolve_market_dataset_bindings(
        symbols=["AAPL", "VIX"],
        instrument_ids=("AAPL.XNAS",),
        market_dataset_id="LEGACY.BARS",
        market_inputs=inputs,
        descriptors=descriptors,
    )

    assert len(bindings) == 2
    binding_map = {binding.symbol: binding for binding in bindings}

    aapl_binding = binding_map["AAPL"]
    assert aapl_binding.dataset_id == "EQUS.MINI"
    assert aapl_binding.descriptor_id == "EQUS.MINI"
    assert "AAPL.XNAS" in aapl_binding.instrument_ids

    fallback_binding = binding_map["VIX"]
    assert fallback_binding.dataset_id == "LEGACY.BARS"
    assert fallback_binding.descriptor_id is None


def test_resolve_market_bindings_uses_instrument_templates_when_missing_lookup() -> None:
    descriptors = load_market_feed_descriptors().as_mapping()

    bindings = resolve_market_dataset_bindings(
        symbols=["MSFT"],
        instrument_ids=(),
        market_dataset_id=None,
        market_inputs=(MarketDatasetInput(descriptor_id="XNAS.ITCH"),),
        descriptors=descriptors,
    )

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.descriptor_id == "XNAS.ITCH"
    assert "MSFT.XNAS" in binding.instrument_ids


def test_resolve_market_bindings_generates_heuristic_instrument_ids_for_fallback() -> None:
    descriptors = load_market_feed_descriptors().as_mapping()

    bindings = resolve_market_dataset_bindings(
        symbols=["TSLA"],
        instrument_ids=(),
        market_dataset_id="LEGACY.BARS",
        market_inputs=(),
        descriptors=descriptors,
    )

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.descriptor_id is None
    assert binding.dataset_id == "LEGACY.BARS"
    assert binding.instrument_ids == (
        "TSLA.ARCX",
        "TSLA.NASDAQ",
        "TSLA.NYSE",
        "TSLA.XNAS",
        "TSLA.XNYS",
    )


def test_binding_stats_to_metadata_converts_timestamps() -> None:
    stats = MarketBindingStats(
        binding_id="binding-1",
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI",
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        schema="ohlcv-1m",
        storage_kind=None,
        source="descriptor",
        license_start="2010-01-01",
        license_end=None,
    )
    ts_start = datetime(2024, 1, 1, tzinfo=UTC)
    ts_end = datetime(2024, 1, 2, tzinfo=UTC)
    stats.record(
        source="store",
        row_count=10,
        ts_min_ns=int(ts_start.timestamp() * 1_000_000_000),
        ts_max_ns=int(ts_end.timestamp() * 1_000_000_000),
    )

    metadata = _binding_stats_to_metadata((stats,))
    assert metadata == (
        MarketBindingMetadata(
            binding_id="binding-1",
            dataset_id="EQUS.MINI",
            descriptor_id="EQUS.MINI",
            schema="ohlcv-1m",
            storage_kind=None,
            symbols=("AAPL",),
            instrument_ids=("AAPL.XNAS",),
            source="descriptor",
            license_start="2010-01-01",
            license_end=None,
            ts_event_start="2024-01-01T00:00:00+00:00",
            ts_event_end="2024-01-02T00:00:00+00:00",
            rows_from_store=10,
            rows_from_catalog=0,
        ),
    )


def test_resolve_market_bindings_sets_provider_dataset_id_from_descriptor() -> None:
    descriptors = load_market_feed_descriptors().as_mapping()

    bindings = resolve_market_dataset_bindings(
        symbols=["AAPL"],
        instrument_ids=("AAPL.XNAS",),
        market_dataset_id=None,
        market_inputs=(MarketDatasetInput(descriptor_id="EQUS.MINI_TBBO"),),
        descriptors=descriptors,
    )

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.dataset_id == "EQUS.MINI_TBBO"
    assert binding.provider_dataset_id == "EQUS.MINI"


def test_resolve_market_bindings_sets_provider_schema_from_descriptor() -> None:
    descriptors = load_market_feed_descriptors().as_mapping()

    bindings = resolve_market_dataset_bindings(
        symbols=["AAPL"],
        instrument_ids=("AAPL.XNAS",),
        market_dataset_id=None,
        market_inputs=(MarketDatasetInput(descriptor_id="EQUS.MINI_QUOTES"),),
        descriptors=descriptors,
    )

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.dataset_id == "EQUS.MINI_QUOTES"
    assert binding.schema == "quotes"
    assert binding.provider_schema == "tbbo"


def test_metadata_round_trip_preserves_provider_dataset_id(tmp_path: Path) -> None:
    metadata = DatasetMetadata(
        dataset_id="demo",
        vintage_policy=VintagePolicy.REAL_TIME,
        vintage_cutoff=None,
        build_ts="2024-01-01T00:00:00+00:00",
        ts_event_start=None,
        ts_event_end=None,
        overall_window=None,
        train_window=None,
        validation_window=None,
        test_window=None,
        macro_observation_counts={},
        market_bindings=(
            MarketBindingMetadata(
                binding_id="binding-1",
                dataset_id="EQUS.MINI_TBBO",
                descriptor_id="EQUS.MINI_TBBO",
                schema="tbbo",
                storage_kind="postgres",
                symbols=("AAPL",),
                instrument_ids=("AAPL.XNAS",),
                source="descriptor",
                license_start="2023-01-01",
                license_end=None,
                ts_event_start=None,
                ts_event_end=None,
                rows_from_store=0,
                rows_from_catalog=0,
                provider_dataset_id="EQUS.MINI",
                provider_schema="tbbo",
            ),
        ),
    )

    payload = _metadata_to_dict(metadata)
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = load_dataset_metadata(metadata_path)
    assert reloaded.market_bindings is not None
    assert reloaded.market_bindings[0].provider_dataset_id == "EQUS.MINI"
    assert reloaded.market_bindings[0].provider_schema == "tbbo"
