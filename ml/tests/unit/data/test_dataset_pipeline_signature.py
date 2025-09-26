from __future__ import annotations

from ml.data import MarketBindingMetadata
from ml.data import compute_dataset_pipeline_signature
from ml.data.vintage import VintagePolicy


def _base_signature_kwargs() -> dict[str, object]:
    return {
        "dataset_id": "tft_dataset",
        "symbols": ("SPY", "QQQ"),
        "instrument_ids": ("SPY.XNAS", "QQQ.XNAS"),
        "macro_series_ids": ("CPI",),
        "include_macro": True,
        "macro_lag_days": 1,
        "vintage_policy": VintagePolicy.REAL_TIME,
        "vintage_cutoff": None,
        "ts_event_start": "2024-01-01T00:00:00+00:00",
        "ts_event_end": "2024-01-31T00:00:00+00:00",
    }


def test_pipeline_signature_changes_with_market_bindings() -> None:
    binding_a = MarketBindingMetadata(
        binding_id="binding-a",
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI",
        schema="ohlcv-1m",
        storage_kind="postgres",
        symbols=("SPY",),
        instrument_ids=("SPY.XNAS",),
        source="descriptor",
        license_start="2010-01-01",
        license_end=None,
        ts_event_start="2024-01-01T00:00:00+00:00",
        ts_event_end="2024-01-31T00:00:00+00:00",
        rows_from_store=100,
        rows_from_catalog=0,
    )
    binding_b = MarketBindingMetadata(
        binding_id="binding-b",
        dataset_id="DBEQ.MINI",
        descriptor_id="DBEQ.MINI",
        schema="tbbo",
        storage_kind="postgres",
        symbols=("QQQ",),
        instrument_ids=("QQQ.XNAS",),
        source="descriptor",
        license_start="2018-01-01",
        license_end=None,
        ts_event_start="2024-01-01T00:00:00+00:00",
        ts_event_end="2024-01-31T00:00:00+00:00",
        rows_from_store=120,
        rows_from_catalog=0,
    )

    base_kwargs = _base_signature_kwargs()

    sig_a = compute_dataset_pipeline_signature(
        **base_kwargs,
        market_bindings=(binding_a,),
    )
    sig_b = compute_dataset_pipeline_signature(
        **base_kwargs,
        market_bindings=(binding_b,),
    )

    assert sig_a != sig_b


def test_pipeline_signature_stable_under_binding_reordering() -> None:
    binding_a = MarketBindingMetadata(
        binding_id="binding-a",
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI",
        schema="ohlcv-1m",
        storage_kind="postgres",
        symbols=("SPY",),
        instrument_ids=("SPY.XNAS",),
        source="descriptor",
        license_start="2010-01-01",
        license_end=None,
        ts_event_start="2024-01-01T00:00:00+00:00",
        ts_event_end="2024-01-31T00:00:00+00:00",
        rows_from_store=100,
        rows_from_catalog=0,
    )
    binding_b = MarketBindingMetadata(
        binding_id="binding-b",
        dataset_id="DBEQ.MINI",
        descriptor_id="DBEQ.MINI",
        schema="tbbo",
        storage_kind="postgres",
        symbols=("QQQ",),
        instrument_ids=("QQQ.XNAS",),
        source="descriptor",
        license_start="2018-01-01",
        license_end=None,
        ts_event_start="2024-01-01T00:00:00+00:00",
        ts_event_end="2024-01-31T00:00:00+00:00",
        rows_from_store=120,
        rows_from_catalog=0,
    )

    base_kwargs = _base_signature_kwargs()

    sig_forward = compute_dataset_pipeline_signature(
        **base_kwargs,
        market_bindings=(binding_a, binding_b),
    )
    sig_reverse = compute_dataset_pipeline_signature(
        **base_kwargs,
        market_bindings=(binding_b, binding_a),
    )

    assert sig_forward == sig_reverse
