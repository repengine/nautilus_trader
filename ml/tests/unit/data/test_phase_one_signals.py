from __future__ import annotations

from ml.data.phase_one_signals import derive_phase_one_signals


def test_derive_phase_one_signals_classifies_expected_columns() -> None:
    columns = [
        "is_market_open",
        "hours_to_fed_meeting",
        "event_clustering_score",
        "PAYEMS_delta_1d",
        "total_events_24h",
        "event_density_week",
        "price",
    ]

    signals = derive_phase_one_signals(columns)

    assert signals["macro_delta_columns"] == ("PAYEMS_delta_1d",)
    assert signals["calendar_lag_columns"] == ("hours_to_fed_meeting",)
    assert signals["clustering_tag_columns"] == ("event_clustering_score", "event_density_week", "total_events_24h")
    assert signals["context_feature_columns"] == ("is_market_open",)
