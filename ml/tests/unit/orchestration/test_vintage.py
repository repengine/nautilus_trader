from __future__ import annotations

from datetime import UTC
from datetime import datetime

import pandas as pd
import pytest

from ml.orchestration.vintage import VintagePolicy
from ml.orchestration.vintage import VintageWindowPolicy


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
)


def test_vintage_window_policy_rejects_non_positive_max_age_days() -> None:
    with pytest.raises(ValueError, match="max_age_days must be > 0"):
        VintageWindowPolicy(max_age_days=0)


def test_filter_by_vintage_raises_when_timestamp_column_missing() -> None:
    policy = VintageWindowPolicy(max_age_days=30)
    frame = pd.DataFrame({"close": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Timestamp column 'timestamp' not found"):
        policy.filter_by_vintage(frame, datetime(2026, 1, 10, tzinfo=UTC))


def test_filter_by_vintage_filters_datetime_window() -> None:
    policy = VintageWindowPolicy(max_age_days=2)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-07T00:00:00+00:00",
                    "2026-01-08T00:00:00+00:00",
                    "2026-01-09T00:00:00+00:00",
                    "2026-01-10T00:00:00+00:00",
                    "2026-01-11T00:00:00+00:00",
                ],
                utc=True,
            ),
            "value": [1, 2, 3, 4, 5],
        },
    )

    filtered = policy.filter_by_vintage(frame, datetime(2026, 1, 10, tzinfo=UTC))

    assert filtered["value"].tolist() == [2, 3, 4]
    assert frame["value"].tolist() == [1, 2, 3, 4, 5]


def test_filter_by_vintage_converts_non_datetime_column() -> None:
    policy = VintageWindowPolicy(max_age_days=1)
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-09T00:00:00+00:00",
                "2026-01-10T00:00:00+00:00",
            ],
            "close": [101.0, 102.0],
        },
    )

    filtered = policy.filter_by_vintage(frame, datetime(2026, 1, 10, tzinfo=UTC))

    assert filtered["close"].tolist() == [101.0, 102.0]
    assert pd.api.types.is_datetime64_any_dtype(filtered["timestamp"])
    assert frame["timestamp"].dtype == object


def test_filter_by_vintage_raises_when_timestamp_conversion_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = VintageWindowPolicy(max_age_days=1)
    frame = pd.DataFrame({"timestamp": ["bad-value"]})

    monkeypatch.setattr(
        pd,
        "to_datetime",
        lambda _values: (_ for _ in ()).throw(ValueError("invalid timestamp")),
    )

    with pytest.raises(ValueError, match="Cannot convert timestamp to datetime: invalid timestamp"):
        policy.filter_by_vintage(frame, datetime(2026, 1, 10, tzinfo=UTC))


def test_compute_vintage_metadata_includes_policy_fields() -> None:
    policy = VintageWindowPolicy(max_age_days=30)

    metadata = policy.compute_vintage_metadata(
        current_date=datetime(2026, 2, 7, tzinfo=UTC),
        original_count=40,
        filtered_count=10,
    )

    vintage = metadata["vintage_policy"]
    assert vintage["max_age_days"] == 30
    assert vintage["cutoff_date"] == "2026-01-08"
    assert vintage["current_date"] == "2026-02-07"
    assert vintage["rows_removed"] == 30
    assert vintage["removal_pct"] == 75.0


def test_compute_vintage_metadata_handles_zero_original_rows() -> None:
    policy = VintageWindowPolicy(max_age_days=7)

    metadata = policy.compute_vintage_metadata(
        current_date=datetime(2026, 2, 7, tzinfo=UTC),
        original_count=0,
        filtered_count=0,
    )

    assert metadata["vintage_policy"]["removal_pct"] == 0.0


def test_vintage_policy_alias_points_to_window_policy() -> None:
    assert VintagePolicy is VintageWindowPolicy
