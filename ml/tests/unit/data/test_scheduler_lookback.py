from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ml.data.scheduler import DataScheduler
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.stores.providers import DAY_NS

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


class _StubCoverage:
    def __init__(self, mapping: dict[str, int | None]) -> None:
        self._mapping = mapping

    def latest_timestamp_ns(self, *, dataset_id: str, instrument_id: str) -> int | None:  # noqa: ARG002
        return self._mapping.get(instrument_id)

def _make_binding(license_start: str | None, license_end: str | None) -> ResolvedMarketBinding:
    return ResolvedMarketBinding(
        binding_id="binding",
        symbol="SPY",
        instrument_ids=("SPY.XNAS",),
        dataset_id="EQUS.MINI",
        descriptor_id="EQUS.MINI_TBBO",
        schema="tbbo",
        storage_kind=None,
        license_start=license_start,
        license_end=license_end,
        start=None,
        end=None,
        source="descriptor",
    )

def test_binding_lookback_clamped_to_license_start() -> None:
    binding = _make_binding(license_start="2024-10-01", license_end=None)
    reference = datetime(2025, 11, 5, tzinfo=UTC)
    result = DataScheduler._binding_lookback_days(binding=binding, base_lookback_days=365, reference_time=reference)
    expected_days = max((reference - datetime(2024, 10, 1, tzinfo=UTC)).days, 1)
    assert result == min(365, expected_days)

def test_binding_lookback_defaults_when_no_license() -> None:
    binding = _make_binding(license_start=None, license_end=None)
    result = DataScheduler._binding_lookback_days(binding=binding, base_lookback_days=180, reference_time=datetime.now(tz=UTC))
    assert result == 180

def test_binding_lookback_handles_expired_dataset() -> None:
    binding = _make_binding(license_start="2023-01-01", license_end="2023-12-31")
    reference = datetime(2025, 1, 1, tzinfo=UTC)
    result = DataScheduler._binding_lookback_days(binding=binding, base_lookback_days=365, reference_time=reference)
    assert result == max((datetime(2023, 12, 31, tzinfo=UTC) - datetime(2023, 1, 1, tzinfo=UTC)).days, 1)


def test_compute_dynamic_lookbacks_uses_sql_staleness() -> None:
    scheduler = object.__new__(DataScheduler)
    now_ns = int(datetime.now(tz=UTC).timestamp() * 1_000_000_000)
    mapping = {
        "SPY.XNAS": now_ns - 2 * DAY_NS,
        "AAPL.XNAS": now_ns - (DAY_NS // 2),
    }
    result = DataScheduler._compute_dynamic_lookbacks(
        scheduler,
        coverage=_StubCoverage(mapping),
        dataset_id="EQUS.MINI",
        instrument_ids=("SPY.XNAS", "AAPL.XNAS"),
        min_days=1,
        max_days=5,
    )
    assert 3 <= result["SPY.XNAS"] <= 5
    assert 2 <= result["AAPL.XNAS"] <= 5  # ceil(0.5)=1 -> 2 days


def test_binding_dynamic_base_prefers_instrument_map() -> None:
    scheduler = object.__new__(DataScheduler)
    scheduler._instrument_dynamic_lookbacks = {  # type: ignore[attr-defined]
        "SPY.XNAS": 2,
        "SPY.ARCX": 5,
    }
    binding = ResolvedMarketBinding(
        binding_id="binding",
        symbol="SPY",
        instrument_ids=("SPY.XNAS", "SPY.ARCX"),
        dataset_id="EQUS.MINI",
        descriptor_id=None,
        schema="tbbo",
        storage_kind=None,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="descriptor",
    )
    assert DataScheduler._binding_dynamic_base(scheduler, binding=binding, fallback=1) == 5
