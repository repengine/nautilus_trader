import os
import pytest

from ml.config.coverage import CoveragePolicy, get_max_lookback_days


def test_coverage_policy_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_L0_LOOKBACK_DAYS", "10")
    monkeypatch.setenv("ML_L1_LOOKBACK_DAYS", "20")
    monkeypatch.setenv("ML_L2_LOOKBACK_DAYS", "30")
    monkeypatch.setenv("ML_L3_LOOKBACK_DAYS", "40")

    p = CoveragePolicy.from_env()
    assert p.l0_max_lookback_days == 10
    assert p.l1_max_lookback_days == 20
    assert p.l2_max_lookback_days == 30
    assert p.l3_max_lookback_days == 40


def test_get_max_lookback_days_mapping_defaults() -> None:
    p = CoveragePolicy()
    assert get_max_lookback_days("bars", p) == p.l0_max_lookback_days
    assert get_max_lookback_days("quotes", p) == p.l1_max_lookback_days
    assert get_max_lookback_days("trades", p) == p.l1_max_lookback_days
    assert get_max_lookback_days("mbp1", p) == p.l2_max_lookback_days
    assert get_max_lookback_days("tbbo", p) == p.l2_max_lookback_days
    assert get_max_lookback_days("unknown", p) == p.l2_max_lookback_days
