from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import cast

import numpy as np

import pytest

from ml.data.ingest.calibration import CalibrationBundle
from ml.data.ingest.calibration import SymbolCalibration
from ml.data.ingest.calibration import dump_calibration_bundle
from ml.data.ingest.service import DatabentoIngestionService
from ml.scripts.verify_eq_itch_parity import CanonicalizedSlice
from ml.scripts.verify_eq_itch_parity import ParityMetrics
from ml.scripts.verify_eq_itch_parity import ParityScenario
from ml.scripts.verify_eq_itch_parity import ParityScenarioResult
from ml.scripts.verify_eq_itch_parity import ParitySuiteReport
from ml.scripts.verify_eq_itch_parity import build_synthetic_minute_bars
from ml.scripts.verify_eq_itch_parity import compare_feature_sets
from ml.scripts.verify_eq_itch_parity import default_parity_suite
from ml.scripts.verify_eq_itch_parity import ensure_calibration_fresh
from ml.scripts.verify_eq_itch_parity import load_parity_suite
from ml.scripts.verify_eq_itch_parity import run_parity_suite
from ml.scripts.verify_eq_itch_parity import serialize_parity_report


def test_compare_feature_sets_identical_series() -> None:
    start = datetime(2024, 1, 2, 8, 0, tzinfo=UTC)
    source = build_synthetic_minute_bars(
        symbol="TEST",
        start=start,
        minutes=120,
        price_base=100.0,
        price_step=0.05,
        volume_base=1_000.0,
        volume_step=5.0,
    )
    parity_a = CanonicalizedSlice(dataframe=source, instrument_id="TEST.XNAS")
    parity_b = CanonicalizedSlice(dataframe=source.copy(deep=True), instrument_id="TEST.XNAS")
    summary = compare_feature_sets(eq_slice=parity_a, fallback_slice=parity_b)
    assert summary["timestamp_count"] == float(len(source.index))
    assert np.isclose(summary["max_abs_diff"], 0.0)
    assert np.isclose(summary["mean_abs_diff"], 0.0)
    assert np.isclose(summary["p99_abs_diff"], 0.0)
    assert summary["worst_timestamp_ns"] is not None
    assert summary["price_close_correlation"] == 1.0
    assert summary["volume_correlation"] == 1.0
    assert summary["volume_ratio_stats"] == {
        "min": 1.0,
        "max": 1.0,
        "median": 1.0,
        "p05": 1.0,
        "p95": 1.0,
    }
    assert summary["volume_residual_abs"] == 0.0
    assert summary["volume_residual_rel"] == 0.0


def test_default_parity_suite_returns_expected_symbols() -> None:
    suite = default_parity_suite()
    assert suite
    for scenario in suite:
        assert scenario.start.tzinfo is not None
        assert scenario.end > scenario.start
        assert scenario.fallback_dataset == "XNAS.ITCH"


def test_run_parity_suite_aggregates_results() -> None:
    start = datetime(2024, 1, 2, 8, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 10, 0, tzinfo=UTC)
    eq_frame = build_synthetic_minute_bars(
        symbol="TEST",
        start=start,
        minutes=120,
        price_base=100.0,
        price_step=0.05,
        volume_base=1_000.0,
        volume_step=5.0,
    )
    fallback_frame = eq_frame.copy(deep=True)
    scenario = ParityScenario(
        label="TEST_SCENARIO",
        eq_symbol="TEST",
        fallback_symbol="TEST.XNAS",
        fallback_dataset="XNAS.ITCH",
        start=start,
        end=end,
    )

    def _stub_ingest(
        *,
        service: DatabentoIngestionService,
        dataset: str,
        symbol: str,
        start: datetime,
        end: datetime,
        schema: str,
    ) -> CanonicalizedSlice:
        frame = eq_frame if dataset == "EQUS.MINI" else fallback_frame
        return CanonicalizedSlice(dataframe=frame.copy(deep=True), instrument_id=f"{symbol}-id")

    report = run_parity_suite(
        service=cast(DatabentoIngestionService, object()),
        scenarios=(scenario,),
        ingest=cast(Callable[..., CanonicalizedSlice], _stub_ingest),
    )
    assert report.generated_at.tzinfo is UTC
    assert len(report.results) == 1
    result = report.results[0]
    assert result.scenario.label == "TEST_SCENARIO"
    assert result.metrics.timestamp_count == float(eq_frame.shape[0])
    assert np.isclose(result.metrics.max_abs_diff, 0.0)
    assert np.isclose(result.metrics.mean_abs_diff, 0.0)
    assert np.isclose(result.metrics.p99_abs_diff, 0.0)
    assert result.metrics.price_close_correlation == 1.0
    assert result.metrics.volume_correlation == 1.0
    assert result.metrics.volume_ratio_stats == {
        "min": 1.0,
        "max": 1.0,
        "median": 1.0,
        "p05": 1.0,
        "p95": 1.0,
    }
    assert result.metrics.volume_residual_abs == 0.0
    assert result.metrics.volume_residual_rel == 0.0


def test_load_parity_suite_parses_json(tmp_path: Path) -> None:
    config = {
        "scenarios": [
            {
                "label": "CUSTOM",
                "eq_symbol": "ABC",
                "fallback_symbol": "ABC.XNAS",
                "fallback_dataset": "XNAS.ITCH",
                "start": "2024-01-10T08:00Z",
                "end": "2024-01-10T16:00Z",
            },
        ],
    }
    config_path = tmp_path / "suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    scenarios = load_parity_suite(config_path)
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario.label == "CUSTOM"
    assert scenario.eq_symbol == "ABC"
    assert scenario.fallback_symbol == "ABC.XNAS"
    assert scenario.start == datetime(2024, 1, 10, 8, 0, tzinfo=UTC)
    assert scenario.end == datetime(2024, 1, 10, 16, 0, tzinfo=UTC)


def test_serialize_parity_report_is_json_serialisable() -> None:
    scenario = ParityScenario(
        label="SERIALISE",
        eq_symbol="SER",
        fallback_symbol="SER.XNAS",
        fallback_dataset="XNAS.ITCH",
        start=datetime(2024, 2, 5, 8, 0, tzinfo=UTC),
        end=datetime(2024, 2, 5, 16, 0, tzinfo=UTC),
    )
    metrics = ParityMetrics(
        timestamp_count=10.0,
        max_abs_diff=0.1,
        mean_abs_diff=0.05,
        p99_abs_diff=0.1,
        worst_timestamp_ns=123,
        worst_timestamp_iso="1970-01-01T00:00:00+00:00",
        price_close_max_abs_diff=0.1,
        price_close_p99_abs_diff=0.1,
        price_close_mean_abs_diff=0.1,
        price_close_correlation=1.0,
        volume_correlation=1.0,
        volume_ratio_stats={"min": 1.0, "max": 1.0, "median": 1.0, "p05": 1.0, "p95": 1.0},
        volume_residual_abs=0.0,
        volume_residual_rel=0.0,
    )
    report = ParitySuiteReport(
        generated_at=datetime(2024, 2, 6, 12, 0, tzinfo=UTC),
        results=(ParityScenarioResult(scenario=scenario, metrics=metrics),),
    )
    payload = serialize_parity_report(report)
    encoded = json.dumps(payload)
    assert "SERIALISE" in encoded


def _bundle(now: datetime) -> CalibrationBundle:
    return CalibrationBundle(
        generated_at=now,
        symbols={
            "INTC": SymbolCalibration(
                sale_condition_allowlist=frozenset(),
                volume_scale_by_minute={},
                price_scaling_by_minute={},
                split_events={},
                exclude_auction_minutes=frozenset(),
            ),
        },
    )


def test_ensure_calibration_fresh_accepts_recent_bundle(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    dump_calibration_bundle(_bundle(datetime.now(tz=UTC)), path)
    ensure_calibration_fresh(
        calibration_path=path,
        max_age_days=2,
        allow_missing=False,
    )


def test_ensure_calibration_fresh_raises_when_stale(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    old = datetime(2020, 1, 1, tzinfo=UTC)
    dump_calibration_bundle(_bundle(old), path)
    with pytest.raises(RuntimeError):
        ensure_calibration_fresh(
            calibration_path=path,
            max_age_days=1,
            allow_missing=False,
        )


def test_ensure_calibration_fresh_allows_missing_when_flagged() -> None:
    ensure_calibration_fresh(
        calibration_path=None,
        max_age_days=1,
        allow_missing=True,
    )
