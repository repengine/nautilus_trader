"""Contract tests for target semantics in datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from ml.config.targets import BinaryTargetConfig
from ml.config.targets import MulticlassTargetConfig
from ml.config.targets import RegressionTargetConfig
from ml.config.targets import TargetHorizonSpec
from ml.config.targets import TargetSemanticsConfig
from ml.data import DatasetBuildConfig
from ml.data import DatasetValidationConfig
from ml.data import build_tft_dataset


pytestmark = [
    pytest.mark.contracts,
    pytest.mark.usefixtures(
        "isolated_prometheus_registry",
        "mock_tracing_backend",
        "isolated_orchestrator_env",
    ),
]


def _patch_market_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubDescriptors:
        def as_mapping(self) -> dict[str, Any]:
            return {}

    monkeypatch.setattr("ml.data.load_market_feed_descriptors", lambda: _StubDescriptors())
    monkeypatch.setattr("ml.data.resolve_market_dataset_bindings", lambda **_: ())


def test_build_tft_dataset_emits_target_semantics_metadata_and_columns(
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset_bars: Callable[..., object],
    sample_bar_series_config_factory: Callable[..., object],
    tmp_path: Path,
) -> None:
    """Ensure dataset outputs include explicit target columns and semantics metadata."""
    pl = pytest.importorskip("polars")

    bar_config = sample_bar_series_config_factory(rows=8, instrument_id="SPY")
    patch_dataset_bars(config=bar_config)
    _patch_market_bindings(monkeypatch)

    target_semantics = TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=1), TargetHorizonSpec(minutes=2)),
        binary=BinaryTargetConfig(enabled=True, threshold_bps=10.0),
        multiclass=MulticlassTargetConfig(
            enabled=True,
            short_threshold_bps=10.0,
            long_threshold_bps=10.0,
        ),
        regression=RegressionTargetConfig(enabled=True),
        primary_target="target_bin_1m",
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = DatasetBuildConfig(
        data_dir=data_dir,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        include_macro=False,
        include_micro=False,
        include_l2=False,
        include_events=False,
        include_calendar=False,
        include_earnings=False,
        auto_refresh_macro=False,
        target_semantics=target_semantics,
        lookback_periods=2,
        validation=DatasetValidationConfig(
            min_positive_rate=None,
            max_positive_rate=None,
        ),
    )

    result = build_tft_dataset(cfg)
    dataset_df = pl.read_parquet(str(result.dataset_parquet))

    expected_columns = {
        "forward_return_1m",
        "forward_return_2m",
        "target_bin_1m",
        "target_bin_2m",
        "target_class_1m",
        "target_class_2m",
        "target_reg_1m",
        "target_reg_2m",
    }
    assert expected_columns.issubset(set(dataset_df.columns))

    assert result.metadata is not None
    semantics = result.metadata.target_semantics
    assert semantics is not None
    assert semantics.get("version") == "v1"
    assert "returns" in semantics
    assert "labels" in semantics
    assert "forward_return_1m" in semantics["returns"]
    assert "target_bin_1m" in semantics["labels"]
    assert semantics.get("primary_target") == "target_bin_1m"
