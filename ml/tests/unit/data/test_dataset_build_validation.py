from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from ml.data import DatasetBuildConfig
from ml.data import DatasetValidationConfig
from ml.data import DatasetValidationError
from ml.data import build_tft_dataset
from ml.tests.utils.targets import build_default_target_semantics


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


def _patch_market_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubDescriptors:
        def as_mapping(self) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(
        "ml.data.build.load_market_feed_descriptors",
        lambda: _StubDescriptors(),
    )
    monkeypatch.setattr(
        "ml.data.build.resolve_market_dataset_bindings",
        lambda **_: (),
    )


def test_build_tft_dataset_raises_when_min_rows_exceeds_dataset(
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset_bars: Callable[..., object],
    sample_bar_series_config_factory: Callable[..., object],
    tmp_path: Path,
) -> None:
    bar_config = sample_bar_series_config_factory(rows=4, instrument_id="SPY")
    patch_dataset_bars(config=bar_config)
    _patch_market_bindings(monkeypatch)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = DatasetBuildConfig(
        data_dir=data_dir,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        instrument_ids=["SPY"],
        include_macro=False,
        include_micro=False,
        include_l2=False,
        include_events=False,
        include_calendar=False,
        include_earnings=False,
        include_macro_deltas=False,
        include_calendar_lags=False,
        include_clustering_tags=False,
        include_context_features=False,
        target_semantics=build_default_target_semantics(horizon_minutes=1, threshold=0.0),
        lookback_periods=1,
        validation=DatasetValidationConfig(
            min_rows=10,
            min_positive_rate=None,
            max_positive_rate=None,
            min_feature_coverage=0.0,
            require_macro_series=(),
        ),
    )

    with pytest.raises(DatasetValidationError):
        build_tft_dataset(cfg)


def test_build_tft_dataset_validates_forward_return_alignment(
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset_bars: Callable[..., object],
    sample_bar_series_config_factory: Callable[..., object],
    tmp_path: Path,
) -> None:
    bar_config = sample_bar_series_config_factory(rows=6, instrument_id="SPY")
    patch_dataset_bars(config=bar_config)
    _patch_market_bindings(monkeypatch)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    validation = DatasetValidationConfig(
        min_rows=1,
        min_positive_rate=None,
        max_positive_rate=None,
        min_feature_coverage=0.0,
        require_macro_series=(),
        forward_return_horizon=1,
        forward_return_column="forward_return_1m",
        forward_return_price_column="close",
        require_numeric_features=False,
    )

    cfg = DatasetBuildConfig(
        data_dir=data_dir,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        instrument_ids=["SPY"],
        include_macro=False,
        include_micro=False,
        include_l2=False,
        include_events=False,
        include_calendar=False,
        include_earnings=False,
        include_macro_deltas=False,
        include_calendar_lags=False,
        include_clustering_tags=False,
        include_context_features=False,
        target_semantics=build_default_target_semantics(horizon_minutes=1, threshold=0.0),
        lookback_periods=1,
        validation=validation,
    )

    result = build_tft_dataset(cfg)

    assert result.dataset_parquet.exists()
