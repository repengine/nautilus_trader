"""Dataset build macro refresh tests."""

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from ml.data import DatasetBuildConfig
from ml.data import DatasetValidationConfig
from ml.data import build_tft_dataset
from ml.tests.utils.targets import build_default_target_semantics
from ml.data.ingest.macro_refresh import MacroRefreshResult
from ml.data.ingest.market_bindings import MarketBindingStats
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.data.validation import DatasetValidationError
from ml.data.vintage import VintagePolicy
from ml.registry.feature_registry import FeatureRegistry
from ml.tasks.datasets import TFTDatasetTaskConfig

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

TARGET_SEMANTICS = build_default_target_semantics(
    horizon_minutes=15,
    threshold=0.001,
    legacy_aliases=True,
)

def _install_recording_builder(monkeypatch: pytest.MonkeyPatch, recorder: dict[str, object]) -> None:
    """Replace TFTDatasetBuilder with a subclass that records init kwargs while invoking real logic."""

    class _RecordingBuilder(TFTDatasetBuilder):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            recorder["builder_params"] = kwargs
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("ml.data.TFTDatasetBuilder", _RecordingBuilder)

def _patch_market_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub market binding resolution to avoid relying on descriptor files."""

    def _resolver(**_: Any) -> tuple[MarketBindingStats, ...]:
        binding = MarketBindingStats(
            binding_id="binding-001",
            dataset_id="EQUS.MINI",
            descriptor_id="EQUS.MINI",
            symbol="SPY",
            instrument_ids=("SPY.XNAS",),
            schema="ohlcv-1m",
            storage_kind=None,
            source="descriptor",
            license_start=None,
            license_end=None,
        )
        ts_start_ns = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
        ts_end_ns = ts_start_ns + 60 * 1_000_000_000
        binding.record(source="store", row_count=2, ts_min_ns=ts_start_ns, ts_max_ns=ts_end_ns)
        return (binding,)

    monkeypatch.setattr("ml.data.resolve_market_dataset_bindings", _resolver)


def _disable_micro_catalog_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the microstructure aggregator from reading the catalog during tests."""

    monkeypatch.setattr(
        "ml.features.micro_aggregate.MicrostructureAggregator._load_catalog_quotes",
        lambda self, symbol, *, start=None, end=None: None,
    )
    monkeypatch.setattr(
        "ml.features.micro_aggregate.MicrostructureAggregator._load_catalog_trades",
        lambda self, symbol, *, start=None, end=None: None,
    )


def test_build_tft_dataset_invokes_macro_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    patch_dataset_bars,
) -> None:
    macro_calls: dict[str, object] = {}
    builder_records: dict[str, object] = {}
    data_store = object()

    def _fake_macro(**kwargs: Any) -> MacroRefreshResult:
        macro_calls.update(kwargs)
        fred_path = cast(Path, kwargs["fred_path"])
        vintage_dir = cast(Path, kwargs["vintage_dir"])
        return MacroRefreshResult(
            fred_refreshed=False,
            alfred_refreshed=False,
            fred_path=fred_path,
            alfred_base_dir=vintage_dir,
        )

    patch_dataset_bars()
    _install_recording_builder(monkeypatch, builder_records)
    monkeypatch.setattr("ml.data.ingest.macro_refresh.ensure_macro_ready", _fake_macro)
    monkeypatch.setattr("ml.data.fred_join.join_fred_asof", lambda df, **_: df)
    _patch_market_bindings(monkeypatch)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    macro_path = tmp_path / "macro" / "fred.parquet"
    macro_path.parent.mkdir(parents=True, exist_ok=True)
    vintage_dir = tmp_path / "macro" / "vintages"
    vintage_dir.mkdir(parents=True, exist_ok=True)

    cfg = DatasetBuildConfig(
        data_dir=data_dir,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=TARGET_SEMANTICS,
        include_macro=True,
        macro_series_ids=("DGS10",),
        macro_fred_path=macro_path,
        fred_vintage_dir=vintage_dir,
        validation=DatasetValidationConfig(
            min_rows=0,
            min_positive_rate=None,
            max_positive_rate=None,
            min_feature_coverage=0.0,
            require_macro_series=(),
            macro_min_vintage_observations=None,
        ),
    )

    result = build_tft_dataset(cfg, data_store=data_store)

    assert macro_calls["fred_path"] == cfg.macro_fred_path
    assert macro_calls["vintage_dir"] == cfg.fred_vintage_dir
    assert macro_calls["data_store"] is data_store

    builder_params = builder_records["builder_params"]
    assert isinstance(builder_params, dict)
    assert builder_params["macro_series_ids"] == cfg.macro_series_ids
    assert builder_params["include_macro"] is True
    assert builder_params["fred_path"] == str(cfg.macro_fred_path)
    assert builder_params["market_bindings"], "expected resolved bindings to be propagated"

    assert result.dataset_parquet.exists()
    assert result.dataset_csv.exists()
    assert result.features_npz.exists()
    assert result.metadata is not None
    assert result.metadata.vintage_policy == VintagePolicy.REAL_TIME
    assert result.metadata.market_bindings
    assert result.metadata.capability_flags["include_macro"] is True
    assert result.metadata.capability_flags["include_calendar"] is False
    assert result.metadata.capability_flags["include_events"] is False
    assert result.metadata.capability_flags["include_micro"] is False
    assert result.metadata.capability_flags["include_l2"] is False
    metadata_path = cfg.out_dir / "dataset_metadata.json"
    assert metadata_path.exists()
    metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_raw["dataset_id"] == cfg.dataset_id
    capabilities_json = metadata_raw.get("capability_flags") or {}
    assert capabilities_json.get("include_macro") is True

def test_build_tft_dataset_marks_capabilities_for_earnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _CatalogStub:
        def __init__(self, path: str) -> None:
            self.path = path

    class _BuilderStub:
        def __init__(self, **kwargs: Any) -> None:
            include_l2 = bool(kwargs.get("include_l2", False))
            include_micro = bool(kwargs.get("include_micro", False))
            self.include_macro = bool(kwargs.get("include_macro", False))
            self.include_macro_revisions = bool(kwargs.get("include_macro_revisions", False))
            self.include_calendar = bool(kwargs.get("include_calendar", False))
            self.include_events = bool(kwargs.get("include_events", False))
            self.include_earnings = bool(kwargs.get("include_earnings", False))
            self.include_l2 = include_l2
            self.include_micro = include_micro or include_l2
            self.student_mode = bool(kwargs.get("student_mode", False))

        def build_training_dataset(self, **kwargs: Any) -> pl.DataFrame:
            del kwargs
            return pl.DataFrame(
                {
                    "time_index": [0, 1],
                    "instrument_id": ["SPY", "SPY"],
                    "ts_event": [
                        datetime.fromtimestamp(1_000_000_000 / 1_000_000_000, tz=UTC),
                        datetime.fromtimestamp(2_000_000_000 / 1_000_000_000, tz=UTC),
                    ],
                    "feature_a": [1.0, 2.0],
                    "y": [0.0, 1.0],
                },
            )

        def get_binding_stats(self) -> tuple[MarketBindingStats, ...]:
            return ()

    monkeypatch.setattr(
        "nautilus_trader.persistence.catalog.parquet.ParquetDataCatalog",
        _CatalogStub,
    )
    monkeypatch.setattr("ml.data.TFTDatasetBuilder", _BuilderStub)

    cfg = DatasetBuildConfig(
        data_dir=tmp_path,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=TARGET_SEMANTICS,
        include_macro=False,
        include_earnings=True,
        include_l2=True,
    )

    result = build_tft_dataset(cfg)
    assert result.metadata is not None
    flags = result.metadata.capability_flags
    assert flags["include_macro"] is False
    assert flags["include_earnings"] is True
    assert flags["include_l2"] is True
    assert flags["include_micro"] is True
    payload = json.loads((cfg.out_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert payload["capability_flags"]["include_earnings"] is True
    assert payload["capability_flags"]["include_macro"] is False
    assert payload["capability_flags"]["include_l2"] is True
    assert payload["capability_flags"]["include_micro"] is True

def test_build_tft_dataset_rejects_missing_macro_observations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_macro(**kwargs: Any) -> MacroRefreshResult:
        fred_path = cast(Path, kwargs["fred_path"])
        vintage_dir = cast(Path, kwargs["vintage_dir"])
        return MacroRefreshResult(
            fred_refreshed=False,
            alfred_refreshed=False,
            fred_path=fred_path,
            alfred_base_dir=vintage_dir,
        )

    class _CatalogStub:
        def __init__(self, path: str) -> None:
            self.path = path

    class _BuilderStub:
        def __init__(self, **kwargs: Any) -> None:
            include_l2 = bool(kwargs.get("include_l2", False))
            include_micro = bool(kwargs.get("include_micro", False))
            self.include_macro = bool(kwargs.get("include_macro", False))
            self.include_macro_revisions = bool(kwargs.get("include_macro_revisions", False))
            self.include_calendar = bool(kwargs.get("include_calendar", False))
            self.include_events = bool(kwargs.get("include_events", False))
            self.include_earnings = bool(kwargs.get("include_earnings", False))
            self.include_l2 = include_l2
            self.include_micro = include_micro or include_l2
            self.student_mode = bool(kwargs.get("student_mode", False))

        def build_training_dataset(self, **kwargs: Any) -> pl.DataFrame:
            del kwargs
            return pl.DataFrame(
                {
                    "time_index": [0, 1],
                    "ts_event": [
                        datetime.fromtimestamp(1_000_000_000 / 1_000_000_000, tz=UTC),
                        datetime.fromtimestamp(2_000_000_000 / 1_000_000_000, tz=UTC),
                    ],
                    "DGS10": [0.0, 0.0],
                    "DGS10__value_vintage_ts": [None, None],
                    "y": [0.0, 1.0],
                },
            )

    monkeypatch.setattr("ml.data.ingest.macro_refresh.ensure_macro_ready", _fake_macro)
    monkeypatch.setattr(
        "nautilus_trader.persistence.catalog.parquet.ParquetDataCatalog",
        _CatalogStub,
    )
    monkeypatch.setattr("ml.data.TFTDatasetBuilder", _BuilderStub)

    cfg = DatasetBuildConfig(
        data_dir=tmp_path,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=TARGET_SEMANTICS,
        include_macro=True,
        macro_series_ids=("DGS10",),
        macro_fred_path=tmp_path / "macro" / "fred.parquet",
        fred_vintage_dir=tmp_path / "macro" / "vintages",
    )

    with pytest.raises(DatasetValidationError):
        build_tft_dataset(cfg)

def test_build_tft_dataset_registers_capability_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    patch_dataset_bars,
) -> None:
    patch_dataset_bars()
    _disable_micro_catalog_queries(monkeypatch)

    def _fake_macro(**kwargs: Any) -> MacroRefreshResult:
        fred_path = cast(Path, kwargs["fred_path"])
        vintage_dir = cast(Path, kwargs["vintage_dir"])
        return MacroRefreshResult(
            fred_refreshed=False,
            alfred_refreshed=False,
            fred_path=fred_path,
            alfred_base_dir=vintage_dir,
        )

    monkeypatch.setattr("ml.data.ingest.macro_refresh.ensure_macro_ready", _fake_macro)
    monkeypatch.setattr("ml.data.fred_join.join_fred_asof", lambda df, **_: df)
    _patch_market_bindings(monkeypatch)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    macro_path = tmp_path / "macro" / "fred.parquet"
    macro_path.parent.mkdir(parents=True, exist_ok=True)
    vintage_dir = tmp_path / "macro" / "vintages"
    vintage_dir.mkdir(parents=True, exist_ok=True)
    registry_dir = tmp_path / "registry"

    cfg = DatasetBuildConfig(
        data_dir=data_dir,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=TARGET_SEMANTICS,
        include_macro=True,
        include_l2=True,
        macro_series_ids=("DGS10",),
        macro_fred_path=macro_path,
        fred_vintage_dir=vintage_dir,
        register_features=True,
        feature_registry_dir=registry_dir,
        validation=DatasetValidationConfig(
            min_rows=0,
            min_positive_rate=None,
            max_positive_rate=None,
            min_feature_coverage=0.0,
            require_macro_series=(),
            macro_min_vintage_observations=None,
        ),
    )

    result = build_tft_dataset(cfg)
    assert result.feature_set_id is not None

    registry = FeatureRegistry(registry_dir)
    info = registry.get_feature_set(result.feature_set_id)
    assert info is not None
    flags = info.manifest.capability_flags
    assert flags["include_macro"] is True
    assert flags["include_l2"] is True
    assert flags["include_micro"] is True

def test_tft_dataset_task_config_overrides_base_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    patch_dataset_bars,
) -> None:
    builder_records: dict[str, object] = {}
    patch_dataset_bars()
    _disable_micro_catalog_queries(monkeypatch)
    _install_recording_builder(monkeypatch, builder_records)
    _patch_market_bindings(monkeypatch)

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    micro_dir = tmp_path / "micro_override"
    l2_dir = tmp_path / "l2_override"
    micro_dir.mkdir()
    l2_dir.mkdir()

    cfg = TFTDatasetTaskConfig(
        data_dir=data_dir,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        target_semantics=TARGET_SEMANTICS,
        include_macro=False,
        include_micro=True,
        include_l2=True,
        micro_base_dir=micro_dir,
        l2_base_dir=l2_dir,
        validation=DatasetValidationConfig(
            min_rows=0,
            min_positive_rate=None,
            max_positive_rate=None,
            min_feature_coverage=0.0,
            require_macro_series=(),
            macro_min_vintage_observations=None,
        ),
    )

    result = build_tft_dataset(cfg)
    builder_params = builder_records["builder_params"]
    assert builder_params["micro_base_dir"] == str(micro_dir)
    assert builder_params["l2_base_dir"] == str(l2_dir)
    assert result.dataset_parquet.exists()
