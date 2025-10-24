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
from ml.data.validation import DatasetValidationError
from ml.data.ingest.macro_refresh import MacroRefreshResult
from ml.data.ingest.market_bindings import MarketBindingStats
from ml.data.vintage import VintagePolicy
from ml.registry.feature_registry import FeatureRegistry
from ml.tasks.datasets import TFTDatasetTaskConfig


def test_build_tft_dataset_invokes_macro_refresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    def _fake_macro(**kwargs: Any) -> MacroRefreshResult:
        recorded.update(kwargs)
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
            recorded["builder_params"] = kwargs
            self.include_macro = bool(kwargs.get("include_macro", False))
            self.include_macro_revisions = bool(kwargs.get("include_macro_revisions", False))
            self.include_calendar = bool(kwargs.get("include_calendar", False))
            self.include_events = bool(kwargs.get("include_events", False))
            self.include_earnings = bool(kwargs.get("include_earnings", False))
            include_micro = bool(kwargs.get("include_micro", False))
            include_l2 = bool(kwargs.get("include_l2", False))
            self.include_l2 = include_l2
            self.include_micro = include_micro or include_l2
            self.student_mode = bool(kwargs.get("student_mode", False))
            stat = MarketBindingStats(
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
            ts_end_ns = int(datetime(2024, 1, 2, tzinfo=UTC).timestamp() * 1_000_000_000)
            stat.record(source="store", row_count=2, ts_min_ns=ts_start_ns, ts_max_ns=ts_end_ns)
            self._stats = (stat,)

        def build_training_dataset(self, **kwargs: Any) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "time_index": [0, 1],
                    "ts_event": [
                        datetime.fromtimestamp(1_000_000_000 / 1_000_000_000, tz=UTC),
                        datetime.fromtimestamp(2_000_000_000 / 1_000_000_000, tz=UTC),
                    ],
                    "feature_a": [1.0, 2.0],
                    "y": [0.0, 1.0],
                },
            )

        def get_binding_stats(self) -> tuple[MarketBindingStats, ...]:
            return self._stats

    monkeypatch.setattr(
        "ml.data.ingest.macro_refresh.ensure_macro_ready",
        _fake_macro,
    )
    monkeypatch.setattr(
        "nautilus_trader.persistence.catalog.parquet.ParquetDataCatalog",
        _CatalogStub,
    )
    monkeypatch.setattr("ml.data.TFTDatasetBuilder", _BuilderStub)

    cfg = DatasetBuildConfig(
        data_dir=tmp_path,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        include_macro=True,
        macro_series_ids=("DGS10",),
        macro_fred_path=tmp_path / "macro" / "fred.parquet",
        fred_vintage_dir=tmp_path / "macro" / "vintages",
        validation=DatasetValidationConfig(
            min_rows=1,
            min_positive_rate=None,
            max_positive_rate=None,
            min_feature_coverage=0.0,
            require_macro_series=(),
            macro_min_vintage_observations=None,
        ),
    )

    result = build_tft_dataset(cfg)

    assert recorded["fred_path"] == cfg.macro_fred_path
    builder_params = recorded["builder_params"]
    assert isinstance(builder_params, dict)
    assert builder_params["fred_path"] == str(cfg.macro_fred_path)
    assert builder_params["macro_series_ids"] == cfg.macro_series_ids
    assert builder_params["market_bindings"] == ()
    assert result.dataset_parquet.exists()
    assert result.dataset_csv.exists()
    assert result.features_npz.exists()
    assert result.metadata is not None
    assert result.metadata.vintage_policy == VintagePolicy.REAL_TIME
    assert result.metadata.market_bindings is not None
    assert len(result.metadata.market_bindings) == 1
    assert result.metadata.capability_flags["include_macro"] is True
    assert result.metadata.capability_flags["include_calendar"] is False
    assert result.metadata.capability_flags["include_events"] is False
    assert result.metadata.capability_flags["include_micro"] is False
    assert result.metadata.capability_flags["include_l2"] is False
    binding_meta = result.metadata.market_bindings[0]
    assert binding_meta.dataset_id == "EQUS.MINI"
    assert binding_meta.rows_from_store == 2
    assert binding_meta.symbols == ("SPY",)
    metadata_path = cfg.out_dir / "dataset_metadata.json"
    assert metadata_path.exists()
    metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_raw["vintage_policy"] == VintagePolicy.REAL_TIME.value
    assert metadata_raw["dataset_id"] == cfg.dataset_id
    assert metadata_raw["train_window"] is not None
    assert metadata_raw["ts_event_start"] is not None
    assert metadata_raw["ts_event_end"] is not None
    capabilities_json = metadata_raw.get("capability_flags") or {}
    assert capabilities_json.get("include_macro") is True
    assert capabilities_json.get("include_calendar") is False
    assert capabilities_json.get("include_micro") is False
    assert builder_params["vintage_policy"] == VintagePolicy.REAL_TIME
    assert builder_params["vintage_as_of"] is None


def test_build_tft_dataset_marks_capabilities_for_earnings(
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
                    "feature_a": [1.0, 2.0],
                    "y": [0.0, 1.0],
                },
            )

        def get_binding_stats(self) -> tuple[MarketBindingStats, ...]:
            return ()

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
        include_macro=False,
        include_earnings=True,
        include_l2=True,
        macro_fred_path=tmp_path / "macro" / "fred.parquet",
        fred_vintage_dir=tmp_path / "macro" / "vintages",
    )

    result = build_tft_dataset(cfg)
    assert result.metadata is not None
    flags = result.metadata.capability_flags
    assert flags["include_macro"] is False
    assert flags["include_earnings"] is True
    assert flags["include_l2"] is True
    assert flags["include_micro"] is True
    metadata_path = cfg.out_dir / "dataset_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
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
                    "time_index": [0, 1, 2],
                    "ts_event": [
                        datetime.fromtimestamp(1_000_000_000 / 1_000_000_000, tz=UTC),
                        datetime.fromtimestamp(2_000_000_000 / 1_000_000_000, tz=UTC),
                        datetime.fromtimestamp(3_000_000_000 / 1_000_000_000, tz=UTC),
                    ],
                    "DGS10": [3.1, 3.2, 3.3],
                    "DGS10__value_vintage_ts": [None, None, None],
                    "feature_a": [1.0, 2.0, 3.0],
                    "y": [0.0, 1.0, 0.0],
                },
            )

        def get_binding_stats(self) -> tuple[MarketBindingStats, ...]:
            return ()

    monkeypatch.setattr("ml.data.ingest.macro_refresh.ensure_macro_ready", _fake_macro)
    monkeypatch.setattr(
        "nautilus_trader.persistence.catalog.parquet.ParquetDataCatalog",
        _CatalogStub,
    )
    monkeypatch.setattr("ml.data.TFTDatasetBuilder", _BuilderStub)

    registry_dir = tmp_path / "registry"
    cfg = DatasetBuildConfig(
        data_dir=tmp_path,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        include_macro=True,
        include_l2=True,
        macro_series_ids=("DGS10",),
        macro_fred_path=tmp_path / "macro" / "fred.parquet",
        fred_vintage_dir=tmp_path / "macro" / "vintages",
        register_features=True,
        feature_registry_dir=registry_dir,
        validation=DatasetValidationConfig(
            min_rows=1,
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
) -> None:
    recorded: dict[str, object] = {}
    pl = pytest.importorskip("polars")

    class _BuilderStub:
        def __init__(self, **kwargs: Any) -> None:
            recorded.update(
                {
                    "micro_base_dir": kwargs.get("micro_base_dir"),
                    "l2_base_dir": kwargs.get("l2_base_dir"),
                },
            )
            self.include_macro = kwargs.get("include_macro", False)
            self.include_micro = kwargs.get("include_micro", False)
            self.include_l2 = kwargs.get("include_l2", False)
            self.include_macro_revisions = kwargs.get("include_macro_revisions", False)
            self.student_mode = kwargs.get("student_mode", False)

        def build_training_dataset(self, **kwargs: Any) -> Any:
            del kwargs
            timestamps = [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, minute=1, tzinfo=UTC),
                datetime(2024, 1, 1, minute=2, tzinfo=UTC),
            ]
            return pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "time_index": [0, 1, 2],
                    "instrument_id": ["SPY"] * 3,
                    "close": [100.0, 100.5, 101.0],
                    "y": [0.0, 1.0, 0.0],
                },
            )

        def get_binding_stats(self) -> tuple[MarketBindingStats, ...]:
            return ()

    monkeypatch.setattr(
        "ml.data.TFTDatasetBuilder",
        _BuilderStub,
    )

    cfg = TFTDatasetTaskConfig(
        data_dir=tmp_path / "source",
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        include_macro=False,
        include_micro=True,
        include_l2=True,
        micro_base_dir=tmp_path / "micro_override",
        l2_base_dir=tmp_path / "l2_override",
    )

    result = build_tft_dataset(cfg)
    assert recorded["micro_base_dir"] == str(tmp_path / "micro_override")
    assert recorded["l2_base_dir"] == str(tmp_path / "l2_override")
    assert result.dataset_parquet.exists()
