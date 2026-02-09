#!/usr/bin/env python3

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import Mock

import pytest

from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.orchestration.config_resolver import ConfigResolver
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.pipeline_orchestrator_facade_helpers import OrchestratorFacadeHelpers
from ml.registry.dataclasses import StorageKind
from ml.tests.utils.targets import build_default_target_semantics_payload


class _HelperHarness(OrchestratorFacadeHelpers):
    def __init__(self) -> None:
        self.coverage = object()
        self.writer = object()
        self.build_main = None
        self.teacher_main = None
        self.registry = None
        self.data_registry = None
        self.ingestor = None
        self.service = None
        self.model_registry = None
        self.feature_registry = None
        self.strategy_registry = None
        self.integration_manager_factory = None
        self.dataset_discovery = None
        self._config_resolver = None
        self._discovery_client = None
        self._ingestion_coordinator = None
        self._registry_synchronizer = None
        self._dataset_builder = None
        self._ingestion_backfill = None
        self._ingestion_backfill_binding = None
        self._ingestion_backfill_coverage = None
        self._ingestion_ensure_dataset_registered = None
        self.backfill = lambda **_kwargs: cast(Any, None)
        self.backfill_binding = lambda **_kwargs: {}
        self.backfill_coverage = lambda **_kwargs: []
        self._ensure_dataset_registered = lambda **_kwargs: None


def _dataset_config(tmp_path: Path) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY.XNAS,QQQ.XNAS",
        out_dir=str(tmp_path / "out"),
        target_semantics=build_default_target_semantics_payload(),
    )


def _binding(symbol: str, *, dataset_id: str = "EQUS.MINI", source: str = "configured") -> ResolvedMarketBinding:
    return ResolvedMarketBinding(
        binding_id=f"{dataset_id}:{symbol}",
        symbol=symbol,
        instrument_ids=(symbol,),
        dataset_id=dataset_id,
        descriptor_id=dataset_id,
        schema="ohlcv-1m",
        storage_kind=StorageKind.PARQUET,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source=source,
    )


def test_resolver_delegation_requires_config_resolver(tmp_path: Path) -> None:
    harness = _HelperHarness()
    cfg = _dataset_config(tmp_path)

    with pytest.raises(RuntimeError, match="ConfigResolver not initialized"):
        harness.apply_default_market_inputs(cfg)

    with pytest.raises(RuntimeError, match="ConfigResolver not initialized"):
        harness.collect_symbol_map()

    with pytest.raises(RuntimeError, match="ConfigResolver not initialized"):
        harness.compute_window_start_iso("2025-01-01")

    with pytest.raises(RuntimeError, match="ConfigResolver not initialized"):
        harness.resolve_window_bounds_ns(cfg)

    with pytest.raises(RuntimeError, match="ConfigResolver not initialized"):
        harness.prepare_dataset_config(cfg=cfg, resolved_inputs=None, bindings=())


def test_health_status_parse_symbols_and_discovery_paths() -> None:
    harness = _HelperHarness()
    harness._config_resolver = ConfigResolver()
    harness._ingestion_coordinator = object()  # type: ignore[assignment]
    harness._dataset_builder = object()  # type: ignore[assignment]

    status = harness._build_health_status()
    assert status["coverage_provider"] == "healthy"
    assert status["binding_resolver"] == "healthy"
    assert harness._parse_symbols(" SPY , QQQ ") == ["SPY", "QQQ"]

    assert (
        harness.discover_market_inputs(
            symbol_map={"SPY": ("SPY.XNAS",)},
            schema="ohlcv-1m",
            start_ns=1,
            end_ns=2,
        )
        is None
    )

    harness._discovery_client = cast(
        Any,
        SimpleNamespace(
            discover_market_inputs=lambda **_kwargs: (
                MarketDatasetInput(descriptor_id="EQUS.MINI", symbols=("SPY.XNAS",)),
            ),
        ),
    )
    discovered = harness.discover_market_inputs(
        symbol_map={"SPY": ("SPY.XNAS",)},
        schema="ohlcv-1m",
        start_ns=1,
        end_ns=2,
        dataset_hint="EQUS.MINI",
    )
    assert discovered is not None
    assert discovered[0].descriptor_id == "EQUS.MINI"

    harness._discovery_client = cast(
        Any,
        SimpleNamespace(
            discover_market_inputs=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
    )
    assert (
        harness.discover_market_inputs(
            symbol_map={"SPY": ("SPY.XNAS",)},
            schema="ohlcv-1m",
            start_ns=1,
            end_ns=2,
        )
        is None
    )


def test_patched_descriptor_loader_and_ingestion_backfill_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _HelperHarness()

    with pytest.raises(RuntimeError, match="IngestionCoordinator not initialized"):
        with harness._patched_ingestion_backfill():
            pass

    coordinator = SimpleNamespace(
        backfill=lambda **_kwargs: "original_backfill",
        backfill_binding=lambda **_kwargs: {"k": "v"},
        backfill_coverage=lambda **_kwargs: [(1, 2)],
        _ensure_dataset_registered=lambda **_kwargs: None,
    )
    harness._ingestion_coordinator = cast(Any, coordinator)
    harness.backfill = lambda **_kwargs: cast(Any, "patched_backfill")
    harness.backfill_binding = lambda **_kwargs: {"patched": cast(Any, None)}
    harness.backfill_coverage = lambda **_kwargs: [(10, 20)]
    harness._ensure_dataset_registered = lambda **_kwargs: None

    with harness._patched_ingestion_backfill():
        assert coordinator.backfill() == "patched_backfill"
        assert "patched" in coordinator.backfill_binding()
        assert coordinator.backfill_coverage() == [(10, 20)]

    assert coordinator.backfill() == "original_backfill"
    assert "k" in coordinator.backfill_binding()
    assert coordinator.backfill_coverage() == [(1, 2)]

    def _override_loader() -> tuple[str, ...]:
        return ("override",)

    monkeypatch.setitem(
        sys.modules,
        "ml.orchestration.pipeline_orchestrator",
        SimpleNamespace(load_market_feed_descriptors=_override_loader),
    )
    from ml.config import market_data as market_data_module
    from ml.orchestration import config_resolver as config_resolver_module

    original_market = market_data_module.load_market_feed_descriptors
    original_config = config_resolver_module.load_market_feed_descriptors
    with harness._patched_descriptor_loader():
        assert market_data_module.load_market_feed_descriptors is _override_loader
        assert config_resolver_module.load_market_feed_descriptors is _override_loader
    assert market_data_module.load_market_feed_descriptors is original_market
    assert config_resolver_module.load_market_feed_descriptors is original_config


def test_prepare_dataset_config_handles_bounds_and_binding_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _HelperHarness()
    cfg = _dataset_config(tmp_path)

    # Resolver unavailable: method short-circuits
    assert harness._prepare_dataset_config(cfg) is cfg

    harness._config_resolver = ConfigResolver()
    monkeypatch.setattr(harness, "apply_default_market_inputs", lambda _cfg: cfg)
    monkeypatch.setattr(harness, "collect_symbol_map", lambda **_kwargs: {})
    monkeypatch.setattr(harness, "resolve_window_bounds_ns", lambda _cfg: (1,))
    assert harness._prepare_dataset_config(cfg) is cfg

    bindings = (_binding("SPY.XNAS"),)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        harness,
        "collect_symbol_map",
        lambda **_kwargs: {".XNAS": (), "SPY.XNAS": ("SPY.XNAS",)},
    )
    monkeypatch.setattr(harness, "resolve_window_bounds_ns", lambda _cfg: (1, 2))
    monkeypatch.setattr(harness, "resolve_market_inputs", lambda **_kwargs: (None, bindings))
    monkeypatch.setattr(
        harness,
        "filter_candidate_bindings",
        lambda **kwargs: kwargs["candidates"] if kwargs["symbol"] == "SPY.XNAS" else (),
    )

    def _capture_prepare(
        *,
        cfg: DatasetBuildConfig,
        resolved_inputs: tuple[Any, ...] | None,
        bindings: tuple[Any, ...],
    ) -> DatasetBuildConfig:
        captured["resolved_inputs"] = resolved_inputs
        captured["bindings"] = bindings
        return cfg

    monkeypatch.setattr(harness, "prepare_dataset_config", _capture_prepare)
    prepared = harness._prepare_dataset_config(cfg)
    assert prepared is cfg
    assert captured["bindings"] == bindings
    resolved_inputs = cast(tuple[MarketDatasetInput, ...], captured["resolved_inputs"])
    assert resolved_inputs[0].descriptor_id == "EQUS.MINI"
    assert resolved_inputs[0].schema_override == "ohlcv-1m"


def test_guard_metadata_and_static_helper_methods(tmp_path: Path) -> None:
    harness = _HelperHarness()
    cfg = _dataset_config(tmp_path)

    with pytest.raises(RuntimeError, match="RegistrySynchronizer not initialized"):
        harness._guard_dataset_metadata(cfg=cfg, metadata=cast(Any, object()))

    harness._registry_synchronizer = cast(Any, SimpleNamespace())
    with pytest.raises(AttributeError, match="missing _guard_dataset_metadata"):
        harness._guard_dataset_metadata(cfg=cfg, metadata=cast(Any, object()))

    guard = Mock()
    harness._registry_synchronizer = cast(Any, SimpleNamespace(_guard_dataset_metadata=guard))
    metadata = cast(Any, object())
    harness._guard_dataset_metadata(cfg=cfg, metadata=metadata)
    guard.assert_called_once_with(cfg=cfg, metadata=metadata)

    metadata_only = SimpleNamespace(
        metadata=SimpleNamespace(overall_window=None, ts_event_start=None, ts_event_end=None),
    )
    assert OrchestratorFacadeHelpers._infer_dataset_row_count(metadata_only) == 0

    csv_only = tmp_path / "dataset.csv"
    csv_only.write_text("timestamp,close\n", encoding="utf-8")
    assert OrchestratorFacadeHelpers._infer_dataset_row_count(SimpleNamespace(dataset_csv=csv_only)) == 0
    csv_only.write_text("timestamp,close\n1,100\n", encoding="utf-8")
    assert OrchestratorFacadeHelpers._infer_dataset_row_count(SimpleNamespace(dataset_csv=csv_only)) is None

    assert OrchestratorFacadeHelpers._resolve_instrument_ids(cfg, ("SPY.XNAS", "")) == ("SPY.XNAS",)
    cfg_with_instruments = DatasetBuildConfig(
        data_dir=cfg.data_dir,
        symbols="SPY.XNAS,QQQ.XNAS",
        out_dir=cfg.out_dir,
        instrument_ids=("AAPL.XNAS", " "),
        target_semantics=cfg.target_semantics,
    )
    assert OrchestratorFacadeHelpers._resolve_instrument_ids(cfg_with_instruments) == ("AAPL.XNAS",)
    assert OrchestratorFacadeHelpers._resolve_instrument_ids(cfg) == ("SPY.XNAS", "QQQ.XNAS")
    assert OrchestratorFacadeHelpers._infer_default_schema(cfg) == "ohlcv-1m"

    assert OrchestratorFacadeHelpers._binding_priority_key(_binding("SPY.XNAS")) == (0, "EQUS.MINI")
    assert OrchestratorFacadeHelpers._binding_priority_key(_binding("SPY.XNAS", dataset_id="XNAS.ITCH")) == (
        1,
        "XNAS.ITCH",
    )
    assert OrchestratorFacadeHelpers._binding_priority_key(_binding("SPY.XNAS", dataset_id="OTHER")) == (
        2,
        "OTHER",
    )

    collected = OrchestratorFacadeHelpers._collect_instrument_ids(
        (_binding("SPY.XNAS"), _binding("QQQ.XNAS")),
        existing=("spy.xnas", " "),
    )
    assert collected == ("SPY.XNAS", "QQQ.XNAS")

    converted = cast(datetime, OrchestratorFacadeHelpers._ns_to_datetime(1_000_000_000))
    assert converted.year == 1970
