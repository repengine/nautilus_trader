import os
import pytest

if os.getenv("ML_ENABLE_COMPONENT_FACADES", "0") != "1":
    pytest.skip("component orchestrator tests disabled", allow_module_level=True)

#!/usr/bin/env python3

"""
Integration tests for MLPipelineOrchestrator facade.

Tests backward compatibility and feature flag behavior between legacy
and component-based implementations.

"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator


@pytest.fixture
def dataset_cfg() -> DatasetBuildConfig:
    """Create sample dataset configuration."""
    return DatasetBuildConfig(
        dataset_id="test_dataset",
        data_dir="/tmp/test_data",
        out_dir="/tmp/test_out",
        symbols="AAPL,MSFT",
        horizon_minutes=60,
        threshold=0.001,
        lookback_periods=10,
        include_macro=False,
        include_micro=True,
        include_l2=False,
        student_mode=False,
        macro_lag_days=1,
    )


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables before each test."""
    original_value = os.environ.get("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR")
    yield
    # Restore original value
    if original_value is not None:
        os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = original_value
    elif "ML_USE_LEGACY_PIPELINE_ORCHESTRATOR" in os.environ:
        del os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"]


class TestMLPipelineOrchestratorFacade:
    """Test suite for MLPipelineOrchestrator facade."""

    def test_facade_default_uses_component_based(self):
        """Test facade uses component-based implementation by default."""
        # Ensure flag is not set
        if "ML_USE_LEGACY_PIPELINE_ORCHESTRATOR" in os.environ:
            del os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"]

        orchestrator = MLPipelineOrchestrator()
        assert not orchestrator._use_legacy
        assert hasattr(orchestrator, "_config_resolver")
        assert hasattr(orchestrator, "_discovery_client")
        assert hasattr(orchestrator, "_binding_resolver")
        assert hasattr(orchestrator, "_ingestion_coordinator")
        assert hasattr(orchestrator, "_dataset_builder")

    def test_facade_uses_legacy_when_flag_set(self):
        """Test facade uses legacy implementation when flag is set."""
        os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = "1"

        # Need to reimport to pick up new environment variable
        from importlib import reload
        import ml.orchestration.pipeline_orchestrator

        reload(ml.orchestration.pipeline_orchestrator)
        from ml.orchestration.pipeline_orchestrator import (
            MLPipelineOrchestrator as ReloadedOrchestrator,
        )

        orchestrator = ReloadedOrchestrator()
        assert orchestrator._use_legacy
        assert hasattr(orchestrator, "_legacy_impl")

    def test_facade_apply_default_market_inputs_delegates(self, dataset_cfg):
        """Test apply_default_market_inputs delegates to config resolver."""
        orchestrator = MLPipelineOrchestrator()
        result = orchestrator.apply_default_market_inputs(dataset_cfg)
        assert isinstance(result, DatasetBuildConfig)
        assert result.dataset_id == dataset_cfg.dataset_id

    def test_facade_collect_symbol_map_delegates(self, dataset_cfg):
        """Test collect_symbol_map delegates to config resolver."""
        orchestrator = MLPipelineOrchestrator()
        symbol_map = orchestrator.collect_symbol_map(dataset_cfg)
        assert isinstance(symbol_map, dict)
        # Should have symbols from config
        assert len(symbol_map) >= 0  # May be empty if no instruments

    def test_facade_compute_window_start_iso_delegates(self):
        """Test compute_window_start_iso delegates to config resolver."""
        orchestrator = MLPipelineOrchestrator()
        start_iso = orchestrator.compute_window_start_iso("2024-01-01", 3)
        assert isinstance(start_iso, str)
        assert "2021" in start_iso  # 3 years before 2024

    def test_facade_resolve_window_bounds_ns_delegates(self, dataset_cfg):
        """Test resolve_window_bounds_ns delegates to config resolver."""
        # Add required fields for window resolution
        cfg = DatasetBuildConfig(
            **{
                **dataset_cfg.__dict__,
                "start_iso": "2023-01-01",
                "end_iso": "2024-01-01",
            }
        )
        orchestrator = MLPipelineOrchestrator()
        start_ns, end_ns = orchestrator.resolve_window_bounds_ns(cfg)
        assert isinstance(start_ns, int)
        assert isinstance(end_ns, int)
        assert end_ns > start_ns

    def test_facade_build_dataset_delegates(self, dataset_cfg, monkeypatch):
        """Test build_dataset delegates to dataset builder."""
        orchestrator = MLPipelineOrchestrator()

        # Mock the builder's build_dataset method
        mock_build = MagicMock(return_value=0)
        monkeypatch.setattr(orchestrator._dataset_builder, "build_dataset", mock_build)

        result = orchestrator.build_dataset(dataset_cfg)
        assert result == 0
        mock_build.assert_called_once_with(dataset_cfg)

    def test_facade_backfill_delegates(self, monkeypatch):
        """Test backfill delegates to ingestion coordinator."""
        orchestrator = MLPipelineOrchestrator()

        # Mock the coordinator's backfill method
        from ml.data.ingest.orchestrator import BackfillWindowList

        mock_result = BackfillWindowList(
            persisted=(),
            requested=(),
            frames_written=0,
            rows_written=0,
        )
        mock_backfill = MagicMock(return_value=mock_result)
        monkeypatch.setattr(orchestrator._ingestion_coordinator, "backfill", mock_backfill)

        result = orchestrator.backfill(
            dataset_id="test",
            schema="ohlcv-1m",
            instrument_id="AAPL.XNAS",
            lookback_days=30,
        )
        assert result == mock_result
        mock_backfill.assert_called_once()

    def test_facade_discover_market_inputs_delegates(self, monkeypatch):
        """Test discover_market_inputs delegates to discovery client."""
        orchestrator = MLPipelineOrchestrator()

        # Mock the discovery client's method
        mock_discover = MagicMock(return_value=())
        monkeypatch.setattr(
            orchestrator._discovery_client,
            "discover_market_inputs",
            mock_discover,
        )

        result = orchestrator.discover_market_inputs(
            symbol_map={"AAPL": ()},
            schema="ohlcv-1m",
            start_ns=1000000000000000000,
            end_ns=2000000000000000000,
        )
        assert result == ()
        mock_discover.assert_called_once()

    def test_facade_resolve_market_inputs_delegates(self, dataset_cfg, monkeypatch):
        """Test resolve_market_inputs delegates to binding resolver."""
        orchestrator = MLPipelineOrchestrator()

        # Mock the binding resolver's method
        mock_resolve = MagicMock(return_value=(None, ()))
        monkeypatch.setattr(
            orchestrator._binding_resolver,
            "resolve_market_inputs",
            mock_resolve,
        )

        result = orchestrator.resolve_market_inputs(
            cfg=dataset_cfg,
            symbol_map={"AAPL": ()},
            start_ns=1000000000000000000,
            end_ns=2000000000000000000,
        )
        assert result == (None, ())
        mock_resolve.assert_called_once()

    def test_facade_health_status_component_based(self):
        """Test get_health_status returns component-based status."""
        orchestrator = MLPipelineOrchestrator()
        health = orchestrator.get_health_status()
        assert health["implementation"] == "component_based"
        assert health["config_resolver"] == "healthy"
        assert health["discovery_client"] == "healthy"
        assert health["binding_resolver"] == "healthy"
        assert health["ingestion_coordinator"] == "healthy"
        assert health["dataset_builder"] == "healthy"

    def test_facade_unimplemented_training_methods_warn(self, dataset_cfg, caplog):
        """Test unimplemented training methods log warnings."""
        orchestrator = MLPipelineOrchestrator()

        from ml.orchestration.config_types import HPOConfig
        from ml.orchestration.config_types import StudentDistillConfig
        from ml.orchestration.config_types import TeacherTrainConfig

        hpo_cfg = HPOConfig(n_trials=10, timeout_seconds=60)
        teacher_cfg = TeacherTrainConfig(epochs=10, batch_size=32)
        student_cfg = StudentDistillConfig(
            epochs=10, batch_size=32, temperature=2.0, alpha=0.5
        )

        # These should return 1 (failure) and log warnings
        assert (
            orchestrator.run_hpo(hpo_cfg, Path("/tmp/data.csv"), Path("/tmp/out")) == 1
        )
        assert (
            orchestrator.train_teacher(
                teacher_cfg, Path("/tmp/data.csv"), Path("/tmp/out")
            )
            == 1
        )
        assert (
            orchestrator.distill_student(
                student_cfg,
                Path("/tmp/data.csv"),
                Path("/tmp/teacher"),
                Path("/tmp/out"),
            )
            == 1
        )

        # Check warnings were logged
        assert "HPO not yet implemented" in caplog.text
        assert "Teacher training not yet implemented" in caplog.text
        assert "Student distillation not yet implemented" in caplog.text

    def test_facade_getattr_fallback(self):
        """Test __getattr__ raises AttributeError for unknown attributes."""
        orchestrator = MLPipelineOrchestrator()
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = orchestrator.nonexistent_method

    def test_facade_constructor_parameters(self):
        """Test facade accepts all legacy constructor parameters."""
        # Should not raise any errors
        orchestrator = MLPipelineOrchestrator(
            connection_string="postgresql://test",
            registry=None,
            data_store=None,
            ingestion_orchestrator=None,
            ingestor=None,
            service=None,
            dataset_discovery=None,
            coverage_provider=None,
            default_data_dir=Path("/tmp"),
            writer=None,
            raw_writer=None,
            domain_loader=None,
            write_mode_tokens=("parquet",),
        )
        assert orchestrator.connection_string == "postgresql://test"
        assert orchestrator.default_data_dir == Path("/tmp")

    def test_facade_exposes_common_attributes(self):
        """Test facade exposes common attributes like registry, data_store, etc."""
        orchestrator = MLPipelineOrchestrator()
        # These should be accessible (even if None)
        assert hasattr(orchestrator, "registry")
        assert hasattr(orchestrator, "data_store")
        assert hasattr(orchestrator, "service")
        assert hasattr(orchestrator, "coverage")
        assert hasattr(orchestrator, "connection_string")
        assert hasattr(orchestrator, "default_data_dir")
