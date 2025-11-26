"""
Unit tests for DailyUpdateOrchestratorComponent.

Tests the daily update orchestration logic extracted from DataScheduler,
covering:
- Successful pipeline runs with proper metrics recording
- Orchestrator mode vs direct collection mode
- Feature computation stage skipping when no feature engineer
- Failure handling and metrics recording
- Pipeline stage timing tracking

"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from ml.data.common.daily_update_orchestrator import (
    DailyUpdateOrchestratorComponent,
    DailyUpdateOrchestratorProtocol,
    track_pipeline_stage,
)


class TestDailyUpdateOrchestratorComponent:
    """Tests for DailyUpdateOrchestratorComponent."""

    @pytest.fixture
    def component(self) -> DailyUpdateOrchestratorComponent:
        """Create component instance for testing."""
        return DailyUpdateOrchestratorComponent()

    @pytest.fixture
    def mock_functions(self) -> dict[str, MagicMock]:
        """Create mock functions for pipeline stages."""
        return {
            "collect_latest_data_fn": MagicMock(),
            "collect_via_orchestrator_fn": MagicMock(),
            "compute_features_fn": MagicMock(),
            "clean_old_data_fn": MagicMock(),
        }

    def test_run_daily_update_success(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test successful daily update runs all stages."""
        feature_engineer = MagicMock()

        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ) as mock_runs, patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ) as mock_latency:
            component.run_daily_update(
                use_orchestrator=False,
                feature_engineer=feature_engineer,
                **mock_functions,
            )

        # Verify direct collection was called (not orchestrator)
        mock_functions["collect_latest_data_fn"].assert_called_once()
        mock_functions["collect_via_orchestrator_fn"].assert_not_called()
        mock_functions["compute_features_fn"].assert_called_once()
        mock_functions["clean_old_data_fn"].assert_called_once()

        # Verify success metrics recorded
        mock_runs.labels.assert_called_with(status="success")
        mock_runs.labels.return_value.inc.assert_called_once()

    def test_run_daily_update_with_orchestrator_mode(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test orchestrator mode uses correct collection function."""
        feature_engineer = MagicMock()

        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ), patch("ml.data.common.daily_update_orchestrator.pipeline_stage_latency"):
            component.run_daily_update(
                use_orchestrator=True,
                feature_engineer=feature_engineer,
                **mock_functions,
            )

        # Verify orchestrator collection was called (not direct)
        mock_functions["collect_via_orchestrator_fn"].assert_called_once()
        mock_functions["collect_latest_data_fn"].assert_not_called()
        mock_functions["compute_features_fn"].assert_called_once()
        mock_functions["clean_old_data_fn"].assert_called_once()

    def test_run_daily_update_with_feature_computation(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test feature computation stage runs when feature engineer provided."""
        feature_engineer = MagicMock()

        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ), patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ) as mock_latency:
            component.run_daily_update(
                use_orchestrator=False,
                feature_engineer=feature_engineer,
                **mock_functions,
            )

        # Verify feature computation was called
        mock_functions["compute_features_fn"].assert_called_once()

        # Verify feature_computation stage was tracked
        stage_calls = [c for c in mock_latency.labels.call_args_list]
        stages_tracked = [c[1]["stage"] for c in stage_calls]
        assert "feature_computation" in stages_tracked

    def test_run_daily_update_without_feature_computation(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test feature computation stage skipped when no feature engineer."""
        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ), patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ) as mock_latency:
            component.run_daily_update(
                use_orchestrator=False,
                feature_engineer=None,  # No feature engineer
                **mock_functions,
            )

        # Verify feature computation was NOT called
        mock_functions["compute_features_fn"].assert_not_called()

        # Verify feature_computation stage was NOT tracked
        stage_calls = [c for c in mock_latency.labels.call_args_list]
        stages_tracked = [c[1]["stage"] for c in stage_calls]
        assert "feature_computation" not in stages_tracked

    def test_run_daily_update_failure_metrics(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test failure metrics recorded on exception."""
        mock_functions["collect_latest_data_fn"].side_effect = RuntimeError(
            "Collection failed"
        )

        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ) as mock_runs, patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ) as mock_latency:
            with pytest.raises(RuntimeError, match="Collection failed"):
                component.run_daily_update(
                    use_orchestrator=False,
                    feature_engineer=MagicMock(),
                    **mock_functions,
                )

        # Verify failure metrics recorded
        mock_runs.labels.assert_called_with(status="failure")
        mock_runs.labels.return_value.inc.assert_called_once()

        # Verify complete_pipeline latency still recorded
        latency_stages = [c[1]["stage"] for c in mock_latency.labels.call_args_list]
        assert "complete_pipeline" in latency_stages

    def test_run_daily_update_pipeline_timing(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test pipeline timing is recorded in metrics."""
        # Make functions take measurable time
        def slow_collect() -> None:
            time.sleep(0.01)

        mock_functions["collect_latest_data_fn"].side_effect = slow_collect

        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ), patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ) as mock_latency:
            component.run_daily_update(
                use_orchestrator=False,
                feature_engineer=None,
                **mock_functions,
            )

        # Verify timing was recorded for complete_pipeline
        complete_call = None
        for c in mock_latency.labels.call_args_list:
            if c[1]["stage"] == "complete_pipeline":
                complete_call = c
                break

        assert complete_call is not None
        mock_latency.labels.return_value.observe.assert_called()

    def test_pipeline_stage_tracking_data_collection(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test data_collection stage is properly tracked."""
        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ), patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ) as mock_latency:
            component.run_daily_update(
                use_orchestrator=False,
                feature_engineer=None,
                **mock_functions,
            )

        # Verify data_collection stage was tracked
        stage_calls = [c[1]["stage"] for c in mock_latency.labels.call_args_list]
        assert "data_collection" in stage_calls

    def test_pipeline_stage_tracking_feature_computation(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test feature_computation stage is properly tracked."""
        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ), patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ) as mock_latency:
            component.run_daily_update(
                use_orchestrator=False,
                feature_engineer=MagicMock(),  # Enable feature computation
                **mock_functions,
            )

        # Verify feature_computation stage was tracked
        stage_calls = [c[1]["stage"] for c in mock_latency.labels.call_args_list]
        assert "feature_computation" in stage_calls

    def test_pipeline_stage_tracking_cleanup(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test data_cleanup stage is properly tracked."""
        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ), patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ) as mock_latency:
            component.run_daily_update(
                use_orchestrator=False,
                feature_engineer=None,
                **mock_functions,
            )

        # Verify data_cleanup stage was tracked
        stage_calls = [c[1]["stage"] for c in mock_latency.labels.call_args_list]
        assert "data_cleanup" in stage_calls

    def test_pipeline_runs_total_metric_success(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test pipeline_runs_total metric increments on success."""
        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ) as mock_runs, patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ):
            component.run_daily_update(
                use_orchestrator=False,
                feature_engineer=None,
                **mock_functions,
            )

        # Verify success label used
        mock_runs.labels.assert_called_with(status="success")
        mock_runs.labels.return_value.inc.assert_called_once()

    def test_pipeline_runs_total_metric_failure(
        self,
        component: DailyUpdateOrchestratorComponent,
        mock_functions: dict[str, MagicMock],
    ) -> None:
        """Test pipeline_runs_total metric increments on failure."""
        mock_functions["clean_old_data_fn"].side_effect = RuntimeError("Cleanup failed")

        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_runs_total"
        ) as mock_runs, patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ):
            with pytest.raises(RuntimeError, match="Cleanup failed"):
                component.run_daily_update(
                    use_orchestrator=False,
                    feature_engineer=None,
                    **mock_functions,
                )

        # Verify failure label used
        mock_runs.labels.assert_called_with(status="failure")
        mock_runs.labels.return_value.inc.assert_called_once()


class TestTrackPipelineStage:
    """Tests for track_pipeline_stage context manager."""

    def test_track_pipeline_stage_records_duration(self) -> None:
        """Test that track_pipeline_stage records stage duration."""
        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ) as mock_latency:
            with track_pipeline_stage("test_stage"):
                time.sleep(0.01)

        mock_latency.labels.assert_called_with(stage="test_stage")
        mock_latency.labels.return_value.observe.assert_called_once()

        # Verify duration is positive
        observed_duration = mock_latency.labels.return_value.observe.call_args[0][0]
        assert observed_duration > 0

    def test_track_pipeline_stage_records_on_exception(self) -> None:
        """Test that track_pipeline_stage records duration even on exception."""
        with patch(
            "ml.data.common.daily_update_orchestrator.pipeline_stage_latency"
        ) as mock_latency:
            with pytest.raises(ValueError, match="Test error"):
                with track_pipeline_stage("failing_stage"):
                    raise ValueError("Test error")

        # Duration should still be recorded
        mock_latency.labels.assert_called_with(stage="failing_stage")
        mock_latency.labels.return_value.observe.assert_called_once()


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_component_satisfies_protocol(self) -> None:
        """Test that component satisfies the protocol."""
        component = DailyUpdateOrchestratorComponent()

        # Verify component has all protocol methods
        assert hasattr(component, "run_daily_update")
        assert callable(component.run_daily_update)

        # Type check passes (structural typing)
        def accepts_protocol(p: DailyUpdateOrchestratorProtocol) -> None:
            pass

        # This should not raise a type error
        accepts_protocol(component)
