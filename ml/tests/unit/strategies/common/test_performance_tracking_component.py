"""
Tests for PerformanceTrackingComponent.

This module tests the performance tracking component extracted from BaseMLStrategy
as part of the Phase 3.4 decomposition. Tests cover:

- Model performance tracking (new model, existing model, win/loss)
- Trade count and win tracking
- Performance data access (returns copies)
- Empty performance handling
- Metrics recording to Prometheus
- Performance reset functionality
- Best model selection

"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.strategies.common.performance_tracking import (
    PerformanceTrackingComponent,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock logger."""
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def performance_tracking_component(
    mock_logger: MagicMock,
) -> PerformanceTrackingComponent:
    """Create a performance tracking component with tracking enabled."""
    return PerformanceTrackingComponent(
        strategy_id="test_strategy",
        track_performance=True,
        log=mock_logger,
    )


@pytest.fixture
def component_tracking_disabled(
    mock_logger: MagicMock,
) -> PerformanceTrackingComponent:
    """Create a performance tracking component with tracking disabled."""
    return PerformanceTrackingComponent(
        strategy_id="test_strategy",
        track_performance=False,
        log=mock_logger,
    )


# ---------------------------------------------------------------------------
# Test Class: Model Performance - New Model
# ---------------------------------------------------------------------------


class TestUpdateModelPerformanceNewModel:
    """Test updating performance for a new model."""

    def test_update_model_performance_new_model(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify new model entry created."""
        # Initially no model
        assert "model_a" not in performance_tracking_component._model_performance

        # Update performance
        performance_tracking_component.update_model_performance("model_a", profit=100.0)

        # Model should now be tracked
        assert "model_a" in performance_tracking_component._model_performance
        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["total_trades"] == 1
        assert perf["total_profit"] == 100.0
        assert perf["wins"] == 1
        assert perf["losses"] == 0

    def test_update_model_performance_new_model_with_loss(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify new model entry created with initial loss."""
        performance_tracking_component.update_model_performance("model_b", profit=-50.0)

        perf = performance_tracking_component._model_performance["model_b"]
        assert perf["total_trades"] == 1
        assert perf["total_profit"] == -50.0
        assert perf["wins"] == 0
        assert perf["losses"] == 1

    def test_update_model_performance_multiple_new_models(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify multiple new models can be tracked."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_b", profit=200.0)
        performance_tracking_component.update_model_performance("model_c", profit=-50.0)

        assert len(performance_tracking_component._model_performance) == 3
        assert "model_a" in performance_tracking_component._model_performance
        assert "model_b" in performance_tracking_component._model_performance
        assert "model_c" in performance_tracking_component._model_performance


# ---------------------------------------------------------------------------
# Test Class: Model Performance - Existing Model (Profit)
# ---------------------------------------------------------------------------


class TestUpdateModelPerformanceExistingModelProfit:
    """Test updating performance for existing model with profit."""

    def test_update_model_performance_existing_model_profit(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify existing model profit is accumulated."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=50.0)

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["total_profit"] == 150.0
        assert perf["total_trades"] == 2
        assert perf["wins"] == 2

    def test_update_model_performance_consecutive_wins(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify consecutive wins are tracked correctly."""
        for i in range(5):
            performance_tracking_component.update_model_performance(
                "model_a", profit=100.0 + i * 10
            )

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["wins"] == 5
        assert perf["losses"] == 0
        assert perf["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Test Class: Model Performance - Existing Model (Loss)
# ---------------------------------------------------------------------------


class TestUpdateModelPerformanceExistingModelLoss:
    """Test updating performance for existing model with loss."""

    def test_update_model_performance_existing_model_loss(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify existing model loss is accumulated."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=-30.0)

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["total_profit"] == 70.0
        assert perf["total_trades"] == 2
        assert perf["wins"] == 1
        assert perf["losses"] == 1

    def test_update_model_performance_consecutive_losses(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify consecutive losses are tracked correctly."""
        for i in range(3):
            performance_tracking_component.update_model_performance(
                "model_a", profit=-50.0 - i * 10
            )

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["wins"] == 0
        assert perf["losses"] == 3
        assert perf["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Test Class: Model Performance - Trade Count Tracking
# ---------------------------------------------------------------------------


class TestUpdateModelPerformanceTracksCounts:
    """Test trade count tracking."""

    def test_update_model_performance_tracks_count(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify total trades count is incremented."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        assert performance_tracking_component._model_performance["model_a"]["total_trades"] == 1

        performance_tracking_component.update_model_performance("model_a", profit=-50.0)
        assert performance_tracking_component._model_performance["model_a"]["total_trades"] == 2

        performance_tracking_component.update_model_performance("model_a", profit=75.0)
        assert performance_tracking_component._model_performance["model_a"]["total_trades"] == 3

    def test_update_model_performance_count_independent_per_model(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify trade counts are independent per model."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=50.0)
        performance_tracking_component.update_model_performance("model_b", profit=200.0)

        assert performance_tracking_component._model_performance["model_a"]["total_trades"] == 2
        assert performance_tracking_component._model_performance["model_b"]["total_trades"] == 1


# ---------------------------------------------------------------------------
# Test Class: Model Performance - Win Tracking
# ---------------------------------------------------------------------------


class TestUpdateModelPerformanceTracksWins:
    """Test win tracking."""

    def test_update_model_performance_tracks_wins(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify wins are tracked correctly."""
        # 3 wins, 2 losses
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=-50.0)
        performance_tracking_component.update_model_performance("model_a", profit=75.0)
        performance_tracking_component.update_model_performance("model_a", profit=-25.0)
        performance_tracking_component.update_model_performance("model_a", profit=50.0)

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["wins"] == 3
        assert perf["losses"] == 2
        assert perf["accuracy"] == pytest.approx(0.6)

    def test_update_model_performance_zero_profit_is_loss(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify zero profit is counted as loss."""
        performance_tracking_component.update_model_performance("model_a", profit=0.0)

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["wins"] == 0
        assert perf["losses"] == 1


# ---------------------------------------------------------------------------
# Test Class: Get Model Performance - Returns Copy
# ---------------------------------------------------------------------------


class TestGetModelPerformanceReturnsCopy:
    """Test that get_model_performance returns copies."""

    def test_get_model_performance_returns_copy(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify performance data is returned as copy."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)

        perf = performance_tracking_component.get_model_performance("model_a")

        # Modify returned data
        perf["total_trades"] = 999
        perf["custom_field"] = "test"

        # Original should be unchanged
        internal = performance_tracking_component._model_performance["model_a"]
        assert internal["total_trades"] == 1
        assert "custom_field" not in internal

    def test_get_model_performance_all_returns_copy(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify all performance data is returned as copy."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_b", profit=200.0)

        all_perf = performance_tracking_component.get_model_performance()

        # Modify returned data
        all_perf["model_a"]["total_trades"] = 999
        all_perf["model_c"] = {"total_trades": 5}

        # Original should be unchanged
        internal_a = performance_tracking_component._model_performance["model_a"]
        assert internal_a["total_trades"] == 1
        assert "model_c" not in performance_tracking_component._model_performance


# ---------------------------------------------------------------------------
# Test Class: Get Model Performance - Empty
# ---------------------------------------------------------------------------


class TestGetModelPerformanceEmpty:
    """Test get_model_performance with empty data."""

    def test_get_model_performance_empty(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify empty dict returned for non-existent model."""
        perf = performance_tracking_component.get_model_performance("non_existent")
        assert perf == {}

    def test_get_model_performance_all_empty(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify empty dict returned when no models tracked."""
        all_perf = performance_tracking_component.get_model_performance()
        assert all_perf == {}

    def test_get_model_performance_none_returns_all(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify None model_id returns all performance data."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_b", profit=200.0)

        all_perf = performance_tracking_component.get_model_performance(None)

        assert "model_a" in all_perf
        assert "model_b" in all_perf


# ---------------------------------------------------------------------------
# Test Class: Record Metrics Usage
# ---------------------------------------------------------------------------


class TestRecordMetricsUsage:
    """Test metrics recording functionality."""

    def test_record_metrics_usage_increments_counters(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify metrics usage increments counters."""
        # Set up mock metrics
        mock_signals_counter = MagicMock()
        mock_signals_labels = MagicMock()
        mock_signals_counter.labels.return_value = mock_signals_labels

        mock_trades_counter = MagicMock()
        mock_trades_labels = MagicMock()
        mock_trades_counter.labels.return_value = mock_trades_labels

        mock_positions_gauge = MagicMock()
        mock_positions_labels = MagicMock()
        mock_positions_gauge.labels.return_value = mock_positions_labels

        performance_tracking_component._signals_counter = mock_signals_counter
        performance_tracking_component._trades_counter = mock_trades_counter
        performance_tracking_component._positions_gauge = mock_positions_gauge

        # Record metrics
        performance_tracking_component.record_metrics_usage(
            signals_received=5,
            trades_executed=2,
            active_positions=3,
        )

        # Verify counter increments
        mock_signals_counter.labels.assert_called_with(strategy_id="test_strategy")
        mock_signals_labels.inc.assert_called_once_with(5)

        mock_trades_counter.labels.assert_called_with(strategy_id="test_strategy")
        mock_trades_labels.inc.assert_called_once_with(2)

        # Verify gauge set
        mock_positions_gauge.labels.assert_called_with(strategy_id="test_strategy")
        mock_positions_labels.set.assert_called_once_with(3)

    def test_record_metrics_usage_zero_values(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify zero values don't increment counters."""
        mock_signals_counter = MagicMock()
        mock_signals_labels = MagicMock()
        mock_signals_counter.labels.return_value = mock_signals_labels

        mock_trades_counter = MagicMock()
        mock_trades_labels = MagicMock()
        mock_trades_counter.labels.return_value = mock_trades_labels

        performance_tracking_component._signals_counter = mock_signals_counter
        performance_tracking_component._trades_counter = mock_trades_counter

        performance_tracking_component.record_metrics_usage(
            signals_received=0,
            trades_executed=0,
            active_positions=0,
        )

        # Counters should not be incremented for zero values
        mock_signals_labels.inc.assert_not_called()
        mock_trades_labels.inc.assert_not_called()

    def test_record_metrics_usage_handles_none_metrics(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify graceful handling when metrics are None."""
        performance_tracking_component._signals_counter = None
        performance_tracking_component._trades_counter = None
        performance_tracking_component._positions_gauge = None

        # Should not raise
        performance_tracking_component.record_metrics_usage(
            signals_received=5,
            trades_executed=2,
            active_positions=3,
        )

    def test_record_metrics_usage_handles_exception(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify graceful handling when metrics raise exception."""
        mock_counter = MagicMock()
        mock_counter.labels.side_effect = Exception("Metrics error")
        performance_tracking_component._signals_counter = mock_counter

        # Should not raise
        performance_tracking_component.record_metrics_usage(
            signals_received=5,
            trades_executed=2,
            active_positions=3,
        )


# ---------------------------------------------------------------------------
# Test Class: Reset Model Performance
# ---------------------------------------------------------------------------


class TestResetModelPerformance:
    """Test reset model performance functionality."""

    def test_reset_model_performance(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify reset clears all model performance data."""
        # Add some performance data
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_b", profit=200.0)
        performance_tracking_component.update_model_performance("model_c", profit=-50.0)

        assert len(performance_tracking_component._model_performance) == 3

        # Reset
        performance_tracking_component.reset_model_performance()

        # Should be empty
        assert performance_tracking_component._model_performance == {}
        assert performance_tracking_component.get_model_performance() == {}

    def test_reset_model_performance_idempotent(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify reset is idempotent."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)

        # Multiple resets should not cause errors
        performance_tracking_component.reset_model_performance()
        performance_tracking_component.reset_model_performance()
        performance_tracking_component.reset_model_performance()

        assert performance_tracking_component._model_performance == {}

    def test_reset_model_performance_allows_new_tracking(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify tracking works after reset."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.reset_model_performance()
        performance_tracking_component.update_model_performance("model_b", profit=200.0)

        perf = performance_tracking_component.get_model_performance()
        assert "model_a" not in perf
        assert "model_b" in perf


# ---------------------------------------------------------------------------
# Test Class: Get Best Model
# ---------------------------------------------------------------------------


class TestGetBestModelByProfit:
    """Test best model selection by profit."""

    def test_get_best_model_by_profit(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify best model by total profit is returned."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_b", profit=200.0)
        performance_tracking_component.update_model_performance("model_c", profit=150.0)

        best = performance_tracking_component.get_best_model("total_profit")
        assert best == "model_b"

    def test_get_best_model_by_accuracy(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify best model by accuracy is returned."""
        # model_a: 1 win, 1 loss = 50% accuracy
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=-50.0)

        # model_b: 2 wins, 1 loss = 66.7% accuracy
        performance_tracking_component.update_model_performance("model_b", profit=100.0)
        performance_tracking_component.update_model_performance("model_b", profit=50.0)
        performance_tracking_component.update_model_performance("model_b", profit=-25.0)

        # model_c: 3 wins = 100% accuracy
        performance_tracking_component.update_model_performance("model_c", profit=50.0)
        performance_tracking_component.update_model_performance("model_c", profit=40.0)
        performance_tracking_component.update_model_performance("model_c", profit=30.0)

        best = performance_tracking_component.get_best_model("accuracy")
        assert best == "model_c"

    def test_get_best_model_by_total_trades(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify best model by total trades is returned."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=50.0)
        performance_tracking_component.update_model_performance("model_a", profit=25.0)

        performance_tracking_component.update_model_performance("model_b", profit=500.0)

        best = performance_tracking_component.get_best_model("total_trades")
        assert best == "model_a"

    def test_get_best_model_by_wins(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify best model by wins is returned."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=50.0)

        performance_tracking_component.update_model_performance("model_b", profit=100.0)
        performance_tracking_component.update_model_performance("model_b", profit=50.0)
        performance_tracking_component.update_model_performance("model_b", profit=25.0)

        best = performance_tracking_component.get_best_model("wins")
        assert best == "model_b"

    def test_get_best_model_empty(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify None returned when no models tracked."""
        best = performance_tracking_component.get_best_model("total_profit")
        assert best is None

    def test_get_best_model_default_metric(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify default metric is total_profit."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_b", profit=200.0)

        best = performance_tracking_component.get_best_model()
        assert best == "model_b"


# ---------------------------------------------------------------------------
# Test Class: Tracking Disabled
# ---------------------------------------------------------------------------


class TestTrackingDisabled:
    """Test behavior when tracking is disabled."""

    def test_update_model_performance_no_op_when_disabled(
        self,
        component_tracking_disabled: PerformanceTrackingComponent,
    ) -> None:
        """Verify update is no-op when tracking disabled."""
        component_tracking_disabled.update_model_performance("model_a", profit=100.0)

        # Should not be tracked
        assert "model_a" not in component_tracking_disabled._model_performance
        assert component_tracking_disabled.get_model_performance() == {}

    def test_enable_tracking_after_init(
        self,
        component_tracking_disabled: PerformanceTrackingComponent,
    ) -> None:
        """Verify tracking can be enabled after initialization."""
        # Initially disabled
        component_tracking_disabled.update_model_performance("model_a", profit=100.0)
        assert component_tracking_disabled.get_model_performance() == {}

        # Enable tracking
        component_tracking_disabled.track_performance = True

        # Now should track
        component_tracking_disabled.update_model_performance("model_b", profit=200.0)
        assert "model_b" in component_tracking_disabled._model_performance


# ---------------------------------------------------------------------------
# Test Class: Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Test component properties."""

    def test_strategy_id_property(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify strategy_id property."""
        assert performance_tracking_component.strategy_id == "test_strategy"

    def test_track_performance_property(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify track_performance property."""
        assert performance_tracking_component.track_performance is True

    def test_track_performance_setter(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify track_performance setter."""
        performance_tracking_component.track_performance = False
        assert performance_tracking_component.track_performance is False

        performance_tracking_component.track_performance = True
        assert performance_tracking_component.track_performance is True


# ---------------------------------------------------------------------------
# Test Class: Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test component initialization."""

    def test_default_initialization(self) -> None:
        """Verify component initializes with defaults."""
        component = PerformanceTrackingComponent(strategy_id="test")

        assert component.strategy_id == "test"
        assert component.track_performance is False
        assert component._model_performance == {}

    def test_custom_initialization(
        self,
        mock_logger: MagicMock,
    ) -> None:
        """Verify component initializes with custom values."""
        component = PerformanceTrackingComponent(
            strategy_id="custom_strategy",
            track_performance=True,
            log=mock_logger,
        )

        assert component.strategy_id == "custom_strategy"
        assert component.track_performance is True

    def test_no_logger_does_not_crash(self) -> None:
        """Verify component works without logger."""
        component = PerformanceTrackingComponent(
            strategy_id="test",
            track_performance=True,
            log=None,
        )

        # Should not raise
        component.update_model_performance("model_a", profit=100.0)
        perf = component.get_model_performance("model_a")
        assert perf["total_trades"] == 1


# ---------------------------------------------------------------------------
# Test Class: Accuracy Calculation
# ---------------------------------------------------------------------------


class TestAccuracyCalculation:
    """Test accuracy calculation."""

    def test_accuracy_calculation_correct(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify accuracy calculated correctly."""
        # 3 wins, 2 losses = 60% accuracy
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=50.0)
        performance_tracking_component.update_model_performance("model_a", profit=-30.0)
        performance_tracking_component.update_model_performance("model_a", profit=25.0)
        performance_tracking_component.update_model_performance("model_a", profit=-10.0)

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["accuracy"] == pytest.approx(0.6)

    def test_accuracy_100_percent(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify 100% accuracy for all wins."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=50.0)
        performance_tracking_component.update_model_performance("model_a", profit=25.0)

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["accuracy"] == 1.0

    def test_accuracy_0_percent(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify 0% accuracy for all losses."""
        performance_tracking_component.update_model_performance("model_a", profit=-100.0)
        performance_tracking_component.update_model_performance("model_a", profit=-50.0)
        performance_tracking_component.update_model_performance("model_a", profit=-25.0)

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Test Class: Profit Accumulation
# ---------------------------------------------------------------------------


class TestProfitAccumulation:
    """Test profit accumulation."""

    def test_profit_accumulation(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify total profit accumulates correctly."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=50.0)
        performance_tracking_component.update_model_performance("model_a", profit=-30.0)

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["total_profit"] == 120.0

    def test_profit_can_go_negative(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify total profit can be negative."""
        performance_tracking_component.update_model_performance("model_a", profit=-100.0)
        performance_tracking_component.update_model_performance("model_a", profit=-50.0)
        performance_tracking_component.update_model_performance("model_a", profit=30.0)

        perf = performance_tracking_component._model_performance["model_a"]
        assert perf["total_profit"] == -120.0


# ---------------------------------------------------------------------------
# Test Class: Summary Statistics
# ---------------------------------------------------------------------------


class TestSummaryStatistics:
    """Test summary statistics."""

    def test_get_summary_statistics(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify summary statistics are calculated correctly."""
        performance_tracking_component.update_model_performance("model_a", profit=100.0)
        performance_tracking_component.update_model_performance("model_a", profit=-50.0)
        performance_tracking_component.update_model_performance("model_b", profit=200.0)
        performance_tracking_component.update_model_performance("model_b", profit=100.0)

        stats = performance_tracking_component.get_summary_statistics()

        assert stats["total_models"] == 2
        assert stats["total_trades"] == 4
        assert stats["total_profit"] == 350.0
        assert stats["total_wins"] == 3
        assert stats["total_losses"] == 1
        assert stats["overall_accuracy"] == pytest.approx(0.75)

    def test_get_summary_statistics_empty(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify summary statistics for empty data."""
        stats = performance_tracking_component.get_summary_statistics()

        assert stats["total_models"] == 0
        assert stats["total_trades"] == 0
        assert stats["total_profit"] == 0.0
        assert stats["total_wins"] == 0
        assert stats["total_losses"] == 0
        assert stats["overall_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Test Class: Individual Metric Methods
# ---------------------------------------------------------------------------


class TestIndividualMetricMethods:
    """Test individual metric increment/set methods."""

    def test_increment_signals_received(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify increment_signals_received works."""
        mock_counter = MagicMock()
        mock_labels = MagicMock()
        mock_counter.labels.return_value = mock_labels
        performance_tracking_component._signals_counter = mock_counter

        performance_tracking_component.increment_signals_received(3)

        mock_counter.labels.assert_called_with(strategy_id="test_strategy")
        mock_labels.inc.assert_called_once_with(3)

    def test_increment_trades_executed(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify increment_trades_executed works."""
        mock_counter = MagicMock()
        mock_labels = MagicMock()
        mock_counter.labels.return_value = mock_labels
        performance_tracking_component._trades_counter = mock_counter

        performance_tracking_component.increment_trades_executed(2)

        mock_counter.labels.assert_called_with(strategy_id="test_strategy")
        mock_labels.inc.assert_called_once_with(2)

    def test_set_active_positions(
        self,
        performance_tracking_component: PerformanceTrackingComponent,
    ) -> None:
        """Verify set_active_positions works."""
        mock_gauge = MagicMock()
        mock_labels = MagicMock()
        mock_gauge.labels.return_value = mock_labels
        performance_tracking_component._positions_gauge = mock_gauge

        performance_tracking_component.set_active_positions(5)

        mock_gauge.labels.assert_called_with(strategy_id="test_strategy")
        mock_labels.set.assert_called_once_with(5)
