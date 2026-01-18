"""
Unit tests for IngestionCoordinator (facade-only).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ml.data.ingest.orchestrator import BackfillWindowList
from ml.orchestration.ingestion_coordinator import IngestionCoordinator


@pytest.fixture
def coordinator() -> IngestionCoordinator:
    """Create an IngestionCoordinator with minimal dependencies."""
    return IngestionCoordinator(
        coverage=MagicMock(),
        writer=MagicMock(),
        registry=MagicMock(),
        ingestor=MagicMock(),
    )


def test_coordinate_ingestion_sums_rows(coordinator: IngestionCoordinator) -> None:
    """coordinate_ingestion aggregates rows across instruments."""
    coordinator.backfill = MagicMock(
        side_effect=[
            BackfillWindowList(persisted=(), requested=(), frames_written=1, rows_written=10),
            BackfillWindowList(persisted=(), requested=(), frames_written=1, rows_written=5),
        ],
    )

    result = coordinator.coordinate_ingestion(
        dataset_id="databento.ohlcv-1s",
        schema="ohlcv-1s",
        instrument_ids=["SPY.XNAS", "QQQ.XNAS"],
        lookback_days=5,
    )

    assert result["fallback_level"] == "primary"
    assert result["rows_written"] == 15


def test_coordinate_ingestion_falls_back_on_error(coordinator: IngestionCoordinator) -> None:
    """coordinate_ingestion uses fallback when primary backfill fails."""
    coordinator.backfill = MagicMock(side_effect=RuntimeError("boom"))
    coordinator._handle_ingestion_fallback = MagicMock(
        return_value={"rows_written": 0, "fallback_level": "cached"},
    )

    result = coordinator.coordinate_ingestion(
        dataset_id="databento.ohlcv-1s",
        schema="ohlcv-1s",
        instrument_ids=["SPY.XNAS"],
        lookback_days=5,
    )

    assert result["fallback_level"] == "cached"
    coordinator._handle_ingestion_fallback.assert_called_once()


def test_backfill_delegates_to_orchestrator(coordinator: IngestionCoordinator) -> None:
    """backfill delegates to the ingestion orchestrator."""
    orchestrator = MagicMock()
    expected = BackfillWindowList(persisted=(), requested=(), frames_written=0, rows_written=0)
    orchestrator.backfill_gaps.return_value = expected
    coordinator._create_ingestion_orchestrator = MagicMock(return_value=orchestrator)

    result = coordinator.backfill(
        dataset_id="databento.ohlcv-1s",
        schema="ohlcv-1s",
        instrument_id="SPY.XNAS",
        lookback_days=5,
    )

    assert result is expected
    orchestrator.backfill_gaps.assert_called_once()


def test_backfill_binding_delegates_to_orchestrator(coordinator: IngestionCoordinator) -> None:
    """backfill_binding delegates to the ingestion orchestrator."""
    orchestrator = MagicMock()
    orchestrator.backfill_binding.return_value = {"SPY.XNAS": BackfillWindowList(persisted=(), requested=(), frames_written=0, rows_written=0)}
    coordinator._create_ingestion_orchestrator = MagicMock(return_value=orchestrator)

    binding = MagicMock()
    result = coordinator.backfill_binding(binding=binding, lookback_days=5)

    assert "SPY.XNAS" in result
    orchestrator.backfill_binding.assert_called_once_with(binding=binding, lookback_days=5)


def test_handle_ingestion_fallback_primary_uses_backfill(coordinator: IngestionCoordinator) -> None:
    """PRIMARY fallback uses component backfill."""
    coordinator.backfill = MagicMock(
        return_value=BackfillWindowList(persisted=(), requested=(), frames_written=1, rows_written=4),
    )

    result = coordinator._handle_ingestion_fallback(
        dataset_id="databento.ohlcv-1s",
        schema="ohlcv-1s",
        instrument_ids=["SPY.XNAS"],
        lookback_days=5,
        level="primary",
    )

    assert result["fallback_level"] == "primary"
    assert result["rows_written"] == 4


def test_handle_ingestion_fallback_cached_uses_backfill_coverage(
    coordinator: IngestionCoordinator,
) -> None:
    """CACHED fallback uses coverage gaps as a signal."""
    coordinator.backfill_coverage = MagicMock(return_value=[(1, 2), (3, 4)])

    result = coordinator._handle_ingestion_fallback(
        dataset_id="databento.ohlcv-1s",
        schema="ohlcv-1s",
        instrument_ids=["SPY.XNAS"],
        lookback_days=5,
        level="cached",
    )

    assert result["fallback_level"] == "cached"
    assert result["rows_written"] == 2
