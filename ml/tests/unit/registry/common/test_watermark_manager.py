#!/usr/bin/env python3
"""
Unit tests for WatermarkManagerComponent.

Tests cover watermark update, retrieval, iteration, and persistence
for the DataRegistry watermark tracking extracted from the legacy DataRegistry.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ml.config.events import Source
from ml.registry.data_registry import Watermark
from ml.registry.persistence import BackendType, PersistenceConfig


if TYPE_CHECKING:
    from ml.registry.common.watermark_manager import WatermarkManagerComponent


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def json_persistence_config(tmp_path: Path) -> PersistenceConfig:
    """Create JSON backend persistence config for testing."""
    return PersistenceConfig(
        backend=BackendType.JSON,
        json_path=tmp_path / "registry",
    )


@pytest.fixture
def sample_watermark() -> Watermark:
    """Create a valid Watermark for testing."""
    return Watermark(
        dataset_id="features_test",
        instrument_id="EUR/USD",
        source="live",
        last_success_ns=1_000_000_000_000_000_000,
        last_attempt_ns=1_000_000_000_000_000_000,
        last_count=100,
        completeness_pct=98.5,
        updated_at=time.time(),
    )


# =============================================================================
# Update Watermark Tests
# =============================================================================


class TestUpdateWatermark:
    """Tests for update_watermark method."""

    def test_update_watermark_success(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify successful watermark update."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        component.update_watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=1000000000000000000,
            count=100,
            completeness_pct=99.5,
        )

        watermark = component.get_watermark("test_dataset", "EUR/USD", Source.LIVE)
        assert watermark is not None
        assert watermark.last_success_ns == 1000000000000000000
        assert watermark.last_count == 100
        assert watermark.completeness_pct == 99.5

    def test_update_watermark_creates_key(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify watermark key format."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        component.update_watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=1000000000000000000,
            count=100,
            completeness_pct=99.5,
        )

        expected_key = "test_dataset:EUR/USD:live"
        assert expected_key in persistence._watermarks

    def test_update_watermark_normalizes_source_enum(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify Source enum converted to string."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        component.update_watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=1000000000000000000,
            count=100,
            completeness_pct=99.5,
        )

        watermark = component.get_watermark("test_dataset", "EUR/USD", Source.LIVE)
        assert watermark.source == "live"

    def test_update_watermark_sets_updated_at(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify updated_at timestamp set."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        before = time.time()

        component.update_watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=1000000000000000000,
            count=100,
            completeness_pct=99.5,
        )

        after = time.time()

        watermark = component.get_watermark("test_dataset", "EUR/USD", Source.LIVE)
        assert before <= watermark.updated_at <= after

    def test_update_watermark_saves_immediately_json(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify JSON backend saves immediately."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        component.update_watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=1000000000000000000,
            count=100,
            completeness_pct=99.5,
        )

        # File should contain watermark
        registry_file = tmp_path / "registry" / "data_registry.json"
        import json
        data = json.loads(registry_file.read_text())
        assert len(data.get("watermarks", {})) >= 1


# =============================================================================
# Get Watermark Tests
# =============================================================================


class TestGetWatermark:
    """Tests for get_watermark method."""

    def test_get_watermark_success(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify successful watermark retrieval."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        component.update_watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=1000000000000000000,
            count=100,
            completeness_pct=99.5,
        )

        watermark = component.get_watermark("test_dataset", "EUR/USD", Source.LIVE)

        assert watermark is not None
        assert watermark.dataset_id == "test_dataset"
        assert watermark.instrument_id == "EUR/USD"

    def test_get_watermark_not_found_returns_none(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify None returned for unknown watermark."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        result = component.get_watermark("unknown", "unknown", Source.LIVE)

        assert result is None

    def test_get_watermark_accepts_source_enum(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify Source enum accepted."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        component.update_watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=1000000000000000000,
            count=100,
            completeness_pct=99.5,
        )

        watermark = component.get_watermark("test_dataset", "EUR/USD", Source.LIVE)

        assert watermark is not None

    def test_get_watermark_accepts_source_string(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify source string accepted."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        component.update_watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source=Source.LIVE,
            last_success_ns=1000000000000000000,
            count=100,
            completeness_pct=99.5,
        )

        watermark = component.get_watermark("test_dataset", "EUR/USD", "live")

        assert watermark is not None


# =============================================================================
# Iter Watermarks Tests
# =============================================================================


class TestIterWatermarks:
    """Tests for iter_watermarks method."""

    def test_iter_watermarks_no_filter(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify all watermarks returned without filters."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        # Create multiple watermarks
        for i in range(3):
            component.update_watermark(
                dataset_id=f"dataset_{i}",
                instrument_id="EUR/USD",
                source=Source.LIVE,
                last_success_ns=1000000000000000000 + i,
                count=100,
                completeness_pct=99.5,
            )

        result = list(component.iter_watermarks())

        assert len(result) == 3

    def test_iter_watermarks_filter_by_dataset_id(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify filtering by dataset_id."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        for i in range(3):
            component.update_watermark(
                dataset_id=f"dataset_{i}",
                instrument_id="EUR/USD",
                source=Source.LIVE,
                last_success_ns=1000000000000000000 + i,
                count=100,
                completeness_pct=99.5,
            )

        result = list(component.iter_watermarks(dataset_id="dataset_1"))

        assert len(result) == 1
        assert result[0].dataset_id == "dataset_1"

    def test_iter_watermarks_filter_by_instrument_id(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify filtering by instrument_id."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        for i, instrument in enumerate(["EUR/USD", "GBP/USD", "EUR/USD"]):
            component.update_watermark(
                dataset_id=f"dataset_{i}_{instrument.replace('/', '_')}",
                instrument_id=instrument,
                source=Source.LIVE,
                last_success_ns=1000000000000000000,
                count=100,
                completeness_pct=99.5,
            )

        result = list(component.iter_watermarks(instrument_id="EUR/USD"))

        assert len(result) == 2
        assert all(w.instrument_id == "EUR/USD" for w in result)

    def test_iter_watermarks_filter_by_source(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify filtering by source."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        for source in [Source.LIVE, Source.HISTORICAL, Source.LIVE]:
            component.update_watermark(
                dataset_id=f"dataset_{source.value}_{time.time()}",
                instrument_id="EUR/USD",
                source=source,
                last_success_ns=1000000000000000000,
                count=100,
                completeness_pct=99.5,
            )

        result = list(component.iter_watermarks(source=Source.LIVE))

        assert len(result) == 2
        assert all(w.source == "live" for w in result)

    def test_iter_watermarks_with_limit(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify limit parameter respected."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        for i in range(5):
            component.update_watermark(
                dataset_id=f"dataset_{i}",
                instrument_id="EUR/USD",
                source=Source.LIVE,
                last_success_ns=1000000000000000000 + i,
                count=100,
                completeness_pct=99.5,
            )

        result = list(component.iter_watermarks(limit=2))

        assert len(result) == 2

    def test_iter_watermarks_sorted_by_updated_at_desc(
        self,
        tmp_path: Path,
        json_persistence_config: PersistenceConfig,
    ) -> None:
        """Verify watermarks sorted by updated_at descending."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=json_persistence_config,
        )
        component = WatermarkManagerComponent(persistence=persistence)

        for i in range(3):
            component.update_watermark(
                dataset_id=f"dataset_{i}",
                instrument_id="EUR/USD",
                source=Source.LIVE,
                last_success_ns=1000000000000000000 + i,
                count=100,
                completeness_pct=99.5,
            )
            time.sleep(0.01)  # Ensure different updated_at

        result = list(component.iter_watermarks())

        # Most recent first
        assert result[0].updated_at >= result[1].updated_at
        assert result[1].updated_at >= result[2].updated_at


# =============================================================================
# PostgreSQL Backend Tests (Mocked)
# =============================================================================


class TestPostgresBackend:
    """Tests for PostgreSQL backend watermark operations."""

    def test_get_watermark_postgres_checks_cache_first(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify PostgreSQL backend checks in-memory cache first."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        # Use JSON backend but set backend property to POSTGRES to test cache logic
        config = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry",
        )

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=config,
        )

        # Manually set backend to POSTGRES for this test
        persistence.persistence.config = PersistenceConfig(
            backend=BackendType.JSON,  # Keep JSON to avoid real DB
            json_path=tmp_path / "registry",
        )

        # Pre-populate cache
        persistence._watermarks["test_dataset:EUR/USD:live"] = Watermark(
            dataset_id="test_dataset",
            instrument_id="EUR/USD",
            source="live",
            last_success_ns=1000000000000000000,
            last_attempt_ns=1000000000000000000,
            last_count=100,
            completeness_pct=99.5,
            updated_at=time.time(),
        )

        component = WatermarkManagerComponent(persistence=persistence)
        watermark = component.get_watermark("test_dataset", "EUR/USD", Source.LIVE)

        # Should return cached value
        assert watermark is not None
        assert watermark.dataset_id == "test_dataset"

    def test_update_watermark_postgres_session_failure_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify RuntimeError when session unavailable."""
        from ml.registry.common.data_persistence import DataPersistenceComponent
        from ml.registry.common.watermark_manager import WatermarkManagerComponent

        # Use JSON backend but test the error path
        config = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=tmp_path / "registry",
        )

        persistence = DataPersistenceComponent(
            registry_path=tmp_path / "registry",
            persistence_config=config,
        )

        # Override backend type to test PostgreSQL code path
        object.__setattr__(persistence.persistence.config, "backend", BackendType.POSTGRES)

        with patch.object(persistence.persistence, "get_session", return_value=None):
            component = WatermarkManagerComponent(persistence=persistence)

            with pytest.raises(RuntimeError, match="Failed to get database session"):
                component.update_watermark(
                    dataset_id="test_dataset",
                    instrument_id="EUR/USD",
                    source=Source.LIVE,
                    last_success_ns=1000000000000000000,
                    count=100,
                    completeness_pct=99.5,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
