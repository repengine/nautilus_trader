"""
Unit tests for DatasetRegistrationComponent.

Tests extracted dataset registration logic from DataScheduler:
- ensure_dataset_registered() for new datasets
- ensure_dataset_registered() for existing datasets
- ensure_dataset_registered() with no registry
- Dataset type mapping for all supported types (bars, trades, tbbo, mbp1)
- Silent failure handling

Test count: 8
Coverage target: 95%

"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.data.common.dataset_registration import DatasetRegistrationComponent
from ml.data.common.dataset_registration import DatasetRegistrationProtocol
from ml.registry.dataclasses import DatasetType


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def component() -> DatasetRegistrationComponent:
    """Create a DatasetRegistrationComponent instance for testing."""
    return DatasetRegistrationComponent()


@pytest.fixture
def mock_registry() -> MagicMock:
    """Create a mock registry for testing."""
    registry = MagicMock()
    return registry


# -----------------------------------------------------------------------------
# Dataset Registration Tests (4 tests)
# -----------------------------------------------------------------------------


class TestEnsureDatasetRegistered:
    """Tests for ensure_dataset_registered method."""

    def test_ensure_dataset_registered_new_dataset(
        self,
        component: DatasetRegistrationComponent,
        mock_registry: MagicMock,
    ) -> None:
        """Test new dataset is registered when manifest doesn't exist."""
        # Setup: get_manifest raises to simulate non-existent dataset
        mock_registry.get_manifest.side_effect = ValueError("Dataset not found")

        with patch(
            "ml.data.dataset_manifest_defaults.build_auto_dataset_manifest"
        ) as mock_build:
            mock_manifest = MagicMock()
            mock_build.return_value = mock_manifest

            component.ensure_dataset_registered(
                registry=mock_registry,
                dataset_id="ohlcv_spy_xnas",
                dataset_type_label="bars",
                location="/data/catalogs/bars",
                retention_days=90,
            )

            # Verify get_manifest was called first
            mock_registry.get_manifest.assert_called_once_with("ohlcv_spy_xnas")

            # Verify manifest was built
            mock_build.assert_called_once()
            call_kwargs = mock_build.call_args.kwargs
            assert call_kwargs["dataset_id"] == "ohlcv_spy_xnas"
            assert call_kwargs["dataset_type"] == DatasetType.BARS
            assert call_kwargs["retention_days"] == 90

            # Verify register_dataset was called with the manifest
            mock_registry.register_dataset.assert_called_once_with(mock_manifest)

    def test_ensure_dataset_registered_existing_dataset(
        self,
        component: DatasetRegistrationComponent,
        mock_registry: MagicMock,
    ) -> None:
        """Test existing dataset is not re-registered."""
        # Setup: get_manifest succeeds (dataset exists)
        mock_registry.get_manifest.return_value = MagicMock()

        component.ensure_dataset_registered(
            registry=mock_registry,
            dataset_id="ohlcv_spy_xnas",
            dataset_type_label="bars",
            location="/data/catalogs/bars",
            retention_days=90,
        )

        # Verify get_manifest was called
        mock_registry.get_manifest.assert_called_once_with("ohlcv_spy_xnas")

        # Verify register_dataset was NOT called (dataset exists)
        mock_registry.register_dataset.assert_not_called()

    def test_ensure_dataset_registered_no_registry(
        self,
        component: DatasetRegistrationComponent,
    ) -> None:
        """Test early return when registry is None."""
        # This should not raise any exception
        component.ensure_dataset_registered(
            registry=None,
            dataset_id="ohlcv_spy_xnas",
            dataset_type_label="bars",
            location="/data/catalogs/bars",
            retention_days=90,
        )
        # No assertion needed - test passes if no exception raised

    def test_ensure_dataset_registered_failure_silent(
        self,
        component: DatasetRegistrationComponent,
        mock_registry: MagicMock,
    ) -> None:
        """Test registration failure is handled silently (debug log only)."""
        # Setup: get_manifest raises, register_dataset also raises
        mock_registry.get_manifest.side_effect = ValueError("Dataset not found")
        mock_registry.register_dataset.side_effect = RuntimeError("DB error")

        with patch(
            "ml.data.dataset_manifest_defaults.build_auto_dataset_manifest"
        ) as mock_build:
            mock_manifest = MagicMock()
            mock_build.return_value = mock_manifest

            # Should not raise - failures are handled silently
            component.ensure_dataset_registered(
                registry=mock_registry,
                dataset_id="ohlcv_spy_xnas",
                dataset_type_label="bars",
                location="/data/catalogs/bars",
                retention_days=90,
            )

            # Verify registration was attempted
            mock_registry.register_dataset.assert_called_once()


# -----------------------------------------------------------------------------
# Dataset Type Mapping Tests (4 tests)
# -----------------------------------------------------------------------------


class TestDatasetTypeMapping:
    """Tests for dataset type label to DatasetType enum mapping."""

    def test_dataset_type_mapping_bars(
        self,
        component: DatasetRegistrationComponent,
    ) -> None:
        """Test 'bars' label maps to DatasetType.BARS."""
        result = component.map_dataset_type("bars")
        assert result == DatasetType.BARS

    def test_dataset_type_mapping_trades(
        self,
        component: DatasetRegistrationComponent,
    ) -> None:
        """Test 'trades' label maps to DatasetType.TRADES."""
        result = component.map_dataset_type("trades")
        assert result == DatasetType.TRADES

    def test_dataset_type_mapping_tbbo(
        self,
        component: DatasetRegistrationComponent,
    ) -> None:
        """Test 'tbbo' label maps to DatasetType.TBBO."""
        result = component.map_dataset_type("tbbo")
        assert result == DatasetType.TBBO

    def test_dataset_type_mapping_mbp1(
        self,
        component: DatasetRegistrationComponent,
    ) -> None:
        """Test 'mbp1' label maps to DatasetType.MBP1."""
        result = component.map_dataset_type("mbp1")
        assert result == DatasetType.MBP1


# -----------------------------------------------------------------------------
# Protocol Compliance Test
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_component_satisfies_protocol(
        self,
        component: DatasetRegistrationComponent,
    ) -> None:
        """Test DatasetRegistrationComponent satisfies DatasetRegistrationProtocol."""
        # Protocol compliance is structural in Python - verify methods exist
        assert hasattr(component, "ensure_dataset_registered")
        assert callable(component.ensure_dataset_registered)


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and default behavior."""

    def test_unknown_dataset_type_defaults_to_bars(
        self,
        component: DatasetRegistrationComponent,
    ) -> None:
        """Test unknown dataset type label defaults to BARS."""
        result = component.map_dataset_type("unknown_type")
        assert result == DatasetType.BARS

    def test_empty_dataset_type_defaults_to_bars(
        self,
        component: DatasetRegistrationComponent,
    ) -> None:
        """Test empty dataset type label defaults to BARS."""
        result = component.map_dataset_type("")
        assert result == DatasetType.BARS

    def test_location_path_expansion(
        self,
        component: DatasetRegistrationComponent,
        mock_registry: MagicMock,
    ) -> None:
        """Test location path is expanded (tilde expansion)."""
        # Setup: get_manifest raises to simulate non-existent dataset
        mock_registry.get_manifest.side_effect = ValueError("Dataset not found")

        with patch(
            "ml.data.dataset_manifest_defaults.build_auto_dataset_manifest"
        ) as mock_build:
            mock_manifest = MagicMock()
            mock_build.return_value = mock_manifest

            component.ensure_dataset_registered(
                registry=mock_registry,
                dataset_id="test_dataset",
                dataset_type_label="bars",
                location="~/data/catalogs",
                retention_days=30,
            )

            # Verify location was expanded
            call_kwargs = mock_build.call_args.kwargs
            assert "~" not in call_kwargs["location"]
            assert call_kwargs["location"].startswith("/")
