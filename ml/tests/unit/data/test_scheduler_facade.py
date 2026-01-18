"""
Unit tests for DataSchedulerFacade shim.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml.registry.dataclasses import DatasetType


@pytest.fixture
def mock_catalog() -> MagicMock:
    """Create a mock ParquetDataCatalog."""
    catalog = MagicMock()
    catalog.path = Path("/data/catalog")
    return catalog


def test_datascheduler_facade_is_alias() -> None:
    """DataSchedulerFacade should alias the canonical DataScheduler."""
    from ml.data.scheduler import DataScheduler
    from ml.data.scheduler_facade import DataSchedulerFacade

    assert DataSchedulerFacade is DataScheduler


def test_create_data_scheduler_forwards_arguments(mock_catalog: MagicMock) -> None:
    """create_data_scheduler forwards arguments to DataScheduler."""
    from ml.data import scheduler_facade

    mock_config = MagicMock()
    mock_collector = MagicMock()
    mock_engineer = MagicMock()
    mock_templates = {DatasetType.BARS: "{instrument_id}-1-MINUTE-LAST-EXTERNAL"}

    with patch.object(scheduler_facade, "DataScheduler") as scheduler_cls:
        scheduler_facade.create_data_scheduler(
            mock_catalog,
            config=mock_config,
            collector=mock_collector,
            feature_engineer=mock_engineer,
            metrics_port=8123,
            start_metrics_server=False,
            connection="postgresql://",
            use_orchestrator=True,
            dual_write=True,
            dual_write_dataset_types={DatasetType.BARS: True},
            dataset_type_identifier_templates=mock_templates,
        )

        scheduler_cls.assert_called_once_with(
            catalog=mock_catalog,
            config=mock_config,
            collector=mock_collector,
            feature_engineer=mock_engineer,
            metrics_port=8123,
            start_metrics_server=False,
            connection="postgresql://",
            use_orchestrator=True,
            dual_write=True,
            dual_write_dataset_types={DatasetType.BARS: True},
            dataset_type_identifier_templates=mock_templates,
        )
