from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ml.config.scheduler_config import SchedulerConfig
from ml.data.scheduler import DataScheduler
from ml.registry.dataclasses import DatasetType
from ml.schema import DATASET_TYPE_IDENTIFIER_DEFAULTS

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


@pytest.mark.contracts
def test_scheduler_defaults_identifier_templates_to_schema_defaults() -> None:
    scheduler = DataScheduler(
        catalog=MagicMock(),
        config=SchedulerConfig(symbols=("AAPL.XNAS",), feature_store_enabled=False),
        collector=MagicMock(),
        start_metrics_server=False,
    )

    for dataset_type, template in DATASET_TYPE_IDENTIFIER_DEFAULTS.items():
        assert scheduler._dataset_type_identifier_templates[dataset_type] == template


@pytest.mark.contracts
def test_scheduler_rejects_invalid_identifier_templates() -> None:
    with pytest.raises(ValueError, match="must include"):
        DataScheduler(
            catalog=MagicMock(),
            config=SchedulerConfig(symbols=("AAPL.XNAS",), feature_store_enabled=False),
            collector=MagicMock(),
            start_metrics_server=False,
            dataset_type_identifier_templates={DatasetType.BARS: "bars"},
        )
