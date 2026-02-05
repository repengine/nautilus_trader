"""
Compatibility shim for DataScheduler canonical import paths.

This module preserves the public API of the legacy facade while routing all
calls to the component-based DataScheduler implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ml.config.scheduler_config import SchedulerConfig
from ml.data.collector import DataCollector
from ml.data.scheduler import DataScheduler
from ml.registry.dataclasses import DatasetType
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    from ml.features.engineering import FeatureEngineer


DataSchedulerFacade = DataScheduler


def create_data_scheduler(
    catalog: ParquetDataCatalog,
    config: SchedulerConfig | None = None,
    collector: DataCollector | None = None,
    feature_engineer: FeatureEngineer | None = None,
    metrics_port: int | None = None,
    start_metrics_server: bool = True,
    connection: str | None = None,
    use_orchestrator: bool = False,
    dual_write: bool = False,
    dual_write_dataset_types: Mapping[DatasetType, bool] | None = None,
    dataset_type_identifier_templates: Mapping[DatasetType, str] | None = None,
) -> DataScheduler:
    """
    Create a DataScheduler instance.

    Args:
        catalog: Nautilus data catalog for data storage.
        config: Configuration for scheduler. If None, uses defaults.
        collector: Data collector for fetching from Databento.
        feature_engineer: Feature engineer for computing features.
        metrics_port: Port for metrics HTTP server. Defaults to 8000.
        start_metrics_server: Whether to start the metrics HTTP server.
        connection: Database connection string for feature store.
        use_orchestrator: Whether to use orchestrator-based collection.
        dual_write: Whether to dual-write to both SQL and catalog.
        dual_write_dataset_types: Optional dataset-type toggles for mirroring.
        dataset_type_identifier_templates: Deprecated and ignored; schema registry defaults
            are enforced for catalog identifiers.

    Returns:
        DataScheduler instance.
    """
    return DataScheduler(
        catalog=catalog,
        config=config,
        collector=collector,
        feature_engineer=feature_engineer,
        metrics_port=metrics_port,
        start_metrics_server=start_metrics_server,
        connection=connection,
        use_orchestrator=use_orchestrator,
        dual_write=dual_write,
        dual_write_dataset_types=dual_write_dataset_types,
        dataset_type_identifier_templates=dataset_type_identifier_templates,
    )


__all__ = [
    "DataScheduler",
    "DataSchedulerFacade",
    "create_data_scheduler",
]
