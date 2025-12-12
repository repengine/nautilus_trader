"""
Data processing components extracted from TFTDatasetBuilder and DataScheduler.

This module contains focused, single-responsibility components for:
- Time series windowing and bounds extraction
- Feature alignment across timestamps
- Target generation for ML models
- Schema validation for TFT datasets
- Scheduler initialization (DataRegistry, FeatureStore)
- Data cleanup and scheduling operations
- Metrics server management
- Dataset registration in DataRegistry
- Feature computation for newly collected data
- Orchestrator-based collection via IngestionOrchestrator
- Data collection from Databento API
- Daily update pipeline orchestration

"""

from __future__ import annotations

from ml.data.common.daily_update_orchestrator import DailyUpdateOrchestratorComponent
from ml.data.common.daily_update_orchestrator import DailyUpdateOrchestratorProtocol
from ml.data.common.daily_update_orchestrator import track_pipeline_stage
from ml.data.common.data_cleanup import DataCleanupComponent
from ml.data.common.data_cleanup import DataCleanupProtocol
from ml.data.common.data_collection import DataCollectionComponent
from ml.data.common.data_collection import DataCollectionProtocol
from ml.data.common.dataset_registration import DatasetRegistrationComponent
from ml.data.common.dataset_registration import DatasetRegistrationProtocol
from ml.data.common.feature_alignment import FeatureAlignmentComponent
from ml.data.common.known_future_features import KnownFutureFeatureComponent
from ml.data.common.metrics_server import MetricsServerComponent
from ml.data.common.metrics_server import MetricsServerProtocol
from ml.data.common.orchestrator_collection import OrchestratorCollectionComponent
from ml.data.common.orchestrator_collection import OrchestratorCollectionProtocol
from ml.data.common.scheduler_feature_job import VENUE_MAP
from ml.data.common.scheduler_feature_job import FeatureComputationComponent
from ml.data.common.scheduler_feature_job import FeatureComputationProtocol
from ml.data.common.scheduler_init import SchedulerInitComponent
from ml.data.common.scheduler_init import SchedulerInitProtocol
from ml.data.common.target_generation import TargetGenerationComponent
from ml.data.common.tft_schema_validator import SchemaValidationError
from ml.data.common.tft_schema_validator import TFTSchemaValidatorComponent
from ml.data.common.time_series_windowing import TimeSeriesWindowingComponent


__all__ = [
    "VENUE_MAP",
    "DailyUpdateOrchestratorComponent",
    "DailyUpdateOrchestratorProtocol",
    "DataCleanupComponent",
    "DataCleanupProtocol",
    "DataCollectionComponent",
    "DataCollectionProtocol",
    "DatasetRegistrationComponent",
    "DatasetRegistrationProtocol",
    "FeatureAlignmentComponent",
    "FeatureComputationComponent",
    "FeatureComputationProtocol",
    "KnownFutureFeatureComponent",
    "MetricsServerComponent",
    "MetricsServerProtocol",
    "OrchestratorCollectionComponent",
    "OrchestratorCollectionProtocol",
    "SchedulerInitComponent",
    "SchedulerInitProtocol",
    "SchemaValidationError",
    "TFTSchemaValidatorComponent",
    "TargetGenerationComponent",
    "TimeSeriesWindowingComponent",
    "track_pipeline_stage",
]
