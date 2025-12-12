"""
Common shared code for orchestration components.

This module provides shared protocols, types, and utilities for the
MLPipelineOrchestrator decomposition.

Exports:
    - Protocols: All component protocol definitions
    - Types: Shared dataclasses (BuildArtifacts, PipelineCheckpoint, etc.)
    - Utils: Utility functions (_ns_to_datetime, _map_schema_to_dataset_type)
    - StageController: Pipeline stage orchestration component

"""

from __future__ import annotations

from ml.orchestration.common.protocols import ConfigResolverProtocol
from ml.orchestration.common.protocols import DatasetBuilderProtocol
from ml.orchestration.common.protocols import DiscoveryClientProtocol
from ml.orchestration.common.protocols import IngestionCoordinatorProtocol
from ml.orchestration.common.protocols import RegistrySynchronizerProtocol
from ml.orchestration.common.protocols import RuntimeAttacherProtocol
from ml.orchestration.common.protocols import StageControllerProtocol
from ml.orchestration.common.protocols import TrainingCoordinatorProtocol
from ml.orchestration.common.stage_controller import StageController
from ml.orchestration.common.types import PipelineCheckpoint
from ml.orchestration.common.utils import map_schema_to_dataset_type
from ml.orchestration.common.utils import ns_to_datetime


__all__ = [
    # Protocols
    "ConfigResolverProtocol",
    "DatasetBuilderProtocol",
    "DiscoveryClientProtocol",
    "IngestionCoordinatorProtocol",
    # Types
    "PipelineCheckpoint",
    "RegistrySynchronizerProtocol",
    "RuntimeAttacherProtocol",
    # Components
    "StageController",
    "StageControllerProtocol",
    "TrainingCoordinatorProtocol",
    # Utils
    "map_schema_to_dataset_type",
    "ns_to_datetime",
]
