#!/usr/bin/env python3

"""
Registry components module.

This module provides decomposed components extracted from registry god classes
following the established TDD decomposition pattern used in Phase 2.x and 3.x.

ModelRegistry Components:
    - ModelPersistenceComponent: Handles JSON/PostgreSQL persistence, serialization,
      batch save management, and SHA-256 integrity verification
    - DeploymentManagerComponent: Manages model deployment, canary releases, and
      gradual rollouts
    - ABTestingComponent: Manages A/B testing and statistical model comparison
    - VersionManagerComponent: Manages model versioning, compatibility, and lineage
      tracking

DataRegistry Components (Phase 3.3):
    - DataPersistenceComponent: Handles JSON/PostgreSQL persistence for datasets
    - ManifestManagerComponent: Handles dataset manifest CRUD operations
    - EventEmissionComponent: Handles data processing event emission
    - WatermarkManagerComponent: Handles data processing watermark tracking
    - LineageTrackerComponent: Handles dataset lineage relationship management

"""

from ml.registry.common.ab_testing import ABTestingComponent
from ml.registry.common.data_persistence import DataPersistenceComponent
from ml.registry.common.deployment_manager import DeploymentManagerComponent
from ml.registry.common.event_emission import EventEmissionComponent
from ml.registry.common.lineage_tracker import LineageTrackerComponent
from ml.registry.common.manifest_defaults import resolve_primary_keys
from ml.registry.common.manifest_manager import ManifestManagerComponent
from ml.registry.common.model_persistence import ModelPersistenceComponent
from ml.registry.common.sql_utils import set_instrumentation_search_path
from ml.registry.common.version_manager import VersionManagerComponent
from ml.registry.common.watermark_manager import WatermarkManagerComponent


__all__ = [
    "ABTestingComponent",
    "DataPersistenceComponent",
    "DeploymentManagerComponent",
    "EventEmissionComponent",
    "LineageTrackerComponent",
    "ManifestManagerComponent",
    "ModelPersistenceComponent",
    "VersionManagerComponent",
    "WatermarkManagerComponent",
    "resolve_primary_keys",
    "set_instrumentation_search_path",
]
