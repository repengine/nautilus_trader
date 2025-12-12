#!/usr/bin/env python3

"""
Protocol re-exports for orchestration module.

This module re-exports all protocols from ml.orchestration.common.protocols
to provide a cleaner import path: `from ml.orchestration.protocols import ...`

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


__all__ = [
    "ConfigResolverProtocol",
    "DatasetBuilderProtocol",
    "DiscoveryClientProtocol",
    "IngestionCoordinatorProtocol",
    "RegistrySynchronizerProtocol",
    "RuntimeAttacherProtocol",
    "StageControllerProtocol",
    "TrainingCoordinatorProtocol",
]
