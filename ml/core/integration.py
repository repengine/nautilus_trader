"""
ML system integration (facade-compatible import path).

This module re-exports the facade implementation to preserve the historical
`ml.core.integration` import path while keeping the component-facade version
as the single runtime implementation.
"""

from __future__ import annotations

from typing import TypeVar

from ml.common.db_connections import collect_postgres_candidates
from ml.core.integration_facade import ActorStoresRegistries
from ml.core.integration_facade import ComponentHealthStatus
from ml.core.integration_facade import DomainHealth
from ml.core.integration_facade import HasDBConnection
from ml.core.integration_facade import HealthDomains
from ml.core.integration_facade import HealthSummary
from ml.core.integration_facade import MLIntegrationManager
from ml.core.integration_facade import MLIntegrationManagerFacade
from ml.core.integration_facade import SystemHealth
from ml.core.integration_facade import create_data_store
from ml.core.integration_facade import get_integration_manager
from ml.core.integration_facade import init_actor_stores_and_registries
from ml.core.integration_facade import init_ml_stores_and_registries
from ml.core.integration_facade import reset_integration_manager


ActorT = TypeVar("ActorT")

__all__ = [
    "ActorStoresRegistries",
    "ActorT",
    "ComponentHealthStatus",
    "DomainHealth",
    "HasDBConnection",
    "HealthDomains",
    "HealthSummary",
    "MLIntegrationManager",
    "MLIntegrationManagerFacade",
    "SystemHealth",
    "collect_postgres_candidates",
    "create_data_store",
    "get_integration_manager",
    "init_actor_stores_and_registries",
    "init_ml_stores_and_registries",
    "reset_integration_manager",
]
