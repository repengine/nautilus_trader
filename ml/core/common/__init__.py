"""
ML Core Components Package.

This package contains decomposed components extracted from MLIntegrationManager
as part of the god-class decomposition effort (Phase 3.6).

Components follow the Protocol-First Interface Design pattern and can be used
independently or composed via the MLIntegrationManagerFacade.

Components
----------
DatabaseLifecycleComponent : Phase 3.6.1
    PostgreSQL connection probing, container startup, migrations.
StoreInitializationComponent : Phase 3.6.2
    Initialize 4 stores with progressive fallback (PostgreSQL -> File -> Dummy).
RegistryInitializationComponent : Phase 3.6.3
    Initialize 4 registries with persistence configuration and DataStore wiring.
HealthMonitoringComponent : Phase 3.6.4
    Health checks for stores/registries, domain aggregation, protocol validation.
ObservabilityComponent : Phase 3.6.5
    Observability pipeline, flush scheduling, async workers, store injection.
ActorFactoryComponent : Phase 3.6.6
    Actor creation, shutdown, message publisher configuration.
EventIngestionComponent : Phase 3.6.7
    Event ingestion pipeline execution, optional backfill on startup.
"""

from ml.core.common.actor_factory import ActorFactoryComponent
from ml.core.common.database_lifecycle import DatabaseLifecycleComponent
from ml.core.common.event_ingestion import EventIngestionComponent
from ml.core.common.health_monitoring import HealthMonitoringComponent
from ml.core.common.observability import ObservabilityComponent
from ml.core.common.registry_initialization import RegistryInitializationComponent
from ml.core.common.store_initialization import StoreInitializationComponent


__all__ = [
    "ActorFactoryComponent",
    "DatabaseLifecycleComponent",
    "EventIngestionComponent",
    "HealthMonitoringComponent",
    "ObservabilityComponent",
    "RegistryInitializationComponent",
    "StoreInitializationComponent",
]
