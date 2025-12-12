#!/usr/bin/env python3

"""
DataStore and FeatureStore component modules.

Phase 2.4.1 - Component Decomposition
Phase 3.7.1 - FeatureStore Decomposition (Write operations)
Phase 3.7.2 - FeatureStore Decomposition (Read operations)
Phase 3.7.3 - FeatureStore Decomposition (Computation operations)
Phase 3.7.4 - FeatureStore Decomposition (Schema operations)
Phase 3.7.5 - FeatureStore Decomposition (Event operations)
Phase 3.7.6 - FeatureStore Decomposition (Health operations)

This package contains extracted components from DataStore and FeatureStore
following the Protocol-First Interface Design pattern. Each component
provides a focused responsibility with explicit type contracts.

Components:
- SchemaValidatorComponent: Pre-write schema validation and contract enforcement
- DataWriterComponent: Write operations for ingestion, features, predictions, signals
- DataReaderComponent: Read operations with time-travel queries
- EventEmitterComponent: Event emission and message bus integration
- ContractEnforcerComponent: Contract retrieval and quality enforcement
- StoreOperationsComponent: Store lifecycle and health monitoring
- FeatureWriterComponent: Feature writing with circuit breaker and publishing
- FeatureReaderComponent: Feature reading with training data and point-in-time lookup
- FeatureComputationComponent: Real-time and historical feature computation
- FeatureSchemaComponent: Table setup, feature set ID, config hashing, timestamp normalization
- FeatureEventComponent: Event emission and DataRegistry integration
- FeatureHealthComponent: Health check, feature clearing, flush operations

"""

from ml.stores.common.contract_enforcer import ContractEnforcerComponent
from ml.stores.common.data_reader import DataReaderComponent
from ml.stores.common.data_writer import DataEvent
from ml.stores.common.data_writer import DataWriterComponent
from ml.stores.common.event_emitter import EventEmitterComponent
from ml.stores.common.feature_computation import FeatureComputationComponent
from ml.stores.common.feature_computation import FeatureComputationConfig
from ml.stores.common.feature_computation import FeatureComputationProtocol
from ml.stores.common.feature_computation import FeatureSchemaProtocol
from ml.stores.common.feature_event import FeatureEventComponent
from ml.stores.common.feature_event import FeatureEventConfig
from ml.stores.common.feature_event import FeatureEventProtocol
from ml.stores.common.feature_health import FeatureHealthComponent
from ml.stores.common.feature_health import FeatureHealthConfig
from ml.stores.common.feature_health import FeatureHealthProtocol
from ml.stores.common.feature_reader import FeatureReaderComponent
from ml.stores.common.feature_reader import FeatureReaderConfig
from ml.stores.common.feature_reader import FeatureReaderProtocol
from ml.stores.common.feature_schema import FeatureSchemaComponent
from ml.stores.common.feature_schema import FeatureSchemaConfig
from ml.stores.common.feature_schema import FeatureSchemaProtocol as FeatureSchemaComponentProtocol
from ml.stores.common.feature_writer import FeatureWriterComponent
from ml.stores.common.feature_writer import FeatureWriterConfig
from ml.stores.common.feature_writer import FeatureWriterProtocol
from ml.stores.common.feature_writer import MessagePublisherProtocol
from ml.stores.common.protocols import ContractEnforcerProtocol
from ml.stores.common.protocols import DataReaderProtocol
from ml.stores.common.protocols import DataWriterProtocol
from ml.stores.common.protocols import EventEmitterProtocol
from ml.stores.common.protocols import SchemaValidatorProtocol
from ml.stores.common.protocols import StoreOperationsProtocol
from ml.stores.common.schema_validator import QualityReport
from ml.stores.common.schema_validator import SchemaValidatorComponent
from ml.stores.common.schema_validator import ValidationViolation
from ml.stores.common.store_operations import StoreOperationsComponent


__all__ = [
    "ContractEnforcerComponent",
    "ContractEnforcerProtocol",
    "DataEvent",
    "DataReaderComponent",
    "DataReaderProtocol",
    "DataWriterComponent",
    "DataWriterProtocol",
    "EventEmitterComponent",
    "EventEmitterProtocol",
    "FeatureComputationComponent",
    "FeatureComputationConfig",
    "FeatureComputationProtocol",
    "FeatureEventComponent",
    "FeatureEventConfig",
    "FeatureEventProtocol",
    "FeatureHealthComponent",
    "FeatureHealthConfig",
    "FeatureHealthProtocol",
    "FeatureReaderComponent",
    "FeatureReaderConfig",
    "FeatureReaderProtocol",
    "FeatureSchemaComponent",
    "FeatureSchemaComponentProtocol",
    "FeatureSchemaConfig",
    "FeatureSchemaProtocol",
    "FeatureWriterComponent",
    "FeatureWriterConfig",
    "FeatureWriterProtocol",
    "MessagePublisherProtocol",
    "QualityReport",
    "SchemaValidatorComponent",
    "SchemaValidatorProtocol",
    "StoreOperationsComponent",
    "StoreOperationsProtocol",
    "ValidationViolation",
]
