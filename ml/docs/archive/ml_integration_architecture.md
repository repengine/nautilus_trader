# ML Integration Architecture

## Overview

The Nautilus Trader ML integration architecture provides a comprehensive framework for seamless integration between ML components and the core trading platform. This document describes the complete cross-domain integration patterns, data flow interactions, and architectural principles that ensure high-performance, reliable ML operations.

## Core Architecture Principles

### 1. Universal Component Protocols
All ML components implement the `MLComponentProtocol` for standardized:

- Health reporting and monitoring
- Performance metrics collection
- Configuration validation
- Lifecycle management

### 2. Domain-Driven Architecture
The ML system is organized into four core domains, each with dedicated bookkeepers:

- **Data Domain**: Raw market data ingestion and quality management
- **Feature Domain**: Feature engineering and transformation pipelines
- **Model Domain**: ML model lifecycle and inference operations
- **Strategy Domain**: Trading signals and decision generation

### 3. Hot/Cold Path Separation
Performance-critical operations are segregated into distinct execution paths:

- **Hot Path**: Real-time inference (<5ms P99), pre-allocated memory, zero GC
- **Cold Path**: Training, analytics, migrations, heavy I/O operations

## Integration Patterns

### Pattern 1: Mandatory 4-Store + 4-Registry Architecture

Every ML component MUST use all four stores and four registries through the `MLIntegrationManager`:

```python
from ml.core.integration import MLIntegrationManager

# Automatic initialization of all components
integration = MLIntegrationManager(config)

# Access to all stores and registries
stores = {
    'feature_store': integration.feature_store,
    'model_store': integration.model_store,
    'strategy_store': integration.strategy_store,
    'data_store': integration.data_store,
}

registries = {
    'feature_registry': integration.feature_registry,
    'model_registry': integration.model_registry,
    'strategy_registry': integration.strategy_registry,
    'data_registry': integration.data_registry,
}
```

#### Architecture Benefits

- **Consistency**: All components follow identical initialization patterns
- **Resilience**: Progressive fallback to dummy implementations when PostgreSQL unavailable
- **Monitoring**: Unified health checks across all components
- **Lifecycle**: Coordinated startup, shutdown, and migration handling

### Pattern 2: Protocol-First Interface Design

Components communicate through well-defined protocols rather than concrete implementations:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MLComponentProtocol(Protocol):
    def get_health_status(self) -> dict[str, Any]: ...
    def get_performance_metrics(self) -> dict[str, float]: ...
    def validate_configuration(self) -> list[str]: ...

# Structural typing enables duck typing and testing flexibility
def monitor_component(component: MLComponentProtocol) -> bool:
    health = component.get_health_status()
    return health.get('status') == 'ok'
```

### Pattern 3: Progressive Fallback Chains

All external dependencies have graceful degradation strategies:

```python
# PostgreSQL fallback hierarchy
try:
    store = FeatureStore(connection_string=db_url)
except ConnectionError:
    logger.warning("PostgreSQL unavailable, using DummyStore")
    store = DummyFeatureStore()  # No persistence, logs warnings

# Model registry fallback
try:
    model = model_registry.load_model(model_id)
except ModelNotFoundError:
    model = fallback_model_loader.load_from_file(backup_path)
```

### Pattern 4: Centralized Metrics Bootstrap

All metrics collection goes through a centralized bootstrap system:

```python
from ml.common.metrics_bootstrap import get_counter, get_histogram

# Prevents registry conflicts and ensures consistent naming
predictions_counter = get_counter(
    "ml_predictions_total",
    "Total ML predictions made",
    labels=["model_id", "instrument_id"]
)

latency_histogram = get_histogram(
    "ml_inference_latency_seconds",
    "ML inference latency distribution",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1]
)
```

## Data Flow Architecture

### End-to-End Data Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Domain   │───▶│ Feature Domain  │───▶│  Model Domain   │───▶│Strategy Domain  │
│                 │    │                 │    │                 │    │                 │
│ • DataRegistry  │    │ • FeatureReg.   │    │ • ModelRegistry │    │ • StrategyReg.  │
│ • DataStore     │    │ • FeatureStore  │    │ • ModelStore    │    │ • StrategyStore │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Market    │    │    Features     │    │   Predictions   │    │     Signals     │
│      Data       │    │                 │    │                 │    │                 │
│ • Bars/Quotes   │    │ • Technical     │    │ • Probabilities │    │ • Buy/Sell     │
│ • Order Books   │    │ • Microstructure│    │ • Confidence    │    │ • Position Size │
│ • Trades        │    │ • Cross-sectional│    │ • Risk Scores   │    │ • Risk Limits  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Cross-Domain Event Flow

Each domain emits standardized events that trigger cascading actions:

```python
# Data Domain Events
DATA_INGESTED = "raw_data_received"
DATA_VALIDATED = "quality_checks_passed"
GAP_DETECTED = "data_gap_identified"

# Feature Domain Events
FEATURES_COMPUTED = "feature_calculation_complete"
FEATURE_DRIFT_DETECTED = "statistical_drift_identified"
PARITY_VIOLATION = "batch_online_mismatch"

# Model Domain Events
PREDICTION_EMITTED = "model_inference_complete"
MODEL_DRIFT_DETECTED = "performance_degradation"
RETRAINING_TRIGGERED = "automatic_retrain_started"

# Strategy Domain Events
SIGNAL_GENERATED = "trading_signal_emitted"
POSITION_RECOMMENDED = "position_size_calculated"
RISK_BREACH = "risk_limit_exceeded"
```

### Event Correlation and Lineage

All events carry correlation metadata enabling full pipeline traceability:

```python
from ml.common.correlation import create_correlation_id, correlate_event

# Create correlation chain
correlation_id = create_correlation_id()

# Data event
data_event = {
    'domain': 'data',
    'event_type': 'DATA_INGESTED',
    'correlation_id': correlation_id,
    'instrument_id': 'EUR/USD',
    'ts_event': ts_now(),
    'payload': {...}
}

# Correlated feature event
feature_event = correlate_event(
    source=data_event,
    target_domain='features',
    event_type='FEATURES_COMPUTED'
)

# Complete lineage tracking
lineage = trace_correlation_chain(correlation_id)
# Returns: [data_event, feature_event, model_event, strategy_event]
```

## Performance Contracts and SLA Definitions

### Hot Path Performance Requirements

| Component | Latency (P99) | Memory | Allocation |
|-----------|---------------|--------|------------|
| Feature Computation | < 1ms | < 64MB | Zero after warmup |
| Model Inference | < 5ms | < 256MB | Pre-allocated arrays |
| Signal Generation | < 10ms | < 512MB | Bounded queues |
| Risk Validation | < 2ms | < 32MB | Stack allocated |

### Cold Path Performance Requirements

| Operation | Target Time | Resource Limits |
|-----------|-------------|-----------------|
| Model Training | < 4 hours | 32GB RAM, 8 CPU cores |
| Feature Backfill | < 2 hours per month | 16GB RAM, 4 CPU cores |
| Registry Migrations | < 30 minutes | 8GB RAM, 2 CPU cores |
| Health Checks | < 10 seconds | 1GB RAM, 1 CPU core |

### SLA Monitoring

```python
from ml.common.metrics_bootstrap import get_histogram

# Automatic SLA tracking
hot_path_latency = get_histogram(
    "ml_hot_path_latency_seconds",
    "Hot path operation latency",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.05],  # 0.5ms to 50ms
    labels=["operation", "component"]
)

@hot_path_latency.time(labels={"operation": "inference", "component": "model"})
def predict(self, features: np.ndarray) -> float:
    # Performance tracked automatically
    return self.model.predict(features)
```

## Error Propagation and Fallback Strategies

### Error Classification

Errors are classified into categories with specific handling strategies:

| Error Type | Strategy | Fallback Action |
|------------|----------|-----------------|
| **Transient** | Retry with backoff | Exponential backoff, circuit breaker |
| **Configuration** | Fail fast | Validate at startup, prevent deployment |
| **Data Quality** | Degrade gracefully | Use cached features, emit warnings |
| **Model Performance** | Auto-remediate | Switch to backup model, trigger retraining |
| **System Resource** | Load shed | Drop non-critical operations, alert ops |

### Error Propagation Chain

```python
class MLErrorPropagator:
    def __init__(self, integration_manager: MLIntegrationManager):
        self.stores = integration_manager
        self.error_handlers = {
            DataQualityError: self._handle_data_error,
            ModelDriftError: self._handle_model_error,
            FeaturePipelineError: self._handle_feature_error,
            SystemResourceError: self._handle_resource_error,
        }

    def propagate_error(self, error: Exception, context: dict) -> None:
        """Propagate error through appropriate channels with context."""
        error_type = type(error)
        handler = self.error_handlers.get(error_type, self._handle_unknown_error)

        # Log with full context
        logger.error(f"ML pipeline error: {error}", extra=context)

        # Execute domain-specific handler
        handler(error, context)

        # Emit cross-domain event for coordination
        self.stores.emit_cross_domain_event({
            'domain': context.get('domain', 'unknown'),
            'event_type': 'ERROR_OCCURRED',
            'error_type': error_type.__name__,
            'severity': self._classify_severity(error),
            'correlation_id': context.get('correlation_id'),
            'timestamp': time.time_ns(),
            'payload': {'message': str(error), 'context': context}
        })
```

### Circuit Breaker Implementation

```python
from ml.common.circuit_breaker import CircuitBreaker

class FeatureEngineeringPipeline:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,      # Open after 5 failures
            recovery_timeout=30,      # Try recovery after 30s
            expected_exception=FeatureComputationError
        )

    @circuit_breaker
    def compute_features(self, data: pd.DataFrame) -> np.ndarray:
        """Feature computation with circuit breaker protection."""
        return self._expensive_feature_computation(data)

    def compute_features_with_fallback(self, data: pd.DataFrame) -> np.ndarray:
        """Primary method with graceful fallback."""
        try:
            return self.compute_features(data)
        except CircuitBreakerOpenError:
            logger.warning("Feature pipeline circuit breaker open, using cached features")
            return self.get_cached_features(data.index[-1])
```

## Security Boundaries and Access Patterns

### Access Control Matrix

| Component | Data Domain | Feature Domain | Model Domain | Strategy Domain |
|-----------|-------------|----------------|--------------|-----------------|
| **Data Ingestion** | Read/Write | Read | - | - |
| **Feature Engineering** | Read | Read/Write | Read | - |
| **Model Training** | Read | Read | Read/Write | - |
| **Strategy Execution** | Read | Read | Read | Read/Write |
| **Monitoring** | Read | Read | Read | Read |

### Security Implementation

```python
from ml.common.security import SecurityContext, require_domain_access

class SecureMLComponent:
    def __init__(self, security_context: SecurityContext):
        self.security = security_context

    @require_domain_access("data", "read")
    def read_market_data(self, instrument_id: str) -> pd.DataFrame:
        """Access controlled data reading."""
        return self.data_store.read_bars(instrument_id)

    @require_domain_access("model", "write")
    def register_model(self, model_manifest: ModelManifest) -> str:
        """Access controlled model registration."""
        return self.model_registry.register_model(model_manifest)
```

### Data Privacy and Encryption

```python
# Sensitive data encryption at rest
from ml.common.encryption import encrypt_sensitive_fields

model_manifest = ModelManifest(
    model_id="lgb_001",
    training_config=encrypt_sensitive_fields({
        "learning_rate": 0.1,
        "api_key": "sensitive_value"  # Automatically encrypted
    }),
    performance_metrics={"accuracy": 0.85}  # Not encrypted
)

# In-transit encryption for inter-service communication
from ml.common.tls import create_secure_channel

secure_channel = create_secure_channel(
    target_service="model-inference",
    cert_path="certs/ml-client.pem",
    key_path="certs/ml-client.key"
)
```

## Component Integration Patterns

### Actor Integration

All ML actors inherit from `BaseMLInferenceActor` for automatic store integration:

```python
from ml.actors.base import BaseMLInferenceActor

class CustomMLActor(BaseMLInferenceActor):
    def __init__(self, config: CustomMLActorConfig):
        super().__init__(config)
        # Stores automatically initialized:
        # - self.feature_store
        # - self.model_store
        # - self.strategy_store
        # - self.data_store

    def on_bar(self, bar: Bar) -> None:
        # Hot path with pre-initialized stores
        features = self.feature_store.get_latest_features(
            bar.instrument_id,
            bar.ts_event
        )
        prediction = self.model.predict(features)
        self.model_store.record_prediction(prediction)
```

### Store Integration

Stores automatically wire together through the integration manager:

```python
# Automatic data flow between stores
class IntegratedFeatureStore(FeatureStore):
    def write_features(self, features: dict) -> None:
        # Write to feature store
        super().write_features(features)

        # Automatically propagate to downstream stores
        if self.integration_manager:
            # Trigger model inference if configured
            self.integration_manager.trigger_downstream_processing(
                domain="features",
                event_type="FEATURES_AVAILABLE",
                data=features
            )
```

### Registry Integration

Registries maintain cross-domain relationships and lineage:

```python
class IntegratedModelRegistry(ModelRegistry):
    def register_model(self, manifest: ModelManifest) -> str:
        model_id = super().register_model(manifest)

        # Automatically link to feature registry
        if manifest.feature_schema_hash:
            feature_set = self.feature_registry.find_by_hash(
                manifest.feature_schema_hash
            )
            if feature_set:
                self.link_model_to_features(model_id, feature_set.id)

        return model_id
```

## Monitoring and Observability

### Unified Health Dashboard

```python
from ml.core.integration import MLIntegrationManager

integration = MLIntegrationManager()
health_status = integration.aggregate_health()

# Structured health reporting
{
    "system": {
        "healthy": True,
        "unhealthy": []
    },
    "domains": {
        "data": {
            "components": ["data_store", "data_registry"],
            "healthy": True
        },
        "features": {
            "components": ["feature_store", "feature_registry"],
            "healthy": True
        },
        "model": {
            "components": ["model_store", "model_registry"],
            "healthy": False  # Issue detected
        },
        "strategy": {
            "components": ["strategy_store", "strategy_registry"],
            "healthy": True
        }
    },
    "components": {
        "model_store": {
            "healthy": False,
            "health": {"last_write": "2024-01-15T10:30:00Z", "error": "Connection timeout"},
            "metrics": {"operations_per_second": 45.2, "error_rate": 0.02}
        }
        # ... other components
    }
}
```

### Performance Monitoring

```python
# Automatic performance tracking across domains
from ml.monitoring.performance import PerformanceTracker

tracker = PerformanceTracker()

@tracker.track_performance(component="feature_engineering", operation="compute_features")
def compute_technical_indicators(self, data: pd.DataFrame) -> np.ndarray:
    # Automatically tracked:
    # - Execution time
    # - Memory usage
    # - CPU utilization
    # - Error rates
    return self.technical_indicators.compute(data)

# Performance metrics available in Prometheus
# ml_component_operation_duration_seconds{component="feature_engineering",operation="compute_features"}
# ml_component_memory_usage_bytes{component="feature_engineering"}
# ml_component_error_rate{component="feature_engineering",operation="compute_features"}
```

### Cross-Domain Event Tracking

```python
from ml.monitoring.events import EventCorrelationTracker

# Track events across domain boundaries
correlation_tracker = EventCorrelationTracker()

def trace_prediction_pipeline(correlation_id: str) -> dict:
    """Trace a prediction from data ingestion to signal generation."""
    events = correlation_tracker.get_correlation_chain(correlation_id)

    return {
        "correlation_id": correlation_id,
        "total_latency_ms": events[-1].timestamp - events[0].timestamp,
        "domain_breakdown": {
            "data_processing": calculate_domain_time(events, "data"),
            "feature_engineering": calculate_domain_time(events, "features"),
            "model_inference": calculate_domain_time(events, "model"),
            "signal_generation": calculate_domain_time(events, "strategy")
        },
        "bottlenecks": identify_bottlenecks(events),
        "error_points": identify_errors(events)
    }
```

## Deployment and Configuration

### Environment-Specific Configuration

```python
from ml.config.environments import EnvironmentConfig

# Development environment
dev_config = EnvironmentConfig(
    environment="development",
    db_connection="postgresql://dev_user:dev_pass@localhost:5432/nautilus_dev",
    auto_start_postgres=True,
    auto_migrate=True,
    strict_protocol_validation=True,
    performance_monitoring=True
)

# Production environment
prod_config = EnvironmentConfig(
    environment="production",
    db_connection="postgresql://prod_user:secure_pass@prod-db:5432/nautilus_prod",
    auto_start_postgres=False,  # Managed externally
    auto_migrate=False,         # Manual migration process
    strict_protocol_validation=False,  # Performance optimized
    performance_monitoring=True,
    circuit_breakers_enabled=True,
    fallback_strategies_enabled=True
)
```

### Container Orchestration

```yaml
# docker-compose.yml for ML components
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: nautilus
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  ml-integration:
    build: .
    environment:
      ML_AUTO_START_DB: "false"  # Use external postgres
      ML_AUTO_MIGRATE: "true"
      DB_CONNECTION: "postgresql://postgres:postgres@postgres:5432/nautilus"
    depends_on:
      - postgres
    ports:
      - "8080:8080"  # Health check endpoint
```

### Health Check Endpoints

```python
from fastapi import FastAPI
from ml.core.integration import get_integration_manager

app = FastAPI()
integration = get_integration_manager()

@app.get("/health")
async def health_check():
    """Kubernetes-compatible health check."""
    try:
        health = integration.aggregate_health()
        return {
            "status": "healthy" if health["system"]["healthy"] else "unhealthy",
            "details": health
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    try:
        integration.ensure_healthy()
        return {"status": "ready"}
    except Exception:
        return {"status": "not_ready"}, 503
```

## Future Architecture Considerations

### Distributed Architecture

Preparation for multi-node deployment:

```python
# Node-aware component distribution
class DistributedMLIntegrationManager(MLIntegrationManager):
    def __init__(self, node_id: str, cluster_config: ClusterConfig):
        self.node_id = node_id
        self.cluster = cluster_config
        super().__init__()

    def _init_stores(self) -> None:
        # Shard stores across cluster nodes
        if self.cluster.is_sharded:
            shard_key = self._calculate_shard_key(self.node_id)
            self.feature_store = ShardedFeatureStore(shard_key=shard_key)
            self.model_store = ShardedModelStore(shard_key=shard_key)
```

### Stream Processing Integration

```python
# Kafka/Pulsar integration for high-throughput scenarios
class StreamingMLPipeline:
    def __init__(self, integration_manager: MLIntegrationManager):
        self.integration = integration_manager
        self.stream_processor = StreamProcessor(
            input_topics=["market_data", "features"],
            output_topics=["predictions", "signals"],
            processing_guarantee="exactly_once"
        )

    async def process_stream(self, event: StreamEvent) -> None:
        # Stream processing with ML integration
        if event.topic == "market_data":
            features = await self.compute_features(event.data)
            await self.stream_processor.emit("features", features)
        elif event.topic == "features":
            prediction = await self.run_inference(event.data)
            await self.stream_processor.emit("predictions", prediction)
```

This comprehensive ML integration architecture ensures that all components work together seamlessly while maintaining high performance, reliability, and observability across the entire ML pipeline.
