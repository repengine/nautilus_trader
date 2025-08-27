# Domain Bookkeeping Architecture

## Core Principle
Each major information system has its own Registry + Store pair that acts as the authoritative bookkeeper for that domain. Together, they provide complete observability and auditability of the ML pipeline.

## The Four Domain Bookkeepers

### 1. 📊 Data Domain (DataRegistry + DataStore)
**Responsibility**: All raw market data and ingestion

**What It Tracks**:
- Every bar, quote, trade, order book update
- Data quality metrics and validation results
- Ingestion timestamps and sources (live vs historical)
- Schema versions and migrations
- Gap detection and backfill status

**Key Events**:
- `DATA_INGESTED` - Raw data received
- `CATALOG_WRITTEN` - Data persisted to catalog
- `DATA_VALIDATED` - Quality checks passed
- `BACKFILL_COMPLETED` - Historical gaps filled

**Storage**: Parquet files in Catalog + PostgreSQL metadata

---

### 2. 🔬 Feature Domain (FeatureRegistry + FeatureStore)
**Responsibility**: All feature engineering and transformations

**What It Tracks**:
- Every feature computation (batch and realtime)
- Feature schemas and versions
- Feature lineage (which raw data → which features)
- Feature drift and quality metrics
- Training/serving feature parity

**Key Events**:
- `FEATURE_COMPUTED` - Features calculated
- `FEATURE_STORED` - Features persisted
- `FEATURE_DRIFT_DETECTED` - Statistical drift identified
- `FEATURE_SCHEMA_CHANGED` - New feature version

**Storage**: PostgreSQL `ml_feature_values` table

---

### 3. 🤖 Model Domain (ModelRegistry + ModelStore)
**Responsibility**: All model lifecycle and predictions

**What It Tracks**:
- Every model training run and hyperparameters
- Model versions and deployments
- Every prediction made (batch and realtime)
- Model performance metrics and drift
- A/B test results and champion/challenger status

**Key Events**:
- `MODEL_TRAINED` - Training completed
- `MODEL_DEPLOYED` - Model promoted to production
- `PREDICTION_EMITTED` - Inference performed
- `MODEL_DRIFT_DETECTED` - Performance degradation
- `MODEL_RETRAINED` - Automatic retraining triggered

**Storage**: PostgreSQL `ml_model_predictions` + ONNX/pickle artifacts

---

### 4. 📈 Strategy Domain (StrategyRegistry + StrategyStore)
**Responsibility**: All trading decisions and signals

**What It Tracks**:
- Every signal generated
- Strategy state and parameters
- Position recommendations
- Risk limits and violations
- Strategy performance metrics

**Key Events**:
- `SIGNAL_EMITTED` - Trading signal generated
- `POSITION_RECOMMENDED` - Size/direction suggested
- `RISK_LIMIT_BREACHED` - Safety threshold exceeded
- `STRATEGY_UPDATED` - Parameters changed
- `STRATEGY_BACKTESTED` - Historical validation completed

**Storage**: PostgreSQL `ml_strategy_signals` table

---

## 🔄 Cross-Domain Coordination

The registries communicate to maintain consistency:

```python
# Example: End-to-end lineage tracking
class MLPipelineCoordinator:
    def __init__(self):
        self.data_registry = DataRegistry()
        self.feature_registry = FeatureRegistry()
        self.model_registry = ModelRegistry()
        self.strategy_registry = StrategyRegistry()

    def trace_prediction_lineage(self, prediction_id: str) -> dict:
        """Trace a prediction back through the entire pipeline."""

        # Start from strategy signal
        signal = self.strategy_registry.get_signal(prediction_id)

        # Trace back to model prediction
        prediction = self.model_registry.get_prediction(signal.prediction_id)

        # Trace back to features
        features = self.feature_registry.get_features(prediction.feature_set_id)

        # Trace back to raw data
        raw_data = self.data_registry.get_data(features.data_ids)

        return {
            "signal": signal,
            "prediction": prediction,
            "features": features,
            "raw_data": raw_data,
            "full_lineage": self._build_lineage_graph(signal, prediction, features, raw_data)
        }
```

---

## 📊 Unified Observability Dashboard

All four bookkeepers feed into a unified monitoring system:

```yaml
# Grafana Dashboard Structure
ML Pipeline Overview:
  Row 1 - Data Domain:
    - Ingestion rate (events/sec)
    - Data quality score
    - Coverage by instrument
    - Backfill queue depth

  Row 2 - Feature Domain:
    - Feature computation latency
    - Feature drift alerts
    - Feature store size
    - Training/serving parity

  Row 3 - Model Domain:
    - Predictions per second
    - Model accuracy trends
    - Drift detection alerts
    - Retraining queue

  Row 4 - Strategy Domain:
    - Signals generated
    - Signal accuracy
    - Risk metrics
    - PnL attribution
```

---

## 🔐 Key Benefits of Domain Bookkeeping

### 1. **Complete Auditability**
Every action in the pipeline is recorded with who/what/when/why:
- Regulatory compliance (MiFID II, SEC requirements)
- Post-mortem analysis of trading decisions
- Debugging production issues

### 2. **Automatic Orchestration**
Bookkeepers can trigger downstream actions:
- Data gaps → Automatic backfill
- Feature drift → Model retraining
- Model degradation → Strategy adjustment
- Risk breach → Position reduction

### 3. **Time Travel Debugging**
Reconstruct exact state at any point:
```python
# Replay the exact conditions that led to a trade
coordinator.replay_pipeline_state(
    timestamp="2024-01-15T14:30:00Z",
    instrument="EUR/USD"
)
```

### 4. **Cost Attribution**
Track resource usage per domain:
- Data ingestion costs (API calls, storage)
- Feature computation costs (CPU hours)
- Model inference costs (GPU time)
- Strategy execution costs (order fees)

---

## 🎯 Implementation Priority

1. **Phase 1**: Data Domain (✅ Completed)
   - DataRegistry + DataStore implemented
   - Event tracking and watermarks working

2. **Phase 2**: Feature Domain (✅ Partial)
   - FeatureRegistry exists
   - FeatureStore enhanced with events

3. **Phase 3**: Model Domain (✅ Partial)
   - ModelRegistry exists
   - ModelStore enhanced with events

4. **Phase 4**: Strategy Domain (✅ Partial)
   - StrategyRegistry exists
   - StrategyStore enhanced with events

5. **Phase 5**: Cross-Domain Integration (🔄 Next)
   - Unified lineage tracking
   - Coordinated orchestration
   - Consolidated monitoring

---

## 📝 Code Example: Universal Bookkeeping Interface

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')  # Domain entity type
E = TypeVar('E')  # Event type

class DomainBookkeeper(ABC, Generic[T, E]):
    """Universal interface for domain bookkeeping."""

    @abstractmethod
    def register(self, entity: T) -> str:
        """Register a new entity in the domain."""
        pass

    @abstractmethod
    def record_event(self, event: E) -> None:
        """Record an event that occurred."""
        pass

    @abstractmethod
    def get_lineage(self, entity_id: str) -> dict:
        """Get complete lineage for an entity."""
        pass

    @abstractmethod
    def get_health(self) -> dict:
        """Get domain health metrics."""
        pass

    @abstractmethod
    def trigger_action(self, condition: dict) -> None:
        """Trigger automated action based on condition."""
        pass

# Each domain implements this interface
class DataBookkeeper(DomainBookkeeper[DatasetManifest, DataEvent]):
    """Bookkeeper for the Data domain."""

    def __init__(self):
        self.registry = DataRegistry()
        self.store = DataStore()

    def register(self, entity: DatasetManifest) -> str:
        return self.registry.register_dataset(entity)

    def record_event(self, event: DataEvent) -> None:
        self.registry.emit_event(**event.dict())

    def get_lineage(self, entity_id: str) -> dict:
        return self.registry.get_lineage(entity_id)

    def get_health(self) -> dict:
        return {
            "coverage": self.registry.get_coverage(),
            "quality": self.store.get_quality_metrics(),
            "watermarks": self.registry.get_watermarks()
        }

    def trigger_action(self, condition: dict) -> None:
        if condition["type"] == "GAP_DETECTED":
            self.store.plan_backfill(condition["dataset"], condition["date"])
```

---

## 🚀 Next Steps

1. **Enhance Cross-Domain Communication**
   - Implement event bus for registry communication
   - Add distributed tracing (OpenTelemetry)

2. **Build Unified Lineage Service**
   - GraphQL API for lineage queries
   - Lineage visualization UI

3. **Implement Orchestration Rules**
   - YAML-based rule definitions
   - Automatic remediation workflows

4. **Create Domain Health Dashboards**
   - Real-time health scores per domain
   - Alerting on degradation

The key insight: **These aren't just storage systems - they're active participants that monitor, validate, orchestrate, and audit everything happening in their domains.**
