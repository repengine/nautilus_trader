# Data Registry Usage Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [API Usage Examples](#api-usage-examples)
3. [CLI Command Examples](#cli-command-examples)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Schema Migration Procedures](#schema-migration-procedures)
6. [Monitoring and Alerting Setup](#monitoring-and-alerting-setup)

## Architecture Overview

The Data Registry system provides centralized metadata management, data lineage tracking, and coverage reporting for the ML pipeline. It consists of several key components:

### Core Components

```
ml/registry/
├── data_registry.py      # Main registry implementation
├── dataclasses.py        # Data models (manifests, contracts, etc.)
├── persistence.py        # Backend abstraction (JSON/PostgreSQL)
├── statistics.py         # Coverage and statistics computation
└── migrations/           # SQL schema migrations
```

### Data Flow

```mermaid
graph LR
    A[Data Ingestion] --> B[Event: CATALOG_WRITTEN]
    B --> C[Feature Engineering]
    C --> D[Event: FEATURE_COMPUTED]
    D --> E[Model Inference]
    E --> F[Event: PREDICTION_EMITTED]
    F --> G[Signal Generation]
    G --> H[Event: SIGNAL_EMITTED]

    B --> I[Watermark Update]
    D --> I
    F --> I
    H --> I

    I --> J[Coverage Report]
```

### Backend Options

1. **JSON Backend** (Development)
   - File-based storage
   - Human-readable format
   - No external dependencies
   - Suitable for testing and development

2. **PostgreSQL Backend** (Production)
   - Scalable database storage
   - ACID compliance
   - Concurrent access support
   - Query performance optimization

## API Usage Examples

### Initializing the Registry

```python
from pathlib import Path
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType, PersistenceConfig

# JSON Backend (Development)
registry = DataRegistry(
    registry_path=Path("/tmp/registry"),
    persistence_config=PersistenceConfig(
        backend=BackendType.JSON
    )
)

# PostgreSQL Backend (Production)
registry = DataRegistry(
    registry_path=Path("/tmp/registry"),
    persistence_config=PersistenceConfig(
        backend=BackendType.POSTGRES,
        connection_string="postgresql://user:pass@localhost/mldb"
    )
)
```

### Registering a Dataset

```python
from ml.registry.dataclasses import DatasetManifest, DatasetType, StorageKind

# Define dataset manifest
manifest = DatasetManifest(
    dataset_id="bars_eurusd_1m",
    dataset_type=DatasetType.BARS,
    storage_kind=StorageKind.PARQUET,
    location="/data/bars/eurusd/1m/",
    partitioning=["date"],
    retention_days=90,
    schema={
        "instrument_id": "string",
        "ts_event": "int64",
        "ts_init": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64"
    },
    ts_field="ts_event",
    primary_keys=["instrument_id", "ts_event"],
    version="1.0.0"
)

# Register the dataset
dataset_id = registry.register_dataset(manifest)
print(f"Registered dataset: {dataset_id}")
```

### Defining Data Contracts

```python
from ml.registry.dataclasses import (
    DataContract,
    ValidationRule,
    ValidationRuleType,
    QualityFlag
)

# Define validation rules
contract = DataContract(
    dataset_id="bars_eurusd_1m",
    validation_rules=[
        ValidationRule(
            field="close",
            rule_type=ValidationRuleType.RANGE,
            params={"min": 0.0, "max": 10000.0},
            severity="error"
        ),
        ValidationRule(
            field="volume",
            rule_type=ValidationRuleType.NOT_NULL,
            params={},
            severity="warning"
        ),
        ValidationRule(
            field="ts_event",
            rule_type=ValidationRuleType.MONOTONIC,
            params={},
            severity="error"
        )
    ],
    quality_flags=[
        QualityFlag.VALIDATED,
        QualityFlag.PRODUCTION_READY
    ],
    sla_hours=1.0,
    owner="market_data_team",
    version="1.0.0"
)

# Update the contract
registry.update_contract("bars_eurusd_1m", contract)
```

### Emitting Pipeline Events

```python
import time

# Emit successful processing event
registry.emit_event(
    dataset_id="bars_eurusd_1m",
    instrument_id="EUR/USD",
    stage="CATALOG_WRITTEN",
    source="historical",
    run_id="run_20240115_001",
    ts_min=int(time.time() * 1e9),
    ts_max=int((time.time() + 3600) * 1e9),
    count=1440,  # Number of bars
    status="success"
)

# Emit failure event
registry.emit_event(
    dataset_id="features_v1",
    instrument_id="EUR/USD",
    stage="FEATURE_COMPUTED",
    source="live",
    run_id="run_20240115_002",
    ts_min=int(time.time() * 1e9),
    ts_max=int((time.time() + 3600) * 1e9),
    count=0,
    status="failure",
    error="Database connection timeout"
)
```

### Updating Watermarks

```python
# Update processing watermark
registry.update_watermark(
    dataset_id="bars_eurusd_1m",
    instrument_id="EUR/USD",
    source="historical",
    last_success_ns=int(time.time() * 1e9),
    count=1440,
    completeness_pct=100.0
)

# Query watermark
watermark = registry.get_watermark(
    "bars_eurusd_1m",
    "EUR/USD",
    "historical"
)
print(f"Last processed: {watermark.last_success_ns}")
print(f"Completeness: {watermark.completeness_pct}%")
```

### Tracking Lineage

```python
# Register derived dataset with lineage
features_manifest = DatasetManifest(
    dataset_id="features_v1",
    dataset_type=DatasetType.FEATURES,
    storage_kind=StorageKind.PARQUET,
    location="/data/features/v1/",
    lineage=["bars_eurusd_1m", "bars_gbpusd_1m"],  # Source datasets
    pipeline_signature="hash_of_pipeline_code",
    # ... other fields
)

registry.register_dataset(features_manifest)

# Query lineage
lineage = registry.get_lineage("features_v1")
print(f"Dependencies: {lineage}")
```

## CLI Command Examples

### Coverage Reporting

```bash
# Generate coverage report for a date range
python -m ml.cli.coverage report \
    --dataset BARS \
    --start 2024-01-01 \
    --end 2024-01-07 \
    --instrument EUR/USD

# Coverage report with specific stages
python -m ml.cli.coverage report \
    --dataset FEATURES \
    --date 2024-01-15 \
    --stages CATALOG_WRITTEN FEATURE_COMPUTED

# Export coverage report to CSV
python -m ml.cli.coverage report \
    --dataset PREDICTIONS \
    --start 2024-01-01 \
    --end 2024-01-31 \
    --output coverage_report.csv
```

### Backfill Planning

```bash
# Plan backfill for missing features
python -m ml.cli.coverage plan-backfill \
    --from BARS \
    --to FEATURES \
    --date 2024-01-15 \
    --instrument EUR/USD GBP/USD

# Plan backfill with specific run parameters
python -m ml.cli.coverage plan-backfill \
    --from FEATURES \
    --to PREDICTIONS \
    --date 2024-01-15 \
    --priority high \
    --batch-size 1000

# Generate backfill job spec
python -m ml.cli.coverage plan-backfill \
    --from BARS \
    --to SIGNALS \
    --start 2024-01-01 \
    --end 2024-01-07 \
    --output backfill_jobs.json
```

### Registry Management

```bash
# List all registered datasets
python -m ml.registry.cli list-datasets

# Show dataset details
python -m ml.registry.cli show-dataset --id bars_eurusd_1m

# Validate data contract
python -m ml.registry.cli validate-contract \
    --dataset bars_eurusd_1m \
    --data-path /data/bars/eurusd/2024-01-15.parquet

# Export registry metadata
python -m ml.registry.cli export \
    --output registry_backup.json
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Registry Lock Contention

**Problem**: Multiple processes trying to update registry simultaneously.

**Solution**:
```python
# Use batch operations
with registry.batch_update():
    registry.emit_event(...)
    registry.update_watermark(...)
    # All updates committed atomically
```

#### 2. Missing Watermarks

**Problem**: Watermark queries return None.

**Solution**:
```python
# Check if watermark exists
watermark = registry.get_watermark(dataset_id, instrument_id, source)
if watermark is None:
    # Initialize watermark
    registry.update_watermark(
        dataset_id=dataset_id,
        instrument_id=instrument_id,
        source=source,
        last_success_ns=0,
        count=0,
        completeness_pct=0.0
    )
```

#### 3. Event Ordering Issues

**Problem**: Events appear out of order.

**Solution**:
```python
# Use consistent timestamps
import time

# Get current time once
current_ns = int(time.time() * 1e9)

# Use for all related events
registry.emit_event(
    ts_min=current_ns,
    ts_max=current_ns + int(3600 * 1e9),
    # ...
)
```

#### 4. PostgreSQL Connection Issues

**Problem**: Database connection failures.

**Solution**:
```python
# Configure connection pooling
persistence_config = PersistenceConfig(
    backend=BackendType.POSTGRES,
    connection_string="postgresql://...",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30
)
```

### Performance Optimization

#### 1. Batch Event Emission

```python
# Instead of individual calls
for event in events:
    registry.emit_event(**event)

# Use batch operation
events_batch = [...]
registry.emit_events_batch(events_batch)
```

#### 2. Caching Frequently Accessed Data

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_watermark(dataset_id: str, instrument_id: str) -> Watermark:
    return registry.get_watermark(dataset_id, instrument_id, "historical")
```

#### 3. Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_events_async(events):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, registry.emit_event, **event)
            for event in events
        ]
        await asyncio.gather(*tasks)
```

## Schema Migration Procedures

### Running Migrations

#### 1. Check Current Schema Version

```bash
# Query current version
psql -d mldb -c "SELECT version FROM schema_versions ORDER BY applied_at DESC LIMIT 1;"
```

#### 2. Apply New Migrations

```bash
# Run migration script
python -m ml.registry.migrate \
    --connection "postgresql://user:pass@localhost/mldb" \
    --migrations-dir ml/registry/migrations/
```

#### 3. Rollback Procedure

```sql
-- Rollback to specific version
BEGIN;
-- Run rollback SQL from migration file
-- Update schema_versions table
ROLLBACK; -- or COMMIT if successful
```

### Creating New Migrations

#### 1. Create Migration File

```sql
-- ml/registry/migrations/003_add_new_feature.sql

-- Up migration
CREATE TABLE IF NOT EXISTS new_feature_table (
    id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(255) NOT NULL,
    -- ... other fields
);

-- Add indexes
CREATE INDEX idx_new_feature_dataset ON new_feature_table(dataset_id);

-- Down migration (in comments for reference)
-- DROP TABLE IF EXISTS new_feature_table;
```

#### 2. Test Migration

```bash
# Test on development database
python -m ml.registry.migrate \
    --connection "postgresql://localhost/mldb_dev" \
    --dry-run
```

## Monitoring and Alerting Setup

### Prometheus Metrics

```python
# Registry metrics exposed
from prometheus_client import Counter, Histogram, Gauge

# Event emission metrics
events_emitted = Counter(
    'data_registry_events_total',
    'Total events emitted',
    ['dataset_id', 'stage', 'status']
)

# Watermark lag metrics
watermark_lag = Gauge(
    'data_registry_watermark_lag_seconds',
    'Watermark lag in seconds',
    ['dataset_id', 'instrument_id']
)

# Query latency
query_latency = Histogram(
    'data_registry_query_duration_seconds',
    'Query duration in seconds',
    ['operation']
)
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Data Registry Monitoring",
    "panels": [
      {
        "title": "Event Rate",
        "targets": [
          {
            "expr": "rate(data_registry_events_total[5m])"
          }
        ]
      },
      {
        "title": "Coverage by Stage",
        "targets": [
          {
            "expr": "data_registry_coverage_percent"
          }
        ]
      },
      {
        "title": "Watermark Lag",
        "targets": [
          {
            "expr": "data_registry_watermark_lag_seconds"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# prometheus/alerts/data_registry.yml
groups:
  - name: data_registry
    rules:
      - alert: HighEventFailureRate
        expr: |
          rate(data_registry_events_total{status="failure"}[5m]) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High event failure rate"
          description: "Failure rate {{ $value }} exceeds threshold"

      - alert: WatermarkLag
        expr: |
          data_registry_watermark_lag_seconds > 3600
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Watermark lag exceeds 1 hour"
          description: "Dataset {{ $labels.dataset_id }} lag: {{ $value }}s"

      - alert: LowCoverage
        expr: |
          data_registry_coverage_percent < 90
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Data coverage below 90%"
          description: "Coverage for {{ $labels.dataset_id }}: {{ $value }}%"
```

### Health Check Endpoint

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Registry health check endpoint."""
    try:
        # Check registry connectivity
        registry.get_manifest("test_dataset")

        # Check database connectivity (if using PostgreSQL)
        if registry.backend == BackendType.POSTGRES:
            registry.persistence.check_connection()

        # Calculate metrics
        total_events = len(registry._events)
        total_datasets = len(registry._manifests)

        return jsonify({
            "status": "healthy",
            "backend": registry.backend.value,
            "total_events": total_events,
            "total_datasets": total_datasets,
            "timestamp": time.time()
        }), 200

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 503
```

## Best Practices

### 1. Event Emission
- Always include both ts_min and ts_max for time ranges
- Use consistent run_id for related events
- Include meaningful error messages for failures

### 2. Watermark Management
- Update watermarks after successful processing
- Use completeness_pct to track partial data
- Reset watermarks when reprocessing historical data

### 3. Contract Definition
- Define contracts early in development
- Include both error and warning severity rules
- Version contracts when making changes

### 4. Performance
- Use batch operations for multiple updates
- Enable connection pooling for PostgreSQL
- Cache frequently accessed metadata

### 5. Monitoring
- Set up alerts for critical metrics
- Monitor watermark lag continuously
- Track coverage trends over time

## Integration Examples

### Integration with ML Pipeline

```python
from ml.actors.base import BaseMLInferenceActor
from ml.registry.data_registry import DataRegistry

class MLPipelineActor(BaseMLInferenceActor):
    def __init__(self, config, registry: DataRegistry):
        super().__init__(config)
        self.registry = registry
        self.run_id = f"run_{uuid.uuid4().hex[:8]}"

    def process_data(self, data):
        # Process data
        result = self._compute_features(data)

        # Emit success event
        self.registry.emit_event(
            dataset_id="features_v1",
            instrument_id=data.instrument_id,
            stage="FEATURE_COMPUTED",
            source="live",
            run_id=self.run_id,
            ts_min=data.ts_min,
            ts_max=data.ts_max,
            count=len(result),
            status="success"
        )

        # Update watermark
        self.registry.update_watermark(
            dataset_id="features_v1",
            instrument_id=data.instrument_id,
            source="live",
            last_success_ns=data.ts_max,
            count=len(result),
            completeness_pct=100.0
        )

        return result
```

### Integration with Airflow

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def check_coverage(**context):
    """Check data coverage before processing."""
    from ml.cli.coverage import CoverageReporter

    reporter = CoverageReporter()
    coverage = reporter.generate_coverage_report(
        dataset_type="BARS",
        date=context['ds'],
        instruments=["EUR/USD", "GBP/USD"]
    )

    # Fail if coverage below threshold
    for row in coverage:
        if row['catalog_pct'] < 95.0:
            raise ValueError(f"Low coverage for {row['instrument']}: {row['catalog_pct']}%")

    return coverage

dag = DAG(
    'ml_pipeline',
    schedule_interval='@daily',
    default_args={'retries': 2}
)

coverage_check = PythonOperator(
    task_id='check_coverage',
    python_callable=check_coverage,
    dag=dag
)
```
