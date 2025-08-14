# Production Deployment Guide: ML System with Mandatory Stores

## Overview

This guide covers deploying the Nautilus ML system in production with mandatory data persistence. All ML actors now automatically persist features, predictions, and signals to ensure complete data tracking and feature parity.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Trading System                            │
├─────────────────────────────────────────────────────────────┤
│  ML Actors (BaseMLInferenceActor)                           │
│  ├── Automatic Feature Persistence                          │
│  ├── Automatic Prediction Storage                           │
│  └── Automatic Signal Tracking                              │
├─────────────────────────────────────────────────────────────┤
│  Stores Layer (Always Active)                               │
│  ├── FeatureStore    → ml_feature_values table              │
│  ├── ModelStore      → ml_model_predictions table           │
│  └── StrategyStore   → ml_strategy_signals table            │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL Database (Time-Partitioned)                     │
│  └── Automatic partitioning for 36 months                   │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker and Docker Compose installed
- PostgreSQL 15+ (via Docker or standalone)
- Python 3.10+ with Nautilus Trader
- ML dependencies: `pip install 'nautilus-trader[ml]'`

## Quick Start

### 1. Start Infrastructure

```bash
cd ml/
docker-compose up -d
```

This starts:
- PostgreSQL with automatic migrations
- Optional: PgAdmin for database management
- Optional: Prometheus for metrics
- Optional: Grafana for visualization

### 2. Verify Database

```bash
# Check that PostgreSQL is running
docker ps | grep nautilus-postgres

# Verify tables were created
docker exec -it nautilus-postgres psql -U postgres -d nautilus -c '\dt'
```

Expected tables:
- `ml_feature_values` (partitioned by time)
- `ml_model_predictions` (partitioned by time)
- `ml_strategy_signals` (partitioned by time)
- `ml_feature_metadata`
- `ml_model_metadata`
- `ml_strategy_metadata`

### 3. Deploy ML Actor

```python
from ml.actors.signal import MLSignalActor
from ml.config.actors import MLSignalActorConfig
from nautilus_trader.config import TradingNodeConfig

# Configure actor with database connection
actor_config = MLSignalActorConfig(
    model_id="xgboost_v1",
    model_path="/models/production_model.onnx",
    db_connection="postgresql://postgres:postgres@localhost:5432/nautilus",
    # Stores are automatically initialized - no need to specify
    prediction_threshold=0.7,
    signal_strategy="adaptive",
)

# Add to trading node
node_config = TradingNodeConfig(
    actors=[actor_config],
    # ... other configuration
)
```

## Production Configuration

### Environment Variables

```bash
# .env file
NAUTILUS_DB_HOST=postgres.production.internal
NAUTILUS_DB_PORT=5432
NAUTILUS_DB_NAME=nautilus
NAUTILUS_DB_USER=nautilus_user
NAUTILUS_DB_PASSWORD=secure_password

# Use in code
import os

db_connection = (
    f"postgresql://{os.getenv('NAUTILUS_DB_USER')}:"
    f"{os.getenv('NAUTILUS_DB_PASSWORD')}@"
    f"{os.getenv('NAUTILUS_DB_HOST')}:"
    f"{os.getenv('NAUTILUS_DB_PORT')}/"
    f"{os.getenv('NAUTILUS_DB_NAME')}"
)
```

### High Availability Setup

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  postgres-primary:
    image: postgres:15
    environment:
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: repl_password
    volumes:
      - primary_data:/var/lib/postgresql/data
    
  postgres-replica:
    image: postgres:15
    environment:
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_MASTER_HOST: postgres-primary
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: repl_password
    volumes:
      - replica_data:/var/lib/postgresql/data
    depends_on:
      - postgres-primary
```

### Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Production connection with pooling
engine = create_engine(
    db_connection,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,  # Verify connections
    pool_recycle=3600,   # Recycle connections after 1 hour
)
```

## Monitoring

### Data Persistence Metrics

```sql
-- Check feature storage rate
SELECT 
    DATE_TRUNC('hour', to_timestamp(ts_event/1e9)) as hour,
    COUNT(*) as features_stored,
    COUNT(DISTINCT instrument_id) as instruments
FROM ml_feature_values
WHERE ts_event > extract(epoch from now() - interval '24 hours') * 1e9
GROUP BY hour
ORDER BY hour DESC;

-- Check prediction latency
SELECT 
    model_id,
    AVG(inference_time_ms) as avg_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY inference_time_ms) as p99_latency_ms
FROM ml_model_predictions
WHERE ts_event > extract(epoch from now() - interval '1 hour') * 1e9
GROUP BY model_id;

-- Check signal generation
SELECT 
    strategy_id,
    signal_type,
    COUNT(*) as signal_count,
    AVG(strength) as avg_strength
FROM ml_strategy_signals
WHERE ts_event > extract(epoch from now() - interval '1 hour') * 1e9
GROUP BY strategy_id, signal_type;
```

### Health Checks

```python
# Health check endpoint
from ml.core.integration import MLIntegrationManager

def health_check():
    """Check ML system health."""
    integration = MLIntegrationManager()
    health = integration.check_health()
    
    if all(health.values()):
        return {"status": "healthy", "components": health}
    else:
        failed = [k for k, v in health.items() if not v]
        return {"status": "unhealthy", "failed_components": failed}
```

## Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh

# Backup database
BACKUP_FILE="nautilus_ml_$(date +%Y%m%d_%H%M%S).sql"
docker exec nautilus-postgres pg_dump -U postgres nautilus > /backups/$BACKUP_FILE

# Compress
gzip /backups/$BACKUP_FILE

# Upload to S3 (optional)
aws s3 cp /backups/${BACKUP_FILE}.gz s3://my-backups/nautilus/

# Keep only last 30 days locally
find /backups -name "nautilus_ml_*.gz" -mtime +30 -delete
```

### Point-in-Time Recovery

```bash
# Restore from backup
gunzip < /backups/nautilus_ml_20240115_120000.sql.gz | \
  docker exec -i nautilus-postgres psql -U postgres nautilus
```

## Performance Tuning

### PostgreSQL Configuration

```sql
-- postgresql.conf optimizations
shared_buffers = 8GB              # 25% of RAM
effective_cache_size = 24GB       # 75% of RAM
maintenance_work_mem = 2GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1            # For SSD storage
effective_io_concurrency = 200    # For SSD storage
work_mem = 32MB
max_connections = 200
```

### Partition Maintenance

```sql
-- Add future partitions
CALL create_monthly_partitions('ml_feature_values', 3);
CALL create_monthly_partitions('ml_model_predictions', 3);
CALL create_monthly_partitions('ml_strategy_signals', 3);

-- Drop old partitions
CALL drop_old_partitions('ml_feature_values', 12);  -- Keep 12 months
```

## Troubleshooting

### Common Issues

#### 1. Connection Refused
```python
# Error: connection to server at "localhost", port 5432 failed
# Solution: Ensure PostgreSQL is running
docker-compose up -d postgres
```

#### 2. Table Does Not Exist
```python
# Error: relation "ml_feature_values" does not exist
# Solution: Run migrations
docker exec -i nautilus-postgres psql -U postgres nautilus < ml/stores/migrations/001_stores_schema.sql
```

#### 3. Disk Space Issues
```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Vacuum and analyze
VACUUM ANALYZE ml_feature_values;
VACUUM ANALYZE ml_model_predictions;
```

## Security Best Practices

### 1. Use SSL Connections
```python
db_connection = (
    "postgresql://user:pass@host:5432/db"
    "?sslmode=require&sslcert=client.crt&sslkey=client.key"
)
```

### 2. Rotate Credentials
```python
# Use AWS Secrets Manager or similar
import boto3

def get_db_connection():
    client = boto3.client('secretsmanager')
    secret = client.get_secret_value(SecretId='nautilus/db/credentials')
    creds = json.loads(secret['SecretString'])
    return f"postgresql://{creds['username']}:{creds['password']}@{creds['host']}/nautilus"
```

### 3. Audit Logging
```sql
-- Enable logging of all data modifications
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
```

## Scaling Considerations

### Horizontal Scaling
- Use read replicas for analytics queries
- Partition data by instrument_id for sharding
- Use connection pooling with PgBouncer

### Vertical Scaling
- Monitor memory usage and adjust shared_buffers
- Use NVMe SSDs for best I/O performance
- Consider TimescaleDB for time-series optimization

## Conclusion

The mandatory store integration ensures that all ML data is automatically persisted without any additional configuration. This provides:

1. **Complete Audit Trail**: Every prediction and signal is tracked
2. **Feature Parity**: Training and inference use identical features
3. **Performance Monitoring**: Built-in latency and accuracy tracking
4. **Disaster Recovery**: All data is persisted and can be restored
5. **Zero Configuration**: Stores work automatically out of the box

For questions or issues, consult the logs:
```bash
# Actor logs
docker logs nautilus-trader

# Database logs
docker logs nautilus-postgres

# Check store statistics
docker exec -it nautilus-postgres psql -U postgres -d nautilus -c "
SELECT 'features' as store, COUNT(*) FROM ml_feature_values
UNION ALL
SELECT 'predictions', COUNT(*) FROM ml_model_predictions
UNION ALL
SELECT 'signals', COUNT(*) FROM ml_strategy_signals;
"
```