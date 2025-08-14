# ML Data Processing Architecture

## Overview
This document outlines the complete data processing pipeline for ML trading systems, covering market data ingestion through strategy execution, with proper storage and retrieval patterns.

## Data Flow Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     Market Data Sources                      │
│         (Databento, Interactive Brokers, Binance)           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
│   • Normalization  • Validation  • Deduplication            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Engineering                       │
│   • Technical Indicators  • Market Microstructure           │
│   • Alternative Data Integration                            │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                     Model Inference                          │
│   • Prediction Generation  • Confidence Scoring             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   Strategy Execution                         │
│   • Signal Generation  • Risk Management  • Order Routing   │
└─────────────────────────────────────────────────────────────┘
```

## 1. Market Data Processing

### Required Processing Steps

#### A. Data Ingestion

```python
# Required transformations
- Timestamp normalization to nanoseconds (Nautilus standard)
- Symbol mapping and instrument ID standardization
- Exchange-specific adjustments (tick size, lot size)
- Corporate action adjustments (splits, dividends)
```

#### B. Data Quality

```python
# Validation requirements
- Outlier detection (price spikes, fat fingers)
- Gap detection and handling
- Sequence validation (monotonic timestamps)
- Cross-validation between data sources
```

#### C. Storage Schema

```sql
-- Market data with metadata
CREATE TABLE market_data (
    instrument_id VARCHAR(100),
    ts_event BIGINT,           -- When event occurred
    ts_init BIGINT,            -- When processed
    bid DECIMAL(20,8),
    ask DECIMAL(20,8),
    bid_size DECIMAL(20,8),
    ask_size DECIMAL(20,8),
    metadata JSONB,            -- Exchange-specific data
    quality_score FLOAT,       -- Data quality metric
    PRIMARY KEY (instrument_id, ts_event)
) PARTITION BY RANGE (ts_event);

-- Market data metadata
CREATE TABLE market_data_metadata (
    instrument_id VARCHAR(100) PRIMARY KEY,
    symbol VARCHAR(50),
    exchange VARCHAR(50),
    asset_class VARCHAR(20),
    tick_size DECIMAL(20,8),
    lot_size DECIMAL(20,8),
    currency VARCHAR(10),
    timezone VARCHAR(50),
    trading_hours JSONB,
    last_updated TIMESTAMP
);

-- Data quality tracking
CREATE TABLE data_quality_metrics (
    instrument_id VARCHAR(100),
    date DATE,
    gaps_detected INTEGER,
    outliers_removed INTEGER,
    duplicates_removed INTEGER,
    completeness_pct FLOAT,
    latency_ms FLOAT,
    PRIMARY KEY (instrument_id, date)
);
```

## 2. Feature Processing

### Required Processing Steps

#### A. Feature Computation

```python
# Real-time requirements
- Incremental computation (no full recalculation)
- State management for indicators
- Windowing and buffering
- Missing data imputation
```

#### B. Feature Validation

```python
# Quality checks
- NaN/Inf detection
- Range validation
- Correlation stability
- Feature importance tracking
```

#### C. Storage Schema

```sql
-- Feature values with lineage
CREATE TABLE ml_feature_values_extended (
    -- Existing fields...
    computation_version VARCHAR(50),  -- Track feature version
    parent_features JSONB,            -- Lineage tracking
    quality_flags INTEGER,            -- Bit flags for quality
    INDEX idx_computation_version (computation_version)
);

-- Feature metadata
CREATE TABLE feature_metadata (
    feature_set_id VARCHAR(255) PRIMARY KEY,
    feature_names TEXT[],
    feature_types JSONB,              -- Type info per feature
    computation_graph JSONB,          -- DAG of dependencies
    statistics JSONB,                 -- Mean, std, quantiles
    last_computed TIMESTAMP,
    computation_cost_ms FLOAT
);

-- Feature drift monitoring
CREATE TABLE feature_drift (
    feature_set_id VARCHAR(255),
    feature_name VARCHAR(100),
    date DATE,
    mean_value FLOAT,
    std_value FLOAT,
    min_value FLOAT,
    max_value FLOAT,
    null_count INTEGER,
    drift_score FLOAT,               -- Statistical drift metric
    PRIMARY KEY (feature_set_id, feature_name, date)
);
```

## 3. Model Processing

### Required Processing Steps

#### A. Inference Pipeline

```python
# Pre-inference
- Feature alignment and scaling
- Missing value handling
- Ensemble preparation

# Post-inference
- Prediction calibration
- Confidence adjustment
- Outlier detection
```

#### B. Model Versioning

```python
# Version tracking
- Model binary storage
- Hyperparameter logging
- Training data snapshot
- Performance baselines
```

#### C. Storage Schema

```sql
-- Extended model predictions
CREATE TABLE ml_model_predictions_extended (
    -- Existing fields...
    model_version VARCHAR(50),
    feature_version VARCHAR(50),
    calibration_applied BOOLEAN,
    ensemble_members JSONB,          -- If ensemble
    explanation JSONB,               -- SHAP/LIME values
    INDEX idx_model_version (model_version)
);

-- Model metadata
CREATE TABLE model_metadata (
    model_id VARCHAR(255),
    version VARCHAR(50),
    algorithm VARCHAR(100),
    hyperparameters JSONB,
    training_metrics JSONB,
    feature_importance JSONB,
    model_size_bytes BIGINT,
    inference_latency_ms FLOAT,
    deployed_at TIMESTAMP,
    retired_at TIMESTAMP,
    PRIMARY KEY (model_id, version)
);

-- Model performance tracking
CREATE TABLE model_performance (
    model_id VARCHAR(255),
    date DATE,
    prediction_count INTEGER,
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    PRIMARY KEY (model_id, date)
);
```

## 4. Strategy Processing

### Required Processing Steps

#### A. Signal Processing

```python
# Signal generation
- Multi-model aggregation
- Signal strength calculation
- Regime-based adjustments
- Temporal alignment
```

#### B. Risk Management

```python
# Risk checks
- Position limits
- Exposure calculation
- Correlation limits
- Drawdown controls
```

#### C. Storage Schema

```sql
-- Extended strategy signals
CREATE TABLE ml_strategy_signals_extended (
    -- Existing fields...
    regime VARCHAR(50),
    volatility_regime VARCHAR(20),
    correlation_matrix JSONB,
    risk_budget_used FLOAT,
    expected_slippage FLOAT,
    INDEX idx_regime (regime)
);

-- Strategy metadata
CREATE TABLE strategy_metadata (
    strategy_id VARCHAR(255) PRIMARY KEY,
    strategy_type VARCHAR(50),
    instruments TEXT[],
    models_used TEXT[],
    risk_parameters JSONB,
    position_limits JSONB,
    active BOOLEAN,
    created_at TIMESTAMP
);

-- Strategy performance
CREATE TABLE strategy_performance (
    strategy_id VARCHAR(255),
    date DATE,
    pnl DECIMAL(20,8),
    trades_count INTEGER,
    win_rate FLOAT,
    avg_win DECIMAL(20,8),
    avg_loss DECIMAL(20,8),
    sharpe_ratio FLOAT,
    sortino_ratio FLOAT,
    max_drawdown FLOAT,
    var_95 DECIMAL(20,8),
    PRIMARY KEY (strategy_id, date)
);
```

## 5. Registry Integration

### Required Processing Steps

#### A. Metadata Synchronization

```python
# Registry updates
- Feature schema changes
- Model deployments
- Strategy activations
- Version migrations
```

#### B. Audit Trail

```python
# Change tracking
- Configuration changes
- Model updates
- Feature modifications
- Performance degradation
```

#### C. Storage Schema

```sql
-- Unified registry audit
CREATE TABLE registry_audit_log (
    id BIGSERIAL PRIMARY KEY,
    registry_type VARCHAR(50),      -- 'feature', 'model', 'strategy'
    entity_id VARCHAR(255),
    action VARCHAR(50),             -- 'create', 'update', 'delete'
    old_value JSONB,
    new_value JSONB,
    user_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT NOW(),
    reason TEXT
);

-- Registry dependencies
CREATE TABLE registry_dependencies (
    id BIGSERIAL PRIMARY KEY,
    from_type VARCHAR(50),
    from_id VARCHAR(255),
    to_type VARCHAR(50),
    to_id VARCHAR(255),
    dependency_type VARCHAR(50),    -- 'uses', 'requires', 'produces'
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(from_type, from_id, to_type, to_id)
);
```

## 6. Data Processing Pipeline

### Complete Processing Implementation

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from datetime import datetime

@dataclass
class ProcessingContext:
    """Context for data processing pipeline."""
    instrument_id: str
    timestamp_ns: int
    processing_version: str
    metadata: Dict[str, Any]

class DataProcessor:
    """Unified data processing pipeline."""

    def process_market_data(self, raw_data: Any) -> Dict[str, Any]:
        """Process raw market data."""
        # 1. Validate timestamps
        self._validate_timestamps(raw_data)

        # 2. Normalize prices
        normalized = self._normalize_prices(raw_data)

        # 3. Detect outliers
        cleaned = self._remove_outliers(normalized)

        # 4. Add metadata
        enriched = self._enrich_metadata(cleaned)

        # 5. Calculate quality score
        quality = self._calculate_quality_score(enriched)

        return {
            "data": enriched,
            "quality": quality,
            "processing_time_ms": self._get_processing_time()
        }

    def process_features(self, market_data: Any,
                        feature_config: Any) -> Dict[str, Any]:
        """Process features with lineage tracking."""
        # 1. Compute features
        features = self._compute_features(market_data, feature_config)

        # 2. Validate features
        validated = self._validate_features(features)

        # 3. Track lineage
        lineage = self._track_lineage(validated, market_data)

        # 4. Monitor drift
        drift = self._monitor_drift(validated)

        return {
            "features": validated,
            "lineage": lineage,
            "drift": drift,
            "version": feature_config.version
        }

    def process_predictions(self, features: np.ndarray,
                          model: Any) -> Dict[str, Any]:
        """Process model predictions."""
        # 1. Generate predictions
        raw_pred = model.predict(features)

        # 2. Calibrate predictions
        calibrated = self._calibrate_predictions(raw_pred)

        # 3. Calculate confidence
        confidence = self._calculate_confidence(calibrated, model)

        # 4. Generate explanations
        explanations = self._generate_explanations(features, model)

        return {
            "prediction": calibrated,
            "confidence": confidence,
            "explanation": explanations,
            "model_version": model.version
        }

    def process_signals(self, predictions: Dict[str, Any],
                       strategy_config: Any) -> Dict[str, Any]:
        """Process strategy signals."""
        # 1. Generate signal
        signal = self._generate_signal(predictions, strategy_config)

        # 2. Apply risk filters
        risk_adjusted = self._apply_risk_filters(signal)

        # 3. Check regime
        regime_adjusted = self._apply_regime_adjustments(risk_adjusted)

        # 4. Calculate execution params
        execution = self._calculate_execution_params(regime_adjusted)

        return {
            "signal": regime_adjusted,
            "execution": execution,
            "risk_metrics": self._get_risk_metrics()
        }
```

## 7. PostgreSQL Container Setup

### Required Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres-ml:
    image: postgres:15-alpine
    container_name: nautilus-ml-postgres
    environment:
      POSTGRES_DB: nautilus
      POSTGRES_USER: nautilus
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --locale=C"
    ports:
      - "5432:5432"
    volumes:
      - ./postgres-data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d
    command:
      - "postgres"
      - "-c"
      - "shared_buffers=2GB"
      - "-c"
      - "max_connections=200"
      - "-c"
      - "effective_cache_size=6GB"
      - "-c"
      - "maintenance_work_mem=512MB"
      - "-c"
      - "random_page_cost=1.1"
      - "-c"
      - "effective_io_concurrency=200"
      - "-c"
      - "work_mem=10MB"
      - "-c"
      - "huge_pages=try"
      - "-c"
      - "max_wal_size=4GB"
      - "-c"
      - "min_wal_size=1GB"
      - "-c"
      - "checkpoint_completion_target=0.9"
      - "-c"
      - "wal_buffers=16MB"
      - "-c"
      - "default_statistics_target=100"
      - "-c"
      - "random_page_cost=1.1"
      # Partitioning optimizations
      - "-c"
      - "constraint_exclusion=partition"
      - "-c"
      - "enable_partition_pruning=on"
      - "-c"
      - "enable_partitionwise_join=on"
      - "-c"
      - "enable_partitionwise_aggregate=on"

  timescaledb:
    # Alternative: Use TimescaleDB for better time-series support
    image: timescale/timescaledb-ha:pg15-latest
    container_name: nautilus-ml-timescale
    environment:
      POSTGRES_DB: nautilus
      POSTGRES_USER: nautilus
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5433:5432"
```

### Indexing Strategy

```sql
-- Critical indexes for performance
CREATE INDEX CONCURRENTLY idx_market_data_lookup
    ON market_data (instrument_id, ts_event DESC);

CREATE INDEX CONCURRENTLY idx_features_lookup
    ON ml_feature_values (feature_set_id, instrument_id, ts_event DESC);

CREATE INDEX CONCURRENTLY idx_predictions_lookup
    ON ml_model_predictions (model_id, instrument_id, ts_event DESC);

CREATE INDEX CONCURRENTLY idx_signals_lookup
    ON ml_strategy_signals (strategy_id, instrument_id, ts_event DESC);

-- BRIN indexes for time-series data (space-efficient)
CREATE INDEX idx_market_data_ts_brin
    ON market_data USING BRIN (ts_event);

CREATE INDEX idx_features_ts_brin
    ON ml_feature_values USING BRIN (ts_event);

-- Partial indexes for common queries
CREATE INDEX idx_live_predictions
    ON ml_model_predictions (ts_event DESC)
    WHERE is_live = TRUE;

CREATE INDEX idx_active_signals
    ON ml_strategy_signals (ts_event DESC)
    WHERE signal_type != 'HOLD';
```

## 8. Data Retrieval Optimization

### Query Patterns

```python
class OptimizedDataRetriever:
    """Optimized data retrieval patterns."""

    def get_training_data(self, start_date: datetime,
                         end_date: datetime) -> pd.DataFrame:
        """Retrieve training data efficiently."""
        query = """
        WITH feature_data AS (
            SELECT
                f.instrument_id,
                f.ts_event,
                f.values,
                f.quality_flags
            FROM ml_feature_values f
            WHERE f.ts_event BETWEEN %s AND %s
            AND f.quality_flags = 0  -- Only clean data
        ),
        market_data AS (
            SELECT
                m.instrument_id,
                m.ts_event,
                m.bid,
                m.ask,
                m.volume
            FROM market_data m
            WHERE m.ts_event BETWEEN %s AND %s
            AND m.quality_score > 0.95
        )
        SELECT
            f.*,
            m.bid,
            m.ask,
            m.volume,
            -- Calculate target (next bar return)
            LEAD(m.bid, 1) OVER (
                PARTITION BY f.instrument_id
                ORDER BY f.ts_event
            ) as next_bid
        FROM feature_data f
        JOIN market_data m
            ON f.instrument_id = m.instrument_id
            AND f.ts_event = m.ts_event
        ORDER BY f.instrument_id, f.ts_event
        """

        return pd.read_sql(query, self.connection,
                          params=[start_date, end_date,
                                 start_date, end_date])

    def get_realtime_features(self, instrument_id: str,
                            lookback_bars: int) -> np.ndarray:
        """Get features for real-time inference."""
        query = """
        SELECT values
        FROM ml_feature_values
        WHERE instrument_id = %s
        ORDER BY ts_event DESC
        LIMIT %s
        """

        # Use prepared statement for performance
        with self.connection.cursor() as cursor:
            cursor.execute("PREPARE realtime_stmt AS " + query)
            cursor.execute("EXECUTE realtime_stmt (%s, %s)",
                         (instrument_id, lookback_bars))
            return np.array(cursor.fetchall())
```

## 9. Data Integrity and Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh - Automated backup with partitioning awareness

# Backup only recent partitions (last 3 months)
TABLES="ml_feature_values ml_model_predictions ml_strategy_signals"
for table in $TABLES; do
    for month in $(seq 0 2); do
        partition_date=$(date -d "$month months ago" +%Y_%m)
        partition_name="${table}_${partition_date}"

        pg_dump -h localhost -U nautilus -t $partition_name \
                --format=custom --compress=9 \
                -f "backup/${partition_name}.dump" nautilus
    done
done

# Backup metadata tables (small, critical)
pg_dump -h localhost -U nautilus \
        -t "*_metadata" -t "*_registry" \
        --format=custom \
        -f "backup/metadata_$(date +%Y%m%d).dump" nautilus
```

### Data Validation

```python
class DataValidator:
    """Validate data integrity across pipeline."""

    def validate_pipeline(self) -> Dict[str, bool]:
        """Run complete pipeline validation."""
        return {
            "market_data": self._validate_market_data(),
            "features": self._validate_features(),
            "predictions": self._validate_predictions(),
            "signals": self._validate_signals(),
            "consistency": self._validate_consistency()
        }

    def _validate_consistency(self) -> bool:
        """Check cross-table consistency."""
        checks = []

        # Check feature-prediction alignment
        query = """
        SELECT COUNT(*) = 0 as valid
        FROM ml_model_predictions p
        LEFT JOIN ml_feature_values f
            ON p.instrument_id = f.instrument_id
            AND p.ts_event = f.ts_event
        WHERE f.ts_event IS NULL
        AND p.ts_event > NOW() - INTERVAL '1 day'
        """
        checks.append(self._run_check(query))

        # Check signal-prediction alignment
        query = """
        SELECT COUNT(*) = 0 as valid
        FROM ml_strategy_signals s
        LEFT JOIN ml_model_predictions p
            ON s.instrument_id = p.instrument_id
            AND s.ts_event = p.ts_event
        WHERE p.ts_event IS NULL
        AND s.ts_event > NOW() - INTERVAL '1 day'
        """
        checks.append(self._run_check(query))

        return all(checks)
```

## 10. Performance Monitoring

### Key Metrics to Track

```sql
-- Processing latency view
CREATE VIEW processing_latency AS
SELECT
    DATE_TRUNC('hour', TO_TIMESTAMP(ts_event/1e9)) as hour,
    AVG((ts_init - ts_event)/1e6) as avg_latency_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY (ts_init - ts_event)/1e6) as p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY (ts_init - ts_event)/1e6) as p95_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY (ts_init - ts_event)/1e6) as p99_ms,
    COUNT(*) as count
FROM ml_model_predictions
WHERE ts_event > EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours') * 1e9
GROUP BY hour
ORDER BY hour DESC;

-- Data quality dashboard
CREATE VIEW data_quality_dashboard AS
SELECT
    instrument_id,
    DATE_TRUNC('day', TO_TIMESTAMP(ts_event/1e9)) as date,
    COUNT(*) as total_records,
    SUM(CASE WHEN quality_flags = 0 THEN 1 ELSE 0 END) as clean_records,
    AVG(quality_score) as avg_quality,
    COUNT(DISTINCT ts_event) as unique_timestamps
FROM market_data
WHERE ts_event > EXTRACT(EPOCH FROM NOW() - INTERVAL '7 days') * 1e9
GROUP BY instrument_id, date;
```

## Summary

Complete data processing requires:

1. **Ingestion**: Normalize, validate, deduplicate
2. **Storage**: Partition, index, compress
3. **Processing**: Transform, enrich, validate
4. **Versioning**: Track schemas, models, configs
5. **Monitoring**: Latency, quality, drift
6. **Recovery**: Backup, replay, reconciliation
7. **Optimization**: Caching, indexing, query planning

This architecture ensures data integrity, performance, and reliability across the entire ML trading pipeline.
