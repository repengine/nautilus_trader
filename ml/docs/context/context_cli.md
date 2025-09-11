# Context: CLI Module

## Overview

The ML CLI module provides command-line interfaces for managing the Nautilus Trader ML pipeline. It includes comprehensive tools for data coverage reporting, backfill planning/execution, system health monitoring, event streaming, and observability management. The CLI tools follow a consistent pattern using argparse for argument parsing and support both PostgreSQL and JSON backend configurations with automatic fallback strategies.

**Key CLI Commands:**

- **coverage**: Data coverage reporting and backfill management for the ML pipeline (1,600+ lines)
- **health**: System health aggregation and monitoring via MLIntegrationManager
- **feature_backfill_cli**: Parallel feature computation and backfilling with thread pools
- **feature_cli**: Feature registry management and lifecycle operations
- **events_consumer**: Redis streams event consumption with topic filtering
- **observability**: Observability data flushing and background processing
- **ingest_backfill**: Gap backfill orchestration with pluggable coverage and writers

## Architecture

### Module Structure

```
ml/cli/
├── coverage.py              # Comprehensive coverage reporting and backfill system (1,600+ lines)
├── health.py               # System health monitoring CLI (42 lines)
├── feature_backfill_cli.py # Parallel feature backfilling (115 lines)
├── feature_cli.py          # Feature registry management (87 lines)
├── events_consumer.py      # Redis streams event consumer (109 lines)
└── observability.py        # Observability flushing CLI (116 lines)
```

### Command Invocation Pattern
All CLI tools follow the Python module execution pattern:

```bash
python -m ml.cli.coverage [command] [options]
python -m ml.cli.health [options]
python -m ml.cli.feature_backfill_cli [options]
python -m ml.cli.events_consumer [options]
python -m ml.cli.observability [command] [options]
python -m ml.cli.ingest_backfill [options]
```

### Backend Configuration Strategy
The CLI tools implement a consistent dual-backend approach:

- **Primary**: PostgreSQL backend via `NAUTILUS_REGISTRY_DB_URL` environment variable
- **Fallback**: JSON file backend with configurable path (default: `ml_registry/`)
- **Auto-detection**: Attempts PostgreSQL first, gracefully falls back to JSON if unavailable

## Key Components

### Coverage CLI (`coverage.py`)

**Purpose**: Comprehensive data coverage analysis and automated backfill orchestration

**Core Classes:**

- `CoverageReporter`: Main class for generating pipeline coverage reports
  - Supports both PostgreSQL and JSON backends
  - Generates tabulated coverage reports with stage-by-stage analysis
  - Tracks data flow through: `CATALOG_WRITTEN → FEATURE_COMPUTED → PREDICTION_EMITTED → SIGNAL_EMITTED`

**Commands:**

- `report`: Generate coverage reports showing data flow through pipeline stages
- `plan-backfill`: Identify gaps and create backfill job specifications
- `apply-backfill`: Execute backfill jobs with rate limiting and retry logic

**Key Features:**

- **Stage Coverage Analysis**: Tracks percentage coverage across all pipeline stages
- **Lag Monitoring**: Measures time since last successful processing per instrument
- **Gap Detection**: Identifies missing data where source exists but target is missing
- **Backfill Planning**: Creates JSON job specifications for missing data gaps
- **Production Execution**: Rate-limited API calls with exponential backoff retry
- **Databento Integration**: Native support for fetching historical data via Databento API

### Health CLI (`health.py`)

**Purpose**: System health aggregation and monitoring dashboard integration

**Core Functionality:**

- Utilizes `MLIntegrationManager` for comprehensive health checks
- Aggregates health status across all ML components (stores, registries, actors)
- Outputs JSON-formatted health summaries suitable for monitoring dashboards
- Supports strict protocol validation mode for enhanced error detection

**Usage Pattern:**

```bash
python -m ml.cli.health [--db-connection <url>] [--strict]
```

### Feature Backfill CLI (`feature_backfill_cli.py`)

**Purpose**: Parallel feature computation and historical backfilling

**Core Functionality:**

- Leverages `FeatureStore.compute_historical_parallel()` for multi-threaded processing
- Supports file-based or comma-separated instrument lists
- Flexible date range specification (ISO 8601 format)
- Configurable worker threads for optimal resource utilization
- Force recompute option for data refresh scenarios

**Key Features:**

- **Parallel Processing**: Configurable worker thread pool (default: 4 workers)
- **Input Flexibility**: Supports both comma-separated lists and file input for instruments
- **Progress Reporting**: Provides completion statistics and per-instrument row counts
- **Error Handling**: Graceful handling of failed instruments with detailed reporting

### Feature CLI (`feature_cli.py`)

**Purpose**: Feature registry lifecycle management and operations

**Core Functions:**

- `cli_register_default()`: Register default FeatureConfig as a feature set
- `cli_promote_with_gates()`: Quality gate validation and promotion
- `cli_deprecate()`: Feature set deprecation with optional reason tracking

**Integration Points:**

- `FeatureRegistry`: Direct interaction with feature registry for lifecycle operations
- `FeatureEngineer`: Manifest generation for feature set registration
- `QualityGate`: Metric-based validation for feature promotion

### Events Consumer CLI (`events_consumer.py`)

**Purpose**: Redis streams event consumption with topic filtering and idempotent processing

**Core Features:**

- **Redis Streams Integration**: Subscribes to Redis streams with configurable stream names
- **Topic Pattern Filtering**: Wildcard pattern matching using `*` and `#` semantics
- **Idempotent Processing**: Built-in watermark gating to prevent duplicate processing
- **JSON Event Handling**: Processes events with `topic` and `payload` fields
- **Configurable Polling**: Supports blocking/non-blocking reads with iteration control

**Key Configuration:**

```python
# Environment variables
ML_BUS_REDIS_URL="redis://localhost:6379/0"  # Redis connection
ML_BUS_REDIS_STREAM="ml-events"              # Stream name
```

**Usage Examples:**

```bash
# Subscribe to feature computation events
python -m ml.cli.events_consumer \
  --redis-url redis://localhost:6379/0 \
  --stream ml-events \
  --pattern events.ml.FEATURE_COMPUTED.# \
  --iterations 1 --count 100

# Multiple pattern filtering
python -m ml.cli.events_consumer \
  --pattern events.ml.FEATURE_* \
  --pattern events.ml.PREDICTION_* \
  --block-ms 5000
```

### Observability CLI (`observability.py`)

**Purpose**: Observability data management with multiple sinks and background processing

**Core Commands:**

- `flush-jsonl`: Export observability data to JSONL/CSV files
- `flush-db`: Flush observability data to PostgreSQL database
- `start`: Start background observability data collection with periodic flushing

**Key Features:**

- **Multi-Sink Support**: File (JSONL/CSV) and database persistence options
- **Background Processing**: Configurable interval-based automatic flushing
- **Comprehensive Metrics**: Latency stages, custom metrics, event correlations, health scores
- **Sample Data Seeding**: `--seed-sample` flag for testing and demonstration

**Integration Points:**

- `MLIntegrationManager`: Core observability pipeline initialization
- `ObservabilityService`: Metrics collection and correlation tracking
- Database schemas for persistent storage of observability data

**Usage Examples:**

```bash
# Flush current observability data to files
python -m ml.cli.observability flush-jsonl \
  --base-path ./observability \
  --format jsonl \
  --seed-sample

# Start background collection
python -m ml.cli.observability start \
  --sink db \
  --db-url postgresql://user:pass@host/db \
  --interval 30.0 \
  --duration 3600.0

# Flush to database
python -m ml.cli.observability flush-db \
  --db-url postgresql://user:pass@host/db
```

## Dependencies

### Internal Dependencies

```python
# Core Integration
from ml.core.integration import MLIntegrationManager

# Feature System
from ml.features.engineering import FeatureConfig, FeatureEngineer
from ml.stores.feature_store import FeatureStore

# Registry System
from ml.registry.data_registry import DataRegistry
from ml.registry.feature_registry import FeatureRegistry, FeatureRole
from ml.registry.persistence import PersistenceManager, PersistenceConfig, BackendType
from ml.registry.dataclasses import DatasetType, QualityGate
from ml.registry.base import DataRequirements

# Configuration
from ml.config.events import Source, Stage
from ml.config.constants import Versions

# Event Processing
from ml.common.topic_filters import match_topic
from ml.consumers.redis_streams_consumer import RedisStreamsConsumer
```

### External Dependencies

```python
# Standard Library
import argparse, json, logging, os, sys, time, uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-Party
import numpy as np
from sqlalchemy import text
```

### Optional Dependencies

- **tabulate**: Enhanced table formatting for coverage reports (graceful degradation)
- **databento**: Historical data fetching for backfill operations (required for production backfill)
- **redis**: Redis streams support for event consumption (required for events_consumer)
- **pandas**: Data export and CSV formatting support (observability CLI)

## Usage Patterns

### Coverage Reporting Workflow

```bash
# 1. Generate coverage report for dataset
python -m ml.cli.coverage report --dataset BARS --start 2024-01-01 --end 2024-01-07

# 2. Identify gaps and plan backfill
python -m ml.cli.coverage plan-backfill --from BARS --to FEATURES --date 2024-01-15

# 3. Execute backfill job with safety measures
python -m ml.cli.coverage apply-backfill --job-file backfill_job.json --dry-run
python -m ml.cli.coverage apply-backfill --job-file backfill_job.json
```

### Feature Management Workflow

```bash
# 1. Backfill historical features in parallel
python -m ml.cli.feature_backfill_cli --db "postgresql://..." --instruments EUR/USD,GBP/USD --max-workers 8

# 2. Monitor system health
python -m ml.cli.health --db-connection "postgresql://..." --strict

# 3. Stream event consumption
python -m ml.cli.events_consumer --redis-url redis://localhost:6379/0 --stream ml-events --pattern events.ml.#

# 4. Observability data management
python -m ml.cli.observability start --sink file --interval 60.0 --duration 3600.0
```

### Environment Configuration

```bash
# Primary backend (PostgreSQL)
export NAUTILUS_REGISTRY_DB_URL="postgresql://user:pass@host:port/db"

# Databento integration (for backfill)
export DATABENTO_API_KEY="your_api_key_here"

# Catalog path (for data storage)
export NAUTILUS_CATALOG_PATH="./catalog"

# Redis event streaming (for events_consumer)
export ML_BUS_REDIS_URL="redis://localhost:6379/0"
export ML_BUS_REDIS_STREAM="ml-events"
```

## Integration Points

### Registry System Integration
All CLI tools integrate with the universal registry system:

- **Data Registry**: Event tracking and watermark management
- **Feature Registry**: Feature set lifecycle and validation
- **Model Registry**: Model deployment tracking (future integration)
- **Strategy Registry**: Strategy compatibility validation (future integration)

### Store Integration
CLI tools leverage the mandatory 4-store pattern:

- **FeatureStore**: Historical feature computation and retrieval
- **DataStore**: Unified data access with contract validation
- **ModelStore**: Prediction storage and performance tracking
- **StrategyStore**: Trading decision persistence

### Pipeline Integration
The coverage CLI provides visibility into the complete ML pipeline:

```
Raw Data → CATALOG_WRITTEN → FEATURE_COMPUTED → PREDICTION_EMITTED → SIGNAL_EMITTED
```

### External System Integration

- **Databento API**: Historical market data fetching with rate limiting
- **PostgreSQL**: Primary persistence layer with connection pooling
- **Prometheus**: Metrics integration via shared bootstrap module
- **Docker/Compose**: Health monitoring integration for containerized deployments
- **Redis Streams**: Event streaming and message bus integration for real-time processing
- **File System**: Observability data export to JSONL/CSV for external analytics

## Implementation Notes

### Performance Considerations

- **Coverage Queries**: Complex SQL queries optimized for PostgreSQL with proper indexing assumptions
- **Parallel Processing**: Feature backfill uses thread pools with configurable worker limits
- **Memory Management**: Streaming processing for large datasets to prevent OOM conditions
- **Rate Limiting**: Databento API calls throttled to prevent quota exhaustion

### Error Handling Patterns

- **Progressive Fallback**: PostgreSQL → JSON backend with warning messages
- **Retry Logic**: Exponential backoff for API failures with configurable max retries
- **Validation**: Input validation with descriptive error messages and early exit
- **Resource Management**: Proper session/connection cleanup in finally blocks

### Security Considerations

- **API Key Management**: Environment variable based configuration only
- **SQL Injection**: Parameterized queries using SQLAlchemy text() with bound parameters
- **File Path Validation**: Path validation for JSON backend and output file operations
- **Connection Security**: Database connection string validation and secure defaults

### Monitoring Integration

- **Health Aggregation**: JSON output suitable for monitoring dashboard consumption
- **Progress Reporting**: Real-time progress updates during long-running operations
- **Metrics Emission**: Integration with shared metrics bootstrap for operational visibility
- **Error Tracking**: Structured logging with appropriate severity levels for operations team

### Future Extensions
The CLI architecture supports easy extension for additional commands:

- Model deployment and rollback operations
- Strategy performance analysis and comparison
- Data quality monitoring and alerting
- Automated pipeline health checks and remediation

### Ingest Backfill CLI (`ingest_backfill.py`)

Purpose: Identify missing UTC day buckets and backfill via an ingestion client (catalog by default), persisting to the canonical SQL store with registry integration.

Options:

- `--db`: Postgres URL (defaults `DB_CONNECTION`)
- `--dataset-id`: e.g., `EQUS.MINI`
- `--schema`: `bars|tbbo|trades` (bars default for catalog client)
- `--instruments`: Comma list or file path
- `--lookback-days`: Default 7 (env `BACKFILL_LOOKBACK_DAYS`)
- `--coverage-mode`: `sql|catalog` (default `sql`)
- `--write-mode`: `sql` (default `sql`; `parquet` not implemented by default)
- `--catalog-path`: Required for catalog coverage/client
- `--table-name`: Target table (default `market_data`)
- `--state-path`: State JSON path (default `checkpoints/ingest_state.json`)
- `--client-mode`: `catalog|databento|noop` (default `catalog`)
- `--api-key`: Databento API key (for `client-mode=databento`, or use `DATABENTO_API_KEY`)
- `--dry-run`: Plan only (no ingestion/writes)

Examples:

```bash
# Plan gaps against SQL store, do not write
python -m ml.cli.ingest_backfill \
  --db postgresql://postgres:postgres@localhost:5433/nautilus \
  --dataset-id EQUS.MINI --schema bars \
  --instruments SPY.XNAS,QQQ.XNAS \
  --lookback-days 7 \
  --dry-run

# Use Parquet catalog for coverage and ingestion client, write to SQL canonical store
python -m ml.cli.ingest_backfill \
  --db postgresql://postgres:postgres@localhost:5433/nautilus \
  --dataset-id EQUS.MINI --schema bars \
  --instruments SPY.XNAS \
  --coverage-mode catalog --client-mode catalog \
  --catalog-path /abs/path/to/catalog \
  --lookback-days 14

# Use Databento client for ingestion (still writing to SQL)
python -m ml.cli.ingest_backfill \
  --db postgresql://postgres:postgres@localhost:5433/nautilus \
  --dataset-id EQUS.MINI --schema bars \
  --instruments SPY.XNAS \
  --client-mode databento --api-key "$DATABENTO_API_KEY" \
  --lookback-days 7
```

Notes:

- Canonical writes are SQL; registry events/watermarks reflect DB persistence.
- Catalog coverage/client is intended for historical workflows; for live backfills, use SQL coverage and a real Databento client.
