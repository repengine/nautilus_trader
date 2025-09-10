# Context: CLI Module

## Overview

The ML CLI module provides command-line interfaces for managing the Nautilus Trader ML pipeline. It includes comprehensive tools for data coverage reporting, backfill planning/execution, system health monitoring, and feature management operations. The CLI tools follow a consistent pattern using argparse for argument parsing and support both PostgreSQL and JSON backend configurations with automatic fallback strategies.

**Key CLI Commands:**

- **coverage**: Data coverage reporting and backfill management for the ML pipeline
- **health**: System health aggregation and monitoring
- **feature_backfill_cli**: Parallel feature computation and backfilling
- **feature_cli**: Feature registry management and lifecycle operations

## Architecture

### Module Structure

```
ml/cli/
├── coverage.py          # Comprehensive coverage reporting and backfill system (1,629 lines)
├── health.py           # System health monitoring CLI (43 lines)
├── feature_backfill_cli.py  # Parallel feature backfilling (111 lines)
└── feature_cli.py      # Feature registry management (87 lines)
```

### Command Invocation Pattern
All CLI tools follow the Python module execution pattern:

```bash
python -m ml.cli.coverage [command] [options]
python -m ml.cli.health [options]
python -m ml.cli.feature_backfill_cli [options]
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
```

### Environment Configuration

```bash
# Primary backend (PostgreSQL)
export NAUTILUS_REGISTRY_DB_URL="postgresql://user:pass@host:port/db"

# Databento integration (for backfill)
export DATABENTO_API_KEY="your_api_key_here"

# Catalog path (for data storage)
export NAUTILUS_CATALOG_PATH="./catalog"
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
