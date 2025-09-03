# Context: Scripts Module

## Overview

The `ml/scripts/` module provides a comprehensive collection of operational scripts for data collection, processing, model training, and system maintenance within the Nautilus Trader ML pipeline. These scripts serve as the automation backbone for production data workflows, including historical backfills, daily updates, health monitoring, and dataset preparation for TFT models.

The scripts are designed to be robust, production-ready tools that handle large-scale data operations, external API integrations (Databento, Yahoo Finance, FRED), and complex data transformations while maintaining the ML system's architectural standards.

## Architecture

The scripts follow a layered architecture with clear separation of concerns:

```
ml/scripts/
├── Data Collection & Population
│   ├── populate_universe.py          # Unified L0/L1/L2/L3 data collection
│   ├── populate_l2_efficient.py      # Efficient L2 market depth collection
│   ├── populate_yahoo_data.py        # Yahoo Finance supplementary data
│   ├── populate_supplementary_simple.py # Alternative data sources
│   ├── populate_alternative_data.py  # CBOE, AAII, COT, sentiment data
│   └── fred_integration_bridge.py    # FRED economic indicators
├── Pipeline Management
│   ├── run_ml_pipeline.py            # Production pipeline orchestrator
│   ├── build_production_dataset.py   # Large-scale dataset construction
│   └── train_tft_quick.py           # Quick TFT model training
├── System Operations
│   ├── check_pipeline_health.py      # Health monitoring & diagnostics
│   ├── check_databento_subscription.py # API limits verification
│   └── sanity_check.py              # Codebase quality checks
└── README_PIPELINE.md               # Operational documentation
```

### Design Patterns

1. **Command-Line Interface**: All scripts use `argparse` or `click` for consistent CLI patterns
2. **Configuration-Driven**: External configuration files and environment variables control behavior
3. **Progress Tracking**: JSON-based progress files enable resume capability
4. **Error Resilience**: Retry logic, graceful degradation, and comprehensive error handling
5. **Resource Management**: Memory-efficient streaming, batching, and temporary file management

## Key Components

### Data Collection Scripts

#### `populate_universe.py` - Unified Data Collection
- **Purpose**: Single interface for collecting multi-level market data across symbol universes
- **Data Levels**: L0 (OHLCV), L1 (quotes/trades), L2/L3 (market depth)
- **Features**: Cost estimation, progress tracking, tier-based symbol management, parallel downloads
- **Integration**: Databento Historical API with EQUS.MINI subscription support

#### `populate_l2_efficient.py` - Efficient L2 Collection  
- **Purpose**: Optimized collection of Level 2 market depth data with gap detection
- **Features**: Day-by-day processing, data integrity validation, streaming merge operations
- **Optimization**: Memory-efficient parquet streaming, corrupted file detection, incremental updates
- **Scale**: Handles 30-day rolling windows for Tier 1 symbols (~70 symbols)

#### `populate_yahoo_data.py` - Supplementary Market Data
- **Purpose**: Collect regime detection data from Yahoo Finance (sectors, factors, commodities)
- **Data Categories**: 11 categories including sector ETFs, international markets, currencies, volatility products
- **Features**: Correlation calculation, spread computation, technical indicators
- **Output**: Parquet files with historical OHLCV + derived features

#### `fred_integration_bridge.py` - Economic Indicators Bridge
- **Purpose**: Convert FRED economic data to ML pipeline format
- **Integration**: Bridges external FRED updater with ML data stores
- **Features**: Format conversion (wide to long), incremental updates, automatic scheduling
- **Indicators**: VIX, treasury rates, credit spreads, dollar index, mortgage rates

### Pipeline Management Scripts

#### `run_ml_pipeline.py` - Production Pipeline Orchestrator
- **Purpose**: Main production entry point for ML pipeline operations  
- **Modes**: Backfill (historical), daily (scheduled updates), realtime (continuous processing)
- **Features**: Configuration management, health checks, graceful shutdown, dry-run capability
- **Integration**: DataScheduler, FeatureEngineer, ParquetDataCatalog, database connectivity

#### `build_production_dataset.py` - Large-Scale Dataset Construction
- **Purpose**: Orchestrate complete pipeline for production-grade datasets (20M+ samples)
- **Phases**: Historical L0/L1 → Microstructure L2/L3 → Cross-sectional features → Regime indicators → TFT dataset
- **Scale**: 7 years historical × 30 symbols + microstructure features
- **Output**: TFT-ready training dataset with comprehensive feature engineering

#### `train_tft_quick.py` - Quick Model Training
- **Purpose**: Fast-path TFT model training on collected data
- **Features**: Automatic data discovery, TFT dataset building, model training with validation
- **Output**: Trained teacher models, prediction validation, dataset statistics
- **Integration**: TFTDatasetBuilder, ParquetDataCatalog

### System Operations Scripts

#### `check_pipeline_health.py` - Health Monitoring
- **Purpose**: Comprehensive ML pipeline health diagnostics
- **Components**: Pipeline status, data collection, feature computation, freshness, errors, model performance
- **Output**: Human-readable reports, JSON for dashboards, critical-only filtering
- **Integration**: PostgreSQL monitoring views, tabulate formatting, Prometheus metrics

#### `check_databento_subscription.py` - API Limits Verification  
- **Purpose**: Verify Databento subscription limits before data collection
- **Checks**: Available datasets, date ranges, schemas (L0/L1/L2/L3), cost estimation
- **Safety**: Generate safe configurations that stay within subscription limits
- **Output**: Dataset availability report, cost estimates, configuration recommendations

#### `sanity_check.py` - Codebase Quality Checks
- **Purpose**: Fast, advisory quality sweep of ML codebase
- **Checks**: Ruff linting, MyPy strict typing, legacy schema references, SQL injection patterns, architecture violations
- **Tools**: Optional integration with pip-audit, bandit, vulture, deptry
- **Output**: Concise advisory report with zero exit code

## Dependencies

### Internal Dependencies
- **ML Core**: `ml._imports` for centralized ML library management
- **ML Config**: `ml.config.scheduler_config` for pipeline configuration
- **ML Data**: `ml.data.collectors`, `ml.data.scheduler`, `ml.data.tft_dataset_builder`
- **ML Features**: `ml.features.engineering` for feature computation
- **ML Stores**: `ml.stores.data_store` for persistence
- **Nautilus Core**: `nautilus_trader.persistence.catalog.parquet`

### External Dependencies
- **Data Sources**: `databento`, `yfinance`, direct HTTP for FRED/Alpha Vantage
- **Data Processing**: `polars`, `pandas`, `numpy` for high-performance data manipulation
- **Database**: `psycopg2` for PostgreSQL connectivity and health monitoring
- **CLI/Config**: `click`, `argparse`, `yaml`, `json` for user interfaces
- **System**: `pathlib`, `os`, `signal`, `subprocess` for system operations

### Optional Dependencies
- **Security**: `bandit`, `pip-audit` for security scanning
- **Code Quality**: `vulture`, `deptry` for dead code and dependency analysis
- **Formatting**: `tabulate` for report formatting (fallback implementation included)

## Usage Patterns

### Production Data Collection Workflow
```bash
# 1. Verify subscription limits
python ml/scripts/check_databento_subscription.py

# 2. Estimate and populate historical data
python ml/scripts/populate_universe.py --estimate-only
python ml/scripts/populate_universe.py --level L0 --tier 1

# 3. Efficient L2 collection with gap filling
python ml/scripts/populate_l2_efficient.py --tier 1 --days 30 --check-gaps

# 4. Supplement with external data
python ml/scripts/populate_yahoo_data.py --all
python ml/scripts/fred_integration_bridge.py
```

### Production Pipeline Operations
```bash
# Daily scheduled update
python ml/scripts/run_ml_pipeline.py --mode daily --config config.yaml

# Historical backfill  
python ml/scripts/run_ml_pipeline.py --mode backfill --start-date 2024-01-01 --end-date 2024-01-31

# Health monitoring
python ml/scripts/check_pipeline_health.py --json --export report.json
```

### Model Training Pipeline
```bash
# Build large-scale dataset
python ml/scripts/build_production_dataset.py --full

# Quick model training
python ml/scripts/train_tft_quick.py

# System quality checks
python ml/scripts/sanity_check.py
```

## Integration Points

### Data Flow Integration
- **Input Sources**: Databento Historical API, Yahoo Finance, FRED API, Alpha Vantage
- **Storage Integration**: ParquetDataCatalog, PostgreSQL feature stores, local parquet files
- **Processing Integration**: Polars/Pandas for data manipulation, streaming for large datasets
- **Output Integration**: TFT-ready datasets, model registries, feature stores

### ML Pipeline Integration
- **DataScheduler**: Production pipeline orchestration and scheduling
- **FeatureEngineer**: Feature computation and storage coordination
- **TFTDatasetBuilder**: Training dataset construction for TFT models
- **DataCollector**: Market data collection and validation

### System Integration
- **Configuration Management**: YAML/JSON config files, environment variables
- **Health Monitoring**: PostgreSQL monitoring views, Prometheus metrics
- **Process Management**: Signal handling, graceful shutdown, progress tracking
- **Deployment**: Systemd services, Docker containers, cron scheduling

## Implementation Notes

### Memory Management
- **Streaming Operations**: Large file operations use streaming to minimize memory usage
- **Batch Processing**: Data processed in configurable batches to control memory consumption
- **Temporary Files**: Efficient use of temporary files for intermediate processing steps
- **Garbage Collection**: Explicit cleanup of large DataFrames and file handles

### Error Handling & Resilience
- **Retry Logic**: Exponential backoff for transient API failures
- **Progress Persistence**: JSON progress files enable resume from interruption
- **Data Validation**: Integrity checks for corrupted files and incomplete data
- **Graceful Degradation**: Fallback strategies when optional dependencies unavailable

### Performance Optimizations
- **Parallel Processing**: Configurable concurrency for symbol-level operations
- **Efficient Formats**: Parquet with compression for optimal storage and query performance
- **Index Usage**: Proper indexing strategies for time-series data
- **Memory Mapping**: Use of memory-mapped files for large dataset operations

### Security & Safety
- **API Key Management**: Environment variable-based API key handling
- **SQL Injection Prevention**: Parameterized queries and f-string detection
- **Data Validation**: Input validation and sanitization for external data
- **Permission Checks**: File system permission validation before operations

### Configuration Management
- **Environment-Driven**: Extensive use of environment variables for deployment flexibility
- **Configuration Files**: Support for YAML and JSON configuration formats
- **Default Fallbacks**: Sensible defaults when configuration not provided
- **Validation**: Configuration validation with clear error messages

The scripts module represents the operational interface of the ML system, providing robust, production-ready tools for all aspects of data pipeline management while maintaining consistency with the broader Nautilus Trader architecture and ML-specific guidelines.