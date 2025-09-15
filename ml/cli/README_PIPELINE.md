# ML Pipeline Runner

Production-ready entry point for running the Nautilus Trader ML pipeline.

## Overview

The `run_ml_pipeline.py` script provides a unified interface for operating the ML data pipeline in different modes:

- **backfill**: Process historical data for a specific date range
- **daily**: Run scheduled daily updates for the previous trading day
- **realtime**: Start continuous real-time data processing (future enhancement)

## Installation

Ensure required dependencies are installed:

```bash
# Core requirements
pip install nautilus-trader polars click pyyaml

# For data collection (non dry-run mode)
pip install databento

# For database features
pip install psycopg2-binary
```

## Configuration

### Environment Variables

```bash
# Required for data collection
export DATABENTO_API_KEY="your_api_key_here"

# Optional: Database connection for feature storage
export DB_CONNECTION="postgresql://user:pass@host:5432/nautilus"
```

### Configuration Files

The script supports YAML and JSON configuration files. See examples in `ml/config/`:

- `pipeline_config.yaml` - Full production configuration
- `pipeline_config_example.json` - Simple example configuration

## Usage

### Daily Update Mode

Run daily data collection and feature computation:

```bash
# Basic usage
python ml/cli/run_ml_pipeline.py --mode daily

# With custom configuration
python ml/cli/run_ml_pipeline.py --mode daily --config ml/config/pipeline_config.yaml

# Dry run (no actual data collection)
python ml/cli/run_ml_pipeline.py --mode daily --dry-run

# Verbose logging
python ml/cli/run_ml_pipeline.py --mode daily --verbose
```

### Backfill Mode

Process historical data for a date range:

```bash
# Backfill January 2024
python ml/cli/run_ml_pipeline.py \
    --mode backfill \
    --start-date 2024-01-01 \
    --end-date 2024-01-31

# Backfill with custom config
python ml/cli/run_ml_pipeline.py \
    --mode backfill \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --config ml/config/pipeline_config.yaml

# Test backfill without actual execution
python ml/cli/run_ml_pipeline.py \
    --mode backfill \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --dry-run
```

### Real-time Mode

Start continuous processing (placeholder for future implementation):

```bash
# Start real-time processing
python ml/cli/run_ml_pipeline.py --mode realtime

# Test real-time setup
python ml/cli/run_ml_pipeline.py --mode realtime --dry-run
```

## Features

### Data Collection

- Fetches market data from Databento API
- Supports multiple data schemas (OHLCV, trades, quotes, L2 depth)
- Automatic retry logic with exponential backoff
- Configurable symbol universe (conservative, moderate, aggressive)

### Feature Engineering

- Technical indicators (SMA, RSI, Bollinger Bands, etc.)
- Statistical features (volatility, skewness, kurtosis)
- Microstructure features (optional, requires L2 data)
- Batch computation with efficient memory usage

### Data Management

- Automatic data retention policies
- Parquet storage for efficient querying
- Integration with Nautilus ParquetDataCatalog
- PostgreSQL feature storage for training/inference parity

### Monitoring & Safety

- Comprehensive logging with configurable levels
- Graceful shutdown handling (SIGINT, SIGTERM)
- Health checks before starting
- Dry-run mode for testing
- Environment validation

## Configuration Options

### Core Settings

- `catalog_path`: Location of Parquet data catalog
- `universe_mode`: Symbol selection strategy (conservative/moderate/aggressive)
- `symbols`: Custom symbol list (overrides universe_mode)

### Databento Settings

- `databento_dataset`: Data source (e.g., "GLBX.MDP3")
- `databento_schema`: Data type (e.g., "ohlcv-1m", "trades", "mbp-1")
- `use_temp_files`: Whether to use temporary DBN files
- `price_precision`: Decimal precision for prices

### Feature Settings

- `enable_features`: Whether to compute features
- `enable_technical_features`: Technical indicators
- `enable_microstructure_features`: L2-based features
- `lookback_periods`: Window sizes for features

### Operational Settings

- `retention_days`: How long to keep historical data
- `collection_time`: Daily collection schedule
- `max_retries`: Retry attempts for failed operations
- `stop_on_error`: Whether to halt on errors

## Production Deployment

### Systemd Service

Create a systemd service for automated daily runs:

```ini
# /etc/systemd/system/nautilus-ml-pipeline.service
[Unit]
Description=Nautilus ML Pipeline Daily Update
After=network.target postgresql.service

[Service]
Type=oneshot
User=nautilus
Group=nautilus
WorkingDirectory=/path/to/nautilus_trader
Environment="DATABENTO_API_KEY=your_key"
Environment="DB_CONNECTION=postgresql://..."
ExecStart=/usr/bin/python3 /path/to/ml/cli/run_ml_pipeline.py --mode daily --config /path/to/config.yaml
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Cron Job

Alternative scheduling with cron:

```bash
# Run daily at 4 AM UTC
0 4 * * * cd /path/to/nautilus_trader && python3 ml/cli/run_ml_pipeline.py --mode daily --config config.yaml >> /var/log/ml_pipeline.log 2>&1
```

### Docker Deployment

Run in containerized environment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV DATABENTO_API_KEY=${DATABENTO_API_KEY}
ENV DB_CONNECTION=${DB_CONNECTION}

# Run pipeline
CMD ["python", "ml/cli/run_ml_pipeline.py", "--mode", "daily"]
```

## Monitoring

### Logs

- Console output with timestamps and levels
- Configurable log files
- Structured logging for parsing

### Metrics

- Feature computation statistics
- Data collection success rates
- Processing latencies
- Error counts

### Health Checks

- Catalog accessibility
- Database connectivity
- API key validation
- Dependency verification

## Troubleshooting

### Common Issues

1. **Missing API Key**

   ```
   ERROR: DATABENTO_API_KEY environment variable is required
   ```

   Solution: Set the environment variable with your Databento API key

2. **Database Connection Failed**

   ```
   WARNING: Database connection check failed
   ```

   Solution: Verify PostgreSQL is running and connection string is correct

3. **Import Errors**

   ```
   ERROR: databento library is required for data collection
   ```

   Solution: Install missing dependencies with pip

4. **Permission Denied**

   ```
   ERROR: Permission denied: './data'
   ```

   Solution: Ensure write permissions for catalog directory

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python ml/cli/run_ml_pipeline.py --mode daily --verbose --dry-run
```

## Best Practices

1. **Start with Dry Run**: Always test configuration with `--dry-run` first
2. **Monitor First Run**: Watch logs closely during initial deployment
3. **Incremental Backfill**: Process historical data in small chunks
4. **Resource Limits**: Set appropriate memory and CPU limits
5. **Error Handling**: Configure alerts for pipeline failures
6. **Data Validation**: Regularly verify data integrity
7. **Version Control**: Track configuration changes in git

## Support

For issues or questions:

1. Check this documentation
2. Review logs for error details
3. Consult ml/docs/context/context_data.md
4. Open an issue with reproduction steps
