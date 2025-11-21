# ML Pipeline Health Monitoring System

## Overview

The ML Pipeline Health Monitoring System provides comprehensive observability for the Nautilus Trader ML infrastructure. It consists of SQL monitoring views, Python health check scripts, and Grafana dashboards for real-time visualization.

## Components

### 1. SQL Monitoring Views (migrations/008_views.sql)

Creates comprehensive database views for monitoring:

- **ml.pipeline_health** - Overall pipeline health status with daily aggregates
- **ml.data_collection_stats** - Data collection metrics per instrument and hour
- **ml.feature_computation_stats** - Feature computation performance and quality
- **ml.data_freshness** - Monitor data staleness per instrument
- **ml.error_summary** - Summary of errors across all ML components
- **ml.model_performance_summary** - Model inference performance metrics
- **ml.strategy_signal_summary** - Strategy signal generation metrics
- **ml.pipeline_processing_summary** - Overall pipeline statistics by stage
- **ml.data_quality_metrics** - Comprehensive data quality tracking

### 2. Health Check Script (`ml/scripts/check_pipeline_health.py`)

Python script that queries monitoring views and generates health reports:

**Features:**

- Comprehensive health checks across all pipeline components
- Human-readable and JSON output formats
- Configurable thresholds for warnings and critical alerts
- Exit codes for integration with monitoring systems (0=healthy, 1=warning, 2=critical)

**Usage:**

```bash
# Human-readable output
python ml/scripts/check_pipeline_health.py

# JSON output for dashboards
python ml/scripts/check_pipeline_health.py --json

# Show only critical issues
python ml/scripts/check_pipeline_health.py --critical-only

# Export to file
python ml/scripts/check_pipeline_health.py --export report.json

# Custom database connection
python ml/scripts/check_pipeline_health.py --connection-string "postgresql://user:pass@host:5432/db"
```

### 3. Grafana Dashboard (`ml/deployment/grafana/ml_pipeline_health.json`)

Comprehensive dashboard with panels for:

- Pipeline overview with health score gauge
- Data freshness indicators
- Active instruments tracking
- Component status table
- Data collection rate time series
- Feature computation latency tracking
- Data freshness heatmap
- Error rate monitoring
- Recent errors table

**Dashboard Features:**

- Auto-refresh every 10 seconds
- Template variables for interval, instrument, and model selection
- Color-coded thresholds for quick issue identification
- Historical trend analysis

### 4. Test Script (`ml/scripts/test_pipeline_health.py`)

Test utility for validating the health monitoring system:

```bash
# Set up test environment with sample data
python ml/scripts/test_pipeline_health.py --setup

# Run health check on test data
python ml/scripts/test_pipeline_health.py --check

# Clean up test data
python ml/scripts/test_pipeline_health.py --cleanup

# Run all steps
python ml/scripts/test_pipeline_health.py --all
```

## Health Check Thresholds

Configurable thresholds defined in `Thresholds` class:

| Metric | Warning | Critical |
|--------|---------|----------|
| Data Staleness | 1 hour | 24 hours |
| Error Rate | 5% | 10% |
| Model Confidence | < 0.6 | - |
| Inference Latency | 500ms | 1000ms |
| Feature Computation | 200ms | 500ms |
| Null Value Rate | 5% | 10% |

## Installation

### Prerequisites

```bash
# Required for health check script
pip install psycopg2-binary

# Optional for better table formatting
pip install tabulate
```

### Database Setup

1. Create monitoring views (canonical migrations):

```bash
psql -U postgres -d nautilus -f ml/stores/migrations/008_views.sql
```

2. Ensure ML tables exist (created by stores migration):

```bash
psql -U postgres -d nautilus -f ml/stores/migrations/002_stores_schema.sql
```

### Grafana Setup

1. Import dashboard:
   - Open Grafana UI (<http://localhost:3000>)
   - Navigate to Dashboards → Import
   - Upload `ml/deployment/grafana/ml_pipeline_health.json`
   - Select Prometheus datasource
   - Click Import

2. Configure alerts (optional):
   - Edit dashboard panels
   - Add alert rules based on thresholds
   - Configure notification channels

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Check ML Pipeline Health
  run: |
    python ml/scripts/check_pipeline_health.py --critical-only
  env:
    ML_DB_CONNECTION: ${{ secrets.DB_CONNECTION }}
```

### Docker Health Check

Add to `docker-compose.yml`:

```yaml
ml_pipeline:
  healthcheck:
    test: ["CMD", "python", "/app/ml/scripts/check_pipeline_health.py", "--critical-only"]
    interval: 60s
    timeout: 10s
    retries: 3
```

### Cron Job for Regular Monitoring

```bash
# Add to crontab for hourly checks
0 * * * * /usr/bin/python3 /path/to/ml/scripts/check_pipeline_health.py --json --export /var/log/ml_health.json
```

## Monitoring Workflow

1. **Continuous Monitoring**:
   - Grafana dashboard provides real-time visualization
   - Prometheus scrapes metrics every 15 seconds
   - Alerts fire based on configured thresholds

2. **Daily Health Checks**:
   - Automated script runs via cron
   - Results exported to monitoring system
   - Email/Slack notifications for issues

3. **Incident Response**:
   - Critical alerts trigger immediate notification
   - Health check script provides detailed diagnostics
   - SQL views enable deep-dive analysis

## Troubleshooting

### Common Issues

1. **No data in views**:
   - Ensure ML pipeline is running
   - Check that data is being written to ML tables
   - Verify timestamp conversions (nanoseconds)

2. **Connection errors**:
   - Verify PostgreSQL is running
   - Check connection string format
   - Ensure database user has SELECT permissions

3. **High staleness values**:
   - Check data collection scheduler
   - Verify Databento connection
   - Review pipeline logs for errors

4. **Performance issues**:
   - Ensure indexes exist on ML tables
   - Consider partitioning for large datasets
   - Optimize view queries if needed

## Performance Considerations

- Views use efficient indexes for time-range queries
- Partition pruning reduces query overhead
- Cached results for frequently accessed metrics
- Batch processing for historical analysis

## Future Enhancements

- [ ] Machine learning-based anomaly detection
- [ ] Predictive alerts for potential issues
- [ ] Automated remediation actions
- [ ] Integration with PagerDuty/OpsGenie
- [ ] Mobile app for on-the-go monitoring
- [ ] Historical trend analysis and reporting

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review logs in `ml/logs/`
3. Run test script to validate setup
4. Consult ML team documentation
