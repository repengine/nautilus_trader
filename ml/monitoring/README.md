# ML Monitoring Stack

Note: Choose Your Monitoring

- Canonical (recommended): Use the monitoring services bundled in `ml/deployment/docker-compose.yml` (Prometheus and Grafana run alongside the ML services on the `nautilus-ml` network). Start with:
  - `docker compose -f ml/deployment/docker-compose.yml up -d prometheus grafana`
- Optional standalone: This `ml/monitoring` directory provides a self-contained monitoring stack for development or separate deployments. Use it only if you intentionally want monitoring isolated from the main stack.

This directory contains the complete monitoring infrastructure for Nautilus Trader's ML components using Prometheus, Grafana, and AlertManager.

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose installed
- Port availability: 3000 (Grafana), 9090 (Prometheus), 9093 (AlertManager)
- ML application running with metrics exposed on port 8000

### 2. Configuration

Copy and customize the environment file:

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Start the Stack

```bash
# Start all monitoring services (standalone stack)
docker compose up -d

# Check service health
docker compose ps

# View logs
docker compose logs -f
```

### 4. Access Services

- **Grafana**: <http://localhost:3000> (admin/nautilus123)
- **Prometheus**: <http://localhost:9090>
- **AlertManager**: <http://localhost:9093>
- **Pushgateway**: <http://localhost:9091>

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  ML Application │────▶│  Prometheus  │────▶│   Grafana    │
│   (Port 8000)   │     │  (Port 9090) │     │ (Port 3000)  │
└─────────────────┘     └──────────────┘     └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ AlertManager │
                        │ (Port 9093)  │
                        └──────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              ┌──────────┐         ┌──────────┐
              │  Slack   │         │  Email   │
              └──────────┘         └──────────┘
```

## Dashboards

### Available Dashboards

1. **ML Overview** - System health and key metrics
2. **Data Quality** - Data pipeline monitoring
3. **Feature Engineering** - Feature computation metrics
4. **Model Lifecycle** - Training and deployment tracking
5. **Performance Degradation** - Model accuracy monitoring
6. **Resource Utilization** - System resource usage

### Dashboard Navigation

All dashboards are organized in the "ML Monitoring" folder and include:

- Template variables for filtering by model/instrument
- Time range selector (default: last 6 hours)
- Auto-refresh (10 seconds for critical metrics)
- Drill-down links between related dashboards

## Metrics Collected

### Core Metrics

- `ml_predictions_total` - Total predictions counter
- `ml_prediction_latency_seconds` - Inference latency histogram
- `ml_model_accuracy_rolling` - Rolling model accuracy
- `ml_feature_drift_score` - Feature drift detection

### Resource Metrics

- `ml_memory_usage_percent` - Memory utilization
- `ml_cpu_usage_percent` - CPU utilization
- `ml_gpu_utilization_percent` - GPU utilization (if available)

### Data Quality Metrics

- `ml_data_missing_values_ratio` - Missing data ratio
- `ml_data_outliers_detected_total` - Outlier detection
- `ml_data_staleness_seconds` - Data freshness

## Alert Configuration

### Alert Severity Levels

- **Critical** (P0): Immediate action required
  - Model accuracy < 60%
  - Inference latency > 5s
  - Memory usage > 95%

- **Warning** (P1): Investigation needed
  - Feature drift detected
  - Cache efficiency < 70%
  - Model accuracy 60-75%

- **Info** (P2): Monitoring only
  - Model retraining recommended
  - Resource trends
  - Optimization opportunities

### Alert Channels

Configure alert destinations in `.env`:

```bash
ALERTMANAGER_SLACK_WEBHOOK_URL=https://hooks.slack.com/...
ALERTMANAGER_EMAIL_TO=team@example.com
```

## Maintenance

### Backup

```bash
# Backup Grafana dashboards
docker exec nautilus_grafana grafana-cli admin export-dashboard-json > dashboards_backup.json

# Backup Prometheus data
docker exec nautilus_prometheus promtool tsdb snapshot /prometheus
```

### Update Services

```bash
# Pull latest images
docker compose pull

# Restart with new images
docker compose down && docker compose up -d
```

### Data Retention

Default retention policies:

- Prometheus: 30 days
- Grafana: Unlimited (database backed)
- AlertManager: 120 hours

Adjust in `prometheus/prometheus.yml`:

```yaml
global:
  storage.tsdb.retention.time: 30d
  storage.tsdb.retention.size: 10GB
```

## Troubleshooting

### Common Issues

1. **Grafana can't connect to Prometheus**
   - Check network connectivity: `docker network ls`
   - Verify Prometheus is running: `curl http://localhost:9090/-/healthy`

2. **No metrics appearing**
   - Ensure ML application is exposing metrics on port 8000
   - Check Prometheus targets: <http://localhost:9090/targets>

3. **Alerts not firing**
   - Verify AlertManager configuration
   - Check alert rules: <http://localhost:9090/alerts>
   - Review AlertManager UI: <http://localhost:9093>

### Debug Commands

```bash
# Check service logs
docker compose logs prometheus
docker compose logs grafana
docker compose logs alertmanager

# Test Prometheus queries
curl -G http://localhost:9090/api/v1/query --data-urlencode 'query=up'

# Validate configuration
docker exec nautilus_prometheus promtool check config /etc/prometheus/prometheus.yml
```

## Development

### Adding New Dashboards

1. Create dashboard in Grafana UI
2. Export as JSON: Settings → JSON Model
3. Save to `grafana/dashboards/`
4. Restart Grafana to auto-provision

### Adding New Alerts

1. Create alert rule in `prometheus/alerts/`
2. Follow naming convention: `ml_<severity>.yml`
3. Reload Prometheus config:

   ```bash
   curl -X POST http://localhost:9090/-/reload
   ```

### Testing Metrics

Use the included Python script to generate test metrics:

```python
from prometheus_client import start_http_server, Counter, Histogram
import time

# Start metrics server
start_http_server(8000)

# Create metrics
predictions = Counter('ml_predictions_total', 'Total predictions')
latency = Histogram('ml_prediction_latency_seconds', 'Latency')

# Generate test data
while True:
    predictions.inc()
    latency.observe(0.1)
    time.sleep(1)
```

## Performance Tuning

### Grafana Optimization

- Limit panels per dashboard to 20-25
- Use recording rules for complex queries
- Enable query caching in datasource settings

### Prometheus Optimization

- Adjust scrape intervals based on metric importance
- Use metric relabeling to reduce cardinality
- Enable compression: `--storage.tsdb.wal-compression`

### Resource Requirements

- Minimum: 4 CPU, 8GB RAM
- Recommended: 8 CPU, 16GB RAM
- Storage: 100GB SSD for 30-day retention

## Security

### Production Deployment

1. **Change default passwords**:

   ```bash
   GRAFANA_ADMIN_PASSWORD=<strong-password>
   ```

2. **Enable TLS**:
   - Add reverse proxy (nginx/traefik)
   - Configure SSL certificates

3. **Restrict access**:
   - Implement IP whitelisting
   - Enable authentication on all services

4. **Audit logging**:
   - Enable Grafana audit logs
   - Monitor access patterns

## Support

For issues or questions:

1. Check the [troubleshooting guide](#troubleshooting)
2. Review logs: `docker compose logs`
3. Open an issue with relevant logs and configuration
