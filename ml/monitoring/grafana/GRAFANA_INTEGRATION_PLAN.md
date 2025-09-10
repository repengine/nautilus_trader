# Grafana Dashboard Integration Plan for ML Monitoring (NEEDS UPDATING)

## Executive Summary

This document outlines the comprehensive plan for integrating Grafana dashboards with the Nautilus Trader ML monitoring system. The integration will provide real-time visualization of 8 specialized Prometheus collectors, enabling proactive ML system observability with minimal performance impact.

## 1. Dashboard Architecture

### 1.1 Dashboard Hierarchy

```
ML Monitoring Overview (Home)
├── Data Pipeline
│   ├── Data Quality Dashboard
│   ├── Data Loading Performance
│   └── Cache Effectiveness
├── Feature Engineering
│   ├── Feature Computation Metrics
│   ├── Feature Drift Analysis
│   └── Feature Cache Performance
├── Model Lifecycle
│   ├── Model Registry
│   ├── Training & Deployment
│   └── Model Version Tracking
├── Performance Monitoring
│   ├── Inference Latency
│   ├── Accuracy Degradation
│   └── Prediction Distribution
├── Resource Utilization
│   ├── System Resources
│   ├── GPU Monitoring
│   └── Memory Management
└── Alerts & Health
    ├── Alert Overview
    ├── SLA Compliance
    └── System Health Score

```

### 1.2 Dashboard Organization Strategy

- **Folder Structure**: Organized by functional domain
- **Navigation**: Breadcrumb navigation with drill-down capability
- **Template Variables**: Shared across dashboards for consistency
- **Time Range**: Unified time picker across all dashboards
- **Refresh Rate**: Configurable per dashboard (5s for critical, 30s for standard)

## 2. Key Metrics Per Dashboard

### 2.1 Data Quality Dashboard

**Primary Panels:**

- Missing Values Ratio (Gauge) - Alert: >5%
- Outliers Detected (Time Series) - Alert: >100/min
- Data Staleness (Stat Panel) - Alert: >300s
- Validation Failures (Counter) - Alert: >10/min
- Data Load Latency P50/P95/P99 (Histogram)
- Cache Hit Ratio (Gauge) - Target: >80%

**Layout:**

```
Row 1: Key Stats (Missing%, Outliers, Staleness, Cache Hit%)
Row 2: Time Series (Load Latency, Validation Failures)
Row 3: Distribution (Data Volume, Error Types)
Row 4: Detailed Tables (Recent Failures, Data Sources)
```

### 2.2 Feature Engineering Dashboard

**Primary Panels:**

- Feature Computation Latency (Histogram) - Alert: P99 >500ms
- Feature Drift Score (Heatmap) - Alert: >0.3
- Feature Importance (Bar Chart)
- Cache Performance (Gauge) - Target: >90%
- Computation Errors (Time Series) - Alert: >5/min
- Feature Null Ratio (Table)

**Layout:**

```
Row 1: Performance Stats (Latency P50/P95/P99, Cache Hit%)
Row 2: Feature Health (Drift Scores, Null Ratios)
Row 3: Feature Importance & Usage
Row 4: Error Analysis & Alerts
```

### 2.3 Model Lifecycle Dashboard

**Primary Panels:**

- Active Model Version (Stat)
- Training Duration (Time Series)
- Model Size (Gauge)
- Load Time (Histogram) - Alert: >5s
- Deployment Count (Counter)
- Training/Validation Scores (Line Chart)

**Layout:**

```
Row 1: Current Model Info (Version, Size, Last Trained)
Row 2: Training Metrics (Duration, Scores)
Row 3: Deployment History
Row 4: Model Registry Table
```

### 2.4 Performance Degradation Dashboard

**Primary Panels:**

- Rolling Accuracy (Time Series) - Alert: <0.7
- Prediction Distribution Shift (Heatmap) - Alert: KS >0.2
- Inference Timeout Ratio (Gauge) - Alert: >1%
- Confidence Percentiles (Line Chart)
- Retraining Required (Binary Indicator)
- Performance Alerts (Table)

**Layout:**

```
Row 1: Critical Indicators (Accuracy, Drift, Timeouts)
Row 2: Performance Trends (7d, 30d rolling)
Row 3: Distribution Analysis
Row 4: Alert History & Actions
```

### 2.5 Resource Utilization Dashboard

**Primary Panels:**

- CPU Usage (Time Series) - Alert: >80%
- Memory Usage (Gauge) - Alert: >90%
- GPU Utilization (Time Series) - Alert: >95%
- Disk I/O (Counter)
- Inference Batch Size (Histogram)
- Feature Store Size (Stat)

**Layout:**

```
Row 1: System Overview (CPU, Memory, GPU, Disk)
Row 2: ML-Specific Resources (Model Memory, Feature Store)
Row 3: Throughput Metrics (Batch Size, Rows Processed)
Row 4: Resource Trends & Forecasting
```

## 3. Alert Rules and Thresholds

### 3.1 Critical Alerts (P0 - Immediate Action)

```yaml
groups:
  - name: ml_critical
    interval: 10s
    rules:
      - alert: ModelAccuracyDegraded
        expr: ml_model_accuracy_rolling < 0.6
        for: 5m
        labels:
          severity: critical
          team: ml-ops
        annotations:
          summary: "Model accuracy critically low: {{ $value }}"

      - alert: InferenceLatencyHigh
        expr: ml_inference_latency_p99 > 5000
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "P99 latency exceeds 5s: {{ $value }}ms"

      - alert: SystemMemoryExhausted
        expr: ml_memory_usage_percent > 95
        for: 1m
        labels:
          severity: critical
```

### 3.2 Warning Alerts (P1 - Investigate Soon)

```yaml
      - alert: DataDriftDetected
        expr: ml_feature_drift_score > 0.3
        for: 10m
        labels:
          severity: warning

      - alert: CacheEfficiencyLow
        expr: ml_cache_hit_ratio < 0.7
        for: 15m
        labels:
          severity: warning

      - alert: ValidationFailuresIncreasing
        expr: rate(ml_validation_failures_total[5m]) > 10
        labels:
          severity: warning
```

### 3.3 Info Alerts (P2 - Monitor)

```yaml
      - alert: ModelRetrainingRecommended
        expr: ml_model_retraining_required == 1
        for: 30m
        labels:
          severity: info

      - alert: FeatureComputationSlow
        expr: ml_feature_computation_latency_p95 > 200
        for: 20m
        labels:
          severity: info
```

## 4. Docker Compose Configuration

### 4.1 Service Definitions

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: nautilus_prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=10GB'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/alerts/:/etc/prometheus/alerts/:ro
      - prometheus_data:/prometheus
    networks:
      - monitoring
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  grafana:
    image: grafana/grafana:10.2.3
    container_name: nautilus_grafana
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_ADMIN_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-nautilus123}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
      - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/ml-overview.json
      - GF_SERVER_ROOT_URL=http://localhost:3000
      - GF_ANALYTICS_REPORTING_ENABLED=false
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana_data:/var/lib/grafana
    networks:
      - monitoring
    depends_on:
      - prometheus
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:3000/api/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  alertmanager:
    image: prom/alertmanager:v0.27.0
    container_name: nautilus_alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--log.level=info'
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    networks:
      - monitoring
    restart: unless-stopped

  pushgateway:
    image: prom/pushgateway:v1.7.0
    container_name: nautilus_pushgateway
    ports:
      - "9091:9091"
    networks:
      - monitoring
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9091/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  node-exporter:
    image: prom/node-exporter:v1.7.0
    container_name: nautilus_node_exporter
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    networks:
      - monitoring
    restart: unless-stopped

  # Optional: GPU monitoring with DCGM exporter
  dcgm-exporter:
    image: nvidia/dcgm-exporter:3.3.0-3.2.0-ubuntu22.04
    container_name: nautilus_dcgm_exporter
    ports:
      - "9400:9400"
    networks:
      - monitoring
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: unless-stopped
    profiles:
      - gpu

volumes:
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  alertmanager_data:
    driver: local

networks:
  monitoring:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 5. Configuration File Structure

### 5.1 Directory Layout

```
ml/monitoring/
├── docker-compose.yml
├── .env.example
├── prometheus/
│   ├── prometheus.yml
│   └── alerts/
│       ├── ml_critical.yml
│       ├── ml_warning.yml
│       └── ml_info.yml
├── grafana/
│   ├── provisioning/
│   │   ├── dashboards/
│   │   │   └── default.yml
│   │   ├── datasources/
│   │   │   └── prometheus.yml
│   │   └── alerting/
│   │       └── notification_channels.yml
│   └── dashboards/
│       ├── ml-overview.json
│       ├── data-quality.json
│       ├── feature-engineering.json
│       ├── model-lifecycle.json
│       ├── performance-degradation.json
│       └── resource-utilization.json
└── alertmanager/
    └── alertmanager.yml
```

### 5.2 Grafana Provisioning Configuration

**datasources/prometheus.yml:**

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
    jsonData:
      timeInterval: "5s"
      queryTimeout: "30s"
      httpMethod: POST
```

**dashboards/default.yml:**

```yaml
apiVersion: 1

providers:
  - name: 'ML Monitoring'
    orgId: 1
    folder: 'ML Monitoring'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: true
```

### 5.3 Dashboard Templates and Variables

**Common Template Variables:**

```json
{
  "templating": {
    "list": [
      {
        "name": "datasource",
        "type": "datasource",
        "query": "prometheus",
        "current": {
          "text": "Prometheus",
          "value": "Prometheus"
        }
      },
      {
        "name": "model",
        "type": "query",
        "query": "label_values(ml_predictions_total, model)",
        "refresh": 2,
        "multi": true,
        "includeAll": true
      },
      {
        "name": "instrument",
        "type": "query",
        "query": "label_values(ml_predictions_total, instrument)",
        "refresh": 2,
        "multi": true,
        "includeAll": true
      },
      {
        "name": "interval",
        "type": "interval",
        "options": [
          {"text": "1m", "value": "1m"},
          {"text": "5m", "value": "5m"},
          {"text": "15m", "value": "15m"},
          {"text": "1h", "value": "1h"}
        ],
        "current": {
          "text": "5m",
          "value": "5m"
        }
      }
    ]
  }
}
```

## 6. Implementation Approach

### Phase 1: Infrastructure Setup (Week 1)

1. Create Docker compose configuration
2. Set up Prometheus with retention policies
3. Configure Grafana with provisioning
4. Set up Alertmanager with routing rules
5. Test basic connectivity

### Phase 2: Dashboard Development (Week 2-3)

1. Create ML Overview dashboard
2. Implement Data Quality dashboard
3. Build Feature Engineering dashboard
4. Develop Model Lifecycle dashboard
5. Create Performance Degradation dashboard
6. Build Resource Utilization dashboard

### Phase 3: Alert Configuration (Week 4)

1. Define alert rules in Prometheus
2. Configure Alertmanager routing
3. Set up notification channels (Slack, Email, PagerDuty)
4. Test alert escalation paths
5. Document runbooks for each alert

### Phase 4: Integration & Testing (Week 5)

1. Integrate with MLDataLoader
2. Connect to FeatureEngineer
3. Test with synthetic load
4. Validate metric accuracy
5. Performance testing

### Phase 5: Documentation & Training (Week 6)

1. Create user documentation
2. Build troubleshooting guides
3. Develop training materials
4. Conduct team training
5. Create maintenance procedures

## 7. Performance Considerations

### 7.1 Data Retention Strategy

- **High-frequency metrics** (1s interval): 7 days
- **Standard metrics** (15s interval): 30 days
- **Aggregated metrics** (5m interval): 90 days
- **Long-term storage**: Downsample to 1h, keep 1 year

### 7.2 Query Optimization

- Use recording rules for frequently accessed queries
- Implement metric aggregation at collection time
- Limit cardinality by controlling label combinations
- Use `increase()` and `rate()` functions efficiently

### 7.3 Dashboard Performance

- Limit panels per dashboard to 20-25
- Use appropriate visualization types
- Implement query caching
- Optimize refresh intervals based on metric importance
- Use streaming for real-time data

### 7.4 Resource Requirements

- **Prometheus**: 4 CPU, 8GB RAM, 100GB SSD
- **Grafana**: 2 CPU, 4GB RAM, 20GB SSD
- **Alertmanager**: 1 CPU, 1GB RAM, 10GB SSD
- **Total**: ~7 CPU, 13GB RAM, 130GB storage

## 8. Security Considerations

### 8.1 Authentication & Authorization

- Enable Grafana authentication (LDAP/OAuth)
- Implement role-based access control
- Secure Prometheus endpoints
- Use TLS for all communications

### 8.2 Data Privacy

- Sanitize sensitive data in metrics
- Implement data masking where needed
- Control metric label cardinality
- Regular audit of exposed metrics

## 9. Maintenance & Operations

### 9.1 Backup Strategy

- Daily Grafana database backup
- Weekly Prometheus TSDB snapshot
- Configuration version control in Git
- Automated restore testing

### 9.2 Monitoring the Monitoring

- Prometheus self-monitoring
- Grafana health checks
- Alert on monitoring system failures
- Dead man's switch for critical alerts

## 10. Success Metrics

### 10.1 KPIs

- Dashboard load time < 2 seconds
- Alert detection latency < 30 seconds
- False positive rate < 5%
- System uptime > 99.9%
- Mean time to detection < 5 minutes

### 10.2 User Adoption

- Active dashboard users > 80% of team
- Custom dashboard creation by users
- Alert response time < 15 minutes
- Positive feedback score > 4/5

## Appendix A: Sample Dashboard JSON

```json
{
  "dashboard": {
    "title": "ML Model Performance Overview",
    "uid": "ml-overview",
    "version": 1,
    "timezone": "browser",
    "refresh": "10s",
    "panels": [
      {
        "id": 1,
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "type": "graph",
        "title": "Model Accuracy (7d Rolling)",
        "targets": [
          {
            "expr": "ml_model_accuracy_rolling{model=~\"$model\"}",
            "legendFormat": "{{model}}"
          }
        ],
        "alert": {
          "conditions": [
            {
              "evaluator": {
                "params": [0.7],
                "type": "lt"
              },
              "query": {
                "params": ["A", "5m", "now"]
              },
              "type": "query"
            }
          ]
        }
      }
    ]
  }
}
```

## Appendix B: PromQL Query Examples

```promql
# Average inference latency by model
avg by (model) (
  rate(ml_prediction_latency_seconds_sum[5m]) /
  rate(ml_prediction_latency_seconds_count[5m])
)

# Data drift detection
max by (feature) (
  ml_feature_drift_score{feature=~".*"}
) > 0.3

# Resource utilization efficiency
(
  rate(ml_predictions_total[5m]) /
  avg(ml_cpu_usage_percent)
) * 100

# Cache effectiveness
ml_cache_hit_ratio /
(1 + rate(ml_cache_misses_total[5m]))
```

## Next Steps

1. Review and approve the integration plan
2. Set up development environment
3. Begin Phase 1 implementation
4. Schedule weekly progress reviews
5. Prepare for production deployment

---

This plan provides a comprehensive approach to integrating Grafana dashboards with the ML monitoring system, ensuring robust observability while maintaining performance and scalability.
