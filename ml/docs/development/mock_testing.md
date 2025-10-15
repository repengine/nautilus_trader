# Mock Testing Environment

## Overview

The ML system includes a comprehensive mock testing environment that allows you to test the entire pipeline without waiting for market hours. This system generates synthetic market data, processes it through the full ML pipeline, and persists all results to a test database.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Mock Data Generator                    │
│  - Synthetic OHLCV bars with Brownian motion            │
│  - Configurable rate (1-100 bars/second)                │
│  - Realistic price movements and volume patterns        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  ML Signal Actor                         │
│  - Feature computation (26 technical indicators)         │
│  - Model inference (ONNX runtime)                       │
│  - Signal generation with thresholds                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Stores & Registries (PostgreSQL)            │
│  - Feature Store: Computed features                     │
│  - Model Store: Predictions and metrics                 │
│  - Strategy Store: Signals and decisions                │
│  - Data Store: Unified facade with validation           │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Using Docker Compose (Recommended)

```bash
# Start the complete test environment
docker compose -f ml/deployment/docker-compose.test.yml up

# View logs in real-time
docker compose -f ml/deployment/docker-compose.test.yml logs -f ml_signal_actor_test

# Check metrics
curl http://localhost:8002/metrics | grep ml_

# Stop the test environment
docker compose -f ml/deployment/docker-compose.test.yml down
```

### 2. Direct Python Execution

```bash
# Test the mock data generator standalone
python ml/deployment/mock_databento.py

# Run with custom parameters
export USE_MOCK_DATA=true
export MOCK_DATA_RATE=10
export MOCK_INITIAL_PRICE=650
export MOCK_VOLATILITY=0.002
python ml/deployment/entrypoint_mock.py
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_DATA` | `false` | Enable mock data generation |
| `MOCK_DATA_RATE` | `1.0` | Bars per second to generate |
| `MOCK_INITIAL_PRICE` | `650.0` | Starting price for synthetic data |
| `MOCK_VOLATILITY` | `0.002` | Price volatility (0.2% per bar) |
| `USE_TEST_DATABASE` | `false` | Use separate test database |
| `TEST_DB_NAME` | `nautilus_test` | Name of test database |
| `LOG_FEATURES` | `false` | Log all computed features |
| `LOG_PREDICTIONS` | `false` | Log all model predictions |
| `LOG_SIGNALS` | `false` | Log all generated signals |

### Test Database Setup

The test environment uses a separate PostgreSQL database to avoid polluting production data:

```yaml
# docker-compose.test.yml
postgres-test:
  environment:
    POSTGRES_DB: nautilus_test  # Separate database
  ports:
    - "5434:5432"  # Different port to avoid conflicts
```

## Mock Data Characteristics

The synthetic data generator creates realistic market data with:

1. **Price Movements**: Brownian motion with configurable drift and volatility
2. **Volume Patterns**: Normal distribution with occasional spikes
3. **Market Events**: 5% chance of volatility spikes (3x normal movement)
4. **Realistic Spreads**: High/Low wicks based on volatility

Example generated bar:
```
OHLC: 648.38 / 648.83 / 648.23 / 648.77
Volume: 820,247
Timestamp: 2025-09-17 20:43:48
```

## Testing Workflow

### 1. Feature Computation Testing

```bash
# Enable feature logging
export LOG_FEATURES=true
docker compose -f ml/deployment/docker-compose.test.yml up

# Check feature values in database
docker exec ml-test-postgres-test-1 psql -U postgres -d nautilus_test -c \
  "SELECT * FROM ml_feature_values_2025_09 ORDER BY ts_event DESC LIMIT 5;"
```

### 2. Model Inference Testing

```bash
# Enable prediction logging
export LOG_PREDICTIONS=true
docker compose -f ml/deployment/docker-compose.test.yml up

# Check predictions in database
docker exec ml-test-postgres-test-1 psql -U postgres -d nautilus_test -c \
  "SELECT * FROM ml_model_predictions_2025_09 ORDER BY ts_event DESC LIMIT 5;"
```

### 3. Signal Generation Testing

```bash
# Lower threshold to generate more signals
export PREDICTION_THRESHOLD=0.3
export LOG_SIGNALS=true
docker compose -f ml/deployment/docker-compose.test.yml up

# Check signals in database
docker exec ml-test-postgres-test-1 psql -U postgres -d nautilus_test -c \
  "SELECT * FROM ml_strategy_signals_2025_09 ORDER BY ts_event DESC LIMIT 5;"
```

## Monitoring & Observability

### Prometheus Metrics

The test environment exposes metrics on port 8002:

```bash
# View all ML metrics
curl -s http://localhost:8002/metrics | grep ml_

# Key metrics to monitor:
# - ml_predictions_total: Total predictions made
# - ml_prediction_latency_seconds: Inference latency
# - ml_feature_computation_duration_seconds: Feature computation time
# - ml_signal_confidence: Distribution of signal confidence scores
```

### Log Locations

Container logs are automatically persisted by Docker:

```bash
# View logs location
docker inspect ml-test-ml_signal_actor_test-1 | grep LogPath

# Typical location:
/var/lib/docker/containers/<container-id>/<container-id>-json.log

# Stream logs to file
docker logs ml-test-ml_signal_actor_test-1 > ml_actor_test.log 2>&1
```

### Database Persistence

All events are persisted to time-partitioned tables:

```sql
-- Check data events
SELECT stage, status, COUNT(*)
FROM ml_data_events_2025_09
GROUP BY stage, status;

-- Check registry activity
SELECT action, entity_type, COUNT(*)
FROM registry_audit_log
GROUP BY action, entity_type;

-- Check feature computation stats
SELECT * FROM ml_feature_computation_stats
ORDER BY created_at DESC LIMIT 10;
```

## Performance Testing

### Load Testing

Increase the mock data rate to stress test the system:

```bash
# Generate 100 bars per second
export MOCK_DATA_RATE=100
docker compose -f ml/deployment/docker-compose.test.yml up

# Monitor performance
watch -n 1 'curl -s localhost:8002/metrics | grep -E "latency|duration"'
```

### Latency Requirements

The system must meet these latency targets:
- Feature computation: <500μs
- Model inference: <2ms
- End-to-end (bar → signal): <5ms

## Debugging

### Common Issues

1. **No predictions generated**
   - Check warm-up period (default: 20 bars)
   - Verify model is loaded: `curl localhost:8002/health`

2. **Database connection errors**
   - Ensure test database is running: `docker ps | grep postgres-test`
   - Check connection string in logs

3. **Mock data not flowing**
   - Verify `USE_MOCK_DATA=true` is set
   - Check generator logs for errors

### Debug Commands

```bash
# Check actor health
curl http://localhost:8002/health | jq

# View recent errors
docker logs ml-test-ml_signal_actor_test-1 2>&1 | grep ERROR

# Database query for failures
docker exec ml-test-postgres-test-1 psql -U postgres -d nautilus_test -c \
  "SELECT * FROM ml_data_events WHERE status='failed' ORDER BY created_at DESC LIMIT 10;"

# Check resource usage
docker stats ml-test-ml_signal_actor_test-1
```

## Integration with CI/CD

Add to your CI pipeline:

```yaml
# .github/workflows/ml-test.yml
test-ml-pipeline:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2

    - name: Start test environment
      run: |
        docker compose -f ml/deployment/docker-compose.test.yml up -d
        sleep 10  # Wait for initialization

    - name: Run mock data test
      run: |
        # Generate 100 bars
        docker exec ml-test-ml_signal_actor_test-1 \
          bash -c "export MOCK_DATA_RATE=50 && timeout 10 python /app/entrypoint_mock.py"

    - name: Verify predictions
      run: |
        PREDICTIONS=$(docker exec ml-test-postgres-test-1 \
          psql -U postgres -d nautilus_test -t -c \
          "SELECT COUNT(*) FROM ml_model_predictions")

        if [ $PREDICTIONS -lt 50 ]; then
          echo "Error: Only $PREDICTIONS predictions generated"
          exit 1
        fi

    - name: Check latency
      run: |
        curl -s localhost:8002/metrics | grep ml_prediction_latency_seconds
```

## Production Readiness

The mock testing environment validates:

- ✅ **Data Flow**: Bars → Features → Predictions → Signals
- ✅ **Persistence**: All stores and registries operational
- ✅ **Performance**: Latency within requirements
- ✅ **Monitoring**: Metrics and health checks working
- ✅ **Error Handling**: Circuit breakers and fallbacks functional
- ✅ **Scaling**: Handles high-frequency data (100+ bars/second)

## Next Steps

1. **Add Custom Models**: Replace dummy model with trained ONNX models
2. **Test Strategies**: Connect ML Strategy to consume signals
3. **Load Testing**: Increase data rate to find system limits
4. **A/B Testing**: Run multiple models in parallel
5. **Feature Engineering**: Add custom features and test computation
