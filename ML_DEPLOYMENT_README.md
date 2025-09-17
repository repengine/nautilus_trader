# ML Signal Actor Deployment Guide

## Current Configuration

The ML Signal Actor is configured to receive live market data from Databento's EQUS.MINI dataset with **EXTERNAL bar aggregation** as the default.

### Key Settings
- **Instrument**: `SPY.EQUS` (aggregated US equities)
- **Bar Type**: `SPY.EQUS-1-MINUTE-LAST-EXTERNAL` (Databento OHLCV-1m bars)
- **Dataset**: `EQUS.MINI` (Databento aggregated equities feed)
- **Model**: `dummy_bullish_model.onnx` (replace with your trained model)

## Deployment Commands

### Quick Start

```bash
# Set your Databento API key
export DATABENTO_API_KEY="your-api-key-here"

# Start the ML Signal Actor only
docker compose -f ml/deployment/docker-compose.yml up -d ml_signal_actor

# Start the full stack (actor + strategy + pipeline)
docker compose -f ml/deployment/docker-compose.yml up -d

# Monitor logs
docker compose -f ml/deployment/docker-compose.yml logs -f ml_signal_actor

# Check health
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### Direct Docker Run (without compose)

```bash
# Build the image
docker build -f ml/deployment/Dockerfile.actor -t ml-signal-actor .

# Run the container
docker run --rm \
  --name ml-signal-actor \
  -e DATABENTO_API_KEY="your-api-key-here" \
  -e USE_DUMMY_STORES="false" \
  -e INSTRUMENT_ID="SPY.EQUS" \
  -e BAR_TYPE="SPY.EQUS-1-MINUTE-LAST-EXTERNAL" \
  -e DB_CONNECTION="postgresql://postgres:postgres@localhost:5432/nautilus" \
  -v $(pwd)/ml/models:/app/models:ro \
  -p 8000:8000 \
  ml-signal-actor
```

## Important Notes

### Market Hours
- **EXTERNAL bars** (OHLCV-1m from Databento) are only available during market hours:
  - Monday-Friday: 9:30 AM - 4:00 PM ET
  - No data outside these hours or on weekends/holidays

### Bar Aggregation Types
- **EXTERNAL** (default): Uses Databento's pre-aggregated OHLCV bars
  - Pros: Lower latency, less CPU usage, official aggregation
  - Cons: Only available during market hours

- **INTERNAL**: Aggregates bars from trades locally
  - Pros: Can aggregate from any data source
  - Cons: Requires instrument definitions, higher CPU usage
  - Note: Currently not configured for EQUS venue

### Port Configuration
- ML Signal Actor: `8000` (metrics/health)
- ML Strategy: `8001` (metrics/health)
- ML Pipeline: `8081` (health)
- PostgreSQL: `5433` (host) → `5432` (container)
- Redis: `6380` (host) → `6379` (container)
- Prometheus: `9090`
- Grafana: `3000` (user: admin, pass: admin)

### Environment Variables

Override defaults using these environment variables:

```bash
# Databento
DATABENTO_API_KEY=your-key-here
DATABENTO_DATASET=EQUS.MINI

# Trading Configuration
INSTRUMENT_ID=SPY.EQUS
BAR_TYPE=SPY.EQUS-1-MINUTE-LAST-EXTERNAL
ACTOR_ID=MLSignalActor-001

# Database
DB_CONNECTION=postgresql://postgres:postgres@postgres:5432/nautilus
USE_DUMMY_STORES=false

# Model
MODEL_PATH=/app/models/dummy_bullish_model.onnx

# Ports (to avoid conflicts)
ML_ACTOR_HOST_PORT=8000
ML_STRATEGY_HOST_PORT=8001
ML_PIPELINE_HOST_PORT=8081
POSTGRES_HOST_PORT=5433
REDIS_HOST_PORT=6380
```

## Monitoring

### Health Checks
```bash
# Actor health
curl http://localhost:8000/health

# Strategy health
curl http://localhost:8001/health

# Pipeline health
curl http://localhost:8081/health
```

### Prometheus Metrics
Access metrics at http://localhost:9090

### Grafana Dashboards
Access dashboards at http://localhost:3000
- Username: admin
- Password: admin

## Troubleshooting

### No bars arriving outside market hours
This is expected behavior. EXTERNAL bars from Databento are only available during market hours (9:30 AM - 4:00 PM ET).

### "No instrument found" error with INTERNAL bars
INTERNAL bar aggregation requires instrument definitions. Currently not configured for EQUS venue. Use EXTERNAL bars instead.

### Port conflicts
If ports are already in use, override with environment variables:
```bash
export ML_ACTOR_HOST_PORT=8100
export POSTGRES_HOST_PORT=5432
```

### Container keeps restarting
Check logs for specific errors:
```bash
docker compose -f ml/deployment/docker-compose.yml logs ml_signal_actor
```

Common issues:
- Invalid API key (must be 32 characters)
- Database connection issues
- Model file not found

## Next Steps

1. **Replace the dummy model** with your trained ONNX model in `ml/models/`
2. **Configure the ML Strategy** to trade based on signals
3. **Set up the ML Pipeline** for data collection and feature computation
4. **Monitor performance** through Prometheus/Grafana dashboards
5. **Test during market hours** to see live data flow

## Testing

To verify the setup during market hours:

```bash
# Check if bars are arriving
docker logs ml-ml_signal_actor-1 --tail 50 | grep -E "Bar received|Processing bar"

# Check signal generation
docker logs ml-ml_signal_actor-1 --tail 50 | grep -E "Signal|Prediction"

# Check metrics
curl -s http://localhost:8000/metrics | grep ml_
```

## Support

For issues or questions:
- Check the logs first: `docker compose logs ml_signal_actor`
- Verify market hours and trading calendar
- Ensure your Databento API key is valid and has appropriate permissions
- Review the Nautilus Trader documentation for additional configuration options