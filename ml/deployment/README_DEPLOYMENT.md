# ML Pipeline Docker Deployment

This directory contains the Docker Compose configuration for deploying the Nautilus Trader ML pipeline in production.

## Quick Start

1. **Configure environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env and add your DATABENTO_API_KEY
   ```

2. **Build and start services:**

   ```bash
   make build
   make up
   ```

3. **Check health:**

   ```bash
   make health

4. **Verify DB prerequisites (recommended):**

   Ensure required DB functions and current-month partitions are present. From your host:

   ```bash
   python -c "from ml.stores.db_preflight import check_db_prereqs; print(check_db_prereqs('postgresql://postgres:postgres@localhost:5432/nautilus'))"
   ```
   ```

4. **View logs:**

   ```bash
   make logs
   ```

## Services

### ml_pipeline
The main ML pipeline service that handles:

- Data collection from Databento
- Feature computation and storage
- Model training coordination
- Real-time processing

**Modes:**

- `daily`: Scheduled daily updates (default)
- `backfill`: Historical data processing
- `realtime`: Continuous real-time updates

### Supporting Services

- `postgres`: PostgreSQL database for persistence
- `redis`: Message bus for inter-service communication
- `ml_signal_actor`: ML inference actor for signal generation
- `ml_strategy`: Trading strategy execution (dry-run mode)
- `prometheus`: Metrics collection
- `grafana`: Metrics visualization

## Configuration

### Environment Variables
See `.env.example` for all available configuration options.

Key variables:

- `DATABENTO_API_KEY`: Your Databento API key (required)
- `PIPELINE_MODE`: Operation mode (daily/backfill/realtime)
- `UNIVERSE_SYMBOLS`: Comma-separated list of symbols to process
- `LOG_LEVEL`: Logging verbosity (DEBUG/INFO/WARNING/ERROR)

### Resource Limits
The pipeline is configured with:

- Memory: 16GB (adjustable via MEMORY_LIMIT)
- CPUs: 4 cores (adjustable via CPU_LIMIT)

## Usage Examples

### Daily Production Mode

```bash
# Standard daily operation (runs at 5 PM)
make deploy
```

### Backfill Historical Data

```bash
# Backfill January 2024
make backfill START=2024-01-01 END=2024-01-31
```

### Real-time Mode

```bash
# Run with 60-second update intervals
make realtime
```

### Development Mode

```bash
# Copy override file for development
cp docker-compose.override.yml.example docker-compose.override.yml

# Run with live code mounting
make dev
```

## Monitoring

### Health Check
The pipeline exposes a health endpoint at `http://localhost:8080/health`:

```bash
curl http://localhost:8080/health
```

### Metrics
Prometheus metrics available at `http://localhost:9090`
Grafana dashboards at `http://localhost:3000` (admin/admin)

### Logs

```bash
# All services
docker-compose logs -f

# Specific service
make logs SERVICE=ml_pipeline

# Pipeline log file
docker-compose exec ml_pipeline tail -f /app/logs/ml_pipeline.log
```

## Database Management

### Backup

```bash
make backup
# Creates backup_YYYYMMDD_HHMMSS.sql
```

### Restore

```bash
make restore FILE=backup_20240101_120000.sql
```

### Direct Access

```bash
# PostgreSQL CLI
docker-compose exec postgres psql -U postgres nautilus

### Migrations

Canonical migrations live under `ml/stores/migrations/`. Apply them in order before first run:

```bash
docker-compose exec -T postgres psql -U postgres nautilus -f /app/ml/stores/migrations/001_stores_schema.sql
docker-compose exec -T postgres psql -U postgres nautilus -f /app/ml/stores/migrations/002_auto_partitioning.sql
docker-compose exec -T postgres psql -U postgres nautilus -f /app/ml/stores/migrations/003_market_data.sql
docker-compose exec -T postgres psql -U postgres nautilus -f /app/ml/stores/migrations/004_data_registry.sql
docker-compose exec -T postgres psql -U postgres nautilus -f /app/ml/stores/migrations/005_schema_hardening.sql
docker-compose exec -T postgres psql -U postgres nautilus -f /app/ml/stores/migrations/005a_feature_values_dedupe.sql
docker-compose exec -T postgres psql -U postgres nautilus -f /app/ml/stores/migrations/006_disable_partition_triggers.sql
```

# pgAdmin web interface (if using override)
# http://localhost:5050
# Login: admin@nautilus.local / admin
```

## Troubleshooting

### Pipeline not starting

1. Check logs: `make logs`
2. Verify database is healthy: `docker-compose ps postgres`
3. Check environment variables: `docker-compose config`

### Memory issues
Adjust limits in docker-compose.yml:

```yaml
deploy:
  resources:
    limits:
      memory: 32g  # Increase as needed
```

### Connection errors
Ensure all services are on the same network:

```bash
docker network ls
docker network inspect nautilus_network
```

### Data not processing

1. Check Databento API key is valid
2. Verify universe symbols are correct
3. Check feature store connectivity
4. Review error logs for specific issues

## Production Deployment Checklist

- [ ] Set strong PostgreSQL password
- [ ] Configure backup strategy
- [ ] Set up monitoring alerts
- [ ] Configure log rotation
- [ ] Set appropriate resource limits
- [ ] Enable SSL/TLS for external connections
- [ ] Configure firewall rules
- [ ] Set up health check monitoring
- [ ] Document recovery procedures
- [ ] Test failover scenarios

## Security Notes

1. **Never commit .env files** - Use secrets management in production
2. **Rotate API keys regularly**
3. **Use read-only mounts where possible**
4. **Restrict network access** to only required services
5. **Enable PostgreSQL SSL** for production deployments
6. **Use Docker secrets** for sensitive data in production

## Support

For issues or questions:

1. Check logs first: `make logs`
2. Review health status: `make health`
3. Consult ml/docs/context/context_deployment.md
4. Check ml/NEXT_STEPS_ACTION_PLAN.md for implementation status
