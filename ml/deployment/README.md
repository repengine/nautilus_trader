# ML Trading System - Production Dry Run Setup

This directory contains everything needed to run the ML trading system with real market data in dry run mode.

## Quick Start

### Option 1: Local Run (Simplest)

```bash
# 1. Set up Databento API key
export DATABENTO_API_KEY=your_key_here

# 2. Install PostgreSQL (optional, will use SQLite if not available)
# On Ubuntu/Debian:
sudo apt install postgresql postgresql-client

# On Mac:
brew install postgresql
brew services start postgresql

# 3. Create database (PostgreSQL required) and apply canonical migrations
createdb nautilus
# Apply canonical migrations (order matters)
psql nautilus -f ../stores/migrations/001_stores_schema.sql
psql nautilus -f ../stores/migrations/002_auto_partitioning.sql
psql nautilus -f ../stores/migrations/003_market_data.sql
psql nautilus -f ../stores/migrations/004_data_registry.sql
psql nautilus -f ../stores/migrations/005_schema_hardening.sql
psql nautilus -f ../stores/migrations/005a_feature_values_dedupe.sql
psql nautilus -f ../stores/migrations/006_disable_partition_triggers.sql

# (Optional) Run DB preflight to verify functions/partitions
python -c "from ml.stores.db_preflight import check_db_prereqs; import os; print(check_db_prereqs(os.getenv('DB_CONNECTION','postgresql://postgres:postgres@localhost:5432/nautilus')))"

# 4. Run the system
python run_local_dry_run.py
```

### Option 2: Docker Compose (Production-like)

```bash
# 1. Set up Databento API key
export DATABENTO_API_KEY=your_key_here

# 2. Build and run with Docker Compose
chmod +x run_dry_run.sh
./run_dry_run.sh
```

## Architecture

```
┌─────────────────────┐
│   Databento Feed    │
│   (Real Market)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ML Signal Actor    │
│  - Load model       │
│  - Calculate features│
│  - Generate signals │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ML Trading Strategy│
│  - Receive signals  │
│  - Make decisions   │
│  - [DRY RUN MODE]   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    PostgreSQL       │
│  - Store features   │
│  - Store signals    │
│  - Store decisions  │
└─────────────────────┘
```

## Components

### 1. ML Signal Actor
- Connects to Databento for real market data
- Calculates technical indicators and features
- Runs model inference
- Publishes ML signals

### 2. ML Trading Strategy  
- Subscribes to ML signals
- Makes trading decisions
- **DRY RUN MODE**: Logs decisions but doesn't execute trades
- Persists all decisions to database

### 3. PostgreSQL Database
- Stores features for analysis
- Stores model predictions
- Stores strategy decisions
- Enables backtesting and analysis

### 4. Monitoring (Optional)
- Prometheus: Metrics collection
- Grafana: Visualization dashboards

## Configuration

### Environment Variables

```bash
# Required
DATABENTO_API_KEY=your_key_here

# Optional
DB_CONNECTION=postgresql://postgres:postgres@localhost:5432/nautilus
DATABENTO_DATASET=GLBX.MDP3  # CME data
INSTRUMENT_ID=ES-USD-FUT.CME  # E-mini S&P 500
BAR_TYPE=ES-USD-FUT.CME-1-MINUTE

# Dry Run Control
EXECUTE_TRADES=false  # Keep this false for dry run!

# Risk Parameters
POSITION_SIZE_PCT=0.02
MIN_CONFIDENCE=0.6
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04
```

### Supported Instruments

Databento provides data for various exchanges:

- **CME**: ES (E-mini S&P), NQ (E-mini Nasdaq), CL (Crude Oil)
- **DERIBIT**: BTC options and futures
- **BINANCE**: Crypto spot and futures
- **FTX**: Historical data (pre-bankruptcy)

Check Databento documentation for full list.

## Monitoring

### Logs

**Local:**
```bash
# Logs are printed to console
python run_local_dry_run.py
```

**Docker:**
```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f ml_signal_actor
docker-compose logs -f ml_strategy
```

### Database

```bash
# Connect to PostgreSQL
psql postgresql://postgres:postgres@localhost:5432/nautilus

# Check signals
SELECT * FROM public.ml_strategy_signals ORDER BY ts_event DESC LIMIT 10;

# Check features
SELECT * FROM public.ml_feature_values ORDER BY ts_event DESC LIMIT 10;

# Check model predictions
SELECT * FROM public.ml_model_predictions ORDER BY ts_event DESC LIMIT 10;
```

### Metrics

Access Prometheus: http://localhost:9090
Access Grafana: http://localhost:3000 (admin/admin)

Key metrics to monitor:
- `ml_signals_generated_total`: Number of signals generated
- `ml_signal_generation_seconds`: Signal generation latency
- `ml_feature_computation_seconds`: Feature calculation time
- `ml_inference_latency_seconds`: Model inference time
- `ml_strategy_dry_run_trades_total`: Dry run trades counter

## Safety Checks

Before going live:

1. **Verify Dry Run Mode**
   - Check logs for "[DRY RUN]" messages
   - Confirm `EXECUTE_TRADES=false` in environment
   - Verify no execution client configured

2. **Test with Small Data**
   - Run for 1 hour first
   - Check all metrics are reasonable
   - Verify persistence is working

3. **Validate Model**
   - Ensure model predictions are in expected range
   - Check feature calculations are correct
   - Verify signal generation logic

4. **Monitor Resources**
   - CPU usage should be < 50%
   - Memory usage should be stable
   - Database connections should not leak

## Transitioning to Live

When ready for real trading:

1. **Set up execution client**
   ```python
   # Add to node config
   exec_clients={
       "DATABENTO": DatabentoExecClientConfig(...),
   }
   ```

2. **Enable trading**
   ```bash
   export EXECUTE_TRADES=true
   ```

3. **Start with minimal risk**
   ```bash
   export POSITION_SIZE_PCT=0.001  # 0.1% per trade
   export MAX_POSITIONS=1
   ```

4. **Monitor closely**
   - Watch position changes
   - Monitor P&L in real-time
   - Have kill switch ready

## Troubleshooting

### "Connection refused" for PostgreSQL
```bash
# Start PostgreSQL
sudo systemctl start postgresql
# or
brew services start postgresql
```

### "API key not valid"
Check your Databento API key and dataset access

### "No signals generated"
- Check warm_up_period (need 20+ bars)
- Verify market is open
- Check model is loading correctly

### "Database schema not found"
```bash
# Apply canonical migrations (not the legacy schema/ files)
psql nautilus -f ../stores/migrations/001_stores_schema.sql
psql nautilus -f ../stores/migrations/002_auto_partitioning.sql
psql nautilus -f ../stores/migrations/003_market_data.sql
psql nautilus -f ../stores/migrations/004_data_registry.sql
psql nautilus -f ../stores/migrations/005_schema_hardening.sql
psql nautilus -f ../stores/migrations/005a_feature_values_dedupe.sql
psql nautilus -f ../stores/migrations/006_disable_partition_triggers.sql

# Optional: verify
python -c "from ml.stores.db_preflight import check_db_prereqs; print(check_db_prereqs('postgresql://postgres:postgres@localhost:5432/nautilus'))"
```

## Support

- Nautilus Trader: https://nautilustrader.io/
- Databento: https://databento.com/docs
- Issues: Create issue in ml/issues/
