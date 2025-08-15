# ML Trading System - Dry Run Checklist

## Prerequisites for Dry Run

### ✅ 1. Model Artifacts
- **Required**: A trained model file in `.pkl`, `.joblib`, or `.onnx` format
- **Quick Start**: Run `python ml/examples/create_dummy_model.py` to create test models
- **Production**: Train your actual model using your data and save it

### ✅ 2. Feature Configuration
Your model expects specific features in a specific order:
```python
feature_config = MLFeatureConfig(
    lookback_window=20,  # Number of bars for indicators
    feature_names=["feature_0", "feature_1", ...],  # Must match model
    normalize_features=True,
    indicators={"rsi": {"period": 14}, ...}  # Technical indicators
)
```

### ✅ 3. Market Data Source
Options:
- **Backtest**: Use historical data (easiest for testing)
- **Paper Trading**: Connect to exchange sandbox/testnet
- **Live Data**: Connect to real exchange (still safe with execute_trades=False)

### ✅ 4. Database/Storage
Options:
- **Quick Start**: Set `use_dummy_stores=True` (no persistence, good for testing)
- **Development**: Use SQLite (automatic fallback)
- **Production**: Set up PostgreSQL with schema from `ml/schema/`

### ✅ 5. Configuration Files

#### MLSignalActor Configuration
```python
actor_config = MLSignalActorConfig(
    model_id="your_model",
    model_path="ml/models/your_model.pkl",
    bar_type=BarType.from_str("BTC-USDT.BINANCE-1-MINUTE"),
    instrument_id=InstrumentId.from_str("BTC-USDT.BINANCE"),
    use_dummy_stores=True,  # For testing without database
    feature_config=feature_config,
    prediction_threshold=0.5,
    warm_up_period=20,  # Bars needed before predictions start
)
```

#### MLStrategy Configuration  
```python
strategy_config = MLStrategyConfig(
    strategy_id="ML-DRY-RUN",
    instrument_id=InstrumentId.from_str("BTC-USDT.BINANCE"),
    ml_signal_source="your_actor_id",
    execute_trades=False,  # DRY RUN MODE!
    position_size_pct=0.02,
    min_confidence=0.6,
    use_strategy_store=True,
    persist_all_signals=True,
)
```

## Quick Start Commands

### 1. Create Dummy Models
```bash
python ml/examples/create_dummy_model.py
```

### 2. Run Backtest Dry Run
```bash
python ml/examples/dry_run_example.py
```

### 3. Run Live Dry Run (requires data connection)
```bash
python ml/examples/dry_run_example.py --live
```

## What Happens in Dry Run Mode

With `execute_trades=False`:

✅ **WILL HAPPEN:**
- ML model loads and runs inference
- Features are calculated from market data
- Signals are generated and published
- Strategy receives and processes signals
- Trading decisions are made and logged
- Metrics are updated (Prometheus)
- Decisions are persisted to stores
- Position sizing is calculated
- Risk parameters are evaluated
- Logs show "[DRY RUN] Would execute trade..."

❌ **WILL NOT HAPPEN:**
- No orders sent to broker/exchange
- No real positions opened
- No real money at risk
- No execution fees
- No slippage (unless simulated)

## Monitoring During Dry Run

### Logs to Watch
```
INFO - Processing ML signal from xgb_model: prediction=0.750, confidence=0.820
INFO - [DRY RUN] Would enter BUY position (execute_trades=False) - Total dry run trades: 1
INFO - Strategy decision persisted: BUY signal for BTC-USDT.BINANCE
```

### Metrics Available
- Signal generation latency
- Feature computation time  
- Model inference time
- Signals received/generated
- Dry run trades counter
- Decision persistence success

## Transitioning to Live Trading

When ready to go live:

1. **Train Production Model**
   - Use real historical data
   - Validate performance metrics
   - Save model with version control

2. **Set Up Production Database**
   ```bash
   # Create PostgreSQL database
   createdb nautilus
   
   # Apply schema
   psql nautilus < ml/schema/features.sql
   psql nautilus < ml/schema/models.sql
   psql nautilus < ml/schema/strategies.sql
   ```

3. **Connect to Broker/Exchange**
   - Set up API keys
   - Configure execution client
   - Test with small amounts first

4. **Enable Live Trading**
   ```python
   strategy_config = MLStrategyConfig(
       ...
       execute_trades=True,  # ENABLE LIVE TRADING
       position_size_pct=0.01,  # Start small!
       ...
   )
   ```

5. **Monitor Closely**
   - Watch Prometheus metrics
   - Set up alerts for anomalies
   - Have kill switch ready
   - Monitor P&L in real-time

## Safety Checks

Before going live, ensure:
- [ ] Model validated on out-of-sample data
- [ ] Risk parameters are conservative
- [ ] Stop losses are configured
- [ ] Position sizing is appropriate
- [ ] Database backups are configured
- [ ] Monitoring/alerting is active
- [ ] You understand the strategy logic
- [ ] Emergency shutdown procedure is ready

## Troubleshooting

### "Model not found"
Run: `python ml/examples/create_dummy_model.py`

### "Database connection failed"  
Set: `use_dummy_stores=True` in configs

### "No signals generated"
Check: `warm_up_period` - need enough bars before predictions start

### "Features don't match model"
Ensure: `feature_names` in config match model training features

## Support

For issues or questions:
- Check logs in detail mode
- Review ml/tests/ for examples
- Consult Nautilus Trader docs
- Review ml/examples/ for patterns