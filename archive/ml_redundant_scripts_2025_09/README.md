# ML Redundant Scripts Archive - September 2025

This directory contains redundant ML scripts removed during the complete cleanup on 2025-09-03.

## Scripts Archived

### FRED Data Scripts (1 script)
- `populate_fred_data.py` - **REDUNDANT** with `simple_fred_updater.py`
  - Complex ML integration version replaced by simpler, more reliable updater
  - Our `simple_fred_updater.py` + `fred_integration_bridge.py` provides same functionality

### Test Scripts (4 scripts)  
- `test_databento_connection.py` - Connection testing
- `test_databento_datasets.py` - Dataset validation testing
- `test_databento_scheduler.py` - Scheduler testing
- `test_pipeline_health.py` - Pipeline health testing

**Reason**: All test functionality now handled by proper unit tests in `ml/tests/`

### Analysis Utilities (2 scripts)
- `analysis/symbol_index.py` - Symbol indexing utility
- `analysis/usage_map.py` - Usage mapping utility

**Reason**: Small utilities consolidated into `tools/data_analysis.py`

## Functionality Replacement

### FRED Data Population
```bash
# OLD: Complex ML-integrated FRED loader
python ml/scripts/populate_fred_data.py --backfill-years 10

# NEW: Simple, reliable FRED updater + ML integration
python simple_fred_updater.py  # Fetch latest data
python fred_integration_bridge.py  # Convert to ML format
```

### Testing
```bash
# OLD: Individual test scripts
python ml/scripts/test_databento_connection.py
python ml/scripts/test_databento_datasets.py

# NEW: Comprehensive test suites
pytest ml/tests/integration/test_databento_*
pytest ml/tests/unit/
```

### Analysis
```bash
# OLD: Small utility scripts
python ml/scripts/analysis/symbol_index.py
python ml/scripts/analysis/usage_map.py

# NEW: Consolidated analysis tool
python tools/data_analysis.py --analysis all
```

## Remaining ML Scripts (11 production scripts)

### Core Production Scripts
- `build_production_dataset.py` - Production dataset builder
- `check_databento_subscription.py` - Subscription validation
- `check_pipeline_health.py` - Health monitoring
- `populate_alternative_data.py` - Alternative data sources
- `populate_l2_efficient.py` - L2 market data (updated version)
- `populate_supplementary_simple.py` - Supplementary data
- `populate_universe.py` - Universe population
- `populate_yahoo_data.py` - Yahoo Finance data
- `run_ml_pipeline.py` - Main ML pipeline runner
- `sanity_check.py` - Pipeline sanity checks
- `train_tft_quick.py` - TFT model training

### Why These Were Kept
- **No redundancy**: Each serves a unique production purpose
- **Active usage**: Still used in ML pipeline workflows  
- **Core functionality**: Essential for ML data processing
- **No better alternatives**: These are the canonical implementations

## Migration Benefits

1. **Eliminated FRED redundancy**: One reliable FRED updater instead of two competing implementations
2. **Proper test organization**: Tests moved to appropriate `ml/tests/` structure
3. **Consolidated utilities**: Small tools merged into comprehensive analysis tool
4. **Cleaner ML directory**: Focus on production scripts only

## Script Count Reduction

- **Before**: 18 ML scripts
- **After**: 11 ML production scripts  
- **Reduction**: 39% fewer scripts
- **Archived**: 7 redundant scripts safely preserved

## Restoration Notes

If any archived functionality is needed:
- FRED: Use `simple_fred_updater.py` + `fred_integration_bridge.py` 
- Testing: Check `ml/tests/` for proper test implementations
- Analysis: Use `tools/data_analysis.py` comprehensive tool

All archived scripts can be restored if absolutely necessary, but modern alternatives are recommended.