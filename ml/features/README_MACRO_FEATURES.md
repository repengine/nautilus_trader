# Macro Feature Engineering with Training/Inference Parity

## Overview

This module provides **revision-aware macro features** from ALFRED/FRED data with guaranteed parity between batch (training) and real-time (inference) computation paths.

### Key Components

- **[macro_cache.py](macro_cache.py)**: Fast cache for real-time macro feature access
- **[macro_transforms.py](macro_transforms.py)**: Feature transform with batch/real-time parity
- **[../data/macro_revisions.py](../data/macro_revisions.py)**: Batch revision computation (cold path)
- **[../data/fred_join.py](../data/fred_join.py)**: ALFRED vintage join logic

## Architecture

### Parity Pattern

```
┌─────────────────────────────────────────────────────────────┐
│ TRAINING (Batch / Cold Path)                                │
├─────────────────────────────────────────────────────────────┤
│  DataFrame → MacroFeatureTransform.compute_batch()           │
│           → join_fred_asof() with point-in-time vintages    │
│           → Returns DF with macro columns                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ INFERENCE (Real-time / Hot Path)                           │
├─────────────────────────────────────────────────────────────┤
│  Bar → MacroFeatureTransform.compute_realtime()             │
│     → MacroDataCache.get_all_features()                     │
│     → Returns dict[str, float] (latest values, no as-of)    │
└─────────────────────────────────────────────────────────────┘

✅ SAME FEATURE NAMES → Parity maintained
```

### Why Real-time is Simpler

**Training** (complex):
- Question: "What was known on 2023-06-15?"
- Need: Point-in-time filtering of ALFRED vintages
- Complexity: As-of joins, release calendar filtering

**Inference** (simple):
- Question: "What is known NOW?"
- Need: Latest released values only
- Complexity: O(1) cache lookup

## Usage

### 1. Create Transform

```python
from ml.features import MacroFeatureTransform

transform = MacroFeatureTransform(
    macro_series_ids=["PAYEMS", "UNRATE", "CPIAUCSL"],
    vintage_base_dir="data/features/macro/fred/vintages",
    include_revisions=True,
    revision_mode="core",  # "minimal", "core", or "full"
    min_coverage=0.85,      # Require at least 85% coverage when assembling data
)
```

### 2. Batch Computation (Training)

```python
import polars as pl

# Load market data
df = pl.read_parquet("ohlcv_data.parquet")

# Add macro features
df_with_macro = transform.compute_batch(df)

# Now has columns: PAYEMS__value_real_time, PAYEMS_revision_1m, etc.
```

### 3. Real-time Computation (Inference)

```python
# In your trading actor's on_bar() handler
def on_bar(self, bar: Bar) -> None:
    # Get macro features (uses cached latest values)
    macro_features = transform.compute_realtime(bar=bar)

    # Combine with technical features
    all_features = {
        **technical_features,  # RSI, SMA, etc.
        **macro_features,      # PAYEMS, revisions, etc.
    }

    # Make prediction
    prediction = model.predict(all_features)
```

### 4. Cache Management

```python
# Refresh cache daily to pick up new FRED releases
transform.refresh_cache()

# Check which series are available
coverage = transform.get_cache_coverage()
# {'PAYEMS': True, 'UNRATE': True, 'CPIAUCSL': True}
```

## Feature Modes

### Minimal (3 features per series)
- `{series}__value_real_time` - Current release
- `{series}_prior_1m` - Value from 1 month ago
- `{series}_revision_1m` - Revision delta

### Core (6 features per series)
Minimal +
- `{series}_mom_1m` - Month-over-month change
- `{series}_pct_1m` - Percentage change
- `{series}_net_signal_1m` - Headline adjusted for revision

### Full (12 features per series)
Core +
- `{series}_prior_3m`, `{series}_prior_12m`
- `{series}_mom_3m`, `{series}_mom_12m`
- `{series}_pct_12m`
- `{series}_revision_3m`

## Example Features

For `PAYEMS` (NFP) in **core** mode:

```python
{
    "PAYEMS__value_real_time": 254000.0,     # September NFP (latest)
    "PAYEMS_prior_1m": 159000.0,             # August NFP (revised)
    "PAYEMS_revision_1m": 17000.0,           # August was revised +17k
    "PAYEMS_mom_1m": 95000.0,                # Sep vs Aug growth
    "PAYEMS_pct_1m": 0.597,                  # 59.7% increase
    "PAYEMS_net_signal_1m": 237000.0,        # Headline - revision
}
```

## Performance

### Real-time Latency
- Cache initialization: ~500ms (once at startup)
- `compute_realtime()`: <1ms per call (O(1) lookups)
- Cache refresh: ~200ms (call daily)

### Memory
- Cache size: ~100KB per series (23 series = 2.3MB)
- Negligible compared to model/indicator overhead

## Integration with Pipeline

### Option A: Manual (Current)

```python
# In TFTDatasetBuilder
df = load_ohlcv()
df = join_fred_asof(df, include_revisions=True)  # Batch
features_df = feature_engineer.compute_batch(df)  # Technical

# In Actor
macro_features = transform.compute_realtime()  # Macro
tech_features = feature_engineer.compute_realtime()  # Technical
all_features = {**macro_features, **tech_features}
```

### Option B: Pipeline Integration (TODO)

```python
# Register macro transform in pipeline
from ml.features.pipeline import PipelineSpec, TransformSpec

spec = PipelineSpec(transforms=[
    TransformSpec(name="returns", ...),
    TransformSpec(name="rsi", ...),
    TransformSpec(name="macro", transform=macro_transform),  # NEW!
])

# FeatureEngineer now handles both technical + macro
feature_engineer = FeatureEngineer(pipeline_spec=spec)

# Batch and real-time now unified!
features = feature_engineer.compute_batch(df)  # All features
features = feature_engineer.compute_realtime(bar)  # All features
```

## Configuration

### In Orchestrator Config (TOML)

```toml
[dataset]
include_macro = true
macro_series_ids = ["PAYEMS", "UNRATE", "CPIAUCSL"]
fred_vintage_dir = "data/features/macro/fred/vintages"
macro_min_coverage = 0.85

# Revision features
include_macro_revisions = true
macro_revision_mode = "core"  # "minimal", "core", "full"
macro_revision_windows = [1, 3, 12]  # Months
```

### In Python Code

```python
from ml.features import create_macro_transform_from_config

transform = create_macro_transform_from_config(
    macro_series_ids=cfg.macro_series_ids,
    vintage_base_dir=cfg.fred_vintage_dir,
    include_revisions=cfg.include_macro_revisions,
    revision_mode=cfg.macro_revision_mode,
    min_coverage=cfg.macro_min_coverage,
)
```

## Testing

### Parity Validation

```python
from ml.tests.unit.features.test_macro_transforms_parity import (
    TestMacroTransformParity,
)

# Run parity tests
pytest ml/tests/unit/features/test_macro_transforms_parity.py -v
```

### Unit Tests

- `test_cache_loads_successfully` - Cache initialization
- `test_realtime_features_match_structure` - Feature name consistency
- `test_feature_names_match_mode` - Mode-specific features
- `test_batch_computation_runs` - Batch path works
- `test_cache_refresh` - Cache update mechanism

## Troubleshooting

### Cache Not Loading

```python
coverage = transform.get_cache_coverage()
# {'PAYEMS': False, ...}

# Check if vintages exist
ls data/features/macro/fred/vintages/PAYEMS/release_calendar.parquet

# Download if missing
python scripts/download_alfred_vintages.py
```

### Feature Count Mismatch

```python
expected = transform.get_feature_names()  # 138 for 23 series, core mode
actual = transform.compute_realtime()     # May have fewer if series missing

# Check which are missing
missing = set(expected) - set(actual.keys())
```

### Performance Issues

```python
import time

# Benchmark real-time computation
start = time.perf_counter()
features = transform.compute_realtime()
elapsed_ms = (time.perf_counter() - start) * 1000

assert elapsed_ms < 5.0, f"Too slow: {elapsed_ms:.2f}ms"
```

## Future Enhancements

### Phase 1 (Current)
- ✅ MacroDataCache for real-time lookups
- ✅ MacroFeatureTransform with batch/real-time methods
- ✅ Basic parity tests

### Phase 2 (TODO)
- ⬜ Full pipeline integration (register in PipelineSpec)
- ⬜ FeatureManifest registration
- ⬜ Comprehensive parity validation (assert_allclose)
- ⬜ Performance benchmarks

### Phase 3 (TODO)
- ⬜ Automatic cache refresh (daily cron)
- ⬜ Prometheus metrics (cache hits, refresh times)
- ⬜ Feature importance analysis (with vs. without revisions)
- ⬜ Multi-frequency support (daily, weekly macro data)

## References

- **ALFRED Documentation**: https://fred.stlouisfed.org/docs/api/fred/
- **Revision Features Prompt**: [ALFRED_REVISION_FEATURES_PROMPT.md](../config/orchestrator/ALFRED_REVISION_FEATURES_PROMPT.md)
- **Architecture Issue**: [REVISION_FEATURES_ARCHITECTURE_ISSUE.md](../config/orchestrator/REVISION_FEATURES_ARCHITECTURE_ISSUE.md)
