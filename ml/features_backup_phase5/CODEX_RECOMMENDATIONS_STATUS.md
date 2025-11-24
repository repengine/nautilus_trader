# Codex Recommendations - Implementation Status

## Summary

This document tracks implementation of Codex's recommendations for enhancing the ML feature stack to support "powerful" predictive models.

---

## ✅ Completed (#1): Extend FeatureConfig and Pipeline Integration

**Recommendation**:
> Extend FeatureConfig to expose macros/calendar/event transforms and patch build_pipeline_spec_from_feature_config accordingly

**Status**: ✅ **COMPLETE**

**Implementation**:

1. **Added to FeatureConfig** ([ml/features/engineering.py:249-257](../features/engineering.py)):
   ```python
   # Macro features (ALFRED/FRED)
   include_macro: bool = False
   macro_series_ids: list[str] = msgspec.field(default_factory=list)
   include_macro_revisions: bool = False
   macro_revision_mode: str = "core"  # "minimal", "core", "full"

   # Calendar features (known-future for TFT)
   include_calendar: bool = False
   calendar_encoding: str = "cyclic"  # "cyclic", "onehot", "fourier"
   ```

2. **Updated Pipeline Builder** ([ml/features/engineering.py:3139-3153](../features/engineering.py)):
   - Macro transforms automatically added when `include_macro=True`
   - Calendar transforms automatically added when `include_calendar=True`
   - Parameters flow through to PipelineSpec correctly

3. **Validation**:
   - 10 tests in `test_feature_config_macro_integration.py` ✅
   - 9 tests in `test_macro_pipeline_integration.py` ✅
   - 6 tests in `test_macro_transforms_parity.py` ✅

**Result**: Users can now enable macro and calendar features via simple config flags, and they will automatically materialize in dataset builds.

---

## 🚧 In Progress (#2): Enrich Macro Layer with Factorized Composites

**Recommendation**:
> Enrich the macro layer with factorized composites (credit spreads, duration ladders, liquidity/funding metrics) and align instrument metadata

**Status**: 🚧 **IN PROGRESS** (80% complete)

**What's Implemented**:

1. **Macro Composites Module** ([ml/features/macro_composites.py](macro_composites.py)):
   - `compute_macro_composites_pl()`: Computes 26 composite features
   - Factorized across 5 economic dimensions:

   | Dimension | Features | Examples |
   |-----------|----------|----------|
   | **Credit/Risk** | 5 | credit_spread_ig, credit_spread_hy, credit_risk_index |
   | **Duration/Term** | 4 | term_spread, yield_curve_slope, fed_policy_stance |
   | **Liquidity** | 4 | liquidity_index, qe_intensity, bank_credit_growth_3m |
   | **Growth/Inflation** | 8 | growth_momentum, inflation_momentum, stagflation_risk, goldilocks_score |
   | **FX** | 3 | dollar_strength, dollar_momentum_3m, fx_stress |

2. **Pipeline Registration** ([ml/features/pipeline.py:552-585](pipeline.py)):
   - `_MacroCompositesTransform` registered in catalog
   - Available via `TransformSpec(name="macro_composites")`

**What's Missing**:

1. **FeatureConfig Integration** ✅:
   ```python
   # TODO: Add to FeatureConfig
   include_macro_composites: bool = False

   # TODO: Add to build_pipeline_spec_from_feature_config
   if getattr(cfg, "include_macro_composites", False):
       transforms.append(TransformSpec(name="macro_composites", params={}))
   ```

2. **Instrument Metadata Mapping**:
   - Need to add instrument-level factors (sector, market cap, beta)
   - Map 95 instruments to macro sensitivity profiles
   - Example: Tech stocks → high dollar sensitivity, rate sensitivity

3. **Real-time Computation**:
   - Batch computation implemented (via `compute_macro_composites_pl`)
   - Real-time path implemented via `_compute_realtime_composites` with cache snapshots

**Next Steps**:
- [x] Add `include_macro_composites` to FeatureConfig
- [x] Wire into pipeline builder
- [x] Add real-time computation to MacroFeatureTransform.compute_realtime()
- [ ] Create instrument metadata mapping (sector → macro factors)

---

## 📋 TODO (#3): Dataset Validation for Macro Coverage

**Recommendation**:
> Upgrade dataset orchestration to enforce macro coverage (validator that macro_series_ids resolved to non-empty columns) and prime MacroDataCache

**Status**: ✅ **DONE**

**Required Implementation**:

1. **Coverage Validator**:
   ```python
   # ml/data/validators.py
   class MacroCoverageValidator:
       def validate_macro_coverage(
           self,
           df: pl.DataFrame,
           expected_series: list[str],
       ) -> ValidationResult:
           """
           Ensure all expected macro series are present as non-empty columns.

           Raises error if:
           - Series missing from DataFrame
           - Series is all NaN/null
           - Series has <90% coverage
           """
           missing = []
           empty = []
           sparse = []

           for series in expected_series:
               if series not in df.columns:
                   missing.append(series)
               elif df[series].null_count() == len(df):
                   empty.append(series)
               elif df[series].null_count() / len(df) > 0.1:
                   sparse.append(series)

           if missing or empty or sparse:
               raise MacroCoverageError(
                   f"Macro coverage failed:\n"
                   f"Missing: {missing}\n"
                   f"Empty: {empty}\n"
                   f"Sparse (>10% null): {sparse}"
               )
   ```

2. **Dataset Build Integration**:
   ```python
   # In MacroFeatureTransform.compute_batch
   validator = MacroCoverageValidator(min_coverage=self._min_coverage)
   validator.validate_macro_coverage(
       output,
       self._series_ids_for_batch,
   )
   ```

3. **Cache Priming**:
   ```python
   # In MacroFeatureTransform.__init__
   if include_revisions:
       # Prime cache on initialization
       self._cache = MacroDataCache(
           vintage_base_dir=vintage_base_dir,
           series_ids=macro_series_ids,
           enable_revisions=True,
       )

       # Validate coverage
       coverage = self._cache.get_coverage()
       missing = [s for s, avail in coverage.items() if not avail]
       if missing:
           logger.warning(f"Missing ALFRED vintages for: {missing}")
   ```

**Validation Checkpoints**:
- [ ] Pre-build: Check ALFRED vintages exist
- [ ] Post-join: Validate macro columns present
- [ ] Post-build: Check feature coverage ≥95%
- [ ] Pre-training: Validate cache primed with correct series

---

## 📋 TODO (#4): Contract-Level Covariates

**Recommendation**:
> Add contract-level covariates (duration buckets, CDS proxies, bid-ask/liquidity metrics) to give TFT explicit signals for the three target forces

**Status**: ⬜ **TODO**

**Required Implementation**:

1. **Instrument Metadata Enrichment**:
   ```python
   # ml/features/instrument_metadata.py
   @dataclass
   class InstrumentMetadata:
       instrument_id: str
       symbol: str

       # Duration/Maturity
       duration_bucket: str  # "short", "medium", "long"
       years_to_maturity: float | None  # For bonds/futures

       # Liquidity
       avg_daily_volume: float
       avg_spread_bps: float
       liquidity_tier: int  # 1=most liquid, 3=least liquid

       # Credit/Risk
       sector: str  # "Technology", "Finance", "Energy", etc.
       market_cap_bucket: str  # "mega", "large", "mid"
       beta_sp500: float  # Systematic risk
       cds_proxy: str | None  # Closest CDS ticker

       # Macro Sensitivity
       rate_sensitivity: float  # Duration-like for equities
       dollar_sensitivity: float  # FX exposure
       credit_sensitivity: float  # Spread sensitivity
   ```

2. **Static Covariate Features** (per TFT known-future inputs):
   ```python
   # These are constant per instrument, known in advance
   static_features = {
       "duration_bucket_short": 1.0 if bucket == "short" else 0.0,
       "duration_bucket_medium": 1.0 if bucket == "medium" else 0.0,
       "duration_bucket_long": 1.0 if bucket == "long" else 0.0,
       "liquidity_tier": tier,  # 1, 2, or 3
       "sector_technology": 1.0 if sector == "Technology" else 0.0,
       "sector_finance": 1.0 if sector == "Finance" else 0.0,
       # ...
       "beta_sp500": beta,
       "rate_sensitivity": rate_beta,
       "dollar_sensitivity": fx_beta,
   }
   ```

3. **Dynamic Liquidity Features** (time-varying):
   ```python
   # Computed from L2 data or rolling windows
   dynamic_liquidity = {
       "bid_ask_spread_bps": spread,
       "order_book_depth_bps": depth,
       "volume_shock": volume / avg_volume,
       "price_impact_bps": estimated_impact,
   }
   ```

4. **Integration with TFT**:
   - Static covariates → TFT `static_covariates` input
   - Dynamic liquidity → TFT `time_varying_known_reals` (if deterministic)
   - Macro composites → TFT `time_varying_known_reals` (known-future)

**Data Sources**:
- Duration/sector/beta: From instrument catalog or external data provider
- Liquidity metrics: Computed from L2 order book data (already have in ml/features/microstructure.py)
- CDS proxies: Mapping table (manual curation for 95 instruments)

**Next Steps**:
- [ ] Create `instrument_metadata.py` with metadata dataclass
- [ ] Build metadata CSV for 95 EQUS.MINI instruments
- [ ] Add static covariate transform to pipeline
- [ ] Add dynamic liquidity features from L2 aggregates
- [ ] Document TFT input mapping (static vs. known vs. unknown)

---

## 📊 Impact Analysis

### Current Feature Stack (With Our Work)

| Category | Count | Signal Density |
|----------|-------|----------------|
| **OHLCV** | 5 | Low (raw data) |
| **Technical** | ~15 | Medium (standard TA) |
| **Macro (base)** | 23 | Medium (economic levels) |
| **Macro (revisions)** | 115 | **High** (trader view) ✅ |
| **Calendar** | ~8 | Medium (time patterns) |
| **Composites** | 26 | **High** (factorized) 🚧 |
| **Instrument Static** | 0 | - ⬜ |
| **Liquidity Dynamic** | 0 | - ⬜ |
| **Total** | **192** | |

### Target Feature Stack (After Codex Recommendations)

| Category | Count | Signal Density |
|----------|-------|----------------|
| **OHLCV** | 5 | Low (raw data) |
| **Technical** | ~15 | Medium (standard TA) |
| **Macro (base)** | 23 | Medium (economic levels) |
| **Macro (revisions)** | 115 | **High** (trader view) |
| **Macro (composites)** | 26 | **High** (factorized) |
| **Calendar** | ~8 | Medium (time patterns) |
| **Instrument Static** | ~20 | **High** (structural) |
| **Liquidity Dynamic** | ~8 | **High** (microstructure) |
| **Total** | **~220** | |

**Improvements**:
- +26 factorized composites (credit spreads, regimes, liquidity)
- +20 instrument-level factors (duration, sector, betas)
- +8 dynamic liquidity metrics (bid-ask, depth, impact)
- **Much higher signal density** vs. raw series

---

## 🎯 Remaining Work Summary

### High Priority (Before Training)

1. **Complete Macro Composites Integration** (2 hours):
   - Add to FeatureConfig
   - Wire into pipeline builder
   - Test end-to-end

2. **Add Dataset Validation** (3 hours):
   - Macro coverage validator
   - Integration with dataset build
   - Cache priming verification

### Medium Priority (Enhances Model Quality)

3. **Instrument Metadata** (4 hours):
   - Create metadata dataclass
   - Curate data for 95 instruments
   - Add static covariate transform

4. **Dynamic Liquidity Features** (2 hours):
   - Extract from L2 microstructure
   - Add to time-varying inputs
   - Test parity

### Low Priority (Future Enhancements)

5. **Advanced Technical Features**:
   - Multi-horizon realized vol
   - Drawdown metrics
   - Regime detection
   - Pattern counts

6. **Feature Selection/Importance**:
   - SHAP analysis
   - Permutation importance
   - Automated feature pruning

---

## ✅ Ready for Training

**Can proceed with dataset build using current implementation**:
- ✅ 95 instruments
- ✅ 23 macro series with 115 revision features
- ✅ Technical indicators
- ✅ Calendar features
- ✅ Training/inference parity

**Will benefit from completing Codex #2-4 before production deployment**:
- Composites add signal density
- Validation prevents silent failures
- Instrument covariates improve TFT modeling

**Recommended path**:
1. Build dataset with current features (test infrastructure)
2. Train baseline TFT model (validate pipeline)
3. Add composites + validation (enhance quality)
4. Retrain with full feature stack (production model)
