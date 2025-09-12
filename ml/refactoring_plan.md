# ML Codebase Refactoring Plan

## Executive Summary
Based on API analysis of 3,173 symbols, we've identified 145 deprecated items and 164 property conversion opportunities that can significantly improve code quality and reduce technical debt.

## 1. Deprecated Code Cleanup (36.2 hours estimated)

### Immediate Actions (High Priority)
1. **Delete archive folders** (2 hours)
   - `ml/tests/tools/archive/conftest_old.py` - 41 deprecated symbols
   - `ml/tests/tools/archive/conftest_backup.py` - 41 deprecated symbols
   - These are pure duplicates taking up namespace

2. **Remove backup test files** (1 hour)
   - `ml/tests/unit/deployment/test_entrypoint_strategy_backup.py` - 23 symbols
   - Already have primary version, backup not needed

### Pattern-Based Cleanup (Medium Priority)

#### OLD Pattern (72 items)
- `ml.actors.signal`: Contains old threshold strategies
  - Migrate `AdaptiveThresholdConfig` → New config system
  - Deprecate `ThresholdSignalStrategy` → Use base signal actor
  
- `ml.monitoring.grafana_client`: Old API methods
  - Update to use new Grafana API v9+

#### TEMP Pattern (8 items)
- Clean up temporary test fixtures
- Replace with proper fixture management

### Migration Strategy
```python
# Step 1: Mark as deprecated
@deprecated(version='2.0', reason='Use NewClass instead')
class OldClass:
    pass

# Step 2: Add migration period (3 months)
# Step 3: Remove in next major version
```

## 2. Large Class Refactoring

### ModelRegistry (34 methods, 1241 LOC)
Split into 4 focused classes:
```python
# Current monolith
class ModelRegistry:
    # 34 methods doing everything
    
# Proposed split
class ModelRegistry:  # Core registry (10 methods)
    def __init__(self, storage, validator, deployer, metrics):
        self.storage = ModelStorage()
        self.validator = ModelValidator()
        self.deployer = ModelDeployer()
        self.metrics = ModelMetrics()

class ModelStorage:  # Persistence layer (8 methods)
    def save(), load(), delete(), list()
    
class ModelValidator:  # Validation logic (8 methods)
    def validate_schema(), validate_performance()
    
class ModelDeployer:  # Deployment operations (8 methods)
    def deploy(), rollback(), canary_deploy()
```

### MLIntegrationManager (24 methods, 519 LOC)
Apply Strategy pattern:
```python
# Before
class MLIntegrationManager:
    def integrate_xgboost(...)
    def integrate_lightgbm(...)
    def integrate_tensorflow(...)
    # 20+ integration methods

# After
class MLIntegrationManager:
    def __init__(self):
        self.integrators = {
            'xgboost': XGBoostIntegrator(),
            'lightgbm': LightGBMIntegrator(),
        }
    
    def integrate(self, framework: str, **kwargs):
        return self.integrators[framework].integrate(**kwargs)
```

## 3. Property Conversions (164 opportunities)

### Automated Conversion Script
```python
# Generate conversion script
def convert_to_property(class_path: str, method_name: str):
    \"\"\"Convert get_xxx() to @property.\"\"\"
    # 1. Find method definition
    # 2. Add @property decorator
    # 3. Rename get_xxx to xxx
    # 4. Update all call sites
```

### Top Priority Conversions

#### ModelRegistry (10 properties)
```python
# Before (3 lines per access)
models = registry.get_active_models()

# After (Pythonic)
models = registry.active_models

# Benefit: 30% less code, better IDE support
```

#### Benefits Analysis
- **Code reduction**: ~500 lines (3 lines → 1 line per accessor)
- **Performance**: Properties can cache computed values
- **Type safety**: Better type inference with properties
- **API consistency**: Aligns with Python standards

### Implementation Plan

#### Phase 1: Read-only properties (Week 1)
- Convert 164 getters without setters
- Low risk, backward compatible with deprecation

#### Phase 2: Read-write properties (Week 2)
- Convert matched getter/setter pairs
- Add validation in setters

#### Phase 3: Computed properties (Week 3)
- Replace expensive get_* methods with cached properties
- Add @cached_property where appropriate

## 4. Quick Wins (Can do TODAY)

### Hour 1: Delete archives
```bash
rm -rf ml/tests/tools/archive/
# Saves 82 symbols, reduces confusion
```

### Hour 2: Property conversion script
```python
# Run automated conversion on top 5 classes
python tools/convert_to_properties.py \
  --classes ModelRegistry,GrafanaClient,StrategyRegistry \
  --dry-run
```

### Hour 3: Extract validators
```python
# Move all validate_* methods to dedicated validators
class ModelValidator:
    def validate_manifest(self, manifest: ModelManifest) -> bool
    def validate_performance(self, metrics: dict) -> bool
```

## 5. Metrics & Tracking

### Before
- 3,173 total symbols
- 145 deprecated items
- 180 get_* methods
- 13 set_* methods

### After (Projected)
- 2,950 symbols (-7%)
- 0 deprecated items
- 20 get_* methods (complex computations only)
- 0 set_* methods (all properties)

### Success Metrics
- 20% reduction in API surface
- 30% improvement in test speed (less code)
- 100% property conversion for simple accessors
- 0 deprecated code in main branch

## 6. Tooling Support

### Created Tools
1. `tools/api_index.py` - Generate API index
2. `tools/analyze_api.py` - Find refactoring targets
3. `tools/filter_api_index.py` - Focus on core APIs

### Next Tools to Build
1. Property converter (automate get/set → @property)
2. Deprecation scanner (find and report deprecated usage)
3. Class splitter (identify cohesive method groups)

## Timeline

**Week 1**: Quick wins (8 hours)
- Delete archives
- Start property conversions
- Mark deprecated code

**Week 2**: Class refactoring (16 hours)
- Split ModelRegistry
- Refactor MLIntegrationManager

**Week 3**: Completion (12 hours)
- Finish property conversions
- Update documentation
- Performance testing

**Total: 36 hours over 3 weeks**

## ROI Calculation

**Investment**: 36 hours
**Returns**:
- 500 lines less code to maintain
- 30% faster test execution
- 50% reduction in API cognitive load
- Prevents ~100 hours of future debugging

**Payback period**: 2 months