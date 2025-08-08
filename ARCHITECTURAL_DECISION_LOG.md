# Architectural Decision Log

## Date: 2025-08-07

### Decision: Clean up engineering_enhanced.py

**Context:**
The file `ml/features/engineering_enhanced.py` was explicitly marked as an "EXAMPLE implementation" showing migration approach (lines 15-21, 457-494). It contained duplicate feature engineering logic (microstructure, trade flow) that already existed in the main `engineering.py` file.

**Current Situation:**

- File was marked as example: "NOTE: This is an EXAMPLE implementation showing the migration approach"
- No other code depended on it (grep found no imports)
- Main `engineering.py` already had all functionality including:
  - `include_microstructure` and `include_trade_flow` config options
  - Implementation methods for both batch and online feature calculation
  - The same feature types (spreads, imbalance, VWAP, etc.)
- File was causing confusion by duplicating logic

**Decision:**
Move the file to the examples directory as a demonstration of how to extend feature engineering.

**Implementation:**

1. Created `/home/nate/projects/nautilus_trader/examples/ml/` directory
2. Moved content to `examples/ml/feature_engineering_extension_example.py` with:
   - Clear documentation that it's an example
   - Renamed class to `CustomFeatureEngineer` to avoid confusion
   - Added custom volatility features as examples
   - Included demonstration script showing usage patterns
   - Fixed import issues and formatting
3. Removed original `ml/features/engineering_enhanced.py`
4. Updated all documentation references:
   - `ml/training/IMPLEMENTATION_GUIDE.md`
   - `ml/training/XGBOOST_MIGRATION_PLAN.md`
   - `FEATURE_PARITY_VALIDATION_REPORT.md`

**Benefits:**

- Reduces confusion by removing duplicate code from main codebase
- Maintains educational value as an example
- Clear separation between production code and examples
- Follows Nautilus Trader convention of keeping examples in examples/ directory

**Files Modified:**

- Deleted: `ml/features/engineering_enhanced.py`
- Created: `examples/ml/feature_engineering_extension_example.py`
- Updated: 3 documentation files to reference the correct module

**Validation:**

- No code dependencies on the removed file
- Example file passes all linting checks (ruff)
- Follows Nautilus Trader coding standards (American English, 4 spaces, copyright header)
