# Redundant Scripts Archive - September 2025

This directory contains redundant development scripts that were removed during the aggressive cleanup on 2025-09-03.

## Scripts Archived

### Test Scripts (Redundant with CI/Makefile)
- `test.sh` - Basic test runner (replaced by CI pipelines)
- `test-coverage.sh` - Coverage testing (handled by CI)  
- `test-examples.sh` - Examples testing (handled by CI)
- `test-performance.sh` - Performance testing (handled by CI)

### Utility Scripts (Redundant/Obsolete)
- `package-version.sh` - Version utility (redundant with pyproject.toml)
- `python-version.sh` - Python version check (redundant)
- `rust-toolchain.sh` - Rust setup (replaced by rust-toolchain.toml)
- `enhanced_collector_smoke.py` - Old smoke test (obsolete)

## Why These Were Removed

### 1. **Testing Scripts**
- All test scripts duplicated functionality already handled by:
  - CI pipelines in `.github/workflows/`
  - Makefile targets (`make test`, `make coverage`)
  - Direct pytest commands

### 2. **Version/Environment Scripts**  
- Python/Rust version management now handled by:
  - `pyproject.toml` for Python dependencies
  - `rust-toolchain.toml` for Rust toolchain
  - Modern dependency management tools (uv, cargo)

### 3. **Obsolete Utilities**
- Enhanced collector smoke tests replaced by proper unit tests
- Package version utilities redundant with poetry/setuptools

## Replaced Functionality

### Testing (CI/Makefile)
```bash
# OLD: Multiple test scripts
./scripts/test.sh
./scripts/test-coverage.sh
./scripts/test-examples.sh

# NEW: Unified commands
make test
make coverage
pytest
```

### Version Management (Configuration Files)
```bash
# OLD: Manual version scripts
./scripts/python-version.sh
./scripts/package-version.sh

# NEW: Declarative configuration
# pyproject.toml - Python dependencies
# rust-toolchain.toml - Rust toolchain
```

## Migration Notes

- **No functionality lost** - All capabilities preserved in CI/config
- **Better maintenance** - Single source of truth for each concern
- **Modern tooling** - Upgraded to current best practices

## Restoration

If any of these scripts are needed, they can be restored from this archive. However, it's recommended to use the modern alternatives listed above.

## Final Script Count
- **Before**: 22+ development scripts
- **After**: 10 core scripts (55% reduction)
- **Archived**: 8 redundant scripts safely preserved