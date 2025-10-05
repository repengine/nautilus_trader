# ML Tests Context Documentation

**Version**: 4.2
**Last Updated**: 2025-09-05
**Status**: Optimized and parallel-ready test infrastructure with DB-safe scoping and two‑phase execution

## ⚠️ Important Database Requirement

**The ML system requires PostgreSQL.** The SQL migrations use PostgreSQL-specific features (partitioning, PL/pgSQL functions, triggers) that are incompatible with SQLite. This is a fundamental architectural requirement.

For speed and stability under parallel execution, DB‑heavy tests run with class/module‑scoped cleanup and integration tests run serially.

## Database & Environment Setup

Tests read database settings from environment variables, with `DATABASE_URL` as the primary source. The pytest configuration auto‑loads an `.env` file from `ml/tests/.env` if present.

Recognized variables (in order of importance):

- `DATABASE_URL` — primary connection string used by tests and fixtures
- `ML_DATABASE_URL` — optional alias some tools/scripts read (mirrors `DATABASE_URL`)
- `NAUTILUS_REGISTRY_DB_URL` — optional alias for registry helpers (mirrors `DATABASE_URL`)

Canonical local configuration (auto‑loaded):

```
# File: ml/tests/.env
DATABASE_URL=postgresql://postgres:postgres@localhost:5434/nautilus_test
ML_DATABASE_URL=postgresql://postgres:postgres@localhost:5434/nautilus_test
NAUTILUS_REGISTRY_DB_URL=postgresql://postgres:postgres@localhost:5434/nautilus_test
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5434/nautilus_test
```

Virtualenv auto‑loads `.env.local` at activation time (see `.venv/bin/activate`). Keep it aligned:

```
# File: .env.local
export DATABASE_URL="postgresql://postgres:postgres@localhost:5434/nautilus_test"
export ML_DATABASE_URL="$DATABASE_URL"
export NAUTILUS_REGISTRY_DB_URL="$DATABASE_URL"
export TEST_DATABASE_URL="$DATABASE_URL"
```

Notes on ports and compose setups:

- Test compose (simple Postgres): `.docker/docker-compose.yml` exposes host port `5432`.
  - Start: `make docker-up-test`
  - Wait: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nautilus uv run --active --no-sync python tools/wait_for_postgres.py`
- ML stack compose: `ml/deployment/docker-compose.yml` maps host port `5433` → container `5432`.
  - Start: `docker compose -f ml/deployment/docker-compose.yml up -d postgres`
  - Use: `DATABASE_URL=postgresql://postgres:postgres@localhost:5433/nautilus`

One‑off overrides (without editing files):

```
DATABASE_URL=postgresql://postgres:postgres@localhost:5434/nautilus_test pytest ml/tests -q
```

## Executive Summary

This document summarizes the ML testing architecture and conventions. The test infrastructure has been consolidated into a single, comprehensive `conftest.py` that uses the `EngineManager` singleton for proper connection pooling. Multiple testing approaches (property-based, metamorphic, contract, and pairwise) ensure thorough coverage while minimizing test count.

### Current State (September 2025)

- **Infrastructure**: Consolidated `ml/tests/conftest.py` with session‑scoped engine and class/module‑scoped DB cleanup fixtures.
- **Connection Management**: EngineManager prevents pool exhaustion (2 base + 3 overflow) and reuses engines per URL.
- **Parallel Readiness**: Non‑integration tests are safe in parallel; integration tests run serially to avoid DDL/DML deadlocks.
- **Performance Stability**: Benchmarks accept optional relax factor `ML_BENCH_RELAX` for CI variance.
- **Testing Approaches**: Property‑based (Hypothesis), metamorphic, contract (Pandera), pairwise.
- **Markers**: `database`, `serial`, and `integration` used consistently for optimal execution strategies.

## Test Infrastructure

### Consolidated Configuration & Fixtures

The ML test configuration is centralized in `ml/tests/conftest.py`. It aggregates
shared fixtures from `ml/tests/fixtures/` (e.g., `integration.py`, `monitoring_collectors.py`,
`database_fixtures.py`, `mock_services.py`) so contributors and code agents have one
obvious entrypoint.

Highlights:

- Session‑scoped engine via `EngineManager` to avoid pool exhaustion
- Transaction‑isolated sessions for function‑scoped DB tests
- Automatic prototype marking and DB gating when PostgreSQL is unreachable
- Central serial marking for `ml/tests/integration/` during collection
- xdist grouping (`xdist_group('db')`) for DB/serial tests when xdist is active
- Per‑scope TRUNCATE with `TEST_DB_SKIP_TRUNCATE=1` to disable per‑test cleanup

```python
# Session-scoped database engine (prevents connection exhaustion)
@pytest.fixture(scope="session")
def database_engine() -> Generator[Engine, None, None]:
    engine = EngineManager.get_engine(
        DATABASE_URL,
        pool_size=2,  # Conservative for tests
        max_overflow=3,  # Limited overflow
        pool_pre_ping=True,  # Test connections
        pool_recycle=300,  # 5-minute recycle
    )
    yield engine
    EngineManager.dispose_all()

# Transaction-isolated test sessions
@pytest.fixture
def database_session(database_session_factory):
    """Isolated session with automatic rollback"""
    # Uses nested transactions for complete isolation
```

### DB Cleanup Scopes & Hypothesis Profiles

Three Hypothesis profiles are available (selected via `HYPOTHESIS_PROFILE`):

- **CI Profile**: Fast (50 examples, 5s deadline, deterministic)
- **Dev Profile**: Thorough (200 examples, no deadline)
- **Debug Profile**: Minimal (10 examples, verbose output)

## Testing Approaches

### 1. Property-Based Testing (`property/`)

Using Hypothesis to verify invariants:

```python
@given(
    instrument_id=instrument_ids(),
    features=feature_values(),
    ts_events=st.lists(nanosecond_timestamps(), min_size=1, unique=True)
)
def test_timestamp_monotonicity_invariant(self, ...):
    """Timestamps must always increase monotonically"""
```

Key invariants tested:

- Timestamp monotonicity
- Feature immutability after write
- Partition consistency
- Data integrity across operations

### 2. Metamorphic Testing (`metamorphic/`)

Testing relationships under controlled transformations:

```python
def test_price_scaling_invariance(self):
    """Returns should be unchanged when prices are scaled"""
    scaled_features = engineer.compute_features(scaled_bars)
    np.testing.assert_allclose(
        original_features['returns'],
        scaled_features['returns']
    )
```

Metamorphic relations tested:

- Price scaling invariance
- Time reversal properties
- Noise addition robustness

### 3. Contract Testing (`contracts/`)

Using Pandera for schema validation:

```python
class FeatureInputSchema(pa.DataFrameModel):
    instrument_id: Series[str] = pa.Field()
    ts_event: Series[int] = pa.Field(ge=0)
    ts_init: Series[int] = pa.Field(ge=0)
    feature_values: Series[object] = pa.Field()

    @pa.check("ts_event")
    def ts_event_monotonic(cls, series):
        return series.is_monotonic_increasing
```

### 4. Pairwise Testing (`combinatorial/`)

Reducing combinatorial explosion with AllPairs:

```python
# 8,748 possible combinations → 15 test cases (99.8% reduction)
pairwise_configs = list(AllPairs([
    return_periods, momentum_periods, volume_periods,
    volatility_windows, use_log_returns, detrend_returns
]))
```

## Test Organization

```
ml/tests/
├── conftest.py              # Consolidated configuration
├── test_smoke.py           # Quick validation tests
├── property/               # Property-based tests
├── metamorphic/           # Metamorphic relation tests
├── contracts/             # Schema contract tests
├── combinatorial/         # Pairwise combination tests
├── unit/                  # Unit tests by domain
│   ├── actors/
│   ├── stores/
│   ├── features/
│   └── strategies/
├── integration/           # Integration tests (serial)
├── e2e/                  # End-to-end tests
├── performance/          # Performance benchmarks
├── fixtures/             # Shared fixtures (imported by conftest.py)
│   ├── integration.py
│   ├── monitoring_collectors.py
│   ├── database_fixtures.py
│   └── mock_services.py
└── tools/                # Test utilities and analysis

```

## Running Tests

### Quick Validation

```bash
# Smoke tests - verify basic functionality
python -m pytest ml/tests/test_smoke.py -xvs

# Unit tests only (fast, mocked)
python -m pytest ml/tests/unit -x --tb=short

# Example scripts (manual, not pytest)
python ml/tests/examples/simple_feature_test.py
python ml/tests/examples/working_feature_test.py
python ml/tests/examples/reproduce_feature_parity_bug.py
```

### Property-Based Tests

```bash
# Run with CI profile (fast)
HYPOTHESIS_PROFILE=ci python -m pytest ml/tests/property -x

# Run with dev profile (thorough)
HYPOTHESIS_PROFILE=dev python -m pytest ml/tests/property
```

### Parallel + Serial Mix (Recommended)

Use the Makefile shortcut to run a fast, stable two‑phase ML test sequence:

```bash
# Parallelize non-integration tests; run integration tests serially
make pytest-ml
```

Equivalent manual invocation:

```bash
# 1) Non-integration in parallel
pytest ml -m "not integration" -n auto --dist=loadscope -q

# 2) Integration serial
pytest ml -m integration -n 1 -q
```

### Fast Dev Loop (Local)

```bash
# Fail fast, concise traces, parallelize what’s safe
HYPOTHESIS_PROFILE=ci \
pytest ml -m "not integration" -n auto --dist=loadscope -q -x --maxfail=1 --tb=short -ra
```

Tip: append `--durations=10` to surface slowest tests.

### Green Lane (DB-free, fast correctness)

```bash
make pytest-green
```

## Developer Tips

- Use `EngineManager.get_engine(...)` for all DB access to avoid pool exhaustion.
- Mark DDL/DB-heavy tests `serial`; parallelize with `-n auto --dist=loadscope` elsewhere.
- Profiles: `HYPOTHESIS_PROFILE=ci|dev|debug` to trade speed vs. depth.
- Monitor connections: add `connection_monitor` fixture to suspect tests.
- Default selection: pytest excludes `prototype` tests by default (marked during collection).
- DB readiness:
  - Start local DB: `make docker-up-test`
  - Wait/check: `make check-db` (uses current `DATABASE_URL`)
- Fast loop: `HYPOTHESIS_PROFILE=ci pytest ml -m "not integration" -n auto --dist=loadscope -q -x --maxfail=1 --tb=short -ra`

## Static Validation & Pattern Compliance

The ML layer ships a comprehensive, fast validation suite to prevent common pitfalls (hot‑path violations, insecure serialization, event/status/topic misuse, coupling, duplication).

- Pre‑commit hooks (always on for changed files):
  - `check_nautilus_patterns.py` (custom AST checker):
    - Hot path: forbids `open()`, network calls, pandas `DataFrame(...)`, and `fit(...)` in `on_*` handlers and actor paths
    - Security: forbids `pickle`/`joblib` in actors/strategies/deployment/inference
    - Events: enforces `EventStatus.<...>.value` (no raw strings)
    - Topics: requires `build_topic_for_stage(...)` in stores/actors (no `build_topic(...)`)
    - Metrics: forbids direct `prometheus_client` imports; use `ml.common.metrics_bootstrap`
    - Architecture: flags store instantiation inside actors; warns on “god classes”
  - `semgrep-ml` (Semgrep rules in `tools/semgrep/ml-rules.yml`): mirrors the above as a second line of defense.

- Manual/advisory checks (developer/CI):
  - Duplication hotspots: `python tools/duplication/check_duplication.py`
  - Architecture contracts: `lint-imports` (Import Linter; see `importlinter.ini`)
  - Complexity budgets: `xenon --max-absolute B --max-modules B --max-average B ml/`
  - Security sweep: `bandit -q -r ml -x ml/tests`
  - Dead code: `vulture ml --min-confidence 90 --exclude ml/tests/*`
  - SQL lint: `sqlfluff lint schema ml/stores/migrations`

- One‑shot suite (advisory):
  - `make validate-nautilus-patterns`

Notes

- Violations in pre‑commit hooks are errors for changed files; the advisory suite is non‑blocking but should be kept clean before PRs.
- These checks map to Roadmap acceptance gates and the Comprehensive Issue Checklist (e.g., C001–C007 duplication, C009–C014 architecture, EventStatus enforcement, hot‑path budgets).

### Performance Tests

```bash
# Benchmark hot path operations (optionally relax thresholds on CI)
ML_BENCH_RELAX=1.5 python -m pytest ml/tests/performance/test_ml_hot_path_benchmarks.py --benchmark-only
```

## Connection Management

### EngineManager Pattern

All database connections go through the EngineManager singleton:

```python
from ml.core.db_engine import EngineManager

# Get or create engine (reuses existing)
engine = EngineManager.get_engine(connection_string)

# Dispose specific engine
EngineManager.dispose_engine(connection_string)

# Dispose all engines (cleanup)
EngineManager.dispose_all()
```

### Preventing Connection Exhaustion & Deadlocks

1. **Session-scoped engine** reuses connections across tests.
2. **Conservative pooling**: 2 base + 3 overflow connections.
3. **Class/Module cleanup**: TRUNCATE once per class/module; per‑test cleanup suppressed via env.
4. **Transaction isolation**: Nested transactions for function‑scoped DB sessions.
5. **Parallel policy**:
   - Non‑integration tests → `-n auto --dist=loadscope`.
   - Integration tests → serial (`-n 1`).
   - DDL‑heavy tests (e.g., partition migrations) are marked `serial` for xdist safety.

## Common Patterns

### Mock Stores for Unit Tests

```python
@pytest.fixture
def mock_feature_store():
    mock_store = MagicMock()
    mock_store.write_features = MagicMock(return_value=True)
    mock_store.get_latest_features = MagicMock(return_value={})
    return mock_store
```

### Isolated SQLite for Hypothesis

```python
@pytest.fixture
def hypothesis_database_session():
    """In-memory SQLite for rapid property test generation"""
    engine = create_engine("sqlite:///:memory:", poolclass=NullPool)
    # ... setup and teardown
```

### Test Data Factories

```python
from ml.tests.fixtures.model_factory import create_test_model
from ml.tests.fixtures.mock_services import create_mock_fred_client

model = create_test_model("xgboost")
fred_client = create_mock_fred_client(test_data)
```

## Debugging Failed Tests

### Connection Issues

```bash
# Monitor PostgreSQL connections
watch -n1 "psql -c 'SELECT count(*) FROM pg_stat_activity;'"

# Check EngineManager pool status
python -c "from ml.core.db_engine import EngineManager; print(EngineManager.get_pool_status('...'))"
```

### Hypothesis Failures

```python
# Use debug profile for verbose output
HYPOTHESIS_PROFILE=debug python -m pytest failing_test.py -xvs

# Reproduce with seed
python -m pytest --hypothesis-seed=12345
```

### Performance Issues

```bash
# Profile test execution
python -m pytest --profile test_slow.py

# Benchmark specific operations
python -m pytest test_file.py::test_function --benchmark-only
```

## Best Practices

1. **Use appropriate fixtures**: Mock stores for unit tests, real stores for integration
2. **Leverage property testing**: Find edge cases automatically with Hypothesis
3. **Test contracts**: Validate data shapes with Pandera schemas
4. **Reduce combinations**: Use pairwise testing for configuration spaces
5. **Monitor connections**: Use connection_monitor for database-heavy tests
6. **Clean up properly**: Ensure all resources are released in teardown

## Known Issues and Workarounds

### PostgreSQL Required

- SQLite is not supported due to PostgreSQL-specific features
- Use Docker for local development if PostgreSQL not installed

### Parallel Test Execution

- Use `-n auto` for non‑integration tests only; keep integration serial to avoid DDL/DML contention.
- The Makefile target `pytest-ml` orchestrates this automatically.

### Memory Usage

- Hypothesis tests can consume significant memory
- Use smaller max_examples in CI environments

## Recent Improvements (September 2025)

### Test Marker Implementation

- Applied pytest markers to all 131 test files
- Database tests marked with `@pytest.mark.serial` to prevent connection exhaustion
- Parallel-safe tests marked for concurrent execution
- Created verification scripts to ensure marker compliance

### PostgreSQL Test Consolidation & Speedups

- Added class/module‑scoped cleanup fixtures to minimize TRUNCATE overhead.
- Cached schema initialization per engine URL.
- Ensured DDL tests are marked `serial` for xdist safety.
- Added `ML_BENCH_RELAX` variable for stable performance thresholds on CI.

- **DB cleanup scopes**:
  - `clean_postgres_db`: function‑scoped compatibility fixture (legacy tests).
  - `clean_postgres_db_class`: TRUNCATE once before/after a test class.
  - `clean_postgres_db_module`: TRUNCATE once before/after a module.
  - Per‑test TRUNCATE is suppressed when higher‑scope cleanup is active.

### Infrastructure Consolidation

- Unified multiple conftest files into single source of truth
- Implemented session-scoped database fixtures
- Added automatic cleanup to prevent connection leaks
- Integrated Hypothesis profiles for different test environments

## Future Improvements

1. **Test Coverage**: Increase from current ~80% to >90%
2. **Mutation Testing**: Add mutmut for test effectiveness validation
3. **Fuzz Testing**: Extend property tests with fuzzing strategies
4. **Performance Regression**: Automated benchmark comparisons
5. **Test Impact Analysis**: Run only affected tests on code changes

## Implementation Review Addendum

**Review Date**: 2025-09-12
**Reviewer**: Claude Code Analysis
**Focus**: Documentation accuracy vs. actual test implementation

### Executive Summary

The ML tests documentation contains several inaccuracies and inflated completion claims compared to the actual implementation. While the test infrastructure is substantial with 239 test files across multiple categories, the documentation overstates coverage and compliance with the Universal ML Architecture Patterns.

### Critical Documentation vs. Implementation Discrepancies

#### 1. Universal ML Architecture Pattern Testing

**Documentation Claims**:

- Complete implementation of 5 Universal ML Architecture Pattern validation
- Comprehensive `UniversalPatternValidator` with AST-based analysis (1,426 lines in universal_patterns_guide.md)
- Automated compliance checking for all ML actors

**Reality**:

- **MISSING**: No `UniversalPatternValidator` class found in codebase
- **MISSING**: No tests validating `Pattern 1-5` compliance found
- **MISSING**: No AST-based pattern compliance analysis
- **LIMITED**: Only basic `metrics_bootstrap` tests exist (`/home/nate/projects/nautilus_trader/ml/tests/unit/monitoring/test_metrics_bootstrap.py`)

**Impact**: Documentation claims sophisticated pattern validation that doesn't exist.

#### 2. Test Coverage Claims

**Documentation Claims**:

- "Current ~80% to >90%" test coverage
- "131 test files" with markers applied (line 428)
- "Comprehensive coverage with multiple testing approaches"

**Reality**:

- **ACTUAL**: 239 total Python test files found (significantly more than claimed 131)
- **NO EVIDENCE**: No coverage metrics validation found in codebase
- **UNVERIFIED**: Coverage percentage claims cannot be substantiated

**Impact**: Numbers don't match actual file count, coverage claims unsubstantiated.

#### 3. Performance Testing Reality

**Documentation Claims**:

- "P99 < 5ms for hot path operations"
- "Zero allocations in hot path after warmup"
- Comprehensive performance SLA validation

**Reality**:

- **PARTIAL**: Performance benchmarks exist (`/home/nate/projects/nautilus_trader/ml/tests/performance/test_ml_hot_path_benchmarks.py`)
- **GOOD**: Actual P99 measurement framework implemented
- **GOOD**: Allocation tracking via tracemalloc exists
- **CONCERN**: Only 1 performance test file found vs. comprehensive claims

**Impact**: Performance testing exists but may be less comprehensive than documented.

#### 4. Test Infrastructure Accuracy

**Documentation Claims**:

- Consolidated `conftest.py` with session-scoped fixtures
- "Applied pytest markers to all 131 test files"
- Comprehensive integration test separation

**Reality**:

- **ACCURATE**: `/home/nate/projects/nautilus_trader/ml/tests/conftest.py` exists and appears comprehensive
- **ACCURATE**: Test markers applied extensively (`@pytest.mark.database`, `@pytest.mark.serial`, `@pytest.mark.integration`)
- **GOOD**: Test structure matches documentation (property/, metamorphic/, contracts/, unit/, etc.)

**Impact**: Test infrastructure documentation is largely accurate.

#### 5. Testing Approach Implementation

**Documentation Claims**:

- Property-based testing with Hypothesis
- Metamorphic testing for transformations
- Contract testing with Pandera
- Pairwise combinatorial testing

**Reality**:

- **VERIFIED**: Property tests exist (16 files in `/home/nate/projects/nautilus_trader/ml/tests/property/`)
- **VERIFIED**: Metamorphic tests exist (7 files in `/home/nate/projects/nautilus_trader/ml/tests/metamorphic/`)
- **VERIFIED**: Contract tests exist (10 files in `/home/nate/projects/nautilus_trader/ml/tests/contracts/`)
- **LIMITED**: Only 2 combinatorial test files found vs. comprehensive claims

**Impact**: Testing approaches are implemented but less extensive than claimed.

### Specific File:Line Discrepancies

#### Universal Pattern Guide (`universal_patterns_guide.md`)

- **Lines 887-1313**: Entire `UniversalPatternValidator` class implementation **NOT FOUND** in codebase
- **Lines 976-1314**: `validate_actor_compliance`, `_validate_store_integration`, etc. methods **DO NOT EXIST**
- **Lines 1322-1425**: Integration test examples reference non-existent validation framework

#### Context Tests Documentation (`context_tests.md`)

- **Line 428**: "Applied pytest markers to all 131 test files" - **ACTUAL COUNT: 239 files**
- **Line 58**: "consolidated into a single, comprehensive conftest.py" - **ACCURATE**
- **Lines 100-103**: Hypothesis profiles documented - **VERIFIED in conftest.py**

### Critical Quality Issues from Validation Reports

Based on `/home/nate/projects/nautilus_trader/ml/tests/validation_reports/COMPREHENSIVE_ISSUE_CHECKLIST.md`:

- **127 total issues found** (28 critical, 46 high priority)
- **God classes**: BaseMLInferenceActor (1,922 lines), BaseMLStrategy (966 lines), BaseMLTrainer (800+ lines)
- **Architecture violations**: Extensive code duplication in store initialization (450+ lines across 4+ files)

### Recommendations for Documentation Accuracy

#### Immediate Corrections Required

1. **Remove Universal Pattern Validator claims** - Delete lines 887-1425 from `universal_patterns_guide.md` or clearly mark as "Future Implementation"
2. **Correct file counts** - Update "131 test files" to "239 test files" in line 428
3. **Qualify coverage claims** - Remove specific percentage claims without verification
4. **Add implementation status** - Clearly distinguish between implemented and planned features

#### Implementation Gaps to Address

1. **Missing Pattern Compliance Testing**: Implement actual Universal Pattern validation or remove claims
2. **Limited Performance Testing**: Expand beyond single benchmark file
3. **Missing Coverage Measurement**: Implement actual coverage tracking to validate claims

#### Documentation Integrity Standards

1. **Ground-truth validation**: All claims must be verifiable against actual implementation
2. **Clear status indicators**: Use ✅ (implemented), 🔄 (in progress), ❌ (not implemented)
3. **Regular accuracy audits**: Quarterly reviews to prevent documentation drift

### Conclusion

While the ML testing infrastructure is substantial and well-structured, the documentation significantly overstates completeness and capabilities. The core testing approaches are implemented, but claims about comprehensive pattern validation, specific coverage percentages, and sophisticated compliance checking are unsupported by the actual codebase.

**Priority Actions**:

1. Correct inflated completion claims in documentation
2. Remove or clearly mark theoretical/future implementations
3. Implement actual Universal Pattern compliance testing if desired
4. Establish regular documentation accuracy validation processes
