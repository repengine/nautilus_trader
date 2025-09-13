# ML Tests Organization

## Directory Structure

```
ml/tests/
├── conftest.py                 # Main pytest configuration
├── conftest.py                 # Central fixtures & pytest config (aggregates fixtures/*)
├── test_smoke.py               # Quick smoke tests (keep in root for easy access)
├── README.md                   # Main test documentation
├── __init__.py                 # Package marker
│
├── unit/                       # Unit tests (isolated, fast)
│   ├── actors/                 # Actor tests
│   ├── config/                 # Configuration tests
│   ├── core/                   # Core functionality tests
│   ├── data/                   # Data handling tests
│   ├── deployment/             # Deployment tests
│   ├── features/               # Feature engineering tests
│   ├── models/                 # Model tests
│   ├── monitoring/             # Monitoring tests
│   ├── registry/               # Registry tests
│   ├── stores/                 # Store tests
│   ├── strategies/             # Strategy tests
│   └── training/               # Training tests
│
├── integration/                # Integration tests (cross-component)
│   ├── test_postgres_*.py     # PostgreSQL integration tests
│   ├── test_stores_*.py       # Store integration tests
│   ├── test_scheduler_*.py    # Scheduler integration tests
│   └── test_end_to_end_*.py   # End-to-end pipeline tests
│
├── property/                   # Property-based tests
│   └── test_store_invariants.py  # Store invariant tests
│
├── contracts/                  # Contract/behavioral tests
│   ├── test_actor_contracts.py
│   ├── test_strategy_contracts.py
│   └── test_store_schemas.py     # Schema validation tests
│
├── metamorphic/               # Metamorphic relationship tests
│   ├── test_feature_transforms.py
│   └── test_signal_predictions.py
│
├── combinatorial/             # Pairwise/combinatorial tests
│   └── test_config_combinations.py
│
├── e2e/                       # End-to-end tests
│   └── test_data_registry_e2e.py
│
├── performance/               # Performance benchmarks
│   └── test_ml_hot_path_benchmarks.py
│
├── examples/                  # Example tests for reference
│   └── test_parameterization_example.py
│
├── fixtures/                  # Test fixtures and utilities
│   ├── database_fixtures.py
│   ├── mock_services.py
│   └── dummy_model.py
│
├── tools/                     # Test analysis and maintenance tools
│   ├── analyze_test_redundancy.py
│   └── cleanup_redundant_tests.py
│
└── docs/                      # Test documentation
    ├── TESTING_STRATEGY.md    # Testing strategy guide
    ├── TEST_REDUNDANCY_REPORT.md  # Redundancy analysis results
    └── VALIDATION_REPORT.md   # Validation report
```

## Test Categories

### By Speed

- **Smoke** (`test_smoke.py`): < 1 second, basic validation
- **Unit** (`unit/`): < 5 seconds per file, isolated
- **Integration** (`integration/`): < 30 seconds per file, database required
- **E2E** (`e2e/`): > 30 seconds, full pipeline

### By Approach

- **Example-based**: Traditional specific test cases
- **Property-based** (`property/`): Hypothesis-generated test cases
- **Contract** (`contracts/`): Behavioral guarantees
- **Metamorphic** (`metamorphic/`): Relationship testing
- **Combinatorial** (`combinatorial/`): Pairwise parameter testing

## Running Tests

### Quick Validation

```bash
# Smoke tests only
pytest ml/tests/test_smoke.py -x

# Fast unit tests
pytest ml/tests/unit -x --ignore=ml/tests/unit/stores

# Property tests
pytest ml/tests/property -x
```

### Full Test Suite

```bash
# All tests (requires PostgreSQL)
pytest ml/tests -x

# With coverage
pytest ml/tests --cov=ml --cov-report=html
```

### Test Analysis

```bash
# Analyze for redundancy
python ml/tests/tools/analyze_test_redundancy.py

## Key Files

- `conftest.py`: Central pytest configuration and fixtures aggregator (imports from `fixtures/`)
- `fixtures/`: Shared fixture modules (integration, monitoring collectors, common builders)
- `test_smoke.py`: Quick validation tests (keep in root for `pytest ml/tests/test_smoke.py`)
- `README.md`: Detailed test documentation

## Notes

- PostgreSQL is required for integration and E2E tests
- Use `use_dummy_stores=True` in configs for unit tests to avoid DB connections
- Property tests provide better coverage than example tests
- Pairwise testing reduces combinatorial explosion
- See `docs/TESTING_STRATEGY.md` for detailed testing philosophy
