# PostgreSQL Test Fixtures Guide

## Core Fixtures Available

### Database Fixtures (from conftest.py)

```python
@pytest.fixture
def postgres_connection() -> str:
    """Returns PostgreSQL connection string."""
    return "postgresql://postgres:postgres@localhost:5432/nautilus_test"

@pytest.fixture
def test_database() -> TestDatabase:
    """Provides TestDatabase instance with automatic cleanup."""
    # Usage:
    # - test_database.engine - SQLAlchemy engine
    # - test_database.connection_string - connection URL
    # - test_database.get_session() - context manager for sessions

@pytest.fixture
def clean_postgres_db():
    """Ensures clean database state before and after test."""
    # Automatically truncates all tables
```

### Store Fixtures (to be used)

```python
@pytest.fixture
def feature_store(test_database) -> FeatureStore:
    """Provides initialized FeatureStore."""
    return FeatureStore(connection_string=test_database.connection_string)

@pytest.fixture  
def model_store(test_database) -> ModelStore:
    """Provides initialized ModelStore."""
    return ModelStore(connection_string=test_database.connection_string)

@pytest.fixture
def strategy_store(test_database) -> StrategyStore:
    """Provides initialized StrategyStore."""
    return StrategyStore(connection_string=test_database.connection_string)
```

## Usage Patterns

### Pattern 1: Using Database Fixture
```python
def test_something(test_database):
    # Access engine
    engine = test_database.engine
    
    # Get session
    with test_database.get_session() as session:
        # Use session for queries
        result = session.execute(text("SELECT 1"))
```

### Pattern 2: Using Store Fixtures
```python
def test_feature_store_operation(feature_store):
    # Store is already initialized with PostgreSQL
    feature_store.write_features(...)
    features = feature_store.read_features(...)
```

### Pattern 3: Clean Database Each Test
```python
@pytest.mark.usefixtures("clean_postgres_db")
def test_with_clean_db(test_database):
    # Database is guaranteed clean
    pass
```

### Pattern 4: Multiple Stores
```python
def test_integration(feature_store, model_store, strategy_store):
    # All stores share same database but are independent instances
    features = feature_store.compute_features(...)
    predictions = model_store.predict(features)
    signals = strategy_store.generate_signals(predictions)
```

## Best Practices

1. **Always use fixtures** - Never hardcode connection strings
2. **Use context managers** - For sessions and transactions
3. **Clean state** - Use clean_postgres_db for isolation
4. **Shared database** - All stores use same test database
5. **No SQLite** - PostgreSQL is the only supported database

## Anti-Patterns to Avoid

❌ **Don't do this:**
```python
# Hardcoded connection
store = FeatureStore("sqlite:///:memory:")

# Direct engine creation  
engine = create_engine("postgresql://...")

# Manual cleanup
session.execute("DELETE FROM ...")
```

✅ **Do this instead:**
```python
# Use fixtures
def test_store(feature_store):
    feature_store.write(...)

# Use test_database
def test_with_db(test_database):
    engine = test_database.engine
    
# Automatic cleanup
@pytest.mark.usefixtures("clean_postgres_db")
def test_isolated(test_database):
    pass
```