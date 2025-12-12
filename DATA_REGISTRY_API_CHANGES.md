# DataRegistry API Changes: Facade Pattern Migration

## EXECUTIVE SUMMARY

The DataRegistry has undergone a significant architectural refactoring from a monolithic "god class" pattern to a **facade + component pattern**. This migration aims to improve maintainability, testability, and follows the project's TDD decomposition standards.

**Key Change**: The facade provides backward compatibility at the public API level, but internal storage access patterns have changed.

---

## ARCHITECTURE CHANGE OVERVIEW

### BEFORE (Monolithic DataRegistry)
```python
# Single god class with all responsibilities
class DataRegistry:
    def __init__(self, registry_path, ...):
        self._manifests = {}        # Direct attribute storage
        self._contracts = {}
        self._events = []
        self._watermarks = {}
        self._lineage = []
    
    def register_dataset(self, manifest) -> str: ...
    def update_watermark(self, ...) -> None: ...
    def get_watermark(...) -> Watermark | None: ...
    def iter_watermarks(...) -> Iterator[Watermark]: ...
    # ... 15+ other methods in single 1900-line class
```

### AFTER (Facade + Components Pattern)
```python
# Facade delegates to specialized components
class DataRegistryFacade:
    def __init__(self, registry_path, ...):
        self._persistence = DataPersistenceComponent(...)
        self._manifest_manager = ManifestManagerComponent(...)
        self._event_emission = EventEmissionComponent(...)
        self._watermark_manager = WatermarkManagerComponent(...)
        self._lineage_tracker = LineageTrackerComponent(...)
    
    def register_dataset(self, manifest) -> str:
        return self._manifest_manager.register_dataset(manifest)
    
    def update_watermark(...) -> None:
        self._watermark_manager.update_watermark(...)
    
    # ... clean delegation pattern
```

**Feature Flag**: `ML_USE_LEGACY_DATA_REGISTRY`
- Set to `"1"` to use legacy DataRegistry (backward compatibility)
- Default (`"0"` or unset): Use DataRegistryFacade (new implementation)

---

## PROBLEM PATTERNS & SOLUTIONS

### Pattern 1: Direct Private Attribute Access

**❌ BROKEN CODE** (accessing `_manifests` directly):
```python
# This will FAIL with new facade
registry = create_data_registry(Path("/tmp"))
manifests = registry._manifests  # AttributeError: DataRegistryFacade has no attribute '_manifests'
```

**✅ CORRECT APPROACH** (using public methods):
```python
# Use the public API - works with both legacy and facade
registry = create_data_registry(Path("/tmp"))
manifests = registry.list_manifests()  # Returns list[DatasetManifest]

# Or get a specific one
manifest = registry.get_manifest("my_dataset")
```

**Key Point**: The facade's internal storage is delegated to `DataPersistenceComponent`. To access raw state for testing:
```python
# For testing/debugging, access through the persistence component
if isinstance(registry, DataRegistryFacade):
    manifests_dict = registry._persistence._manifests
    events = registry._persistence._events
    watermarks = registry._persistence._watermarks
```

---

### Pattern 2: Watermark Methods Returning None

**❌ FAILURE PATTERN** (assumption of false positives):
```python
watermark = registry.get_watermark("dataset", "EUR/USD", Source.LIVE)
assert watermark is not None  # May FAIL if watermark not initialized first
```

**Root Cause**: Watermarks are lazily created. The method returns `None` if:
1. No `update_watermark()` call has been made yet, OR
2. For PostgreSQL backend, the database session fails (missing `RuntimeError`)

**✅ CORRECT APPROACH** (defensive check):
```python
# Always check for None
watermark = registry.get_watermark("dataset", "EUR/USD", Source.LIVE)
if watermark is not None:
    print(f"Last success: {watermark.last_success_ns}")
else:
    # Initialize a watermark first
    registry.update_watermark(
        dataset_id="dataset",
        instrument_id="EUR/USD",
        source=Source.LIVE,
        last_success_ns=1000000000000000000,
        count=0,
        completeness_pct=0.0,
    )
    watermark = registry.get_watermark("dataset", "EUR/USD", Source.LIVE)
    assert watermark is not None
```

---

### Pattern 3: PostgreSQL Backend Database Session Failures

**❌ BROKEN CODE** (missing session check):
```python
# May raise: RuntimeError: Failed to get database session
manifest = registry.get_manifest("my_dataset")
```

**Root Causes** (PostgreSQL only):
- Database connection pool exhausted
- PostgreSQL service unavailable
- Invalid connection credentials
- Network timeout

**✅ CORRECT APPROACH** (with error handling):
```python
from ml.registry.persistence import BackendType

try:
    manifest = registry.get_manifest("my_dataset")
except RuntimeError as e:
    if "Failed to get database session" in str(e):
        logger.error("Database unavailable, check PostgreSQL connection", exc_info=True)
        # Fallback to cache if available
        if hasattr(registry, '_manifests') and "my_dataset" in registry._manifests:
            manifest = registry._manifests["my_dataset"]
        else:
            raise
    else:
        raise
```

**Prevention**: Use JSON backend for tests, PostgreSQL for production:
```python
# For testing
config = PersistenceConfig(
    backend=BackendType.JSON,
    json_path=Path("/tmp/test_registry"),
)

# For production
config = PersistenceConfig(
    backend=BackendType.POSTGRES,
    db_url="postgresql://...",
)
```

---

## INTERNAL STORAGE STRUCTURE

### DataPersistenceComponent Attributes

The actual state is now stored in `DataPersistenceComponent`:

```python
class DataPersistenceComponent:
    def __init__(self, ...):
        self._manifests: dict[str, DatasetManifest] = {}
        self._contracts: dict[str, DataContract] = {}
        self._events: list[dict[str, Any]] = []
        self._watermarks: dict[str, Watermark] = {}    # Key: "dataset_id:instrument_id:source"
        self._lineage: list[dict[str, Any]] = []
        self._lock = threading.RLock()  # Thread-safe access
        self.backend: BackendType  # JSON or POSTGRES
```

### Key Access Paths

```python
# Via DataRegistryFacade (public API)
facade = create_data_registry(Path("/tmp"))

# Public API (RECOMMENDED)
manifests = facade.list_manifests()
manifest = facade.get_manifest("dataset_id")
watermark = facade.get_watermark("dataset_id", "EUR/USD", Source.LIVE)
events = []
for event in facade._event_emission.iter_events():  # No public iter_events() yet
    events.append(event)

# For testing: Access persistence directly
persistence = facade._persistence
manifests_dict = persistence._manifests  # dict[str, DatasetManifest]
events_list = persistence._events        # list[dict[str, Any]]
watermarks_dict = persistence._watermarks  # dict[str, Watermark]
lineage_list = persistence._lineage      # list[dict[str, Any]]

# Thread-safe access (for long operations)
with persistence._lock:
    for dataset_id, manifest in persistence._manifests.items():
        # Process safely
        pass
```

### Watermark Key Format

Watermarks use a composite key:
```python
watermark_key = f"{dataset_id}:{instrument_id}:{source_value}"
# Example: "bars_eurusd_1m:EUR/USD:live"

# Access pattern
watermark = persistence._watermarks.get(watermark_key)
```

---

## MIGRATION GUIDE FOR TEST CODE

### Issue #1: Tests Accessing `registry._manifests`

**BEFORE (Legacy DataRegistry)**:
```python
legacy_registry = DataRegistry(Path("/tmp"), ...)
legacy_manifests = legacy_registry._manifests
assert len(legacy_manifests) == 1
```

**AFTER (DataRegistryFacade)**:
```python
facade = create_data_registry(Path("/tmp"))

# Option A: Use public API (PREFERRED)
manifests = facade.list_manifests()
assert len(manifests) == 1

# Option B: For detailed testing, access persistence
if isinstance(facade, DataRegistryFacade):
    manifests_dict = facade._persistence._manifests
    assert len(manifests_dict) == 1
```

### Issue #2: Parity Tests Comparing Internal State

**BEFORE (Legacy vs Legacy)**:
```python
legacy1_events = legacy_registry._events
legacy2_events = another_registry._events
assert len(legacy1_events) == len(legacy2_events)
```

**AFTER (Legacy vs Facade)**:
```python
legacy_events = legacy_registry._events
facade_events = facade._persistence._events

# Both now return list[dict[str, Any]]
assert len(legacy_events) == len(facade_events)
```

### Issue #3: Watermark Tests Returning None

**BEFORE (Direct Access)**:
```python
registry.update_watermark("dataset", "EUR/USD", Source.LIVE, 1000000000, 100, 99.5)
wm = registry.get_watermark("dataset", "EUR/USD", Source.LIVE)
assert wm is not None  # Always true if update_watermark succeeded
assert wm.completeness_pct == 99.5
```

**AFTER (Same API, but defensive checks)**:
```python
# Register dataset first (for PostgreSQL compatibility)
registry.register_dataset(manifest)

# Update watermark
registry.update_watermark("dataset", "EUR/USD", Source.LIVE, 1000000000, 100, 99.5)

# Get and verify
wm = registry.get_watermark("dataset", "EUR/USD", Source.LIVE)
assert wm is not None, "Watermark not found - may indicate database issue"
assert wm.completeness_pct == 99.5
```

---

## WATERMARK BEHAVIOR DETAILS

### When `get_watermark()` Returns None

| Scenario | Returns | Notes |
|----------|---------|-------|
| Never updated | `None` | Watermark doesn't exist yet |
| Updated once | `Watermark` | Subsequent calls return it |
| JSON backend | Always found | Stored in-memory `_watermarks` dict |
| PostgreSQL, no DB | Raises `RuntimeError` | "Failed to get database session" |
| PostgreSQL, no row | `None` | Entry doesn't exist in table |
| Queried again | Cached locally | In-memory cache within `_lock` |

### Update Watermark Behavior

```python
# Creates if not exists, updates if exists
registry.update_watermark(
    dataset_id="bars_eurusd_1m",
    instrument_id="EUR/USD",
    source=Source.LIVE,
    last_success_ns=1234567890000000000,
    count=1000,
    completeness_pct=99.5,
)

# For JSON backend: writes immediately (or batched)
# For PostgreSQL: calls SQL function update_watermark()
#   - If function missing, raises IntegrityError
#   - If table missing, raises ProgrammingError
```

### Iterator Behavior

```python
# iter_watermarks() with filters
for wm in registry.iter_watermarks(
    dataset_id="bars_eurusd_1m",
    instrument_id="EUR/USD",
    source=Source.LIVE,
    limit=10,
):
    # Yields Watermark objects matching filters
    # For JSON: filters in-memory, then sorts by updated_at DESC
    # For PostgreSQL: queries table with WHERE clause, sorts DESC
```

---

## COMPONENT ARCHITECTURE

### Five Components Extracted from God Class

```
DataRegistryFacade
├── DataPersistenceComponent          ← Storage & serialization
│   ├── _manifests: dict[str, DatasetManifest]
│   ├── _contracts: dict[str, DataContract]
│   ├── _events: list[dict]
│   ├── _watermarks: dict[str, Watermark]
│   ├── _lineage: list[dict]
│   └── _lock: RLock
│
├── ManifestManagerComponent          ← Dataset registration/updates
│   ├── register_dataset()
│   ├── get_manifest()
│   ├── list_manifests()
│   └── deprecate()
│
├── EventEmissionComponent            ← Event recording
│   ├── emit_event()
│   └── iter_events() [not public]
│
├── WatermarkManagerComponent         ← Progress tracking
│   ├── update_watermark()
│   ├── get_watermark()
│   └── iter_watermarks()
│
└── LineageTrackerComponent           ← Dataset lineage
    ├── link_lineage()
    ├── iter_lineage()
    └── pipeline_signature operations
```

---

## BACKWARD COMPATIBILITY

### Public API is 100% Compatible

All public methods have identical signatures:
```python
# Works with both DataRegistry and DataRegistryFacade
registry.register_dataset(manifest) -> str
registry.get_manifest(dataset_id) -> DatasetManifest
registry.update_watermark(...) -> None
registry.get_watermark(...) -> Watermark | None
registry.iter_watermarks(...) -> Iterator[Watermark]
registry.emit_event(...) -> None
registry.link_lineage(...) -> None
registry.iter_lineage(...) -> Iterator[DatasetLineageRecord]
registry.list_manifests() -> list[DatasetManifest]
registry.get_contract(dataset_id) -> DataContract
registry.deprecate(dataset_id) -> None
registry.flush() -> None
```

### Exceptions to Watch

Only private attribute access breaks:
```python
# ❌ BREAKS with facade
registry._manifests
registry._events
registry._watermarks
registry._contracts
registry._lineage

# ✅ WORKS with both
registry.list_manifests()
registry.get_manifest()
# ... all public methods
```

---

## TESTING BEST PRACTICES

### Setup
```python
import pytest
from pathlib import Path
from ml.registry import create_data_registry, PersistenceConfig, BackendType

@pytest.fixture
def registry_json(tmp_path: Path):
    """Use JSON for fast, isolated tests."""
    config = PersistenceConfig(
        backend=BackendType.JSON,
        json_path=tmp_path / "registry",
    )
    return create_data_registry(tmp_path / "registry", persistence_config=config)

@pytest.fixture
def registry_postgres(tmp_path: Path):
    """Use PostgreSQL for integration tests (slow, requires DB)."""
    config = PersistenceConfig(
        backend=BackendType.POSTGRES,
        db_url=os.environ["ML_POSTGRES_URL"],  # Must be set
    )
    return create_data_registry(tmp_path / "registry", persistence_config=config)
```

### Test Pattern: Manifest Operations
```python
def test_register_and_retrieve(registry_json):
    manifest = DatasetManifest(
        dataset_id="test_dataset",
        dataset_type=DatasetType.BARS,
        storage_kind=StorageKind.PARQUET,
        location="/data/bars",
        # ... required fields
    )
    
    # Register
    dataset_id = registry_json.register_dataset(manifest)
    assert dataset_id == "test_dataset"
    
    # Retrieve via public API
    retrieved = registry_json.get_manifest("test_dataset")
    assert retrieved.dataset_type == DatasetType.BARS
    
    # List via public API
    all_manifests = registry_json.list_manifests()
    assert len(all_manifests) >= 1
    assert any(m.dataset_id == "test_dataset" for m in all_manifests)
```

### Test Pattern: Watermark Operations
```python
def test_watermark_lifecycle(registry_json):
    # Register dataset first
    registry_json.register_dataset(manifest)
    
    # Initially no watermark
    wm = registry_json.get_watermark("test_dataset", "EUR/USD", Source.LIVE)
    assert wm is None
    
    # Update creates watermark
    registry_json.update_watermark(
        dataset_id="test_dataset",
        instrument_id="EUR/USD",
        source=Source.LIVE,
        last_success_ns=1000000000,
        count=100,
        completeness_pct=99.5,
    )
    
    # Now it exists
    wm = registry_json.get_watermark("test_dataset", "EUR/USD", Source.LIVE)
    assert wm is not None
    assert wm.completeness_pct == 99.5
    
    # Iterate
    all_wms = list(registry_json.iter_watermarks())
    assert len(all_wms) >= 1
```

### Test Pattern: Parity Tests (Legacy vs Facade)
```python
def test_parity_register_dataset(legacy_registry, facade_registry, manifest):
    legacy_id = legacy_registry.register_dataset(manifest)
    facade_id = facade_registry.register_dataset(manifest)
    
    assert legacy_id == facade_id
    
    # Compare via public API
    legacy_retrieved = legacy_registry.get_manifest(manifest.dataset_id)
    facade_retrieved = facade_registry.get_manifest(manifest.dataset_id)
    
    assert legacy_retrieved.dataset_type == facade_retrieved.dataset_type
    assert legacy_retrieved.version == facade_retrieved.version
    
    # If testing internal state (not recommended):
    if isinstance(facade_registry, DataRegistryFacade):
        legacy_manifests = legacy_registry._manifests
        facade_manifests = facade_registry._persistence._manifests
        assert len(legacy_manifests) == len(facade_manifests)
```

---

## TROUBLESHOOTING

### Problem: `AttributeError: 'DataRegistryFacade' object has no attribute '_manifests'`

**Solution**: Use public API instead
```python
# ❌ WRONG
manifests = registry._manifests

# ✅ CORRECT
manifests = registry.list_manifests()
```

### Problem: `assert None is not None` in watermark tests

**Solution**: Initialize watermark first
```python
# Register dataset
registry.register_dataset(manifest)

# Update watermark (creates it if not exists)
registry.update_watermark(
    dataset_id=manifest.dataset_id,
    instrument_id="EUR/USD",
    source=Source.LIVE,
    last_success_ns=1000000000,
    count=0,
    completeness_pct=0.0,
)

# Now get returns non-None
wm = registry.get_watermark(manifest.dataset_id, "EUR/USD", Source.LIVE)
assert wm is not None
```

### Problem: `RuntimeError: Failed to get database session`

**Solution**: Check PostgreSQL connectivity, use JSON backend for tests
```python
# For tests: use JSON
config = PersistenceConfig(
    backend=BackendType.JSON,
    json_path=tmp_path / "registry",
)

# For production: ensure PostgreSQL is running
# Check connection string in environment variable
```

### Problem: Events/Watermarks/Lineage not persisting

**Solution**: Call `registry.flush()` to force immediate save (JSON backend)
```python
registry.emit_event(...)
registry.update_watermark(...)
registry.link_lineage(...)

# Force save for immediate persistence
registry.flush()
```

---

## IMPLEMENTATION DETAILS FOR CONTRIBUTORS

### Adding New Methods to DataRegistry

**Old Pattern** (single class):
```python
class DataRegistry:
    def my_new_operation(self):
        # ... implementation
```

**New Pattern** (facade + component):
```python
# 1. Add to the appropriate component (e.g., ManifestManagerComponent)
class ManifestManagerComponent:
    def my_new_operation(self):
        # ... implementation using self._persistence._manifests

# 2. Add delegation to facade
class DataRegistryFacade:
    def my_new_operation(self):
        return self._manifest_manager.my_new_operation()
```

### Component Responsibilities

| Component | Responsibility | Public Methods |
|-----------|---|---|
| DataPersistenceComponent | Storage, serialization, thread-safety | `_load_registry()`, `_save_registry()`, `flush()` |
| ManifestManagerComponent | Dataset registration, listing, contracts | `register_dataset()`, `get_manifest()`, `list_manifests()`, etc. |
| EventEmissionComponent | Event recording | `emit_event()` |
| WatermarkManagerComponent | Progress tracking | `update_watermark()`, `get_watermark()`, `iter_watermarks()` |
| LineageTrackerComponent | Lineage tracking | `link_lineage()`, `iter_lineage()`, signature operations |

---

## SUMMARY: QUICK REFERENCE

| Aspect | Details |
|--------|---------|
| **Public API** | 100% backward compatible |
| **Private attributes** | Moved to `DataPersistenceComponent` |
| **Access pattern** | Use public methods; avoid `registry._*` attributes |
| **Watermarks** | Return `None` if not yet updated; check before use |
| **Database sessions** | PostgreSQL may raise `RuntimeError` on connection loss |
| **Feature flag** | `ML_USE_LEGACY_DATA_REGISTRY=1` for backward compatibility |
| **Thread-safety** | All operations protected by `persistence._lock` |
| **JSON backend** | Use for tests (fast, no DB required) |
| **PostgreSQL backend** | Use for production (persistent, concurrent) |
| **Persistence** | Call `registry.flush()` to force immediate save |
