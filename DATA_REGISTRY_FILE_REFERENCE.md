# DataRegistry Architecture - File Reference Map

## Core Files Structure

```
ml/registry/
├── data_registry.py                    ← LEGACY class (still 1900 lines)
├── data_registry_facade.py             ← NEW facade (delegation layer)
├── data_registry_legacy.py             ← Backup of legacy impl
├── common/
│   ├── data_persistence.py             ← Storage & serialization
│   ├── manifest_manager.py             ← Dataset management
│   ├── event_emission.py               ← Event recording
│   ├── watermark_manager.py            ← Watermark tracking
│   └── lineage_tracker.py              ← Lineage management
├── persistence.py                      ← Backend abstraction (JSON/PostgreSQL)
├── dataclasses.py                      ← Type definitions
└── protocols.py                        ← Interface definitions
```

---

## File Details

### 1. `/ml/registry/data_registry.py` (LEGACY - 1900 lines)

**Current Status**: Still exists for backward compatibility

**Contains**:
- `Watermark` dataclass (lines 46-80)
- `DataRegistry` class (lines 82-1915)
  - All monolithic methods (register, update, deprecate, emit, etc.)
  - Direct `_manifests`, `_events`, `_watermarks`, `_lineage` storage
  - Thread-safe with `self._lock`

**Key Methods** (all work with legacy class):
```python
def register_dataset(manifest: DatasetManifest) -> str
def get_manifest(dataset_id: str) -> DatasetManifest
def list_manifests() -> list[DatasetManifest]
def update_watermark(...) -> None
def get_watermark(...) -> Watermark | None
def iter_watermarks(...) -> Iterator[Watermark]
def emit_event(...) -> None
def link_lineage(...) -> None
def iter_lineage(...) -> Iterator[DatasetLineageRecord]
```

**Private Attributes** (problematic access):
```python
self._manifests: dict[str, DatasetManifest]
self._contracts: dict[str, DataContract]
self._events: list[dict[str, Any]]
self._watermarks: dict[str, Watermark]
self._lineage: list[dict[str, Any]]
self._lock: threading.RLock()
```

---

### 2. `/ml/registry/data_registry_facade.py` (NEW - 670 lines)

**Current Status**: NEW facade implementation (recommended)

**Contains**:
- `DataRegistryFacade` class (lines 52-622)
  - Delegates to 5 components
  - Clean public API (same as legacy)
  - No direct `_manifests` on facade itself

**Key Methods** (same signatures as legacy):
```python
def register_dataset(manifest: DatasetManifest) -> str
def get_manifest(dataset_id: str) -> DatasetManifest
def list_manifests() -> list[DatasetManifest]
def update_watermark(...) -> None
def get_watermark(...) -> Watermark | None
def iter_watermarks(...) -> Iterator[Watermark]
def emit_event(...) -> None
def link_lineage(...) -> None
def iter_lineage(...) -> Iterator[DatasetLineageRecord]
```

**Component Delegation**:
```python
self._persistence = DataPersistenceComponent(...)         # Storage
self._manifest_manager = ManifestManagerComponent(...)    # Manifests
self._event_emission = EventEmissionComponent(...)        # Events
self._watermark_manager = WatermarkManagerComponent(...)  # Watermarks
self._lineage_tracker = LineageTrackerComponent(...)      # Lineage
```

**Factory Function** (lines 624-669):
```python
def create_data_registry(
    registry_path: Path,
    batch_save_interval: float = 0.1,
    persistence_config: PersistenceConfig | None = None,
) -> DataRegistryFacade:
    """
    Returns DataRegistryFacade (or legacy DataRegistry if ML_USE_LEGACY_DATA_REGISTRY=1)
    """
```

---

### 3. `/ml/registry/common/data_persistence.py` (NEW - 800+ lines)

**Current Status**: NEW component for storage operations

**Contains**:
- `DataPersistenceComponent` class (lines 42-500+)
  - Extracted from DataRegistry
  - Manages `_manifests`, `_events`, `_watermarks`, `_lineage`
  - Thread-safe with `self._lock`

**Key Attributes** (where data actually lives in facade):
```python
self._manifests: dict[str, DatasetManifest]
self._contracts: dict[str, DataContract]
self._events: list[dict[str, Any]]
self._watermarks: dict[str, Watermark]  # Key: "dataset_id:instrument_id:source"
self._lineage: list[dict[str, Any]]
self._lock: threading.RLock()
self.backend: BackendType  # JSON or POSTGRES
```

**Key Methods** (internal, not public):
```python
def _load_registry() -> None
def _save_registry(immediate: bool = False) -> None
def _do_save() -> None
def _dict_to_manifest(data: dict) -> DatasetManifest
def _manifest_to_dict(manifest: DatasetManifest) -> dict
def _dict_to_watermark(data: dict) -> Watermark
def _watermark_to_dict(watermark: Watermark) -> dict
def _manifest_from_row(row: Any) -> DatasetManifest  # PostgreSQL
def flush() -> None  # Public: force immediate save
```

---

### 4. `/ml/registry/common/manifest_manager.py` (NEW)

**Current Status**: NEW component for dataset operations

**Contains**:
- `ManifestManagerComponent` class
  - `register_dataset(manifest)` - delegates to `_persistence`
  - `get_manifest(dataset_id)` - delegates to `_persistence`
  - `list_manifests()` - delegates to `_persistence`
  - `update_manifest(dataset_id, changes)` - delegates to `_persistence`
  - `deprecate(dataset_id)` - delegates to `_persistence`
  - `get_contract(dataset_id)` - delegates to `_persistence`
  - `_create_contract_from_manifest(manifest)` - helper

**Delegation Pattern**:
```python
def __init__(self, persistence: DataPersistenceComponent) -> None:
    self._persistence = persistence

def register_dataset(self, manifest: DatasetManifest) -> str:
    with self._persistence._lock:
        # ... validation and logic
        self._persistence._manifests[manifest.dataset_id] = manifest
        return manifest.dataset_id
```

---

### 5. `/ml/registry/common/event_emission.py` (NEW)

**Current Status**: NEW component for event tracking

**Contains**:
- `EventEmissionComponent` class
  - `emit_event(dataset_id, instrument_id, ...)` - records event
  - `iter_events(...)` - iterates stored events (not public)

**Delegation Pattern**:
```python
def emit_event(
    self,
    dataset_id: str,
    instrument_id: str,
    stage: Stage,
    source: Source,
    ...
) -> None:
    with self._persistence._lock:
        event = {...}
        self._persistence._events.append(event)
        self._persistence._save_registry()
```

---

### 6. `/ml/registry/common/watermark_manager.py` (NEW)

**Current Status**: NEW component for watermark tracking

**Contains**:
- `WatermarkManagerComponent` class
  - `update_watermark(...)` - creates or updates watermark
  - `get_watermark(...)` -> Watermark | None - retrieves watermark
  - `iter_watermarks(...)` -> Iterator[Watermark] - filters & yields

**Key Detail** (watermark key format):
```python
watermark_key = f"{dataset_id}:{instrument_id}:{source_value}"
# Examples:
# "bars_eurusd_1m:EUR/USD:live"
# "quotes_eurusd:EUR/USD:historical"
# "features_momentum:BTC/USD:backfill"
```

**Delegation Pattern**:
```python
def get_watermark(self, dataset_id: str, ...) -> Watermark | None:
    with self._persistence._lock:
        watermark_key = f"{dataset_id}:{instrument_id}:{source_val}"
        return self._persistence._watermarks.get(watermark_key)  # May be None
```

---

### 7. `/ml/registry/common/lineage_tracker.py` (NEW)

**Current Status**: NEW component for lineage tracking

**Contains**:
- `LineageTrackerComponent` class
  - `link_lineage(child, parents, transform_id, ...)` - records lineage
  - `iter_lineage(child=..., parent=...)` -> Iterator - filters & yields
  - `get_pipeline_signature(dataset_id)` -> str | None
  - `set_pipeline_signature(dataset_id, signature)` -> None

**Delegation Pattern**:
```python
def link_lineage(self, child_dataset_id: str, parent_ids: list[str], ...) -> None:
    with self._persistence._lock:
        for parent_id in parent_ids:
            entry = {...}
            self._persistence._lineage.append(entry)
        self._persistence._save_registry()
```

---

### 8. `/ml/registry/persistence.py` (EXISTING)

**Current Status**: Backend abstraction (no changes to public API)

**Contains**:
- `BackendType` enum: JSON, POSTGRES
- `PersistenceConfig` dataclass
  - `backend: BackendType`
  - `json_path: Path` (for JSON backend)
  - `db_url: str` (for PostgreSQL backend)
- `PersistenceManager` class
  - `get_session()` - returns SQLAlchemy session (PostgreSQL only)
  - `load_json(key)` - loads JSON (JSON only)
  - `save_json(data, key)` - saves JSON (JSON only)

---

### 9. `/ml/registry/dataclasses.py` (EXISTING)

**Current Status**: Type definitions (no changes)

**Contains**:
- `DatasetManifest` - dataset metadata
- `DataContract` - data quality contracts
- `DatasetLineageRecord` - lineage relationship
- `DatasetType` enum - BARS, QUOTES, FEATURES, FUNDAMENTALS, EARNINGS_ACTUALS, EARNINGS_ESTIMATES
- `StorageKind` enum - PARQUET, CSV, DELTA, JSONL
- `ValidationRule`, `ValidationRuleType` - contract rules
- `QualityFlag` enum - PASS, WARN, FAIL

---

### 10. `/ml/registry/__init__.py` (EXISTING)

**Current Status**: Public exports (updated to include facade)

**Key Exports**:
```python
from ml.registry.data_registry import DataRegistry, Watermark
from ml.registry.data_registry_facade import DataRegistryFacade, create_data_registry
from ml.registry.dataclasses import DatasetManifest, DataContract, ...
from ml.registry.persistence import PersistenceConfig, BackendType, PersistenceManager
```

---

## Test File Mappings

### Unit Tests for Components

```
ml/tests/unit/registry/common/
├── test_data_persistence.py            ← DataPersistenceComponent tests
├── test_manifest_manager.py            ← ManifestManagerComponent tests
├── test_event_emission.py              ← EventEmissionComponent tests
├── test_watermark_manager.py           ← WatermarkManagerComponent tests
└── test_lineage_tracker.py             ← LineageTrackerComponent tests
```

### Integration Tests

```
ml/tests/unit/registry/
├── test_data_registry_facade.py        ← Facade delegation tests
├── test_data_registry_json_backend.py  ← JSON backend tests
└── test_data_registry_unit.py          ← Legacy compatibility tests
```

### Parity Tests

```
ml/tests/facades/
└── test_data_registry_parity.py        ← DataRegistry vs DataRegistryFacade parity
```

---

## Access Pattern Reference

### Accessing Manifests

```python
# ✅ PUBLIC API (works with both legacy and facade)
registry = create_data_registry(Path("/tmp"))
manifests: list[DatasetManifest] = registry.list_manifests()
manifest: DatasetManifest = registry.get_manifest("dataset_id")

# ❌ LEGACY DIRECT ACCESS (only works with DataRegistry, not facade)
registry._manifests  # AttributeError on facade

# ⚠️ TESTING WORKAROUND (access through facade's persistence)
if isinstance(registry, DataRegistryFacade):
    manifests_dict = registry._persistence._manifests
```

### Accessing Events

```python
# ✅ PUBLIC API (works with both legacy and facade)
registry.emit_event(dataset_id, instrument_id, stage, source, run_id, ...)

# ❌ LEGACY DIRECT ACCESS (only works with DataRegistry, not facade)
events = registry._events  # AttributeError on facade

# ⚠️ TESTING WORKAROUND (access through facade's persistence)
if isinstance(registry, DataRegistryFacade):
    events_list = registry._persistence._events
```

### Accessing Watermarks

```python
# ✅ PUBLIC API (works with both legacy and facade)
wm = registry.get_watermark("dataset_id", "EUR/USD", Source.LIVE)
registry.update_watermark("dataset_id", "EUR/USD", Source.LIVE, 1000000000, 100, 99.5)
for wm in registry.iter_watermarks(dataset_id="dataset_id"):
    print(wm)

# ❌ LEGACY DIRECT ACCESS (only works with DataRegistry, not facade)
watermarks = registry._watermarks  # AttributeError on facade

# ⚠️ TESTING WORKAROUND (access through facade's persistence)
if isinstance(registry, DataRegistryFacade):
    watermarks_dict = registry._persistence._watermarks
    # Key format: "dataset_id:instrument_id:source_value"
    wm = watermarks_dict.get("dataset_id:EUR/USD:live")
```

### Accessing Lineage

```python
# ✅ PUBLIC API (works with both legacy and facade)
registry.link_lineage("child", ["parent1", "parent2"], "transform_id", ...)
for record in registry.iter_lineage(child="child_dataset"):
    print(record)

# ❌ LEGACY DIRECT ACCESS (only works with DataRegistry, not facade)
lineage = registry._lineage  # AttributeError on facade

# ⚠️ TESTING WORKAROUND (access through facade's persistence)
if isinstance(registry, DataRegistryFacade):
    lineage_list = registry._persistence._lineage
```

---

## Threading & Locks

### Thread-Safety Guarantee

All components use `persistence._lock` (RLock) for thread-safe access:

```python
# In all components
def some_method(self):
    with self._persistence._lock:
        # All operations are serialized here
        # No concurrent access to _manifests, _events, etc.
```

### Accessing Lock for Long Operations

```python
registry = create_data_registry(Path("/tmp"))
if isinstance(registry, DataRegistryFacade):
    persistence = registry._persistence
    with persistence._lock:
        # Long-running operation
        for dataset_id in persistence._manifests:
            # Process safely
            pass
```

---

## Quick File Lookup

| Need | File | Class | Key Methods |
|------|------|-------|-------------|
| Create registry | `data_registry_facade.py` | - | `create_data_registry()` |
| Register dataset | `common/manifest_manager.py` | ManifestManagerComponent | `register_dataset()` |
| Get manifest | `common/manifest_manager.py` | ManifestManagerComponent | `get_manifest()` |
| List manifests | `common/manifest_manager.py` | ManifestManagerComponent | `list_manifests()` |
| Emit event | `common/event_emission.py` | EventEmissionComponent | `emit_event()` |
| Update watermark | `common/watermark_manager.py` | WatermarkManagerComponent | `update_watermark()` |
| Get watermark | `common/watermark_manager.py` | WatermarkManagerComponent | `get_watermark()` |
| Iter watermarks | `common/watermark_manager.py` | WatermarkManagerComponent | `iter_watermarks()` |
| Link lineage | `common/lineage_tracker.py` | LineageTrackerComponent | `link_lineage()` |
| Iter lineage | `common/lineage_tracker.py` | LineageTrackerComponent | `iter_lineage()` |
| Storage layer | `common/data_persistence.py` | DataPersistenceComponent | `_load_registry()`, `_save_registry()` |
| Persistence config | `persistence.py` | PersistenceConfig | - |
| Data types | `dataclasses.py` | Various | DatasetManifest, DataContract, etc. |

