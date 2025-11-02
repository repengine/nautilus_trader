# Test Anti-Patterns Guide: Preventing Test Pollution

**Last Updated:** 2025-10-31
**Status:** MANDATORY - All new tests MUST follow these patterns

## Executive Summary

**Problem:** AI code generation and example-based test writing introduce subtle pollution issues that cause tests to pass individually but fail in parallel execution.

**Audit Results:** 445 potential issues found across 3000 tests:
- 130 config equality comparisons (HIGH risk)
- 315 weak assertions (MEDIUM risk)
- Unknown enum identity checks (HIGH risk - from previous investigation)

**This document is MANDATORY reading before writing ANY test.**

---

## Critical Anti-Patterns & Fixes

### 1. Config Object Equality (HIGH SEVERITY)

**❌ WRONG - Object Identity Comparison:**
```python
def test_config_propagation(self):
    config = FeatureConfig(lookback_window=50)
    actor = MLSignalActor(config=config)

    # FAILS in parallel: compares object identity, not values
    assert actor._feature_config == config
```

**Why it fails:** In parallel execution, previous tests may mutate shared default values or the msgspec struct internals, causing object identity checks to fail even when values are identical.

**✅ CORRECT - Value Comparison:**
```python
def test_config_propagation(self):
    config = FeatureConfig(lookback_window=50)
    actor = MLSignalActor(config=config)

    # Compare VALUES, not object identity
    import msgspec
    actual = msgspec.to_builtins(actor._feature_config)
    expected = msgspec.to_builtins(config)
    assert actual == expected, f"Config mismatch: {actual} != {expected}"

    # OR test specific fields if that's all you care about
    assert actor._feature_config.lookback_window == 50
```

**Affected Files:** 130 instances found
**Priority:** P0 - Fix ASAP
**Pattern to Search:** `grep -rn "assert.*Config.*==" ml/tests/`

---

### 2. Enum Identity Checks (HIGH SEVERITY)

**❌ WRONG - Enum Member Comparison:**
```python
def test_deployment_status(self):
    model_info = registry.get_model(model_id)

    # FAILS with module reloading: enum identity changes
    assert model_info.deployment_status == DeploymentStatus.ACTIVE
```

**Why it fails:** When pytest reloads modules in parallel, enum members get new object identities. The `==` check compares identities, not values.

**✅ CORRECT - Value Comparison:**
```python
def test_deployment_status(self):
    model_info = registry.get_model(model_id)

    # Compare string VALUES
    assert model_info.deployment_status.value == "active"

    # OR use string directly
    assert model_info.deployment_status.value == DeploymentStatus.ACTIVE.value
```

**Affected Files:** 10+ instances found (from previous investigation)
**Priority:** P0 - Fix ASAP
**Pattern Proven:** Fixed 11 tests in commits f925feba2 and 89b383a69

---

### 2b. Enum isinstance() Checks in Parallel Tests (HIGH SEVERITY - UPDATED)

**Problem:** `isinstance()` checks fail when pytest-xdist workers import enums separately.

**Why it happens:**
- pytest-xdist runs tests in parallel across multiple worker processes
- Each worker imports modules independently
- Python's `isinstance()` checks object identity (memory address), not value
- Same enum imported in different workers = different identities
- Result: `isinstance(<Stage.FEATURE_COMPUTED: 'FEATURE_COMPUTED'>, Stage)` returns `False`

**Example of failure:**
```python
# WRONG - fails in parallel execution
def test_event_emission(mock_registry):
    emit_dataset_event(
        mock_registry,
        stage=Stage.FEATURE_COMPUTED,
        source=Source.LIVE,
        status=EventStatus.SUCCESS,
        ...
    )

    call_args = mock_registry.emit_event.call_args

    # This FAILS in pytest-xdist parallel execution
    assert isinstance(call_args.kwargs["stage"], Stage)
```

**Solution - Pattern A: Value Comparison (PREFERRED)**
```python
# CORRECT - compare enum values
def test_event_emission(mock_registry):
    emit_dataset_event(
        mock_registry,
        stage=Stage.FEATURE_COMPUTED,
        source=Source.LIVE,
        status=EventStatus.SUCCESS,
        ...
    )

    call_args = mock_registry.emit_event.call_args

    # Compare values, not identity
    assert call_args.kwargs["stage"].value == "FEATURE_COMPUTED"
    assert call_args.kwargs["source"].value == "live"
    assert call_args.kwargs["status"].value == "success"
```

**Solution - Pattern B: String Comparison (BACKUP)**
```python
# CORRECT - compare string representation
assert str(call_args.kwargs["stage"]) == "Stage.FEATURE_COMPUTED"
```

**Solution - Pattern C: Class Name (RARE)**
```python
# CORRECT - type-only check (don't care about value)
assert call_args.kwargs["stage"].__class__.__name__ == "Stage"
```

**When to use each pattern:**
- **Pattern A (90% of cases):** When you know the expected enum value
- **Pattern B (5% of cases):** When you need to verify both type and value
- **Pattern C (5% of cases):** When you only care about the type

**Migration guide:**
1. Search for: `isinstance(.*Stage\|Source\|EventStatus.*)`
2. Identify the expected enum value from test context
3. Replace with: `obj.value == "expected_value"`
4. Run test individually: `pytest path/to/test.py::test_name -xvs` (should pass)
5. Run test in full suite: `make pytest-ml` (should still pass)

**Verification:**
```bash
# Find all isinstance checks on our enum types
grep -rn "isinstance.*Stage\|isinstance.*Source\|isinstance.*EventStatus" ml/tests/

# After fixes, this should return zero results:
grep -rn "isinstance.*Stage\|isinstance.*Source\|isinstance.*EventStatus" ml/tests/contracts/
```

**Affected Files:** Fixed 9 instances in Phase 0.0 (test_dataset_event_contracts.py)
**Priority:** P0 - CRITICAL
**Status:** FIXED in Phase 0.0

---

### 3. Parallel Execution Without Markers (HIGH SEVERITY)

**❌ WRONG - Database Tests Without Serial Marker:**
```python
@pytest.mark.parametrize("run", [1, 2, 3])
def test_database_cleanup(fresh_store_bundle, run):
    # Parameterized tests run in PARALLEL by default
    # Cleanup from run=1 may not finish before run=2 starts
    pass
```

**Why it fails:** pytest-xdist runs parameterized tests in parallel. Database cleanup (table truncation, timer cancellation) may not complete before the next parameter instance starts.

**✅ CORRECT - Serial Execution for Database Tests:**
```python
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.parametrize("run", [1, 2, 3])
def test_database_cleanup(fresh_store_bundle, run):
    # Sequential execution ensures cleanup completes
    pass
```

**Affected Files:** Fixed in commit 89b383a69
**Priority:** P0 - ALWAYS required for database tests
**Rule:** ALL tests using PostgreSQL fixtures MUST have `@pytest.mark.serial`

---

### 4. Weak Assertions (MEDIUM SEVERITY)

**❌ WRONG - Vague Checks:**
```python
def test_store_initialization(self):
    assert feature_store is not None  # Too weak!
    assert model_store is not None    # Doesn't verify behavior
```

**Why it's weak:** Tests pass even if stores are broken or misconfigured. Doesn't catch actual bugs.

**✅ CORRECT - Specific Behavior Checks:**
```python
def test_store_initialization(self):
    # Verify stores actually work
    assert hasattr(feature_store, "write_features")
    assert hasattr(feature_store, "read_features")

    # Even better: test actual behavior
    feature_store.write_features(...)
    features = feature_store.read_features(...)
    assert len(features) > 0
```

**Affected Files:** 315 instances found
**Priority:** P2 - Improve over time
**Pattern to Search:** `grep -rn "assert.*is not None" ml/tests/`

---

### 5. Mock State Pollution (HIGH SEVERITY)

**❌ WRONG - Module-Level Mocks:**
```python
# Module level - shared across all tests!
@patch("ml.actors.signal.MLSignalActor._load_model")
class TestMLSignalActor:
    def test_prediction(self):
        # This mock persists across ALL tests in the class
        pass
```

**Why it fails:** Mocks at class/module level are shared. Previous tests configure them differently, causing later tests to fail with unexpected calls.

**✅ CORRECT - Test-Level Mocks:**
```python
class TestMLSignalActor:
    def test_prediction(self):
        # Mock only within this test scope
        with patch("ml.actors.signal.MLSignalActor._load_model") as mock_load:
            mock_load.return_value = Mock(spec=InferenceSession)
            # Test code
```

**Affected Files:** Unknown - needs manual review
**Priority:** P1 - Review when debugging flaky tests
**Rule:** NEVER use class-level `@patch` decorators

---

## Testing Checklist (MANDATORY)

Before committing ANY test, verify:

- [ ] **No object identity checks** - Use `.value` for enums, `msgspec.to_builtins()` for configs
- [ ] **Database tests have `@pytest.mark.serial`** - ALWAYS
- [ ] **Mocks are test-scoped** - Use `with patch()` context managers
- [ ] **Assertions are specific** - Test behavior, not just "is not None"
- [ ] **Test passes individually** - `pytest path/to/test.py::test_name -xvs`
- [ ] **Test passes in full suite** - `make pytest-ml` (or at least the module)
- [ ] **No hardcoded values** - Use fixtures for connections, configs
- [ ] **Type annotations complete** - `mypy ml/tests --strict` passes

---

## Automated Detection

### Pre-Commit Hook (Coming Soon)

```bash
# Add to .pre-commit-config.yaml
- repo: local
  hooks:
    - id: test-anti-patterns
      name: Check for test anti-patterns
      entry: python scripts/check_test_patterns.py
      language: python
      files: ^ml/tests/.*\.py$
```

### Semgrep Rules (Coming Soon)

```yaml
# .semgrep/test-anti-patterns.yml
rules:
  - id: test-config-object-equality
    pattern: assert $X == $Y
    where:
      - metavariable-pattern:
          metavariable: $Y
          pattern: |
            ...Config(...)
    message: Use msgspec.to_builtins() for config comparison
    severity: ERROR
```

### Manual Scan

```bash
# Find config equality issues
grep -rn "assert.*Config.*==" ml/tests/ --include="*.py"

# Find enum identity checks
grep -rn "assert.*Status\." ml/tests/ --include="*.py"
grep -rn "assert.*Role\." ml/tests/ --include="*.py"

# Find database tests without serial marker
grep -l "@pytest.mark.database" ml/tests/**/*.py | \
  xargs grep -L "@pytest.mark.serial"
```

---

## Migration Guide

### Step 1: Audit Existing Tests (Estimated: 2-3 days)

1. Run the scan script to find all 445 issues
2. Categorize by severity (P0, P1, P2)
3. Create tickets for each category

### Step 2: Fix P0 Issues First (Estimated: 1-2 weeks)

**Priority Order:**
1. Fix 130 config equality comparisons
2. Find and fix remaining enum identity checks
3. Add serial markers to database tests

**Batch Size:** Fix 10-20 per day, validate with full suite after each batch

### Step 3: Prevent Future Issues (Estimated: 1 week)

1. Add semgrep rules to catch patterns
2. Update CLAUDE.md with test-specific guidance
3. Create pre-commit hook validator
4. Document in onboarding guide

---

## Examples: Before & After

### Example 1: Config Equality

**Before (fails in suite):**
```python
def test_feature_store_config_propagation(self):
    feature_config = FeatureConfig()
    actor = MLSignalActor(feature_config=feature_config)
    assert cast(Any, actor._feature_store).feature_config == feature_config  # ❌
```

**After (passes reliably):**
```python
def test_feature_store_config_propagation(self):
    feature_config = FeatureConfig()
    actor = MLSignalActor(feature_config=feature_config)

    # Compare values, not object identity
    import msgspec
    actual = msgspec.to_builtins(cast(Any, actor._feature_store).feature_config)
    expected = msgspec.to_builtins(feature_config)
    assert actual == expected  # ✅
```

### Example 2: Enum Identity

**Before (fails with module reload):**
```python
def test_deployment_status(self):
    model_info = registry.get_model(model_id)
    assert model_info.deployment_status == DeploymentStatus.ACTIVE  # ❌
```

**After (passes reliably):**
```python
def test_deployment_status(self):
    model_info = registry.get_model(model_id)
    assert model_info.deployment_status.value == "active"  # ✅
```

### Example 3: Parallel Database Tests

**Before (fails in suite):**
```python
@pytest.mark.parametrize("run", [1, 2, 3])
def test_fresh_store_bundle_consistency(fresh_store_bundle, run):  # ❌
    # Runs in parallel, cleanup doesn't complete
    pass
```

**After (passes reliably):**
```python
@pytest.mark.database
@pytest.mark.serial
@pytest.mark.parametrize("run", [1, 2, 3])
def test_fresh_store_bundle_consistency(fresh_store_bundle, run):  # ✅
    # Sequential execution ensures cleanup
    pass
```

---

## Reference Material

### Related Documents

- **TESTING_STRATEGY.md** - Overall testing philosophy and strategies
- **FIXTURE_GUIDE.md** - Fixture usage patterns
- **CLAUDE.md** - General coding standards (will be updated with test guidance)

### Proven Patterns

**Commits demonstrating correct fixes:**
- `f925feba2` - Fixed 9 enum identity issues with `.value` pattern
- `89b383a69` - Added serial markers to prevent parallel pollution
- `[current]` - Fixed config equality with `msgspec.to_builtins()`

---

## FAQ

**Q: Why can't I just use `==` for configs?**
A: In parallel execution, msgspec structs may have different object identities even when values are identical. Use `msgspec.to_builtins()` to compare values.

**Q: How do I know if my test needs `@pytest.mark.serial`?**
A: If it uses ANY database fixture (`test_database`, `fresh_store_bundle`, store instances), it MUST be serial.

**Q: What about `@pytest.mark.database` vs `@pytest.mark.serial`?**
A: Use BOTH. `database` marks it as integration, `serial` prevents parallel pollution. They serve different purposes.

**Q: Can I fix these issues incrementally?**
A: Yes, but prioritize P0 issues (config equality, enum identity) first. These cause the most flakiness.

**Q: How do I validate my fix?**
A: Run the test individually (`pytest path/to/test.py::test_name -xvs`) AND in full suite (`make pytest-ml`). It must pass BOTH.

---

## Conclusion

Test pollution is a systemic issue requiring systematic fixes. This guide provides:

1. **Detection:** Patterns to search for existing issues
2. **Prevention:** Rules to avoid introducing new issues
3. **Remediation:** Step-by-step fixes for each anti-pattern
4. **Automation:** Tools to catch issues before they reach CI

**Remember:** A test that passes individually but fails in the suite is a BUG in the test, not the code.

---

**This document is a LIVING guide. Update it as new patterns emerge.**
