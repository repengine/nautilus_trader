# Critical Safeguards - Lessons Learned & Failure Prevention

**Purpose:** Prevent catastrophic failures during god class decomposition refactoring

**Last Failed Attempt Issues:**
1. ✅ Agents only implemented stubs (not full functionality)
2. ✅ Broke legacy code before verifying testing parity
3. ✅ Tests only collected, not executed ("X collected" vs "X passed")

**Date:** 2025-10-15
**Status:** Pre-execution review

---

## Category 1: Code Preservation & Rollback Safety

### ⚠️ RISK: Deleting legacy code before facade is proven

**What went wrong before:** Agents modified/deleted legacy code before verifying new code works.

**SAFEGUARD:**
```yaml
Rule: NEVER delete, rename, or modify legacy class until ALL conditions met:
  1. ✅ Facade implementation complete
  2. ✅ Static validation PASSED (Phase 3)
  3. ✅ Integration validation PASSED (Phase 4)
  4. ✅ Parity tests show 100% identical behavior
  5. ✅ Feature flag tested in BOTH modes
  6. ✅ System validation PASSED (Phase 5 if applicable)

Implementation:
  - Legacy class → rename to {ClassName}Legacy
  - Keep legacy file unchanged until ALL validations pass
  - Feature flag controls which implementation is used
  - Only after full validation: deprecate legacy (don't delete yet)
```

**Git Strategy:**
```bash
# Phase 2: Implementation - DO NOT TOUCH LEGACY
git add ml/stores/data_store_components/*.py  # New components
git add ml/stores/data_store_facade.py        # Facade
git commit -m "feat: add DataStore components (legacy untouched)"

# Phase 3: Static validation passes
# (no commit - just validation)

# Phase 4: Integration validation passes
git add ml/tests/facades/test_data_store_parity.py
git commit -m "test: add DataStore parity tests (all pass)"

# Phase 5: System validation passes
# NOW we can deprecate legacy
git add ml/stores/data_store.py  # Add deprecation warnings
git commit -m "refactor: deprecate DataStoreLegacy (facade proven)"

# After 1 release cycle with warnings
git rm ml/stores/data_store_legacy.py
git commit -m "refactor: remove DataStoreLegacy (deprecated 1 release ago)"
```

**Enforcement:** Static validation agent REJECTS if legacy code modified before parity proven.

---

## Category 2: No Stubs, No TODOs, No NotImplementedError

### ⚠️ RISK: Implementation agent writes stubs instead of real code

**What went wrong before:** Agent wrote `raise NotImplementedError` or `# TODO: implement this`

**SAFEGUARD:**
```yaml
Rule: Implementation agent MUST write FULL, WORKING implementations:
  - NO "raise NotImplementedError"
  - NO "# TODO: implement"
  - NO "pass" in function bodies (except valid cases like protocols)
  - NO empty functions
  - NO placeholder code

Validation:
  Phase 3 (Static): grep for forbidden patterns:
    - grep -r "NotImplementedError" [changed files] → FAIL if found
    - grep -r "# TODO" [changed files] → FAIL if found (except tests)
    - grep -r "^\s*pass\s*$" [changed files] → CHECK (valid for protocols only)

  Phase 4 (Integration): ALL tests must PASS (not just collect)
    - "X passed" required, "X collected" is FAILURE
```

**Implementation Agent Prompt Addition:**
```markdown
## STRICT REQUIREMENTS - NO EXCEPTIONS

You are FORBIDDEN from writing:
- `raise NotImplementedError` (implement it for real!)
- `# TODO: implement this later` (implement it NOW!)
- Empty function bodies with just `pass` (write the logic!)
- Placeholder code that "will be implemented later"

If you cannot implement something fully:
1. State clearly what you need (dependencies, data, context)
2. Request those inputs
3. THEN implement fully

REJECTION CRITERIA:
- Any "TODO" in production code → AUTOMATIC REJECTION
- Any "NotImplementedError" → AUTOMATIC REJECTION
- Any stub implementation → AUTOMATIC REJECTION
```

---

## Category 3: Test Execution Verification (Not Just Collection)

### ⚠️ RISK: Tests collected but not executed, giving false confidence

**What went wrong before:** `pytest` showed "47 collected" but tests didn't run (import errors, missing fixtures)

**SAFEGUARD:**
```yaml
Rule: Phase 4 MUST verify tests ACTUALLY RUN:

  Required output parsing:
    ✅ PASS: "47 passed in 3.2s"
    ❌ FAIL: "47 collected"
    ❌ FAIL: "47 collected, 2 errors"
    ❌ FAIL: "47 passed, 1 skipped" (investigate skip)

  Verification commands:
    pytest ml/tests/unit/stores/test_data_store.py -v 2>&1 | tee test_output.txt

    # Parse output - MUST contain "passed"
    if grep -q "passed" test_output.txt && ! grep -q "collected$" test_output.txt; then
        echo "✅ Tests EXECUTED and PASSED"
    else
        echo "❌ Tests only COLLECTED or FAILED"
        exit 1
    fi

  Integration agent MUST:
    1. Run pytest with -v flag
    2. Capture FULL output
    3. Parse for "X passed" string
    4. Count passed tests
    5. Verify passed count > 0
    6. Verify no "collected" without "passed"
    7. REJECT if tests didn't execute
```

**Phase 4 Agent Prompt Addition:**
```markdown
## CRITICAL: VERIFY TESTS ACTUALLY RUN

⚠️ "47 collected" ≠ "47 passed"

You MUST verify pytest output contains "X passed" (not just "X collected").

REJECTION CRITERIA:
- Output shows "collected" but not "passed" → AUTOMATIC REJECTION
- Output shows "0 passed" → AUTOMATIC REJECTION
- Output shows import errors → AUTOMATIC REJECTION
- Output shows fixture errors → AUTOMATIC REJECTION

REQUIRED OUTPUT FORMAT:
```
======================== test session starts =========================
collected 47 items

ml/tests/unit/stores/test_data_store.py::test_write_data PASSED [ 2%]
ml/tests/unit/stores/test_data_store.py::test_read_data PASSED [ 4%]
...
======================== 47 passed in 3.21s ==========================
```

Look for "47 passed" - that's the ONLY acceptable outcome.
```

---

## Category 4: Feature Flag Implementation & Testing

### ⚠️ RISK: Feature flags implemented incorrectly or not tested

**New risk:** Feature flags control legacy vs facade, but what if the flag logic is broken?

**SAFEGUARD:**
```yaml
Rule: Feature flags MUST be implemented and tested FIRST:

  Phase 2 (Implementation) - Step 1: Implement feature flag infrastructure
    1. Create feature flag in ml/config/feature_flags.py:
       ```python
       import os

       def use_legacy_data_store() -> bool:
           return os.getenv("ML_USE_LEGACY_DATA_STORE", "0") == "1"
       ```

    2. Write feature flag tests FIRST:
       ```python
       def test_feature_flag_legacy_mode():
           with env_var("ML_USE_LEGACY_DATA_STORE", "1"):
               assert use_legacy_data_store() is True

       def test_feature_flag_facade_mode():
           with env_var("ML_USE_LEGACY_DATA_STORE", "0"):
               assert use_legacy_data_store() is False

       def test_feature_flag_default_is_facade():
           # No env var set
           assert use_legacy_data_store() is False  # Default to new code
       ```

    3. Run feature flag tests → verify PASS
    4. ONLY THEN proceed with facade implementation

  Phase 4 (Integration) - Feature flag parity testing:
    1. Run ALL tests with ML_USE_LEGACY_X=1 → count passed: N
    2. Run ALL tests with ML_USE_LEGACY_X=0 → count passed: M
    3. REQUIRE: N == M (same number of tests pass in both modes)
    4. If N ≠ M → FAIL and investigate discrepancy
```

**Enforcement:** Integration agent verifies feature flag parity before approving.

---

## Category 5: No Schema Changes During Refactoring

### ⚠️ RISK: Refactoring changes database schema, breaking backward compatibility

**New risk:** Components might try to add new columns or change table structure.

**SAFEGUARD:**
```yaml
Rule: ZERO database schema changes allowed during refactoring:

  Forbidden actions:
    - Adding new tables
    - Dropping tables
    - Adding columns to existing tables
    - Changing column types
    - Changing indexes
    - Changing constraints

  If schema change is REQUIRED:
    1. Stop refactoring
    2. Create separate schema migration task
    3. Deploy schema migration FIRST
    4. Wait for migration to production
    5. THEN resume refactoring with new schema

  Validation:
    Phase 3 (Static): Check for schema changes:
      grep -r "CREATE TABLE" [changed files] → WARN (review carefully)
      grep -r "ALTER TABLE" [changed files] → FAIL if found
      grep -r "DROP TABLE" [changed files] → FAIL if found

      Compare schema before/after:
        python -c "from ml.stores import DataStore; DataStore._setup_tables()"
        # Capture table definitions before refactoring
        # After refactoring, verify IDENTICAL table definitions
```

---

## Category 6: Circular Dependency Prevention

### ⚠️ RISK: Refactoring reintroduces circular dependencies (Phase 0 fixed these!)

**New risk:** New components might import in ways that create cycles.

**SAFEGUARD:**
```yaml
Rule: Run circular dependency check after EVERY phase:

  Phase 3 (Static validation) - MANDATORY check:
    python -c "import ml.stores; import ml.actors; import ml.registry"
    # Must complete without ImportError

    # Check for circular imports
    python -c "
    import importlib
    import ml.stores as stores
    import ml.actors as actors
    importlib.reload(stores)
    importlib.reload(actors)
    # If this fails → circular dependency detected
    "

  After each phase:
    make validate-nautilus-patterns
    # Must show ZERO circular dependencies

  Enforcement:
    Phase 3 agent REJECTS if circular dependencies detected
    Phase 3 agent REJECTS if import order matters
```

---

## Category 7: Performance Parity (Not Just Functional Parity)

### ⚠️ RISK: Facade works correctly but is 10x slower than legacy

**New risk:** Functional parity tests pass, but performance regresses.

**SAFEGUARD:**
```yaml
Rule: Performance must be within 10% of legacy:

  Phase 4 (Integration) - Performance tests REQUIRED:

    Test 1: Hot path latency (P99)
      Legacy:
        with timer():
            for i in range(1000):
                data_store_legacy.write_data(bar)
        legacy_p99 = measure_p99()

      Facade:
        with timer():
            for i in range(1000):
                data_store_facade.write_data(bar)
        facade_p99 = measure_p99()

      REQUIRE: facade_p99 <= legacy_p99 * 1.10  # Within 110%

    Test 2: Memory usage
      Legacy: memory_before = get_memory_usage()
              data_store_legacy.write_data(large_dataset)
              memory_after = get_memory_usage()
              legacy_delta = memory_after - memory_before

      Facade: memory_before = get_memory_usage()
              data_store_facade.write_data(large_dataset)
              memory_after = get_memory_usage()
              facade_delta = memory_after - memory_before

      REQUIRE: facade_delta <= legacy_delta * 1.10  # Within 110%

    Test 3: Throughput
      Legacy: count = benchmark_throughput(data_store_legacy, duration=10s)
      Facade: count = benchmark_throughput(data_store_facade, duration=10s)

      REQUIRE: facade_count >= legacy_count * 0.90  # At least 90%

  If performance fails:
    - Profile facade to find bottleneck
    - Optimize facade
    - Re-run performance tests
    - DO NOT approve until performance parity achieved
```

---

## Category 8: Configuration Backward Compatibility

### ⚠️ RISK: Old configuration files don't work with refactored code

**New risk:** Users have existing configs that might break.

**SAFEGUARD:**
```yaml
Rule: ALL existing configurations MUST work unchanged:

  Phase 4 (Integration) - Config compatibility tests:

    Test 1: Legacy config files work
      # Load actual legacy config from tests/data/legacy_configs/
      config = load_config("tests/data/legacy_configs/data_store_2023.yaml")
      store = DataStore(config)  # Must not raise
      assert store is not None

    Test 2: Legacy parameters work
      # Old code might use deprecated parameter names
      store = DataStore(
          connection_string="postgresql://...",
          pool_size=5,  # Legacy param
          legacy_mode=True  # Deprecated param - should still work with warning
      )
      assert store is not None
      # Check for deprecation warning in logs

    Test 3: Config migration NOT required
      # Users should NOT have to migrate configs
      # Facade should accept old config format
      old_config = {"db_url": "postgresql://..."}  # Old format
      store = DataStore.from_config(old_config)  # Should work
      assert store is not None

  Enforcement:
    Phase 4 agent REJECTS if any legacy config fails to load
```

---

## Category 9: Commit Strategy & Rollback Procedures

### ⚠️ RISK: Committing broken code, making rollback difficult

**New risk:** When should we commit? What if we need to rollback?

**SAFEGUARD:**
```yaml
Commit strategy - commit after each APPROVED phase:

  ❌ DO NOT commit:
    - After Phase 1 (test design) - tests might fail during implementation
    - After Phase 2 (implementation) - not validated yet
    - After Phase 3 (static validation) - not runtime validated yet

  ✅ DO commit:
    - After Phase 4 PASSES (integration validation complete)
      OR
    - After Phase 5 PASSES (system validation complete, if applicable)

  Commit structure:
    Phase 4 pass → Commit with message:
      "feat(stores): add DataStore facade components (validated)

      - Add DataStore components (schema validator, reader, writer)
      - Add DataStoreFacade with backward compatibility
      - Add feature flag ML_USE_LEGACY_DATA_STORE
      - Add parity tests (100% pass in both modes)
      - Legacy DataStore preserved as DataStoreLegacy

      All tests passing:
      - Unit tests: 55/55 passed
      - Integration tests: 12/12 passed
      - E2E tests: 7/7 passed
      - Parity tests: 3/3 passed
      - Performance: within 5% of legacy

      Phase 4 validation: PASSED

      🤖 Generated with Claude Code
      Co-Authored-By: Claude <noreply@anthropic.com>"

  Rollback procedure:
    If Phase 5 fails or production issues detected:
      git revert HEAD  # Revert the facade commit
      # Legacy code still exists, immediately rollback to legacy
      # No data loss, no downtime
```

---

## Category 10: Agent Communication & Report Standardization

### ⚠️ RISK: Phase 4 agent doesn't understand Phase 3 report format

**New risk:** Agents might misinterpret each other's outputs.

**SAFEGUARD:**
```yaml
Rule: All reports MUST follow standardized templates:

  Phase 1 - TEST_DESIGN_REPORT.md:
    Required sections:
      - Existing Test Discovery (count, files)
      - Gap Analysis (what's missing)
      - New Tests Design (only new tests)
      - Test Reuse Strategy
      - Coverage Expectations

  Phase 2 - IMPLEMENTATION_REPORT.md:
    Required sections:
      - Files Changed (exact paths and line counts)
      - Implementation Approach
      - Feature Flag Implementation (exact env var name)
      - How Tests Are Satisfied
      - Handoff to Phase 3 (what to validate)

  Phase 3 - STATIC_VALIDATION_REPORT.md:
    Required sections:
      - Summary: ✅ PASS or ❌ FAIL
      - Ruff Results: (exact output)
      - MyPy Results: (exact output)
      - Import Verification: (commands run + output)
      - Circular Dependency Check: (PASS/FAIL)
      - Decision: Proceed to Phase 4 / Return to Phase 2

  Phase 4 - INTEGRATION_VALIDATION_REPORT.md:
    Required sections:
      - Summary: ✅ PASS or ❌ FAIL
      - Test Execution Results (MUST show "X passed")
      - Feature Flag Parity (counts for both modes)
      - Performance Results (latency, memory, throughput)
      - Decision: Proceed to Phase 5 / APPROVED / Return to Phase 2

  Phase 5 - SYSTEM_VALIDATION_REPORT.md:
    Required sections:
      - Summary: ✅ PASS or ❌ FAIL
      - Docker Build Results
      - Service Boot Results
      - Health Check Results
      - Smoke Test Results
      - Decision: APPROVED / Return to Phase 2

Enforcement:
  Each agent validates previous agent's report format
  If format invalid → REJECT and request correct format
```

---

## Category 11: Task Exclusivity & Dependency Management

### ⚠️ RISK: Two agents trying to refactor overlapping code simultaneously

**New risk:** Parallel work could create conflicts.

**SAFEGUARD:**
```yaml
Rule: Strict task exclusivity - no parallel work on same files:

  Before starting any task:
    1. Check task dependencies in REFACTORING_PLAN.md
    2. Verify all prerequisite tasks COMPLETE
    3. Lock files being modified (git lock or file marker)

  Task dependency rules:
    Phase 2.1 (FeatureEngineer) → blocks nothing
    Phase 2.2 (MLPipelineOrchestrator) → depends on 2.1 (uses FeatureEngineer)
    Phase 2.3 (BaseMLInferenceActor) → blocks 2.4, 2.5 (they inherit from it)
    Phase 2.4 (DataStore) → depends on 2.3
    Phase 2.5 (MLSignalActor) → depends on 2.3
    Phase 2.6 (TFTDatasetBuilder) → depends on 2.4

  File exclusivity:
    Create .refactoring_locks/phase_2_1.lock when starting Phase 2.1
    Lock files:
      - ml/features/engineering.py
      - ml/features/*.py (all feature modules)

    Remove lock only after Phase 4 PASSES

    Next agent checks for locks before starting:
      if ls .refactoring_locks/*.lock 2>/dev/null; then
          echo "ERROR: Another refactoring task in progress"
          echo "Locked files: $(cat .refactoring_locks/*.lock)"
          exit 1
      fi
```

---

## Category 12: Hot Path Validation (P99 Latency <5ms)

### ⚠️ RISK: Refactoring accidentally slows down critical hot paths

**New risk:** MLSignalActor.predict() must stay under 5ms P99.

**SAFEGUARD:**
```yaml
Rule: Hot path performance MUST be maintained:

  Identify hot paths (from CLAUDE.md):
    - MLSignalActor.predict() - <5ms P99
    - FeatureEngineer.compute_features() - <5ms P99 per bar
    - DataStore.read_data() - <5ms P99
    - All actor on_data() handlers - <5ms P99

  Phase 4 (Integration) - Hot path benchmarks REQUIRED:

    import pytest_benchmark

    def test_hot_path_predict_latency(benchmark):
        actor = MLSignalActor(config)
        bar = create_test_bar()

        # Benchmark
        result = benchmark(actor.predict, bar)

        # Verify P99 <5ms
        p99_ms = benchmark.stats.stats.p99 * 1000
        assert p99_ms < 5.0, f"P99 latency {p99_ms}ms exceeds 5ms threshold"

    Run benchmark:
      pytest ml/tests/performance/test_hot_path.py --benchmark-only

    Compare legacy vs facade:
      Legacy P99: 3.2ms
      Facade P99: 3.5ms
      Delta: +9% (acceptable, <10%)

  If hot path regression detected:
    1. Profile facade to find bottleneck
    2. Optimize (reduce allocations, cache, etc.)
    3. Re-run benchmark
    4. REJECT if still slower than 5ms P99
```

---

## Category 13: Memory Leak Detection

### ⚠️ RISK: New components have memory leaks

**New risk:** Facade might hold references, preventing garbage collection.

**SAFEGUARD:**
```yaml
Rule: Memory must not grow unbounded:

  Phase 4 (Integration) - Memory leak test:

    import tracemalloc
    import gc

    def test_no_memory_leaks():
        gc.collect()
        tracemalloc.start()

        # Baseline
        snapshot1 = tracemalloc.take_snapshot()

        # Run operation 1000 times
        store = DataStore(config)
        for i in range(1000):
            store.write_data(create_test_bar())
            store.read_data("SPY")

        # Force cleanup
        del store
        gc.collect()

        # Measure memory
        snapshot2 = tracemalloc.take_snapshot()

        # Compare
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        # Total memory growth
        total_growth = sum(stat.size_diff for stat in top_stats)

        # Should be near zero (some growth acceptable for caches)
        assert total_growth < 10_000_000, f"Memory leak detected: {total_growth} bytes"

    Run test:
      pytest ml/tests/performance/test_memory_leaks.py -v
      # Must pass
```

---

## Enforcement Checklist - Print Before Each Phase

Before starting ANY god class decomposition, print this checklist:

```markdown
# Pre-Flight Checklist for Phase [X.Y]: [God Class Name]

## Code Preservation
- [ ] Legacy code will NOT be modified until Phase 4 passes
- [ ] Legacy class will be renamed to {ClassName}Legacy
- [ ] Feature flag implemented and tested FIRST

## Implementation Quality
- [ ] NO stubs, NO TODOs, NO NotImplementedError allowed
- [ ] Implementation agent will write FULL, WORKING code
- [ ] All functions will have real implementations

## Test Execution
- [ ] Phase 4 will verify "X passed" (not "X collected")
- [ ] Test output will be parsed for actual execution
- [ ] Zero tolerance for test collection without execution

## Feature Flags
- [ ] Feature flag infrastructure tested BEFORE facade
- [ ] Parity tests will run in BOTH modes
- [ ] Pass counts must be IDENTICAL in both modes

## Schema Safety
- [ ] ZERO schema changes during refactoring
- [ ] Table definitions will be byte-identical before/after

## Dependency Safety
- [ ] Circular dependency check will run after Phase 3
- [ ] Import order will be verified

## Performance
- [ ] Hot path latency will be benchmarked
- [ ] Performance must be within 10% of legacy
- [ ] P99 latency must stay <5ms for hot paths

## Memory
- [ ] Memory leak tests will run
- [ ] Memory growth must be <10MB over 1000 iterations

## Configuration
- [ ] Legacy configs will be tested
- [ ] Old parameter names will work (with deprecation warnings)
- [ ] NO config migration required

## Commits
- [ ] Will commit ONLY after Phase 4 PASSES (or Phase 5 if applicable)
- [ ] Commit message will include test counts and validation status
- [ ] Rollback procedure documented

## Exclusivity
- [ ] Task dependencies verified
- [ ] File locks checked
- [ ] No concurrent work on same files

ALL BOXES CHECKED? ✅ Proceed with Phase 1 (Test Design)
ANY BOX UNCHECKED? ❌ STOP and address missing safeguard
```

---

**This document MUST be reviewed before starting Phase 2.0 Analysis.**

Any violation of these safeguards → AUTOMATIC REJECTION.
