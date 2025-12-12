# Critical Safeguards - Lessons Learned & Failure Prevention

**Purpose:** Prevent catastrophic failures during god class decomposition refactoring

**Last Failed Attempt Issues:**
1. ✅ Agents only implemented stubs (not full functionality)
2. ✅ Broke legacy code before verifying testing parity
3. ✅ Tests only collected, not executed ("X collected" vs "X passed")

**🆕 Phase 2.3 Success (2025-10-29):** Completed all 5 sub-phases with proven patterns
- 127 tests designed across 5 components
- 50 API mismatches caught by Codex MCP verification before implementation
- 100% compilation success rate after API corrections
- 0 runtime failures after V2 documents created
- **KEY LESSON: Codex verification is NON-NEGOTIABLE**

**🆕 Decomposition Audit (2025-11-26):** Comprehensive quality review revealed gaps
- 7/16 decompositions were SHALLOW (components unused, facade wraps legacy)
- 4/16 decompositions PROPER but need cleanup (duplication issues)
- 5/16 decompositions PROPER (done correctly)
- **KEY LESSON: TDD verifies behavior, not decomposition quality → Category 14 added**

**Date:** 2025-11-26 (Updated with decomposition quality audit)
**Status:** 14 categories of safeguards (Categories 0-13 + Category 14)

---

## 🆕 CATEGORY 0: Codex MCP Verification (Phase 2.3 Proven Essential)

### ⚠️ RISK: Test specifications contain invented/incorrect APIs or fixture violations

**Phase 2.3 Evidence:** 50 API mismatches caught; 100% compilation after V2 corrections; 20-30 hours saved.
**Task 1.1 Evidence:** 6/7 test files had fixture violations (missing pytest_plugins, duplicate fixtures).

**SAFEGUARD:**
```yaml
Rule: ALL test designs MUST be Codex-verified BEFORE implementation

## Part A: API Error Patterns (from 127 tests)

  - Invented methods (35%): Methods don't exist (e.g., `_buffer_bar()` → actual `on_bar()`)
  - Wrong signatures (25%): Parameters/return types wrong
  - Invented metrics (20%): Assertions for non-existent metrics
  - Config field errors (10%): Wrong field names
  - Attribute vs method (5%): Callable vs property confusion
  - Invented classes (5%): Component classes don't exist

## Part B: 🆕 Fixture Violation Patterns (from Task 1.1 audit)

  - Missing pytest_plugins (85%): Test packages don't register shared fixtures
    ❌ WRONG: No pytest_plugins in conftest.py
    ✅ CORRECT: pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

  - Duplicate fixtures (71%): Same fixture defined in multiple test files
    ❌ WRONG: @pytest.fixture def feature_config() in 5+ files
    ✅ CORRECT: Single definition in conftest.py or ml/tests/fixtures/

  - Inline fixture definitions (57%): Fixtures defined in test files
    ❌ WRONG: @pytest.fixture in test_*.py files
    ✅ CORRECT: Fixtures in conftest.py or shared fixture modules

  - Direct fixture imports (14%): Importing fixtures instead of injection
    ❌ WRONG: from ml.tests.fixtures import mock_feature_store
    ✅ CORRECT: def test_something(mock_feature_store):

## Part C: 🆕 Test Anti-Patterns (from ml/tests/docs/TEST_ANTI_PATTERNS.md)

  - Config equality (common): assert config == expected_config
    ✅ CORRECT: assert msgspec.to_builtins(config) == msgspec.to_builtins(expected)

  - Enum identity (common): assert status == DeploymentStatus.ACTIVE
    ✅ CORRECT: assert status.value == "active"

  - Missing serial markers: DB tests without @pytest.mark.serial
    ✅ CORRECT: @pytest.mark.database\n@pytest.mark.serial

  - Module-level patches: @patch at class/module level
    ✅ CORRECT: with patch(...) inside test function

## Part D: 🆕 Value Testing (from Task 1.1 lesson - 2025-12-01)

  WHY: Parity tests must verify NUMERICAL EQUIVALENCE, not container types.
       Hot path (numpy) and cold path (dict/DataFrame) must produce SAME VALUES.

  - Wrong approach (assumes return type):
    ❌ WRONG:
      legacy_features = legacy.compute_features(bars)
      facade_features = facade.compute_features(bars)
      np.testing.assert_allclose(legacy_features, facade_features)
      # Fails if one returns dict and other returns numpy!

  - Correct approach (compares values):
    ✅ CORRECT:
      for feature_name in ["rsi_14", "bb_upper", "ema_12"]:
          legacy_val = legacy_features[feature_name]
          facade_val = facade_features[feature_name]
          assert legacy_val == pytest.approx(facade_val, rel=1e-10)

  - Training/Serving Skew Prevention:
    ✅ REQUIRED: Batch mode values == Online mode values
    ✅ REQUIRED: Legacy mode values == Facade mode values
    ✅ REQUIRED: All modes produce identical feature computations

Verification Process:
  1. Test Design Agent creates specs
  2. Codex verifies:
     a) API: methods, signatures, attributes, config fields, exceptions, metrics
     b) Fixtures: pytest_plugins, no duplicates, proper placement
     c) Anti-patterns: config equality, enum identity, serial markers, patch scope
     d) Value testing: parity tests compare VALUES not container types
  3. Issues found → Create V2 with corrections
  4. Implementation uses V2 (corrected)

Decision Tree:
  0 issues → ✅ PASS → Implement
  1-5 API issues → ⚠️ Create V2 with API corrections
  1+ fixture issues → ⚠️ Fix conftest.py before implementation
  6+ API issues OR structural fixture problems → ❌ Major revision → Phase 1

Phase 2.3 Results: 11→8→14→15→2 API issues (87% improvement by Phase 2.3.5)
Task 1.1 Results: 6/7 files had fixture violations → Fixed to 0 violations
ROI: 2-8 hours invested → 20-30 hours saved (250-1500% ROI)
```

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

### ⚠️ RISK: Tests collected but not executed

**SAFEGUARD:**
```yaml
Rule: Phase 4 MUST verify tests ACTUALLY RUN

Required output:
  ✅ PASS: "47 passed in 3.2s"
  ❌ FAIL: "47 collected" (no "passed")
  ❌ FAIL: Import/fixture errors

Verification: grep -q "passed" output && ! grep -q "collected$" output
Integration agent REJECTS if tests only collected, not executed.
```

---

## Category 4: Feature Flag Implementation & Testing

### ⚠️ RISK: Feature flags incorrectly implemented

**SAFEGUARD:**
```yaml
Rule: Feature flags MUST be implemented and tested FIRST

Implementation (ml/config/feature_flags.py):
  def use_legacy_data_store() -> bool:
      return os.getenv("ML_USE_LEGACY_DATA_STORE", "0") == "1"

Phase 4 Parity Test:
  ML_USE_LEGACY_X=1 pytest → N passed
  ML_USE_LEGACY_X=0 pytest → M passed
  REQUIRE: N == M (identical pass counts)
```

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

## Category 14: Decomposition Quality Gates (2025-11-26)

### ⚠️ RISK: "Decomposition" is just a wrapper with duplicated code

**What the audit revealed:** 7 of 16 decompositions (44%) were SHALLOW - components exist but are unused, facade wraps legacy, code copied instead of extracted. TDD/parity tests PASSED because they verify behavioral correctness, not structural quality.

**Evidence from comprehensive audit:**
- BaseMLInferenceActor: Facade (2,273 lines) LARGER than legacy (2,046 lines)
- FeatureEngineer: 5 components instantiated but NEVER CALLED
- MLPipelineOrchestrator: Components return placeholder values `{"rows_written": 0}`
- Code growth: +172% instead of expected +50%

**SAFEGUARD:**
```yaml
Rule: Verify ACTUAL decomposition, not just functional parity:

Phase 4 (Integration) - Decomposition Quality Checks (MANDATORY):

  1. Facade Size Check (MUST be thin)
     ```bash
     FACADE_LINES=$(wc -l < *_facade.py)
     if [ $FACADE_LINES -gt 400 ]; then
         echo "❌ FAIL: Facade too large ($FACADE_LINES lines)"
         echo "   Facade should be ~200-400 lines of pure delegation"
         echo "   If larger, logic belongs in components, not facade"
         exit 1
     fi
     ```
     Thresholds:
       - Excellent: <300 lines
       - Acceptable: 300-400 lines
       - REJECT: >400 lines (logic not extracted)

  2. Component Usage Check (Components MUST be called)
     ```bash
     # Facade should NOT delegate to legacy
     LEGACY_CALLS=$(grep -c "self._legacy" *_facade.py || echo 0)
     if [ $LEGACY_CALLS -gt 0 ]; then
         echo "❌ FAIL: Facade delegates to legacy ($LEGACY_CALLS calls)"
         echo "   Components should own the logic, not legacy wrapper"
         exit 1
     fi

     # Facade MUST call components
     COMPONENT_CALLS=$(grep -c "self\._[a-z_]*\." *_facade.py || echo 0)
     if [ $COMPONENT_CALLS -lt 5 ]; then
         echo "❌ FAIL: Facade doesn't use components ($COMPONENT_CALLS calls)"
         echo "   Components are instantiated but never called"
         exit 1
     fi
     ```

  3. Code Duplication Check (<10% allowed)
     ```bash
     # Install jscpd if not present: npm install -g jscpd
     jscpd --min-lines 10 --reporters console \
           legacy.py *_facade.py components/*.py \
           --format "python" 2>&1 | tee duplication_report.txt

     DUPE_PERCENT=$(grep "duplicated" duplication_report.txt | grep -oP '\d+\.\d+' | head -1)
     if (( $(echo "$DUPE_PERCENT > 10" | bc -l) )); then
         echo "❌ FAIL: ${DUPE_PERCENT}% code duplication (max 10%)"
         echo "   Extract shared code to components/common.py"
         exit 1
     fi
     ```
     Common duplication patterns to check:
       - Dataclasses (DataEvent, ValidationViolation) - extract to common.py
       - Helper methods (_validate_*, _generate_*) - extract to utils component
       - Protocols - define ONCE in protocols.py, import everywhere
       - Constants (VENUE_MAP, STATIC_FEATURE_MAP) - define in config

  4. Code Growth Check (<200% allowed)
     ```bash
     BEFORE=$(wc -l < legacy.py)
     AFTER=$(cat *_facade.py components/*.py 2>/dev/null | wc -l)
     GROWTH=$((AFTER * 100 / BEFORE))

     if [ $GROWTH -gt 200 ]; then
         echo "❌ FAIL: Code grew ${GROWTH}% (max 200%)"
         echo "   Expected: Legacy N lines → Facade+Components ~1.5N lines"
         echo "   Actual growth indicates copy-paste, not extraction"
         exit 1
     fi
     ```
     Expected growth patterns:
       - Excellent: <150% (logic moved, not copied)
       - Acceptable: 150-200% (some duplication for clarity)
       - REJECT: >200% (copy-paste extraction)

  5. Dead Component Check (All components MUST be used)
     ```bash
     for component in components/*.py; do
         # Extract class name
         CLASS=$(grep -oP "^class \K\w+" "$component" | head -1)
         if [ -z "$CLASS" ]; then continue; fi

         # Count usages in facade (excluding imports)
         USAGES=$(grep -c "$CLASS" *_facade.py 2>/dev/null || echo 0)
         IMPORTS=$(grep -c "from.*import.*$CLASS\|import.*$CLASS" *_facade.py 2>/dev/null || echo 0)
         ACTUAL=$((USAGES - IMPORTS))

         if [ $ACTUAL -lt 1 ]; then
             echo "❌ FAIL: $CLASS instantiated but never used"
             echo "   Component exists in $component but facade doesn't call it"
             exit 1
         fi
     done
     ```

  6. Legacy Removal Readiness Check
     ```bash
     # After decomposition, legacy file should be deletable
     # Check if facade imports from legacy (it shouldn't)
     LEGACY_IMPORTS=$(grep -c "from.*legacy import\|import.*legacy" *_facade.py || echo 0)
     if [ $LEGACY_IMPORTS -gt 0 ]; then
         echo "⚠️ WARN: Facade still imports from legacy ($LEGACY_IMPORTS imports)"
         echo "   True decomposition = legacy file can be deleted"
     fi

     # Check if tests still reference legacy directly
     TEST_LEGACY_REFS=$(grep -r "legacy" tests/ --include="*.py" | grep -v "parity\|compare" | wc -l)
     if [ $TEST_LEGACY_REFS -gt 5 ]; then
         echo "⚠️ WARN: Tests still reference legacy ($TEST_LEGACY_REFS refs)"
         echo "   Update tests to use facade exclusively"
     fi
     ```

Why Parity Tests Don't Catch This:
  # This PASSES all parity tests but is SHALLOW:
  class FeatureEngineerFacade:
      def __init__(self):
          self._legacy = LegacyFeatureEngineer()  # Full legacy instantiated
          self.calculator = FeatureCalculator()    # Component created but...

      def calculate_features(self, bars):
          return self._legacy.calculate_features(bars)  # ...NEVER USED!

  # Parity test:
  assert legacy.calculate_features(bars) == facade.calculate_features(bars)
  # Result: ✅ PASS (both return same value)
  # Reality: Facade is a wrapper, not a decomposition

Decision Criteria:
  ✅ PASS (proceed) if ALL checks pass:
    - Facade <400 lines
    - 0 legacy delegation calls
    - >5 component method calls
    - <10% code duplication
    - <200% code growth
    - All components actually used

  ❌ FAIL (return to Phase 2) if ANY check fails:
    - Facade too large → Extract logic to components
    - Legacy delegation → Wire facade to use components
    - Components unused → Remove or wire them up
    - High duplication → Extract to common.py
    - Code bloat → Refactor, don't copy-paste
```

**Implementation:** Add these checks to Phase 4 (Integration Validation) in AGENT_TASK_FRAMEWORK.md

**Audit Results Reference:** See REFACTORING_PLAN.md Appendix F for complete findings.

---

## Enforcement Checklist - Print Before Each Phase

**🆕 UPDATED (2025-10-29):** Added Codex verification requirements based on Phase 2.3 lessons.

Before starting ANY god class decomposition, print this checklist:

```markdown
# Pre-Flight Checklist for Phase [X.Y]: [God Class Name]

## 🆕 Codex MCP Verification (Phase 2.3 + Task 1.1 Proven Essential)
- [ ] Codex verification will run AFTER Phase 1 (Test Design)
- [ ] ALL API calls will be verified against legacy code
- [ ] ALL test files will be verified for FIXTURE_GUIDE.md compliance
- [ ] V2 documents will be created if Codex finds issues
- [ ] Implementation will use V2 (corrected), not V1 (raw)
- [ ] Understand: 50 API errors + 6/7 fixture violations caught - this step is NON-NEGOTIABLE

## 🆕 Fixture Compliance (Task 1.1 Proven Essential)
- [ ] `pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)` in conftest.py
- [ ] No inline `@pytest.fixture` definitions in test files
- [ ] Fixtures consolidated in package conftest.py (not scattered)
- [ ] No duplicate fixture definitions across test files
- [ ] Shared fixtures in ml/tests/fixtures/{module}.py
- [ ] No test anti-patterns (config equality, enum identity, missing serial markers)

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

## 🆕 Decomposition Quality (2025-11-26 Audit Lessons)
- [ ] Facade will be <400 lines (pure delegation, no business logic)
- [ ] Components will be ACTUALLY CALLED (not just instantiated)
- [ ] Code duplication will be <10% (extract shared code to common.py)
- [ ] Code growth will be <200% (refactor, don't copy-paste)
- [ ] Legacy file will be DELETABLE after decomposition
- [ ] No `self._legacy` delegation in facade (components own logic)

ALL BOXES CHECKED? ✅ Proceed with Phase 1 (Test Design)
ANY BOX UNCHECKED? ❌ STOP and address missing safeguard

🆕 SPECIAL NOTES:
- Codex verification is MANDATORY (proven in Phase 2.3) - 50 API errors caught
- Fixture compliance is MANDATORY (proven in Task 1.1) - 6/7 test files had violations
- Decomposition quality checks are MANDATORY (proven in 2025-11-26 audit) - 7/16 were shallow
```

---

## Summary of Phase 2.3 Lessons (2025-10-29)

**What We Learned:**
1. **Codex verification is NON-NEGOTIABLE** - 50 API errors caught = 20-30 hours saved
2. **V2 documents are essential** - API corrections ensure compilation success
3. **Learning curves are real** - 87% error reduction from phase 1 to phase 5
4. **Common patterns exist** - Invented methods, wrong signatures, config errors
5. **Parity testing reduces errors** - Comparing existing APIs prevents invention
6. **Type annotations help** - Complete typing catches many issues early
7. **Documentation quality matters** - Line references enable Codex verification

**Statistics:**
- Phases completed: 5/5 (Phase 2.3.1 through 2.3.5)
- Tests designed: 127 across all phases
- API errors caught: 50 (would have blocked all implementation)
- V2 documents created: 5 (one per phase)
- Compilation success: 100% after corrections
- Time invested in Codex: ~2-8 hours total
- Time saved: ~20-30 hours of debugging

**Pattern Proven:**
```
Phase 1: Test Design Agent → Creates specifications
Phase 1.5: Codex MCP → Verifies APIs + Fixtures (NEW - MANDATORY)
Phase 2: Implementation Agent → Uses corrected V2
Phase 3: Static Validation → Code quality
Phase 4: Integration Validation → Runtime behavior
Phase 5: System Validation → Deployment (optional)
```

**Key Takeaway:** The 4-step workflow (design → codex → corrections → consolidation) is now proven and should be standard for all test design work. Skip Codex at your peril - every phase had issues, even the "best" one.

---

## 🆕 Summary of Task 1.1 Lessons (2025-12-01)

**What We Learned:**
1. **Fixture compliance is NON-NEGOTIABLE** - 6/7 test files had violations
2. **pytest_plugins registration is essential** - Without it, shared fixtures aren't available
3. **Fixture duplication is common** - Same fixtures defined 3-5 times across files
4. **conftest.py consolidation works** - Single source of truth for package fixtures
5. **Anti-patterns cause flaky tests** - Config equality, enum identity, missing markers
6. **FIXTURE_GUIDE.md exists but wasn't followed** - Need Phase 1.5 enforcement
7. **🆕 Value testing prevents training/serving skew** - Parity tests must compare VALUES not containers

**Statistics:**
- Test files reviewed: 7
- Files with violations: 6 (86%)
- Duplicate fixtures found: 20+ (12 unique)
- Fixtures consolidated: 12 to conftest.py
- Missing pytest_plugins: 6/7 files
- Wrong parity tests: 13 tests assumed numpy return type
- Final result: 27 passed, 53 skipped, 0 failed, 0 errors

**Common Violation Patterns:**
- Missing `pytest_plugins` registration (85%)
- Duplicate `feature_config` fixture (5 files)
- Duplicate `sample_ohlcv_dataframe` fixture (4 files)
- Inline fixtures in test files (57%)
- **🆕 Parity tests assuming container type instead of comparing values** (100% of parity file)

**Key Takeaway:** Phase 1.5 must verify:
1. FIXTURE_GUIDE.md compliance (pytest_plugins, no duplicates)
2. VALUE TESTING for parity (compare numerical values, not container types)
3. Anti-pattern avoidance (config equality, enum identity, serial markers)

---

**This document MUST be reviewed before starting any refactoring work.**

**Updates Applied:**
- **Phase 2.3 (2025-10-29):** Codex verification now mandatory (Category 0)
- **Audit (2025-11-26):** Decomposition quality gates added (Category 14) - 7/16 shallow decompositions caught
- **Task 1.1 (2025-12-01):** Fixture compliance added to Category 0 - 6/7 test files had violations

Any violation of these safeguards → AUTOMATIC REJECTION.
