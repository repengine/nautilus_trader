# Agent Task Framework for ML Refactoring

**Purpose:** Systematically execute the refactoring plan using specialized agents with **Test-Driven Development (TDD)** and **mandatory runtime validation**.

## Critical Lesson Learned

**⚠️ Type Checking ≠ Correctness | Imports ≠ Functionality | Syntax ≠ Semantics**

Previous workflow validated **form** (code looks right) but not **function** (code works).
This framework now enforces:
1. **TDD approach**: Tests written BEFORE implementation
2. **5-phase validation**: Design → Implement → Static Check → Runtime Check → System Check
3. **Smaller agent scope**: Each agent has ONE focused responsibility

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                        │
│  - Reads REFACTORING_PLAN.md                                │
│  - Spawns agents sequentially through 5 phases              │
│  - Manages handoffs between phases                          │
│  - Enforces validation before approval                      │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: TEST DESIGN AGENT                      │
│  🎯 Mission: Design tests BEFORE implementation (TDD)       │
│  Input Context:                                              │
│    - REFACTORING_PLAN.md (specific task section)            │
│    - tasks/phase_X_Y_task_name.md (task definition)        │
│    - CODING_STANDARDS.md                                     │
│    - universal_patterns_guide.md                            │
│    - CLAUDE.md (AI agent guide)                             │
│  Responsibilities:                                           │
│    - Design comprehensive test cases                        │
│    - Write test skeletons/stubs (initially failing/skip)   │
│    - Document expected behavior in test docstrings         │
│    - Define fixtures and test data                         │
│    - Specify edge cases (nulls, empty, boundaries)         │
│  Output:                                                     │
│    - Test files (unit, integration, e2e)                   │
│    - TEST_DESIGN_REPORT.md (strategy, coverage plan)       │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│            PHASE 2: IMPLEMENTATION AGENT                     │
│  🎯 Mission: Write code to satisfy test specifications      │
│  Input Context:                                              │
│    - TEST_DESIGN_REPORT.md (specification!)                │
│    - Test files (define the contract)                      │
│    - tasks/phase_X_Y_task_name.md                          │
│    - CODING_STANDARDS.md                                     │
│    - universal_patterns_guide.md                            │
│  Responsibilities:                                           │
│    - Read test cases to understand requirements            │
│    - Implement code with 100% type annotations             │
│    - Follow architectural patterns (protocols, facades)    │
│    - Make tests pass one by one                            │
│    - Preserve backward compatibility                       │
│    - Add docstrings and comments                           │
│  Output:                                                     │
│    - Production code changes                                │
│    - IMPLEMENTATION_REPORT.md (what changed, how tests pass)│
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│          PHASE 3: STATIC VALIDATION AGENT (mandatory)        │
│  🎯 Mission: Verify CODE QUALITY (not runtime behavior)     │
│  Input Context:                                              │
│    - IMPLEMENTATION_REPORT.md                               │
│    - TEST_DESIGN_REPORT.md                                  │
│    - CODING_STANDARDS.md                                     │
│  Validation Steps (Code Quality):                            │
│    1. Run: ruff check [files] (must be 0 violations)       │
│    2. Run: mypy [files] --strict (must be 0 errors)        │
│    3. Run: make validate-nautilus-patterns (must pass)      │
│    4. Test: All imports work (python -c "import ...")      │
│    5. Verify: Public API preserved (no breaking changes)   │
│    6. Check: 100% type annotations present                  │
│    7. Verify: No circular dependencies introduced           │
│  Output:                                                     │
│    - STATIC_VALIDATION_REPORT.md (pass/fail + issues)       │
│    - Decision: PASS → Phase 4 | FAIL → Back to Phase 2     │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│      PHASE 4: INTEGRATION VALIDATION AGENT (mandatory)       │
│  🎯 Mission: RUNTIME verification - tests MUST RUN          │
│  ⚠️ CRITICAL: "X collected" ≠ "X passed" - tests must RUN!  │
│  Input Context:                                              │
│    - STATIC_VALIDATION_REPORT.md (must be PASS)             │
│    - IMPLEMENTATION_REPORT.md                               │
│    - TEST_DESIGN_REPORT.md                                  │
│  Validation Steps (Runtime Verification):                    │
│    1. Run: pytest [unit tests] -v (verify "X passed")      │
│       - ⚠️ CRITICAL: NOT "X collected" - must RUN!          │
│       - Must: 100% pass rate (0 failed, 0 errors)          │
│    2. Run: pytest [integration tests] -v -m integration    │
│       - Verify: All tests PASS                              │
│    3. Run: pytest [e2e tests] -v                           │
│       - Verify: All tests PASS                              │
│    4. Test: Can instantiate classes (python -c "...")      │
│    5. Test: Methods work with real data                     │
│    6. Test: Config classes accept legacy parameters         │
│    7. Test: Feature flags work in BOTH modes (if applicable)│
│       - Legacy mode: ML_USE_LEGACY_X=1 pytest [...]        │
│       - Facade mode: ML_USE_LEGACY_X=0 pytest [...]        │
│       - Pass counts MUST match                              │
│    8. Check: No infinite loops/recursion (timeout test)    │
│    9. Verify: All public APIs from original preserved       │
│   10. Check: Coverage maintained or improved                │
│  Output:                                                     │
│    - INTEGRATION_VALIDATION_REPORT.md (pass/fail + output)  │
│    - Decision: PASS → Phase 5 or APPROVED | FAIL → Phase 2 │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│      PHASE 5: SYSTEM VALIDATION AGENT (optional)             │
│  🎯 Mission: Deployment verification                        │
│  Only required for major changes (stores, orchestrator, etc) │
│  Input Context:                                              │
│    - INTEGRATION_VALIDATION_REPORT.md (must be PASS)         │
│    - Service definitions                                     │
│  Validation Steps (Deployment Verification):                 │
│    1. Build: docker build (must succeed)                    │
│    2. Boot: Start services (actor, strategy, dashboard)     │
│       - Verify: No exceptions in logs                       │
│       - Verify: Services respond to health checks           │
│    3. Smoke tests: Basic workflows end-to-end               │
│       - Can write to stores                                 │
│       - Can read from stores                                │
│       - Can register datasets                               │
│       - Can run pipeline stages                             │
│    4. Check: No 503 errors on critical endpoints            │
│  Output:                                                     │
│    - SYSTEM_VALIDATION_REPORT.md (pass/fail)                │
│    - Decision: PASS → APPROVED | FAIL → Back to Phase 2    │
└─────────────────────────────────────────────────────────────┘
                             ↓
                  ┌──────────┴──────────┐
                  │                     │
            ✅ APPROVED           ❌ REJECTED
                  │                     │
          Commit changes         Return to Phase 2
          Move to next task      (Implementation Agent)
```

---

## Task Template Structure

Each task follows this structure:

### 1. Task Definition File

**Location:** `/home/nate/projects/nautilus_trader/tasks/phase_X_Y_task_name.md`

```markdown
# Task: [Phase X.Y] [Task Name]

## Context
**Phase:** X - [Phase Name]
**Task ID:** X.Y
**Depends On:** [Previous task IDs or "none"]
**Estimated Effort:** [hours]

## Scope
[Precise description of what to change]

## Required Reading
- [ ] REFACTORING_PLAN.md (Phase X section)
- [ ] ml/docs/development/CODING_STANDARDS.md
- [ ] ml/docs/architecture/universal_patterns_guide.md
- [ ] [Domain-specific doc if applicable]

## Definition of Done
- [ ] [Specific criterion 1]
- [ ] [Specific criterion 2]
- [ ] All tests pass
- [ ] Ruff check passes
- [ ] MyPy --strict passes
- [ ] make validate-nautilus-patterns passes
- [ ] Coverage ≥ baseline

## Files to Modify
- [ ] /path/to/file1.py (lines X-Y)
- [ ] /path/to/file2.py (create new)

## Implementation Steps
1. [Concrete step 1]
2. [Concrete step 2]
3. Run tests: `pytest ml/tests/path/to/test_*.py -v`
4. Run validation: `make validate-nautilus-patterns`

## Testing Requirements
- [ ] Unit tests for new functions
- [ ] Integration tests if database/stores involved
- [ ] Backward compatibility tests (if applicable)

## Rollback Plan
[How to undo changes if validation fails]

## Success Metrics
- Lines reduced: [target]
- DRY impact score reduced: [target]
- Test coverage: [maintained or +X%]
```

### 2. Phase 1: Test Design Agent Prompt Template

```markdown
You are a TEST DESIGN AGENT responsible for designing comprehensive tests BEFORE implementation (TDD approach).

## YOUR MISSION
Design tests for Phase [X.Y]: [Task Name]

## REQUIRED CONTEXT (Read these FIRST)
1. Task Definition: tasks/phase_X_Y_task_name.md
2. Overall Plan: REFACTORING_PLAN.md (Phase X section)
3. Coding Standards: ml/docs/development/CODING_STANDARDS.md
4. Architecture Patterns: ml/docs/architecture/universal_patterns_guide.md
5. CLAUDE.md (AI agent guide)

## YOUR RESPONSIBILITIES
1. Read ALL required context documents
2. Understand requirements from task definition
3. Design comprehensive test cases covering:
   - Happy path scenarios
   - Error conditions and edge cases
   - Boundary conditions (empty, null, extreme values)
   - Backward compatibility (if applicable)
   - Feature flag parity (if facade created)
4. Write test skeletons/stubs with clear assertions
5. Document expected behavior in test docstrings
6. Define test data fixtures needed
7. Plan integration test scenarios
8. Identify e2e workflows to verify

## TEST DESIGN CHECKLIST
- [ ] Unit tests for each new function/method
- [ ] Integration tests if touching stores/DB/external systems
- [ ] E2E tests if workflow changes
- [ ] Property tests for invariants (using hypothesis)
- [ ] Backward compatibility tests (legacy params work)
- [ ] Feature flag parity tests (both modes pass)
- [ ] Error condition tests (invalid inputs handled)
- [ ] Edge case tests (boundaries, nulls, empty arrays)
- [ ] Performance tests for hot paths (if applicable)

## CONSTRAINTS
- Tests should be FAILING initially or marked @pytest.mark.skip
- Tests define the CONTRACT for implementation
- Cover happy path AND failure modes
- Test coverage target: ≥90% for ML modules, ≥80% general
- NEVER write implementation code (only tests)
- Use clear, descriptive test names

## OUTPUT FORMAT
Generate TEST_DESIGN_REPORT.md with:
1. Test strategy overview
2. List of test files created/modified
3. Test cases with expected outcomes
4. Fixtures and test data requirements
5. Coverage expectations
6. Handoff notes for implementation agent

BEGIN TEST DESIGN:
```

### 3. Phase 2: Implementation Agent Prompt Template

```markdown
You are an IMPLEMENTATION AGENT responsible for writing code to satisfy test specifications.

## YOUR MISSION
Implement Phase [X.Y]: [Task Name]

## REQUIRED CONTEXT (Read these FIRST)
1. TEST_DESIGN_REPORT.md (YOUR SPECIFICATION!)
2. Test files (define the contract you must satisfy)
3. Task Definition: tasks/phase_X_Y_task_name.md
4. Coding Standards: ml/docs/development/CODING_STANDARDS.md
5. Architecture Patterns: ml/docs/architecture/universal_patterns_guide.md
6. CLAUDE.md (AI agent guide)

## YOUR RESPONSIBILITIES
1. Read TEST_DESIGN_REPORT.md to understand requirements
2. Review test files to see expected behavior
3. Implement code with complete type annotations
4. Make tests pass one by one
5. Follow architectural patterns (protocols, facades, etc.)
6. Preserve backward compatibility
7. Add comprehensive docstrings and comments
8. Run tests locally to verify they pass

## IMPLEMENTATION CHECKLIST
- [ ] Read all test cases to understand contracts
- [ ] Implement with 100% type annotations
- [ ] Follow Protocol-First pattern
- [ ] Use config-driven approach (no hardcoded values)
- [ ] Preserve all public APIs
- [ ] Support legacy parameters (if applicable)
- [ ] Add comprehensive docstrings with examples
- [ ] Run pytest locally - verify tests pass
- [ ] Check test coverage meets target (≥90% ML, ≥80% general)

## CONSTRAINTS
- NEVER modify test expectations without justification
- NEVER skip tests to make things "pass"
- NEVER deviate from coding standards
- ALWAYS use protocols over concrete types
- ALWAYS preserve backward compatibility
- Focus on making tests green, not adding features
- DO NOT run linters or validators (Phase 3's job)

## OUTPUT FORMAT
Generate IMPLEMENTATION_REPORT.md with:
1. Files changed (with line ranges)
2. Implementation approach/strategy
3. How each test is satisfied
4. Any deviations from plan (with justification)
5. Current test pass rate (from local pytest run)
6. Handoff notes for validation agents

BEGIN IMPLEMENTATION:
```

### 4. Phase 3: Static Validation Agent Prompt Template

```markdown
You are a STATIC VALIDATION AGENT responsible for verifying code quality (Phase 3).

⚠️ IMPORTANT: You check CODE QUALITY only. Runtime behavior is Phase 4's job.

## YOUR MISSION
Static Validation for Phase [X.Y]: [Task Name]

## INPUT DOCUMENTS
1. IMPLEMENTATION_REPORT.md from implementation agent
2. TEST_DESIGN_REPORT.md from test design agent
3. tasks/phase_X_Y_task_name.md (original task definition)
4. CODING_STANDARDS.md
5. universal_patterns_guide.md

## VALIDATION CHECKLIST (STATIC ONLY)

### A. Code Quality (MANDATORY - Must be 0 errors)
- [ ] Run: `ruff check [changed files]`
     - Must output: "All checks passed!"
     - Any violations → REJECT
- [ ] Run: `mypy [changed files] --strict`
     - Must output: "Success: no issues found"
     - Any errors → REJECT (timeout acceptable if documented)
- [ ] Run: `make validate-nautilus-patterns`
     - Must not show errors for changed files
     - God-class warnings acceptable for legacy files
- [ ] Verify: No new warnings introduced
     - Compare before/after warning counts

### B. Import Verification
- [ ] Test: `python -c "import ml.module"`
     - Must complete without exceptions
     - Test for each changed module
- [ ] Check: No circular import errors
     - `python -c "import ml.X; import ml.Y; importlib.reload(ml.X)"`
- [ ] Verify: 100% type annotations present
     - No missing return types
     - No `Any` without justification

### C. API Surface Verification
- [ ] Extract public API from original file
     - `grep "^    def [^_]" original.py`
- [ ] Extract public API from modified file
     - `grep "^    def [^_]" modified.py`
- [ ] Compare: All original methods present
     - Additions OK
     - Removals → REJECT (breaking change)
- [ ] Check: Method signatures match
     - Parameter names/types unchanged
     - Return types unchanged

### D. Config Class Verification (if applicable)
- [ ] Check: All original parameters accepted
     - Old code using config must still work
     - New parameters OK, removed ones → REJECT
- [ ] Verify: `__post_init__` handles legacy params
     - Deprecation warnings acceptable

### E. Architecture Compliance
- [ ] Verify: Follows Protocol-First pattern
- [ ] Verify: No circular dependencies introduced
- [ ] Verify: Metrics use centralized bootstrap
- [ ] Verify: No direct prometheus_client imports

### F. Documentation
- [ ] Check: Changed functions have docstrings
- [ ] Check: Complex logic is commented
- [ ] Verify: __init__.py updated if needed

## VALIDATION COMMANDS
Run these in order:

```bash
# 1. Linting (MUST pass)
ruff check [changed files]
# Expected: "All checks passed!"

# 2. Type checking (MUST pass or documented timeout)
mypy [changed files] --strict
# Expected: "Success: no issues found"

# 3. Pattern validation (errors must not be in changed files)
make validate-nautilus-patterns
# Check output for errors in changed files

# 4. Import tests (MUST complete without exception)
python -c "import ml.module"
# For each changed module

# 5. API diff (public methods preserved)
diff <(grep "^    def [^_]" original.py | sort) \
     <(grep "^    def [^_]" modified.py | sort)
# Additions OK, removals are breaking changes
```

## OUTPUT: STATIC_VALIDATION_REPORT.md

Generate a report with this structure:

```markdown
# Static Validation Report: Phase [X.Y] [Task Name]

**Validation Date:** [timestamp]
**Validator:** Static Validation Agent (Phase 2)
**Task Agent:** [agent ID]

## Summary
**Status:** ✅ PASS (proceed to Phase 3) / ❌ FAIL (return to Fix Agent)

## Code Quality Results
- Ruff: [PASS/FAIL + output]
- MyPy: [PASS/TIMEOUT/FAIL + output]
- Patterns: [PASS/FAIL + errors in changed files]
- Imports: [PASS/FAIL + which failed]

## API Compatibility
- Public methods preserved: [YES/NO + list missing]
- Method signatures match: [YES/NO + differences]
- Config classes backward compatible: [YES/NO + breaking changes]

## Issues Found
[If failed, list specific issues with file:line]

## Decision
[If PASS]: Proceed to Phase 4 (Integration Validation)
[If FAIL]: Return to Implementation Agent (Phase 2) with issues list
```

BEGIN STATIC VALIDATION:
```

### 5. Phase 4: Integration Validation Agent Prompt Template

```markdown
You are an INTEGRATION VALIDATION AGENT responsible for runtime verification (Phase 4).

⚠️ CRITICAL: This is where we failed before. You MUST actually RUN tests, not just collect them.
⚠️ CRITICAL: "X collected" ≠ "X passed" - tests must actually EXECUTE and PASS!

## YOUR MISSION
Integration Validation for Phase [X.Y]: [Task Name]

## INPUT DOCUMENTS
1. STATIC_VALIDATION_REPORT.md (must be PASS)
2. IMPLEMENTATION_REPORT.md from implementation agent
3. TEST_DESIGN_REPORT.md from test design agent
4. Test file paths from task
5. tasks/phase_X_Y_task_name.md

## PRE-REQUISITE
STATIC_VALIDATION_REPORT.md must show PASS status. If not, REJECT immediately and return to Phase 2.

## VALIDATION CHECKLIST (RUNTIME - THIS IS CRITICAL!)

### A. Unit Tests (MANDATORY - Must be 100% pass)
- [ ] Run: `pytest ml/tests/unit/[module]/ -v`
     - ⚠️ CRITICAL: Verify output contains "X passed" NOT "X collected"
     - Must: 100% pass rate (0 failed, 0 errors)
     - If any failures → REJECT
- [ ] Parse output: Extract pass/fail counts
     - Example: "15 passed in 2.3s" → 15 passed, 0 failed ✅
     - Example: "15 collected" → NOT RUN ❌ REJECT
- [ ] Verify: New tests added for new functionality
     - Check test file was modified/created

### B. Integration Tests (MANDATORY if stores/DB involved)
- [ ] Run: `pytest ml/tests/integration/[module]/ -v -m integration`
     - Must: All tests PASS
     - Parse: "X passed" in output
- [ ] Verify: Tests actually executed (not skipped)
     - Check for "X passed" not "X skipped"

### C. E2E Tests (MANDATORY if exists)
- [ ] Run: `pytest ml/tests/e2e/test_[module]_e2e.py -v`
     - Must: All tests PASS
     - Parse pass count from output
- [ ] Verify: Tests cover main workflows
     - Check test names are descriptive

### D. Class Instantiation Test (MANDATORY)
- [ ] Test: Can create instances
     ```python
     python -c "
     from ml.[module] import [Class]
     obj = [Class]([minimal args])
     assert obj is not None
     print('✓ Instantiation works')
     "
     ```
     - Must complete without exception
     - Try for each new/modified class

### E. Method Execution Test (MANDATORY)
- [ ] Test: Can call methods with real data
     ```python
     python -c "
     from ml.[module] import [Class]
     obj = [Class]([args])
     result = obj.method([real data])
     print(f'✓ Method returned: {type(result)}')
     "
     ```
     - Must complete without exception
     - Try for critical public methods

### F. Config Backward Compatibility (MANDATORY if config changed)
- [ ] Test: Old config patterns still work
     ```python
     python -c "
     from ml.config import [Config]
     # Try old pattern
     cfg = [Config](old_param=value, ...)
     print('✓ Legacy params accepted')
     "
     ```
     - Must not raise TypeError
     - Deprecation warnings OK

### G. Feature Flag Parity Test (MANDATORY if facade created)
- [ ] Test legacy mode:
     ```bash
     ML_USE_LEGACY_[MODULE]=1 pytest ml/tests/unit/[module]/test_basic.py -v
     # Record pass count
     ```
- [ ] Test facade mode:
     ```bash
     ML_USE_LEGACY_[MODULE]=0 pytest ml/tests/unit/[module]/test_basic.py -v
     # Record pass count
     ```
- [ ] Compare: Pass counts must match
     - If different → REJECT (parity broken)

### H. Recursion/Infinite Loop Check (MANDATORY)
- [ ] Test: Initialize stores/registries
     ```python
     timeout 10s python -c "
     from ml.stores import DataStore
     store = DataStore([args])
     print('✓ No infinite loops')
     "
     ```
     - Must complete within 10 seconds
     - If timeout → REJECT (infinite loop)

### I. Public API Preservation Test (MANDATORY)
- [ ] Test: Old usage patterns still work
     - Try examples from existing tests
     - Try patterns from documentation
     - All must work without modification

### J. Coverage Check (MANDATORY)
- [ ] Run: `pytest [tests] --cov=[module] --cov-report=term-missing`
     - Coverage must be ≥ baseline
     - New code must have tests

## VALIDATION COMMANDS
Run these in order (DO NOT SKIP ANY):

```bash
# 1. Unit tests (MUST RUN, not collect)
pytest ml/tests/unit/[module]/ -v
# ⚠️ CHECK OUTPUT: "X passed" means tests RAN
# ⚠️ "X collected" means tests NOT RUN → REJECT

# 2. Integration tests (if applicable)
pytest ml/tests/integration/[module]/ -v -m integration
# CHECK: "X passed" (not skipped)

# 3. E2E tests (if exists)
pytest ml/tests/e2e/test_[module]_e2e.py -v
# CHECK: "X passed"

# 4. Instantiation test
python -c "from ml.[module] import [Class]; obj = [Class]([args]); print('✓')"
# MUST complete without exception

# 5. Method test
python -c "from ml.[module] import [Class]; obj = [Class]([args]); result = obj.method([data]); print('✓')"
# MUST complete without exception

# 6. Config compatibility (if applicable)
python -c "from ml.config import [Config]; cfg = [Config](old_param=value); print('✓')"
# MUST not raise TypeError

# 7. Feature flag parity (if facade)
ML_USE_LEGACY_[MODULE]=1 pytest ml/tests/unit/[module]/test_basic.py -v > legacy.txt
ML_USE_LEGACY_[MODULE]=0 pytest ml/tests/unit/[module]/test_basic.py -v > facade.txt
diff <(grep "passed" legacy.txt) <(grep "passed" facade.txt)
# Pass counts MUST match

# 8. Recursion check
timeout 10s python -c "from ml.[module] import [Class]; obj = [Class]([args]); print('✓')"
# MUST complete within 10s

# 9. Coverage
pytest [tests] --cov=[module] --cov-report=term-missing
# Coverage MUST be ≥ baseline
```

## OUTPUT: INTEGRATION_VALIDATION_REPORT.md

Generate a report with this structure:

```markdown
# Integration Validation Report: Phase [X.Y] [Task Name]

**Validation Date:** [timestamp]
**Validator:** Integration Validation Agent (Phase 4)
**Static Validation:** [PASS status confirmed from Phase 3]

## Summary
**Status:** ✅ PASS (proceed to Phase 5 or APPROVED) / ❌ FAIL (return to Phase 2)

## Test Execution Results

### Unit Tests
```
[Full pytest output]
```
- Tests RUN (not just collected): [YES/NO - CRITICAL]
- Passed: [count]
- Failed: [count]
- Pass rate: [percentage]

### Integration Tests
```
[Full pytest output]
```
- Passed: [count]
- Failed: [count]

### E2E Tests
```
[Full pytest output]
```
- Passed: [count]
- Failed: [count]

### Runtime Verification
- Instantiation test: [PASS/FAIL]
- Method execution test: [PASS/FAIL]
- Config compatibility: [PASS/FAIL/N/A]
- Feature flag parity: [PASS/FAIL/N/A]
- Recursion check: [PASS/FAIL]

### Coverage
- Before: [X%]
- After: [Y%]
- Change: [+/-Z%]

## Issues Found
[If failed, list specific test failures with output]

## Decision
[If PASS]: All runtime tests passed. Proceed to Phase 5 (System Validation) if major change, or APPROVED if Phase 5 not required.
[If FAIL]: Tests failed. Return to Implementation Agent (Phase 2) with failures list.
```

## DECISION CRITERIA
**APPROVE (PASS) if:**
- All unit tests RAN and PASSED (not just collected)
- All integration tests PASSED
- All E2E tests PASSED
- Instantiation test PASSED
- No infinite loops detected
- Feature flag parity confirmed (if applicable)
- Coverage ≥ baseline

**REJECT (FAIL) if:**
- Tests only collected (not run)
- Any test failures
- Instantiation fails
- Infinite loop detected
- Feature flag parity broken
- Coverage decreased

BEGIN INTEGRATION VALIDATION:
```

### 6. Phase 5: System Validation Agent Prompt Template (Optional)

```markdown
You are a SYSTEM VALIDATION AGENT responsible for deployment verification (Phase 5).

⚠️ This phase is OPTIONAL - only required for major changes affecting stores, orchestrators, registries, or system-level components.

## YOUR MISSION
System Validation for Phase [X.Y]: [Task Name]

## INPUT DOCUMENTS
1. INTEGRATION_VALIDATION_REPORT.md (must be PASS)
2. IMPLEMENTATION_REPORT.md
3. Service definitions and deployment configs
4. tasks/phase_X_Y_task_name.md

## PRE-REQUISITE
INTEGRATION_VALIDATION_REPORT.md must show PASS status. If not, REJECT immediately.

## WHEN TO USE THIS PHASE
**Required for:**
- Store modifications (DataStore, FeatureStore, ModelStore, StrategyStore)
- Orchestrator changes (pipeline, dataset builders)
- Registry system changes (feature, model, strategy, data registries)
- Database schema updates
- Actor system modifications
- API/service interface changes

**Skip for:**
- Minor utility function changes
- Test-only changes
- Documentation updates
- Config adjustments without deployment impact

## VALIDATION CHECKLIST (DEPLOYMENT)

### A. Build Verification (MANDATORY)
- [ ] Run: docker build
     - Must: Complete without errors
     - Verify: Image builds successfully

### B. Service Boot (MANDATORY)
- [ ] Start services: actor, strategy, dashboard
     - Must: Boot without exceptions
     - Check: Logs for startup errors
     - Verify: Services reach ready state

### C. Health Checks (MANDATORY)
- [ ] Test: Health endpoints respond
     - GET /health → 200 OK
     - All critical services responding
     - No timeouts or 503 errors

### D. Smoke Tests (MANDATORY)
- [ ] Store operations:
     - Can write to stores → succeeds
     - Can read from stores → succeeds
     - Data persisted correctly
- [ ] Registry operations:
     - Can register datasets → succeeds
     - Can query registries → succeeds
     - Metadata consistent
- [ ] Pipeline operations:
     - Can run pipeline stages → succeeds
     - Data flows correctly
     - No stage failures

### E. Critical Endpoint Check (MANDATORY)
- [ ] Test critical API endpoints
     - No 503 Service Unavailable errors
     - Response times within SLA
     - Error rates acceptable

## VALIDATION COMMANDS
Run these in order:

```bash
# 1. Build
docker build -t ml-service:test .
# MUST complete without errors

# 2. Start services
docker-compose up -d
# Wait for services to boot (check logs)

# 3. Health checks
curl http://localhost:8000/health
# MUST return 200 OK

# 4. Store smoke tests
python -c "
from ml.stores import DataStore
store = DataStore(...)
store.write_data(...)  # MUST succeed
data = store.read_data(...)  # MUST succeed
print('✓ Store operations work')
"

# 5. Registry smoke tests
python -c "
from ml.registry import DataRegistry
registry = DataRegistry(...)
registry.register_dataset(...)  # MUST succeed
print('✓ Registry operations work')
"

# 6. Pipeline smoke tests
pytest ml/tests/e2e/test_pipeline_e2e.py -v
# MUST pass

# 7. Teardown
docker-compose down
```

## OUTPUT: SYSTEM_VALIDATION_REPORT.md

Generate a report with this structure:

```markdown
# System Validation Report: Phase [X.Y] [Task Name]

**Validation Date:** [timestamp]
**Validator:** System Validation Agent (Phase 5)
**Integration Validation:** [PASS status confirmed from Phase 4]

## Summary
**Status:** ✅ PASS (APPROVED - ready to commit) / ❌ FAIL (return to Phase 2)

## Build Results
```
[docker build output]
```
- Build successful: [YES/NO]
- Image size: [MB]
- Build time: [seconds]

## Service Boot Results
```
[docker-compose up output]
```
- Services started: [list]
- Boot time: [seconds]
- Exceptions in logs: [YES/NO + details]

## Health Check Results
- Actor service: [PASS/FAIL + response]
- Strategy service: [PASS/FAIL + response]
- Dashboard service: [PASS/FAIL + response]

## Smoke Test Results
### Store Operations
- Write operations: [PASS/FAIL]
- Read operations: [PASS/FAIL]
- Data consistency: [PASS/FAIL]

### Registry Operations
- Dataset registration: [PASS/FAIL]
- Registry queries: [PASS/FAIL]
- Metadata integrity: [PASS/FAIL]

### Pipeline Operations
- Stage execution: [PASS/FAIL]
- Data flow: [PASS/FAIL]
- E2E workflow: [PASS/FAIL]

## Critical Endpoint Check
- 503 errors detected: [YES/NO + endpoints]
- Response times: [within SLA / degraded]
- Error rates: [acceptable / elevated]

## Issues Found
[If failed, list specific deployment issues]

## Decision
[If PASS]: All deployment checks passed. APPROVED - ready to commit.
[If FAIL]: Deployment verification failed. Return to Implementation Agent (Phase 2) with issues.
```

## DECISION CRITERIA
**APPROVE (PASS) if:**
- Docker build succeeds
- All services boot without exceptions
- All health checks pass
- Store operations work
- Registry operations work
- Pipeline smoke tests pass
- No 503 errors on critical endpoints

**REJECT (FAIL) if:**
- Build fails
- Services fail to boot
- Health checks fail
- Store/registry operations fail
- Pipeline smoke tests fail
- 503 errors on critical endpoints

BEGIN SYSTEM VALIDATION:
```

---

## Critical Workflow Rules

### Rule 0: TDD is Mandatory
**Tests MUST be designed BEFORE implementation.**

- Phase 1 (Test Design) writes tests first
- Phase 2 (Implementation) makes tests pass
- This ensures code satisfies requirements, not the other way around

**No exceptions.**

### Rule 1: Phases 3 and 4 are MANDATORY
**Every refactoring task MUST pass both validation phases:**

- Phase 3 (Static) checks that code looks right (syntax, types, imports)
- Phase 4 (Integration) checks that code **works** right (runtime behavior)

Phase 5 (System) is optional for major changes.

**No exceptions.**

### Rule 2: Tests Must ACTUALLY RUN
**"X collected" ≠ "X passed"**

Phase 4 agents must verify pytest output contains "X passed" not "X collected".
Test collection proves nothing about functionality.

### Rule 3: Both Feature Flag Modes Must Work
**If you create a facade, BOTH modes must pass the same tests:**
- Legacy mode: `ML_USE_LEGACY_X=1 pytest [...]`
- Facade mode: `ML_USE_LEGACY_X=0 pytest [...]`

Pass counts must match. Parity is non-negotiable.

### Rule 4: Backward Compatibility is Sacred
**All public APIs must be preserved unless explicitly approved as breaking change.**

- Check: All original public methods present
- Check: Method signatures unchanged
- Check: Config classes accept legacy parameters
- Check: Old usage patterns still work

### Rule 5: Runtime Verification Required
**Static checks (ruff, mypy, imports) are necessary but not sufficient.**

Must also verify (Phase 4):
- Classes can be instantiated
- Methods work with real data
- No infinite loops/recursion
- System can boot (Phase 5 for major changes)

### Rule 6: Each Agent Has ONE Job
**Agents must NOT overstep their responsibilities:**

- Test Design Agent: Design tests only (no implementation)
- Implementation Agent: Write code only (no validation)
- Static Validation Agent: Check code quality only (no runtime tests)
- Integration Validation Agent: Run tests only (no system deployment)
- System Validation Agent: Check deployment only

### Rule 7: When in Doubt, Run More Tests
**If validation is uncertain, err on the side of more testing.**

Better to catch issues in validation than in production.

---

## Workflow Summary

```
Test Design Agent (Phase 1)
    ↓ (designs tests, writes test skeletons, defines contracts)
Implementation Agent (Phase 2)
    ↓ (writes code to make tests pass, generates implementation report)
Static Validation Agent (Phase 3)
    ├─ PASS → Integration Validation Agent (Phase 4)
    └─ FAIL → Back to Implementation Agent (Phase 2)
             ↓ (actually runs tests, verifies runtime behavior)
Integration Validation Agent (Phase 4)
    ├─ PASS → System Validation Agent (Phase 5, optional) or APPROVED
    └─ FAIL → Back to Implementation Agent (Phase 2) → Phase 3 → Phase 4
             ↓ (boots system, smoke tests, deployment checks)
System Validation Agent (Phase 5, optional)
    ├─ PASS → APPROVED (commit)
    └─ FAIL → Back to Implementation Agent (Phase 2) → Phase 3 → Phase 4 → Phase 5
```

**Key Points:**
- Phase 1: Test-first approach (TDD) - tests define the contract
- Phase 2: Implementation follows test specifications
- Phase 3: Quick validation (seconds) - Syntax, types, imports
- Phase 4: Runtime validation (minutes) - Test execution, actual behavior
- Phase 5: System validation (minutes-hours) - Docker, deployment, smoke tests
- Phases 1-4 are always required; Phase 5 only for major changes
- All fixes go back to Phase 2 (Implementation), not Phase 1 (tests stay fixed)

---

## Phase 0 Task Breakdown (Example Tasks)

### Task 0.1: Remove stores → actors circular dependency
**File:** `tasks/phase_0_1_remove_stores_actors_import.md`

```markdown
# Task: [Phase 0.1] Remove stores → actors Circular Dependency

## Context
**Phase:** 0 - Foundation (Critical Blockers)
**Task ID:** 0.1
**Depends On:** none
**Estimated Effort:** 0.5 hours

## Scope
Remove the circular import between `ml/stores/__init__.py` and `ml/actors/base.py` by eliminating the runtime import of `BaseMLInferenceActor` in the stores module.

## Required Reading
- [x] REFACTORING_PLAN.md (Phase 0.1)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md

## Definition of Done
- [ ] `ml/stores/__init__.py` does NOT import `BaseMLInferenceActor` at runtime
- [ ] `BaseMLInferenceActor` can be imported for TYPE_CHECKING only
- [ ] All tests pass
- [ ] No import errors when `import ml.stores` is executed standalone
- [ ] No import errors when `import ml.actors` is executed standalone
- [ ] Circular dependency broken (verify with import order test)

## Files to Modify
- [ ] ml/stores/__init__.py (line 20)

## Implementation Steps
1. Read `ml/stores/__init__.py` to understand current imports
2. Identify why `BaseMLInferenceActor` is imported (likely for re-export)
3. Check if import is only needed for type hints
4. If yes: Move import inside `if TYPE_CHECKING:` block
5. If no (used at runtime): Remove from `__all__` and update consumers
6. Run: `python -c "import ml.stores; import ml.actors"`
7. Run: `python -c "import ml.actors; import ml.stores"`
8. Run: `pytest ml/tests/unit/stores/ -v`

## Testing Requirements
- [ ] Create test: `tests/test_no_circular_imports.py`
  ```python
  def test_stores_import_standalone():
      import ml.stores
      assert ml.stores is not None

  def test_actors_import_standalone():
      import ml.actors
      assert ml.actors is not None

  def test_import_order_independence():
      # Should work in either order
      import ml.stores
      import ml.actors
      import importlib
      importlib.reload(ml.stores)
      importlib.reload(ml.actors)
  ```

## Rollback Plan
```bash
git checkout ml/stores/__init__.py
```

## Success Metrics
- Circular dependency chain count: 3 → 2
- Import time: (measure before/after)
- Test suite: 100% pass rate maintained
```

### Task 0.2: Extract dataset constants to config
**File:** `tasks/phase_0_2_extract_dataset_constants.md`

```markdown
# Task: [Phase 0.2] Extract Dataset Constants to Config

## Context
**Phase:** 0 - Foundation (Critical Blockers)
**Task ID:** 0.2
**Depends On:** 0.1
**Estimated Effort:** 1 hour

## Scope
Move `EARNINGS_ACTUALS_DATASET_ID` and `EARNINGS_ESTIMATES_DATASET_ID` from `ml/stores/data_store.py` to a new centralized config module to break the registry → stores circular dependency.

## Required Reading
- [x] REFACTORING_PLAN.md (Phase 0.2)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md

## Definition of Done
- [ ] New file created: `ml/config/dataset_ids.py`
- [ ] Constants moved from `ml/stores/data_store.py`
- [ ] `ml/registry/bootstrap_datasets.py` imports from config
- [ ] `ml/stores/data_store.py` imports from config
- [ ] All existing usages updated
- [ ] All tests pass
- [ ] Circular dependency broken (registry ↔ stores)

## Files to Modify
- [ ] ml/config/dataset_ids.py (CREATE NEW)
- [ ] ml/stores/data_store.py (UPDATE: remove constants)
- [ ] ml/registry/bootstrap_datasets.py (UPDATE: lines 29-30)
- [ ] ml/config/__init__.py (UPDATE: export new constants)

## Implementation Steps
1. Create `ml/config/dataset_ids.py`:
   ```python
   """Dataset ID constants for ML module."""
   from typing import Final

   # Earnings dataset IDs
   EARNINGS_ACTUALS_DATASET_ID: Final[str] = "earnings.actuals"
   EARNINGS_ESTIMATES_DATASET_ID: Final[str] = "earnings.estimates"

   __all__ = [
       "EARNINGS_ACTUALS_DATASET_ID",
       "EARNINGS_ESTIMATES_DATASET_ID",
   ]
   ```

2. Update `ml/config/__init__.py`:
   ```python
   from ml.config.dataset_ids import (
       EARNINGS_ACTUALS_DATASET_ID,
       EARNINGS_ESTIMATES_DATASET_ID,
   )

   # Add to __all__ (keep alphabetically sorted)
   ```

3. Update `ml/registry/bootstrap_datasets.py:29-30`:
   ```python
   # OLD:
   # from ml.stores.data_store import EARNINGS_ACTUALS_DATASET_ID

   # NEW:
   from ml.config.dataset_ids import (
       EARNINGS_ACTUALS_DATASET_ID,
       EARNINGS_ESTIMATES_DATASET_ID,
   )
   ```

4. Update `ml/stores/data_store.py`:
   - Remove constant definitions
   - Import from `ml.config.dataset_ids`

5. Search for other usages:
   ```bash
   grep -r "EARNINGS_ACTUALS_DATASET_ID" ml/
   ```
   Update all imports to use `ml.config.dataset_ids`

6. Run tests:
   ```bash
   pytest ml/tests/ -k "earnings" -v
   ```

## Testing Requirements
- [ ] Existing tests pass unchanged
- [ ] Add test to verify constants accessible from config:
   ```python
   def test_dataset_ids_accessible_from_config():
       from ml.config import (
           EARNINGS_ACTUALS_DATASET_ID,
           EARNINGS_ESTIMATES_DATASET_ID,
       )
       assert EARNINGS_ACTUALS_DATASET_ID == "earnings.actuals"
       assert EARNINGS_ESTIMATES_DATASET_ID == "earnings.estimates"
   ```

## Rollback Plan
```bash
git checkout ml/config/dataset_ids.py ml/config/__init__.py
git checkout ml/stores/data_store.py
git checkout ml/registry/bootstrap_datasets.py
```

## Success Metrics
- Circular dependency chain count: 2 → 1
- Files affected: 4
- Test suite: 100% pass rate maintained
- Lines of code: +15 (new file) -10 (removed duplication) = +5 net
```

---

## Usage Instructions

### For You (Orchestrator)

1. **Generate Task Files:**
   ```bash
   mkdir -p /home/nate/projects/nautilus_trader/tasks
   mkdir -p /home/nate/projects/nautilus_trader/reports/tests
   mkdir -p /home/nate/projects/nautilus_trader/reports/implementations
   mkdir -p /home/nate/projects/nautilus_trader/reports/validations
   ```

2. **Execute Phase 0.1 (5-Agent TDD Workflow):**
   ```
   You: "Execute task Phase 0.1: Remove stores → actors circular dependency"

   Orchestrator spawns agents sequentially:

   Phase 1 - Test Design Agent:
   - Context: tasks/phase_0_1_remove_stores_actors_import.md
   - Output: reports/tests/phase_0_1_test_design_report.md + test files

   Phase 2 - Implementation Agent:
   - Input: TEST_DESIGN_REPORT.md + test files
   - Output: reports/implementations/phase_0_1_implementation_report.md + code

   Phase 3 - Static Validation Agent:
   - Input: IMPLEMENTATION_REPORT.md + TEST_DESIGN_REPORT.md
   - Output: reports/validations/phase_0_1_static_validation_report.md
   - Decision: PASS → Phase 4 | FAIL → Back to Phase 2

   Phase 4 - Integration Validation Agent:
   - Input: STATIC_VALIDATION_REPORT.md + test paths
   - Output: reports/validations/phase_0_1_integration_validation_report.md
   - Decision: PASS → Phase 5 or APPROVED | FAIL → Back to Phase 2

   Phase 5 - System Validation Agent (if major change):
   - Input: INTEGRATION_VALIDATION_REPORT.md
   - Output: reports/validations/phase_0_1_system_validation_report.md
   - Decision: PASS → APPROVED | FAIL → Back to Phase 2
   ```

3. **Review Results:**
   - Read all validation reports
   - If ✅ APPROVED: Commit and move to next task
   - If ❌ REJECTED: Implementation Agent fixes and re-validates

4. **Iterate Through All Tasks:**
   - Phase 0.1 → 0.2 → 0.3
   - Phase 1.1 → 1.2 → 1.3
   - ... continue through all phases

### For Me (Specialized Agents)

When you say:
```
"Execute task Phase 0.1"
```

I will:
1. Spawn **Test Design Agent** (Phase 1)
   - Reads task definition
   - Designs comprehensive tests
   - Generates TEST_DESIGN_REPORT.md + test files

2. Spawn **Implementation Agent** (Phase 2)
   - Reads TEST_DESIGN_REPORT.md
   - Implements code to make tests pass
   - Generates IMPLEMENTATION_REPORT.md

3. Spawn **Static Validation Agent** (Phase 3)
   - Runs linters, type checkers, import tests
   - Generates STATIC_VALIDATION_REPORT.md
   - If FAIL → back to step 2

4. Spawn **Integration Validation Agent** (Phase 4)
   - Runs tests (verifies "X passed" not "X collected")
   - Tests runtime behavior
   - Generates INTEGRATION_VALIDATION_REPORT.md
   - If FAIL → back to step 2

5. Spawn **System Validation Agent** (Phase 5, if major change)
   - Tests deployment
   - Runs smoke tests
   - Generates SYSTEM_VALIDATION_REPORT.md
   - If FAIL → back to step 2

6. Return all reports to you for review

---

## Benefits of This Approach

1. **Test-Driven Development:** Tests designed before implementation
   - Tests define the contract and requirements
   - Implementation satisfies pre-defined specifications
   - Reduces scope creep and gold-plating

2. **Smaller Agent Scope:** Each agent has ONE focused responsibility
   - Test Design: Only designs tests
   - Implementation: Only writes code
   - Static Validation: Only checks code quality
   - Integration Validation: Only runs tests
   - System Validation: Only checks deployment

3. **Systematic Execution:** Each task has clear boundaries and phases
   - Phase 1 → 2 → 3 → 4 → (5 optional)
   - Clear handoffs between agents
   - Each phase has specific deliverables

4. **Built-in Quality:** Multi-phase validation before approval
   - Static checks (syntax, types, imports)
   - Runtime checks (tests actually run)
   - System checks (deployment works)

5. **Audit Trail:** Complete reports for every phase
   - TEST_DESIGN_REPORT.md
   - IMPLEMENTATION_REPORT.md
   - STATIC_VALIDATION_REPORT.md
   - INTEGRATION_VALIDATION_REPORT.md
   - SYSTEM_VALIDATION_REPORT.md (if applicable)

6. **Rollback Safety:** Each task can be independently reverted
   - Git commits per approved task
   - Clear task boundaries

7. **Parallelizable:** Independent tasks can run concurrently
   - After Phase 0 blockers removed
   - Within same refactoring phase

8. **Coding Standards Enforced:** Agents read standards before every task
   - Phase 3 validates compliance automatically

9. **Architecture Compliance:** Universal patterns verified automatically
   - Protocol-First, Hot/Cold separation, etc.

10. **Test Coverage:** Required for every task
    - ≥90% for ML modules, ≥80% general
    - Verified in Phase 4

11. **Documentation:** Reports explain all changes
    - What changed, why, how to test
    - Deviations justified

---

## Next Steps

Ready to execute? Say:
- **"Generate all Phase 0 task files"** - I'll create the 3 task definition files
- **"Execute Phase 0.1"** - I'll spawn all 5 agents sequentially (TDD workflow)
- **"Execute all Phase 0 tasks"** - I'll run 0.1 → 0.2 → 0.3 with full 5-agent workflow
- **"Generate all task files for Phases 0-4"** - Complete task definitions for entire refactoring

**Example Full Workflow:**

```
You: "Execute Phase 0.1"

Me:
  1. Spawns Test Design Agent → generates TEST_DESIGN_REPORT.md + tests
  2. Spawns Implementation Agent → generates IMPLEMENTATION_REPORT.md + code
  3. Spawns Static Validation Agent → generates STATIC_VALIDATION_REPORT.md
     - If FAIL: Returns to step 2
  4. Spawns Integration Validation Agent → generates INTEGRATION_VALIDATION_REPORT.md
     - If FAIL: Returns to step 2
  5. Spawns System Validation Agent (if major change) → generates SYSTEM_VALIDATION_REPORT.md
     - If FAIL: Returns to step 2
  6. Returns all reports to you

You: Review reports, approve if all pass, or request fixes if any fail
```

**Your call.**
