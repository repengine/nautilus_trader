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

**🚨 CRITICAL UPDATE (2025-10-29):** After completing Phase 2.3 (all 5 sub-phases), we've proven a **4-step workflow** is essential:

1. **Test Design Agent** → Creates gap fix specifications
2. **Codex MCP Verification** → Verifies ALL APIs against legacy code (caught 50 errors!)
3. **API Corrections (V2)** → Fixes mismatches to ensure compilation success
4. **Consolidation** → Production-ready deliverable

**This Codex verification step is NON-NEGOTIABLE** - without it, 50 API mismatches would have blocked implementation across 5 phases.

**🚨 CRITICAL UPDATE (2025-11-26):** Comprehensive audit revealed 7/16 decompositions were SHALLOW:

**Root Cause:** Jumped straight to TDD without proper concern mapping. Components were created arbitrarily.

**Solution:** New **Phase 0: Decomposition Design** added BEFORE TDD:

1. **Decomposition Design Agent** → Maps concerns, defines component boundaries
2. **Test Design Agent** → Designs tests FOR the mapped components
3. **Implementation Agent** → Implements the designed components
4. **Validation Agents** → Now includes Category 14 decomposition quality checks

**This decomposition design step is NON-NEGOTIABLE** - without it, 7/16 decompositions were wrappers, not decompositions.

### Phase Flow

| Phase | Agent | Mission | Key Output |
|-------|-------|---------|------------|
| **0** | Planning Agent | Research codebase, design components BEFORE code | DECOMPOSITION_MAP.md |
| **1** | Test Design Agent | Design tests using decomposition map (TDD) | TEST_DESIGN_REPORT.md |
| **1.5** | Codex MCP | Verify APIs + FIXTURE_GUIDE compliance (MANDATORY) | CODEX_VERIFICATION_REPORT.md |
| **2** | Implementation Agent | Write code to satisfy test specs | IMPLEMENTATION_REPORT.md |
| **3** | Static Validation | ruff, mypy, imports, API preservation | STATIC_VALIDATION_REPORT.md |
| **4** | Integration Validation | Runtime tests, parity, Category 14 checks | INTEGRATION_VALIDATION_REPORT.md |
| **5** | System Validation | Docker, deployment, smoke tests (optional) | SYSTEM_VALIDATION_REPORT.md |

**Decision Flow:**
- Phase 3 PASS → Phase 4 | FAIL → Phase 2
- Phase 4 PASS → Phase 5 or APPROVED | FAIL → Phase 2
- Phase 5 PASS → APPROVED | FAIL → Phase 2

**Quality Gates (CRITICAL_SAFEGUARDS.md Category 14):**
- Facade <400 lines (delegation only)
- Single responsibility per component
- No duplication (uses common/ modules)
- Growth <200% of baseline

**Codex Verification Stats:** 50 API errors caught in Phase 2.3 (11+8+14+15+2 across 5 sub-phases)

**🆕 Fixture Compliance (2025-12-01):** Phase 1.5 now also verifies FIXTURE_GUIDE.md adherence:
- `pytest_plugins` registration required in test packages
- No inline fixture definitions (use `ml/tests/fixtures/`)
- Fixture reuse from shared modules (no duplication)

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

### 1.5 🆕 Phase 0: Planning Agent Prompt Template (2025-11-26)

**Agent:** `planning-architect` (Claude Code custom agent)

```markdown
You are a PLANNING AGENT responsible for researching the codebase and designing modular implementations BEFORE any tests or code are written.

## YOUR MISSION
Design the decomposition for [God Class Name] (~X,XXX lines)

## WHY THIS PHASE EXISTS
Audit of 16 god class decompositions revealed:
- 7/16 (44%) were SHALLOW - components exist but unused, facade wraps legacy
- Root cause: Jumped to TDD without mapping concerns first
- Result: +172% code growth instead of expected +50%

This phase ensures we UNDERSTAND the class before decomposing it.

## REQUIRED CONTEXT (Read these FIRST - spend 2-4 hours here!)
1. **Legacy source file** (the god class to decompose) - READ EVERY LINE
2. REFACTORING_PLAN.md (task section for this class)
3. CRITICAL_SAFEGUARDS.md Category 14 (decomposition quality gates)
4. Examples of PROPER decompositions:
   - ml/stores/feature_store_facade.py + components/ (FeatureStore - done right)
   - ml/training/base_facade.py + components/ (TrainerBase - done right)
   - ml/core/integration_facade.py + components/ (MLIntegrationManager - done right)

## YOUR RESPONSIBILITIES

### Step 1: DEEP ANALYSIS (2-4 hours minimum)
Read the entire legacy class and document:

```
## Method Inventory

### Public Methods
| Method | Lines | Purpose | Dependencies |
|--------|-------|---------|--------------|
| process_data() | 45-89 | Validates and stores data | _validate(), _store() |
| get_features() | 112-156 | Retrieves computed features | _read_cache(), _compute() |
...

### Private Methods
| Method | Lines | Purpose | Called By |
|--------|-------|---------|-----------|
| _validate() | 200-245 | Schema validation | process_data() |
| _compute() | 300-380 | Feature computation | get_features() |
...

### Attributes/Properties
| Name | Type | Purpose |
|------|------|---------|
| _cache | dict | In-memory feature cache |
| _engine | Engine | Database connection |
...

### Dataclasses/Protocols/Constants
| Name | Lines | Used By |
|------|-------|---------|
| DataEvent | 50-65 | process_data(), _emit_event() |
| VENUE_MAP | 20-35 | _resolve_venue() |
...
```

### Step 2: IDENTIFY CONCERNS (Group by responsibility)
Group methods by what they DO, not where they are:

```
## Concern Analysis

### Concern 1: Schema Validation
- _validate_schema() (lines 200-245)
- _check_constraints() (lines 250-280)
- _validate_types() (lines 285-310)
Total: ~115 lines
Why grouped: All deal with validating data against schema

### Concern 2: Data Storage
- _write_to_db() (lines 400-450)
- _batch_insert() (lines 455-490)
- _handle_conflicts() (lines 495-530)
Total: ~135 lines
Why grouped: All deal with database persistence

### Concern 3: Feature Computation
...
```

### Step 3: MAP TO COMPONENTS
Assign each method to exactly one component:

```
## Component Mapping

### SchemaValidatorComponent (components/schema_validator.py)
Concern: Schema Validation
Methods:
  - validate_schema() ← from _validate_schema()
  - check_constraints() ← from _check_constraints()
  - validate_types() ← from _validate_types()
Estimated size: ~150 lines (115 + overhead)
Dependencies: None (stateless)

### DataWriterComponent (components/data_writer.py)
Concern: Data Storage
Methods:
  - write() ← from _write_to_db()
  - batch_insert() ← from _batch_insert()
  - handle_conflicts() ← from _handle_conflicts()
Estimated size: ~180 lines (135 + overhead)
Dependencies: SchemaValidatorComponent (validates before write)

### Facade (god_class_facade.py)
Role: DELEGATION ONLY - zero business logic
Methods (thin wrappers):
  - process_data() → validator.validate() then writer.write()
  - get_features() → reader.read() or computer.compute()
Estimated size: ~250 lines (delegation + __init__)
```

### Step 4: IDENTIFY SHARED CODE TO EXTRACT FIRST
Before ANY component work, extract:

```
## Shared Code Extraction Plan

### components/common.py (extract FIRST)
Dataclasses to move:
  - DataEvent (lines 50-65) - used by 3 components
  - ValidationViolation (lines 70-85) - used by validator, writer
  - QualityReport (lines 90-110) - used by validator, facade

### components/protocols.py (extract SECOND)
Protocols to define:
  - DataReaderProtocol - for reader component
  - DataWriterProtocol - for writer component
  - ValidatorProtocol - for schema validator

### config/constants.py (if needed)
Constants to centralize:
  - VENUE_MAP (lines 20-35) - used by 2 components
  - DEFAULT_BATCH_SIZE (line 40) - used by writer
```

### Step 5: ESTIMATE SIZES
Verify decomposition meets Category 14 thresholds:

```
## Size Estimates

| Component | Estimated Lines | Status |
|-----------|-----------------|--------|
| Facade | ~250 | ✅ <400 (delegation only) |
| SchemaValidatorComponent | ~150 | ✅ Focused |
| DataWriterComponent | ~180 | ✅ Focused |
| DataReaderComponent | ~200 | ✅ Focused |
| FeatureComputerComponent | ~220 | ✅ Focused |
| common.py | ~80 | ✅ Shared code |
| protocols.py | ~60 | ✅ Interfaces |

Total: ~1,140 lines
Original: ~1,000 lines
Growth: ~14% ✅ (<200% threshold)
```

## OUTPUT: DECOMPOSITION_MAP.md

Generate a report with this structure:

```markdown
# Decomposition Map: [God Class Name]

**Date:** [timestamp]
**Legacy file:** [path] (~X,XXX lines)
**Designer:** Decomposition Design Agent (Phase 0)

## Executive Summary
- Components planned: [N]
- Estimated facade size: [X] lines (must be <400)
- Estimated total size: [Y] lines
- Expected growth: [Z]% (must be <200%)

## Method Inventory
[Full table from Step 1]

## Concern Analysis
[Groupings from Step 2]

## Component Mapping
[Assignments from Step 3 - THIS IS THE KEY DELIVERABLE]

## Shared Code Extraction Plan
[What to extract first from Step 4]

## Size Estimates
[Table from Step 5]

## Component Dependency Graph
```
                    ┌─────────────┐
                    │   Facade    │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Validator │    │  Writer  │    │  Reader  │
    └──────────┘    └────┬─────┘    └──────────┘
                         │
                         ▼
                  ┌──────────┐
                  │ Validator │ (validates before write)
                  └──────────┘
```

## Facade Delegation Pattern
```python
class GodClassFacade:
    def __init__(self, config):
        # Initialize components
        self._validator = SchemaValidatorComponent(config)
        self._writer = DataWriterComponent(config, self._validator)
        self._reader = DataReaderComponent(config)

    def process_data(self, data):
        # DELEGATION ONLY - no business logic here
        violations = self._validator.validate(data)
        if violations:
            return ValidationResult(success=False, violations=violations)
        return self._writer.write(data)
```

## Quality Gate Checklist
- [ ] Facade <400 lines? [X] lines ✅/❌
- [ ] All methods assigned to components? [Y/Z] ✅/❌
- [ ] No business logic in facade? ✅/❌
- [ ] Shared code identified? [N] items ✅/❌
- [ ] Growth <200%? [X]% ✅/❌
- [ ] Component boundaries make sense? ✅/❌

## Decision
✅ APPROVE - Proceed to Phase 1 (Test Design)
❌ REVISE - [Specific issues to address]
```

## QUALITY GATES (from CRITICAL_SAFEGUARDS.md Category 14)

Before approving, verify:
1. **Facade <400 lines** - If larger, more logic needs to move to components
2. **Every method assigned** - No orphan methods that "will be handled later"
3. **Zero business logic in facade** - Facade is delegation only
4. **Shared code identified** - Dataclasses, protocols, constants listed for extraction
5. **Growth <200%** - If higher, you're copying instead of moving

## ANTI-PATTERNS TO AVOID (from audit failures)

❌ **Don't create arbitrary components:**
   Bad: "Let's have 5 components because that seems like a good number"
   Good: "Method X, Y, Z all handle validation, so ValidationComponent"

❌ **Don't leave methods unassigned:**
   Bad: "We'll figure out where _complex_method() goes later"
   Good: "Every method has exactly one component home"

❌ **Don't plan business logic in facade:**
   Bad: "Facade will validate data and then decide which component to call"
   Good: "Facade calls validator.validate(), then writer.write()"

❌ **Don't skip shared code extraction:**
   Bad: "Each component can have its own DataEvent class"
   Good: "DataEvent in common.py, imported by all components"

## REFERENCE: PROPER vs SHALLOW Decomposition

### PROPER (MLIntegrationManager - <1% duplication)
- Facade: 287 lines (pure delegation)
- Components: Each owns ONE concern
- Shared code: Extracted to common modules
- Result: Clean, maintainable, testable

### SHALLOW (BaseMLInferenceActor - 40% duplication)
- Facade: 2,273 lines (LARGER than legacy!)
- Components: Exist but NEVER CALLED
- Shared code: Duplicated in facade AND components
- Result: Wrapper, not decomposition

Your goal is PROPER decomposition.

BEGIN DECOMPOSITION DESIGN:
```

---

### 2. Phase 1: Test Design Agent Prompt Template

```markdown
You are a TEST DESIGN AGENT responsible for designing comprehensive tests BEFORE implementation (TDD approach).

## YOUR MISSION
Design tests for Phase [X.Y]: [Task Name]

## REQUIRED CONTEXT (Read these FIRST)
1. 🆕 **DECOMPOSITION_MAP.md** (component boundaries - use this as your blueprint!)
2. Task Definition: tasks/phase_X_Y_task_name.md
3. Overall Plan: REFACTORING_PLAN.md (Phase X section)
4. Coding Standards: ml/docs/development/CODING_STANDARDS.md
5. Architecture Patterns: ml/docs/architecture/universal_patterns_guide.md
6. CLAUDE.md (AI agent guide)
7. **CRITICAL: ml/tests/fixtures/FIXTURE_GUIDE.md** (existing fixture patterns)
7. **CRITICAL: ml/tests/fixtures/__init__.py** (available fixtures)

## FIXTURE REUSE REQUIREMENTS (MANDATORY)
Before designing ANY new fixtures, you MUST:

1. **Audit Existing Fixtures**
   - Read ml/tests/fixtures/__init__.py for available exports
   - Read ml/tests/conftest.py for core fixtures
   - Search: `grep -r "@pytest.fixture" ml/tests/fixtures/`

2. **Document Fixture Reuse Plan**
   For each test category, specify:
   - Which EXISTING fixtures to use (name + module)
   - Why existing fixtures are suitable
   - Only if no suitable fixture exists: propose NEW fixture with justification

3. **Fixture Creation Rules** (if new fixtures needed)
   - Place in ml/tests/fixtures/{appropriate_module}.py
   - Add to module's `__all__` export list
   - Follow naming conventions (mock_*, create_*, sample_*)
   - Include comprehensive docstrings
   - NEVER define fixtures inline in test files

4. **Anti-Patterns to AVOID**
   ```python
   # BAD: Fixture defined inline (duplicates existing!)
   @pytest.fixture
   def mock_feature_store():
       return MagicMock()

   # GOOD: Use existing fixture by name
   def test_something(mock_feature_store):
       ...

   # BAD: Import fixture directly
   from ml.tests.fixtures import mock_feature_store

   # GOOD: Request via dependency injection (automatic)
   def test_something(mock_feature_store):
       ...
   ```

## AVAILABLE FIXTURE CATEGORIES
Reference these before creating duplicates:

### Database Fixtures (conftest.py)

- `test_database` - PostgreSQL TestDatabase with cleanup
- `database_session` - Isolated session with rollback
- `clean_postgres_db` - Truncates ml_* tables
- `isolated_engine` - In-memory SQLite for unit tests

### Store Fixtures (conftest.py)

- `store_bundle` - Feature/Model/Strategy stores (reset each test)
- `feature_store`, `model_store`, `strategy_store` - Individual stores
- `mock_feature_store`, `mock_model_store`, `mock_strategy_store` - Mocks

### Common Type Fixtures (fixtures/common.py)

- `default_instrument_id`, `default_bar_type`, `default_venue`
- `test_timestamps` - (ts_event, ts_init) nanoseconds
- `sample_features`, `sample_predictions` - Test data
- `dummy_onnx_model` - Minimal ONNX model bytes

### Integration Fixtures (fixtures/integration.py)

- `test_instrument` - Full Equity instrument
- `generate_test_bars` - Factory for Bar sequences
- `onnx_test_model_path`, `xgboost_test_model`, `lightgbm_test_model`

### Monitoring Fixtures (fixtures/monitoring_collectors.py)

- `metric_name_manager` - Unique metric names
- `prometheus_registry_cleanup` - Clean registry state

## YOUR RESPONSIBILITIES

1. Read ALL required context documents
2. **AUDIT EXISTING FIXTURES before designing new ones**
3. Understand requirements from task definition
4. Design comprehensive test cases covering:
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

### Test Coverage
- [ ] Unit tests for each new function/method
- [ ] Integration tests if touching stores/DB/external systems
- [ ] E2E tests if workflow changes
- [ ] Property tests for invariants (using hypothesis)
- [ ] Backward compatibility tests (legacy params work)
- [ ] Feature flag parity tests (both modes pass)
- [ ] Error condition tests (invalid inputs handled)
- [ ] Edge case tests (boundaries, nulls, empty arrays)
- [ ] Performance tests for hot paths (if applicable)

### 🆕 Fixture Compliance (MANDATORY - Phase 1.5 will verify)
- [ ] `pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)` in conftest.py
- [ ] No inline `@pytest.fixture` definitions (use shared fixtures)
- [ ] Fixtures consolidated in package conftest.py (not scattered)
- [ ] Shared fixtures in ml/tests/fixtures/{module}.py

### 🆕 Anti-Pattern Avoidance (See ml/tests/docs/TEST_ANTI_PATTERNS.md)
- [ ] No `assert config == other_config` (use msgspec.to_builtins)
- [ ] No `assert status == EnumMember` (use .value comparison)
- [ ] All DB tests marked `@pytest.mark.serial`
- [ ] No class/module-scoped `@patch` decorators

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
2. **Fixture Reuse Plan** (REQUIRED)
   - Existing fixtures to be used (name + source module)
   - New fixtures proposed (with justification for why existing don't suffice)
   - Where new fixtures will be placed (ml/tests/fixtures/{module}.py or conftest.py)
   - `pytest_plugins` registration confirmation
3. List of test files created/modified
4. Test cases with expected outcomes
5. Fixtures and test data requirements
6. Coverage expectations
7. **Handoff notes for Codex verification (Phase 1.5)** - MUST include:
   - ALL API methods to verify against legacy code
   - Fixture compliance checklist status
   - Anti-pattern avoidance confirmation
8. Handoff notes for implementation agent

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

### 🆕 Rule -1: Planning is Mandatory (2025-11-26 Audit Lesson)
**Codebase MUST be researched and design planned BEFORE writing any tests or code.**

- Phase 0 uses **planning-architect** agent (Claude Code custom agent)
- Research existing ml/common/ and ml/{domain}/common/ modules first
- Design modular components following Pattern 2 (scoped organization)
- Facade/main module <400 lines (delegation only)
- This ensures we UNDERSTAND the codebase before modifying it

**Lesson Learned:** Audit of 16 decompositions revealed 7/16 (44%) were SHALLOW because they skipped this step.

**Evidence:**
- Without Phase 0: Components created arbitrarily, existing common/ ignored
- With Phase 0: Proper reuse of existing modules, thin facades, no duplication
- Code growth: 172% (bad) vs expected 50% (good)

**Quality Gates (from Category 14):**
- Facade/main module <400 lines?
- Single responsibility per component?
- No duplication (uses existing common/ modules)?
- Follows Pattern 2 scoped organization?
- Growth estimate <200%?

**No exceptions for significant tasks.**

---

### Rule 0: TDD is Mandatory
**Tests MUST be designed BEFORE implementation.**

- Phase 0 (Decomposition Design) maps concerns first (**NEW - proven essential**)
- Phase 1 (Test Design) writes tests for the mapped components
- Phase 1.5 (Codex) verifies APIs match legacy code (**NEW - proven essential**)
- Phase 2 (Implementation) makes tests pass
- This ensures code satisfies requirements, not the other way around

**No exceptions.**

### 🆕 Rule 0.5: Codex Verification is NON-NEGOTIABLE
**ALL test designs MUST be verified against legacy code AND FIXTURE_GUIDE.md before implementation.**

**Details:** See CRITICAL_SAFEGUARDS.md Category 0 for error patterns and decision tree.

**Scope (updated 2025-12-01):**
1. **API Verification** - Method names, signatures, return types match legacy code
2. **Fixture Compliance** - Tests follow FIXTURE_GUIDE.md patterns (pytest_plugins, no duplicates)
3. **Anti-Pattern Detection** - No config equality, enum identity, missing serial markers
4. **🆕 Value Testing** - Parity tests compare NUMERICAL VALUES, not container types

**Value Testing Requirement:**
```python
# ❌ WRONG - assumes specific return type
legacy_features = legacy.compute_features(bars)
facade_features = facade.compute_features(bars)
np.testing.assert_allclose(legacy_features, facade_features)  # Fails if dict vs array

# ✅ CORRECT - compares VALUES regardless of container
for feature_name in feature_names:
    assert legacy_features[feature_name] == pytest.approx(facade_features[feature_name], rel=1e-10)
```

**Summary:** Phase 2.3 caught 50 API mismatches; Task 1.1 caught 6/7 files with fixture violations + wrong parity test assumptions.

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

**🆕 UPDATED (2025-11-26):** Added Phase 0 (Decomposition Design) based on audit findings. 7/16 decompositions were shallow because they skipped concern mapping.

### Phase 0: Planning (NEW - MANDATORY for all significant tasks)

```
🆕 Planning Agent (Phase 0) - planning-architect
    ↓ (researches codebase, designs modular components, plans organization)
    ├─ Quality gates pass → ✅ APPROVE → Proceed to Phase 1
    ├─ Facade >400 lines → ❌ REVISE → Extract more to components
    ├─ Duplication planned → ❌ REVISE → Use common/ modules
    └─ Pattern 2 violated → ❌ REVISE → Fix scoped organization

Output: PLANNING_DOCUMENT.md (or DECOMPOSITION_MAP.md) with:
    • Codebase research (existing common/ to reuse)
    • Component design (scope, responsibility, dependencies)
    • Shared code plan (what goes to which common/)
    • File organization tree
    • Handoff notes for TDD agent
```

### Test Design Workflow (Phases 1-1.5)

```
Test Design Agent (Phase 1)
    ↓ (designs tests FOR the mapped components, uses DECOMPOSITION_MAP.md)
🆕 Codex MCP Verification (Phase 1.5) - MANDATORY
    ↓ (verifies APIs + FIXTURE_GUIDE.md compliance)
    ├─ 0 issues → ✅ PASS → Proceed to Phase 2
    ├─ API issues → ⚠️ PARTIAL → Create V2 with API corrections → Phase 2
    ├─ Fixture issues → ⚠️ PARTIAL → Fix conftest.py/plugins → Phase 2
    └─ Major issues → ❌ FAIL → Major revision → Phase 1
```

**Phase 1.5 Verification Checklist:**
1. ✅ API methods exist in legacy code
2. ✅ Method signatures match (params, return types)
3. ✅ `pytest_plugins` registered in test package
4. ✅ No inline fixture definitions (use `ml/tests/fixtures/`)
5. ✅ No duplicate fixtures across test files
6. ✅ No test anti-patterns (config equality, enum identity, missing serial markers)
7. ✅ **Value Testing** - Parity tests compare VALUES not container types

### Implementation & Validation Workflow (Phases 2-5)

```
Implementation Agent (Phase 2)
    ↓ (writes code to make tests pass, FOLLOWS DECOMPOSITION_MAP.md)
Static Validation Agent (Phase 3)
    ├─ PASS → Integration Validation Agent (Phase 4)
    └─ FAIL → Back to Implementation Agent (Phase 2)
             ↓ (actually runs tests, verifies runtime behavior)
Integration Validation Agent (Phase 4)
    ├─ PASS → System Validation Agent (Phase 5, optional) or APPROVED
    ├─ 🆕 Category 14 checks FAIL → Back to Phase 2 (decomposition quality)
    └─ FAIL → Back to Implementation Agent (Phase 2) → Phase 3 → Phase 4
             ↓ (boots system, smoke tests, deployment checks)
System Validation Agent (Phase 5, optional)
    ├─ PASS → APPROVED (commit)
    └─ FAIL → Back to Implementation Agent (Phase 2) → Phase 3 → Phase 4 → Phase 5
```

**Key Points:**

- **Phase 0: Decomposition design - maps concerns BEFORE any code (PROVEN ESSENTIAL)**
- Phase 1: Test-first approach (TDD) - tests FOR the designed components
- **Phase 1.5: Codex verification - prevents 50+ API errors + fixture violations (PROVEN ESSENTIAL)**
- Phase 2: Implementation follows DECOMPOSITION_MAP.md + test specifications
- Phase 3: Quick validation (seconds) - Syntax, types, imports
- Phase 4: Runtime validation (minutes) - Test execution + **Category 14 decomposition quality**
- Phase 5: System validation (minutes-hours) - Docker, deployment, smoke tests
- Phase 0 required for all god class decompositions; Phases 1-4 for all tasks; Phase 5 for major changes
- If Category 14 fails in Phase 4 → back to Phase 2 to fix decomposition structure
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

## Next Steps

Ready to execute? Say:

- **"Generate all Phase 0 task files"** - I'll create the 3 task definition files
- **"Execute Phase 0.1"** - I'll spawn all 5 agents sequentially (TDD workflow)
- **"Execute all Phase 0 tasks"** - I'll run 0.1 → 0.2 → 0.3 with full 5-agent workflow
- **"Generate all task files for Phases 0-4"** - Complete task definitions for entire refactoring

**Example Full Workflow:**

```
User: "Execute Phase 0.1"

CLAUDE:
  1. Spawns Test Design Agent → generates TEST_DESIGN_REPORT.md + tests
  2. Spawns Implementation Agent → generates IMPLEMENTATION_REPORT.md + code
  3. Spawns Static Validation Agent → generates STATIC_VALIDATION_REPORT.md
     - If FAIL: Returns to step 2
  4. Spawns Integration Validation Agent → generates INTEGRATION_VALIDATION_REPORT.md
     - If FAIL: Returns to step 2
  5. Spawns System Validation Agent (if major change) → generates SYSTEM_VALIDATION_REPORT.md
     - If FAIL: Returns to step 2
  6. Returns all reports to you

USER: Review reports, approve if all pass, or request fixes if any fail
```
