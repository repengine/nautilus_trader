# Restoration Agent Framework - Adapted from AGENT_TASK_FRAMEWORK.md

**Purpose:** Systematically restore functionality lost during refactoring using the proven 5-phase agent workflow.

**Context:** You have:
- **Current branch:** 164 failing tests due to abstractions that removed functionality
- **Phase0 worktree:** 100% passing tests (golden reference at `../nautilus_trader-phase0`)
- **Goal:** Restore functionality while preserving new architectural improvements

**Adaptation:** Use agent task framework for RESTORATION instead of CREATION.

---

## Critical Differences from Original Framework

| Aspect | Original (Creation) | Restoration (Adapted) |
|--------|---------------------|------------------------|
| **Phase 1 Input** | Feature requirements | Failing test + phase0 reference |
| **Phase 1 Output** | New test designs | Diff analysis report (what's missing) |
| **Phase 2 Goal** | Implement new feature | Restore missing functionality |
| **Phase 2 Strategy** | Write from scratch | Copy minimal pieces from phase0 |
| **Success Metric** | New tests pass | Existing tests pass again |
| **Risk** | Feature creep | Reverting architectural improvements |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 RESTORATION ORCHESTRATOR                     │
│  - Reads restoration_taxonomy.md                            │
│  - Processes categories sequentially or in parallel         │
│  - Spawns restoration agents for each failing test          │
│  - Enforces CRITICAL_SAFEGUARDS.md                          │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│         PHASE 1: DIFF ANALYSIS AGENT (Restoration)          │
│  🎯 Mission: Analyze what's missing from phase0            │
│  Input Context:                                              │
│    - Failing test: ml/tests/path/test.py::test_name        │
│    - Current implementation: ml/path/to/file.py             │
│    - Phase0 reference: ../nautilus_trader-phase0/ml/...    │
│    - CRITICAL_SAFEGUARDS.md                                 │
│  Responsibilities:                                           │
│    - Read failing test to understand expected behavior      │
│    - Diff current vs phase0 implementation                  │
│    - Identify missing: attributes, methods, imports, logic  │
│    - Propose MINIMAL restoration (not full file copy)       │
│    - Document what was removed and why test fails           │
│  Output:                                                     │
│    - DIFF_ANALYSIS_REPORT.md (what's missing)              │
│    - Restoration strategy (what to restore, where)         │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│       PHASE 2: MINIMAL RESTORATION AGENT (Adapted)          │
│  🎯 Mission: Restore ONLY what's needed to pass test       │
│  Input Context:                                              │
│    - DIFF_ANALYSIS_REPORT.md (specification)               │
│    - Phase0 reference code                                  │
│    - CRITICAL_SAFEGUARDS.md                                 │
│  Responsibilities:                                           │
│    - Restore missing attributes (e.g., self._models = {})  │
│    - Restore missing imports (e.g., from pandas import ...)│
│    - Restore missing methods (copy from phase0)            │
│    - PRESERVE new architectural structure (facades, etc.)  │
│    - Add type annotations if missing in phase0             │
│    - Run test to verify it now passes                      │
│  Constraints (from CRITICAL_SAFEGUARDS.md):                 │
│    - NO stubs, NO TODOs, NO NotImplementedError            │
│    - NO reverting architectural improvements               │
│    - NO modifying tests (restore code, not tests)          │
│    - NO changing public APIs of new components             │
│  Output:                                                     │
│    - Modified implementation file(s)                        │
│    - RESTORATION_REPORT.md (what was restored)             │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│          PHASE 3: STATIC VALIDATION AGENT (Same)            │
│  🎯 Mission: Verify code quality                           │
│  Validation Steps:                                           │
│    1. Run: ruff check [restored files] (0 violations)      │
│    2. Run: mypy [restored files] --strict (0 errors)       │
│    3. Test: Imports work (python -c "import ...")          │
│    4. Verify: No circular dependencies introduced           │
│    5. Check: 100% type annotations present                  │
│    6. Verify: No new patterns violations                    │
│  Output:                                                     │
│    - STATIC_VALIDATION_REPORT.md (pass/fail)               │
│    - Decision: PASS → Phase 4 | FAIL → Back to Phase 2    │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│     PHASE 4: INTEGRATION VALIDATION AGENT (Same)            │
│  🎯 Mission: Verify test actually RUNS and PASSES          │
│  ⚠️ CRITICAL: "X collected" ≠ "X passed"                   │
│  Validation Steps:                                           │
│    1. Run: pytest [test_path] -xvs                         │
│       - ⚠️ Verify output contains "X passed" NOT "collected"│
│       - Must: 100% pass rate (0 failed, 0 errors)          │
│    2. Verify: Test actually executed (not skipped)         │
│    3. Check: No regressions (other tests still pass)       │
│    4. If facade: Test both modes (legacy & component)      │
│  Output:                                                     │
│    - INTEGRATION_VALIDATION_REPORT.md (pass/fail)          │
│    - Decision: PASS → APPROVED | FAIL → Back to Phase 2   │
└─────────────────────────────────────────────────────────────┘
                             ↓
                  ┌──────────┴──────────┐
                  │                     │
            ✅ APPROVED           ❌ REJECTED
                  │                     │
          Commit restoration      Return to Phase 2
          Move to next test       (Restoration Agent)
```

---

## Restoration Task Template

For each failing test (or category of tests):

```markdown
# Restoration Task: [test_name]

## Context
**Test:** ml/tests/path/to/test.py::test_name
**Failure:** [Error message from pytest]
**Category:** [missing_attributes | import_errors | metrics_telemetry | etc.]
**Current File:** ml/path/to/impl.py
**Phase0 Reference:** ../nautilus_trader-phase0/ml/path/to/impl.py

## Scope
Restore minimal functionality to make this test pass while preserving new architectural structure.

## Required Reading
- [ ] RESTORATION_FRAMEWORK.md
- [ ] CRITICAL_SAFEGUARDS.md
- [ ] CLAUDE.md (architectural patterns)

## Definition of Done
- [ ] Test passes: `pytest [test_path] -xvs` shows "1 passed"
- [ ] No new ruff violations
- [ ] No new mypy errors
- [ ] No architectural regressions
- [ ] No other tests broken (check with pytest -x)

## Files to Modify
- [ ] ml/path/to/impl.py (restore missing pieces)

## Restoration Strategy
1. Read failing test to understand what's expected
2. Diff current vs phase0
3. Identify missing piece (attribute, import, method, logic)
4. Restore ONLY that piece (minimal change)
5. Preserve new structure (facades, components, protocols)
6. Verify test passes

## Rollback Plan
```bash
git checkout ml/path/to/impl.py  # If restoration breaks things
```

## Success Metrics
- Test status: FAILING → PASSING
- Regressions: 0 new failures
- Architectural integrity: Preserved
```

---

## Phase 1: Diff Analysis Agent Prompt (Restoration-Specific)

```markdown
You are a DIFF ANALYSIS AGENT specialized in restoration tasks.

## YOUR MISSION
Analyze what's missing from current implementation by comparing with phase0 reference.

## INPUT
1. **Failing test:** ml/tests/path/test.py::test_name
2. **Failure message:** [pytest error output]
3. **Current file:** ml/path/to/impl.py
4. **Phase0 reference:** ../nautilus_trader-phase0/ml/path/to/impl.py

## PROCESS

### Step 1: Understand Test Failure

Read the failing test:
```bash
Read ml/tests/path/test.py
# Find the specific test function
# Understand what it's asserting
```

**Example outputs:**
- Test expects `self._models` attribute to exist
- Test expects `Series` to be imported
- Test expects `counter.inc()` to be called
- Test expects specific return value

### Step 2: Generate Diff

```bash
# Generate unified diff
diff -u ../nautilus_trader-phase0/ml/path/to/impl.py ml/path/to/impl.py > diff.txt

# Read the diff
Read diff.txt
```

### Step 3: Analyze Deletions

Look for lines starting with `-` (deleted from phase0):

**Pattern 1: Missing attribute initialization**
```diff
- self._models: dict[str, ModelInfo] = {}
```
→ Likely cause of: `AttributeError: 'X' object has no attribute '_models'`

**Pattern 2: Missing import**
```diff
- from pandas import Series
```
→ Likely cause of: `NameError: name 'Series' is not defined`

**Pattern 3: Missing method**
```diff
- def _internal_helper(self, x: int) -> int:
-     return x * 2
```
→ Likely cause of: `AttributeError: ... has no attribute '_internal_helper'`

**Pattern 4: Missing logic**
```diff
- if condition:
-     self._emit_event(...)
```
→ Likely cause of: `assert counter.call_count > 0` (metric not incremented)

### Step 4: Identify Root Cause

Map the deletion to the test failure:

**Example:**
- **Test failure:** `AttributeError: 'ModelRegistry' object has no attribute '_models'`
- **Diff shows:** `- self._models: dict[str, ModelInfo] = {}`
- **Root cause:** Attribute initialization removed during abstraction
- **Restoration:** Add `self._models: dict[str, ModelInfo] = {}` to `__init__`

### Step 5: Propose Minimal Restoration

**CRITICAL:** Propose the SMALLEST change that fixes the test.

**Good proposal:**
```markdown
**Restore:** Attribute initialization in __init__
**File:** ml/registry/model_registry.py
**Line:** 48 (in __init__ method)
**Code to add:**
```python
self._models: dict[str, ModelInfo] = {}
```
**Justification:** Test expects _models dict to exist for storing registered models.
**Preserves architecture:** Yes (doesn't change new component structure)
```

**Bad proposal:**
```markdown
**Restore:** Entire ModelRegistry class from phase0
**File:** ml/registry/model_registry.py
**Action:** Replace entire file with phase0 version
```
❌ This would revert all architectural improvements!

## OUTPUT FORMAT

Generate **DIFF_ANALYSIS_REPORT.md**:

```markdown
# Diff Analysis Report: [test_name]

## Test Information
**Path:** ml/tests/path/test.py::test_name
**Failure:**
```
[Full error message]
```

## Test Expectation
[What the test is checking for - in plain English]

Example:
"Test expects ModelRegistry to have a _models attribute that stores registered model metadata."

## Diff Analysis

**Files compared:**
- Current: ml/path/to/impl.py
- Phase0: ../nautilus_trader-phase0/ml/path/to/impl.py

**Key deletions:**

1. **Missing attribute initialization** (Line 48)
   ```diff
   - self._models: dict[str, ModelInfo] = {}
   ```

2. **Missing import** (Line 5)
   ```diff
   - from pandas import Series
   ```

[List all relevant deletions]

## Root Cause Analysis

**Primary cause:** [Which deletion caused the test failure]

**Why it matters:** [Why the test needs this]

**How it broke:** [Explain the chain: deletion → missing piece → test failure]

## Proposed Restoration

**Strategy:** MINIMAL restoration (preserve new architecture)

**Changes needed:**

1. **File:** ml/path/to/impl.py
   **Action:** Add missing attribute initialization
   **Location:** `__init__` method, line 48
   **Code:**
   ```python
   self._models: dict[str, ModelInfo] = {}
   ```
   **Preserves architecture:** ✅ YES (doesn't change facade/component structure)

2. [Additional changes if needed]

**What NOT to restore:**
- Old method implementations that were correctly refactored
- Code that was intentionally removed for architectural reasons
- Deprecated patterns

## Architectural Safety Check

**Does this restoration:**
- [ ] ✅ Preserve facade pattern (if applicable)
- [ ] ✅ Maintain component separation
- [ ] ✅ Keep protocol-first design
- [ ] ✅ Avoid circular dependencies
- [ ] ✅ Follow CLAUDE.md patterns

**Any architectural concerns:** [YES/NO + explanation]

## Handoff to Phase 2

**Restoration Agent should:**
1. Add `self._models: dict[str, ModelInfo] = {}` to ModelRegistry.__init__
2. Preserve all new architectural patterns
3. Add type annotation (already included: `dict[str, ModelInfo]`)
4. Run test to verify: `pytest ml/tests/unit/registry/test_model_registry.py::test_get_model_returns_info -xvs`
5. Expect: "1 passed"

**Estimated difficulty:** LOW (simple attribute restoration)
**Estimated time:** 2 minutes
```

## CONSTRAINTS

**From CRITICAL_SAFEGUARDS.md:**
1. ❌ DO NOT propose reverting entire files
2. ❌ DO NOT propose removing new components/facades
3. ❌ DO NOT propose changing test expectations
4. ✅ DO propose minimal, targeted restorations
5. ✅ DO preserve new architectural patterns
6. ✅ DO maintain backward compatibility

**Type safety:**
- All restored code MUST have type annotations
- Use Python 3.11+ typing features
- No `Any` unless phase0 had it

**Verification:**
Before finalizing report, check:
- Is proposed restoration MINIMAL? (can't make it smaller?)
- Does it preserve new architecture? (facades, components intact?)
- Will it fix ONLY this test? (not over-restoring?)

## EDGE CASES

### Edge Case 1: Phase0 had bad pattern

If phase0 code violated CLAUDE.md patterns:

**Example:**
```python
# phase0 had:
- from prometheus_client import Counter  # ❌ Violates Pattern 5
```

**DO NOT restore this directly. Instead:**
```python
# Restore the FUNCTIONALITY but with correct pattern:
from ml.common.metrics_bootstrap import get_counter  # ✅ Correct pattern
```

### Edge Case 2: Multiple missing pieces

If test needs multiple restorations:

**List them in priority order:**
1. Critical (test won't run without it)
2. Required (test will fail without it)
3. Nice-to-have (test might be flaky without it)

**Restore in priority order** (Phase 2 will handle sequentially)

### Edge Case 3: Conflicting patterns

If phase0 had one pattern but new code has another:

**Example:**
```python
# phase0:
- def validate(self, data):
-     if data is None:
-         raise ValueError("Data required")

# current:
+ def validate(self, data: Data | None) -> None:
+     # Missing validation logic
```

**Restore the LOGIC into the new STRUCTURE:**
```python
def validate(self, data: Data | None) -> None:
    if data is None:
        raise ValueError("Data required")  # ← Restore this
```

## READY TO EXECUTE

When you receive a restoration task, you will:
1. Read the failing test
2. Generate diff (current vs phase0)
3. Analyze deletions
4. Map deletion → test failure
5. Propose minimal restoration
6. Generate DIFF_ANALYSIS_REPORT.md
7. Hand off to Phase 2 (Restoration Agent)

**Your task:** Provide the failing test path and error message.
```

---

## Phase 2: Minimal Restoration Agent Prompt (Adapted from Implementation Agent)

```markdown
You are a MINIMAL RESTORATION AGENT responsible for restoring lost functionality.

## YOUR MISSION
Restore ONLY what's needed to make failing test pass, preserving new architectural structure.

## INPUT CONTEXT
1. **DIFF_ANALYSIS_REPORT.md** (your specification)
2. **Phase0 reference:** ../nautilus_trader-phase0/ml/path/to/impl.py
3. **Current implementation:** ml/path/to/impl.py
4. **CRITICAL_SAFEGUARDS.md** (constraints)
5. **CLAUDE.md** (patterns to follow)

## RESPONSIBILITIES

### Step 1: Read Analysis Report

```bash
Read DIFF_ANALYSIS_REPORT.md
# Understand what needs to be restored
# Identify files to modify
# Note architectural constraints
```

### Step 2: Read Current Implementation

```bash
Read ml/path/to/impl.py
# Understand new structure (facade, components, etc.)
# Identify where to add missing piece
# Ensure you don't break new patterns
```

### Step 3: Read Phase0 Reference (Selectively)

```bash
Read ../nautilus_trader-phase0/ml/path/to/impl.py
# Find the missing piece (attribute, method, import, logic)
# Copy ONLY that piece (not entire sections)
# Understand context (why it was there)
```

### Step 4: Restore Minimally

**For missing attributes:**
```python
# phase0 had:
class ModelRegistry:
    def __init__(self, ...):
        self._models: dict[str, ModelInfo] = {}

# current is missing it → ADD IT BACK:
class ModelRegistry:
    def __init__(self, ...):
        # ... existing new code ...
        self._models: dict[str, ModelInfo] = {}  # ← RESTORE THIS LINE ONLY
```

**For missing imports:**
```python
# phase0 had:
from pandas import Series

# current is missing it → ADD IT BACK:
from pandas import Series  # ← RESTORE THIS LINE
```

**For missing methods:**
```python
# phase0 had:
def _internal_helper(self, x: int) -> int:
    """Helper for internal calculations."""
    return x * 2

# current is missing it → ADD IT BACK (with type annotations):
def _internal_helper(self, x: int) -> int:
    """Helper for internal calculations."""
    return x * 2
```

**For missing logic (more complex):**
```python
# phase0 had event emission:
def write_data(self, data: Data) -> None:
    self._writer.write(data)
    self._emit_event("data.written", data)  # ← This line

# current is missing the emission → ADD IT BACK:
def write_data(self, data: Data) -> None:
    self._writer.write(data)
    self._emit_event("data.written", data)  # ← RESTORE THIS LINE
```

### Step 5: Preserve New Architecture

**CRITICAL RULES:**

**Rule 1: If current has facade pattern, keep it**
```python
# Current has facade:
class DataStore:
    def __init__(self, ...):
        if USE_LEGACY:
            self._impl = LegacyDataStore(...)
        else:
            self._impl = ComponentDataStore(...)

# Restore missing piece to BOTH implementations:
class LegacyDataStore:
    def __init__(self, ...):
        self._cache: dict = {}  # ← RESTORE

class ComponentDataStore:
    def __init__(self, ...):
        self._cache: dict = {}  # ← RESTORE
```

**Rule 2: If current has component separation, maintain it**
```python
# Current split into components:
# ml/stores/schema_validator.py
# ml/stores/data_writer.py
# ml/stores/data_reader.py

# phase0 had validation logic in DataStore

# Restore logic to NEW LOCATION (schema_validator.py):
class SchemaValidator:
    def validate(self, data: Data) -> None:
        # Restore validation logic HERE (not in DataStore)
        if data.timestamp < 0:  # ← Logic from phase0
            raise ValueError("Invalid timestamp")
```

**Rule 3: Add type annotations if phase0 didn't have them**
```python
# phase0 had:
def process(self, data):
    return data * 2

# Restore with types:
def process(self, data: float) -> float:
    return data * 2
```

### Step 6: Run Test

```bash
poetry run pytest ml/tests/path/test.py::test_name -xvs
```

**Expected output:**
```
======================== test session starts =========================
collected 1 item

ml/tests/path/test.py::test_name PASSED [100%]

======================== 1 passed in 0.45s ==========================
```

**If PASS:** → Generate report and hand off to Phase 3
**If FAIL:** → Analyze why, adjust restoration, retry

### Step 7: Check for Regressions

```bash
# Run related tests to ensure no breakage
poetry run pytest ml/tests/unit/path/ -x
```

**Must:** No new failures introduced

## CONSTRAINTS (from CRITICAL_SAFEGUARDS.md)

**FORBIDDEN:**
- ❌ `raise NotImplementedError`
- ❌ `# TODO: implement`
- ❌ Empty function bodies with just `pass`
- ❌ Reverting entire files to phase0
- ❌ Removing facade/component structure
- ❌ Changing test expectations
- ❌ Modifying legacy code (if still in use)

**REQUIRED:**
- ✅ Minimal changes (smallest diff possible)
- ✅ Preserve new architectural patterns
- ✅ 100% type annotations
- ✅ Follow CLAUDE.md patterns
- ✅ Test must PASS (not just collect)

## OUTPUT FORMAT

Generate **RESTORATION_REPORT.md**:

```markdown
# Restoration Report: [test_name]

## Test Information
**Path:** ml/tests/path/test.py::test_name
**Status:** ✅ RESTORED (now passing)

## Failure Analysis
**Original error:**
```
[Error message]
```

**Root cause:** [What was missing]

## Restoration Applied

**Files modified:**
- ml/path/to/impl.py (lines 48)

**Changes:**

### ml/path/to/impl.py (Line 48)

**Added:**
```python
self._models: dict[str, ModelInfo] = {}
```

**Context:**
```python
class ModelRegistry:
    def __init__(self, engine: Engine, ...):
        super().__init__()
        self.engine = engine
        self._models: dict[str, ModelInfo] = {}  # ← RESTORED
```

**Why it's safe:**
- Minimal change (1 line)
- Doesn't alter new structure
- Preserves facade pattern (N/A for this file)
- Type-annotated
- No side effects

## Phase0 Reference

**From:** ../nautilus_trader-phase0/ml/registry/model_registry.py:48

```python
# phase0 had this exact line:
self._models: dict[str, ModelInfo] = {}
```

**Reason it was there:** Internal storage for registered model metadata

**Why it was removed:** Accidentally deleted during abstraction

**Why it's needed:** Test expects registry to track models internally

## Architectural Preservation

**New patterns preserved:**
- [x] Protocol-first design: N/A
- [x] Facade pattern: N/A
- [x] Component separation: N/A
- [x] Type annotations: YES (added)
- [x] CLAUDE.md compliance: YES

**No architectural regressions:** ✅ Confirmed

## Test Verification

**Command run:**
```bash
poetry run pytest ml/tests/unit/registry/test_model_registry.py::test_get_model_returns_info -xvs
```

**Output:**
```
======================== test session starts =========================
collected 1 item

ml/tests/unit/registry/test_model_registry.py::test_get_model_returns_info PASSED [100%]

======================== 1 passed in 0.45s ==========================
```

**Result:** ✅ PASS

## Regression Check

**Command run:**
```bash
poetry run pytest ml/tests/unit/registry/ -x
```

**Result:** No new failures (all tests that were passing still pass)

## Handoff to Phase 3

**Static validation needed:**
- Ruff check: ml/registry/model_registry.py
- MyPy check: ml/registry/model_registry.py
- Import verification: `python -c "from ml.registry import ModelRegistry"`

**Expected outcome:** All pass (minimal change, type-safe)

## Commit Message (if approved)

```
restore(registry): add _models dict to ModelRegistry

Restored internal state attribute that was removed during abstraction.
Test now passes.

Test: ml/tests/unit/registry/test_model_registry.py::test_get_model_returns_info
Status: ✅ PASSING

Preserves new architecture, no regressions.

Co-Authored-By: Claude <noreply@anthropic.com>
```
```

## READY TO EXECUTE

When you receive a DIFF_ANALYSIS_REPORT.md, you will:
1. Read the analysis
2. Read current implementation
3. Identify restoration location
4. Copy minimal piece from phase0
5. Restore with preservation of new structure
6. Add type annotations
7. Run test → verify PASS
8. Check regressions
9. Generate RESTORATION_REPORT.md
10. Hand off to Phase 3

**Your task:** Provide the DIFF_ANALYSIS_REPORT.md path.
```

---

## Phase 3-5: Same as Original Framework

Phases 3-5 remain identical to AGENT_TASK_FRAMEWORK.md:
- **Phase 3:** Static Validation (ruff, mypy, imports)
- **Phase 4:** Integration Validation (tests ACTUALLY RUN)
- **Phase 5:** System Validation (optional, for major changes)

See AGENT_TASK_FRAMEWORK.md for full prompts.

---

## Orchestration Strategy for 164 Failing Tests

### Sequential Processing (Per Category)

```
Restoration Orchestrator:
  1. Read restoration_taxonomy.md
  2. For each category in priority order:
     - Load test list (restoration_tasks/category_X_tests.txt)
     - For each test in list:
       a. Spawn Diff Analysis Agent (Phase 1)
       b. Spawn Minimal Restoration Agent (Phase 2)
       c. Spawn Static Validation Agent (Phase 3)
       d. Spawn Integration Validation Agent (Phase 4)
       e. If PASS: Commit and continue
       f. If FAIL: Escalate to human or retry with more context
```

### Parallel Processing (Across Categories)

```
Restoration Orchestrator (Parallel Mode):

  Spawn Category Agent #1 (missing_attributes):
    - Processes 15 tests sequentially (5-phase workflow each)
    - Commits after each success

  Spawn Category Agent #2 (import_errors):
    - Processes 20 tests sequentially (5-phase workflow each)
    - Commits after each success

  Spawn Category Agent #3 (file_not_found):
    - Processes 10 tests sequentially (5-phase workflow each)
    - Commits after each success

  (After mechanical fixes complete)

  Spawn Category Agent #4 (metrics_telemetry):
    - Processes 15 tests sequentially (5-phase workflow each)
    - Requires wiring analysis (more complex)
    - Commits after each success
```

**Why this works:**
- **No file conflicts:** Missing attributes ≠ import errors ≠ config paths
- **Independent execution:** Each category agent has isolated scope
- **Clear git history:** Each restoration is a separate commit
- **Fast recovery:** 3x speedup for mechanical fixes

---

## Progress Tracking

Create `restoration_dashboard.py`:

```python
#!/usr/bin/env python3
"""Real-time restoration progress tracking."""

import subprocess
from pathlib import Path
from collections import defaultdict

def get_category_progress():
    """Track restoration progress per category."""
    categories = [
        "missing_attributes",
        "import_errors",
        "file_not_found",
        "metrics_telemetry",
        "assertion_failures",
    ]

    progress = {}
    for cat in categories:
        test_file = Path(f"restoration_tasks/{cat}_tests.txt")
        if not test_file.exists():
            continue

        total = len(test_file.read_text().strip().split('\n'))

        # Count restoration commits for this category
        result = subprocess.run(
            ["git", "log", "--oneline", "--grep", f"restore.*{cat}"],
            capture_output=True,
            text=True
        )
        restored = len(result.stdout.strip().split('\n')) if result.stdout else 0

        progress[cat] = {
            "total": total,
            "restored": restored,
            "remaining": total - restored,
            "percent": (restored / total * 100) if total > 0 else 0
        }

    return progress

def print_dashboard():
    """Print restoration dashboard."""
    progress = get_category_progress()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          RESTORATION PROGRESS DASHBOARD                  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    for cat, stats in progress.items():
        bar_length = 40
        filled = int(stats["percent"] / 100 * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"{cat:.<30} [{bar}] {stats['percent']:.1f}%")
        print(f"  {stats['restored']}/{stats['total']} restored, {stats['remaining']} remaining\n")

    # Overall stats
    total_tests = sum(s["total"] for s in progress.values())
    total_restored = sum(s["restored"] for s in progress.values())
    overall_percent = (total_restored / total_tests * 100) if total_tests > 0 else 0

    print(f"\n{'OVERALL':.<30} {total_restored}/{total_tests} ({overall_percent:.1f}%)\n")

if __name__ == "__main__":
    print_dashboard()
```

Run with:
```bash
chmod +x tools/restoration_dashboard.py
watch -n 60 python tools/restoration_dashboard.py
```

---

## Success Metrics

**Phase 0 (Today):**
- [ ] PostgreSQL running
- [ ] Categorization complete (restoration_taxonomy.md exists)
- [ ] Decision made (GO / NO-GO)

**Phase 1 (Day 1-2):**
- [ ] Category agents launched (missing_attributes, import_errors, file_not_found)
- [ ] First 10 tests restored
- [ ] Zero architectural regressions

**Phase 2 (Day 3):**
- [ ] 40+ tests restored (HIGH confidence categories complete)
- [ ] Metrics/telemetry agent launched
- [ ] Pass rate improved to >98%

**Phase 3 (Day 4-5):**
- [ ] Human review queue triaged
- [ ] Behavioral failures resolved or documented
- [ ] Final pass rate >97.5% (target: 5276/5410 tests)

**Final Validation:**
- [ ] `make pytest-ml` passes
- [ ] `make validate-nautilus-patterns` passes
- [ ] No circular dependencies
- [ ] No god class regressions
- [ ] Architectural improvements preserved

---

## Rollback Criteria (from CRITICAL_SAFEGUARDS.md)

**Abort restoration if:**
1. ❌ After 2 days, <30 tests restored (too slow)
2. ❌ Restoration introduces circular dependencies
3. ❌ Restoration reverts architectural improvements
4. ❌ Pass rate decreases (regressions introduced)
5. ❌ Time exceeds 5 days total

**Rollback procedure:**
```bash
# Preserve attempt
git branch feat/strategy-integration-restoration-attempt-$(date +%Y%m%d)

# Reset to pre-refactoring state
git reset --hard $(git merge-base develop HEAD)

# Cherry-pick successful pieces from phase0
# (manual review required)
```

---

## Next Steps

**RIGHT NOW (15 minutes):**

```bash
# 1. Fix PostgreSQL
docker-compose up -d postgres

# 2. Run categorization
poetry run pytest ml/tests --tb=line -q > test_output.txt 2>&1
python tools/analyze_test_failures.py test_output.txt

# 3. Review taxonomy
cat restoration_taxonomy.md

# 4. Make decision
```

**If GO (Day 1):**

```bash
# Launch first restoration agent (missing attributes)
# You: "Launch restoration agent for missing_attributes category"
# Provide:
#   - Test list: restoration_tasks/missing_attributes_tests.txt
#   - Agent prompts: This framework document
#   - Phase0 reference: ../nautilus_trader-phase0
```

**If NO-GO:**

```bash
# Strategic rollback to phase0
# Document lessons learned
# Plan better approach
```

---

## Key Advantages of This Adapted Framework

1. ✅ **Proven workflow:** Same 5-phase validation as original
2. ✅ **Restoration-specific:** Phase 1 analyzes diffs, Phase 2 restores minimally
3. ✅ **Preserves architecture:** Explicitly checks for architectural regressions
4. ✅ **Enforces safeguards:** CRITICAL_SAFEGUARDS.md built into every phase
5. ✅ **Parallelizable:** Categories can be processed concurrently
6. ✅ **Audit trail:** Every restoration documented with diff + justification
7. ✅ **Safe rollback:** Each restoration is a separate commit

**This framework transforms your restoration from a slog into a systematic, trackable, parallelizable process.**

---

**Ready to launch? Your call.**
