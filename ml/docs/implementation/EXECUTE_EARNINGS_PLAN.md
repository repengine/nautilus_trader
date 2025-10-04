# Execute Earnings Integration Plan - Quick Reference

## Status: ~60% Complete → Target: 100%

**Remaining Work:** 10 tasks, 20-29 hours (3-4 days)

---

## Quick Start

### Review Documents (Do This First!)
1. ✅ Read refactor status: Agent already reviewed progress
2. ✅ Read task breakdown: `playground/EARNINGS_TASK_BREAKDOWN.md`
3. ✅ Understand architecture: `playground/EARNINGS_ARCHITECTURE_DESIGN.md`

### Execute Plan (Use These Commands)

Each task follows this pattern:
```bash
# 1. Assign implementation agent
# 2. Agent reads docs and implements
# 3. Assign validation agent
# 4. Validation agent checks and approves
```

---

## Task Execution Commands

### Phase 1: Critical Path (Sequential Execution)

#### Task 1: FileEarningsStore (2-4 hours)
```
Assign implementation agent to:
- Read: playground/EARNINGS_TASK_BREAKDOWN.md (Task 1 section)
- Read: ml/stores/protocols.py (EarningsStoreProtocol)
- Read: ml/stores/earnings_store.py (reference implementation)
- Implement: FileEarningsStore class in ml/stores/file_backed.py
- Follow: Template and requirements from task breakdown

Then assign validation agent to:
- Run: mypy --strict ml/stores/file_backed.py
- Run: ruff check ml/stores/file_backed.py
- Verify: Protocol compliance
- Report: Validation results
```

#### Task 2: FileDataStore Earnings Methods (1-2 hours)
**⚠️ Depends on Task 1**
```
Assign implementation agent to:
- Read: playground/EARNINGS_TASK_BREAKDOWN.md (Task 2 section)
- Read: ml/stores/data_store.py (reference implementation)
- Modify: FileDataStore class in ml/stores/file_backed.py
- Add: 4 earnings methods
- Follow: Template from task breakdown

Then assign validation agent to:
- Run: mypy --strict ml/stores/file_backed.py
- Run: ruff check ml/stores/file_backed.py
- Run: pytest ml/tests/unit/stores/ -k file
- Verify: DataStoreFacadeProtocol compliance
- Report: Validation results
```

#### Task 3: Integration Manager Wiring (1 hour)
**⚠️ Depends on Tasks 1 & 2**
```
Assign implementation agent to:
- Read: playground/EARNINGS_TASK_BREAKDOWN.md (Task 3 section)
- Read: ml/core/integration.py (_init_stores method)
- Modify: _init_stores method in ml/core/integration.py
- Add: FileEarningsStore initialization in file fallback path
- Follow: Template from task breakdown

Then assign validation agent to:
- Run: mypy --strict ml/core/integration.py
- Run: ruff check ml/core/integration.py
- Run: pytest ml/tests/unit/core/test_integration.py
- Report: Validation results
```

#### Task 4 & 5: Parallel Execution

**Task 4: TFT Builder Integration (4-6 hours)**
```
Assign implementation agent to:
- Read: playground/EARNINGS_TASK_BREAKDOWN.md (Task 4 section)
- Read: playground/EARNINGS_ARCHITECTURE_DESIGN.md (TFT section)
- Read: ml/data/tft_dataset_builder.py (existing code)
- Implement: _fetch_earnings_features() method
- Add: Earnings integration in _process_symbol_polars() and _process_symbol_pandas()
- Follow: Detailed template from task breakdown

Then assign validation agent to:
- Run: mypy --strict ml/data/tft_dataset_builder.py
- Run: ruff check ml/data/tft_dataset_builder.py
- Run: pytest ml/tests/unit/data/ -k tft
- Report: Validation results
```

**Task 5: DataRegistry Contracts (2-3 hours)**
```
Assign implementation agent to:
- Read: playground/EARNINGS_TASK_BREAKDOWN.md (Task 5 section)
- Read: ml/registry/dataclasses.py (DataContract structure)
- Modify: ml/registry/data_registry.py
- Add: EARNINGS_ACTUALS_CONTRACT and EARNINGS_ESTIMATES_CONTRACT
- Register: Contracts in DataRegistry initialization
- Follow: Template from task breakdown

Then assign validation agent to:
- Run: mypy --strict ml/registry/data_registry.py
- Run: ruff check ml/registry/data_registry.py
- Run: pytest ml/tests/unit/registry/ -k data_registry
- Report: Validation results
```

#### Task 6: DataStore Tests (4-6 hours)
**⚠️ Depends on Tasks 1-5**
```
Assign implementation agent to:
- Read: playground/EARNINGS_TASK_BREAKDOWN.md (Task 6 section)
- Read: playground/EARNINGS_INTEGRATION_SUMMARY.md (Testing section)
- Create: ml/tests/unit/stores/test_data_store_earnings.py
- Create: ml/tests/integration/test_earnings_datastore_integration.py
- Implement: Unit and integration tests from templates
- Follow: Test patterns from task breakdown

Then assign validation agent to:
- Run: pytest ml/tests/unit/stores/test_data_store_earnings.py -v
- Run: pytest ml/tests/integration/test_earnings_datastore_integration.py -v
- Verify: Coverage ≥90%
- Verify: Performance benchmarks meet SLA
- Report: Validation results
```

---

### Phase 2: Documentation (Parallel Execution)

#### Task 7: Integration Guide (2-3 hours)
```
Assign implementation agent to:
- Read: playground/EARNINGS_TASK_BREAKDOWN.md (Task 7 section)
- Read: playground/EARNINGS_ARCHITECTURE_DESIGN.md (full)
- Create: ml/docs/guides/earnings_integration_guide.md
- Include: Overview, ingestion guide, query guide, code examples
- Follow: Template from task breakdown

Then assign validation agent to:
- Verify: All code examples work
- Verify: Links are correct
- Verify: Markdown formatting is correct
- Report: Validation results
```

#### Task 8: Deprecation Warnings (1 hour)
```
Assign implementation agent to:
- Read: playground/EARNINGS_TASK_BREAKDOWN.md (Task 8 section)
- Modify: ml/stores/earnings_store.py
- Add: Deprecation warning in EarningsStore.__init__()
- Follow: Template from task breakdown

Then assign validation agent to:
- Run: ruff check ml/stores/earnings_store.py
- Verify: Warning is emitted
- Report: Validation results
```

#### Task 9: Migration Guide (1-2 hours)
```
Assign implementation agent to:
- Read: playground/EARNINGS_TASK_BREAKDOWN.md (Task 9 section)
- Read: playground/EARNINGS_INTEGRATION_SUMMARY.md (Migration section)
- Create: ml/docs/migrations/earnings_datastore_migration.md
- Include: Why migrate, before/after examples, timeline, FAQ
- Follow: Template from task breakdown

Then assign validation agent to:
- Verify: All code examples work
- Verify: Timeline is clear
- Report: Validation results
```

#### Task 10: Architecture Docs (1 hour)
```
Assign implementation agent to:
- Read: playground/EARNINGS_TASK_BREAKDOWN.md (Task 10 section)
- Modify: ml/docs/architecture/universal_patterns_guide.md
- Modify: ml/docs/development/CODING_STANDARDS.md
- Add: Earnings examples to relevant sections
- Follow: Content templates from task breakdown

Then assign validation agent to:
- Verify: Examples are correct
- Verify: No breaking changes to existing content
- Report: Validation results
```

---

## Agent Command Templates

### Implementation Agent Command
```
Task an agent to:

"Your task is to implement [TASK NAME] for the earnings integration refactor.

## Required Reading (Read these FIRST)
1. playground/EARNINGS_TASK_BREAKDOWN.md - Read [TASK NUMBER] section completely
2. playground/EARNINGS_ARCHITECTURE_DESIGN.md - Read [RELEVANT SECTIONS]
3. ml/docs/development/CODING_STANDARDS.md - Read full document
4. ml/docs/architecture/universal_patterns_guide.md - Read [RELEVANT PATTERNS]
5. [ADDITIONAL FILES TO READ]

## Your Task
[COPY TASK DESCRIPTION FROM TASK BREAKDOWN]

## Implementation Steps
1. Read all required documentation (listed above)
2. Create an implementation plan with:
   - File path to modify: [FILE PATH]
   - Changes to make: [SPECIFIC CHANGES]
   - Code structure: [CLASS/FUNCTION NAMES]
   - Type hints and docstrings: [REQUIREMENTS]
3. Implement the changes following:
   - Template code from task breakdown
   - Coding standards from CODING_STANDARDS.md
   - Architectural patterns from universal_patterns_guide.md
4. Self-validate:
   - Check type hints are complete
   - Check docstrings are comprehensive
   - Check error handling is present
   - Check follows existing code patterns
5. Report completion with:
   - File modified: [PATH]
   - Lines changed: [LINE NUMBERS]
   - Summary of changes
   - Any issues encountered

IMPORTANT: Follow the exact template and requirements from the task breakdown. Do not deviate from the architectural patterns."
```

### Validation Agent Command
```
Task an agent to:

"Your task is to validate the implementation of [TASK NAME] for earnings integration.

## Files to Validate
[LIST FILES MODIFIED BY IMPLEMENTATION AGENT]

## Validation Steps

1. **Static Analysis**
   Run these commands and report results:
   ```bash
   mypy --strict [MODIFIED_FILE]
   ruff check [MODIFIED_FILE]
   ```

2. **Protocol Compliance** (if applicable)
   - Verify class implements required protocol
   - Check all protocol methods have correct signatures
   - Use isinstance() checks to verify

3. **Tests**
   Run these commands and report results:
   ```bash
   pytest [RELATED_TEST_FILES] -v
   pytest ml/tests/unit/ -k [KEYWORD]
   pytest ml/tests/integration/ -k [KEYWORD]
   ```

4. **Performance** (if applicable)
   - Run performance benchmarks
   - Verify P99 latency < 10ms for queries
   - Report actual measurements

5. **Code Quality**
   - Check all functions have docstrings
   - Check all parameters have type hints
   - Check error handling is present
   - Check follows coding standards

## Validation Report Format

```markdown
# Validation Report: [TASK NAME]

## Static Analysis
- mypy --strict: [PASSED/FAILED - details]
- ruff check: [PASSED/FAILED - details]

## Tests
- Unit tests: [X/X passed - details]
- Integration tests: [X/X passed - details]

## Protocol Compliance
- Implements [PROTOCOL]: [YES/NO - details]

## Performance (if applicable)
- P99 latency: [X.X ms - target <10ms]

## Code Quality
- Docstrings: [COMPLETE/INCOMPLETE]
- Type hints: [COMPLETE/INCOMPLETE]
- Error handling: [PRESENT/MISSING]

## Issues Found
[List any issues, or write "None"]

## Recommendation
[APPROVED/NEEDS REVISION - explanation]
```

IMPORTANT: Be thorough. Report ALL issues found. Do not approve if there are mypy errors or test failures."
```

---

## Progress Tracking Checklist

### Phase 1: Critical Path
- [ ] Task 1: FileEarningsStore ✅ Implementation ✅ Validation
- [ ] Task 2: FileDataStore earnings methods ✅ Implementation ✅ Validation
- [ ] Task 3: Integration manager wiring ✅ Implementation ✅ Validation
- [ ] Task 4: TFT builder integration ✅ Implementation ✅ Validation
- [ ] Task 5: DataRegistry contracts ✅ Implementation ✅ Validation
- [ ] Task 6: DataStore tests ✅ Implementation ✅ Validation

### Phase 2: Documentation
- [ ] Task 7: Integration guide ✅ Implementation ✅ Validation
- [ ] Task 8: Deprecation warnings ✅ Implementation ✅ Validation
- [ ] Task 9: Migration guide ✅ Implementation ✅ Validation
- [ ] Task 10: Architecture docs ✅ Implementation ✅ Validation

### Final Verification
- [ ] All mypy checks pass
- [ ] All ruff checks pass
- [ ] All tests pass
- [ ] Performance benchmarks meet SLA
- [ ] Documentation complete
- [ ] Architecture compliance verified

---

## Common Issues & Solutions

### Issue: Agent doesn't follow template
**Solution:** Be very explicit in the task description. Copy the exact template from task breakdown and say "Follow this template EXACTLY"

### Issue: Validation agent approves broken code
**Solution:** Make validation criteria more explicit. Require specific command outputs, not just "check if it works"

### Issue: Tasks overlap (file conflicts)
**Solution:** Refer to dependency graph in task breakdown. Never run dependent tasks in parallel

### Issue: Agent forgets to read documentation
**Solution:** List documentation as "Required Reading (Read these FIRST)" and make it the first step in the task

### Issue: Mypy errors after implementation
**Solution:** Ask validation agent to run mypy and report ALL errors. Do not approve until zero errors

---

## Estimated Timeline

### Aggressive (3 days)
- Day 1: Tasks 1-3 (critical path start)
- Day 2: Tasks 4-6 (critical path complete)
- Day 3: Tasks 7-10 (documentation parallel)

### Conservative (4 days)
- Day 1: Tasks 1-2
- Day 2: Tasks 3-4
- Day 3: Tasks 5-6
- Day 4: Tasks 7-10

### Realistic (4-5 days including validation iterations)
- Day 1: Task 1 (impl + validation + fixes)
- Day 2: Tasks 2-3 (impl + validation + fixes)
- Day 3: Tasks 4-5 parallel (impl + validation)
- Day 4: Task 6 (impl + validation + fixes)
- Day 5: Tasks 7-10 parallel (docs + validation)

---

## Final Success Criteria

Before declaring the refactor complete, verify:

✅ **Architectural Compliance**
- [ ] FileEarningsStore implements EarningsStoreProtocol
- [ ] FileDataStore implements DataStoreFacadeProtocol
- [ ] Progressive fallback works: PostgreSQL → File → Dummy
- [ ] TFT builder uses DataStore facade (no direct EarningsStore)
- [ ] DataRegistry validates earnings contracts

✅ **Code Quality**
- [ ] Zero mypy --strict errors
- [ ] Zero ruff violations
- [ ] All tests pass (unit + integration)
- [ ] Test coverage ≥90% on new code
- [ ] Performance benchmarks meet SLA (P99 < 10ms)

✅ **Documentation**
- [ ] Integration guide complete
- [ ] Migration guide published
- [ ] Architecture docs updated
- [ ] Deprecation warnings added
- [ ] All code examples work

✅ **Integration**
- [ ] Actors can access earnings via self.data_store
- [ ] TFT builder includes earnings features correctly
- [ ] Point-in-time correctness verified
- [ ] ASOF join logic correct
- [ ] File fallback mode works

When all checkboxes are ✅, the refactor is **COMPLETE**.
