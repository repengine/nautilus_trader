# Earnings Integration Refactor - Master Index

## 📚 Documentation Hub

This is the master index for the earnings integration refactor. All documents are organized by purpose and reading order.

---

## 🚀 Quick Start (Read These First)

### 1. **Refactor Summary** → [`EARNINGS_REFACTOR_SUMMARY.md`](EARNINGS_REFACTOR_SUMMARY.md)
**Purpose:** Executive overview and quick reference
**Read Time:** 5 minutes
**You'll Learn:**
- Current status (~60% complete)
- What's done and what's missing
- Why this refactor matters
- How to get started

### 2. **Execution Plan** → [`EXECUTE_EARNINGS_PLAN.md`](EXECUTE_EARNINGS_PLAN.md)
**Purpose:** Step-by-step guide to execute the refactor
**Read Time:** 10 minutes
**You'll Learn:**
- How to assign agents to tasks
- Command templates for implementation and validation
- Progress tracking checklist
- Common issues and solutions

---

## 📋 Planning Documents (Read Before Implementing)

### 3. **Architecture Design** → [`EARNINGS_ARCHITECTURE_DESIGN.md`](EARNINGS_ARCHITECTURE_DESIGN.md)
**Purpose:** Complete technical design specification
**Read Time:** 30 minutes
**You'll Learn:**
- Current state analysis and architectural violations
- Two-layer integration model (DataStore + FeatureStore)
- Implementation details with code examples
- Migration path and testing strategy
- Files to modify with line numbers

### 4. **Architecture Diagrams** → [`EARNINGS_ARCHITECTURE_DIAGRAM.md`](EARNINGS_ARCHITECTURE_DIAGRAM.md)
**Purpose:** Visual representations of data flow
**Read Time:** 15 minutes
**You'll Learn:**
- Current vs. target state comparison
- Progressive fallback flow diagrams
- End-to-end data flow example
- Layer separation (raw data vs. computed features)

### 5. **Integration Summary** → [`EARNINGS_INTEGRATION_SUMMARY.md`](EARNINGS_INTEGRATION_SUMMARY.md)
**Purpose:** Executive summary with code examples
**Read Time:** 20 minutes
**You'll Learn:**
- Problem statement and solution
- Key design decisions
- Implementation plan (Phases 1-3)
- Code examples for all use cases
- Testing strategy
- Success criteria

---

## 🛠️ Implementation Documents (Use During Execution)

### 6. **Task Breakdown** → [`EARNINGS_TASK_BREAKDOWN.md`](EARNINGS_TASK_BREAKDOWN.md)
**Purpose:** Detailed task specifications for agent execution
**Read Time:** 45 minutes (reference document)
**Contains:**
- **10 discrete tasks** with no file overlap
- **Agent instructions** for each task (what to read, what to implement)
- **Code templates** to follow
- **Validation criteria** for each task
- **Task dependency graph**

**Tasks Overview:**
- **Critical Path (Tasks 1-6):** Must complete sequentially
  - Task 1: FileEarningsStore (2-4 hours)
  - Task 2: FileDataStore earnings methods (1-2 hours)
  - Task 3: Integration manager wiring (1 hour)
  - Task 4: TFT builder integration (4-6 hours)
  - Task 5: DataRegistry contracts (2-3 hours)
  - Task 6: DataStore tests (4-6 hours)

- **Documentation (Tasks 7-10):** Can execute in parallel
  - Task 7: Integration guide (2-3 hours)
  - Task 8: Deprecation warnings (1 hour)
  - Task 9: Migration guide (1-2 hours)
  - Task 10: Architecture docs (1 hour)

---

## 🔍 Reference Documents (Keep Open While Working)

### 7. **Original Investigation Plan** → [`EARNINGS_INTEGRATION_PLAN.md`](EARNINGS_INTEGRATION_PLAN.md)
**Purpose:** Original investigation report (historical reference)
**Read Time:** 60 minutes
**Contains:**
- How macro/micro/L2 features currently work
- Pattern analysis for TFT dataset builder
- Missing value handling and point-in-time correctness
- Original implementation pseudocode

### 8. **Coding Standards** → [`../ml/docs/development/CODING_STANDARDS.md`](../ml/docs/development/CODING_STANDARDS.md)
**Purpose:** Project coding standards (MUST follow)
**Read Time:** 30 minutes
**Contains:**
- Schema adherence rules
- Type annotation requirements
- Testing and coverage requirements
- Linting and formatting standards

### 9. **Universal Patterns Guide** → [`../ml/docs/architecture/universal_patterns_guide.md`](../ml/docs/architecture/universal_patterns_guide.md)
**Purpose:** ML architecture patterns (MUST follow)
**Read Time:** 45 minutes
**Contains:**
- Pattern 1: 4-Store + 4-Registry Integration
- Pattern 2: Protocol-First Interface Design
- Pattern 3: Hot/Cold Path Separation
- Pattern 4: Progressive Fallback Chains
- Pattern 5: Centralized Metrics Bootstrap

---

## 📊 Status & Progress Tracking

### Current Status: ~90% Complete

**✅ Completed (90%)**
- Raw data storage protocols and DataStore implementation
- Feature computation (all functions + TransformSpecs)
- PostgreSQL → DummyStore fallback
- Unit tests for feature computation
- Integration + data-quality tests for earnings pipelines
- DataStore earnings façade adoption across unit/integration/perf suites
- Performance benchmarks (ml/tests/performance/test_earnings_performance.py) back within SLA without allocations
- Core developer docs updated (`ml/features/README_EARNINGS.md`, `ml/data/earnings/README.md`)
- FileEarningsStore + FileDataStore earnings methods wired for progressive fallback
- MLIntegrationManager + DataStore fallback chain now initialize file-backed earnings store
- Earnings contracts seeded in registry bootstrap + JSON backend normalisation
- File-backed/unit tests covering earnings round-trip & fallback (`ml/tests/unit/stores/test_file_backed_store.py`, `test_data_store_earnings_fallback.py`)

- Documentation refresh for execution/migration guides (playground docs still reference legacy flows)

- TFT builder DataStore integration (earnings feature join)

### Execution Timeline

**Aggressive:** 3 days
**Conservative:** 4 days
**Realistic:** 4-5 days (including validation iterations)

---

## 🎯 How to Use This Index

### If You're Starting Fresh
1. Read **Refactor Summary** (5 min) → Get oriented
2. Read **Execution Plan** (10 min) → Understand process
3. Read **Architecture Design** (30 min) → Understand what to build
4. Read **Task Breakdown** for Task 1 (15 min) → Start implementing
5. Assign agents using templates from **Execution Plan**

### If You're Validating Work
1. Find the task in **Task Breakdown**
2. Check validation criteria section
3. Use validation agent template from **Execution Plan**
4. Run all checks and generate report

### If You Need Architecture Guidance
1. Check **Architecture Design** for detailed explanation
2. Check **Architecture Diagrams** for visual reference
3. Check **Universal Patterns Guide** for pattern details
4. Check **Coding Standards** for implementation rules

### If You're Writing Documentation
1. Read **Integration Summary** for code examples
2. Check Tasks 7-10 in **Task Breakdown** for templates
3. Follow markdown formatting standards
4. Validate all code examples work

---

## 📁 File Structure

```
playground/
├── EARNINGS_REFACTOR_INDEX.md              ← YOU ARE HERE (master index)
├── EARNINGS_REFACTOR_SUMMARY.md            ← Start here (executive summary)
├── EXECUTE_EARNINGS_PLAN.md                ← Execution guide (step-by-step)
├── EARNINGS_TASK_BREAKDOWN.md              ← Task details (agent instructions)
├── EARNINGS_ARCHITECTURE_DESIGN.md         ← Technical design (what to build)
├── EARNINGS_ARCHITECTURE_DIAGRAM.md        ← Visual diagrams (data flow)
├── EARNINGS_INTEGRATION_SUMMARY.md         ← Code examples (reference)
└── EARNINGS_INTEGRATION_PLAN.md            ← Original investigation (historical)

ml/docs/
├── development/CODING_STANDARDS.md         ← Coding standards (must follow)
└── architecture/universal_patterns_guide.md ← ML patterns (must follow)

ml/stores/
├── protocols.py                            ← Store protocols (reference)
├── data_store.py                           ← DataStore implementation (reference)
├── earnings_store.py                       ← EarningsStore (existing)
└── file_backed.py                          ← File stores (to modify)

ml/data/
└── tft_dataset_builder.py                  ← TFT builder (to modify)

ml/registry/
└── data_registry.py                        ← DataRegistry (to modify)

ml/tests/
├── unit/stores/                            ← Unit tests (to create)
└── integration/                            ← Integration tests (to create)
```

---

## 🔑 Key Concepts

### Two-Layer Integration Model

**Layer 1: Raw Data → DataStore + DataRegistry**
- Earnings actuals/estimates are **raw inputs** (like bars, ticks)
- Stored via `DataStore.write_earnings_actual()`
- Validated via `DataRegistry` contracts
- Point-in-time correctness at data layer

**Layer 2: Computed Features → FeatureStore + FeatureRegistry**
- Earnings surprise/growth/momentum are **computed features** (like RSI, MACD)
- Computed via `ml.features.earnings` functions
- Stored via `FeatureStore.write_features()`
- Validated via `FeatureRegistry` schemas

### Progressive Fallback Chain

```
Production:  PostgreSQL EarningsStore
Testing:     FileEarningsStore (Parquet files)
CI/Dummy:    DummyEarningsStore (in-memory dict)
```

Every component must support all three modes.

### Protocol-First Design

```python
# Define protocol (interface)
class EarningsStoreProtocol(Protocol):
    def get_actuals(...) -> list[dict]: ...

# Implementations conform without inheritance
class FileEarningsStore:  # No base class
    def get_actuals(...) -> list[dict]: ...  # Conforms

# Type checking validates protocol compliance
isinstance(store, EarningsStoreProtocol)  # True
```

### Point-in-Time Correctness

```python
# Backtest at 2024-04-01
actuals = data_store.get_earnings_actuals_at_or_before(
    ticker="AAPL",
    ts_event=1711929600000000000,  # 2024-04-01 in nanoseconds
    limit=5,
)
# Only returns filings with ts_event < 2024-04-01
# Future filings are invisible (no look-ahead bias)
```

---

## ✅ Pre-Flight Checklist

Before starting implementation, ensure:

- [ ] Read **Refactor Summary** (understand what we're doing)
- [ ] Read **Execution Plan** (understand how to execute)
- [ ] Read **Architecture Design** (understand what to build)
- [ ] Read **Coding Standards** (understand quality requirements)
- [ ] Read **Universal Patterns Guide** (understand architectural patterns)
- [ ] Have agent system available (for parallel execution)
- [ ] Have validation tools installed (mypy, ruff, pytest)
- [ ] Have test database available (PostgreSQL or Docker)

---

## 🚦 Execution Workflow

### Phase 1: Critical Path (Sequential)

```
┌─────────────────────────────────────────────────────┐
│ Task 1: FileEarningsStore                           │
│ Agent reads docs → Plans → Implements → Reports     │
└─────────────────────────┬───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ Validation: mypy + ruff + tests + approval          │
└─────────────────────────┬───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ Task 2: FileDataStore earnings methods              │
│ Agent reads docs → Plans → Implements → Reports     │
└─────────────────────────┬───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ Validation: mypy + ruff + tests + approval          │
└─────────────────────────┬───────────────────────────┘
                          ↓
                       [Continue...]
```

### Phase 2: Documentation (Parallel)

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Task 7     │  │   Task 8     │  │   Task 9     │  │   Task 10    │
│ Integration  │  │ Deprecation  │  │  Migration   │  │ Architecture │
│    Guide     │  │   Warnings   │  │    Guide     │  │     Docs     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       └─────────────────┴─────────────────┴─────────────────┘
                                 ↓
                         All validate in parallel
```

---

## 🎓 Learning Resources

### For Understanding Architecture
- Read: **Architecture Design** (complete technical spec)
- Read: **Architecture Diagrams** (visual data flow)
- Read: **Universal Patterns Guide** (ML architecture patterns)

### For Implementation Guidance
- Read: **Task Breakdown** (detailed task specs)
- Read: **Coding Standards** (quality requirements)
- Reference: Existing implementations in codebase

### For Code Examples
- Read: **Integration Summary** (actor, TFT builder examples)
- Reference: `ml/stores/data_store.py` (DataStore implementation)
- Reference: `ml/features/earnings/` (feature computation)

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** Agent doesn't follow template
→ Solution: Be explicit, copy exact template, say "Follow EXACTLY"

**Issue:** Validation fails with mypy errors
→ Solution: Read error carefully, fix type hints, re-validate

**Issue:** Tests fail after implementation
→ Solution: Check test output, fix implementation, ensure backward compatibility

**Issue:** Performance benchmarks fail
→ Solution: Profile code, optimize hot paths, add caching

**Issue:** File conflicts between agents
→ Solution: Follow dependency graph, never run dependent tasks in parallel

### Getting Help

1. Check relevant section in **Task Breakdown**
2. Review **Architecture Design** for guidance
3. Look at reference implementations
4. Check **Coding Standards** for patterns
5. Ask for clarification before implementing

---

## 📈 Success Metrics

### Definition of Done

✅ All 10 tasks completed and validated
✅ Zero architectural violations
✅ Zero mypy --strict errors
✅ Zero ruff violations
✅ All tests pass (unit + integration)
✅ Test coverage ≥90% on new code
✅ Performance benchmarks meet SLA
✅ Documentation complete
✅ Migration path documented
✅ Backward compatibility maintained

### Final Verification Checklist

- [ ] FileEarningsStore implements EarningsStoreProtocol
- [ ] FileDataStore implements DataStoreFacadeProtocol
- [ ] Progressive fallback: PostgreSQL → File → Dummy
- [ ] TFT builder uses DataStore (no direct EarningsStore)
- [ ] DataRegistry validates earnings contracts
- [ ] Actors access earnings via self.data_store
- [ ] Point-in-time correctness verified
- [ ] ASOF join logic correct
- [ ] File fallback mode works
- [ ] All code examples in docs work

---

## 🎉 Completion Criteria

When you can check all boxes above, the refactor is **COMPLETE** and earnings data is:

✅ Integrated with Universal ML Architecture (4-store + 4-registry)
✅ Validated by DataRegistry contracts
✅ Tracked with full lineage
✅ Available in all environments (PostgreSQL, file, in-memory)
✅ Accessible to actors via standard protocols
✅ Integrated with TFT builder
✅ Fully tested (≥90% coverage)
✅ Completely documented

**The result:** A clean, observable, type-safe data flow that treats earnings as a first-class citizen in the ML architecture.

---

## 🚀 Ready to Begin?

**Next Step:** Open [`EXECUTE_EARNINGS_PLAN.md`](EXECUTE_EARNINGS_PLAN.md) and start with Task 1!

Good luck! 🎯
