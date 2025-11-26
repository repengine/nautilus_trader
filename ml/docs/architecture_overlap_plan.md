# Architecture Consolidation & DRY Plan

> **⚠️ SUPERSEDED:** This plan has been merged into the unified consolidation roadmap.
> See: `reports/CONSOLIDATION_PLAN.md` (Phase 6: Component Consolidation)
>
> **Status as of 2024-11-25:**
> - Phases 1-5 (Import Discipline) ✅ Complete
> - Phase 6 (Component Consolidation) 🔲 Ready to start
> - Phase 7 (Legacy Deletion) 🔲 Blocked on Phase 6

---

*Original plan preserved below for reference:*

Plan to converge legacy/root modules with extracted `components/` implementations across domains (registry, orchestration, stores, etc.). Goal: one canonical implementation per concern, public APIs exposed via domain `__init__.py`, internal code kept private, and callers routed through the public surface.

## Context
- God-class decomposition produced `components/` folders; legacy root modules often remain.
- Some components are fully implemented (registry), others are placeholders (orchestration).
- Backup/phase folders add noise and risk accidental imports.
- Strategy: pick a canonical per concern, shim/deprecate the duplicate, expose only the canonical surface via `__init__.py`, and migrate callers to that surface.

## Candidates (from inventory + fuzzy matching)
- **Registry**
  - `ab_testing_manager.py` ↔ `components/ab_testing.py`
  - `persistence.py` ↔ `components/data_persistence.py`
  - `model_persistence.py` ↔ `components/model_persistence.py`
  - `event_manager.py` ↔ `components/event_emission.py`
  - `manifest_manager.py` ↔ `components/manifest_manager.py`
  - `lineage_manager.py` ↔ `components/lineage_tracker.py`
  - `model_deployment_mgr.py` / `deployment_manager.py` ↔ `components/deployment_manager.py`
  - `version_manager.py` ↔ `components/version_manager.py` (parity check)
- **Orchestration** (and `orchestration_backup_phase2`)
  - `ingestion_coordinator.py` ↔ `components/training_coordinator.py`
  - `discovery_client.py` ↔ `components/discovery_service.py`
  - `config_loader.py` / `binding_resolver.py` ↔ `components/config_resolver.py`
- **Stores** (and `stores_backup_phase3`)
  - `raw_protocols.py` ↔ `components/protocols.py`
  - `data_reader.py` ↔ `components/data_writer.py`
  - `data_store_facade.py` ↔ `components/data_reader.py`
  - `schema_audit.py` ↔ `components/schema_validator.py`
- Tests mirror these overlaps; will need aligning once canonical paths are chosen.

## Findings to Date
- **Registry A/B testing**: `components/ab_testing.py` is lock/persistence/policy-aware; `ab_testing_manager.py` is a lighter, dict-based duplicate. Canonical should be the component; legacy manager should shim or retire.
- **Registry persistence layering**: `persistence.py` provides backend plumbing; `components/data_persistence.py` holds data-registry state with locking/timers. Not duplication—layering is intentional.
- **Registry model persistence**: `model_persistence.py` (richer cache/SHA logic) vs `components/model_persistence.py` (componentized persistence with locking). Needs an API diff to pick canonical or merge; likely keep component as wrapper and lift missing features from the root class.
- **Registry events/manifests/lineage/deployment/version**: roots vs components handle the same concerns; components tend to enforce locking/persistence, roots operate on loose dicts. Bias toward components; map gaps (e.g., emissions, pipeline signatures) before shimming roots.
- **Orchestration**: components are structural placeholders; roots carry the working logic. Canonical today is the root; either implement components from root logic or deprecate placeholders to avoid dual APIs.
- **Stores**: fuzzy overlaps are largely orthogonal (raw protocols vs component protocols; reader vs writer; schema audit vs validator). Likely no consolidation needed; clarify boundaries to avoid misuse.

## Status Tracker (by domain/pair)
- Registry
  - [x] A/B testing: canonical = `components/ab_testing.py`; root to shim/deprecate.
  - [ ] Model persistence: needs API diff; target canonical = component, with cache/SHA features lifted if needed.
  - [ ] Deployment/version managers: needs API diff; likely component canonical.
  - [ ] Manifest manager: component canonical; verify events/lock parity, then shim root.
  - [ ] Lineage tracker: component canonical; verify pipeline signature helpers, then shim root.
  - [ ] Event emission: component canonical; ensure parity, then shim root.
  - [ ] Public surface: define exports in `ml/registry/__init__.py`; migrate callers/tests.
- Orchestration
  - [ ] Decide: implement components from root logic or deprecate placeholders; avoid dual APIs.
  - [ ] Public surface: define exports in `ml/orchestration/__init__.py`; migrate callers/tests.
- Stores
  - [ ] Confirm no consolidation needed; document boundaries (protocols vs raw protocols, audit vs validator).
  - [ ] Public surface: keep exports in `ml/stores/__init__.py`; ensure callers use surface, not internals.
- Cross-cutting
  - [ ] Quarantine/remove backup/phase dirs from importable paths.
  - [ ] Add import discipline (lint/test) to enforce public-surface imports only.

## Target Architecture
- **Public API per domain**: expose supported facades/managers via `ml/<domain>/__init__.py`.
- **Internal modules stay private**: keep components/internal helpers un-exported (or underscore-prefixed). Only public facades/managers are re-exported.
- **One canonical implementation per concern**: pick component vs legacy root; the loser becomes a shim/deprecated wrapper or is removed after caller migration.
- **Import discipline**: external callers import from the domain `__init__`; internal code can reach into private modules. Enforce via lint/tests if feasible.
- **Backups/phase dirs**: quarantine or remove from importable paths to reduce accidental use.

## Working Steps
- [ ] For each domain, decide canonical implementation per pair (favor components where they enforce locks/persistence/config; favor roots where components are stubs).
- [ ] Define the public surface in each domain `__init__.py` (facades/managers only) and re-export it.
- [ ] Add shims/deprecation warnings in legacy modules that lose, delegating to the canonical implementation.
- [ ] Migrate callers/tests to the public surface; reduce duplicated tests to shim coverage + canonical coverage.
- [ ] Clarify naming where overlaps are orthogonal (short docstrings/README per domain).
- [ ] Validation when code changes: `poetry run mypy ml --strict`, `poetry ruff check ml`, and focused tests (`poetry run pytest -k <domain>`).

## Next Actions (suggested)
- Start with **registry**: choose canonical implementations, update `ml/registry/__init__.py`, and shim/deprecate `ab_testing_manager.py` et al.
- Decide on **orchestration** components: implement or deprecate placeholders to avoid parallel APIs.
- Confirm **stores** need no consolidation; add brief docs to prevent misrouting (protocols/audit vs validators).
