# Code-Only Verification Ledger Design

## Intent
Focus the verification ledger on current code, config, entrypoints, docs, and tests so that the audit surfaces present-day gaps between plan claims and implemented behavior. Historical run artifacts and logs are explicitly out of scope because the code that produced them no longer reflects the current system state.

## Scope
In scope:
- CODE/CONFIG/ENTRY/DOC/TEST claims that can be verified by static inspection.
- Gaps where plan documents assert capabilities that are missing or contradicted by current code.

Out of scope:
- Run artifacts, logs, and historical metrics (e.g., `ml_out/`, `reports/`).
- Evidence that depends on the runtime environment or previous executions.

## Planned Changes
- Remove ARTIFACT and LOG evidence types from the ledger and workflow.
- Delete claims whose only evidence is artifacts/logs (run summaries, slice results, OOM logs).
- Rewrite mixed claims to code-only where the intent is still useful (e.g., ONNX-only inference, order-intent serialization path).
- Remove empty sections left behind by artifact/log claim removal to keep the ledger readable.
- Keep failures that are provable by code (e.g., missing fine-tune wiring or facade imports) so the ledger highlights real development gaps.

## Success Criteria
- The ledger contains only CODE/CONFIG/ENTRY/DOC/TEST evidence types.
- Every remaining claim can be validated by static inspection or test presence.
- No references to `ml_out/` or `reports/` remain in the ledger.
- The workflow reflects code-first verification and clear gap surfacing.
