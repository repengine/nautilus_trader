# Azure Spot Training Checkpointing Plan

## Context

- Goal: run streaming training workloads on Azure spot VMs via VS Code remote sessions without losing progress when preemptions occur.
- Existing stack: Lightning-based streaming worker (`LightningStreamingWorker`), streaming runner CLI, orchestrator/persistence services, dashboards.
- Risk: spot evictions terminate the VM with short notice (~30 seconds), so we need durable checkpoints and fast resume logic.

## Requirements

1. Periodic model checkpoints persisted to durable storage outside the spot VM.
2. Automatic resume when a spot VM is reprovisioned with the same plan/workload.
3. Graceful handling of Azure eviction notices (shutdown hooks).
4. Operational visibility: metrics/logging for checkpoint writes and resume attempts.

## Design Overview

### Checkpoint Cadence

- Extend `StreamingWorkerConfig`/CLI with:
  - `checkpoint_dir`: path (local or cloud-mounted) where checkpoints are written.
  - `checkpoint_interval_seconds` and/or `checkpoint_interval_steps`: cadence for checkpoint callbacks.
- Leverage PyTorch Lightning checkpoint callbacks:
  - Keep the latest checkpoint (`plan_id_latest.ckpt`) and optionally a previous version.
  - Write lightweight metadata alongside the checkpoint (`plan_id_latest.json`) with fields: plan ID, epoch, global step, val metrics, timestamp.
- Storage options:
  - Azure Blob Storage: mount via `blobfuse2` or upload via SDK after local write.
  - Azure Files: SMB mount for cross-VM access.
  - Ensure the mount path is configurable via infra scripts.

### Resume Logic

- On runner start:
  - Discover the latest checkpoint in `checkpoint_dir` for the active plan (match by plan ID).
  - Load checkpoint into the Lightning trainer before calling `fit`.
  - Update the persistence/manifest state to note “resume from checkpoint” with the recorded global step.
- On normal completion:
  - Promote the final checkpoint to `plan_id_final.ckpt`.
  - Clean up old checkpoints if desired (configurable retention).

### Eviction Handling

- Install a signal handler in `StreamingTrainingRunner`:
  - Catch `SIGTERM` and `SIGINT`.
  - Call a new `save_checkpoint_now()` hook on the worker to flush a final checkpoint.
  - Publish a telemetry event (e.g., `ml_streaming_checkpoint_evictions_total`) and exit cleanly.
- Azure scheduled events:
  - Background watcher queries `169.254.169.254/metadata/scheduledevents` for `Preempt` notices.
  - On notice, trigger the same save routine and set a flag to prevent duplicate checkpointing.

### Storage & Credentials

- Mount cloud storage at boot using managed identity or service principal:
  - For Azure Blob: use `blobfuse2` with MSI authentication (`--use-azure-identity`).
  - For Azure Files: store credentials in Azure Key Vault; mount via SMB with secure options.
- Ensure checkpoint directory is accessible to both the streaming worker and any resume job.
- Document mounting steps in infra scripts (Terraform/cloud-init).

### Job Lifecycle

- Spot VM startup script:
  1. Sync repo / install dependencies (Poetry install).
  2. Mount checkpoint storage.
  3. Fetch latest state snapshot (optional) to determine resume target.
  4. Launch streaming runner with `--checkpoint-dir` pointing at the mount.
- Orchestrator/persistence services remain on stable (non-spot) infrastructure to preserve plan state.
- Optionally use Azure Batch or Scale Sets to manage spot VM lifecycle; the runner can be part of a custom script extension.

### Observability & Alerts

- Metrics:
  - `ml_streaming_checkpoints_total{status=created|failed}` updated on every checkpoint save attempt.
  - `ml_streaming_checkpoint_resumes_total{outcome=success|failure}` tracked when restarting from checkpoints.
  - `ml_streaming_checkpoint_evictions_total` increments when an eviction-triggered checkpoint runs.
- Logs:
  - Include checkpoint path, global step, hash of checkpoint file for traceability.
  - Log resume decisions (e.g., “resuming plan X from step Y after eviction”).
- Alerts:
  - Optional Prometheus rule: alert if `evictions_total` spans >N per hour or if checkpoint failures occur consecutively.

## Implementation Tasks

1. **Runner & worker enhancements**
   - [x] Add CLI/config options for checkpoint directory and interval.
   - [x] Implement Lightning checkpoint callbacks and expose `save_checkpoint_now`.
   - [x] Wire signal handlers to call the save hook.
2. **Azure integration**
   - [x] Provide cloud-init/Terraform scripts to mount Blob/Files storage and poll scheduled events (`ml/deployment/azure/cloud-init-checkpoint-mount.yaml`, `ml/deployment/azure/terraform_checkpoint_mount.tf`).
   - [x] Create documentation snippet for VS Code remote setup referencing the checkpoint-mount path (see “VS Code Remote Checklist” below).
3. **Resume workflow**
   - [x] Modify runner start-up to auto-load checkpoints when present.
   - [x] Update persistence/manifest logic to record resumed runs (plan metadata + manifest entries).
   - [x] Add unit/integration test that simulates interruption (`SIGTERM`) and resume.
4. **Observability**
   - [x] Register new metrics in `ml/common/metrics_bootstrap`.
   - [x] Add logging entries for checkpoint saves/resumes.
   - [x] Update dashboards to surface checkpoint telemetry (Grafana panel added to `ml/deployment/grafana/ml_pipeline_health.json`).
5. **Documentation**
   - [x] Update runbooks (streaming scaling experiments, dashboard) with checkpoint usage.
   - [x] Provide operator checklist for spot VM lifecycle (start, monitor, resume, teardown).

### Scheduled-Event Polling Details

- Runner config now exposes `AzureScheduledEventsConfig` (see `ml/config/streaming_pipeline.py`).
- CLI/env overrides:
  - `--azure-events-enabled` / `ML_STREAMING_AZURE_EVENTS_ENABLED=1`
  - `--azure-events-poll-interval` (seconds, default `5`)
  - `--azure-events-timeout-seconds` (HTTP timeout, default `2`)
  - `--azure-events-endpoint` (default `http://169.254.169.254/metadata/scheduledevents`)
  - `--azure-events-api-version` (default `2020-07-01`)
  - `--azure-events-resource` (repeatable) to scope resources (fallback accepts any or `*`)
  - `--azure-events-event-type` (repeatable, default `Preempt`)
  - `--azure-events-status` (repeatable, default `Scheduled`, `InProgress`)
- `ml/training/event_driven/azure_events.py` hosts the watcher. When a matching event arrives it:
  - Logs `azure_eviction_notice_received` with event metadata.
  - Calls `LightningStreamingWorker.save_checkpoint_now(triggered_by_signal=True)`.
  - Sets `StreamingTrainingRunner._stop_requested = True` to avoid launching new plans.

### VS Code Remote Checklist

1. Mount the checkpoint container and Azure Files share (if used) using the cloud-init systemd unit or by running `/usr/local/bin/mount-checkpoint.sh` manually after connecting.
2. Validate `/etc/environment` now contains `ML_STREAMING_CHECKPOINT_DIR`. Run `printenv ML_STREAMING_CHECKPOINT_DIR` inside the remote session; VS Code inherits this value automatically.
3. Update `.vscode/settings.json` (or the Remote SSH workspace settings) with:

   ```json
   {
     "terminal.integrated.env.linux": {
       "ML_STREAMING_CHECKPOINT_DIR": "/mnt/ml-checkpoints",
       "ML_STREAMING_AZURE_EVENTS_ENABLED": "1"
     }
   }
   ```

4. Use the new runner flags when launching cohorts inside VS Code:

   ```bash
   poetry run python -m ml.cli.streaming_training_runner \
     --azure-events-enabled \
     --azure-events-resource "$(hostname)" \
     --checkpoint-dir "$ML_STREAMING_CHECKPOINT_DIR" \
     --plan-interval-seconds 120
   ```

5. Confirm `azure_eviction_notice_received` and `checkpoint_saved` logs reach the integrated terminal (or `journalctl -u blobfuse2-checkpoint` for mount troubleshooting).

## Open Questions

- Preferred checkpoint format: Lightning `.ckpt` vs. custom (possible need for ONNX snapshot?). Lightning format suffices unless inference pipeline requires conversion.
- Retention policy for checkpoints (keep N latest vs. clean-up after promotion).
- Whether to integrate with model registry artefacts or keep checkpoints separate from final logits.

## Next Actions

1. Validate end-to-end on a spot VM: run training, trigger eviction, confirm resume continues training, and verify metrics/logging reflect the event.
2. Capture operator feedback on the new Grafana panels and refine thresholds/alerting as needed.
3. Prototype multi-cloud alignment once Azure flow is stable, incorporating operator feedback from the updated runbooks.
