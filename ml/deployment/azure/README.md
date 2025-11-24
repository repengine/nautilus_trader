# Azure Spot Checkpointing Artifacts

This directory wires the streaming training runner to durable Azure storage so
spot-eviction checkpoints survive host re-provisioning.

## Files

- `cloud-init-checkpoint-mount.yaml` — cloud-init blueprint that:
  - Installs `blobfuse2` and `cifs-utils`.
  - Mounts the checkpoint Blob container to `${CHECKPOINT_MOUNT:-/mnt/ml-checkpoints}` with MSI auth.
  - Optionally binds an Azure Files share for manifest/log file fan-out.
  - Exports `ML_STREAMING_CHECKPOINT_DIR` for the runner.
- `terraform_checkpoint_mount.tf` — Terraform snippet creating a user-assigned
  managed identity, assigning Blob/File roles, and attaching the cloud-init
  payload to a spot VM.

## Usage

1. Populate the placeholders in `cloud-init-checkpoint-mount.yaml` via
   `templatefile` (see Terraform example) or render the file manually before
   uploading. Required parameters:

   | Placeholder | Description |
   | ----------- | ----------- |
   | `AZURE_STORAGE_ACCOUNT` | Storage account hosting the checkpoint container |
   | `AZURE_BLOB_CONTAINER` | Container name for checkpoint archives |
   | `AZURE_USER_ASSIGNED_IDENTITY` | Resource ID of the managed identity granted Blob/File permissions |
   | `AZURE_FILES_STORAGE_ACCOUNT` | Storage account for Azure Files (often the same as above) |
   | `AZURE_FILES_SHARE` | Share name for manifests/logs |
   | `AZURE_FILES_SAS_TOKEN` | SAS token used by the CIFS mount |
   | `CHECKPOINT_MOUNT` | Filesystem mount point (mirrors `ML_STREAMING_CHECKPOINT_DIR`) |

2. Apply the Terraform snippet or copy the relevant resources into your
   existing module. Attach the rendered cloud-init as `custom_data`.

3. When the VM boots it mounts the checkpoint container and exports
   `ML_STREAMING_CHECKPOINT_DIR`, `AZURE_FILES_*`, and `CHECKPOINT_MOUNT` so the
   streaming runner can resume cohorts on eviction.

The streaming runner automatically polls Azure scheduled events (metadata
endpoint `169.254.169.254/metadata/scheduledevents`) and requests a checkpoint
when a `Preempt` notice is received. Ensure the VM retains outbound access to
the Azure Instance Metadata Service.
