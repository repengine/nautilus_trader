variable "location" {
  description = "Azure region for the spot training VM."
  type        = string
}

variable "resource_group_name" {
  description = "Resource group that hosts the training VM."
  type        = string
}

variable "spot_vm_name" {
  description = "Name of the Azure spot VM used for streaming training."
  type        = string
}

variable "storage_account_name" {
  description = "Existing storage account that hosts the checkpoint container."
  type        = string
}

variable "blob_container_name" {
  description = "Container within the storage account for Lightning checkpoints."
  type        = string
}

variable "files_share_name" {
  description = "Azure Files share used for cross-VM manifests/logs."
  type        = string
}

variable "files_sas_token" {
  description = "SAS token granting mount permissions on the Azure Files share."
  type        = string
  sensitive   = true
}

variable "checkpoint_mount_path" {
  description = "Filesystem mount used by ML_STREAMING_CHECKPOINT_DIR."
  type        = string
  default     = "/mnt/ml-checkpoints"
}

data "azurerm_storage_account" "checkpoint" {
  name                = var.storage_account_name
  resource_group_name = var.resource_group_name
}

resource "azurerm_user_assigned_identity" "checkpoint" {
  name                = "${var.spot_vm_name}-checkpoint-msi"
  location            = var.location
  resource_group_name = var.resource_group_name
}

resource "azurerm_role_assignment" "blob_data_contributor" {
  scope                = data.azurerm_storage_account.checkpoint.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.checkpoint.principal_id
}

resource "azurerm_role_assignment" "files_data_smb_share" {
  scope                = "${data.azurerm_storage_account.checkpoint.id}/fileServices/default"
  role_definition_name = "Storage File Data SMB Share Contributor"
  principal_id         = azurerm_user_assigned_identity.checkpoint.principal_id
}

resource "azurerm_linux_virtual_machine" "spot_trainer" {
  name                  = var.spot_vm_name
  resource_group_name   = var.resource_group_name
  location              = var.location
  size                  = "Standard_NC6s_v3"
  admin_username        = "azureuser"
  disable_password_authentication = true
  network_interface_ids = [] # Attach to an ip config / subnet in the parent module.

  priority        = "Spot"
  eviction_policy = "Deallocate"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.checkpoint.id]
  }

  custom_data = base64encode(templatefile("${path.module}/cloud-init-checkpoint-mount.yaml", {
    AZURE_STORAGE_ACCOUNT         = var.storage_account_name,
    AZURE_BLOB_CONTAINER          = var.blob_container_name,
    AZURE_USER_ASSIGNED_IDENTITY  = azurerm_user_assigned_identity.checkpoint.id,
    AZURE_FILES_STORAGE_ACCOUNT   = var.storage_account_name,
    AZURE_FILES_SHARE             = var.files_share_name,
    AZURE_FILES_SAS_TOKEN         = var.files_sas_token,
    CHECKPOINT_MOUNT              = var.checkpoint_mount_path,
  }))

  tags = {
    role                 = "ml-streaming-trainer"
    checkpoint_mount     = var.checkpoint_mount_path
    checkpoint_container = var.blob_container_name
  }

  lifecycle {
    ignore_changes = [
      custom_data, # allow cloud-init reruns without reprovision
    ]
  }
}
