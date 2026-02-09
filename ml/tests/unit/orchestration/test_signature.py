from __future__ import annotations

import pytest

from ml.orchestration.signature import PipelineSignatureValidator
from ml.orchestration.signature import compute_pipeline_signature


class _RegistryStub:
    def __init__(self, signature: str | None) -> None:
        self.signature = signature
        self.dataset_ids: list[str] = []

    def get_pipeline_signature(self, dataset_id: str) -> str | None:
        self.dataset_ids.append(dataset_id)
        return self.signature


def test_compute_pipeline_signature_is_order_invariant() -> None:
    signature_a = compute_pipeline_signature(
        features=["returns_1", "rsi_14"],
        version="1.0",
    )
    signature_b = compute_pipeline_signature(
        features=["rsi_14", "returns_1"],
        version="1.0",
    )

    assert signature_a == signature_b
    assert len(signature_a) == 64


def test_compute_pipeline_signature_changes_with_config_hash() -> None:
    base_signature = compute_pipeline_signature(
        features=["returns_1", "rsi_14"],
        version="1.0",
    )
    with_config = compute_pipeline_signature(
        features=["returns_1", "rsi_14"],
        version="1.0",
        config_hash="cfg-1",
    )

    assert base_signature != with_config


def test_compute_pipeline_signature_validates_inputs() -> None:
    with pytest.raises(ValueError, match="Features list cannot be empty"):
        compute_pipeline_signature(features=[], version="1.0")

    with pytest.raises(ValueError, match="Version string cannot be empty"):
        compute_pipeline_signature(features=["returns_1"], version="")


def test_pipeline_signature_validator_passes_when_no_stored_signature() -> None:
    registry = _RegistryStub(signature=None)
    validator = PipelineSignatureValidator()

    validator.validate_signature(
        dataset_id="dataset-a",
        new_signature="abc123",
        registry=registry,
    )

    assert registry.dataset_ids == ["dataset-a"]


def test_pipeline_signature_validator_raises_on_mismatch() -> None:
    registry = _RegistryStub(signature="stored-signature")
    validator = PipelineSignatureValidator()

    with pytest.raises(ValueError, match="Pipeline signature mismatch for dataset dataset-a"):
        validator.validate_signature(
            dataset_id="dataset-a",
            new_signature="new-signature",
            registry=registry,
        )


def test_pipeline_signature_validator_accepts_matching_signature() -> None:
    registry = _RegistryStub(signature="same-signature")
    validator = PipelineSignatureValidator()

    validator.validate_signature(
        dataset_id="dataset-a",
        new_signature="same-signature",
        registry=registry,
    )

    assert registry.dataset_ids == ["dataset-a"]
