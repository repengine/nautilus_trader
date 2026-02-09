from __future__ import annotations

import random

import numpy as np
import pytest

import ml.common.reproducibility as reproducibility_module
from ml.common.reproducibility import ReproducibilityHelper
from ml.common.reproducibility import apply_reproducibility_seed
from ml.common.reproducibility import build_configured_reproducibility_provenance
from ml.common.reproducibility import build_reproducibility_provenance
from ml.common.reproducibility import resolve_configured_seed
from ml.common.reproducibility import validate_reproducibility_provenance


pytestmark = pytest.mark.unit


def test_apply_reproducibility_seed_when_reapplied_repeats_python_and_numpy_draws() -> None:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        seed_result = apply_reproducibility_seed(123, include_torch=False)
        first_python = random.random()
        first_numpy = float(np.random.random())

        apply_reproducibility_seed(123, include_torch=False)
        second_python = random.random()
        second_numpy = float(np.random.random())
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    assert seed_result.python_random_seeded is True
    assert seed_result.numpy_seeded is True
    assert seed_result.torch_seeded is False
    assert second_python == pytest.approx(first_python)
    assert second_numpy == pytest.approx(first_numpy)


def test_reproducibility_helper_when_seed_negative_raises_value_error() -> None:
    with pytest.raises(ValueError, match="seed must be >= 0"):
        ReproducibilityHelper.apply_seed(-1)


def test_resolve_configured_seed_returns_primary_when_available() -> None:
    resolved = resolve_configured_seed(primary_seed=13, fallback_seed=7)
    assert resolved == 13


def test_resolve_configured_seed_returns_fallback_when_primary_missing() -> None:
    resolved = resolve_configured_seed(primary_seed=None, fallback_seed=17)
    assert resolved == 17


def test_resolve_configured_seed_returns_none_when_optional_and_missing() -> None:
    resolved = resolve_configured_seed(primary_seed=None, fallback_seed=None, required=False)
    assert resolved is None


def test_resolve_configured_seed_when_required_and_missing_raises_value_error() -> None:
    with pytest.raises(ValueError, match="worker seed must be configured"):
        resolve_configured_seed(
            primary_seed=None,
            fallback_seed=None,
            required=True,
            context="worker seed",
        )


def test_resolve_configured_seed_when_negative_raises_value_error() -> None:
    with pytest.raises(ValueError, match="worker seed must be >= 0"):
        resolve_configured_seed(primary_seed=-1, context="worker seed")


def test_build_reproducibility_provenance_includes_core_metadata() -> None:
    provenance = build_reproducibility_provenance(
        seed=7,
        deterministic_mode=True,
        extra={"run_id": "slice-pr02"},
    )

    assert provenance["seed"] == 7
    assert provenance["deterministic_mode"] is True
    assert isinstance(provenance["python_version"], str)
    assert isinstance(provenance["numpy_version"], str)
    assert isinstance(provenance["platform"], str)
    assert provenance["run_id"] == "slice-pr02"
    assert isinstance(provenance["generated_at_utc"], str)


def test_build_configured_reproducibility_provenance_when_deterministic_without_seed_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dataset seed must be configured"):
        build_configured_reproducibility_provenance(
            primary_seed=None,
            deterministic_mode=True,
            context="dataset seed",
        )


def test_validate_reproducibility_provenance_when_runtime_field_missing_raises_value_error() -> None:
    payload = build_reproducibility_provenance(
        seed=5,
        deterministic_mode=True,
    )
    payload.pop("python_version", None)

    with pytest.raises(ValueError, match="python_version"):
        validate_reproducibility_provenance(
            payload=payload,
            context="unit provenance",
        )


class _TorchCudaStub:
    def __init__(self) -> None:
        self.seed_all_calls: list[int] = []

    def is_available(self) -> bool:
        return True

    def manual_seed_all(self, seed: int) -> None:
        self.seed_all_calls.append(seed)


class _TorchCudnnStub:
    def __init__(self) -> None:
        self.deterministic = False
        self.benchmark = True


class _TorchBackendsStub:
    def __init__(self) -> None:
        self.cudnn = _TorchCudnnStub()


class _TorchStub:
    __version__ = "2.4.0-test"

    def __init__(self) -> None:
        self.seed_calls: list[int] = []
        self.deterministic_calls: list[bool] = []
        self.cuda = _TorchCudaStub()
        self.backends = _TorchBackendsStub()

    def manual_seed(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def use_deterministic_algorithms(self, mode: bool) -> None:
        self.deterministic_calls.append(mode)


def test_reproducibility_helper_when_torch_available_seeds_torch_and_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_stub = _TorchStub()
    monkeypatch.setattr(reproducibility_module, "HAS_TORCH", True)
    monkeypatch.setattr(reproducibility_module, "_torch", torch_stub)

    result = ReproducibilityHelper.apply_seed(
        321,
        deterministic_mode=True,
        include_torch=True,
    )

    assert result.torch_seeded is True
    assert result.torch_cuda_seeded is True
    assert torch_stub.seed_calls == [321]
    assert torch_stub.cuda.seed_all_calls == [321]
    assert torch_stub.deterministic_calls == [True]
    assert torch_stub.backends.cudnn.deterministic is True
    assert torch_stub.backends.cudnn.benchmark is False


def test_build_reproducibility_provenance_when_torch_missing_returns_none_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reproducibility_module, "HAS_TORCH", False)
    monkeypatch.setattr(reproducibility_module, "_torch", None)

    provenance = ReproducibilityHelper.build_provenance(seed=None, deterministic_mode=False)

    assert provenance["seed"] is None
    assert provenance["torch_version"] is None
