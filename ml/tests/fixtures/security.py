#!/usr/bin/env python3
"""
Security-focused fixtures for ONNX runtime isolation.

These helpers provide deterministic ONNX Runtime mocks so tests can exercise
model-loading flows without requiring the optional dependency or instantiating
real sessions.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Generator, Iterable, Sequence
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class OnnxRuntimeHarness:
    """Expose ONNX runtime mocks and helpers for tests."""

    ort: MagicMock
    check_dependencies: MagicMock
    reload_modules: Callable[[Iterable[str] | str], None]
    _availability_targets: Sequence[Any] = field(repr=False)

    def set_available(self, available: bool) -> None:
        """
        Toggle ONNX availability for patched modules.

        Args:
            available: Whether ONNX should be considered available.
        """
        for module in self._availability_targets:
            if hasattr(module, "HAS_ONNX"):
                setattr(module, "HAS_ONNX", available)


@dataclass(slots=True)
class DeterministicOnnxSession:
    """Minimal ONNX Runtime session stub with deterministic outputs."""

    prediction: float = 0.5
    confidence: float = 0.9
    raise_on_run: bool = False

    _inputs: tuple[SimpleNamespace, ...] = (
        SimpleNamespace(name="features"),
    )
    _outputs: tuple[SimpleNamespace, ...] = (
        SimpleNamespace(name="prediction"),
        SimpleNamespace(name="confidence"),
    )

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=item.name) for item in self._inputs]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=item.name) for item in self._outputs]

    def run(
        self,
        _: object,
        input_feed: dict[str, npt.NDArray[np.float32]],
    ) -> list[npt.NDArray[np.float32]]:
        if self.raise_on_run:
            raise RuntimeError("mock onnx inference failure")

        values = next(iter(input_feed.values()))
        batch = np.asarray(values, dtype=np.float32)
        batch_size = batch.shape[0]
        predictions = np.full((batch_size,), self.prediction, dtype=np.float32)
        confidences = np.full((batch_size,), self.confidence, dtype=np.float32)
        return [predictions, confidences]


@contextmanager
def patch_onnx_runtime(
    modules: Iterable[str] | None = None,
) -> Generator[OnnxRuntimeHarness, None, None]:
    """
    Patch ONNX runtime imports with deterministic mocks.

    Args:
        modules: Optional iterable of module paths whose ``HAS_ONNX`` / ``ort``
            attributes should be patched. When omitted, a safe default set of ML
            runtime modules is used.

    Yields:
        OnnxRuntimeHarness exposing the patched runtime mocks and helpers.
    """

    default_modules = (
        "ml._imports",
        "ml.common.security",
        "ml.registry.model_persistence",
        "ml.registry.model_registry",
        "ml.registry.model_registry_facade",  # Add facade to ONNX patching
    )
    target_modules = tuple(modules) if modules is not None else default_modules

    ort_mock = MagicMock(name="onnxruntime")
    ort_mock.InferenceSession = MagicMock(name="InferenceSession")
    check_mock = MagicMock(name="check_ml_dependencies")
    availability_modules: list[Any] = []

    with ExitStack() as stack:
        for module_path in target_modules:
            module = importlib.import_module(module_path)

            if hasattr(module, "HAS_ONNX"):
                stack.enter_context(patch.object(module, "HAS_ONNX", True))
                availability_modules.append(module)
            if hasattr(module, "ort"):
                stack.enter_context(patch.object(module, "ort", ort_mock))
            if hasattr(module, "check_ml_dependencies"):
                stack.enter_context(patch.object(module, "check_ml_dependencies", check_mock))

        def _reload(targets: Iterable[str] | str) -> None:
            resolved = (targets,) if isinstance(targets, str) else tuple(targets)
            for target in resolved:
                sys.modules.pop(target, None)

        harness = OnnxRuntimeHarness(
            ort=ort_mock,
            check_dependencies=check_mock,
            reload_modules=_reload,
            _availability_targets=tuple(availability_modules),
        )
        try:
            yield harness
        finally:
            ort_mock.reset_mock()
            check_mock.reset_mock()


@pytest.fixture
def mock_onnx_runtime() -> Generator[OnnxRuntimeHarness, None, None]:
    """
    Provide deterministic ONNX runtime mocks for tests.
    """

    with patch_onnx_runtime() as harness:
        yield harness


@pytest.fixture
def onnx_session_stub_factory() -> Callable[..., DeterministicOnnxSession]:
    """
    Provide a factory for deterministic ONNX Runtime session stubs.
    """

    def _factory(
        *,
        prediction: float = 0.5,
        confidence: float = 0.9,
        raise_on_run: bool = False,
    ) -> DeterministicOnnxSession:
        return DeterministicOnnxSession(
            prediction=prediction,
            confidence=confidence,
            raise_on_run=raise_on_run,
        )

    return _factory


__all__ = [
    "DeterministicOnnxSession",
    "OnnxRuntimeHarness",
    "mock_onnx_runtime",
    "onnx_session_stub_factory",
    "patch_onnx_runtime",
]
