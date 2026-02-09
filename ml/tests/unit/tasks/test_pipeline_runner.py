from __future__ import annotations

import importlib

import pytest

from ml.orchestration.pipeline_runner import MLPipelineRunner as CanonicalMLPipelineRunner
from ml.orchestration.pipeline_runner import PipelineRunConfig as CanonicalPipelineRunConfig
from ml.orchestration.pipeline_runner import load_config as canonical_load_config
from ml.orchestration.pipeline_runner import run_pipeline as canonical_run_pipeline
from ml.orchestration.pipeline_runner import setup_logging as canonical_setup_logging


def test_pipeline_runner_canonical_symbols_are_available() -> None:
    assert CanonicalMLPipelineRunner.__name__ == "MLPipelineRunner"
    assert CanonicalPipelineRunConfig.__name__ == "PipelineRunConfig"
    assert callable(canonical_load_config)
    assert callable(canonical_setup_logging)
    assert callable(canonical_run_pipeline)


def test_pipeline_runner_task_shim_module_is_retired() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ml.tasks.pipelines.runner")
