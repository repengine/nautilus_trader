#!/usr/bin/env python3

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ml._imports import HAS_ONNX
from ml.registry.base import DataRequirements
from ml.registry.base import ModelRole
from ml.registry.model_registry import ModelRegistry
from ml.tests.builders import RegistryBuilder


@pytest.mark.skipif(not HAS_ONNX, reason="ONNX not installed")
def test_model_registry_register_load_deploy_lineage() -> None:
    # Build a simple ONNX model: y = sum(x)
    import onnx
    from onnx import TensorProto
    from onnx import helper

    from ml.training.export import DEFAULT_ONNX_OPSET

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])
    node = helper.make_node("ReduceSum", inputs=["input"], outputs=["output"], keepdims=1, axes=[1])
    graph = helper.make_graph([node], "sum_model", [input_tensor], [output_tensor])
    # Use compatible opset and IR version
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", min(DEFAULT_ONNX_OPSET, 10))],
    )
    model.ir_version = 10  # Explicitly set for compatibility

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        onnx_path = base / "model.onnx"
        onnx.save(model, str(onnx_path))

        registry = ModelRegistry(base)

        manifest = RegistryBuilder.model_manifest(
            model_id="",
            role=ModelRole.STUDENT,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="ONNX",
            feature_schema={"input": "float32"},
            feature_schema_hash="abc123",
            parent_id="teacher_model_001",  # Add parent_id for student model
            performance_metrics={"inference_latency_ms": 1.2},
        )

        mid = registry.register_model(
            model_path=onnx_path,
            manifest=manifest,
            auto_deploy=True,
        )

        # Query
        info = registry.get_model(mid)
        assert info is not None
        assert info.manifest.role == ModelRole.STUDENT
        assert info.deployment_status.name.lower() in {"active", "testing"}

        # Load
        session = registry.load_model(mid)
        assert session is not None
        outputs = session.run(None, {"input": np.random.randn(2, 4).astype(np.float32)})
        assert outputs[0].shape == (2, 1)

        # Lineage (no parent/children yet)
        lineage = registry.get_model_lineage(mid)
        assert len(lineage) == 1
