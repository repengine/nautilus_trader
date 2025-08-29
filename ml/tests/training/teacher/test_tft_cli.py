from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash
from ml.training.teacher.tft_cli import main as tft_main


def _make_feature_registry(tmp_path: Path, feature_names: list[str]) -> tuple[Path, str, str]:
    registry_dir = tmp_path / "feature_registry"
    freg = FeatureRegistry(registry_dir)
    dtypes = ["float32"] * len(feature_names)
    schema = compute_schema_hash(feature_names, dtypes, pipeline_signature="sig_v1")
    manifest = FeatureManifest(
        feature_set_id="",
        name="test_features",
        version="0.0.1",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=feature_names,
        feature_dtypes=dtypes,
        schema_hash=schema,
        pipeline_signature="sig_v1",
        pipeline_version="1",
    )
    fid = freg.register_feature_set(manifest)
    return registry_dir, fid, schema


@pytest.mark.parallel_safe
def test_tft_cli_registry_calibration_with_z_val(tmp_path: Path) -> None:
    feature_names = ["f1", "f2", "f3"]
    registry_dir, feature_set_id, schema_hash = _make_feature_registry(tmp_path, feature_names)

    # Prepare NPZ with logits and true labels
    n = 32
    rng = np.random.default_rng(42)
    z_val = rng.normal(size=(n,)).astype(np.float64)
    y_val_true = (rng.random(n) > 0.5).astype(np.float64)
    npz_path = tmp_path / "val_slice.npz"
    np.savez_compressed(npz_path, z_val=z_val, y_val_true=y_val_true)

    out_dir = tmp_path / "out"
    args = [
        "--student_window_npz",
        str(npz_path),
        "--out_dir",
        str(out_dir),
        "--model_id",
        "tft_test_model",
        "--feature_registry_dir",
        str(registry_dir),
        "--feature_set_id",
        feature_set_id,
    ]
    rc = tft_main(args)
    assert rc == 0

    preds_path = out_dir / "teacher_preds.npz"
    meta_path = out_dir / "teacher_meta.json"
    assert preds_path.exists(), "predictions file not created"
    assert meta_path.exists(), "meta file not created"

    data = np.load(preds_path)
    q_train = data["q_train"]
    assert q_train.shape == z_val.shape
    assert np.all(q_train >= 0.0) and np.all(q_train <= 1.0)

    meta = json.loads(meta_path.read_text())
    assert meta["feature_set_id"] == feature_set_id
    assert meta["feature_schema_hash"] == schema_hash


def test_tft_cli_registry_enforces_shape_on_X_val(tmp_path: Path) -> None:
    feature_names = ["f1", "f2", "f3"]
    registry_dir, feature_set_id, _ = _make_feature_registry(tmp_path, feature_names)

    # Prepare NPZ with wrong-shaped X_val
    n = 8
    X_val = np.random.randn(n, 2).astype(np.float32)  # should be 3
    y_val_true = (np.random.rand(n) > 0.5).astype(np.float64)
    npz_path = tmp_path / "val_x.npz"
    np.savez_compressed(npz_path, X_val=X_val, y_val_true=y_val_true)

    out_dir = tmp_path / "out2"
    args = [
        "--student_window_npz",
        str(npz_path),
        "--out_dir",
        str(out_dir),
        "--model_id",
        "tft_shape_model",
        "--feature_registry_dir",
        str(registry_dir),
        "--feature_set_id",
        feature_set_id,
    ]

    try:
        tft_main(args)
        assert False, "Expected SystemExit due to shape mismatch"
    except SystemExit as e:
        assert "does not match feature manifest width" in str(e)


def test_tft_cli_training_minimal_flow(tmp_path: Path) -> None:
    # Small synthetic dataset
    n = 50
    times = np.arange(n, dtype=np.int64)
    inst = ["A"] * n
    rng = np.random.default_rng(0)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    # Create a simple target related to f1
    y = (f1 + 0.1 * f2 + rng.normal(scale=0.1, size=n) > 0).astype(int)

    df = pd.DataFrame(
        {
            "time_index": times,
            "instrument_id": inst,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "y": y,
        }
    )
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    feature_names = ["f1", "f2", "f3"]
    registry_dir, feature_set_id, _ = _make_feature_registry(tmp_path, feature_names)

    out_dir = tmp_path / "out_train"
    args = [
        "--out_dir",
        str(out_dir),
        "--model_id",
        "tft_train_model",
        "--feature_registry_dir",
        str(registry_dir),
        "--feature_set_id",
        feature_set_id,
        "--train_data_csv",
        str(csv_path),
        "--target_col",
        "y",
        "--time_index_col",
        "time_index",
        "--group_id_col",
        "instrument_id",
        "--max_encoder_length",
        "10",
        "--max_epochs",
        "1",
    ]

    rc = tft_main(args)
    assert rc == 0

    preds_path = out_dir / "teacher_preds.npz"
    meta_path = out_dir / "teacher_meta.json"
    assert preds_path.exists()
    assert meta_path.exists()

    data = np.load(preds_path)
    q_train = data["q_train"]
    y_val_true = data["y_val_true"]
    assert len(q_train) == len(y_val_true)
    assert np.all(q_train >= 0.0) and np.all(q_train <= 1.0)


def test_tft_cli_training_with_registration(tmp_path: Path) -> None:
    # Synthetic data
    n = 30
    times = np.arange(n, dtype=np.int64)
    inst = ["A"] * n
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "time_index": times,
            "instrument_id": inst,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    df["y"] = (df["f1"] + 0.2 * df["f2"] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    csv_path = tmp_path / "train2.csv"
    df.to_csv(csv_path, index=False)

    feature_names = ["f1", "f2", "f3"]
    registry_dir, feature_set_id, _ = _make_feature_registry(tmp_path, feature_names)

    model_registry_dir = tmp_path / "model_registry"

    out_dir = tmp_path / "out_train_reg"
    args = [
        "--out_dir",
        str(out_dir),
        "--model_id",
        "tft_teacher_reg",
        "--feature_registry_dir",
        str(registry_dir),
        "--feature_set_id",
        feature_set_id,
        "--train_data_csv",
        str(csv_path),
        "--target_col",
        "y",
        "--time_index_col",
        "time_index",
        "--group_id_col",
        "instrument_id",
        "--max_encoder_length",
        "8",
        "--max_epochs",
        "1",
        "--register_teacher",
        "--model_registry_dir",
        str(model_registry_dir),
    ]

    rc = tft_main(args)
    assert rc == 0

    # Verify registry contains the teacher with artifact inside registry root
    from ml.registry.model_registry import ModelRegistry

    mreg = ModelRegistry(model_registry_dir)
    # Registered with provided model_id
    info = mreg.get_model("tft_teacher_reg")
    if info is None:
        # Inspect raw registry file for debugging context
        import json

        reg_json = model_registry_dir / "registry.json"
        if reg_json.exists():
            data = json.loads(reg_json.read_text())
            print("registry models:", list(data.get("models", {}).keys()))
        assert info is not None
    assert not info.manifest.serveable
    path = mreg.get_artifact_path("tft_teacher_reg")
    assert path is not None and path.exists()
