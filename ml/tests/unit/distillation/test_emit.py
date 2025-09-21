from __future__ import annotations

import numpy as np

from ml.training.distillation.emit import generate_teacher_targets


def test_generate_teacher_targets(tmp_path) -> None:
    features_path = tmp_path / "features.npz"
    out_path = tmp_path / "teacher_logits.npz"
    X_train = np.array([[1.0, 2.0], [3.0, 5.0]])
    X_val = np.array([[2.0, 4.0]])
    np.savez(features_path, X_train=X_train, X_val=X_val)

    result = generate_teacher_targets(features_path, out_path)
    assert result.exists()
    data = np.load(result)
    assert np.allclose(data["teacher_logits_train"], X_train.mean(axis=1))
    assert np.allclose(data["teacher_logits_val"], X_val.mean(axis=1))
