"""
Utility to generate teacher prediction targets for student distillation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def generate_teacher_targets(features_npz: Path, out_npz: Path) -> Path:
    """
    Generate placeholder teacher logits for student distillation scaffolding.
    """
    outputs: dict[str, NDArray[np.float64]] = {}
    with np.load(features_npz, allow_pickle=True) as data:
        for split in ("train", "val", "test"):
            key = f"X_{split}"
            if key not in data:
                continue
            arr = np.asarray(data[key], dtype=np.float64)
            logits: NDArray[np.float64]
            if arr.ndim == 2:
                logits = arr.mean(axis=1)
            else:
                logits = arr.astype(np.float64)
            outputs[f"teacher_logits_{split}"] = logits

    np.savez(out_npz, **cast(dict[str, Any], outputs))
    return out_npz


__all__ = ["generate_teacher_targets"]
