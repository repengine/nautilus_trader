"""
Backwards-compatible dummy model utilities.

Historically the project exposed :mod:`ml.models.save_dummy_model` which provided
the ``DummyModel`` helper used in a variety of smoke tests. During the ONNX
migration the implementation moved under ``ml.examples.create_dummy_model`` but
the import path in ``ml.models`` (and several tests) was never updated, leaving
``ml.models`` un-importable.

To restore the public API we simply re-export the modern ``DummyModel`` (and its
helpers) from the examples module. Keeping this thin wrapper avoids duplicating
logic while ensuring `import ml.models` succeeds when optional dependencies are
installed.
"""

from __future__ import annotations

from pathlib import Path

from ml.examples.create_dummy_model import DummyModel
from ml.examples.create_dummy_model import create_dummy_models


def save_dummy_model(output_dir: str | Path | None = None) -> Path:
    """
    Create dummy ONNX artefacts and return the directory path.

    Parameters
    ----------
    output_dir : str | Path | None, optional
        Optional target directory. When omitted the models are written under
        ``ml/models`` (matching historical behaviour).

    Returns
    -------
    Path
        Path to the directory containing the generated ONNX files.

    Notes
    -----
    The helper delegates to :func:`create_dummy_models` which implements the
    ONNX-first export pipeline used across tests and documentation. Keeping the
    wrapper preserves the name that external scripts may still reference.
    """
    default_dir = create_dummy_models()
    source_dir = Path(default_dir)

    if output_dir is None:
        return source_dir

    target_dir = Path(output_dir)
    if target_dir.resolve() == source_dir.resolve():
        return source_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    for model_file in source_dir.glob("*.onnx"):
        destination = target_dir / model_file.name
        if not destination.exists():
            destination.write_bytes(model_file.read_bytes())

    return target_dir


__all__ = ["DummyModel", "create_dummy_models", "save_dummy_model"]
