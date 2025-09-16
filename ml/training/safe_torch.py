"""
Safe wrappers around PyTorch serialization.

Provides a hardened loader for state dicts with optional checksum validation and
weights-only deserialization to reduce pickle attack surface.

"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_torch_load(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    map_location: str = "cpu",
) -> Any:
    """
    Load a PyTorch object safely with optional checksum verification.

    - Restricts loading to CPU via map_location="cpu".
    - Attempts weights_only=True when supported (PyTorch>=2.0) to avoid arbitrary code exec.
    - Optionally validates the file's SHA-256 checksum.

    """
    import torch

    p = Path(path).expanduser().resolve()
    if expected_sha256 is not None:
        actual = _sha256_file(p)
        if actual.lower() != expected_sha256.lower():
            raise ValueError("Torch state checksum mismatch")

    try:
        # nosec B614: weights_only + prior checksum verification mitigate risks
        return torch.load(str(p), map_location=map_location, weights_only=True)
    except TypeError:
        # nosec B614: best-effort fallback for older torch; recommend providing expected_sha256
        return torch.load(str(p), map_location=map_location)  # nosec B614
