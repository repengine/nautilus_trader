from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.scripts import ensure_peer_logits_metadata as script


def _write_logits(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)


def _base_payload() -> dict[str, np.ndarray]:
    return {
        "z_train": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "z_val": np.asarray([0.4, 0.5], dtype=np.float32),
        "y_val": np.asarray([0.0, 1.0], dtype=np.float32),
        "train_row_ids": np.asarray(["AAPL::1", "AAPL::2", "SPY::3"], dtype=np.str_),
        "train_instrument_ids": np.asarray(["AAPL.XNAS", "AAPL.XNAS", "SPY.XNAS"], dtype=np.str_),
        "train_time_indices": np.asarray([1, 2, 3], dtype=np.int64),
        "val_row_ids": np.asarray(["AAPL::4", "SPY::5"], dtype=np.str_),
        "val_instrument_ids": np.asarray(["AAPL.XNAS", "SPY.XNAS"], dtype=np.str_),
        "val_time_indices": np.asarray([4, 5], dtype=np.int64),
    }


def test_run_adds_missing_metadata(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    peer = tmp_path / "peer.npz"
    payload = _base_payload()
    _write_logits(reference, **payload)
    _write_logits(peer, z_train=payload["z_train"], z_val=payload["z_val"], y_val=payload["y_val"])

    exit_code = script.run(reference, [peer], dry_run=False, backup_suffix="")
    assert exit_code == 0

    with np.load(peer, allow_pickle=False) as handle:
        for key in ("train_row_ids", "train_instrument_ids", "train_time_indices"):
            assert key in handle.files
            np.testing.assert_array_equal(handle[key], payload[key])
        for key in ("val_row_ids", "val_instrument_ids", "val_time_indices"):
            assert key in handle.files
            np.testing.assert_array_equal(handle[key], payload[key])


def test_run_skips_when_peer_already_complete(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    peer = tmp_path / "peer_complete.npz"
    payload = _base_payload()
    _write_logits(reference, **payload)
    _write_logits(peer, **payload)

    exit_code = script.run(reference, [peer], dry_run=False, backup_suffix="")
    assert exit_code == 0
    # Ensure no backup file is produced because no rewrite happened
    assert not peer.with_name(f"{peer.name}.bak").exists()


def test_run_raises_for_shape_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npz"
    peer = tmp_path / "peer_bad.npz"
    payload = _base_payload()
    _write_logits(reference, **payload)
    _write_logits(
        peer,
        z_train=np.asarray([0.1, 0.2], dtype=np.float32),
        z_val=payload["z_val"],
        y_val=payload["y_val"],
    )

    with pytest.raises(ValueError):
        script.run(reference, [peer], dry_run=False, backup_suffix="")
