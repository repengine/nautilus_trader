#!/usr/bin/env python3
r"""
Annotate ensemble peer logits files with row metadata copied from a reference artifact.

The event-driven streaming worker now persists per-row identifiers inside every logits
bundle so ensemble peers can validate alignment before blending. Older peer artifacts
produced before this guardrail may lack the ``train_row_ids`` / ``val_row_ids`` keys,
causing the worker to skip the member or raise when the member is required.

This utility copies the metadata arrays from a reference logits artifact (typically the
teacher run generated alongside the peers) into one or more peer artifacts after
validating that the logits lengths and metadata payloads match.

Example:
    $ python -m ml.scripts.ensure_peer_logits_metadata \\
        --reference ml_out/tft_streaming_artifacts/full_tft_95/full_tft_95-098aefc6abb0_logits.npz \\
        --peer ml_out/tft_streaming_artifacts/full_tft_95/best_peer_logits.npz \\
        --peer ml_out/tft_streaming_artifacts/full_tft_95/canary_logits.npz
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

_TRAIN_METADATA_KEYS: tuple[str, ...] = (
    "train_row_ids",
    "train_instrument_ids",
    "train_time_indices",
)
_VAL_METADATA_KEYS: tuple[str, ...] = (
    "val_row_ids",
    "val_instrument_ids",
    "val_time_indices",
)


@dataclass(slots=True)
class LogitsArtifact:
    """In-memory representation of a logits artefact."""

    path: Path
    payload: dict[str, npt.NDArray[np.generic]]

    @property
    def train_size(self) -> int:
        return int(np.asarray(self.payload["z_train"]).size)

    @property
    def val_size(self) -> int:
        return int(np.asarray(self.payload["z_val"]).size)

    def has_train_metadata(self) -> bool:
        return all(key in self.payload for key in _TRAIN_METADATA_KEYS)

    def has_val_metadata(self) -> bool:
        return all(key in self.payload for key in _VAL_METADATA_KEYS)

    def save(self) -> None:
        np.savez_compressed(file=self.path, **cast(dict[str, Any], self.payload))


def _load_artifact(path: Path) -> LogitsArtifact:
    if not path.exists():
        raise FileNotFoundError(f"logits artifact missing: {path}")
    with np.load(path, allow_pickle=False) as handle:
        payload: dict[str, npt.NDArray[np.generic]] = {}
        for key in handle.files:
            payload[key] = np.asarray(handle[key])
    return LogitsArtifact(path=path, payload=payload)


def _collect_peer_paths(args: argparse.Namespace) -> list[Path]:
    peers: list[Path] = []
    if args.peer:
        peers.extend(Path(item).expanduser() for item in args.peer)
    if args.peer_dir:
        peers.extend(sorted(Path(args.peer_dir).expanduser().glob(args.pattern)))
    unique_peers: list[Path] = []
    seen: set[Path] = set()
    for path in peers:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_peers.append(resolved)
    return unique_peers


def _copy_missing_metadata(
    reference: LogitsArtifact,
    peer: LogitsArtifact,
) -> tuple[bool, list[str]]:
    if not reference.has_train_metadata() or not reference.has_val_metadata():
        raise ValueError(f"reference logits missing metadata: {reference.path}")
    if peer.train_size != reference.train_size:
        raise ValueError(
            f"train logits length mismatch for peer {peer.path} "
            f"(peer={peer.train_size}, reference={reference.train_size})",
        )
    if peer.val_size != reference.val_size:
        raise ValueError(
            f"validation logits length mismatch for peer {peer.path} "
            f"(peer={peer.val_size}, reference={reference.val_size})",
        )
    added_keys: list[str] = []
    for key in _TRAIN_METADATA_KEYS + _VAL_METADATA_KEYS:
        if key in peer.payload:
            continue
        peer.payload[key] = np.asarray(reference.payload[key]).copy()
        added_keys.append(key)
    return bool(added_keys), added_keys


def _write_with_backup(path: Path, backup_suffix: str) -> None:
    if not backup_suffix:
        return
    backup_path = path.with_name(f"{path.name}{backup_suffix}")
    shutil.copy2(path, backup_path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy row metadata arrays from a reference logits artefact into peer artefacts.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to the teacher logits artefact that already contains metadata.",
    )
    parser.add_argument(
        "--peer",
        action="append",
        help="Peer logits artefact path (repeatable).",
    )
    parser.add_argument(
        "--peer-dir",
        type=Path,
        help="Directory to scan for peer logits artefacts (combined with --pattern).",
    )
    parser.add_argument(
        "--pattern",
        default="*_logits.npz",
        help="Glob pattern used when --peer-dir is provided. Defaults to '*_logits.npz'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing files.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix appended to backups before overwriting peers (set to '' to disable).",
    )
    return parser.parse_args(argv)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def run(reference_path: Path, peer_paths: Iterable[Path], *, dry_run: bool, backup_suffix: str) -> int:
    if not peer_paths:
        logger.info("No peer artefacts provided; nothing to do.")
        return 0
    reference = _load_artifact(reference_path)
    updated = 0
    skipped = 0
    for peer_path in peer_paths:
        if peer_path == reference.path:
            logger.debug("Skipping reference artefact in peer set: %s", peer_path)
            continue
        peer = _load_artifact(peer_path)
        if peer.has_train_metadata() and peer.has_val_metadata():
            logger.info("Peer already metadata-complete: %s", peer_path)
            skipped += 1
            continue
        changed, keys_added = _copy_missing_metadata(reference, peer)
        if not changed:
            logger.info("Peer already contains metadata: %s", peer_path)
            skipped += 1
            continue
        logger.info("Ready to annotate %s with keys %s", peer_path, ", ".join(keys_added))
        if dry_run:
            continue
        _write_with_backup(peer_path, backup_suffix)
        peer.save()
        updated += 1
    logger.info("Peer metadata refresh complete: updated=%s skipped=%s", updated, skipped)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _configure_logging()
    args = _parse_args(argv)
    peer_paths = _collect_peer_paths(args)
    try:
        return run(
            Path(args.reference).expanduser().resolve(),
            peer_paths,
            dry_run=bool(args.dry_run),
            backup_suffix=str(args.backup_suffix),
        )
    except Exception:  # pragma: no cover - CLI safety net
        logger.error("Failed to refresh peer logits metadata", exc_info=True)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
