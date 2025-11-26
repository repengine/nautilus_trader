"""
Shared helpers for resolving symbol directories and selecting parquet files.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path


def resolve_symbol_data_dir(base_dir: Path, symbol: str) -> Path | None:
    """
    Resolve the on-disk directory for ``symbol`` under ``base_dir``.

    Args:
        base_dir: Root directory that contains per-symbol subdirectories.
        symbol: Symbol with or without venue suffix (e.g., ``"SPY"`` or ``"SPY.XNAS"``).

    Returns:
        Directory path for the symbol if it exists, otherwise ``None``.
    """
    normalized = symbol.strip()
    if not normalized:
        return None
    resolved_root = base_dir.expanduser().resolve()
    return _resolve_symbol_dir_cached(resolved_root, normalized.upper())


@lru_cache(maxsize=2048)
def _resolve_symbol_dir_cached(base_dir: Path, symbol_upper: str) -> Path | None:
    if not base_dir.exists():
        return None
    direct = _match_case_insensitive(base_dir, symbol_upper)
    if direct is not None:
        return direct
    head, _, _tail = symbol_upper.partition(".")
    if head and head != symbol_upper:
        head_match = _match_case_insensitive(base_dir, head)
        if head_match is not None:
            return head_match
    if "." not in symbol_upper:
        prefix = f"{symbol_upper}."
        for entry in sorted(base_dir.iterdir()):
            if entry.is_dir() and entry.name.upper().startswith(prefix):
                return entry
    return None


def _match_case_insensitive(root: Path, candidate: str) -> Path | None:
    direct = root / candidate
    if direct.exists():
        return direct
    candidate_upper = candidate.upper()
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.upper() == candidate_upper:
            return entry
    return None


def select_latest_symbol_file(
    directory: Path,
    symbol_prefix: str,
    token: str,
) -> Path | None:
    """
    Select the freshest parquet file for ``symbol_prefix`` and ``token``.

    Args:
        directory: Directory containing parquet files.
        symbol_prefix: Directory name (often includes venue) used as file prefix.
        token: Token inserted between symbol and suffix (e.g., ``"bbo"`` or ``"mbp-10"``).

    Returns:
        Path to the preferred parquet file or ``None`` if no candidates exist.
    """
    candidates = list(_iter_symbol_files(directory, symbol_prefix, token))
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    return ranked[-1]


def _iter_symbol_files(directory: Path, symbol_prefix: str, token: str) -> Iterable[Path]:
    pattern = f"{symbol_prefix}_{token}_*.parquet"
    yield from directory.glob(pattern)
    fallback = directory / f"{symbol_prefix}_{token}.parquet"
    if fallback.exists():
        yield fallback
