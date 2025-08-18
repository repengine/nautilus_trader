"""
Typed config loader helpers.

Layered merge order (lowest to highest priority):
- Defaults (code)
- File (YAML/JSON; optional)
- Environment (prefix ML_)
- CLI overrides (call-site)

"""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar, cast

import msgspec


T = TypeVar("T")


def load_from_file(path: str | None, t: type[T], default: T) -> T:
    if not path:
        return default
    try:
        with open(path) as f:
            raw = f.read()
        data = json.loads(raw)
        return msgspec.json.decode(msgspec.json.encode(data), type=t)
    except Exception:
        return default


def merge_env(prefix: str, t: type[T], base: T) -> T:
    # Simple env overlay: expects a single JSON blob in {PREFIX}_JSON
    key = f"{prefix}_JSON"
    blob = os.getenv(key)
    if not blob:
        return base
    try:
        data = json.loads(blob)
        partial = msgspec.json.decode(msgspec.json.encode(data), type=t)
        # Shallow merge: prefer env fields when not None
        return cast(T, _merge_structs(base, partial))
    except Exception:
        return base


def _merge_structs(a: Any, b: Any) -> Any:
    # Best-effort shallow merge for msgspec structs
    if type(a) is not type(b):
        return a
    fields = getattr(a, "__struct_fields__", []) or getattr(a, "__annotations__", {}).keys()
    updates: dict[str, Any] = {}
    for name in fields:
        av = getattr(a, name)
        bv = getattr(b, name)
        updates[name] = bv if bv is not None else av
    return type(a)(**updates)


__all__ = ["load_from_file", "merge_env"]
