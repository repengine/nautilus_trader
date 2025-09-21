"""
Persistence helpers for IngestState.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ml.data.ingest.resume import IngestState


def load_state(path: str | Path) -> IngestState:
    p = Path(path)
    if not p.exists():
        return IngestState()
    try:
        data = json.loads(p.read_text())
        mapping = data.get("last_ts_ns_by_instrument", {}) if isinstance(data, dict) else {}
        if not isinstance(mapping, dict):
            mapping = {}
        # Coerce keys to str and values to int
        fixed = {str(k): int(v) for k, v in mapping.items()}
        return IngestState(last_ts_ns_by_instrument=fixed)
    except Exception:
        return IngestState()


def save_state(path: str | Path, state: IngestState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    payload = asdict(state)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(p)
