#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ml"

ALLOWLIST = {
    str(ML_DIR / "config" / "events.py"),
}

STAGE_LITERALS = (
    "CATALOG_WRITTEN",
    "FEATURE_COMPUTED",
    "PREDICTION_EMITTED",
    "SIGNAL_EMITTED",
    "INGESTED",
)

STAGE_RE = re.compile(r"\b(" + "|".join(map(re.escape, STAGE_LITERALS)) + r")\b")


def main() -> int:
    violations: list[str] = []
    for py in ML_DIR.rglob("*.py"):
        p = str(py)
        if p in ALLOWLIST:
            continue
        if "/tests/" in p or "/docs/" in p or "/migrations/" in p:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        # Allow Stage.<name>.value usages
        if "Stage." in text:
            # Potential mix; still check raw literals
            pass
        for m in STAGE_RE.finditer(text):
            # Skip occurrences that look like Stage.NAME.value
            start = max(0, m.start() - 20)
            context = text[start:m.end() + 20]
            if ".value" in context or "Stage." in context:
                continue
            violations.append(f"{py}: raw stage literal '{m.group(1)}'")

    if violations:
        print("Event constants validation FAILED:\n" + "\n".join(violations))
        return 1
    print("Event constants validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

