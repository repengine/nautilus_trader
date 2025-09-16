#!/usr/bin/env python3
"""
Promote FeatureRegistry feature sets based on quality gates and metrics.
This CLI loads a metrics JSON file (e.g., from evaluate_predictions.py), updates the
FeatureRegistry perf_digest for the given feature_set_id, and validates against a set of
quality gates. If all required gates pass, the feature set is promoted to PROD.

Examples
--------
1) Promote with inline gates:
   python -m ml.scripts.promote_features \
       --feature_registry_dir ~/.nautilus/ml/features \
       --feature_set_id feature_set_123 \
       --metrics_json /tmp/metrics.json \
       --gate pr_auc gte 0.70 required \
       --gate logloss lte 0.60 required

2) Promote with gates from a file (JSON):
   python -m ml.scripts.promote_features \
       --feature_registry_dir ~/.nautilus/ml/features \
       --feature_set_id feature_set_123 \
       --metrics_json /tmp/metrics.json \
       --gates_json /tmp/gates.json

metrics.json schema (example)::
    {"roc_auc": 0.78, "pr_auc": 0.72, "logloss": 0.58}

gates.json schema (example)::
    {
      "gates": [
        {"metric": "pr_auc", "comparison": "gte", "threshold": 0.7, "required": true},
        {"metric": "logloss", "comparison": "lte", "threshold": 0.6, "required": true}
      ]
    }

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.registry.dataclasses import QualityGate
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureStage


def _parse_inline_gates(gate_args: list[str]) -> list[QualityGate]:
    """
    Parse inline gates of the form: metric comparison threshold required|optional.

    Examples
    --------
    gate_args = ["pr_auc", "gte", "0.7", "required", "logloss", "lte", "0.6", "required"]
    """
    gates: list[QualityGate] = []
    if not gate_args:
        return gates
    if len(gate_args) % 4 != 0:
        raise SystemExit(
            "--gate must be provided in groups of 4: metric comparison threshold required|optional",
        )
    for i in range(0, len(gate_args), 4):
        metric, comp, thresh, req = gate_args[i : i + 4]
        required = req.lower() in ("req", "required", "true", "1")
        try:
            threshold = float(thresh)
        except ValueError as exc:  # pragma: no cover - argparse validation should catch
            raise SystemExit(f"Invalid threshold: {thresh}") from exc
        gates.append(
            QualityGate(
                metric_name=metric,
                threshold=threshold,
                comparison=comp,
                required=required,
            ),
        )
    return gates


def _load_gates_from_file(path: Path) -> list[QualityGate]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result: list[QualityGate] = []
    for g in data.get("gates", []):
        result.append(
            QualityGate(
                metric_name=str(g["metric"]),
                threshold=float(g["threshold"]),
                comparison=str(g.get("comparison", "gte")),
                required=bool(g.get("required", True)),
            ),
        )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Promote feature sets based on metrics and gates")
    ap.add_argument("--feature_registry_dir", required=True)
    ap.add_argument("--feature_set_id", required=True)
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--gates_json", required=False)
    ap.add_argument(
        "--gate",
        nargs="*",
        default=[],
        help="Inline gate: metric comparison threshold required|optional (repeatable, in groups of 4)",
    )
    args = ap.parse_args(argv)

    registry = FeatureRegistry(Path(args.feature_registry_dir))
    info = registry.get_feature_set(args.feature_set_id)
    if info is None:
        raise SystemExit(f"Unknown feature_set_id: {args.feature_set_id}")

    # Load metrics and persist into perf_digest
    metrics: dict[str, Any] = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int | float))}
    registry.update_manifest(args.feature_set_id, perf_digest=numeric_metrics)

    # Build gates
    gates: list[QualityGate] = []
    if args.gates_json:
        gates.extend(_load_gates_from_file(Path(args.gates_json)))
    gates.extend(_parse_inline_gates(args.gate))
    if not gates:
        raise SystemExit("No quality gates specified. Use --gate or --gates_json.")

    # Validate and promote
    ok = registry.validate_and_promote(args.feature_set_id, gates)
    stage = registry.get_feature_set(args.feature_set_id).manifest.stage  # type: ignore[union-attr]
    print(json.dumps({"promoted": bool(ok), "stage": stage.value}))
    # Optional explicit success/failure exit code:
    return 0 if ok and stage == FeatureStage.PROD else 1


if __name__ == "__main__":
    raise SystemExit(main())
