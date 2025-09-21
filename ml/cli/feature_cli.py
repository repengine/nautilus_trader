#!/usr/bin/env python3
"""
Feature registry CLI wrappers leveraging :mod:`ml.tasks.registry`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureRole
from ml.tasks.registry import FeaturePromotionGate
from ml.tasks.registry import deprecate_feature_set
from ml.tasks.registry import promote_feature_set
from ml.tasks.registry import register_default_feature_set


def _parse_gates(raw_gates: Sequence[dict[str, Any]]) -> list[FeaturePromotionGate]:
    return [
        FeaturePromotionGate(
            metric_name=str(g["metric_name"]),
            threshold=float(g["threshold"]),
            comparison=str(g.get("comparison", "gte")),
            required=bool(g.get("required", True)),
        )
        for g in raw_gates
    ]


def cli_register_default(
    registry_path: str,
    name: str = "default",
    version: str | None = None,
    role: str = "student",
    data_requirements: str = "l1_only",
) -> str:
    role_enum = FeatureRole(role)
    req_enum = DataRequirements(data_requirements)
    return register_default_feature_set(
        Path(registry_path),
        name=name,
        version=version,
        role=role_enum,
        data_requirements=req_enum,
    )


def cli_promote_with_gates(
    registry_path: str,
    feature_set_id: str,
    gates: list[dict[str, Any]],
) -> bool:
    gate_objs = _parse_gates(gates)
    return promote_feature_set(Path(registry_path), feature_set_id=feature_set_id, gates=gate_objs)


def cli_deprecate(
    registry_path: str,
    feature_set_id: str,
    reason: str | None = None,
) -> None:
    deprecate_feature_set(Path(registry_path), feature_set_id=feature_set_id, reason=reason)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature registry operations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser(
        "register-default",
        help="Register default feature config",
    )
    register_parser.add_argument("registry_path")
    register_parser.add_argument("--name", default="default")
    register_parser.add_argument("--version")
    register_parser.add_argument("--role", default="student")
    register_parser.add_argument("--data-requirements", default="l1_only")

    promote_parser = subparsers.add_parser("promote", help="Promote a feature set with gates")
    promote_parser.add_argument("registry_path")
    promote_parser.add_argument("feature_set_id")
    promote_parser.add_argument(
        "--gate",
        action="append",
        nargs=3,
        metavar=("metric", "comparison", "threshold"),
    )

    deprecate_parser = subparsers.add_parser("deprecate", help="Deprecate a feature set")
    deprecate_parser.add_argument("registry_path")
    deprecate_parser.add_argument("feature_set_id")
    deprecate_parser.add_argument("--reason")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "register-default":
        result = cli_register_default(
            registry_path=args.registry_path,
            name=args.name,
            version=args.version,
            role=args.role,
            data_requirements=args.data_requirements,
        )
        print(result)
        return 0

    if args.command == "promote":
        gates: list[dict[str, Any]] = []
        if args.gate:
            for metric, comparison, threshold in args.gate:
                gates.append(
                    {
                        "metric_name": metric,
                        "comparison": comparison,
                        "threshold": float(threshold),
                        "required": True,
                    },
                )
        success = cli_promote_with_gates(
            registry_path=args.registry_path,
            feature_set_id=args.feature_set_id,
            gates=gates,
        )
        print("PROMOTED" if success else "FAILED")
        return 0 if success else 1

    if args.command == "deprecate":
        cli_deprecate(
            registry_path=args.registry_path,
            feature_set_id=args.feature_set_id,
            reason=args.reason,
        )
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
