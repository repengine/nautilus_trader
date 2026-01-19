#!/usr/bin/env python3
"""Validate streaming teacher export manifests against the model registry."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml.common.logging_config import configure_logging
from ml.training.event_driven.teacher_export import validate_streaming_teacher_export


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate streaming teacher export manifests against the model registry.",
    )
    parser.add_argument(
        "--manifest",
        dest="manifest_path",
        type=Path,
        required=True,
        help="Path to the streaming manifest JSON.",
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=None,
        help="Optional override for registry root path.",
    )
    parser.add_argument(
        "--allow-missing-registry",
        action="store_true",
        help="Allow validation to pass when registry is unavailable.",
    )
    return parser


def main() -> None:
    configure_logging()
    args = _build_parser().parse_args()

    result = validate_streaming_teacher_export(
        args.manifest_path,
        registry_root=args.registry_root,
        require_registry=not args.allow_missing_registry,
    )

    if result.errors:
        logger.error(
            "streaming_export_validation_failed",
            extra={
                "model_id": result.model_id,
                "errors": list(result.errors),
                "manifest": str(args.manifest_path),
            },
        )
        result.raise_for_errors()

    logger.info(
        "streaming_export_validation_ok",
        extra={
            "model_id": result.model_id,
            "manifest": str(args.manifest_path),
        },
    )


if __name__ == "__main__":
    main()
