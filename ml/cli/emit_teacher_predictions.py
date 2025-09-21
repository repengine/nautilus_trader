#!/usr/bin/env python3
"""
CLI to generate teacher prediction targets for student distillation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml.training.distillation.emit import generate_teacher_targets


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit teacher predictions for student distillation",
    )
    parser.add_argument(
        "--features_npz",
        required=True,
        help="Path to features npz containing X_train/X_val",
    )
    parser.add_argument("--out_npz", required=True, help="Output npz path for teacher logits")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output",
    )
    args = parser.parse_args(argv)

    features_path = Path(args.features_npz)
    out_path = Path(args.out_npz)
    if out_path.exists() and not args.overwrite:
        logger.error("Output file %s already exists. Use --overwrite to replace.", out_path)
        return 1

    result_path = generate_teacher_targets(features_path, out_path)
    print(f"Saved teacher logits to {result_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
