#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Pre-commit hook to ensure mypy passes with zero errors.
"""

import shutil
import subprocess
import sys


def _has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def check_mypy(changed_files):
    """
    Run mypy on changed Python files.
    """
    # Filter for Python files under ml/ except tests and __init__.py
    allowed_prefixes = ("ml/",)
    blocked_prefixes = ("ml/tests/",)
    allowed_singles: set[str] = set()
    python_files = [
        f
        for f in changed_files
        if f.endswith(".py")
        and not f.endswith("__init__.py")
        and f.startswith(allowed_prefixes)
        and not f.startswith(blocked_prefixes)
        and (not allowed_singles or f in allowed_singles or True)
    ]

    if not python_files:
        return True, "No Python files to check"

    print(f"Running mypy on {len(python_files)} file(s) (strict mode)...")

    # Prefer the project's Poetry venv if available; fallback to uv; then system
    if _has_cmd("poetry"):
        cmd = [
            "poetry",
            "run",
            "mypy",
            "--config-file",
            "pyproject.toml",
            "--strict",
            "--no-error-summary",
            *python_files,
        ]
    elif _has_cmd("uv"):
        cmd = [
            "uv",
            "run",
            "--active",
            "--no-sync",
            "mypy",
            "--config-file",
            "pyproject.toml",
            "--strict",
            "--no-error-summary",
            *python_files,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            "pyproject.toml",
            "--strict",
            "--no-error-summary",
            *python_files,
        ]

    # Run mypy
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Parse errors
        errors = []
        error_count = 0

        for line in result.stdout.split("\n"):
            if ": error:" in line:
                error_count += 1
                # Show first 10 errors
                if len(errors) < 10:
                    errors.append(line.strip())
            elif ": note:" in line and len(errors) < 10:
                # Include notes for context
                errors.append(f"  {line.strip()}")

        message = f"Found {error_count} mypy error(s):\n"
        message += "\n".join(errors)

        if error_count > 10:
            message += f"\n... and {error_count - 10} more errors"

        return False, message

    # Check if mypy actually ran
    if "Success:" in result.stdout:
        match = result.stdout.strip().split()
        if len(match) >= 2:
            return True, f"✅ Mypy passed: {match[-1]}"

    # No errors
    return True, "✅ Mypy passed with no errors"


def main():
    """
    Execute the main mypy checking process.
    """
    changed_files = [f for f in sys.argv[1:] if f.startswith("ml/")]

    if not changed_files:
        return 0

    passed, message = check_mypy(changed_files)

    print(message)

    if not passed:
        print("\n❌ Mypy type checking failed!")
        print("Fix the type errors above before committing.")
        print("\nCommon fixes:")
        print("- Add type annotations to function arguments and return values")
        print("- Use Optional[T] for values that can be None")
        print("- Import types from typing module")
        print("- Check that all class attributes are properly typed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
