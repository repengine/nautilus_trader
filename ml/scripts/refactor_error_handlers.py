#!/usr/bin/env python3
"""
Automated refactoring script for Phase 1.3: Standardize Error Handling.

This script automatically refactors try/except patterns to use standardized
error handling utilities from ml.common.error_handlers.

Usage:
    python ml/scripts/refactor_error_handlers.py --dry-run  # Preview changes
    python ml/scripts/refactor_error_handlers.py            # Apply changes
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import NamedTuple

from ml.common.subprocess_utils import SubprocessExecutionError
from ml.common.subprocess_utils import run_command


class RefactoringPattern(NamedTuple):
    """A refactoring pattern to apply."""

    name: str
    search_pattern: re.Pattern[str]
    replacement: str
    description: str


# Define refactoring patterns
PATTERNS = [
    RefactoringPattern(
        name="registry_fallback",
        search_pattern=re.compile(
            r"try:\s*\n"
            r"([\s\S]*?)"
            r"except Exception as e:\s*\n"
            r"\s*logger\.(warning|warn)\(['\"].*?['\"], e\).*\n"
            r"\s*return (None|\[\]|\{\})",
            re.MULTILINE,
        ),
        replacement=r"@with_fallback(fallback_value=\3, log_level='warning')\n\1",
        description="Convert registry operations with warning + return to @with_fallback decorator",
    ),
]


def _to_text(stream: str | bytes | None) -> str:
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="ignore")
    return stream or ""


def find_files_with_error_patterns(ml_dir: Path) -> list[tuple[Path, int]]:
    """Find all Python files with 'except Exception as e:' patterns."""
    try:
        result = run_command(
            ["grep", "-r", "except Exception as e:", str(ml_dir), "--include=*.py", "-c"],
            capture_output=True,
            text=True,
            check=False,
        )

        files_with_counts = []
        stdout_text = _to_text(result.stdout)
        for line in stdout_text.strip().split("\n"):
            if ":" in line:
                file_path, count_str = line.rsplit(":", 1)
                try:
                    count = int(count_str)
                    files_with_counts.append((Path(file_path), count))
                except ValueError:
                    continue

        # Sort by count descending
        files_with_counts.sort(key=lambda x: x[1], reverse=True)
        return files_with_counts

    except SubprocessExecutionError as exc:
        print(f"Error finding files: {exc}")
        return []


def needs_import_added(file_path: Path) -> tuple[bool, bool, bool, bool]:
    """Check which error handler imports are needed."""
    content = file_path.read_text()

    needs_db_op = "db_operation_handler" not in content and "except Exception as e:" in content
    needs_registry_op = "registry_operation_handler" not in content and "registry" in content.lower()
    needs_with_db = "with_db_error_handling" not in content
    needs_with_fallback = "with_fallback" not in content and "fallback" in content.lower()

    return needs_db_op, needs_registry_op, needs_with_db, needs_with_fallback


def add_error_handler_imports(file_path: Path, dry_run: bool = False) -> bool:
    """Add error handler imports to a file if needed."""
    content = file_path.read_text()

    # Check if already has the imports
    if "from ml.common.error_handlers import" in content:
        return False

    needs_db_op, needs_registry_op, needs_with_db, needs_with_fallback = needs_import_added(file_path)

    if not (needs_db_op or needs_registry_op or needs_with_db or needs_with_fallback):
        return False

    # Find the last 'from ml.' import
    import_pattern = re.compile(r"^from ml\.", re.MULTILINE)
    matches = list(import_pattern.finditer(content))

    if not matches:
        # No ml imports, skip
        return False

    # Insert after the last ml import
    last_match = matches[-1]
    insert_pos = content.find("\n", last_match.end()) + 1

    imports_needed = []
    if needs_db_op:
        imports_needed.append("db_operation_handler")
    if needs_registry_op:
        imports_needed.append("registry_operation_handler")
    if needs_with_db:
        imports_needed.append("with_db_error_handling")
    if needs_with_fallback:
        imports_needed.append("with_fallback")

    import_line = f"from ml.common.error_handlers import {', '.join(imports_needed)}\n"

    new_content = content[:insert_pos] + import_line + content[insert_pos:]

    if not dry_run:
        file_path.write_text(new_content)
        print(f"✓ Added imports to {file_path}")
    else:
        print(f"[DRY RUN] Would add imports to {file_path}")

    return True


def refactor_file(file_path: Path, dry_run: bool = False) -> int:
    """Refactor a single file. Returns number of patterns refactored."""
    content = file_path.read_text()
    original_content = content
    changes_count = 0

    for pattern in PATTERNS:
        matches = list(pattern.search_pattern.finditer(content))
        if matches:
            content = pattern.search_pattern.sub(pattern.replacement, content)
            changes_count += len(matches)
            if dry_run:
                print(f"  [DRY RUN] Would apply pattern '{pattern.name}': {len(matches)} matches")
            else:
                print(f"  ✓ Applied pattern '{pattern.name}': {len(matches)} matches")

    if content != original_content and not dry_run:
        file_path.write_text(content)

    return changes_count


def main() -> None:
    """Main refactoring entry point."""
    parser = argparse.ArgumentParser(description="Refactor error handling patterns")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top files to refactor")
    parser.add_argument("--ml-dir", type=Path, default=Path("ml"), help="ML directory path")

    args = parser.parse_args()

    ml_dir = Path(args.ml_dir)
    if not ml_dir.exists():
        print(f"Error: {ml_dir} does not exist")
        return

    print(f"Finding files with error handling patterns in {ml_dir}...")
    files_with_counts = find_files_with_error_patterns(ml_dir)

    print(f"\nFound {len(files_with_counts)} files with error patterns")
    print(f"Refactoring top {args.top_n} files...")
    print()

    total_patterns_refactored = 0
    total_imports_added = 0
    files_modified = 0

    for file_path, count in files_with_counts[: args.top_n]:
        print(f"\n📝 {file_path} ({count} patterns)")

        # Add imports if needed
        if add_error_handler_imports(file_path, dry_run=args.dry_run):
            total_imports_added += 1

        # Refactor patterns
        patterns_count = refactor_file(file_path, dry_run=args.dry_run)
        if patterns_count > 0:
            total_patterns_refactored += patterns_count
            files_modified += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed: {min(args.top_n, len(files_with_counts))}")
    print(f"Files modified: {files_modified}")
    print(f"Imports added: {total_imports_added}")
    print(f"Patterns refactored: {total_patterns_refactored}")

    if args.dry_run:
        print("\n⚠️  DRY RUN - No files were modified")
        print("Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
