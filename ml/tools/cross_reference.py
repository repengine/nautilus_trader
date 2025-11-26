"""Find same-named methods/functions across different files.

Detects conceptual duplication where the same concern appears in multiple places.

Usage:
    python -m ml.tools.cross_reference --input reports/architecture/api_index.json \
        --output reports/architecture/cross_refs.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def find_cross_references(
    api_index: list[dict[str, Any]],
    *,
    min_occurrences: int = 2,
    exclude_tests: bool = True,
    exclude_dunder: bool = True,
) -> dict[str, list[dict[str, str]]]:
    """Find methods/functions with the same name in different files.

    Args:
        api_index: List of API entries from api_index.json
        min_occurrences: Minimum occurrences to be considered a cross-reference
        exclude_tests: Exclude test files from analysis
        exclude_dunder: Exclude __dunder__ methods

    Returns:
        Dict mapping method name → list of locations
    """
    # Group by name
    by_name: dict[str, list[dict[str, str]]] = defaultdict(list)

    for entry in api_index:
        # Support both old and new API index formats
        entry_kind = entry.get("kind") or entry.get("type", "")
        if entry_kind not in ("method", "function"):
            continue

        name = entry.get("name", "")
        file_path = entry.get("path") or entry.get("file", "")

        # Skip test files if requested
        if exclude_tests and "/tests/" in file_path:
            continue

        # Skip dunder methods if requested
        if exclude_dunder and name.startswith("__") and name.endswith("__"):
            continue

        # Extract class from qualname if not provided
        qualname = entry.get("qualname") or entry.get("qualified_name", "")
        class_name = entry.get("class", "")
        if not class_name and "." in qualname and entry_kind == "method":
            parts = qualname.rsplit(".", 2)
            if len(parts) >= 2:
                class_name = parts[-2]

        by_name[name].append({
            "file": file_path,
            "class": class_name,
            "line": entry.get("line_start") or entry.get("line", 0),
            "qualified_name": qualname,
            "type": entry_kind,
        })

    # Filter to those with multiple occurrences in different files
    cross_refs: dict[str, list[dict[str, str]]] = {}
    for name, locations in by_name.items():
        # Get unique files
        files = {loc["file"] for loc in locations}
        if len(files) >= min_occurrences:
            cross_refs[name] = sorted(locations, key=lambda x: x["file"])

    return cross_refs


def analyze_overlap_pairs(
    cross_refs: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """Identify file pairs that share many method names.

    Args:
        cross_refs: Cross-reference map from find_cross_references

    Returns:
        List of file pairs with shared method counts, sorted by overlap
    """
    pair_methods: dict[tuple[str, str], list[str]] = defaultdict(list)

    for method_name, locations in cross_refs.items():
        files = sorted({loc["file"] for loc in locations})
        # Generate all pairs
        for i, file1 in enumerate(files):
            for file2 in files[i + 1 :]:
                pair_methods[(file1, file2)].append(method_name)

    # Convert to list and sort by overlap
    pairs = [
        {
            "file1": pair[0],
            "file2": pair[1],
            "shared_methods": methods,
            "count": len(methods),
        }
        for pair, methods in pair_methods.items()
        if len(methods) >= 3  # Only pairs with 3+ shared methods
    ]

    return sorted(pairs, key=lambda x: -x["count"])


def filter_by_domain(
    cross_refs: dict[str, list[dict[str, str]]],
    domain: str,
) -> dict[str, list[dict[str, str]]]:
    """Filter cross-references to entries within a domain."""
    filtered: dict[str, list[dict[str, str]]] = {}
    for name, locations in cross_refs.items():
        domain_locs = [loc for loc in locations if loc["file"].startswith(domain)]
        if len(domain_locs) >= 2:
            filtered[name] = domain_locs
    return filtered


def generate_summary(cross_refs: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    """Generate summary statistics."""
    total_names = len(cross_refs)
    total_locations = sum(len(locs) for locs in cross_refs.values())

    # Most duplicated names
    top_duplicated = sorted(
        [(name, len(locs)) for name, locs in cross_refs.items()],
        key=lambda x: -x[1],
    )[:30]

    # Files with most cross-referenced methods
    file_counts: dict[str, int] = defaultdict(int)
    for locations in cross_refs.values():
        for loc in locations:
            file_counts[loc["file"]] += 1

    return {
        "total_cross_referenced_names": total_names,
        "total_locations": total_locations,
        "top_duplicated_names": dict(top_duplicated),
        "top_files_by_cross_refs": dict(
            sorted(file_counts.items(), key=lambda x: -x[1])[:20]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for cross-reference finder."""
    parser = argparse.ArgumentParser(
        description="Find same-named methods across different files",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to api_index.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for cross_refs.json",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter to specific domain (e.g., ml/features)",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Minimum occurrences to report (default: 2)",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files in analysis",
    )
    parser.add_argument(
        "--include-overlap-pairs",
        action="store_true",
        help="Include file pair overlap analysis",
    )

    args = parser.parse_args(argv)

    # Load API index
    with args.input.open() as f:
        api_index = json.load(f)

    # Find cross-references
    cross_refs = find_cross_references(
        api_index,
        min_occurrences=args.min_occurrences,
        exclude_tests=not args.include_tests,
    )

    # Filter by domain if specified
    if args.domain:
        cross_refs = filter_by_domain(cross_refs, args.domain)

    # Generate summary
    summary = generate_summary(cross_refs)

    # Build output
    output: dict[str, Any] = {
        "summary": summary,
        "cross_references": cross_refs,
    }

    # Add overlap pairs if requested
    if args.include_overlap_pairs:
        output["overlap_pairs"] = analyze_overlap_pairs(cross_refs)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Found {summary['total_cross_referenced_names']} names in multiple files")
    print(f"Total locations: {summary['total_locations']}")
    print(f"Output written to: {args.output}")

    if args.include_overlap_pairs:
        pairs = output.get("overlap_pairs", [])
        if pairs:
            print(f"\nTop overlap pairs:")
            for pair in pairs[:5]:
                print(f"  {pair['file1']} <-> {pair['file2']}: {pair['count']} shared")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
