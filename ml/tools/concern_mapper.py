"""Map API methods to concerns by pattern-matching method names.

Usage:
    python -m ml.tools.concern_mapper --input reports/architecture/api_index.json \
        --output reports/architecture/concern_map.json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


# Concern patterns: method name patterns → concern category
CONCERN_PATTERNS: dict[str, list[str]] = {
    "persistence_load": ["load_*", "read_*", "fetch_*", "get_from_*", "retrieve_*"],
    "persistence_save": ["save_*", "write_*", "persist_*", "store_*", "put_*"],
    "event_emission": ["emit_*", "publish_*", "notify_*", "broadcast_*", "send_event_*"],
    "validation": ["validate_*", "check_*", "verify_*", "assert_*", "ensure_*", "is_valid_*"],
    "lifecycle": ["start_*", "stop_*", "init_*", "shutdown_*", "dispose_*", "cleanup_*", "close_*"],
    "computation": ["compute_*", "calculate_*", "process_*", "transform_*", "derive_*"],
    "registration": ["register_*", "unregister_*", "deregister_*"],
    "collection_add": ["add_*", "append_*", "insert_*", "push_*"],
    "collection_remove": ["remove_*", "delete_*", "pop_*", "clear_*"],
    "query": ["get_*", "find_*", "lookup_*", "search_*", "query_*", "list_*", "filter_*"],
    "update": ["update_*", "set_*", "modify_*", "patch_*", "refresh_*"],
    "configuration": ["configure_*", "config_*", "setup_*", "with_*"],
    "serialization": ["to_dict", "from_dict", "to_json", "from_json", "serialize_*", "deserialize_*"],
    "health_metrics": ["health_*", "status_*", "metrics_*", "stats_*", "report_*"],
    "locking": ["lock_*", "unlock_*", "acquire_*", "release_*", "with_lock_*"],
    "caching": ["cache_*", "cached_*", "invalidate_*", "evict_*"],
    "batching": ["batch_*", "bulk_*", "flush_*"],
}


def match_concern(method_name: str) -> str | None:
    """Return the concern name if method matches a pattern, else None."""
    for concern, patterns in CONCERN_PATTERNS.items():
        for pattern in patterns:
            if fnmatch.fnmatch(method_name, pattern):
                return concern
    return None


def map_concerns(api_index: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Map API index entries to concerns.

    Args:
        api_index: List of API entries from api_index.json

    Returns:
        Tuple of (concern map, unmatched entries)
    """
    concern_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmatched: list[dict[str, Any]] = []

    for entry in api_index:
        # Support both old and new API index formats
        entry_kind = entry.get("kind") or entry.get("type", "")
        if entry_kind != "method":
            continue

        method_name = entry.get("name", "")
        # Extract file path - support both formats
        file_path = entry.get("path") or entry.get("file", "")
        # Extract class from qualname if not provided directly
        qualname = entry.get("qualname") or entry.get("qualified_name", "")
        class_name = entry.get("class", "")
        if not class_name and "." in qualname:
            parts = qualname.rsplit(".", 2)
            if len(parts) >= 2:
                class_name = parts[-2]
        # Line number
        line = entry.get("line_start") or entry.get("line", 0)

        concern = match_concern(method_name)

        if concern:
            concern_map[concern].append({
                "file": file_path,
                "class": class_name,
                "method": method_name,
                "line": line,
                "qualified_name": qualname,
            })
        else:
            unmatched.append({
                "file": file_path,
                "class": class_name,
                "method": method_name,
                "line": line,
            })

    return dict(concern_map), unmatched


def filter_by_domain(
    concern_map: dict[str, list[dict[str, Any]]],
    domain: str,
) -> dict[str, list[dict[str, Any]]]:
    """Filter concern map to only include entries from a specific domain.

    Args:
        concern_map: Full concern map
        domain: Domain path prefix (e.g., 'ml/features', 'ml/registry')

    Returns:
        Filtered concern map
    """
    filtered: dict[str, list[dict[str, Any]]] = {}
    for concern, entries in concern_map.items():
        domain_entries = [e for e in entries if e["file"].startswith(domain)]
        if domain_entries:
            filtered[concern] = domain_entries
    return filtered


def generate_summary(concern_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Generate summary statistics for the concern map."""
    total_methods = sum(len(entries) for entries in concern_map.values())

    # Count by file
    file_counts: dict[str, int] = defaultdict(int)
    for entries in concern_map.values():
        for entry in entries:
            file_counts[entry["file"]] += 1

    # Count by class
    class_counts: dict[str, int] = defaultdict(int)
    for entries in concern_map.values():
        for entry in entries:
            if entry.get("class"):
                class_counts[entry["class"]] += 1

    return {
        "total_concerns": len(concern_map),
        "total_methods": total_methods,
        "methods_by_concern": {c: len(e) for c, e in sorted(concern_map.items())},
        "top_files": dict(sorted(file_counts.items(), key=lambda x: -x[1])[:20]),
        "top_classes": dict(sorted(class_counts.items(), key=lambda x: -x[1])[:20]),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for concern mapper tool."""
    parser = argparse.ArgumentParser(
        description="Map API methods to concerns by pattern matching",
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
        help="Output path for concern_map.json",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter to specific domain (e.g., ml/features)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only output summary statistics",
    )

    args = parser.parse_args(argv)

    # Load API index
    with args.input.open() as f:
        api_index = json.load(f)

    # Map concerns
    concern_map, unmatched = map_concerns(api_index)

    # Filter by domain if specified
    if args.domain:
        concern_map = filter_by_domain(concern_map, args.domain)

    # Generate summary
    summary = generate_summary(concern_map)

    # Build output
    output: dict[str, Any] = {
        "summary": summary,
        "patterns": CONCERN_PATTERNS,
    }

    if not args.summary_only:
        output["concerns"] = concern_map
        output["unmatched_count"] = len(unmatched)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Mapped {summary['total_methods']} methods to {summary['total_concerns']} concerns")
    print(f"Unmatched methods: {len(unmatched)}")
    print(f"Output written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
