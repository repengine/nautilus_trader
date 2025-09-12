#!/usr/bin/env python3
"""Filter API index to focus on core functionality."""

import json
import argparse
from pathlib import Path
from typing import Any

def should_include(item: dict[str, Any], filters: dict) -> bool:
    """Determine if an API item should be included based on filters."""
    
    module = item['module']
    name = item['name']
    kind = item['kind']
    
    # Exclude test code unless explicitly included
    if not filters['include_tests'] and 'test' in module.lower():
        return False
    
    # Exclude archived/backup code
    if any(x in module.lower() for x in ['archive', 'backup', '_old', '_temp']):
        return False
    
    # Exclude examples unless explicitly included
    if not filters['include_examples'] and '.examples.' in module:
        return False
    
    # Exclude CLI internals unless requested
    if not filters['include_cli'] and '.cli.' in module and name != 'main':
        return False
    
    # Include only documented items if requested
    if filters['documented_only'] and not item.get('docstring'):
        return False
    
    # Filter by kind
    if filters['kinds'] and kind not in filters['kinds']:
        return False
    
    # Include only core modules if specified
    if filters['core_only']:
        core_modules = [
            '.actors.', '.stores.', '.registry.', '.features.',
            '.strategies.', '.training.', '.core.', '.monitoring.'
        ]
        if not any(m in module for m in core_modules):
            return False
    
    # Exclude deep nesting if requested
    if filters['max_depth'] and module.count('.') > filters['max_depth']:
        return False
    
    return True

def create_summary_index(data: list[dict], include_docstrings: bool = False) -> list[dict]:
    """Create a condensed version with optional docstring removal."""
    
    summary = []
    for item in data:
        entry = {
            'module': item['module'],
            'kind': item['kind'],
            'name': item['name'],
            'qualname': item['qualname'],
            'link': item['link']
        }
        
        # Include summary line only
        if item.get('summary'):
            entry['summary'] = item['summary']
        
        # Optionally include full docstring
        if include_docstrings and item.get('docstring'):
            entry['docstring'] = item['docstring']
            
        summary.append(entry)
    
    return summary

def group_by_module(data: list[dict]) -> dict[str, list[dict]]:
    """Group API items by module for better organization."""
    
    grouped = {}
    for item in data:
        module = item['module']
        if module not in grouped:
            grouped[module] = []
        grouped[module].append({
            'kind': item['kind'],
            'name': item['name'],
            'summary': item.get('summary', '')
        })
    
    return grouped

def main():
    parser = argparse.ArgumentParser(description="Filter API index")
    parser.add_argument("--input", default="ml/public_api_index.json")
    parser.add_argument("--output", default="ml/filtered_api_index.json")
    
    # Filtering options
    parser.add_argument("--core-only", action="store_true",
                       help="Include only core modules (actors, stores, etc)")
    parser.add_argument("--documented-only", action="store_true",
                       help="Include only items with docstrings")
    parser.add_argument("--no-tests", dest="include_tests", action="store_false",
                       help="Exclude test modules (default)")
    parser.add_argument("--no-examples", dest="include_examples", action="store_false",
                       help="Exclude example modules (default)")
    parser.add_argument("--no-cli", dest="include_cli", action="store_false",
                       help="Exclude CLI internals")
    parser.add_argument("--kinds", nargs="+", choices=["function", "class", "method"],
                       help="Include only specific kinds")
    parser.add_argument("--max-depth", type=int,
                       help="Maximum module nesting depth")
    
    # Output options
    parser.add_argument("--summary", action="store_true",
                       help="Create condensed summary without full docstrings")
    parser.add_argument("--grouped", action="store_true",
                       help="Group by module in output")
    parser.add_argument("--stats", action="store_true",
                       help="Print statistics about filtering")
    
    args = parser.parse_args()
    
    # Load data
    with open(args.input) as f:
        data = json.load(f)
    
    original_count = len(data)
    
    # Apply filters
    filters = {
        'include_tests': args.include_tests,
        'include_examples': args.include_examples,
        'include_cli': args.include_cli,
        'documented_only': args.documented_only,
        'kinds': args.kinds,
        'core_only': args.core_only,
        'max_depth': args.max_depth
    }
    
    filtered = [item for item in data if should_include(item, filters)]
    
    # Create output format
    if args.summary:
        output = create_summary_index(filtered, include_docstrings=False)
    else:
        output = filtered
    
    if args.grouped:
        output = group_by_module(output)
    
    # Write output
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print statistics
    if args.stats:
        print(f"Original items: {original_count}")
        print(f"Filtered items: {len(filtered)}")
        print(f"Reduction: {(1 - len(filtered)/original_count)*100:.1f}%")
        
        if not args.grouped:
            # Module count
            modules = set(item['module'] for item in filtered)
            print(f"Modules: {len(modules)}")
            
            # Kind breakdown
            from collections import Counter
            kinds = Counter(item['kind'] for item in filtered)
            for kind, count in kinds.items():
                print(f"  {kind}: {count}")

if __name__ == "__main__":
    main()