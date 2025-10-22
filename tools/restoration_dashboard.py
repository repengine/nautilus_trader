#!/usr/bin/env python3
"""Real-time restoration progress tracking."""

import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def get_category_progress():
    """Track restoration progress per category."""
    categories = [
        "db_connection",
        "missing_attributes",
        "import_errors",
    ]

    progress = {}
    for cat in categories:
        test_file = Path(f"restoration_tasks/{cat}_tests.txt")
        if not test_file.exists():
            continue

        total = len([line for line in test_file.read_text().strip().split('\n') if line.strip()])

        # Count restoration commits for this category
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--grep", f"restore.*{cat}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            restored = len([line for line in result.stdout.strip().split('\n') if line.strip()]) if result.stdout.strip() else 0
        except Exception:
            restored = 0

        progress[cat] = {
            "total": total,
            "restored": restored,
            "remaining": total - restored,
            "percent": (restored / total * 100) if total > 0 else 0
        }

    return progress


def get_test_stats():
    """Get current test pass/fail stats."""
    try:
        result = subprocess.run(
            ["poetry", "run", "pytest", "ml/tests", "--co", "-q"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Parse output for test counts
        last_line = result.stdout.strip().split('\n')[-1]
        # Expected format: "5360 tests collected"
        if "test" in last_line:
            total = int(last_line.split()[0])
            return {"total": total, "status": "counted"}
    except Exception as e:
        return {"total": "?", "status": f"error: {e}"}

    return {"total": "?", "status": "unknown"}


def print_dashboard():
    """Print restoration dashboard."""
    progress = get_category_progress()
    test_stats = get_test_stats()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          RESTORATION PROGRESS DASHBOARD                  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Category progress
    print("Category Progress:")
    print("─" * 60)
    for cat, stats in progress.items():
        bar_length = 40
        filled = int(stats["percent"] / 100 * bar_length) if stats["percent"] > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        cat_display = cat.replace("_", " ").title()
        print(f"{cat_display:.<25} [{bar}] {stats['percent']:.1f}%")
        print(f"  {stats['restored']}/{stats['total']} restored, {stats['remaining']} remaining\n")

    # Overall stats
    total_tests = sum(s["total"] for s in progress.values())
    total_restored = sum(s["restored"] for s in progress.values())
    overall_percent = (total_restored / total_tests * 100) if total_tests > 0 else 0

    print("─" * 60)
    print(f"{'OVERALL PROGRESS':.<25} {total_restored}/{total_tests} ({overall_percent:.1f}%)\n")

    # Test suite stats
    print("─" * 60)
    print(f"Test Suite Status:")
    print(f"  Total tests: {test_stats['total']}")
    print(f"  Status: {test_stats['status']}")

    # Recent commits
    print("\n─" * 60)
    print("Recent Restoration Commits:")
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--grep", "restore:", "-10"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n')[:5]:
                print(f"  {line}")
        else:
            print("  No restoration commits yet")
    except Exception:
        print("  Unable to fetch commits")

    print("\n─" * 60)
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        print_dashboard()
    except KeyboardInterrupt:
        print("\n\nDashboard interrupted.")
    except Exception as e:
        print(f"Error: {e}")
