#!/usr/bin/env python3
"""
Automatic partition management scheduler for ML stores.

This script can be run as a cron job or scheduled task to automatically
maintain database partitions.

Usage:
    # Run once
    python schedule_partitions.py

    # Run as daemon with periodic checks
    python schedule_partitions.py --daemon --interval 3600

    # Dry run to see what would be done
    python schedule_partitions.py --dry-run

Cron example (daily at 2 AM):
    0 2 * * * /usr/bin/python3 /path/to/schedule_partitions.py

"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from ml.common.logging_config import configure_logging


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.common.db_connections import ConnectionRole
from ml.common.db_connections import collect_postgres_candidates
from ml.common.db_connections import select_first_working_connection
from ml.stores.infrastructure import PartitionManager


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up logging configuration.
    """
    level = logging.DEBUG if verbose else logging.INFO

    configure_logging(level=logging.getLevelName(level))

    return logging.getLogger("partition_scheduler")


def main() -> None:
    """
    Run partition scheduler entry point.
    """
    parser = argparse.ArgumentParser(
        description="Automatic partition management for ML stores",
    )

    parser.add_argument(
        "--connection-string",
        type=str,
        default=None,
        help="PostgreSQL connection string (or set ML_DB_CONNECTION env var)",
    )

    parser.add_argument(
        "--months-ahead",
        type=int,
        default=3,
        help="Number of months to create partitions in advance (default: 3)",
    )

    parser.add_argument(
        "--retention-months",
        type=int,
        default=24,
        help="Number of months to retain old partitions (default: 24)",
    )

    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon with periodic checks",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=86400,  # 24 hours
        help="Interval between checks in seconds when running as daemon (default: 86400)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show partition statistics without making changes",
    )

    args = parser.parse_args()

    candidates = collect_postgres_candidates(
        ConnectionRole.PARTITION,
        explicit=args.connection_string,
    )
    if not candidates.urls:
        raise SystemExit(
            "No PostgreSQL connection candidates found. Configure ML_DB_CONNECTION or use --connection-string."
        )

    resolved_connection = candidates.urls[0]
    probe_warning: str | None = None
    if args.connection_string is None:
        try:
            resolved_connection = select_first_working_connection(candidates.urls)
        except RuntimeError as exc:
            probe_warning = "PostgreSQL connectivity probe failed; using first candidate"
            resolved_connection = candidates.urls[0]
            logging.getLogger("partition_scheduler").debug(
                "partition_connection_probe_failed: %s",
                exc,
            )

    # Setup logging
    logger = setup_logging(args.verbose)
    if probe_warning is not None:
        logger.warning("%s (%s)", probe_warning, resolved_connection)

    # Create partition manager
    manager = PartitionManager(
        connection_string=resolved_connection,
        months_ahead=args.months_ahead,
        retention_months=args.retention_months,
        logger=logger,
    )

    # Stats only mode
    if args.stats_only:
        logger.info("Fetching partition statistics...")
        stats = manager.get_partition_stats()

        for table, partitions in stats.items():
            print(f"\n{table}:")
            print("-" * 50)

            if not partitions:
                print("  No partitions found")
            else:
                total_size = sum(int(p["size_bytes"]) for p in partitions if "size_bytes" in p)
                print(f"  Total partitions: {len(partitions)}")
                print(f"  Total size: {total_size / (1024**3):.2f} GB")
                print("\n  Partitions:")
                for partition in partitions:
                    print(f"    {partition['name']}: {partition['size']}")

        return

    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

        print("\nDry run analysis:")
        print("-" * 50)

        # Check what partitions would be created
        current_date = datetime.now().date()
        print(f"Current date: {current_date}")
        print(f"Would create partitions up to {args.months_ahead} months ahead")
        print(f"Would remove partitions older than {args.retention_months} months")

        stats = manager.get_partition_stats()
        for table in manager.tables:
            partitions = stats.get(table, [])
            if partitions:
                oldest = min(p["name"] for p in partitions)
                newest = max(p["name"] for p in partitions)
                print(f"\n{table}:")
                print(f"  Current range: {oldest} to {newest}")
                print(f"  Total partitions: {len(partitions)}")

        return

    # Run maintenance
    def run_maintenance() -> bool:
        """
        Run partition maintenance cycle.
        """
        try:
            logger.info("Starting partition maintenance cycle")
            results = manager.run_maintenance()

            logger.info(
                f"Maintenance complete - Created: {results['created']}, Removed: {results['removed']}",
            )

            # Log statistics
            stats = manager.get_partition_stats()
            for table, partitions in stats.items():
                logger.info(f"{table}: {len(partitions)} partitions")

            return True

        except Exception as e:
            logger.error(f"Maintenance failed: {e}", exc_info=True)
            return False

    # Single run or daemon mode
    if args.daemon:
        logger.info(f"Starting daemon mode with {args.interval}s interval")

        while True:
            success = run_maintenance()

            if not success:
                logger.warning("Maintenance failed, will retry at next interval")

            logger.info(f"Sleeping for {args.interval} seconds...")
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("Daemon stopped by user")
                break
    else:
        # Single run
        success = run_maintenance()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
