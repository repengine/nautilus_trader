#!/usr/bin/env python3

"""
Export Grafana dashboards from running instance.

This script exports ML monitoring dashboards from a running Grafana instance
to local JSON files for version control and backup purposes.

Usage:
    python export_dashboards.py [options]

Example:
    # Export all ML dashboards
    python export_dashboards.py --url http://localhost:3000 --token <api_token>

    # Export specific dashboard
    python export_dashboards.py --url http://localhost:3000 --token <api_token> --uid ml-overview

    # Export with custom output directory
    python export_dashboards.py --url http://localhost:3000 --token <api_token> --output ./backups

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ml.common.logging_config import configure_logging
from ml.monitoring.grafana_client import GrafanaClient


# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def export_dashboard(client: GrafanaClient, uid: str, output_dir: Path) -> bool:
    """
    Export a single dashboard by UID.

    Parameters
    ----------
    client : GrafanaClient
        Grafana API client
    uid : str
        Dashboard UID
    output_dir : Path
        Output directory for exported dashboards

    Returns
    -------
    bool
        True if export successful, False otherwise

    """
    try:
        logger.info(f"Exporting dashboard: {uid}")
        dashboard = client.get_dashboard(uid)

        if not dashboard:
            logger.error(f"Dashboard not found: {uid}")
            return False

        # Clean up the dashboard data for export
        if isinstance(dashboard, dict):
            dashboard_data = dashboard.get("dashboard", {})
        else:
            logger.error(f"Unexpected dashboard response type for {uid}")
            return False

        if dashboard_data:
            # Remove runtime fields
            dashboard_data.pop("id", None)
            dashboard_data.pop("version", None)
            dashboard_data.pop("iteration", None)

            # Ensure UID is set
            dashboard_data["uid"] = uid

            # Save to file
            output_file = output_dir / f"{uid}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully exported: {output_file}")
            return True
        else:
            logger.error(f"No dashboard data found for: {uid}")
            return False

    except Exception as e:
        logger.error(f"Failed to export dashboard {uid}: {e}")
        return False


def export_all_ml_dashboards(client: GrafanaClient, output_dir: Path) -> dict[str, bool]:
    """
    Export all ML monitoring dashboards.

    Parameters
    ----------
    client : GrafanaClient
        Grafana API client
    output_dir : Path
        Output directory for exported dashboards

    Returns
    -------
    dict[str, bool]
        Dictionary mapping dashboard UIDs to export success status

    """
    # ML dashboard UIDs
    ml_dashboard_uids = [
        "ml-overview",
        "data-quality",
        "feature-engineering",
        "model-lifecycle",
        "performance-degradation",
        "resource-utilization",
    ]

    results = {}
    for uid in ml_dashboard_uids:
        results[uid] = export_dashboard(client, uid, output_dir)

    return results


def export_dashboards_by_tag(client: GrafanaClient, tag: str, output_dir: Path) -> dict[str, bool]:
    """
    Export all dashboards with a specific tag.

    Parameters
    ----------
    client : GrafanaClient
        Grafana API client
    tag : str
        Dashboard tag to filter by
    output_dir : Path
        Output directory for exported dashboards

    Returns
    -------
    dict[str, bool]
        Dictionary mapping dashboard UIDs to export success status

    """
    try:
        dashboards = client.search_dashboards(tag=tag)
        results = {}

        for dashboard in dashboards:
            uid = dashboard.get("uid")
            if uid:
                results[uid] = export_dashboard(client, uid, output_dir)
            else:
                logger.warning(f"Dashboard missing UID: {dashboard.get('title', 'Unknown')}")

        return results

    except Exception as e:
        logger.error(f"Failed to search dashboards by tag '{tag}': {e}")
        return {}


def create_export_manifest(
    results: dict[str, bool],
    output_dir: Path,
    client: GrafanaClient | None = None,
) -> None:
    """
    Create export manifest file with results.

    Parameters
    ----------
    results : dict[str, bool]
        Export results mapping UID to success status
    output_dir : Path
        Output directory
    client : GrafanaClient | None, optional
        Grafana client for timestamp

    """
    manifest = {
        "export_timestamp": client.get_server_time() if client else None,
        "total_dashboards": len(results),
        "successful_exports": sum(1 for success in results.values() if success),
        "failed_exports": sum(1 for success in results.values() if not success),
        "results": results,
    }

    manifest_file = output_dir / "export_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Export manifest saved: {manifest_file}")


def main() -> int:
    """
    Export Grafana dashboards from running instance.
    """
    parser = argparse.ArgumentParser(
        description="Export Grafana dashboards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--url",
        required=True,
        help="Grafana server URL (e.g., http://localhost:3000)",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Grafana API token",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./exported_dashboards"),
        help="Output directory for exported dashboards",
    )
    parser.add_argument(
        "--uid",
        help="Export specific dashboard by UID",
    )
    parser.add_argument(
        "--tag",
        default="ml-monitoring",
        help="Export dashboards by tag (default: ml-monitoring)",
    )
    parser.add_argument(
        "--all-ml",
        action="store_true",
        help="Export all known ML monitoring dashboards",
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Verify SSL certificates",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting to: {args.output}")

    # Initialize Grafana client
    try:
        client = GrafanaClient(
            base_url=args.url,
            api_token=args.token,
            verify_ssl=args.verify_ssl,
            timeout=args.timeout,
        )

        # Test connection
        if not client.health_check():
            logger.error("Failed to connect to Grafana server")
            return 1

        logger.info("Connected to Grafana server")

    except Exception as e:
        logger.error(f"Failed to initialize Grafana client: {e}")
        return 1

    # Export dashboards
    results: dict[str, bool] = {}

    try:
        if args.uid:
            # Export specific dashboard
            results[args.uid] = export_dashboard(client, args.uid, args.output)

        elif args.all_ml:
            # Export all ML dashboards
            results = export_all_ml_dashboards(client, args.output)

        else:
            # Export by tag
            results = export_dashboards_by_tag(client, args.tag, args.output)

        # Create export manifest
        create_export_manifest(results, args.output, client)

        # Print summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)

        logger.info(f"Export completed: {successful}/{total} dashboards successful")

        if successful < total:
            logger.warning(f"{total - successful} dashboards failed to export")
            return 1

        return 0

    except KeyboardInterrupt:
        logger.info("Export cancelled by user")
        return 1

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
